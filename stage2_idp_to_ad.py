#!/usr/bin/env python3
"""Stage 2: IDP -> AD classifier. Three branch NNs (one per IDP modality,
DTI/FreeSurfer/ASHS) + a combo NN over their concatenated penultimate
features. Trained ONLY on ADNI's real measured IDP + real diagnosis label;
external cohorts (e.g. ADSP, via fuse_stage1_predictions.py's genome-
predicted IDP) are never touched here -- score them separately with
predict.py.

Hyperparameters are chosen by 5-fold CV on train+val (not a single small
val split): with ~450 ADNI subjects split 60/20/20, a single ~90-subject
val fold is noisy enough that Optuna can pick hyperparameters that look
great on that fold and collapse on the untouched test set. Averaging over
5 folds fixes this.

Usage:
    python stage2_idp_to_ad.py --n_trials 50 --seed 42
"""
import argparse
import os

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import config
from data_loader_functions import get_adni_patient_info, getIDPinfo
from nn_functions import train_nn_with_params_cls
from optuna_objectives import predict_nn_ad
from utils import set_seed

IDP_CSVID = {1: 8, 2: 10, 3: 14}


def aggregate_by_subject(ids, values):
    uniq, inv = np.unique(ids, return_inverse=True)
    out = np.zeros((len(uniq), values.shape[1]), dtype=np.float32)
    counts = np.zeros(len(uniq))
    for i, row in zip(inv, values):
        out[i] += np.nan_to_num(row, nan=0.0)
        counts[i] += 1
    return uniq, out / counts[:, None]


def _params_from_trial(trial):
    hidden = trial.suggest_categorical("hidden", [32, 48, 64, 72, 96, 128, 256, 512, 1024])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    wd = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4])
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 24, 32])
    return dict(hidden_units=[hidden], dropout=dropout, learning_rate=lr,
                weight_decay=wd, batch_size=batch_size, patience=10, max_epochs=100, grad_clip=1.0)


def _cv_mean_auc(params, X, y, input_size, seed, n_splits=5):
    aucs = []
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        model, _ = train_nn_with_params_cls(
            X[tr], y[tr].astype(np.float32), X[va], y[va].astype(np.float32),
            input_size=input_size, output_size=1, params=dict(params),
            model_dir=config.OPTUNA_TMP_DIR, model_name=f"_cv_{np.random.randint(1e9)}.pt",
        )
        aucs.append(roc_auc_score(y[va], predict_nn_ad(model, X[va])))
    return float(np.mean(aucs))


def train_branch(X_tr, y_tr, X_val, y_val, input_size, n_trials, seed, model_dir, model_name):
    X_trval, y_trval = np.concatenate([X_tr, X_val]), np.concatenate([y_tr, y_val])
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(
        lambda t: 1.0 - _cv_mean_auc(_params_from_trial(t), X_trval, y_trval, input_size, seed),
        n_trials=n_trials,
    )
    params = dict(_params_from_trial(optuna.trial.FixedTrial(study.best_params)))
    model, _ = train_nn_with_params_cls(
        X_tr, y_tr.astype(np.float32), X_val, y_val.astype(np.float32),
        input_size=input_size, output_size=1, params=params,
        model_dir=model_dir, model_name=model_name,
    )
    return model, study.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(config.STAGE2_OUT_DIR, exist_ok=True)

    _, pat_dict, pat_to_label = get_adni_patient_info()
    idp, ids = {}, {}
    for k in (1, 2, 3):
        raw_ids, raw_data = getIDPinfo(IDP_CSVID[k], pat_dict)
        raw_ids = np.array(raw_ids)
        valid = raw_ids != -1
        ids[k], idp[k] = aggregate_by_subject(
            raw_ids[valid], np.vstack(np.array(raw_data, dtype=object)[valid]).astype(np.float32)
        )

    common = sorted(set(ids[1]) & set(ids[2]) & set(ids[3]))
    labels = np.array([pat_to_label.get(s, 2) for s in common])
    keep = np.isin(labels, [0, 1])
    subjects = [s for s, k in zip(common, keep) if k]
    y = labels[keep]

    idx_trainval, idx_test = train_test_split(np.arange(len(subjects)), test_size=0.2,
                                               random_state=args.seed, stratify=y)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.25,
                                           random_state=args.seed, stratify=y[idx_trainval])
    print(f"ADNI subjects: {len(subjects)}  AD={int(y.sum())}  CN={int((y == 0).sum())}  "
          f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    pen = {"train": [], "val": [], "test": []}
    for k in (1, 2, 3):
        pos = {s: i for i, s in enumerate(ids[k])}
        X = np.stack([idp[k][pos[s]] for s in subjects])
        scaler = StandardScaler().fit(X[idx_train])
        Xs = {split: scaler.transform(X[idx]).astype(np.float32)
              for split, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test))}

        model, cv_auc = train_branch(
            Xs["train"], y[idx_train], Xs["val"], y[idx_val], X.shape[1],
            args.n_trials, args.seed, config.STAGE2_OUT_DIR, f"branch_idp{k}.pt",
        )
        test_auc = roc_auc_score(y[idx_test], predict_nn_ad(model, Xs["test"]))
        print(f"  idp{k} branch: 5-fold CV AUC={cv_auc:.4f}  test AUC={test_auc:.4f}")

        with torch.no_grad():
            for split, Xsp in Xs.items():
                pen[split].append(model.forward_features(torch.tensor(Xsp)).numpy())

    X_combo = {s: np.concatenate(v, axis=1).astype(np.float32) for s, v in pen.items()}
    combo, cv_auc = train_branch(
        X_combo["train"], y[idx_train], X_combo["val"], y[idx_val], X_combo["train"].shape[1],
        args.n_trials, args.seed, config.STAGE2_OUT_DIR, "combo.pt",
    )
    test_auc = roc_auc_score(y[idx_test], predict_nn_ad(combo, X_combo["test"]))
    print(f"\nCombo model: 5-fold CV AUC={cv_auc:.4f}  ADNI held-out test AUC={test_auc:.4f}")
    print(f"Saved branch_idp{{1,2,3}}.pt + combo.pt -> {config.STAGE2_OUT_DIR}")


if __name__ == "__main__":
    main()
