#!/usr/bin/env python3
"""Stage 1: per-chromosome SNP -> IDP regression (NN, Optuna-tuned).

One subject = one row (duplicate genotyping batches / imaging visits are
mean-aggregated). Trains on ADNI subjects with both genome and real IDP;
also scores the fitted model on the ADSP cohort for later fusion.

Usage:
    python stage1_snp_to_idp.py --idp 1 --chr 1 --n_trials 50 --seed 42
"""
import argparse
import os

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import config
from data_loader_functions import get_adni_patient_info, getIDPinfo, read_all_raw_data
from nn_functions import train_nn_with_params
from optuna_objectives import nn_objective, predict_nn_idp

IDP_CSVID = {1: 8, 2: 10, 3: 14}  # DTI, FreeSurfer, ASHS -- see data_loader_functions._IDP_FILES


def aggregate_by_subject(ids, values):
    uniq, inv = np.unique(ids, return_inverse=True)
    out = np.zeros((len(uniq), values.shape[1]), dtype=np.float32)
    counts = np.zeros(len(uniq))
    for i, row in zip(inv, values):
        out[i] += np.nan_to_num(row, nan=0.0)
        counts[i] += 1
    return uniq, out / counts[:, None]


def r2_safe(y_true, y_pred):
    scores = []
    for j in range(y_true.shape[1]):
        m = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        if m.sum() >= 2 and y_true[m, j].var() > 1e-12:
            scores.append(r2_score(y_true[m, j], y_pred[m, j]))
    return float(np.mean(scores)) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idp", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--chr", type=int, required=True, choices=range(1, 23))
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--max_epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _, pat_dict, pat_to_label = get_adni_patient_info()
    idp_ids_raw, idp_data_raw = getIDPinfo(IDP_CSVID[args.idp], pat_dict)
    idp_ids = np.array(idp_ids_raw)
    valid = idp_ids != -1
    idp_subjects, Y_subj = aggregate_by_subject(
        idp_ids[valid], np.vstack(np.array(idp_data_raw, dtype=object)[valid]).astype(np.float32)
    )

    final_data, _, all_ids = read_all_raw_data(args.chr, pat_to_label)
    gen_ids = np.concatenate([all_ids[c] for c in ("ADNI1", "ADNI31", "ADNI32", "ADNI4")])
    gen_snp = np.concatenate([final_data[c] for c in ("ADNI1", "ADNI31", "ADNI32", "ADNI4")], axis=0)
    gen_subjects, G_subj = aggregate_by_subject(gen_ids, gen_snp.astype(np.float32))
    adsp_subjects, G_adsp = aggregate_by_subject(np.asarray(all_ids["ADSP"]), final_data["ADSP"].astype(np.float32))

    common = np.intersect1d(gen_subjects, idp_subjects)
    g_pos = {s: i for i, s in enumerate(gen_subjects)}
    y_pos = {s: i for i, s in enumerate(idp_subjects)}
    G = G_subj[[g_pos[s] for s in common]]
    Y = Y_subj[[y_pos[s] for s in common]]

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, test_idx = next(iter(kf.split(common)))

    imputer = SimpleImputer(strategy="mean")
    X_tr = imputer.fit_transform(G[train_idx]).astype(np.float32)
    X_te = imputer.transform(G[test_idx]).astype(np.float32)
    X_adsp = imputer.transform(G_adsp).astype(np.float32)
    x_scaler = StandardScaler()
    X_tr = x_scaler.fit_transform(X_tr).astype(np.float32)
    X_te = x_scaler.transform(X_te).astype(np.float32)
    X_adsp = x_scaler.transform(X_adsp).astype(np.float32)

    y_scaler = StandardScaler()
    Y_tr = y_scaler.fit_transform(Y[train_idx]).astype(np.float32)
    Y_te = y_scaler.transform(Y[test_idx]).astype(np.float32)

    n_val = max(1, int(0.1 * len(X_tr)))
    X_fit, X_val, Y_fit, Y_val = X_tr[n_val:], X_tr[:n_val], Y_tr[n_val:], Y_tr[:n_val]

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=args.seed))
    study.optimize(
        lambda t: nn_objective(t, X_fit, Y_fit, X_val, Y_val, X_tr.shape[1], Y_tr.shape[1]),
        n_trials=args.n_trials,
    )

    out_dir = os.path.join(config.STAGE1_OUT_DIR, f"idp{args.idp}")
    os.makedirs(out_dir, exist_ok=True)
    best_params = dict(study.best_trial.params, max_epochs=args.max_epochs, patience=10)
    model, _ = train_nn_with_params(
        X_tr, Y_tr, X_te, Y_te, X_tr.shape[1], Y_tr.shape[1], best_params,
        model_dir=out_dir, model_name=f"chr{args.chr:02d}.pt",
    )

    preds_te = y_scaler.inverse_transform(predict_nn_idp(model, X_te))
    preds_adsp = y_scaler.inverse_transform(predict_nn_idp(model, X_adsp))
    r2 = r2_safe(Y[test_idx], preds_te)

    np.savez_compressed(
        os.path.join(out_dir, f"chr{args.chr:02d}_preds.npz"),
        preds_test=preds_te, y_test=Y[test_idx],
        preds_adsp=preds_adsp, adsp_ids=adsp_subjects,
    )
    print(f"idp{args.idp} chr{args.chr}: R2_test={r2:.4f}  best_val_R2={study.best_value:.4f}  "
          f"-> {out_dir}/chr{args.chr:02d}.pt")


if __name__ == "__main__":
    main()
