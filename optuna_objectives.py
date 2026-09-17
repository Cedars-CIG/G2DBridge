# optuna_objectives.py
# =============================================================================
# Optuna objective functions and inference helpers.
#
# Directions:
#   Regression  objectives return safe R²  → study(direction="maximize")
#   Classification objectives return 1-AUC → study(direction="minimize")
# =============================================================================

from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

import config
from utils import set_seed, safe_r2
from nn_functions import train_nn_with_params, train_nn_with_params_cls


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_nn_ad(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Binary classification probabilities for IDP→AD NN. Returns shape (N,)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                    batch_size=batch_size, shuffle=False)
    model.eval()
    outs = []
    for (xb,) in dl:
        logits = model(xb.to(device))
        if logits.ndim == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(1)
        outs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def predict_nn_idp(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Regression predictions for SNP→IDP NN. Returns shape (N, D)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                    batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    for (xb,) in dl:
        preds.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# SNP → IDP (regression) objectives
# ---------------------------------------------------------------------------

def lasso_objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    set_seed(42)
    alpha = trial.suggest_categorical(
        "alpha", [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.01, 0.001, 0.0001]
    )
    model = Lasso(alpha=alpha, max_iter=1000, warm_start=True, selection="random")
    model.fit(X_train, y_train)
    return float(safe_r2(y_val, model.predict(X_val)))


def rf_objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    set_seed(42)
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
        max_depth=trial.suggest_int("max_depth", 2, 12),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return float(safe_r2(y_val, model.predict(X_val)))


def nn_objective(
    trial: optuna.Trial,
    X_train, y_train, X_val, y_val,
    input_size: int, output_size: int,
) -> float:
    """Regression NN with monotone width search; returns safe R² on validation."""
    set_seed(42)

    MAX_LAYERS = 3
    num_layers = trial.suggest_int("num_layers", 1, MAX_LAYERS)

    layer_selection = [4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256, 192, 128, 96, 64, 32, 16, 8, 4]
    cap = min(4096, max(int(input_size), int(output_size), layer_selection[-1]))
    allowed = [h for h in layer_selection if h <= cap and h >= int(output_size)]
    if not allowed:
        ge = [h for h in layer_selection if h >= int(output_size)]
        allowed = ge if ge else [max(int(output_size), 4)]
    L = len(allowed)

    idxs_raw = [trial.suggest_int(f"h_idx_{k}", 0, L - 1) for k in range(1, MAX_LAYERS + 1)]
    idxs_monotone, running_max = [], -1
    for idx in idxs_raw:
        running_max = max(running_max, idx)
        idxs_monotone.append(running_max)
    hidden_units = [allowed[i] for i in idxs_monotone[:num_layers]]

    dropout_input = trial.suggest_float("dropout_input", 0.0, 0.5, step=0.01)
    dropouts = [dropout_input] + [
        trial.suggest_float(f"dropout_layer{k}", 0.0, 0.5, step=0.01)
        for k in range(1, num_layers + 1)
    ]

    act        = trial.suggest_categorical("activation", ["relu", "gelu", "leakyrelu", "silu"])
    norm       = trial.suggest_categorical("norm", [None, "layernorm", "batchnorm"])
    res_every  = trial.suggest_categorical("residual_every", [None, 2, 3])
    optimizer  = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    if optimizer in ("adam", "adamw"):
        lr      = trial.suggest_float("lr_adam", 1e-5, 1e-2, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [False, True]) if optimizer == "adam" else False
        momentum = None
    else:
        lr       = trial.suggest_float("lr_sgd", 1e-4, 1.0, log=True)
        momentum = trial.suggest_float("momentum_sgd", 0.0, 0.99)
        amsgrad  = False

    wd          = trial.suggest_categorical("weight_decay", [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    loss_name   = trial.suggest_categorical("loss", ["mse", "huber"])
    huber_delta = trial.suggest_float("huber_delta", 0.5, 2.0) if loss_name == "huber" else 1.0
    sched       = trial.suggest_categorical("scheduler", [None, "cosine", "onecycle"])
    grad_clip   = trial.suggest_float("grad_clip", 0.0, 1.0)

    params: Any = dict(
        hidden_units=hidden_units, dropouts=dropouts, learning_rate=lr,
        weight_decay=wd, batch_size=batch_size, patience=10,
        activation=act, norm=norm, residual_every=res_every,
        optimizer=optimizer, amsgrad=amsgrad, momentum=momentum,
        loss=loss_name, huber_delta=huber_delta, scheduler=sched,
        max_epochs=100, grad_clip=grad_clip, trial=trial,
    )

    tmp_dir = config.OPTUNA_TMP_DIR
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_name = f"_optuna_t{trial.number}_{uuid.uuid4().hex[:6]}.pt"

    model, model_path = train_nn_with_params(
        X_train, y_train, X_val, y_val,
        input_size, output_size, params,
        model_dir=tmp_dir, model_name=tmp_name,
    )

    try:
        return float(safe_r2(y_val, predict_nn_idp(model, X_val)))
    finally:
        try:
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
        except OSError:
            pass


def xgb_reg_objective(
    trial: optuna.Trial,
    X_train, y_train, X_val, y_val,
    use_gpu: bool = True,
) -> float:
    """Multi-output XGBoost regression; returns safe R² (maximize)."""
    set_seed(42)
    want_cuda = use_gpu and torch.cuda.is_available()
    params: Any = dict(
        n_estimators     = trial.suggest_int("n_estimators", 400, 1600, step=200),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
        max_depth        = trial.suggest_int("max_depth", 2, 8),
        min_child_weight = trial.suggest_int("min_child_weight", 2, 10),
        subsample        = trial.suggest_float("subsample", 0.6, 0.95, step=0.05),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 0.6, step=0.05),
        reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 0.02, step=0.002),
        reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 10.0, step=0.5),
        objective        = "reg:squarederror",
        tree_method      = "hist",
        device           = "cuda" if want_cuda else "cpu",
        n_jobs           = -1,
        random_state     = 42,
    )

    D = y_train.shape[1] if y_train.ndim == 2 else 1
    preds_val = np.zeros((len(X_val), D), dtype=np.float32)
    for j in range(D):
        yj = y_train[:, j] if y_train.ndim == 2 else y_train
        valid = np.isfinite(yj)
        if not np.any(valid):
            preds_val[:, j] = np.nan
            continue
        m = XGBRegressor(**params)
        m.fit(X_train[valid], yj[valid])
        preds_val[:, j] = m.predict(X_val).astype(np.float32)
    return float(safe_r2(y_val, preds_val))


# ---------------------------------------------------------------------------
# IDP → AD (classification) objectives
# ---------------------------------------------------------------------------

def idp_lr_objective(trial: optuna.Trial, X_tr, y_tr, X_val, y_val) -> float:
    set_seed(42)
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    lr = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                            max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)
    auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
    return 1.0 - float(auc)


def idp_rf_objective(trial: optuna.Trial, X_tr, y_tr, X_val, y_val) -> float:
    set_seed(42)
    rf = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_int("max_depth", 6, 32, step=2),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    return 1.0 - float(auc)


def idp_lasso_objective(trial: optuna.Trial, X_tr, y_tr, X_val, y_val) -> float:
    """L1 linear model, sigmoid-squashed, scored with AUC."""
    set_seed(42)
    alpha = trial.suggest_categorical("alpha", [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 1e-2])
    lasso = Lasso(alpha=alpha, max_iter=5000)
    lasso.fit(X_tr, y_tr)
    p = 1.0 / (1.0 + np.exp(-lasso.predict(X_val)))
    auc = roc_auc_score(y_val, p)
    return 1.0 - float(auc)


def idp_nn_objective(
    trial: optuna.Trial,
    X_tr, y_tr, X_val, y_val,
    input_size: int,
) -> float:
    """Compact NN search for IDP→AD classification. Returns (1-AUC)."""
    set_seed(42)
    hidden      = trial.suggest_categorical("hidden", [4, 8, 12, 16, 24, 32, 48, 64, 72, 96, 128])
    dropout     = trial.suggest_categorical("dropout", [0.0, 0.2])
    lr          = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    wd          = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4])
    batch_size  = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 24, 32])

    params: Any = dict(
        hidden_units=[hidden], dropout=dropout, learning_rate=lr,
        weight_decay=wd, batch_size=batch_size, patience=10,
        max_epochs=100, grad_clip=1.0,
    )

    tmp_dir = config.OPTUNA_TMP_DIR
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_name = f"_optuna_idp2ad_t{trial.number}_{uuid.uuid4().hex[:6]}.pt"

    model, model_path = train_nn_with_params_cls(
        X_tr, y_tr, X_val, y_val,
        input_size=input_size, output_size=1, params=params,
        model_dir=tmp_dir, model_name=tmp_name,
    )

    try:
        p_val = predict_nn_ad(model, X_val)
        auc = roc_auc_score(y_val, p_val) if np.unique(y_val).size >= 2 else np.nan
        loss = 1.0 - float(auc) if np.isfinite(auc) else 1.0
        return loss
    finally:
        try:
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
        except OSError:
            pass
