# nn_functions.py
# =============================================================================
# Neural building blocks and training utilities for regression (SNP→IDP)
# and classification (IDP→AD).
# =============================================================================

import os
import uuid
import math
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

import config
from early_stopping import EarlyStopping

try:
    import optuna  # noqa: F401
except Exception:
    pass


# -------------------------
# Reproducibility
# -------------------------
def set_torch_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


# -------------------------
# Supported activations
# -------------------------
_ACTS: Dict[str, Any] = {
    "relu":      nn.ReLU,
    "gelu":      nn.GELU,
    "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "silu":      nn.SiLU,
}


class FlexibleBlock(nn.Module):
    """
    MLP block: Linear → [Norm] → Activation → [Dropout] → [Residual].
    Residual is applied only when in_f == out_f.
    """
    def __init__(
        self,
        in_f: int,
        out_f: int,
        act: str = "relu",
        norm: Optional[str] = None,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_f, out_f)]
        if norm == "layernorm":
            layers.append(nn.LayerNorm(out_f))
        elif norm == "batchnorm":
            layers.append(nn.BatchNorm1d(out_f))
        layers.append(_ACTS[act]())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)
        self.residual = bool(residual and (in_f == out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.residual:
            y = y + x
        return y


class AdvancedNN(nn.Module):
    """
    Fully-connected neural network with:
      - Multiple hidden layers
      - Configurable per-layer dropout (input + per hidden)
      - Optional residual connections (every N-th block if dims match)
      - Optional normalization (layernorm / batchnorm)

    Exposes:
      .input_drop — first dropout / identity
      .hidden     — Sequential of hidden blocks (used by feature-extractors)
      .out        — final linear layer
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_units: List[int],
        dropouts: Optional[List[float]] = None,
        act: str = "relu",
        norm: Optional[str] = None,
        residual_every: Optional[int] = None,
    ):
        super().__init__()
        hidden_units = list(hidden_units) if hidden_units else []
        dropouts = (list(dropouts) if dropouts is not None
                    else [0.0] * (len(hidden_units) + 1))
        assert len(dropouts) == len(hidden_units) + 1, (
            "dropouts must have length len(hidden_units) + 1"
        )

        self.input_drop = nn.Dropout(dropouts[0]) if dropouts[0] > 0 else nn.Identity()

        blocks: List[nn.Module] = []
        in_f = int(input_size)
        for i, h in enumerate(hidden_units):
            h = int(h)
            use_res = False
            if residual_every is not None and residual_every > 0:
                use_res = ((i + 1) % int(residual_every) == 0) and (in_f == h)
            blocks.append(
                FlexibleBlock(in_f, h, act=act, norm=norm,
                              dropout=float(dropouts[i + 1]), residual=use_res)
            )
            in_f = h
        self.hidden = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.out = nn.Linear(in_f, int(output_size))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate (last hidden) representation."""
        x = self.input_drop(x)
        return self.hidden(x)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_features(x)
        return self.out(h), h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.forward_features(x))


# =============================================================================
# Regression training  (SNP → IDP)
# =============================================================================
def train_nn_with_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    input_size: int,
    output_size: int,
    params: Dict[str, Any],
    model_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[AdvancedNN, str]:
    """
    Train AdvancedNN for regression (MSE or Huber loss).
    Saves and reloads the best model on validation loss.
    Returns (best_model, model_path).
    """
    if model_dir is None:
        model_dir = config.MODELS_DIR
    os.makedirs(model_dir, exist_ok=True)

    if model_name is None:
        save_dir = os.path.join(model_dir, "tmp")
        os.makedirs(save_dir, exist_ok=True)
        model_name = f"optuna_snp2idp_{uuid.uuid4().hex[:8]}_nn.pt"
    else:
        save_dir = model_dir
    if not model_name.endswith(".pt"):
        model_name += ".pt"
    model_path = os.path.join(save_dir, model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AdvancedNN(
        input_size=input_size,
        output_size=output_size,
        hidden_units=params.get("hidden_units", [32]),
        dropouts=params.get("dropouts", None),
        act=params.get("activation", "relu"),
        norm=params.get("norm", None),
        residual_every=params.get("residual_every", None),
    ).to(device)

    batch_size  = int(params.get("batch_size", 32))
    patience    = int(params.get("patience", 10))
    lr          = float(params.get("learning_rate", 1e-3))
    wd          = float(params.get("weight_decay", 1e-4))
    max_epochs  = int(params.get("max_epochs", 150))
    grad_clip   = float(params.get("grad_clip", 0.0))
    opt_name    = str(params.get("optimizer", "adamw")).lower()
    loss_name   = str(params.get("loss", "mse")).lower()
    huber_delta = float(params.get("huber_delta", 1.0))
    sched_name  = params.get("scheduler", None)

    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd,
                         amsgrad=bool(params.get("amsgrad", False)))

    scheduler = None
    if sched_name == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(5, max_epochs // 10), T_mult=2)
    elif sched_name == "onecycle":
        steps_per_epoch = max(1, math.ceil(len(train_ds) / batch_size))
        scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=max_epochs,
                               steps_per_epoch=steps_per_epoch)

    def loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if loss_name == "huber":
            return F.huber_loss(pred, tgt, delta=huber_delta)
        return F.mse_loss(pred, tgt)

    early    = EarlyStopping(patience=patience)
    best_val = float("inf")
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    for ep in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler and isinstance(scheduler, OneCycleLR):
                scheduler.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), yv).item()

        if scheduler and isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(ep + 1)

        if vloss < best_val:
            best_val = vloss
            torch.save(model, model_path)

        early(vloss)
        if early.early_stop:
            break

    best_model: AdvancedNN = torch.load(model_path, map_location=device, weights_only=False)
    best_model.eval()
    return best_model, model_path


# =============================================================================
# Classification training  (IDP → AD)
# =============================================================================
def _normalize_dropouts_config(
    hidden_units: List[int],
    params: Dict[str, Any],
) -> List[float]:
    """
    Normalise various dropout inputs into a list of length len(hidden_units)+1.
    Supports scalar, list, or missing (defaults to 0.0 everywhere).
    """
    hidden_units = list(hidden_units) if hidden_units else []
    p_scalar = params.pop("dropout", None)
    dropouts = params.get("dropouts", None)

    def _broadcast(p: float, n: int) -> List[float]:
        return [float(p)] + [float(p)] * n

    if dropouts is None:
        return _broadcast(0.0 if p_scalar is None else float(p_scalar), len(hidden_units))

    dips = list(dropouts) if not isinstance(dropouts, (float, int)) else [float(dropouts)]
    if len(dips) == 1:
        return _broadcast(dips[0], len(hidden_units))
    target = len(hidden_units) + 1
    if len(dips) < target:
        dips = dips + [float(dips[-1]) if dips else 0.0] * (target - len(dips))
    elif len(dips) > target:
        dips = dips[:target]
    return [float(x) for x in dips]


def train_nn_with_params_cls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    input_size: int,
    output_size: int,
    params: Dict[str, Any],
    model_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[AdvancedNN, str]:
    """
    Train AdvancedNN for binary classification (BCEWithLogitsLoss).
    Saves and reloads the best model.
    Returns (best_model, model_path).
    """
    if model_dir is None:
        model_dir = config.MODELS_DIR
    os.makedirs(model_dir, exist_ok=True)

    if model_name is None:
        model_name = f"optuna_idp2ad_{uuid.uuid4().hex[:8]}_nn.pt"
    if not model_name.endswith(".pt"):
        model_name += ".pt"
    model_path = os.path.join(model_dir, model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_units = list(params.get("hidden_units", [32]))
    dropouts = _normalize_dropouts_config(hidden_units, params)
    params["dropouts"] = dropouts

    model = AdvancedNN(
        input_size=input_size,
        output_size=output_size,
        hidden_units=hidden_units,
        dropouts=dropouts,
        act=params.get("activation", "relu"),
        norm=params.get("norm", None),
        residual_every=params.get("residual_every", None),
    ).to(device)

    batch_size = int(params.get("batch_size", 32))
    patience   = int(params.get("patience", 10))
    lr         = float(params.get("learning_rate", 1e-3))
    wd         = float(params.get("weight_decay", 1e-4))
    max_epochs = int(params.get("max_epochs", 100))
    grad_clip  = float(params.get("grad_clip", 0.0))
    opt_name   = str(params.get("optimizer", "adamw")).lower()
    sched_name = params.get("scheduler", None)
    amsgrad    = bool(params.get("amsgrad", False))

    Xtr = torch.as_tensor(X_train, dtype=torch.float32)
    ytr = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    Xv  = torch.as_tensor(X_val, dtype=torch.float32).to(device)
    yv  = torch.as_tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=True)

    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=amsgrad)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    if sched_name == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(5, max_epochs // 10), T_mult=2)
    elif sched_name == "onecycle":
        steps_per_epoch = max(1, math.ceil(len(loader.dataset) / batch_size))
        scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=max_epochs,
                               steps_per_epoch=steps_per_epoch)

    bce      = nn.BCEWithLogitsLoss()
    early    = EarlyStopping(patience=patience)
    best_val = float("inf")

    for ep in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = bce(model(xb), yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler and isinstance(scheduler, OneCycleLR):
                scheduler.step()

        model.eval()
        with torch.no_grad():
            vloss = bce(model(Xv), yv).item()

        if scheduler and isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(ep + 1)

        if vloss < best_val:
            best_val = vloss
            torch.save(model, model_path)

        early(vloss)
        if early.early_stop:
            break

    best_model: AdvancedNN = torch.load(model_path, map_location=device, weights_only=False)
    best_model.eval()
    return best_model, model_path
