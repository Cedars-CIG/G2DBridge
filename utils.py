import numpy as np
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If using GPU, ensure deterministic behavior
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_r2(y_true, y_pred, clip=(-1.0, 1.0), eps=1e-12):
    """
    Robust R² using absolute errors, with clipping and numeric safeguards.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    clip : (float, float), optional
        (min, max) bounds for the returned score. Default (-10.0, 1.0).
    eps : float, optional
        Small number to avoid divide-by-zero. Default 1e-12.

    Returns
    -------
    float
        Clipped robust R² score.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()

    # Drop NaN/Inf pairs
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return np.nan
    yt, yp = yt[mask], yp[mask]

    # Absolute-error version of residual and total "sum of squares"
    ss_res = np.sum(np.abs(yt - yp))
    ss_tot = np.sum(np.abs(yt - np.mean(yt)))

    # Handle (near-)constant target safely
    if ss_tot < eps:
        score = 1.0 if ss_res < eps else 0.0  # define convention
    else:
        score = 1.0 - (ss_res / max(ss_tot, eps))

    # Final clip (upper bound typically 1.0 for R²-type metrics)
    return float(np.clip(score, clip[0], clip[1]))

import numpy as np

def safe_r2(y_true, y_pred, eps_var: float = 1e-12) -> float:
    """
    Robust multi-output R² score.

    This function avoids exploding negative values when the target variance
    is close to zero, and ensures NaN/Inf predictions do not break the metric.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values (N, D).
    y_pred : array-like
        Predicted values (N, D).
    eps_var : float, optional
        Minimum variance threshold. Targets with variance <= eps_var
        are excluded from the mean R² calculation.

    Returns
    -------
    float
        Mean R² score across valid targets. Returns 0.0 if no valid targets remain.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    # Ensure 2D arrays
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    # Sanity check: number of rows must match
    n = yt.shape[0]
    if yp.shape[0] != n:
        raise ValueError(
            f"safe_r2: y_true ({yt.shape}) and y_pred ({yp.shape}) have different number of samples."
        )

    # Replace NaN/Inf predictions with zeros
    yp = np.where(np.isfinite(yp), yp, 0.0)

    # Compute per-target variance and mask out near-constant targets
    var = yt.var(axis=0, ddof=1)
    mask = var > eps_var
    if not np.any(mask):
        return 0.0  # No valid targets

    yt = yt[:, mask]
    yp = yp[:, mask]

    # Residual and total sum of squares
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)

    # Avoid division by zero
    ss_tot = np.maximum(ss_tot, eps_var)

    # Per-target R²
    r2_cols = 1.0 - (ss_res / ss_tot)

    # Optionally, clip extreme negatives:
    r2_cols = np.clip(r2_cols, -1.0, 1.0)

    # Return mean across targets
    return float(np.mean(r2_cols))
