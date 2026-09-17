# cv_split.py
"""
Generates a leakage-free 5-fold cross-validation split with an explicit
60% train / 20% validation / 20% test partition of each fold, computed once
per subject (never per visit/row) so no subject appears in two roles.

For classification labels, folds are stratified so each split preserves the
overall class balance. Regression targets (e.g. Stage 1's IDP vectors) use
plain KFold since there is no class to stratify on.
"""
from typing import List, Optional, Sequence

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def make_folds(
    subject_ids: Sequence,
    labels: Optional[np.ndarray] = None,
    n_folds: int = 5,
    val_frac: float = 0.2,
    seed: int = 42,
) -> List[dict]:
    """
    Splits `subject_ids` into `n_folds` folds, each with disjoint
    train/val/test subject ID arrays.

    Per fold: test = 1/n_folds of all subjects (20% for n_folds=5); the
    remaining subjects are split again so that val is exactly `val_frac`
    of the TOTAL subject count (not of the remainder), leaving
    1 - 1/n_folds - val_frac for train (60% for the default 5-fold/20%-val
    setup).

    Parameters
    ----------
    subject_ids : array-like of shape (N,)
        One entry per subject (must already be deduplicated).
    labels : array-like of shape (N,), optional
        Binary/categorical label per subject. If given, both the outer
        (test) and inner (val) splits are stratified on it.
    n_folds : int
        Number of CV folds (outer test-fold size = 1/n_folds).
    val_frac : float
        Fraction of the TOTAL subject count to reserve for validation.
    seed : int
        Random seed, applied identically to every fold for reproducibility.

    Returns
    -------
    List of length `n_folds`, each a dict with keys "train", "val", "test",
    holding subject-ID arrays.
    """
    subject_ids = np.asarray(subject_ids)
    n = len(subject_ids)
    if labels is not None:
        labels = np.asarray(labels)
        assert len(labels) == n, "labels must align 1:1 with subject_ids"

    test_frac = 1.0 / n_folds
    if val_frac + test_frac >= 1.0:
        raise ValueError(f"val_frac ({val_frac}) + test_frac ({test_frac:.3f}) must be < 1.0")
    # val_frac is a fraction of the TOTAL; re-express as a fraction of the
    # (1 - test_frac) remainder so the final val slice is val_frac of the total.
    val_frac_of_remainder = val_frac / (1.0 - test_frac)

    if labels is not None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(subject_ids, labels)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(subject_ids)

    folds = []
    for trainval_idx, test_idx in split_iter:
        ids_trainval = subject_ids[trainval_idx]
        labels_trainval = labels[trainval_idx] if labels is not None else None

        train_idx_rel, val_idx_rel = train_test_split(
            np.arange(len(ids_trainval)),
            test_size=val_frac_of_remainder,
            random_state=seed,
            stratify=labels_trainval,
        )

        folds.append({
            "train": ids_trainval[train_idx_rel],
            "val": ids_trainval[val_idx_rel],
            "test": subject_ids[test_idx],
        })

    # Sanity check: no subject appears in two roles within the same fold,
    # and train+val+test reconstructs the full subject set for every fold.
    for f in folds:
        assert set(f["train"]) & set(f["val"]) == set()
        assert set(f["train"]) & set(f["test"]) == set()
        assert set(f["val"]) & set(f["test"]) == set()
        assert len(f["train"]) + len(f["val"]) + len(f["test"]) == n

    return folds
