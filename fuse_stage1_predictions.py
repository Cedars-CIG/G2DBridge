#!/usr/bin/env python3
"""Averages each IDP set's 22 per-chromosome Stage-1 predictions into one
fused vector per subject (ADNI held-out test, for a quick R^2 sanity check,
and ADSP, which feeds Stage 2). Run after stage1_snp_to_idp.py has produced
chr01_preds.npz .. chr22_preds.npz for a given --idp.

Usage:
    python fuse_stage1_predictions.py --idp 1
"""
import argparse
import os

import numpy as np
from sklearn.metrics import r2_score

import config


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
    args = ap.parse_args()

    idp_dir = os.path.join(config.STAGE1_OUT_DIR, f"idp{args.idp}")
    preds_test, preds_adsp, y_test, adsp_ids = [], [], None, None
    for chrn in range(1, 23):
        d = np.load(os.path.join(idp_dir, f"chr{chrn:02d}_preds.npz"))
        preds_test.append(d["preds_test"])
        preds_adsp.append(d["preds_adsp"])
        if y_test is None:
            y_test, adsp_ids = d["y_test"], d["adsp_ids"]

    fused_test = np.mean(preds_test, axis=0)
    fused_adsp = np.mean(preds_adsp, axis=0)
    print(f"idp{args.idp}: fused ADNI-test R2={r2_safe(y_test, fused_test):.4f}  "
          f"(mean single-chr R2={np.mean([r2_safe(y_test, p) for p in preds_test]):.4f})")

    np.save(os.path.join(idp_dir, "fused_adsp.npy"), fused_adsp)
    np.save(os.path.join(idp_dir, "fused_adsp_ids.npy"), adsp_ids)
    print(f"Saved {idp_dir}/fused_adsp.npy")


if __name__ == "__main__":
    main()
