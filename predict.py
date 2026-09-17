#!/usr/bin/env python3
"""Genome -> IDP -> AD risk inference for one or more subjects, using
pre-trained Stage-1 (per-chromosome SNP->IDP) and Stage-2 (IDP->AD) models.

Two ways to call it:
  1. From raw genotypes: pass --genome-dir with chr01.npy .. chr22.npy
     (SNP dosage vectors per subject, column order matching snp_lists/chrNN.txt).
  2. From already-known IDP values (e.g. your own DTI/FreeSurfer/ASHS
     measurements): pass --idp1/--idp2/--idp3 .npy files directly and skip
     Stage 1 entirely.

Usage:
    python predict.py --genome-dir patient_snps/ --output result.json
    python predict.py --idp1 dti.npy --idp2 fs.npy --idp3 ashs.npy
"""
import argparse
import json
import os

import numpy as np
import torch

import config

IDP_DIMS = {1: 292, 2: 323, 3: 35}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_idp_from_genome(genome_dir, idp):
    """Averages 22 per-chromosome Stage-1 models' predictions for one IDP set."""
    preds = []
    for chrn in range(1, 23):
        X = np.load(os.path.join(genome_dir, f"chr{chrn:02d}.npy")).astype(np.float32)
        if X.ndim == 1:
            X = X[None, :]
        model = torch.load(
            os.path.join(config.STAGE1_OUT_DIR, f"idp{idp}", f"chr{chrn:02d}.pt"),
            map_location="cpu", weights_only=False,
        )
        model.eval()
        with torch.no_grad():
            preds.append(model(torch.tensor(X)).numpy())
    return np.mean(preds, axis=0)


def predict_ad_risk(idp_vectors):
    """idp_vectors: dict {1: array(N, 292), 2: array(N, 323), 3: array(N, 35)}."""
    pen = []
    for k in (1, 2, 3):
        model = torch.load(
            os.path.join(config.STAGE2_OUT_DIR, f"branch_idp{k}.pt"),
            map_location="cpu", weights_only=False,
        )
        model.eval()
        with torch.no_grad():
            pen.append(model.forward_features(torch.tensor(idp_vectors[k].astype(np.float32))).numpy())

    combo = torch.load(os.path.join(config.STAGE2_OUT_DIR, "combo.pt"), map_location="cpu", weights_only=False)
    combo.eval()
    with torch.no_grad():
        logits = combo(torch.tensor(np.concatenate(pen, axis=1).astype(np.float32))).numpy().squeeze(-1)
    return sigmoid(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", help="Directory with chr01.npy .. chr22.npy SNP dosage arrays")
    ap.add_argument("--idp1", help="Precomputed DTI IDP .npy (N, 292); skips Stage 1 for this modality")
    ap.add_argument("--idp2", help="Precomputed FreeSurfer IDP .npy (N, 323)")
    ap.add_argument("--idp3", help="Precomputed ASHS IDP .npy (N, 35)")
    ap.add_argument("--output", help="Write JSON result here instead of stdout")
    args = ap.parse_args()

    idp_vectors = {}
    for k, override in ((1, args.idp1), (2, args.idp2), (3, args.idp3)):
        if override:
            idp_vectors[k] = np.load(override)
        elif args.genome_dir:
            idp_vectors[k] = predict_idp_from_genome(args.genome_dir, k)
        else:
            raise SystemExit(f"Need either --genome-dir or --idp{k} for IDP set {k}")

    risk = predict_ad_risk(idp_vectors)
    result = {"ad_risk_probability": risk.tolist()}

    out = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Saved -> {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
