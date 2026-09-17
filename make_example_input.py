#!/usr/bin/env python3
"""Generates a random (not real-patient) example input, correctly shaped for
predict.py, so you can try the REAL pretrained models end-to-end without
ADNI/ADSP access. Run this after downloading the model bundle (see README).

Usage:
    python make_example_input.py --out example_genome/
"""
import argparse
import os

import numpy as np

IDP_DIMS = {1: 292, 2: 323, 3: 35}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="example_genome")
    ap.add_argument("--n_subjects", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for chr_n in range(1, 23):
        n_snps = sum(1 for _ in open(f"snp_lists/chr{chr_n:02d}.txt"))
        dosage = np.random.randint(0, 3, size=(args.n_subjects, n_snps)).astype(np.float32)
        np.save(os.path.join(args.out, f"chr{chr_n:02d}.npy"), dosage)

    print(f"Wrote random genome input for {args.n_subjects} subject(s) -> {args.out}/")
    print(f"Try:  python predict.py --genome-dir {args.out} --output result.json")


if __name__ == "__main__":
    main()
