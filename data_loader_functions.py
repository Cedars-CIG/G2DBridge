#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader_functions.py
Low-level data helpers for G2DBridge.

Public API
----------
- get_adni_patient_info()
- getIDPinfo(csvid, pat_dict, return_col=0)
- read_all_raw_data(chr_index, pat_to_label, verbose=0)
- filter_binary_classes(...)

All file paths are read from config.py; edit that file to point to your data.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

__all__ = [
    "get_adni_patient_info",
    "getIDPinfo",
    "read_all_raw_data",
    "filter_binary_classes",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def getSNPnames(file_path: str) -> List[str]:
    snp_names = []
    with open(file_path, "r") as f:
        for line in f:
            name = line.strip()
            if name:
                snp_names.append(name)
    return snp_names


def get_ad_labels(file_path: str) -> Dict[str, str]:
    ad_labels: Dict[str, str] = {}
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            subject_id = row["SUBJID"].strip()
            ad_status = row.get("AD", "").strip()
            if ad_status:
                ad_labels[subject_id] = ad_status
    return ad_labels


def count_effect_allele(subject_id, snp_name: str, df: pd.DataFrame) -> int:
    effect_alleles = {
        "rs7412": "C", "rs429358": "T", "rs4420638": "A", "rs11136000": "C",
        "rs3851179": "C", "rs6656401": "C", "rs744373": "A", "rs75932628": "C",
        "rs11218343": "T",
    }
    row = df[df["subject_id"] == str(subject_id)]
    if row.empty or snp_name not in df.columns:
        return 0
    snp_value = row[snp_name].values[0]
    if pd.isna(snp_value):
        return 0
    return snp_value.count(effect_alleles.get(snp_name, ""))


# ---------------------------------------------------------------------------
# 1) Patient info / labels
# ---------------------------------------------------------------------------

def get_adni_patient_info():
    """
    Parse ADNI diagnosis and info CSVs.

    Returns
    -------
    fid_dict : dict  {visit_id -> label}
    pat_dict : dict  {visit_id -> subject_id}
    pat_to_label : dict  {subject_id -> label (0=CN, 1=AD, 2=other)}
    """
    pat_info_file = config.ADNI_DXSUM_CSV
    info_file_path = config.ADNI_INFO_CSV

    pat_to_label: Dict[str, int] = {}
    pat_dict2: Dict[int, str] = {}

    with open(pat_info_file, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            try:
                diagnosis = int(row[6].strip('"'))
                subject_id = row[1].strip('"')
                visit_id = int(row[2].strip('"'))

                if diagnosis == 3:
                    label_item = 1
                elif diagnosis == 1:
                    label_item = 0
                else:
                    label_item = 2

                if subject_id not in pat_to_label:
                    pat_to_label[subject_id] = label_item
                else:
                    prev = pat_to_label[subject_id]
                    if prev == 0 and label_item in (1, 2):
                        pat_to_label[subject_id] = label_item
                    elif prev == 2 and label_item == 1:
                        pat_to_label[subject_id] = label_item

                pat_dict2[visit_id] = subject_id
            except ValueError:
                print("Skipping invalid row:", row)

    fid_dict: Dict[int, int] = {}
    pat_dict: Dict[int, str] = {}

    with open(info_file_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            try:
                visit_id = int(row[2].strip('"'))
                subject_id = row[1].strip('"')
                if row[6] != '""':
                    fid_dict[visit_id] = pat_to_label.get(subject_id, 2)
                    pat_dict[visit_id] = subject_id
            except ValueError:
                print("Skipping invalid row:", row)

    print("Processing complete.")
    return fid_dict, pat_dict, pat_to_label


# ---------------------------------------------------------------------------
# 2) IDP info
# ---------------------------------------------------------------------------

# Ordered list of ADNI IDP CSV file names; index == csvid argument
_IDP_FILES = [
    "UCSFFSL51ALL_08_01_16_04Sep2024.csv",    # 0
    "UASPMVBM_04Sep2024.csv",                  # 1
    "UCSFFSL51_03_01_22_04Sep2024.csv",        # 2
    "UCSDVOL_04Sep2024.csv",                   # 3
    "UCSFFSL_02_01_16_04Sep2024.csv",          # 4
    "ADNI_DTIROI_MEAN_V2_04Sep2024.csv",       # 5
    "UCSFFSX51_11_08_19_04Sep2024.csv",        # 6
    "ADNI_PICSL_ASHS_04Sep2024.csv",           # 7
    "ADNI_DTIROI_ROBUSTMEAN_V2_04Sep2024.csv", # 8  (IDP set 1 — DTI)
    "ADNI_DTIROI_V1_04Sep2024.csv",            # 9
    "UCSFFSX6_07_06_23_04Sep2024.csv",         # 10 (IDP set 2 — FreeSurfer)
    "UCSFASLFSCBF_08_17_22_04Sep2024.csv",     # 11
    "UCSFFSX51_ADNI1_3T_02_01_16_04Sep2024.csv", # 12
    "UCSFFSX_11_02_15_04Sep2024.csv",          # 13
    "ADNI_PICSLASHS_02_15_22_04Sep2024.csv",   # 14 (IDP set 3 — ASHS)
    "UPENNROI_MARS_06_01_16_04Sep2024.csv",    # 15
    "UCSFFSL51Y1_08_01_16_04Sep2024.csv",      # 16
    "UCSFASLFS_11_02_15_V2_04Sep2024.csv",     # 17
]

_IDP_START_COL = {
    "UCSFFSL51ALL_08_01_16_04Sep2024.csv": 22,
    "UASPMVBM_04Sep2024.csv": 8,
    "UCSFFSL51_03_01_22_04Sep2024.csv": 22,
    "UCSDVOL_04Sep2024.csv": 8,
    "UCSFFSL_02_01_16_04Sep2024.csv": 22,
    "ADNI_DTIROI_MEAN_V2_04Sep2024.csv": 12,
    "UCSFFSX51_11_08_19_04Sep2024.csv": 22,
    "ADNI_PICSL_ASHS_04Sep2024.csv": 22,
    "ADNI_DTIROI_ROBUSTMEAN_V2_04Sep2024.csv": 12,
    "ADNI_DTIROI_V1_04Sep2024.csv": 12,
    "UCSFFSX6_07_06_23_04Sep2024.csv": 22,
    "UCSFASLFSCBF_08_17_22_04Sep2024.csv": 22,
    "UCSFFSX51_ADNI1_3T_02_01_16_04Sep2024.csv": 22,
    "UCSFFSX_11_02_15_04Sep2024.csv": 10,
    "ADNI_PICSLASHS_02_15_22_04Sep2024.csv": 22,
    "UPENNROI_MARS_06_01_16_04Sep2024.csv": 9,
    "UCSFFSL51Y1_08_01_16_04Sep2024.csv": 22,
    "UCSFASLFS_11_02_15_V2_04Sep2024.csv": 22,
}


def getIDPinfo(csvid: int, pat_dict: dict, return_col: int = 0):
    """
    Load IDP features from a CSV file.

    Parameters
    ----------
    csvid : int
        Index into the ordered IDP file list.
    pat_dict : dict
        {visit_id -> subject_id} mapping from get_adni_patient_info().
    return_col : int
        If 1, also return feature column names.

    Returns
    -------
    (all_idp_id, all_idp_data) or (all_idp_id, all_idp_data, feature_names)
    """
    file_name = _IDP_FILES[csvid]
    start_id = _IDP_START_COL[file_name]
    file_path = os.path.join(config.ADNI_IDP_DIR, file_name)

    print(f"Processing IDP file: {file_name}")

    with open(file_path, "r") as f:
        data = f.read().split("\n")

    col_names = data[0]
    data = data[1:]

    all_idp_id: List = []
    all_idp_data: List = []

    rid_index = col_names.split(",").index('"RID"')

    for row in data:
        row_values = row.replace('"', "").split(",")
        try:
            rid_item = int(row.split(",")[rid_index].replace('"', ""))
        except ValueError:
            continue

        float_data = []
        for value in row_values[start_id:-1]:
            try:
                float_data.append(float(value))
            except ValueError:
                float_data.append(np.nan)

        all_idp_data.append(np.asarray(float_data))

        if rid_item in pat_dict:
            all_idp_id.append(pat_dict[rid_item])
        else:
            all_idp_id.append(-1)

    if return_col == 1:
        feature_names = col_names.replace('"', "").split(",")[start_id:-1]
        return all_idp_id, all_idp_data, feature_names

    return all_idp_id, all_idp_data


# ---------------------------------------------------------------------------
# 3) Genotypes + labels
# ---------------------------------------------------------------------------

def read_all_raw_data(chr_index: int, pat_to_label: dict, verbose: int = 0):
    """
    Load per-chromosome PLINK .raw genotype files for all cohorts.

    Parameters
    ----------
    chr_index : int
        Chromosome number (1–22).
    pat_to_label : dict
        {subject_id -> label} from get_adni_patient_info().
    verbose : int
        If > 0, print array shapes.

    Returns
    -------
    (final_data, final_labels, all_data_ids)
        Each is a dict keyed by cohort name.
    """
    raw_root = config.RAW_FILES_ROOT
    snp_root = config.SNP_FILES_ROOT

    raw_files = {
        "ADNI1":  os.path.join(raw_root, "ADNI1",  f"Chr{chr_index}.ADNI.raw"),
        "ADNI31": os.path.join(raw_root, "ADNI3",  f"Chr{chr_index}.ADNI31.raw"),
        "ADNI32": os.path.join(raw_root, "ADNI3",  f"Chr{chr_index}.ADNI32.raw"),
        "ADNI4":  os.path.join(raw_root, "ADNI4",  f"Chr{chr_index}.ADNI.raw"),
        "ADSP":   os.path.join(raw_root, "ADSP",   f"Chr{chr_index}.ADNI.raw"),
    }

    rare_allele_file_paths = {
        "ADNI1":  os.path.join(snp_root, f"ADNI1_ExtractAllele_{chr_index}.txt"),
        "ADNI31": os.path.join(snp_root, f"ADNI31_ExtractAllele_{chr_index}.txt"),
        "ADNI32": os.path.join(snp_root, f"ADNI32_ExtractAllele_{chr_index}.txt"),
        "ADNI4":  os.path.join(snp_root, f"ADNI4_ExtractAllele_{chr_index}.txt"),
        "ADSP":   os.path.join(snp_root, f"ADSP_ExtractAllele_{chr_index}.txt"),
    }

    important_snp_df = pd.read_csv(config.IMPORTANT_SNPS_CSV)

    all_id_data: Dict[str, List] = {}
    all_snp_data: Dict[str, List] = {}
    all_data_ids: Dict[str, List] = {}

    for key, file_path in raw_files.items():
        try:
            with open(file_path) as f:
                adni_genome = f.read().strip().split("\n")

            snp_names = adni_genome[0].split("\t")[6:]
            snp_names = [s[:-2] for s in snp_names]
            all_snp_data[key] = snp_names

            all_id_data[key] = []
            all_data_ids[key] = []
            for row in adni_genome[1:]:
                row_info = row.replace("NA", "0").split("\t")
                all_id_data[key].append(np.asarray(row_info[6:], dtype=int))
                all_data_ids[key].append(row_info[1])
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    final_data: Dict[str, np.ndarray] = {}
    for key in all_snp_data:
        raw_snps = set(all_snp_data[key])
        expected_snp_list = getSNPnames(rare_allele_file_paths[key])
        expected_snps = set(expected_snp_list)
        missing_snps = expected_snps - raw_snps

        if not missing_snps:
            final_data[key] = np.array(all_id_data[key])
        else:
            snp_matrix = np.array(all_id_data[key])
            new_data = np.zeros((snp_matrix.shape[0], len(expected_snps)))
            for x, snp in enumerate(expected_snp_list):
                if snp in raw_snps:
                    snp_index = all_snp_data[key].index(snp)
                    new_data[:, x] = snp_matrix[:, snp_index]
                else:
                    for y in range(len(all_data_ids[key])):
                        new_data[y, x] = count_effect_allele(
                            all_data_ids[key][y], snp, important_snp_df
                        )
            final_data[key] = new_data

    if verbose:
        for k, v in final_data.items():
            print(f"{k} shape: {v.shape}")

    adsp_ad_dict = get_ad_labels(config.ADSP_PHENOTYPE_FILE)

    final_labels: Dict[str, np.ndarray] = {}
    for key in all_data_ids:
        labels = []
        if key == "ADSP":
            for id_item in all_data_ids[key]:
                if id_item in adsp_ad_dict:
                    try:
                        labels.append(int(adsp_ad_dict[id_item]))
                    except Exception:
                        labels.append(3)
                else:
                    labels.append(3)
        else:
            for id_item in all_data_ids[key]:
                labels.append(int(pat_to_label.get(id_item, 3)))
        final_labels[key] = np.asarray(labels)

    # Keep only ADSP subjects not in ADNI (no overlap)
    adsp_selection = np.array(
        [1 if "ADNI" not in sid else 0 for sid in all_data_ids["ADSP"]]
    )
    final_labels["ADSP"] = final_labels["ADSP"][adsp_selection == 1]
    final_data["ADSP"] = final_data["ADSP"][adsp_selection == 1]
    all_data_ids["ADSP"] = [
        sid for sid, sel in zip(all_data_ids["ADSP"], adsp_selection) if sel == 1
    ]

    # Remove ADNI4 subjects already present in ADNI1
    adni1_ids = set(all_data_ids["ADNI1"])
    keep = [
        (i, sid) for i, sid in enumerate(all_data_ids["ADNI4"])
        if sid not in adni1_ids
    ]
    if keep:
        keep_idx = [i for i, _ in keep]
        final_data["ADNI4"] = final_data["ADNI4"][keep_idx]
        final_labels["ADNI4"] = final_labels["ADNI4"][keep_idx]
        all_data_ids["ADNI4"] = np.asarray([sid for _, sid in keep])
    else:
        final_data["ADNI4"] = np.empty((0, final_data["ADNI4"].shape[1]))
        final_labels["ADNI4"] = np.empty(0, dtype=int)
        all_data_ids["ADNI4"] = np.empty(0, dtype=object)

    return final_data, final_labels, all_data_ids


# ---------------------------------------------------------------------------
# 4) Binary-class filter + NaN imputation
# ---------------------------------------------------------------------------

def filter_binary_classes(
    genome_data_list, genome_labels, genome_ids, genome_source,
    idp_data, idp_labels, idp_ids,
):
    """
    Remove samples with labels outside {0, 1} from both genome and IDP views.
    Replace NaN IDP values with column means; drop all-NaN IDP columns.
    """
    valid_genome = np.where((genome_labels == 0) | (genome_labels == 1))[0]
    valid_idp    = np.where((idp_labels   == 0) | (idp_labels   == 1))[0]

    genome_data_filtered   = [chrom[valid_genome] for chrom in genome_data_list]
    genome_labels_filtered = genome_labels[valid_genome]
    genome_ids_filtered    = genome_ids[valid_genome]
    genome_source_filtered = genome_source[valid_genome]

    idp_data_filtered   = idp_data[valid_idp]
    idp_labels_filtered = idp_labels[valid_idp]
    idp_ids_filtered    = idp_ids[valid_idp]

    # Drop all-NaN columns, then mean-impute remaining NaNs
    nan_columns = np.isnan(idp_data_filtered).all(axis=0)
    idp_data_filtered = idp_data_filtered[:, ~nan_columns]
    col_means = np.nanmean(idp_data_filtered, axis=0)
    for col in range(idp_data_filtered.shape[1]):
        mask = np.isnan(idp_data_filtered[:, col])
        idp_data_filtered[mask, col] = col_means[col]

    return (
        genome_data_filtered, genome_labels_filtered,
        genome_ids_filtered, genome_source_filtered,
        idp_data_filtered, idp_labels_filtered, idp_ids_filtered,
    )
