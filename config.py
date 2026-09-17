"""
config.py  —  G2DBridge path configuration
===========================================
Edit the variables below to match your local filesystem before running any
training or inference scripts.
"""

import os

# ---------------------------------------------------------------------------
# Raw data roots
# ---------------------------------------------------------------------------

# Directory containing ADNI IDP CSV files (FreeSurfer / DTI / ASHS)
ADNI_IDP_DIR = "/path/to/ADNI_IDP"

# ADNI subject-level diagnosis summary CSV  (columns include RID, visit, DXSUM)
ADNI_DXSUM_CSV = "/path/to/All_Subjects_DXSUM.csv"

# ADNI info CSV  (RID <-> subject-ID mapping with diagnosis)
ADNI_INFO_CSV = "/path/to/ADNI_Info.csv"

# Root directory for per-chromosome PLINK .raw files, organised as:
#   {RAW_FILES_ROOT}/{cohort}/Chr{chr}.{cohort}.raw
#   cohorts: ADNI1, ADNI31, ADNI32, ADNI4, ADSP
RAW_FILES_ROOT = "/path/to/GeneFiles/rawFiles/ADNI134-ADSP"

# Per-chromosome SNP-name list files (ExtractAllele), organised as:
#   {SNP_FILES_ROOT}/{cohort}_ExtractAllele_{chr}.txt
SNP_FILES_ROOT = "/path/to/SNP_files"

# ADSP case/control phenotype file (tab-separated, columns SUBJID, AD, ...)
ADSP_PHENOTYPE_FILE = "/path/to/ADSPCaseControlPhenotypes.txt"

# CSV with important/sentinel SNPs used to fill missing columns
IMPORTANT_SNPS_CSV = "/path/to/important_snps.csv"

# ---------------------------------------------------------------------------
# Output roots  (created automatically at runtime if they don't exist)
# ---------------------------------------------------------------------------

# Stage-1 checkpoints: {STAGE1_OUT_DIR}/idp{1,2,3}/chr{01..22}.pt
# Defaults to models/stage1 so a downloaded pretrained bundle (see README)
# works with no config changes -- point elsewhere to train your own.
STAGE1_OUT_DIR = "./models/stage1"

# Stage-2 checkpoints: {STAGE2_OUT_DIR}/branch_idp{1,2,3}.pt, combo.pt
STAGE2_OUT_DIR = "./models/stage2"

# Generic fallback model directory (used only if a training call doesn't
# pass its own model_dir explicitly)
MODELS_DIR = "./models"

# Temporary directory for Optuna trial checkpoints (cleaned after each trial)
OPTUNA_TMP_DIR = "./models/.optuna_tmp"
