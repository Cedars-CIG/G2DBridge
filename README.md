# G2DBridge

**G2DBridge** links germline genetic variants (SNPs) to imaging-derived phenotypes (IDPs) and, from there, to Alzheimer's Disease (AD) diagnosis risk.

```
SNPs (chr1–22)  →  IDPs (DTI / FreeSurfer / ASHS)  →  AD risk score
     Stage 1                  Stage 2
```

Stage 1 trains one NN per (chromosome, IDP modality) — 66 models — and
averages their predictions. Stage 2 trains one branch NN per IDP modality
on ADNI's real measured IDP, plus a combo NN over the three branches'
concatenated penultimate features. External cohorts (e.g. ADSP) are only
ever scored, never trained on.

## Repository structure

```
G2DBridge_release/
├── config.py                    # All data/model paths — edit before running
├── data_loader_functions.py     # Raw ADNI/ADSP file I/O
├── nn_functions.py              # AdvancedNN architecture + training loops
├── optuna_objectives.py         # Optuna search spaces (regression + classification)
├── early_stopping.py / utils.py
├── cv_split.py                  # Leakage-free 60/20/20 CV splitter
├── stage1_snp_to_idp.py         # Stage 1: one (idp, chromosome) NN
├── fuse_stage1_predictions.py   # Averages the 22 per-chromosome predictions
├── stage2_idp_to_ad.py          # Stage 2: 3 branch NNs + combo NN
├── predict.py                   # Inference: genome or IDP -> AD risk
├── make_example_input.py        # Random (not real-patient) input for a quickstart
├── snp_lists/chr01.txt..chr22.txt   # SNP order expected by each Stage-1 model
└── models/stage1/, models/stage2/   # Pretrained weights (see below, NOT in git)
```

## Setup

```bash
pip install -r requirements.txt
```

### Pretrained real models

The trained Stage-1 (66 models, ~10 GB) and Stage-2 (4 models) weights are
hosted on Google Drive (too large for git):

**[Download link — TBD]**

Unzip so the layout matches `models/stage1/idp{1,2,3}/chr{01..22}.pt` and
`models/stage2/{branch_idp1,branch_idp2,branch_idp3,combo}.pt`.

### Quickstart (real models, random input)

ADNI/ADSP genotypes are access-restricted, so there's no real-patient
example data here — but you can still run the actual pretrained models
end-to-end on a randomly generated (correctly shaped) input:

```bash
python make_example_input.py --out example_genome/
python predict.py --genome-dir example_genome/ --output result.json
```

### Training from scratch

Requires access-restricted ADNI/ADSP genotype (PLINK `.raw`) and IDP files.
Edit **`config.py`** to point at your copies, then:

```bash
# Stage 1: one run per (idp, chromosome) -- 3 x 22 = 66 total
for idp in 1 2 3; do
  for chr in $(seq 1 22); do
    python stage1_snp_to_idp.py --idp $idp --chr $chr --n_trials 50
  done
done
python fuse_stage1_predictions.py --idp 1
python fuse_stage1_predictions.py --idp 2
python fuse_stage1_predictions.py --idp 3

# Stage 2
python stage2_idp_to_ad.py --n_trials 50
```

SNP input order must match `snp_lists/chrNN.txt` for the corresponding
chromosome — this is the order `stage1_snp_to_idp.py` trains on and
`predict.py` expects at inference time. Each line is `SNP_ID<TAB>EffectAllele`
(the allele your dosage value counts, i.e. 0/1/2 copies of that allele).

## Inference

```bash
# From raw genotypes (chr01.npy .. chr22.npy, SNP order = snp_lists/chrNN.txt)
python predict.py --genome-dir patient_snps/ --output result.json

# From IDP values you already have (skips Stage 1)
python predict.py --idp1 dti.npy --idp2 freesurfer.npy --idp3 ashs.npy
```
Outputs `{"ad_risk_probability": [...]}`.

## Citation

> Ozdemir, O. B., Chen, R., Wu, O., & Li, R. (2026). G2DBridge: A Multimodal Framework Linking Genetics to Disease through Imaging Intermediates. *npj Digital Medicine* (in press).

```bibtex
@article{ozdemir2026g2dbridge,
  title   = {G2DBridge: A Multimodal Framework Linking Genetics to Disease through Imaging Intermediates},
  author  = {Ozdemir, Okan Bilge and Chen, Raelynn and Wu, Olivia and Li, Ruowang},
  journal = {npj Digital Medicine},
  year    = {2026},
  note    = {In press}
}
```
