# RESEARCH_LOG.md — Experiment Record

This log is the single source of truth for every meaningful experiment
in this project. It exists so results can be reproduced, compared
honestly, and defended in the viva and the paper.

## How to use this log

1. **Every meaningful experiment gets a unique Experiment ID** before
   it is run. Suggested format: `EXP-YYYYMMDD-NN`
   (e.g. the first run on a given day is `-01`). Planned ablations can
   also reference their design ID (E1–E5, OCR) in the
   "Research Question" field.
2. Copy the template below, fill it in, and append it to the
   **Experiment Entries** section (newest at the bottom).
3. **Status** is one of: `PLANNED`, `RUNNING`, `COMPLETED`, `FAILED`,
   `ABANDONED`. Failed and negative experiments are recorded too.
4. **Only write numbers that were actually measured.** Copy metrics from
   the files the code produced (e.g. `results/metrics/...`) and link
   them under "Artifacts". Never estimate, round favourably, or fill in
   "expected" values.
5. Keep **Validation Metrics** and **Test Metrics** separate. Test
   metrics are only filled in for a final, frozen configuration; write
   `NOT EVALUATED` otherwise.
6. Write observations as observations ("macro-F1 on validation was
   lower than E1") and hypotheses as hypotheses ("this *may* be because
   ..."). Do not claim statistical significance without a test.
7. Do not edit a completed entry's results. If something was wrong,
   add a correction note with the date.

---

## Template

```markdown
### Experiment ID:

Date:

Status:

Research Question:

Dataset:

Dataset Version:

Dataset Source:

Train Split:

Validation Split:

Test Split:

Input Modality:

Preprocessing:

Feature Representation:

Model:

Model Version / Pretrained Checkpoint:

Hyperparameters:

Random Seed:

Training Environment:

Training Details:

Validation Metrics:

Test Metrics:

Runtime / Resource Notes:

Artifacts:

Observations:

Error Analysis Notes:

Limitations:

Conclusion:

Next Experiment:
```

---

## Experiment Entries

_No ML experiments have been run yet. Data-inspection records use the
prefix `INS-` instead of `EXP-` because they train/evaluate nothing._

### Experiment ID: INS-20261010-01

Date: 2026-10-10

Status: COMPLETED

Research Question: Not an experiment. Task 3 dataset inventory: verify what the available data contains before choosing a dataset strategy (supports RQ1–RQ3).

Dataset: MIMIC

Dataset Version: local file `MIMIC2024.csv`, SHA-256 prefix `98dd8843ef81` (no official version string found)

Dataset Source: Paper https://doi.org/10.1145/3656169; download source TO_BE_VERIFIED

Train Split: NOT_APPLICABLE (no split created; file has no split column)

Validation Split: NOT_APPLICABLE

Test Split: NOT_APPLICABLE

Input Modality: Inspection of text + images (no modelling)

Preprocessing: None (raw files read only; `data/raw` fingerprint identical before/after)

Feature Representation: NOT_APPLICABLE

Model: NOT_APPLICABLE

Model Version / Pretrained Checkpoint: NOT_APPLICABLE

Hyperparameters: NOT_APPLICABLE

Random Seed: NOT_APPLICABLE (inspection is deterministic)

Training Environment: Windows 11, Python 3.10.0, pandas 2.3.3, Pillow 12.3.0

Training Details: `python scripts/inspect_datasets.py`

Validation Metrics: NOT_APPLICABLE

Test Metrics: NOT_APPLICABLE

Runtime / Resource Notes: ~12 s for 5,054 records + 5,054 images (438 MB)

Artifacts: `results/metrics/dataset_inspection_mimic.json`, `results/tables/dataset_inventory.csv`, `results/tables/dataset_compatibility.csv`, `docs/DATASET_REPORT.md`

Observations (measured):
- 5,054 records, 5,054 readable JPEG images, 0 missing.
- `Misogyny` 0: 2,539 (50.24%), 1: 2,515 (49.76%).
- 148 empty texts (106 in label 1, 42 in label 0).
- 18 duplicate-text groups (37 rows), 4 with conflicting labels.
- 8 byte-identical image groups, 2 with conflicting primary labels.
- File names perfectly encode the label (`_M_` / `_NM_`).

Error Analysis Notes: NOT_APPLICABLE

Limitations: language profile is heuristic; only exact duplicates detected; label meaning, text origin, licence and official split TO_BE_VERIFIED.

Conclusion: MIMIC is recommended as the single primary dataset for text-only, image-only and multimodal models on the same samples. No merge with other datasets.

Next Experiment: TASK 4: Dataset Cleaning and Final Dataset Strategy (duplicate/conflict policy, empty-text policy, split).

### Experiment ID: INS-20261010-02

Date: 2026-10-10

Status: COMPLETED

Research Question: Not an experiment. Task 4: final dataset strategy decision and deterministic metadata cleaning.

Dataset: MIMIC (decision: single PRIMARY TRAINING DATASET; no datasets merged; `Downloads/DATASET 0x` files NOT USED)

Dataset Version: `MIMIC2024.csv`, SHA-256 `98dd8843ef81…`; licence CC-BY-4.0 and label meaning (1 = misogynistic) per dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card above

Train Split: NOT_APPLICABLE (no split created; the release has no official split)

Validation Split: NOT_APPLICABLE

Test Split: NOT_APPLICABLE

Input Modality: Metadata + image integrity (no modelling)

Preprocessing: None. `original_text` is the raw OCR text, byte-for-byte; images referenced, not copied.

Feature Representation: NOT_APPLICABLE

Model: NOT_APPLICABLE

Model Version / Pretrained Checkpoint: NOT_APPLICABLE

Hyperparameters: NOT_APPLICABLE

Random Seed: NOT_USED (cleaning is deterministic)

Training Environment: Windows 11, Python 3.10.0, pandas 2.3.3, Pillow 12.3.0

Training Details: `python scripts/clean_dataset.py`. Cleaning version 1.0, policy in `src/data/cleaning.py` (`POLICY`).

Validation Metrics: NOT_APPLICABLE

Test Metrics: NOT_APPLICABLE

Runtime / Resource Notes: ~5 s

Artifacts: `data/processed/mimic_cleaned.csv` (git-ignored, SHA-256 `1ff17281346f…`), `results/tables/excluded_samples.csv`, `results/tables/duplicate_decisions.csv`, `results/tables/cleaning_summary.csv`, `results/metrics/mimic_cleaning_fingerprint.json`, `docs/DATA_CLEANING_REPORT.md`

Observations (measured):
- 5,054 → 5,044 samples; 10 excluded:
  - 6 later copies of identical image + text + label;
  - 4 = 2 identical memes annotated with both labels.
- `Misogyny` 0: 2,539 → 2,537; 1: 2,515 → 2,507.
- 148 empty texts kept (`has_text=False`; 106 misogynistic / 42 not).
- 10 same-text groups on different images kept and grouped.
- 23 samples flagged for review.
- All 10 post-cleaning checks passed; two runs gave byte-identical outputs.

Error Analysis Notes: NOT_APPLICABLE

Limitations: exact duplicates only; conflicting identical memes excluded rather than resolved; `ExtractedText` is OCR output, so RQ3's "reference vs OCR" comparison needs reframing.

Conclusion: Cleaned MIMIC metadata supports text-only, image-only and multimodal experiments on the same 5,044 samples and labels.

Next Experiment: TASK 5: Exploratory Data Analysis (EDA).

### Experiment ID: INS-20261010-03

Date: 2026-10-10

Status: COMPLETED

Research Question: Not an experiment. Task 5: exploratory data analysis (supports RQ1–RQ3 design and MSE-1 dataset / model-identification sections).

Dataset: MIMIC, cleaned (5,044 samples)

Dataset Version: `data/processed/mimic_cleaned.csv`, SHA-256 `1ff17281346f…` (Task 4 cleaning v1.0; verified against fingerprint before analysis)

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: NOT_APPLICABLE (no split created)

Validation Split: NOT_APPLICABLE

Test Split: NOT_APPLICABLE

Input Modality: Text statistics + image metadata (no modelling, no embeddings)

Preprocessing: None. EDA-only tokenization (lowercase letter runs) for counting; never written back.

Feature Representation: NOT_APPLICABLE

Model: NOT_APPLICABLE

Model Version / Pretrained Checkpoint: NOT_APPLICABLE

Hyperparameters: NOT_APPLICABLE

Random Seed: NOT_USED (deterministic)

Training Environment: Windows 11, Python 3.10.0, pandas 2.3.3, matplotlib 3.10.9, Pillow 12.3.0

Training Details: `python scripts/run_eda.py`; `notebooks/02_eda.ipynb` (executed end-to-end with nbclient: 14 code cells, 0 errors)

Validation Metrics: NOT_APPLICABLE

Test Metrics: NOT_APPLICABLE

Runtime / Resource Notes: ~27 s (near-duplicate dHash ~20 s)

Artifacts: `results/tables/eda/` (17 CSV), `results/figures/eda/` (13 PNG, 300 dpi), `results/metrics/eda_summary.json`, `docs/EDA_REPORT.md`

Observations (measured):
- **Labels:** 2,537 / 2,507 (ratio 1.012).
- **Text length:** median 55 chars / 11 words (non-empty texts).
- **Vocabulary:** 14,244 EDA token types, 65% occurring once.
- **Script:** about 29% of samples contain Devanagari (heuristic).
- **Text features:** 0 URLs and 0 emojis; uppercase-heavy text 8.1% (misogynistic) vs 2.9%.
- **Watermark handles:** `@doublemeaning` (59) and `@doublemean` (56) occur only in misogynistic samples.
- **Image resolutions:** class-correlated (1440×1800: 57/57 misogynistic).
- **Near-duplicate images:** 55 candidate pairs (dHash ≤ 4), 48 not grouped by Task 4, 4 cross-class.

Error Analysis Notes: NOT_APPLICABLE

Limitations: heuristic language indicators; crude EDA tokenization; near-duplicates are candidates needing manual review; no significance tests; text is OCR output.

Conclusion: Classes are balanced; the main risks are shortcut/leakage features (handles, file names, resolutions, near-duplicate images). These must be addressed in splitting and tested in preprocessing ablations.

Next Experiment: TASK 6: Text Preprocessing and Hinglish Normalization.
