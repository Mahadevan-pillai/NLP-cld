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

### Experiment ID: INS-20261012-01

Date: 2026-10-12

Status: COMPLETED

Research Question: Not an experiment. Task 6: build the three controlled text variants for RQ1 (RAW vs BASIC vs HINGLISH_NORMALIZED). Effectiveness NOT measured.

Dataset: MIMIC, cleaned (5,044 samples)

Dataset Version: input `data/processed/mimic_cleaned.csv` SHA-256 `1ff17281346f…` (verified against the Task 4 fingerprint); output `data/processed/mimic_text_preprocessed_v1.0.csv` SHA-256 `51faa312cd1b…` (git-ignored)

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: NOT_APPLICABLE (no split created)

Validation Split: NOT_APPLICABLE

Test Split: NOT_APPLICABLE

Input Modality: Text

Preprocessing: text_preprocessing version 1.0, three modes.
- RAW: unchanged.
- BASIC: NFC, HTML unescape, zero-width/NBSP cleanup, URL → `<URL>`, @mention → `<USER>`, hashtag `#` removed, whitespace collapsed; lowercase off.
- HINGLISH_NORMALIZED: BASIC + elongation 3+ → 2 + rules file `src/preprocessing/resources/hinglish_normalization.yaml` v1.0 (32 rules / 40 variants, each with ≥ 2 corpus occurrences and no ambiguity; rules file SHA-256 `d57d474f5cab…`).

Feature Representation: NOT_APPLICABLE

Model: NOT_APPLICABLE

Model Version / Pretrained Checkpoint: NOT_APPLICABLE

Hyperparameters: NOT_APPLICABLE

Random Seed: NOT_USED (deterministic)

Training Environment: Windows 11, Python 3.10.0, pandas 2.3.3, PyYAML 6.0.3

Training Details: `python scripts/preprocess_text.py`; 9/9 validation checks passed (IDs, labels and image references preserved; `text_raw` == `original_text`; no new empty texts; deterministic)

Validation Metrics: NOT_APPLICABLE

Test Metrics: NOT_APPLICABLE

Runtime / Resource Notes: ~3 s

Artifacts: `results/tables/preprocessing/` (preprocessing_summary, operation_counts, hinglish_rule_frequencies, normalization_audit; preprocessing_examples is git-ignored), `results/metrics/mimic_preprocessing_fingerprint.json`, `docs/TEXT_PREPROCESSING_REPORT.md`

Observations (measured):
- **Samples changed vs RAW:** BASIC 671 (13.30%), HINGLISH_NORMALIZED 1,255 (24.88%); 655 differ between BASIC and HINGLISH.
- **Operations:** 761 rule replacements in 597 samples; 523 mentions masked; 81 elongations shortened.
- **Vocabulary (case-sensitive words):** 20,235 → 19,919 → 19,810.
- **Safety:** 0 texts newly empty.
- **Audit:** 48 samples flagged for review (1 negation-count change caused by masking the handle `@ni`).

Error Analysis Notes: NOT_APPLICABLE

Limitations:
- Rules are derived from the whole corpus (all future splits).
- Rules are context-free.
- No Devanagari↔Latin transliteration.
- OCR artefacts (e.g. `ः`) are not repaired.
- 18 watermark names without "@" remain.

Conclusion: The three variants are ready for E1/E2 and E4/E5. BASIC vs HINGLISH_NORMALIZED is the clean comparison for RQ1, because RAW vs BASIC also includes mention (watermark) masking.

Next Experiment: TASK 7: Leakage-Safe Train/Validation/Test Splitting.

### Experiment ID: INS-20261012-02

Date: 2026-10-12

Status: COMPLETED

Research Question: Not an experiment. Task 7: create the single canonical, leakage-safe split used by all later experiments (text-only, image-only, multimodal, preprocessing ablations).

Dataset: MIMIC, cleaned + preprocessed (5,044 samples)

Dataset Version: `data/processed/mimic_text_preprocessed_v1.0.csv` SHA-256 `51faa312cd1b…` (preprocessing v1.0; raw `MIMIC2024.csv` `98dd8843ef81…`)

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (70.02%): 1,776 non-misogynistic / 1,756 misogynistic

Validation Split: 754 (14.95%): 379 / 375

Test Split: 758 (15.03%): 382 / 376. LOCKED for final evaluation only.

Input Modality: Split by sample_id (applies to text, image and multimodal)

Preprocessing: None (splitting only)

Feature Representation: NOT_APPLICABLE

Model: NOT_APPLICABLE

Model Version / Pretrained Checkpoint: NOT_APPLICABLE

Hyperparameters: sklearn `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`, stratified by target_label; folds 0–2 → test, 3–5 → validation, 6–19 → train

Random Seed: 42

Training Environment: Windows 11, Python 3.10.0, scikit-learn 1.7.2, pandas 2.3.3

Training Details: `python scripts/create_splits.py`.
- Leakage groups formed by union-find over: identical image SHA-256 (none remain), Task 5 near-duplicate pairs (dHash ≤ 4; 55 pairs), and Task 4 identical-caption groups with ≥ 3 words (7 of 10; 3 trivial captions not grouped).
- Result: 43 multi-sample groups (93 samples; largest 7; 4 mixed-label).

Validation Metrics: NOT_APPLICABLE

Test Metrics: NOT_APPLICABLE

Runtime / Resource Notes: ~10 s

Artifacts:
- `data/splits/split_assignments.csv` (canonical, tracked, SHA-256 `47fb02a87b36…`)
- `data/splits/split_manifest.json` (tracked)
- `data/splits/{train,validation,test}.csv` (git-ignored)
- `results/tables/splits/` (`split_distribution.csv`, `leakage_audit.csv`, `leakage_groups.csv`)
- `docs/DATA_SPLIT_REPORT.md`

Observations (measured):
- **Leakage audit:** all 11 blocking checks pass. 0 groups, identical images, near-duplicate pairs or identical records cross splits.
- **Class balance:** every split within 0.13 pp of the overall class balance.
- **Reproducibility:** the in-script re-run is identical, and full regeneration is byte-identical.
- **Watermark samples:** spread 70 / 6 / 14 across train / validation / test (a shortcut, not leakage).

Error Analysis Notes: NOT_APPLICABLE

Limitations:
- Near-duplicate grouping is not manually confirmed.
- The 3-word caption cut-off is a judgment call.
- Shortcut features are present in all splits.
- A single split and seed (no variance estimate).
- The test set is small (758).

Conclusion: The canonical split is ready. All learned preprocessing must be fitted on train only; the test set is locked.

Next Experiment: TASK 8: Traditional Text Baselines (TF-IDF + Logistic Regression + SVM).

### Experiment ID: B01

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only baseline; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included as all-zero vectors)

Preprocessing: `raw` (Task 6 v1.0), applied inside the pipeline from `text_raw`; verified identical to the stored Task 6 column

Feature Representation: TF-IDF word 1–2-grams, `min_df=2`, `sublinear_tf`, lowercase, Devanagari-safe `token_pattern`; 6,042 features (fitted on train only)

Model: TF-IDF + Logistic Regression

Model Version / Pretrained Checkpoint: baseline v1.0 (no pretrained weights)

Hyperparameters: {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "class_weight": null} (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11 CPU, Python 3.10.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_baselines.py`; train time 0.11 s; converged: True; no warnings

Validation Metrics: macro-F1 **0.6538** (95% bootstrap CI 0.619–0.685); accuracy 0.6538; precision (misogynistic) 0.6541; recall 0.6453; weighted-F1 0.6538; TN 251, FP 128, FN 133, TP 242; macro-F1 with text only 0.6456; without watermark 0.6509

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: validation prediction 0.015 s; artifact 181,220 bytes

Artifacts: `models/text/B01_logistic_regression_tfidf_raw_v1.0.joblib` (git-ignored); `results/tables/text_baselines/`; `results/figures/text_baselines/B01_confusion_matrix.png`

Observations: Watermark tokens (`doublemean`, `doublemeaning`, bigrams) are among the top-25 misogynistic features.

Error Analysis Notes: see `docs/TEXT_BASELINE_REPORT.md` §12 (best model B03)

Limitations: untuned; small validation set (CI about ±3.4 pp); shortcut features (watermark pattern, empty text predicted misogynistic, OCR `meः`) are learned

Conclusion: Within the uncertainty of the best baseline.

Next Experiment: TASK 9: Transformer / Pretrained Text Model.

### Experiment ID: B02

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only baseline; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included as all-zero vectors)

Preprocessing: `basic` (Task 6 v1.0), applied inside the pipeline from `text_raw`; verified identical to the stored Task 6 column

Feature Representation: TF-IDF word 1–2-grams, `min_df=2`, `sublinear_tf`, lowercase, Devanagari-safe `token_pattern`; 6,058 features (fitted on train only)

Model: TF-IDF + Logistic Regression

Model Version / Pretrained Checkpoint: baseline v1.0 (no pretrained weights)

Hyperparameters: {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "class_weight": null} (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11 CPU, Python 3.10.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_baselines.py`; train time 0.17 s; converged: True; no warnings

Validation Metrics: macro-F1 **0.6565** (95% bootstrap CI 0.622–0.690); accuracy 0.6565; precision (misogynistic) 0.6559; recall 0.6507; weighted-F1 0.6565; TN 251, FP 128, FN 131, TP 244; macro-F1 with text only 0.6484; without watermark 0.6536

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: validation prediction 0.029 s; artifact 181,700 bytes

Artifacts: `models/text/B02_logistic_regression_tfidf_basic_v1.0.joblib` (git-ignored); `results/tables/text_baselines/`; `results/figures/text_baselines/B02_confusion_matrix.png`

Observations: Mention masking replaces handles, but the placeholder bigrams `user user` and `user in` become top misogynistic features (watermark proxy).

Error Analysis Notes: see `docs/TEXT_BASELINE_REPORT.md` §12 (best model B03)

Limitations: untuned; small validation set (CI about ±3.4 pp); shortcut features (watermark pattern, empty text predicted misogynistic, OCR `meः`) are learned

Conclusion: Within the uncertainty of the best baseline.

Next Experiment: TASK 9: Transformer / Pretrained Text Model.

### Experiment ID: B03

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only baseline; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included as all-zero vectors)

Preprocessing: `hinglish_normalized` (Task 6 v1.0), applied inside the pipeline from `text_raw`; verified identical to the stored Task 6 column

Feature Representation: TF-IDF word 1–2-grams, `min_df=2`, `sublinear_tf`, lowercase, Devanagari-safe `token_pattern`; 6,052 features (fitted on train only)

Model: TF-IDF + Logistic Regression

Model Version / Pretrained Checkpoint: baseline v1.0 (no pretrained weights)

Hyperparameters: {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "class_weight": null} (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11 CPU, Python 3.10.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_baselines.py`; train time 0.27 s; converged: True; no warnings

Validation Metrics: macro-F1 **0.6671** (95% bootstrap CI 0.633–0.701); accuracy 0.6671; precision (misogynistic) 0.6676; recall 0.6587; weighted-F1 0.6671; TN 256, FP 123, FN 128, TP 247; macro-F1 with text only 0.6593; without watermark 0.6643

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: validation prediction 0.036 s; artifact 181,860 bytes

Artifacts: `models/text/B03_logistic_regression_tfidf_hinglish_normalized_v1.0.joblib` (git-ignored); `results/tables/text_baselines/`; `results/figures/text_baselines/B03_confusion_matrix.png`

Observations: Best validation macro-F1 of the six. +0.0106 vs B02 and +0.0133 vs B01; confidence intervals overlap and no significance is claimed. Error rates are highest for Devanagari (39.6%) and very short texts (40.0%); 207/251 errors are shared by all six models.

Error Analysis Notes: see `docs/TEXT_BASELINE_REPORT.md` §12 (best model B03)

Limitations: untuned; small validation set (CI about ±3.4 pp); shortcut features (watermark pattern, empty text predicted misogynistic, OCR `meः`) are learned

Conclusion: Best development baseline.

Next Experiment: TASK 9: Transformer / Pretrained Text Model.

### Experiment ID: B04

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only baseline; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included as all-zero vectors)

Preprocessing: `raw` (Task 6 v1.0), applied inside the pipeline from `text_raw`; verified identical to the stored Task 6 column

Feature Representation: TF-IDF word 1–2-grams, `min_df=2`, `sublinear_tf`, lowercase, Devanagari-safe `token_pattern`; 6,042 features (fitted on train only)

Model: TF-IDF + Linear SVM (LinearSVC)

Model Version / Pretrained Checkpoint: baseline v1.0 (no pretrained weights)

Hyperparameters: {"C": 1.0, "loss": "squared_hinge", "dual": "auto", "max_iter": 10000, "class_weight": null} (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11 CPU, Python 3.10.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_baselines.py`; train time 0.10 s; converged: True; no warnings

Validation Metrics: macro-F1 **0.6512** (95% bootstrap CI 0.614–0.684); accuracy 0.6512; precision (misogynistic) 0.6497; recall 0.6480; weighted-F1 0.6512; TN 248, FP 131, FN 132, TP 243; macro-F1 with text only 0.6430; without watermark 0.6483

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: validation prediction 0.017 s; artifact 181,088 bytes

Artifacts: `models/text/B04_linear_svm_tfidf_raw_v1.0.joblib` (git-ignored); `results/tables/text_baselines/`; `results/figures/text_baselines/B04_confusion_matrix.png`

Observations: SVM on RAW. Same watermark artefacts as B01.

Error Analysis Notes: see `docs/TEXT_BASELINE_REPORT.md` §12 (best model B03)

Limitations: untuned; small validation set (CI about ±3.4 pp); shortcut features (watermark pattern, empty text predicted misogynistic, OCR `meः`) are learned

Conclusion: Within the uncertainty of the best baseline.

Next Experiment: TASK 9: Transformer / Pretrained Text Model.

### Experiment ID: B05

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only baseline; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included as all-zero vectors)

Preprocessing: `basic` (Task 6 v1.0), applied inside the pipeline from `text_raw`; verified identical to the stored Task 6 column

Feature Representation: TF-IDF word 1–2-grams, `min_df=2`, `sublinear_tf`, lowercase, Devanagari-safe `token_pattern`; 6,058 features (fitted on train only)

Model: TF-IDF + Linear SVM (LinearSVC)

Model Version / Pretrained Checkpoint: baseline v1.0 (no pretrained weights)

Hyperparameters: {"C": 1.0, "loss": "squared_hinge", "dual": "auto", "max_iter": 10000, "class_weight": null} (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11 CPU, Python 3.10.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_baselines.py`; train time 0.14 s; converged: True; no warnings

Validation Metrics: macro-F1 **0.6525** (95% bootstrap CI 0.615–0.685); accuracy 0.6525; precision (misogynistic) 0.6507; recall 0.6507; weighted-F1 0.6525; TN 248, FP 131, FN 131, TP 244; macro-F1 with text only 0.6444; without watermark 0.6497

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: validation prediction 0.024 s; artifact 181,552 bytes

Artifacts: `models/text/B05_linear_svm_tfidf_basic_v1.0.joblib` (git-ignored); `results/tables/text_baselines/`; `results/figures/text_baselines/B05_confusion_matrix.png`

Observations: Placeholder bigrams again among the strongest features.

Error Analysis Notes: see `docs/TEXT_BASELINE_REPORT.md` §12 (best model B03)

Limitations: untuned; small validation set (CI about ±3.4 pp); shortcut features (watermark pattern, empty text predicted misogynistic, OCR `meः`) are learned

Conclusion: Within the uncertainty of the best baseline.

Next Experiment: TASK 9: Transformer / Pretrained Text Model.

### Experiment ID: B06

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only baseline; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included as all-zero vectors)

Preprocessing: `hinglish_normalized` (Task 6 v1.0), applied inside the pipeline from `text_raw`; verified identical to the stored Task 6 column

Feature Representation: TF-IDF word 1–2-grams, `min_df=2`, `sublinear_tf`, lowercase, Devanagari-safe `token_pattern`; 6,052 features (fitted on train only)

Model: TF-IDF + Linear SVM (LinearSVC)

Model Version / Pretrained Checkpoint: baseline v1.0 (no pretrained weights)

Hyperparameters: {"C": 1.0, "loss": "squared_hinge", "dual": "auto", "max_iter": 10000, "class_weight": null} (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11 CPU, Python 3.10.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_baselines.py`; train time 0.23 s; converged: True; no warnings

Validation Metrics: macro-F1 **0.6499** (95% bootstrap CI 0.615–0.683); accuracy 0.6499; precision (misogynistic) 0.6480; recall 0.6480; weighted-F1 0.6499; TN 247, FP 132, FN 132, TP 243; macro-F1 with text only 0.6417; without watermark 0.6470

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: validation prediction 0.045 s; artifact 181,728 bytes

Artifacts: `models/text/B06_linear_svm_tfidf_hinglish_normalized_v1.0.joblib` (git-ignored); `results/tables/text_baselines/`; `results/figures/text_baselines/B06_confusion_matrix.png`

Observations: Hinglish normalization did not help the SVM (-0.0026 vs B05).

Error Analysis Notes: see `docs/TEXT_BASELINE_REPORT.md` §12 (best model B03)

Limitations: untuned; small validation set (CI about ±3.4 pp); shortcut features (watermark pattern, empty text predicted misogynistic, OCR `meः`) are learned

Conclusion: Within the uncertainty of the best baseline.

Next Experiment: TASK 9: Transformer / Pretrained Text Model.

### Experiment ID: INS-20261012-03

Date: 2026-10-12

Status: COMPLETED

Research Question: Not an experiment. Task 8.5: can the available annotations support multi-category / hierarchical harmful-content classification? (Feasibility and design only.)

Dataset: MIMIC (cleaned, 5,044). Three other local files (`Downloads/DATASET 00.csv`, `DATASET 01.csv`, `DATASET 02.xlsx`) inspected read-only with the developer's approval: agriculture, mortality statistics and football data. They are not content datasets.

Dataset Version: cleaned SHA-256 `1ff17281346f…`; canonical split `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: label counts only (no training)

Validation Split: label counts only

Test Split: label counts only (descriptive). NO EVALUATION.

Input Modality: Annotations only

Preprocessing: NOT_APPLICABLE

Feature Representation: NOT_APPLICABLE

Model: NOT_APPLICABLE (nothing trained; Task 8 B01–B06 untouched)

Model Version / Pretrained Checkpoint: NOT_APPLICABLE

Hyperparameters: NOT_APPLICABLE

Random Seed: NOT_USED

Training Environment: Windows 11, Python 3.10.0, pandas 2.3.3

Training Details: `python scripts/analyze_category_feasibility.py`

Validation Metrics: NOT_APPLICABLE

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: < 5 s

Artifacts: `results/tables/category_feasibility/` (`all_dataset_labels`, `label_mapping_analysis`, `category_sufficiency`, `subcategory_structure`, `modality_support`), `config/categories.yaml` (proposed), `docs/MULTICATEGORY_FEASIBILITY_REPORT.md`

Observations (measured):
- **Verified labels:** only `Misogyny` (binary) plus `Objectification`, `Prejudice` and `Humiliation`.
- **Nesting:** the sub-labels are strictly nested (0 positives among non-misogynistic memes).
- **Co-occurrence:** 11.45% of misogynistic memes have more than one sub-label, so the formulation is MULTILABEL.
- **Positives (train/validation):** Misogyny 1,756/375 (STRONG); Objectification 1,049/214 (STRONG); Prejudice 642/152 (MODERATE); Humiliation 262/55 (WEAK).
- **Unsupported:** SAFE, HATE, ABUSE, THREAT, OBSCENE and SARCASM have no annotation.

Error Analysis Notes: NOT_APPLICABLE

Limitations:
- Single dataset.
- The meaning of sub-label values is inferred (1 = present) from the dataset card and the nesting.
- Sufficiency tiers are a project judgment.

Conclusion:
- A general safe/harmful or multi-harm hierarchy is NOT supported.
- A two-stage misogyny hierarchy is supported: binary detection, then a conditional multilabel sub-category stage.
- Stage 2 and a narrowed RQ4 await the developer's decision.

Next Experiment: TO BE DECIDED AFTER REVIEW OF MULTICATEGORY_FEASIBILITY_REPORT.md

### Experiment ID: T01

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only; pretrained model vs TF-IDF; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included)

Preprocessing: `raw` (Task 6 v1.0), applied with `preprocess()` from `text_raw`

Feature Representation: MuRIL last-layer attention-mask mean pooling, 768 dimensions, encoder frozen; embeddings cached per variant/split outside the repo

Model: Frozen MuRIL embeddings + StandardScaler + Logistic Regression

Model Version / Pretrained Checkpoint: `google/muril-base-cased` @ `afd9f36c7923d54e97903922ff1b260d091d202f` (weights SHA-256 `e65e3b7aaa3c…`); tokenizer from the same revision

Hyperparameters: {"C": 1.0, "max_iter": 2000, "class_weight": null}; max_length 64 (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, 12 threads), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_transformers.py frozen`; classifier fit 0.48 s; embedding train+validation 165 s; converged: True; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.6151** (95% bootstrap CI 0.580–0.651); accuracy 0.6154; precision (misogynistic) 0.6184; recall 0.5920; weighted-F1 0.6152; TN 242, FP 137, FN 153, TP 222; macro-F1 with text only 0.6054; without watermark 0.6118

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: classifier prediction 0.006 s (embeddings precomputed); artifact 25,937 bytes (+ the shared 953 MB encoder)

Artifacts: `models/text/T01_muril_frozen_logreg_raw_v1.joblib` (git-ignored); embedding cache outside the repo; `results/metrics/text_models/T01.json`; `results/figures/text_models/T01_confusion_matrix.png`

Observations: Frozen mean-pooled MuRIL + LR on RAW text is 0.0387 below the weakest TF-IDF baseline counterpart (B01 0.6538). Lowest of the three frozen variants. FN > FP (153 vs 137).

Error Analysis Notes: see `docs/TRANSFORMER_TEXT_MODEL_REPORT.md` §15

Limitations: untuned; single seed and split; small validation set (CI about ±3.5 pp); OCR text; shortcut features (watermark, empty text) present

Conclusion: Frozen MuRIL embeddings do not beat TF-IDF on RAW text.

Next Experiment: T02 (BASIC) / T03 (HINGLISH_NORMALIZED) frozen; then T04/T05 fine-tuning.

### Experiment ID: T02

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only; pretrained model vs TF-IDF; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included)

Preprocessing: `basic` (Task 6 v1.0), applied with `preprocess()` from `text_raw`

Feature Representation: MuRIL last-layer attention-mask mean pooling, 768 dimensions, encoder frozen; embeddings cached per variant/split outside the repo

Model: Frozen MuRIL embeddings + StandardScaler + Logistic Regression

Model Version / Pretrained Checkpoint: `google/muril-base-cased` @ `afd9f36c7923d54e97903922ff1b260d091d202f` (weights SHA-256 `e65e3b7aaa3c…`); tokenizer from the same revision

Hyperparameters: {"C": 1.0, "max_iter": 2000, "class_weight": null}; max_length 64 (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, 12 threads), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_transformers.py frozen`; classifier fit 0.62 s; embedding train+validation 129 s; converged: True; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.6298** (95% bootstrap CI 0.595–0.661); accuracy 0.6300; precision (misogynistic) 0.6319; recall 0.6133; weighted-F1 0.6299; TN 245, FP 134, FN 145, TP 230; macro-F1 with text only 0.6207; without watermark 0.6267

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: classifier prediction 0.010 s (embeddings precomputed); artifact 25,937 bytes (+ the shared 953 MB encoder)

Artifacts: `models/text/T02_muril_frozen_logreg_basic_v1.joblib` (git-ignored); embedding cache outside the repo; `results/metrics/text_models/T02.json`; `results/figures/text_models/T02_confusion_matrix.png`

Observations: Best frozen variant (+0.0147 vs RAW, +0.0107 vs HINGLISH_NORMALIZED); therefore selected automatically as the T04 text variant. Still 0.0373 below B03 (0.6671). FN > FP (145 vs 134).

Error Analysis Notes: see `docs/TRANSFORMER_TEXT_MODEL_REPORT.md` §15

Limitations: untuned; single seed and split; small validation set (CI about ±3.5 pp); OCR text; shortcut features (watermark, empty text) present

Conclusion: Best frozen configuration, but weaker than every Task 8 baseline; BASIC chosen for T04 by validation.

Next Experiment: T04: fine-tuned MuRIL on BASIC.

### Experiment ID: T03

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only; pretrained model vs TF-IDF; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included)

Preprocessing: `hinglish_normalized` (Task 6 v1.0), applied with `preprocess()` from `text_raw`

Feature Representation: MuRIL last-layer attention-mask mean pooling, 768 dimensions, encoder frozen; embeddings cached per variant/split outside the repo

Model: Frozen MuRIL embeddings + StandardScaler + Logistic Regression

Model Version / Pretrained Checkpoint: `google/muril-base-cased` @ `afd9f36c7923d54e97903922ff1b260d091d202f` (weights SHA-256 `e65e3b7aaa3c…`); tokenizer from the same revision

Hyperparameters: {"C": 1.0, "max_iter": 2000, "class_weight": null}; max_length 64 (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, 12 threads), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_transformers.py frozen`; classifier fit 0.54 s; embedding train+validation 127 s; converged: True; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.6191** (95% bootstrap CI 0.585–0.652); accuracy 0.6194; precision (misogynistic) 0.6222; recall 0.5973; weighted-F1 0.6192; TN 243, FP 136, FN 151, TP 224; macro-F1 with text only 0.6095; without watermark 0.6159

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: classifier prediction 0.010 s (embeddings precomputed); artifact 25,937 bytes (+ the shared 953 MB encoder)

Artifacts: `models/text/T03_muril_frozen_logreg_hinglish_normalized_v1.joblib` (git-ignored); embedding cache outside the repo; `results/metrics/text_models/T03.json`; `results/figures/text_models/T03_confusion_matrix.png`

Observations: Hinglish normalization is -0.0107 vs BASIC and +0.0040 vs RAW at the frozen level (single run; within CIs).

Error Analysis Notes: see `docs/TRANSFORMER_TEXT_MODEL_REPORT.md` §15

Limitations: untuned; single seed and split; small validation set (CI about ±3.5 pp); OCR text; shortcut features (watermark, empty text) present

Conclusion: No benefit from normalization for frozen MuRIL embeddings (not significant either way).

Next Experiment: T05: fine-tuned MuRIL on HINGLISH_NORMALIZED.

### Experiment ID: T04

Date: 2026-10-12

Status: COMPLETED

Research Question: RQ1 (text-only; pretrained model vs TF-IDF; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included)

Preprocessing: `basic` (Task 6 v1.0), applied with `preprocess()` from `text_raw`

Feature Representation: MuRIL WordPiece tokens, max_length 64, dynamic padding

Model: Fine-tuned MuRIL (AutoModelForSequenceClassification, 2-logit head, cross-entropy, softmax score)

Model Version / Pretrained Checkpoint: `google/muril-base-cased` @ `afd9f36c7923d54e97903922ff1b260d091d202f` (weights SHA-256 `e65e3b7aaa3c…`); tokenizer from the same revision

Hyperparameters: {"learning_rate": 2e-05, "weight_decay": 0.01, "batch_size": 16, "max_epochs": 4, "warmup_ratio": 0.1, "gradient_clip_norm": 1.0, "early_stopping_patience": 1, "freeze_word_embeddings": true, "max_length": 64} (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, 12 threads), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_transformers.py finetune --exp T04`; 4 epochs run, best epoch 4 (validation macro-F1); epoch 1: train loss 0.6924, val loss 0.6930, val macro-F1 0.3322; epoch 2: train loss 0.6761, val loss 0.6496, val macro-F1 0.6483; epoch 3: train loss 0.6098, val loss 0.6286, val macro-F1 0.6721; epoch 4: train loss 0.5612, val loss 0.6302, val macro-F1 0.6751; trainable parameters 86,042,882 of 237,557,762; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.6751** (95% bootstrap CI 0.641–0.707); accuracy 0.6751; precision (misogynistic) 0.6693; recall 0.6853; weighted-F1 0.6750; TN 252, FP 127, FN 118, TP 257; macro-F1 with text only 0.6679; without watermark 0.6738

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: training 44.5 min on CPU; validation inference 24.8 s for 754 texts; checkpoint 957 MB

Artifacts: `{MODEL_DIR}/T04_muril_finetuned_basic_v1` (outside the repo, `$NLP_CLD_MODEL_DIR`); `results/tables/text_models/T04_training_history.csv`; `results/metrics/text_models/T04_training.log` (local only, git-ignored); `results/figures/text_models/T04_training_curves.png`; `results/metrics/text_models/T04.json`; `results/figures/text_models/T04_confusion_matrix.png`

Observations: Epoch 1 predicted almost only one class (val macro-F1 0.3322, train loss about ln 2); learning started in epoch 2. Best epoch = last epoch (4); training loss still falling, so not clearly converged. +0.0453 vs frozen BASIC (T02), +0.0080 vs B03 (CIs overlap). Recall > precision (0.685 vs 0.669). Empty text -> misogynistic (all 22).

Error Analysis Notes: see `docs/TRANSFORMER_TEXT_MODEL_REPORT.md` §15

Limitations: untuned; single seed and split; small validation set (CI about ±3.5 pp); OCR text; shortcut features (watermark, empty text) present; word embeddings frozen during fine-tuning

Conclusion: Fine-tuning clearly beats frozen embeddings; roughly on par with the best TF-IDF baseline.

Next Experiment: T05 (same settings, HINGLISH_NORMALIZED).

### Experiment ID: T05

Date: 2026-10-13

Status: COMPLETED

Research Question: RQ1 (text-only; pretrained model vs TF-IDF; effect of Hinglish normalization). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned + preprocessed (Task 6 v1.0)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text only (empty texts included)

Preprocessing: `hinglish_normalized` (Task 6 v1.0), applied with `preprocess()` from `text_raw`

Feature Representation: MuRIL WordPiece tokens, max_length 64, dynamic padding

Model: Fine-tuned MuRIL (AutoModelForSequenceClassification, 2-logit head, cross-entropy, softmax score)

Model Version / Pretrained Checkpoint: `google/muril-base-cased` @ `afd9f36c7923d54e97903922ff1b260d091d202f` (weights SHA-256 `e65e3b7aaa3c…`); tokenizer from the same revision

Hyperparameters: {"learning_rate": 2e-05, "weight_decay": 0.01, "batch_size": 16, "max_epochs": 4, "warmup_ratio": 0.1, "gradient_clip_norm": 1.0, "early_stopping_patience": 1, "freeze_word_embeddings": true, "max_length": 64} (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, 12 threads), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_transformers.py finetune --exp T05`; 4 epochs run, best epoch 3 (validation macro-F1); epoch 1: train loss 0.6930, val loss 0.6915, val macro-F1 0.6133; epoch 2: train loss 0.6773, val loss 0.6512, val macro-F1 0.6554; epoch 3: train loss 0.6329, val loss 0.6247, val macro-F1 0.6869; epoch 4: train loss 0.5921, val loss 0.6217, val macro-F1 0.6788; trainable parameters 86,042,882 of 237,557,762; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.6869** (95% bootstrap CI 0.654–0.719); accuracy 0.6870; precision (misogynistic) 0.6759; recall 0.7120; weighted-F1 0.6868; TN 251, FP 128, FN 108, TP 267; macro-F1 with text only 0.6803; without watermark 0.6858

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: training 33.1 min on CPU; validation inference 24.1 s for 754 texts; checkpoint 957 MB

Artifacts: `{MODEL_DIR}/T05_muril_finetuned_hinglish_normalized_v1` (outside the repo, `$NLP_CLD_MODEL_DIR`); `results/tables/text_models/T05_training_history.csv`; `results/metrics/text_models/T05_training.log` (local only, git-ignored); `results/figures/text_models/T05_training_curves.png`; `results/metrics/text_models/T05.json`; `results/figures/text_models/T05_confusion_matrix.png`

Observations: Best Task 9 and best overall validation text model. +0.0118 vs T04 (BASIC), +0.0198 vs B03, +0.0678 vs frozen T03; CIs overlap with B03 and T04, no significance claimed. Best epoch 3 of 4. Lower error rate than B03 on Devanagari (31.8% vs 39.6%) but higher on negation (39.4% vs 30.3%) and on texts with mentions. Fixes 126 of B03's 251 errors and adds 111 new ones. Softmax scores compressed to 0.33-0.68 and barely separate errors from correct predictions. Empty text -> misogynistic (all 22, score 0.526).

Error Analysis Notes: see `docs/TRANSFORMER_TEXT_MODEL_REPORT.md` §15

Limitations: untuned; single seed and split; small validation set (CI about ±3.5 pp); OCR text; shortcut features (watermark, empty text) present; word embeddings frozen during fine-tuning

Conclusion: Best development text model (validation only). Normalization helps the fine-tuned model by +1.2 pp in this single run; not established.

Next Experiment: TASK 10 - Image-Only Vision Model (text tuning, if any, in Task 12 on validation only).

### Experiment ID: I01

Date: 2026-10-13

Status: COMPLETED

Research Question: RQ2 groundwork (image-only baseline for text vs image vs multimodal). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned (Task 4); images `data/raw/mimic/Files` (read-only, verified unchanged by SHA-256)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NO TEST IMAGE OPENED, NOT EVALUATED.

Input Modality: Image only (no OCR, no caption text, no file name)

Preprocessing: load_image() v1.0 (decode, EXIF orientation, RGB) + SiglipImageProcessorPil from the pinned revision (resize 224x224 bicubic, no crop; rescale 1/255; normalise mean 0.5 / std 0.5)

Feature Representation: Frozen SigLIP vision tower `pooler_output` (attention-pooling head), 768-d; encoder in eval/no-grad; embeddings cached per split outside the repo

Model: frozen SigLIP + Logistic Regression

Model Version / Pretrained Checkpoint: `google/siglip-base-patch16-224` @ `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` (model.safetensors SHA-256 `2c63cb7d1f2e…`), vision tower only

Hyperparameters: {"C": 1.0, "max_iter": 2000, "class_weight": null} with StandardScaler (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_image_models.py train`; converged: True; save→reload reproduces predictions: True; predict_image() reproduces all 754 validation predictions

Validation Metrics: macro-F1 **0.7586** (95% bootstrap CI 0.731–0.789); accuracy 0.7586; precision (misogynistic) 0.7533; recall 0.7653; weighted-F1 0.7586; TN 285, FP 94, FN 88, TP 287; macro-F1 on samples with caption 0.7581; without watermark 0.7567

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: embedding extraction 713 s (train 593 s + validation 120 s) on CPU; classifier fit 0.27 s; classifier inference 5 ms for 754 images; predict_image end-to-end about 210 ms/image; artifact 25,937 bytes

Artifacts: `models/image/I01_siglip_logistic_regression_v1.0.joblib` (git-ignored); `results/metrics/image_model_registry.json`; `results/metrics/image_models/I01.json`; `results/figures/image_models/I01_confusion_matrix.png`

Observations: Best image-only model and best validation model so far: +0.0717 vs T05 (text), +0.0915 vs B03; CIs do not overlap with T05. Errors nearly symmetric (FP 94, FN 88). Not driven by the resolution shortcut (0.753 outside source resolutions) or near-duplicates (no train neighbour with cosine >= 0.95). Complementary to text: errors weakly correlated with T05 (phi 0.134); Objectification recall 0.855 vs T05 0.687, but Prejudice/Humiliation lower than text. Manual review: positive predictions mostly picture a woman (gender-presence shortcut suspected; false positives on harmless photos of women). LR scores extremely polarised and overconfident.

Error Analysis Notes: see `docs/IMAGE_MODEL_REPORT.md` §13–14

Limitations: untuned; frozen encoder (fine-tuning DEFERRED TO GPU); single seed and split; small validation set (CI about ±3 pp); encoder can see text rendered in the image; suspected gender-presence shortcut; 224x224 stretch

Conclusion: Best development image-only model (validation only); strong but shortcut-prone. Motivates fusion.

Next Experiment: TASK 11 - Multimodal Image + Text Fusion Model

### Experiment ID: I02

Date: 2026-10-13

Status: COMPLETED

Research Question: RQ2 groundwork (image-only baseline for text vs image vs multimodal). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned (Task 4); images `data/raw/mimic/Files` (read-only, verified unchanged by SHA-256)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NO TEST IMAGE OPENED, NOT EVALUATED.

Input Modality: Image only (no OCR, no caption text, no file name)

Preprocessing: load_image() v1.0 (decode, EXIF orientation, RGB) + SiglipImageProcessorPil from the pinned revision (resize 224x224 bicubic, no crop; rescale 1/255; normalise mean 0.5 / std 0.5)

Feature Representation: Frozen SigLIP vision tower `pooler_output` (attention-pooling head), 768-d; encoder in eval/no-grad; embeddings cached per split outside the repo

Model: frozen SigLIP + Linear SVM

Model Version / Pretrained Checkpoint: `google/siglip-base-patch16-224` @ `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` (model.safetensors SHA-256 `2c63cb7d1f2e…`), vision tower only

Hyperparameters: {"C": 1.0, "loss": "squared_hinge", "dual": "auto", "max_iter": 10000, "class_weight": null} with StandardScaler (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_image_models.py train`; converged: True; save→reload reproduces predictions: True; predict_image() reproduces all 754 validation predictions

Validation Metrics: macro-F1 **0.7507** (95% bootstrap CI 0.720–0.781); accuracy 0.7507; precision (misogynistic) 0.7493; recall 0.7493; weighted-F1 0.7507; TN 285, FP 94, FN 94, TP 281; macro-F1 on samples with caption 0.7498; without watermark 0.7486

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: embedding extraction 713 s (train 593 s + validation 120 s) on CPU; classifier fit 2.38 s; classifier inference 4 ms for 754 images; predict_image end-to-end about 210 ms/image; artifact 25,789 bytes

Artifacts: `models/image/I02_siglip_linear_svm_v1.0.joblib` (git-ignored); `results/metrics/image_model_registry.json`; `results/metrics/image_models/I02.json`; `results/figures/image_models/I02_confusion_matrix.png`

Observations: 0.0079 below I01 (within CIs). Scores are signed SVM margins, not probabilities.

Error Analysis Notes: see `docs/IMAGE_MODEL_REPORT.md` §13–14

Limitations: untuned; frozen encoder (fine-tuning DEFERRED TO GPU); single seed and split; small validation set (CI about ±3 pp); encoder can see text rendered in the image; suspected gender-presence shortcut; 224x224 stretch

Conclusion: Comparable to I01; LR preferred (best validation macro-F1).

Next Experiment: TASK 11 - Multimodal Image + Text Fusion Model

### Experiment ID: D01

Date: 2026-10-13

Status: COMPLETED (DIAGNOSTIC, not a model)

Research Question: RQ2 groundwork (image-only baseline for text vs image vs multimodal). Development comparison on VALIDATION. Tests the class-correlated resolution shortcut (Task 5).

Dataset: MIMIC, cleaned (Task 4); images `data/raw/mimic/Files` (read-only, verified unchanged by SHA-256)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (canonical Task 7)

Test Split: 758. NOT LOADED, NO TEST IMAGE OPENED, NOT EVALUATED.

Input Modality: Image only (no OCR, no caption text, no file name)

Preprocessing: NOT_APPLICABLE (image header only)

Feature Representation: One-hot of the exact image resolution read from the file header (resolutions seen < 5 times in train grouped); no pixels, no file name, no text

Model: DIAGNOSTIC: exact resolution one-hot + LR

Model Version / Pretrained Checkpoint: none

Hyperparameters: {"OneHotEncoder": {"min_frequency": 5, "handle_unknown": "infrequent_if_exist"}, "LogisticRegression": {"max_iter": 2000}}

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_image_models.py train` (diagnostic part)

Validation Metrics: macro-F1 **0.5153** (95% bootstrap CI 0.479–0.549); accuracy 0.5703; precision (misogynistic) 0.7040; recall 0.2347; weighted-F1 0.5162; TN 342, FP 37, FN 287, TP 88; macro-F1 on samples with caption 0.5247; without watermark 0.5094

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: negligible (header reads only)

Artifacts: `results/metrics/image_models/D01.json`; `results/tables/image_models/resolution_shortcut_check.csv`

Observations: Shortcut diagnostic, not a model. Exact resolution alone: macro-F1 0.5153 overall; 0.903 inside the 53 validation images with train-defined class-correlated resolutions, 0.489 elsewhere. The shortcut exists but covers about 7% of validation; I01 does not depend on it overall.

Error Analysis Notes: see `docs/IMAGE_MODEL_REPORT.md` §13–14

Limitations: untuned; frozen encoder (fine-tuning DEFERRED TO GPU); single seed and split; small validation set (CI about ±3 pp); encoder can see text rendered in the image; suspected gender-presence shortcut; 224x224 stretch

Conclusion: Resolution shortcut is real but small; keep reporting per-resolution results.

Next Experiment: Repeat this check for the Task 11 fusion model.

### Experiment ID: M01

Date: 2026-10-13

Status: COMPLETED

Research Question: RQ2 (does image + text outperform text-only and image-only on the same samples?). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7; all have image and caption entry; 105 empty captions)

Validation Split: 754 (22 empty captions)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Image + text (MIMIC OCR caption; no new OCR). Modalities joined on sample_id (alignment audit: 0 duplicates, 0 missing)

Preprocessing: text `hinglish_normalized` (Task 6 v1.0); image load_image() v1.0 + SigLIP PIL processor (224x224 stretch)

Feature Representation: text: T05 fine-tuned MuRIL pooled [CLS] (input of T05's head), 768-d; image: frozen SigLIP pooler_output (Task 10 cache), 768-d; standardised with TRAIN statistics

Model: early fusion: concat + StandardScaler + LR (concatenation (early))

Model Version / Pretrained Checkpoint: text `google/muril-base-cased` @ `afd9f36c…` fine-tuned as T05; image `google/siglip-base-patch16-224` @ `7fd15f06…` (frozen)

Hyperparameters: {"C": 1.0, "max_iter": 3000, "class_weight": null} with StandardScaler on the 1,536-d concatenation (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_multimodal.py train`; converged: True; train macro-F1 0.9465; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.7679** (95% bootstrap CI 0.736–0.798); accuracy 0.7679; precision (misogynistic) 0.7688; recall 0.7627; weighted-F1 0.7679; TN 293, FP 86, FN 89, TP 286; macro-F1 with non-empty caption 0.7675; without watermark 0.7660

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: classifier training 3.371 s on cached features; classifier inference 0.0431 s for 754 samples; text features T05 413 s / frozen 293 s (train + validation); image features reused from Task 10; artifact 50,513 bytes

Artifacts: `models/multimodal/M01_concat_logreg_finetuned_pooled_v1.0.joblib` (git-ignored); `results/metrics/multimodal_model_registry.json`; `results/metrics/multimodal/M01.json`; `results/figures/multimodal/M01_confusion_matrix.png`

Observations: Early-fusion baseline. +0.0093 vs I01 (image), +0.0810 vs T05 (text) on the same 754 samples. Strong overfitting (train macro-F1 0.9465). Coefficient L2 norm text 10.80 vs image 8.52.

Error Analysis Notes: see `docs/MULTIMODAL_MODEL_REPORT.md` §14–15

Limitations: untuned; single seed and split; epoch selected on validation (optimistic); MLP overfits quickly; uncalibrated scores; inherits image gender-presence and text empty-caption shortcuts

Conclusion: Linear concatenation adds little over the image-only model.

Next Experiment: M02 (neural fusion).

### Experiment ID: M02

Date: 2026-10-13

Status: COMPLETED

Research Question: RQ2 (does image + text outperform text-only and image-only on the same samples?). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7; all have image and caption entry; 105 empty captions)

Validation Split: 754 (22 empty captions)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Image + text (MIMIC OCR caption; no new OCR). Modalities joined on sample_id (alignment audit: 0 duplicates, 0 missing)

Preprocessing: text `hinglish_normalized` (Task 6 v1.0); image load_image() v1.0 + SigLIP PIL processor (224x224 stretch)

Feature Representation: text: T05 fine-tuned MuRIL pooled [CLS] (input of T05's head), 768-d; image: frozen SigLIP pooler_output (Task 10 cache), 768-d; standardised with TRAIN statistics

Model: fusion MLP (fine-tuned text + image) (projection (Linear+LayerNorm per modality) + concat + MLP)

Model Version / Pretrained Checkpoint: text `google/muril-base-cased` @ `afd9f36c…` fine-tuned as T05; image `google/siglip-base-patch16-224` @ `7fd15f06…` (frozen)

Hyperparameters: {"projection_dim": 256, "hidden_dim": 256, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.01, "batch_size": 64, "max_epochs": 50, "early_stopping_patience": 5, "seed": 42}; AdamW; cross-entropy (2 classes); no class weights (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_multimodal.py train`; 7 epochs, early stopping (patience 5), best epoch 2; epoch 1: train loss 0.454, val loss 0.453, val macro-F1 0.7837; epoch 2: train loss 0.329, val loss 0.433, val macro-F1 0.8076; epoch 3: train loss 0.292, val loss 0.453, val macro-F1 0.8064; epoch 4: train loss 0.253, val loss 0.515, val macro-F1 0.7891; epoch 5: train loss 0.229, val loss 0.539, val macro-F1 0.7851; epoch 6: train loss 0.187, val loss 0.591, val macro-F1 0.7904; epoch 7: train loss 0.158, val loss 0.653, val macro-F1 0.7944; parameters 526,594; train macro-F1 0.8915; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.8076** (95% bootstrap CI 0.779–0.834); accuracy 0.8077; precision (misogynistic) 0.8194; recall 0.7867; weighted-F1 0.8076; TN 314, FP 65, FN 80, TP 295; macro-F1 with non-empty caption 0.8027; without watermark 0.8059

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: classifier training 8.83 s on cached features; classifier inference 0.0245 s for 754 samples; text features T05 413 s / frozen 293 s (train + validation); image features reused from Task 10; artifact 2,121,071 bytes

Artifacts: `models/multimodal/M02_fusion_mlp_finetuned_pooled_v1.0` (git-ignored); `results/metrics/multimodal_model_registry.json`; `results/metrics/multimodal/M02.json`; `results/figures/multimodal/M02_confusion_matrix.png`; `results/figures/multimodal/M02_training_curves.png`; `results/tables/multimodal/M02_training_history.csv`

Observations: MAIN multimodal model (declared in advance). +0.0490 vs I01, +0.1207 vs T05 on the same samples; every epoch (0.784-0.808) is above I01. Best epoch 2 of 7 (selected peak about 1.4 pp above the epoch mean 0.794). When T05 and I01 disagree (264 samples) M02 is right in 191, siding with the image 178 times. Reliance diagnostic: image->train mean drops macro-F1 to 0.682, text->train mean to 0.786 (image-led). Validation loss lowest at epoch 2, then rises (overfitting).

Error Analysis Notes: see `docs/MULTIMODAL_MODEL_REPORT.md` §14–15

Limitations: untuned; single seed and split; epoch selected on validation (optimistic); MLP overfits quickly; uncalibrated scores; inherits image gender-presence and text empty-caption shortcuts

Conclusion: Best development model so far together with M03 (validation only); multimodal > image-only > text-only on identical samples. No significance claimed; test not evaluated.

Next Experiment: TASK 12 - Hyperparameter Tuning and Model Optimization

### Experiment ID: M03

Date: 2026-10-13

Status: COMPLETED

Research Question: RQ2 (does image + text outperform text-only and image-only on the same samples?). Development comparison on VALIDATION.

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7; all have image and caption entry; 105 empty captions)

Validation Split: 754 (22 empty captions)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Image + text (MIMIC OCR caption; no new OCR). Modalities joined on sample_id (alignment audit: 0 duplicates, 0 missing)

Preprocessing: text `hinglish_normalized` (Task 6 v1.0); image load_image() v1.0 + SigLIP PIL processor (224x224 stretch)

Feature Representation: text: frozen MuRIL attention-mask mean pooling (T03 representation), 768-d; image: frozen SigLIP pooler_output (Task 10 cache), 768-d; standardised with TRAIN statistics

Model: fusion MLP (frozen text + image) (projection (Linear+LayerNorm per modality) + concat + MLP)

Model Version / Pretrained Checkpoint: text `google/muril-base-cased` @ `afd9f36c…` (frozen); image `google/siglip-base-patch16-224` @ `7fd15f06…` (frozen)

Hyperparameters: {"projection_dim": 256, "hidden_dim": 256, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.01, "batch_size": 64, "max_epochs": 50, "early_stopping_patience": 5, "seed": 42}; AdamW; cross-entropy (2 classes); no class weights (fixed defaults, NOT tuned)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_multimodal.py train`; 13 epochs, early stopping (patience 5), best epoch 8; epoch 1: train loss 0.478, val loss 0.445, val macro-F1 0.7864; epoch 2: train loss 0.333, val loss 0.452, val macro-F1 0.7864; epoch 3: train loss 0.280, val loss 0.463, val macro-F1 0.7984; epoch 4: train loss 0.217, val loss 0.531, val macro-F1 0.7957; epoch 5: train loss 0.183, val loss 0.614, val macro-F1 0.7957; epoch 6: train loss 0.131, val loss 0.645, val macro-F1 0.8062; epoch 7: train loss 0.115, val loss 0.753, val macro-F1 0.7917; epoch 8: train loss 0.104, val loss 0.800, val macro-F1 0.8077; epoch 9: train loss 0.076, val loss 0.928, val macro-F1 0.7891; epoch 10: train loss 0.062, val loss 1.010, val macro-F1 0.7838; epoch 11: train loss 0.060, val loss 0.983, val macro-F1 0.8050; epoch 12: train loss 0.050, val loss 1.153, val macro-F1 0.7719; epoch 13: train loss 0.054, val loss 1.089, val macro-F1 0.7997; parameters 526,594; train macro-F1 0.9918; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.8077** (95% bootstrap CI 0.780–0.836); accuracy 0.8077; precision (misogynistic) 0.8026; recall 0.8133; weighted-F1 0.8077; TN 304, FP 75, FN 70, TP 305; macro-F1 with non-empty caption 0.8045; without watermark 0.8061

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: classifier training 16.1 s on cached features; classifier inference 0.0281 s for 754 samples; text features T05 413 s / frozen 293 s (train + validation); image features reused from Task 10; artifact 2,121,066 bytes

Artifacts: `models/multimodal/M03_fusion_mlp_frozen_mean_v1.0` (git-ignored); `results/metrics/multimodal_model_registry.json`; `results/metrics/multimodal/M03.json`; `results/figures/multimodal/M03_confusion_matrix.png`; `results/figures/multimodal/M03_training_curves.png`; `results/tables/multimodal/M03_training_history.csv`

Observations: Control for the T05 feature bias (T05 train 0.748 vs validation 0.687): same MLP with frozen MuRIL text features. 0.8077, indistinguishable from M02 (0.8076), so the fine-tuned features gave no measurable fusion advantage and no visible harm. Best epoch 8 of 13 while validation loss had risen to 0.80 (overconfident). Relies on the image even more (image->mean 0.455, text->mean 0.760).

Error Analysis Notes: see `docs/MULTIMODAL_MODEL_REPORT.md` §14–15

Limitations: untuned; single seed and split; epoch selected on validation (optimistic); MLP overfits quickly; uncalibrated scores; inherits image gender-presence and text empty-caption shortcuts

Conclusion: Frozen text representation is sufficient inside fusion at this stage.

Next Experiment: TASK 12 - Hyperparameter Tuning and Model Optimization

### Experiment ID: TUNE-A

Date: 2026-10-13

Status: COMPLETED

Research Question: Task 12 hyperparameter search: TF-IDF + Logistic Regression (base B03). Selection on VALIDATION macro-F1.

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (used for selection)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text

Preprocessing: fixed (Task 6 v1.0 / Task 10 v1.0), unchanged across trials

Feature Representation: fixed across trials (see model)

Model: TF-IDF + Logistic Regression on HINGLISH_NORMALIZED text

Model Version / Pretrained Checkpoint: unchanged from the base experiment

Hyperparameters: search space ngram_range {(1,1),(1,2),(1,3)} x min_df {1,2,3} x sublinear_tf {T,F} x C {0.1,0.3,1,3,10} (grid, 90 + baseline)

Random Seed: 42 (all search trials)

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: 91 trials (`results/tables/tuning/hyperparameter_trials.csv`, search `traditional_text`); statuses {'COMPLETED': 91}; validation macro-F1 min 0.5962 / median 0.6512 / max 0.6684; best TT-073 {"C": 1.0, "min_df": 2, "ngram_range": [1, 3], "sublinear_tf": true}

Validation Metrics: best trial TT-073: macro-F1 0.6684, accuracy 0.6684, weighted-F1 0.6684

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: sum of trial times 72.8 s

Artifacts: trial log only; selected model in its own entry

Observations: Baseline TT-000 reproduced B03 (0.6671) exactly. Broad plateau: median trial 0.6512; best TT-073 (1-3-grams, min_df 2, sublinear, C 1) only +0.0013 (one more correct sample). min_df 1 raises train macro-F1 to 0.92-0.99 without validation gain.

Error Analysis Notes: see `docs/HYPERPARAMETER_TUNING_REPORT.md`

Limitations: selection on the same validation set inflates the best score; single split

Conclusion: see the selected-model entry.

Next Experiment: TASK 13 - Final Evaluation, Ablation Study & Result Analysis

### Experiment ID: B07

Date: 2026-10-13

Status: COMPLETED (Task 12 tuned / selected configuration)

Research Question: Task 12: does tuning improve the B03 baseline on VALIDATION?

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (used for selection)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text

Preprocessing: hinglish_normalized (Task 6 v1.0) inside the pipeline

Feature Representation: TF-IDF word 1-3-grams

Model: TF-IDF + Logistic Regression (tuned)

Model Version / Pretrained Checkpoint: none (no pretrained weights)

Hyperparameters: {"C": 1.0, "min_df": 2, "ngram_range": [1, 3], "sublinear_tf": true}; fixed {"max_df": 1.0, "max_features": null, "class_weight": null}

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: selected trial TT-073; rebuilt as the full raw-text pipeline (TextPreprocessor inside): predictions identical to the search run: True; reload reproduces predictions: True; 6650 TF-IDF features

Validation Metrics: macro-F1 **0.6684** (95% bootstrap CI 0.634–0.701); accuracy 0.6684; precision (misogynistic) 0.6694; recall 0.6587; weighted-F1 0.6684; TN 257, FP 122, FN 128, TP 247

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: fit 1.232 s (incl. preprocessing); artifact 204,372 bytes

Artifacts: `models/text/B07_logistic_regression_tfidf_hinglish_normalized_tuned_v1.0.joblib` (git-ignored); `results/metrics/tuning/B07.json`

Observations: Best TF-IDF configuration; +0.0013 vs B03 is not a meaningful improvement. Rebuilt full pipeline gives the same predictions as the search run.

Error Analysis Notes: see `docs/HYPERPARAMETER_TUNING_REPORT.md`

Limitations: validation-selected (optimistic); single split; untested on test

Conclusion: Tuning confirms the Task 8 defaults were near-optimal. Cheap fallback text model.

Next Experiment: TASK 13 - Final Evaluation, Ablation Study & Result Analysis

### Experiment ID: T06

Date: 2026-10-13

Status: COMPLETED (Task 12 tuned / selected configuration)

Research Question: Task 12: does tuning improve the T05 baseline on VALIDATION?

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (used for selection)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text

Preprocessing: hinglish_normalized (Task 6 v1.0)

Feature Representation: MuRIL WordPiece, max_length 64

Model: fine-tuned MuRIL (T05 setup, learning rate 3e-05)

Model Version / Pretrained Checkpoint: `google/muril-base-cased` @ `afd9f36c…`

Hyperparameters: {"learning_rate": 3e-05, "weight_decay": 0.01, "batch_size": 16, "max_epochs": 4, "warmup_ratio": 0.1, "gradient_clip_norm": 1.0, "early_stopping_patience": 1, "freeze_word_embeddings": true, "max_length": 64, "seed": 42, "max_batches_per_epoch": null} (only learning_rate changed vs T05)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_transformers.py finetune --exp T05 --run-id T06 --learning-rate 3e-05 --out tuning`; 2 epochs, best epoch 1; epoch 1: train loss 0.693, val loss 0.693, val macro-F1 0.3322; epoch 2: train loss 0.693, val loss 0.693, val macro-F1 0.3322; train macro-F1 0.3327; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.3322** (95% bootstrap CI 0.316–0.348); accuracy 0.4973; precision (misogynistic) 0.4973; recall 1.0000; weighted-F1 0.3304; TN 0, FP 379, FN 0, TP 375

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: training 29.8 min on CPU (see report §16 on machine load); checkpoint 957 MB

Artifacts: `{MODEL_DIR}/T06_muril_finetuned_hinglish_normalized_v1` (outside the repo); `results/metrics/tuning/T06.json`

Observations: Learning rate 3e-5 (only change vs T05). DEGENERATE: predicts every validation meme as misogynistic (TP 375, FP 379); train macro-F1 0.3327; training loss stayed at ln 2 for both epochs; stopped by patience 1 after epoch 2. First epoch slowed by a VM sharing the machine.

Error Analysis Notes: see `docs/HYPERPARAMETER_TUNING_REPORT.md`

Limitations: validation-selected (optimistic); single split; untested on test

Conclusion: Failed run (no learning); not registered for inference; T05 kept.

Next Experiment: TASK 13 - Final Evaluation, Ablation Study & Result Analysis

### Experiment ID: T07

Date: 2026-10-13

Status: COMPLETED (Task 12 tuned / selected configuration)

Research Question: Task 12: does tuning improve the T05 baseline on VALIDATION?

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (used for selection)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Text

Preprocessing: hinglish_normalized (Task 6 v1.0)

Feature Representation: MuRIL WordPiece, max_length 64

Model: fine-tuned MuRIL (T05 setup, learning rate 5e-05)

Model Version / Pretrained Checkpoint: `google/muril-base-cased` @ `afd9f36c…`

Hyperparameters: {"learning_rate": 5e-05, "weight_decay": 0.01, "batch_size": 16, "max_epochs": 4, "warmup_ratio": 0.1, "gradient_clip_norm": 1.0, "early_stopping_patience": 1, "freeze_word_embeddings": true, "max_length": 64, "seed": 42, "max_batches_per_epoch": null} (only learning_rate changed vs T05)

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: `python scripts/train_text_transformers.py finetune --exp T05 --run-id T07 --learning-rate 5e-05 --out tuning`; 3 epochs, best epoch 2; epoch 1: train loss 0.693, val loss 0.693, val macro-F1 0.3322; epoch 2: train loss 0.693, val loss 0.693, val macro-F1 0.3345; epoch 3: train loss 0.693, val loss 0.693, val macro-F1 0.3345; train macro-F1 0.3346; save→reload reproduces predictions: True

Validation Metrics: macro-F1 **0.3345** (95% bootstrap CI 0.318–0.350); accuracy 0.5027; precision (misogynistic) 0.0000; recall 0.0000; weighted-F1 0.3363; TN 379, FP 0, FN 375, TP 0

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: training 25.2 min on CPU (see report §16 on machine load); checkpoint 957 MB

Artifacts: `{MODEL_DIR}/T07_muril_finetuned_hinglish_normalized_v1` (outside the repo); `results/metrics/tuning/T07.json`

Observations: Learning rate 5e-5 (only change vs T05). DEGENERATE: predicts every validation meme as non-misogynistic; train macro-F1 0.3346; training loss at ln 2 for 3 epochs; stopped by patience 1 after epoch 3.

Error Analysis Notes: see `docs/HYPERPARAMETER_TUNING_REPORT.md`

Limitations: validation-selected (optimistic); single split; untested on test

Conclusion: Failed run (no learning); not registered for inference; T05 kept as FINAL_TEXT_MODEL.

Next Experiment: TASK 13 - Final Evaluation, Ablation Study & Result Analysis

### Experiment ID: TUNE-C

Date: 2026-10-13

Status: COMPLETED

Research Question: Task 12 hyperparameter search: classifier on frozen SigLIP embeddings (base I01). Selection on VALIDATION macro-F1.

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (used for selection)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Image

Preprocessing: fixed (Task 6 v1.0 / Task 10 v1.0), unchanged across trials

Feature Representation: fixed across trials (see model)

Model: StandardScaler + LR / Linear SVM on cached SigLIP pooler_output (768-d)

Model Version / Pretrained Checkpoint: unchanged from the base experiment

Hyperparameters: search space LR C {0.001..3} (8) + Linear SVM C {0.0003..1} (8) (grid, 16 + baseline)

Random Seed: 42 (all search trials)

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: 17 trials (`results/tables/tuning/hyperparameter_trials.csv`, search `image`); statuses {'COMPLETED': 17}; validation macro-F1 min 0.7507 / median 0.7652 / max 0.7997; best TI-001 {"C": 0.001, "classifier": "logistic_regression"}

Validation Metrics: best trial TI-001: macro-F1 0.7997, accuracy 0.7997, weighted-F1 0.7997

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: sum of trial times 49.8 s

Artifacts: trial log only; selected model in its own entry

Observations: Baseline TI-000 reproduced I01 (0.7586) exactly. Validation macro-F1 increases steadily as C decreases for both LR and SVM; the best values (LR 0.001, SVM 0.0003) are at the lower edge of the pre-declared grid.

Error Analysis Notes: see `docs/HYPERPARAMETER_TUNING_REPORT.md`

Limitations: selection on the same validation set inflates the best score; single split

Conclusion: see the selected-model entry.

Next Experiment: TASK 13 - Final Evaluation, Ablation Study & Result Analysis

### Experiment ID: I03

Date: 2026-10-13

Status: COMPLETED (Task 12 tuned / selected configuration)

Research Question: Task 12: does tuning improve the I01 baseline on VALIDATION?

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (used for selection)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Image only

Preprocessing: load_image() v1.0 + SigLIP PIL processor (224x224 stretch)

Feature Representation: SigLIP pooler_output, 768-d (Task 10 cache)

Model: frozen SigLIP + logistic_regression (tuned)

Model Version / Pretrained Checkpoint: `google/siglip-base-patch16-224` @ `7fd15f06…` (frozen)

Hyperparameters: {"C": 0.001, "max_iter": 2000, "class_weight": null}

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: selected trial TI-001; train macro-F1 0.8494; reload reproduces predictions: True

Validation Metrics: macro-F1 **0.7997** (95% bootstrap CI 0.772–0.829); accuracy 0.7997; precision (misogynistic) 0.8043; recall 0.7893; weighted-F1 0.7997; TN 307, FP 72, FN 79, TP 296

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: fit 0.309 s on cached embeddings; artifact 25,937 bytes

Artifacts: `models/image/I03_siglip_logistic_regression_tuned_v1.0.joblib` (git-ignored); `results/metrics/tuning/I03.json`

Observations: +0.0411 vs I01 with the same embeddings and cost; train-validation gap shrinks from 0.14 to 0.05 (I01 was overfitted). Best C at the grid boundary (smaller C untested).

Error Analysis Notes: see `docs/HYPERPARAMETER_TUNING_REPORT.md`

Limitations: validation-selected (optimistic); single split; untested on test

Conclusion: Selected as FINAL_IMAGE_MODEL.

Next Experiment: TASK 13 - Final Evaluation, Ablation Study & Result Analysis

### Experiment ID: TUNE-D

Date: 2026-10-13

Status: COMPLETED

Research Question: Task 12 hyperparameter search: fusion MLP (base M02), random search + seed robustness. Selection on VALIDATION macro-F1.

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (used for selection)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Image + text

Preprocessing: fixed (Task 6 v1.0 / Task 10 v1.0), unchanged across trials

Feature Representation: fixed across trials (see model)

Model: fusion MLP on T05 pooled text + SigLIP image features

Model Version / Pretrained Checkpoint: unchanged from the base experiment

Hyperparameters: search space learning_rate {3e-4,1e-3,3e-3} x dropout {0.1,0.3,0.5} x hidden {128,256,512} x projection {128,256} x weight_decay {1e-3,1e-2,1e-1} x batch {32,64,128} (486 points; 30 sampled + baseline); early stopping on validation macro-F1, patience 5, min improvement 0.0

Random Seed: 42 (all search trials)

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: 31 trials (`results/tables/tuning/hyperparameter_trials.csv`, search `multimodal_random_search`); statuses {'EARLY_STOPPED': 31}; validation macro-F1 min 0.8050 / median 0.8101 / max 0.8183; best TM-007 {"batch_size": 64, "dropout": 0.1, "hidden_dim": 512, "learning_rate": 0.0003, "projection_dim": 256, "weight_decay": 0.1}

Validation Metrics: best trial TM-007: macro-F1 0.8183, accuracy 0.8183, weighted-F1 0.8183

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: sum of trial times 234.6 s

Artifacts: trial log only; selected model in its own entry

Observations: Baseline TM-000 reproduced M02 (0.8076) exactly. Flat landscape: all 31 configurations 0.805-0.818 at seed 42; only learning rate 3e-4 is mildly better on average. All trials early-stopped (7-12 epochs, best epochs 2-7). Ran with 4 threads concurrently with the MuRIL sweep.

Error Analysis Notes: see `docs/HYPERPARAMETER_TUNING_REPORT.md`

Limitations: selection on the same validation set inflates the best score; single split

Conclusion: see the selected-model entry.

Next Experiment: TASK 13 - Final Evaluation, Ablation Study & Result Analysis

### Experiment ID: M04

Date: 2026-10-13

Status: COMPLETED (Task 12 tuned / selected configuration)

Research Question: Task 12: does tuning improve the M02 baseline on VALIDATION?

Dataset: MIMIC, cleaned (Task 4) + preprocessed text (Task 6 v1.0) + images (read-only)

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (canonical Task 7)

Validation Split: 754 (used for selection)

Test Split: 758. NOT LOADED, NOT EVALUATED.

Input Modality: Image + text

Preprocessing: text hinglish_normalized (Task 6 v1.0); image load_image() v1.0 + SigLIP processor

Feature Representation: T05 pooled [CLS] 768-d + SigLIP 768-d, train-standardised

Model: fusion MLP (tuned configuration)

Model Version / Pretrained Checkpoint: text MuRIL `afd9f36c…` fine-tuned as T05 (pooled features); image SigLIP `7fd15f06…` (frozen)

Hyperparameters: {"projection_dim": 256, "hidden_dim": 512, "dropout": 0.1, "learning_rate": 0.0003, "weight_decay": 0.1, "batch_size": 64, "max_epochs": 50, "early_stopping_patience": 5, "min_improvement": 0.0, "seed": 42}

Random Seed: 42

Training Environment: Windows 11, CPU only (i5-13500H, no CUDA), Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2

Training Details: selected trial TM-007; 8 epochs, best epoch 3; train macro-F1 0.9119; seeds: baseline config 0.8059 ± 0.0055, tuned config 0.8081 ± 0.0133 (seeds 42/123/2026); adopted by the pre-declared rule: tuned; reload reproduces predictions: True

Validation Metrics: macro-F1 **0.8183** (95% bootstrap CI 0.790–0.846); accuracy 0.8183; precision (misogynistic) 0.8199; recall 0.8133; weighted-F1 0.8183; TN 312, FP 67, FN 70, TP 305

Test Metrics: NOT EVALUATED

Runtime / Resource Notes: training 9.08 s on cached features (4 threads, concurrent with the MuRIL sweep); artifact 2,648,474 bytes

Artifacts: `models/multimodal/M04_fusion_mlp_finetuned_pooled_tuned_v1.0` (git-ignored, safetensors + JSON); `results/metrics/tuning/M04.json`

Observations: TM-007 (lr 3e-4, dropout 0.1, hidden 512, projection 256, weight decay 0.1, batch 64). Seed 42: 0.8183 (+0.0107), but 3-seed mean 0.8081 +/- 0.0133 vs baseline config 0.8059 +/- 0.0055 (+0.0022, within noise; less stable). Same config with frozen text: 0.8028 +/- 0.0020. Still overfits after epoch 2-3. Only +0.008 (seed mean) above the tuned image model I03.

Error Analysis Notes: see `docs/HYPERPARAMETER_TUNING_REPORT.md`

Limitations: validation-selected (optimistic); single split; untested on test

Conclusion: Adopted by the pre-declared rule as FINAL_MULTIMODAL_MODEL; improvement over M02 not meaningful; M02 equivalent.

Next Experiment: TASK 13 - Final Evaluation, Ablation Study & Result Analysis

### Experiment ID: FINAL-T05

Date: 2026-10-13

Status: COMPLETED: **FINAL TEST EVALUATION** (FINAL_TEXT_MODEL)

Research Question: RQ2/RQ3 (final modality comparison) and RQ1 (preprocessing ablation), evaluated once on TEST with a frozen model

Dataset: MIMIC (cleaned, Task 4) with Task 6 v1.0 text and the original images

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (used only to fit the frozen models, before Task 13)

Validation Split: 754 (used only to reproduce stored results at the freeze)

Test Split: 758 (382 non-misogynistic / 376 misogynistic). **EVALUATED ONCE (FINAL TEST EVALUATION)** after the freeze commit `ed0de53`

Input Modality: text

Preprocessing: frozen (text v1.0, image loader v1.0, pil backend)

Feature Representation: as in the frozen model (fine-tuned MuRIL, HINGLISH_NORMALIZED)

Model: T05: fine-tuned MuRIL, HINGLISH_NORMALIZED

Model Version / Pretrained Checkpoint: artifact SHA-256 `697692cc1d9e111f…`; encoders google/muril-base-cased@afd9f36c7923d54e97903922ff1b260d091d202f / google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed (as applicable)

Hyperparameters: frozen (see `results/final/model_freeze_manifest.json` and `config.yaml` tuning.selected)

Random Seed: 42 (bootstrap: 10,000 resamples, seed 42)

Training Environment: no training in Task 13 (evaluation only; CPU, Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2)

Training Details: none; decision rule argmax (frozen)

Validation Metrics: macro-F1 0.6869 (reproduced by the evaluation code at the freeze)

Test Metrics: **macro-F1 0.6476 (95% bootstrap CI 0.6133–0.6815)**; accuracy 0.6478; precision (misogynistic) 0.6373; recall 0.6729; weighted-F1 0.6476; TN 238, FP 144, FN 123, TP 253; macro-F1 with non-empty caption 0.6540 (n 737); without watermark 0.6425 (n 744)

Runtime / Resource Notes: evaluation of all 19 frozen models about 8 min on CPU

Artifacts: `results/final/metrics/T05.json`; `results/predictions/final/T05_test_predictions.csv` (git-ignored); `results/figures/final/T05_test_confusion_matrix.png`; `results/final/final_model_registry.json`

Observations: Validation 0.6869 -> test 0.6476 (-3.9 pp). On TEST its BASIC twin T04 scores 0.6736 (T05 - T04 = -0.026, CI -0.055 to +0.002); T05 is kept as the frozen final model. Leans misogynistic (FP 144 > FN 123).

Error Analysis Notes: see `docs/FINAL_EVALUATION_REPORT.md` §12–14

Limitations: single split and seed; MIMIC OCR captions; uncalibrated scores; known shortcuts not removed

Conclusion: reported as the frozen final result; no development after TEST

Next Experiment: TASK 14 - OCR Pipeline & OCR Robustness Experiment

### Experiment ID: FINAL-I03

Date: 2026-10-13

Status: COMPLETED: **FINAL TEST EVALUATION** (FINAL_IMAGE_MODEL)

Research Question: RQ2/RQ3 (final modality comparison) and RQ1 (preprocessing ablation), evaluated once on TEST with a frozen model

Dataset: MIMIC (cleaned, Task 4) with Task 6 v1.0 text and the original images

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (used only to fit the frozen models, before Task 13)

Validation Split: 754 (used only to reproduce stored results at the freeze)

Test Split: 758 (382 non-misogynistic / 376 misogynistic). **EVALUATED ONCE (FINAL TEST EVALUATION)** after the freeze commit `ed0de53`

Input Modality: image

Preprocessing: frozen (text v1.0, image loader v1.0, pil backend)

Feature Representation: as in the frozen model (frozen SigLIP + LR (C=0.001, tuned))

Model: I03: frozen SigLIP + LR (C=0.001, tuned)

Model Version / Pretrained Checkpoint: artifact SHA-256 `73c5497e95dad8f8…`; encoders google/muril-base-cased@afd9f36c7923d54e97903922ff1b260d091d202f / google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed (as applicable)

Hyperparameters: frozen (see `results/final/model_freeze_manifest.json` and `config.yaml` tuning.selected)

Random Seed: 42 (bootstrap: 10,000 resamples, seed 42)

Training Environment: no training in Task 13 (evaluation only; CPU, Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2)

Training Details: none; decision rule argmax (frozen)

Validation Metrics: macro-F1 0.7997 (reproduced by the evaluation code at the freeze)

Test Metrics: **macro-F1 0.8047 (95% bootstrap CI 0.7757–0.8324)**; accuracy 0.8047; precision (misogynistic) 0.8098; recall 0.7926; weighted-F1 0.8047; TN 312, FP 70, FN 78, TP 298; macro-F1 with non-empty caption 0.8045 (n 737); without watermark 0.8008 (n 744)

Runtime / Resource Notes: evaluation of all 19 frozen models about 8 min on CPU

Artifacts: `results/final/metrics/I03.json`; `results/predictions/final/I03_test_predictions.csv` (git-ignored); `results/figures/final/I03_test_confusion_matrix.png`; `results/final/final_model_registry.json`

Observations: Best TEST macro-F1 of all 19 models. Validation 0.7997 -> test 0.8047 (stable). Tuning generalised: I03 - I01 = +0.028 (CI +0.003 to +0.054). Balanced errors (FP 70, FN 78).

Error Analysis Notes: see `docs/FINAL_EVALUATION_REPORT.md` §12–14

Limitations: single split and seed; MIMIC OCR captions; uncalibrated scores; known shortcuts not removed

Conclusion: reported as the frozen final result; no development after TEST

Next Experiment: TASK 14 - OCR Pipeline & OCR Robustness Experiment

### Experiment ID: FINAL-M04

Date: 2026-10-13

Status: COMPLETED: **FINAL TEST EVALUATION** (FINAL_MULTIMODAL_MODEL)

Research Question: RQ2/RQ3 (final modality comparison) and RQ1 (preprocessing ablation), evaluated once on TEST with a frozen model

Dataset: MIMIC (cleaned, Task 4) with Task 6 v1.0 text and the original images

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (used only to fit the frozen models, before Task 13)

Validation Split: 754 (used only to reproduce stored results at the freeze)

Test Split: 758 (382 non-misogynistic / 376 misogynistic). **EVALUATED ONCE (FINAL TEST EVALUATION)** after the freeze commit `ed0de53`

Input Modality: image+text

Preprocessing: frozen (text v1.0, image loader v1.0, pil backend)

Feature Representation: as in the frozen model (fusion MLP, T05 text, tuned config TM-007)

Model: M04: fusion MLP, T05 text, tuned config TM-007

Model Version / Pretrained Checkpoint: artifact SHA-256 `5b132e127981eb73…`; encoders google/muril-base-cased@afd9f36c7923d54e97903922ff1b260d091d202f / google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed (as applicable)

Hyperparameters: frozen (see `results/final/model_freeze_manifest.json` and `config.yaml` tuning.selected)

Random Seed: 42 (bootstrap: 10,000 resamples, seed 42)

Training Environment: no training in Task 13 (evaluation only; CPU, Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2)

Training Details: none; decision rule argmax (frozen)

Validation Metrics: macro-F1 0.8183 (reproduced by the evaluation code at the freeze)

Test Metrics: **macro-F1 0.8031 (95% bootstrap CI 0.7741–0.8308)**; accuracy 0.8034; precision (misogynistic) 0.8215; recall 0.7713; weighted-F1 0.8032; TN 319, FP 63, FN 86, TP 290; macro-F1 with non-empty caption 0.8056 (n 737); without watermark 0.7991 (n 744)

Runtime / Resource Notes: evaluation of all 19 frozen models about 8 min on CPU

Artifacts: `results/final/metrics/M04.json`; `results/predictions/final/M04_test_predictions.csv` (git-ignored); `results/figures/final/M04_test_confusion_matrix.png`; `results/final/final_model_registry.json`

Observations: Validation 0.8183 (seed 42; 3-seed mean 0.8081) -> test 0.8031. vs T05 +0.1555 (CI +0.119 to +0.191; Holm-adjusted McNemar p 2e-16: significant by the frozen rule). vs I03 -0.0015 (CI -0.025 to +0.021; p 1.0). With image replaced by its train mean it reproduces T05's test predictions exactly (758/758); with text replaced it scores 0.8021. Precision-oriented (FP 63, FN 86).

Error Analysis Notes: see `docs/FINAL_EVALUATION_REPORT.md` §12–14

Limitations: single split and seed; MIMIC OCR captions; uncalibrated scores; known shortcuts not removed

Conclusion: reported as the frozen final result; no development after TEST

Next Experiment: TASK 14 - OCR Pipeline & OCR Robustness Experiment

### Experiment ID: FINAL-B07

Date: 2026-10-13

Status: COMPLETED: **FINAL TEST EVALUATION** (traditional text baseline)

Research Question: RQ2/RQ3 (final modality comparison) and RQ1 (preprocessing ablation), evaluated once on TEST with a frozen model

Dataset: MIMIC (cleaned, Task 4) with Task 6 v1.0 text and the original images

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (used only to fit the frozen models, before Task 13)

Validation Split: 754 (used only to reproduce stored results at the freeze)

Test Split: 758 (382 non-misogynistic / 376 misogynistic). **EVALUATED ONCE (FINAL TEST EVALUATION)** after the freeze commit `ed0de53`

Input Modality: text

Preprocessing: frozen (text v1.0, image loader v1.0, pil backend)

Feature Representation: as in the frozen model (TF-IDF + LR, HINGLISH_NORMALIZED (tuned))

Model: B07: TF-IDF + LR, HINGLISH_NORMALIZED (tuned)

Model Version / Pretrained Checkpoint: artifact SHA-256 `a2aaae3912886379…`; encoders google/muril-base-cased@afd9f36c7923d54e97903922ff1b260d091d202f / google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed (as applicable)

Hyperparameters: frozen (see `results/final/model_freeze_manifest.json` and `config.yaml` tuning.selected)

Random Seed: 42 (bootstrap: 10,000 resamples, seed 42)

Training Environment: no training in Task 13 (evaluation only; CPU, Python 3.10.0, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-learn 1.7.2)

Training Details: none; decision rule argmax (frozen)

Validation Metrics: macro-F1 0.6684 (reproduced by the evaluation code at the freeze)

Test Metrics: **macro-F1 0.6359 (95% bootstrap CI 0.6015–0.6700)**; accuracy 0.6359; precision (misogynistic) 0.6323; recall 0.6356; weighted-F1 0.6359; TN 243, FP 139, FN 137, TP 239; macro-F1 with non-empty caption 0.6417 (n 737); without watermark 0.6288 (n 744)

Runtime / Resource Notes: evaluation of all 19 frozen models about 8 min on CPU

Artifacts: `results/final/metrics/B07.json`; `results/predictions/final/B07_test_predictions.csv` (git-ignored); `results/figures/final/B07_test_confusion_matrix.png`; `results/final/final_model_registry.json`

Observations: Validation 0.6684 -> test 0.6359 (identical to untuned B03 on test: tuning did not generalise).

Error Analysis Notes: see `docs/FINAL_EVALUATION_REPORT.md` §12–14

Limitations: single split and seed; MIMIC OCR captions; uncalibrated scores; known shortcuts not removed

Conclusion: reported as the frozen final result; no development after TEST

Next Experiment: TASK 14 - OCR Pipeline & OCR Robustness Experiment

### Experiment ID: FINAL-ABLATIONS

Date: 2026-10-13

Status: COMPLETED: **FINAL TEST EVALUATION** (pre-specified ablation models)

Research Question: RQ1 (RAW / BASIC / HINGLISH_NORMALIZED), fusion ablation, image classifier ablation

Dataset: MIMIC (cleaned, Task 4) with Task 6 v1.0 text and the original images

Dataset Version: canonical split `data/splits/split_assignments.csv` SHA-256 `47fb02a87b36…`

Dataset Source: Paper https://doi.org/10.1145/3656169; dataset card https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset

Train Split: 3,532 (used only to fit the frozen models, before Task 13)

Validation Split: 754 (used only to reproduce stored results at the freeze)

Test Split: 758 (382 non-misogynistic / 376 misogynistic). **EVALUATED ONCE (FINAL TEST EVALUATION)** after the freeze commit `ed0de53`

Input Modality: text (B01–B06, T01–T04), image (I01, I02), image + text (M01–M03)

Preprocessing: frozen per model (Task 6 v1.0 modes)

Feature Representation: as frozen in Tasks 8–11

Model: 15 frozen ablation models listed in the freeze manifest

Model Version / Pretrained Checkpoint: artifact checksums in `results/final/model_freeze_manifest.json`

Hyperparameters: frozen (Tasks 8–11 baseline settings)

Random Seed: 42 (bootstrap 10,000 resamples, seed 42)

Training Environment: no training in Task 13

Training Details: none

Validation Metrics: each reproduced exactly by the evaluation code at the freeze

Test Metrics: macro-F1 [95% CI]: B01 0.6319 [0.598–0.666]; B02 0.6346 [0.600–0.669]; B03 0.6359 [0.601–0.670]; B04 0.6187 [0.584–0.653]; B05 0.6238 [0.589–0.658]; B06 0.6172 [0.583–0.651]; T01 0.6330 [0.598–0.667]; T02 0.6264 [0.591–0.660]; T03 0.6172 [0.582–0.652]; T04 0.6736 [0.639–0.707]; I01 0.7768 [0.747–0.805]; I02 0.7793 [0.749–0.808]; M01 0.7517 [0.720–0.781]; M02 0.8016 [0.773–0.829]; M03 0.7929 [0.763–0.822]

Runtime / Resource Notes: included in the single evaluation run

Artifacts: `results/final/metrics/*.json`; `results/tables/final/preprocessing_ablation.csv`, `rq1_normalization_contrasts.csv`, `fusion_ablation.csv`, `statistical_tests.csv`

Observations: RQ1 controlled contrasts HINGLISH - BASIC: TF-IDF LR +0.0013, SVM -0.0066, frozen MuRIL -0.0092, fine-tuned MuRIL -0.0260 (all CIs include 0) -> INCONCLUSIVE. Fusion: M01 concatenation -0.0515 vs M04 (CI excludes 0); M02 and M03 within CI of M04. Image: I02 0.7793 and I01 0.7768 below tuned I03.

Error Analysis Notes: see `docs/FINAL_EVALUATION_REPORT.md`

Limitations: descriptive (no significance claims outside the two primary comparisons)

Conclusion: normalization not beneficial on TEST; projection-MLP fusion > concatenation; tuning of fusion not beneficial

Next Experiment: TASK 14 - OCR Pipeline & OCR Robustness Experiment
