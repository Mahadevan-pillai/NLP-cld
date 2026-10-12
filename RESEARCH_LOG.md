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
