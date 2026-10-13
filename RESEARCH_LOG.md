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
