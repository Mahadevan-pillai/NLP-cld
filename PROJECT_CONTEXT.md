# PROJECT_CONTEXT.md — Stable Project Context

Status vocabulary used in this document:

- **PLANNED** — intended, not yet implemented.
- **PROPOSED** — a candidate choice that may change after evidence.
- **TO BE VERIFIED** — a claim or assumption that must be checked
  (e.g. by dataset inspection or literature review).
- **IMPLEMENTED** — exists in the repository and has been tested.

Nothing in this document is IMPLEMENTED unless explicitly marked so.

---

## 1. Project title

**Explainable Multimodal Hinglish Misogynistic Meme Detection Using
Text and Vision-Language Models**

Short name: *Hinglish Multimodal Meme Detection*
Type: B.Tech Final-Year Project + Research Paper
Domains: Natural Language Processing, Computer Vision, Multimodal ML

## 2. Problem statement

Social-media memes combine an image with overlaid text. Misogynistic
content in memes may be expressed through the text, the image, or only
through the combination of both. In the Indian context much of this
text is Hinglish (Hindi-English code-mixed, often Romanized Hindi),
which has non-standard spelling, mixed scripts and informal grammar.

The project aims to build and evaluate models that classify
English/Hinglish text and memes as **Misogynistic** or
**Non-Misogynistic**, and to analyse when multimodal modeling,
Hinglish-specific normalization, and OCR quality affect performance.

## 3. Motivation

- Text-only moderation can miss harm that depends on the image, and
  image-only models cannot read the overlaid text.
- Code-mixed Hinglish is poorly served by tools built for standard
  English, so normalization may matter (TO BE VERIFIED, see RQ1).
- Real-world memes do not come with clean transcriptions; an applied
  system must rely on OCR, whose errors may degrade classification
  (TO BE VERIFIED, see RQ3).
- Moderation decisions benefit from explanations, provided those
  explanations are presented honestly as model attributions, not as
  proof of reasoning.

Specific prevalence statistics or prior-work claims will only be added
after verified literature review, with real citations.

## 4. Text Analysis mode (PLANNED)

```
User Text -> Input Validation -> Text Preprocessing
          -> Hinglish Normalization -> Text Model -> Prediction
```

Primary objective: Misogynistic vs Non-Misogynistic.

May display: original text, normalized text, predicted class,
confidence/probability (only when technically meaningful, e.g. a
calibrated or probabilistic model), explanation/important features
where appropriate, model version.

The text model does **not** initially require a separate text-only
dataset: if the primary multimodal dataset contains meme text and
labels, a text-only model can be trained on the text modality alone.
Whether additional text datasets help is decided after inspection.

## 5. Meme Analysis mode (PLANNED)

```
                       Uploaded Meme
                             |
                  +----------+----------+
                  |                     |
                IMAGE                  OCR
                  |                Extracted Text
                  |               Text Preprocessing
                  |             Hinglish Normalization
                  v                     v
            Image Encoder          Text Encoder
                  |                     |
            Image Features         Text Features
                  +----------+----------+
                             v
                      Multimodal Fusion
                             v
                         Classifier
                             v
                       Final Prediction
```

May display: uploaded meme, OCR text, normalized text, text-only
prediction, image-only prediction, multimodal prediction, final
classification, confidence where valid, explanation where valid.

The **multimodal prediction is the primary prediction**. Text-only and
image-only predictions are shown for research comparison and
interpretability.

## 6. Conceptual architecture (PLANNED)

```
            +-------------------- Streamlit app (app/) --------------------+
            | Home | Analyze Text | Analyze Meme | History | Dashboard | About |
            +-------------------------------+------------------------------+
                                            |
                         shared inference code in src/
       (preprocessing, normalization, OCR wrapper, feature extraction, models)
                                            |
              +-----------------------------+----------------------------+
              |                                                          |
     trained model artifacts (models/)                       MongoDB Atlas (src/database/)
                                                     analysis history, feedback, model versions

  Offline research pipeline (scripts/ + src/):
  raw data -> preprocessing -> splits -> training -> validation tuning
           -> final test evaluation -> results/ -> RESEARCH_LOG.md -> paper
```

Key principle: the app imports the same preprocessing and model code
used in training, so training and inference stay consistent.

## 7. Datasets (FINAL strategy, Task 4)

- **MIMIC**: *Misogyny Identification in Multimodal Internet Content
  in Hindi-English Code-Mixed Language* (Singh, Sharma & Singh, 2024,
  ACM TALLIP, https://doi.org/10.1145/3656169; dataset card
  https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset).
  - **Role:** the single **PRIMARY TRAINING DATASET** for all
    experiments.
  - **Contents:** 5,054 memes (image + OCR text), binary `Misogyny`
    label (documented: 1 = misogynistic, 0 = non-misogynistic) plus
    sub-labels Objectification / Prejudice / Humiliation. Licence
    CC-BY-4.0 (per dataset card).
  - **Cleaned (Task 4):** 5,044 samples in
    `data/processed/mimic_cleaned.csv`. 10 excluded: 6 identical
    copies, plus 2 identical memes labelled both ways.
- No datasets are merged. Other local files (`Downloads/DATASET 0x`)
  are NOT USED. Offensive/hate data, if ever used, is a separate
  auxiliary task. No external-validation dataset is available yet.
- Details: `docs/DATASET_REPORT.md` (inspection) and
  `docs/DATA_CLEANING_REPORT.md` (cleaning).

## 8. Remaining dataset caveats

- The dataset has **no official split**; the project creates its own.
  It must keep `duplicate_group` members together.
- File names encode the label; they must never be used as features.
- 148 samples have empty text (106 misogynistic / 42 not). The
  experiment design must state how the text-only model handles them.
- `ExtractedText` is **OCR output** (EasyOCR per the dataset card), not
  a human transcription (see the RQ3 note in §10).

Datasets will **not** be combined until label/task compatibility has
been demonstrated.

## 9. Planned model families (PROPOSED, none finalized)

| Family | Candidates | Status |
|---|---|---|
| Text baselines | TF-IDF + Logistic Regression; TF-IDF + Linear SVM | PROPOSED |
| Text transformer | A multilingual / Hinglish-capable pretrained encoder, chosen by language support, compute, license, reproducibility | TO BE SELECTED |
| Image | A pretrained vision or vision-language encoder (CLIP or another justified option) | TO BE SELECTED |
| Multimodal | Image embedding + text embedding -> simple fusion (e.g. concatenation) -> classifier, before any complex architecture | PROPOSED |

Listing a model here does not finalize it.

## 10. Research questions (PROPOSED — novelty not claimed)

- **RQ1** — Does Hinglish-specific text normalization improve
  misogynistic content classification?
- **RQ2** — Does multimodal image + text modeling outperform text-only
  and image-only approaches?
- **RQ3** — How does OCR noise affect real-world meme classification?

These may be refined after dataset inspection and literature review.
Novelty must not be claimed until relevant literature is reviewed.

**RQ4 (PROPOSED in Task 8.5, not adopted):** how effectively can
hierarchical multilabel models identify misogyny sub-categories
(objectification, prejudice, humiliation)? It is supported by MIMIC
sub-task 2; Humiliation has weak sample support. RQ1–RQ3 stay specific
to misogyny: the data cannot support a broader "harmful content" claim.

**RQ3 / OCR experiment: TO BE REFRAMED (finding from Task 4).** The
dataset text is itself OCR output (EasyOCR per the dataset card), so
there is no human "reference text". Options to decide before the OCR
task:
- dataset OCR vs the project's own OCR pipeline;
- a small manually transcribed subset as reference.

The research question has not been changed yet; that is a decision for
the developer.

## 11. Planned experiments (PLANNED — no results exist)

Text variants for these experiments exist (Task 6): `text_raw`,
`text_basic` and `text_hinglish_normalized`. "Without" normalization
means `text_basic`; "With" means `text_hinglish_normalized`.

| ID | Modality | Hinglish normalization |
|---|---|---|
| E1 | Text only | Without |
| E2 | Text only | With |
| E3 | Image only | n/a |
| E4 | Image + Text | Without |
| E5 | Image + Text | With |

OCR experiment (PLANNED; TO BE REFRAMED, see §10): reference dataset
text vs OCR-extracted text vs OCR-extracted + normalized text. Note
that the "reference" dataset text is itself OCR output.

Every run is recorded in `RESEARCH_LOG.md`.

## 12. Evaluation strategy (PLANNED)

- Metrics: Accuracy, Precision, Recall, F1, Macro-F1, Weighted-F1,
  per-class metrics, confusion matrix; others where justified.
- Accuracy alone is not sufficient, especially with class imbalance.
  The primary selection metric is TO BE DECIDED after inspecting the
  class distribution.
- Train / validation / test are kept separate. Tuning uses validation
  only; the test set is not used repeatedly during development.
- IMPLEMENTED (Task 7): one canonical stratified-group split,
  70 / 15 / 15 with seed 42 (MIMIC has no official split). It is shared
  by all experiments, and the test set is locked for final evaluation.
  Learned preprocessing (TF-IDF, scaling, class weights) is fitted on
  train only. See `docs/DATA_SPLIT_REPORT.md`.
- Statistical significance is only claimed if a suitable test is
  actually performed.

## 13. Database strategy (foundation IMPLEMENTED in Task 2)

- Technology: **MongoDB Atlas**, driver **PyMongo**. Stores application
  data only — **training datasets are never stored in MongoDB**.
- Collections (IMPLEMENTED): `analysis_history`, `user_feedback`,
  `model_versions`. Schemas, validation rules and indexes are
  documented in `docs/database.md`.
- Stored fields: analysis ID, analysis type (text/meme), UTC
  timestamps, submitted text, OCR text, normalized text, basic image
  metadata/optional reference, text/image/multimodal/final
  predictions, model versions, processing time, optional feedback.
- Code: `src/database/` (connection manager, schemas/validation,
  repositories, `ensure_indexes()`); scripts `scripts/check_mongodb.py`
  and `scripts/init_database.py`. Not yet used by the app (no UI yet).
- Classification labels are not hard-coded; they stay configurable
  until verified by dataset inspection.
- **Image binaries are not stored in normal MongoDB documents.** Whether
  images are temporary only, stored as references, kept in object
  storage, or in GridFS is **NOT YET DECIDED**.
- Connection string comes from the `MONGODB_URI` environment variable.
- No authentication is planned for now; may be added only if useful.

## 14. Application strategy (PLANNED)

Streamlit app with pages: Home, Analyze Text, Analyze Meme, Analysis
History, Research Dashboard, About, plus user feedback where
appropriate. OCR engine (Tesseract / EasyOCR / PaddleOCR / other) is
**NOT YET SELECTED** — a later task evaluates requirements.
Explainability (SHAP / LIME / token attribution / image heatmaps) is
PLANNED and must correspond to the actual trained model.
Deployment target (e.g. Hugging Face Spaces) is TO BE EVALUATED.

Privacy: avoid retaining uploads unnecessarily, store only necessary
data, document what is persisted, avoid logging raw user input.

## 15. Planned technology stack

| Concern | Choice |
|---|---|
| Language | Python |
| Traditional ML | scikit-learn |
| Deep learning | PyTorch |
| NLP / transformers | Hugging Face Transformers (+ possibly sentence-transformers) |
| Image processing | Pillow, OpenCV |
| Frontend | Streamlit |
| Database | MongoDB Atlas via PyMongo |
| Configuration | YAML (`config.yaml`) |
| Secrets | Environment variables / local `.env` |
| Version control | Git + GitHub |
| Deployment | TO BE EVALUATED (e.g. Hugging Face Spaces) |

## 16. Research integrity requirements (mandatory)

1. Never fabricate experimental results.
2. Never fabricate dataset statistics.
3. Never fabricate citations, papers, DOIs or authors.
4. Never claim research novelty without evidence.
5. Never manipulate metrics to make results look better.
6. Report negative/failed experiments honestly.
7. Never use the test set for routine hyperparameter tuning.
8. Prevent train/validation/test leakage.
9. Prevent duplicate or near-duplicate samples leaking across splits
   where reasonably detectable.
10. Preserve raw datasets.
11. Keep preprocessing reproducible.
12. Use fixed/configurable random seeds.
13. Record important experiments.
14. Distinguish validation results from final test results.
15. Do not merge datasets until label/task compatibility is shown.
16. Do not use an external LLM API as the actual classifier.
17. Pretrained models are allowed, but our classifiers must actually be
    trained/evaluated where the methodology requires.
18. Do not call a difference statistically significant without an
    appropriate statistical test.
19. Clearly distinguish measured observations from hypotheses.
20. Maintain enough documentation to defend the project in a viva.

## 17. Current project stage

**Task 4 — Dataset Cleaning and Final Dataset Strategy (complete).**

IMPLEMENTED in Task 1: directory structure, documentation
(`CLAUDE.md`, `PROJECT_CONTEXT.md`, `RESEARCH_LOG.md`, `README.md`),
`config.yaml` with placeholders, `.env.example`, `.gitignore`,
`requirements.txt`, a minimal config loader (`src/utils/config.py`)
and its tests.

IMPLEMENTED in Task 2: MongoDB database layer (`src/database/`),
health-check and init scripts, unit tests with a mocked client and an
in-memory fake database, `docs/database.md`. Live Atlas connectivity is
verified separately with `scripts/check_mongodb.py`.

IMPLEMENTED in Task 3: read-only dataset inspection (`src/data/`,
`scripts/inspect_datasets.py`), inventory and compatibility tables in
`results/tables/`, `docs/DATASET_REPORT.md`, verified column names in
`config.yaml`, log entry `INS-20261010-01`. MIMIC recommended as the
single primary dataset.

IMPLEMENTED in Task 4:
- MIMIC finalized as the single primary dataset.
- Deterministic cleaning pipeline (`src/data/cleaning.py`,
  `scripts/clean_dataset.py`), producing `data/processed/mimic_cleaned.csv`
  (5,044 samples, git-ignored).
- Exclusion log, duplicate decisions and cleaning summary in
  `results/tables/`; fingerprint in `results/metrics/`.
- `docs/DATA_CLEANING_REPORT.md`; log entry `INS-20261010-02`.
- Label meaning, licence and text origin documented from the dataset
  card.

IMPLEMENTED in Task 5:
- Reproducible EDA (`src/data/eda.py`, `src/data/eda_plots.py`,
  `scripts/run_eda.py`, `notebooks/02_eda.ipynb`).
- 17 tables in `results/tables/eda/`, 13 figures in
  `results/figures/eda/`, `docs/EDA_REPORT.md`, log entry
  `INS-20261010-03`.

Key EDA findings for later tasks:
- The classes are balanced.
- **Shortcut and leakage risks:**
  - class-exclusive `@handles` (watermarks);
  - class-correlated image resolutions;
  - 48 near-duplicate image pairs not yet grouped;
  - label-encoding file names.
- The splitting task must group near-duplicates and `duplicate_group`s
  together.
- Handle masking should be tested in preprocessing ablations.

IMPLEMENTED in Task 6:
- Text preprocessing v1.0 (`src/preprocessing/`), shared by training
  and the future app: modes `raw`, `basic`, `hinglish_normalized`.
- Rule file `src/preprocessing/resources/hinglish_normalization.yaml`
  (32 rules / 40 variants, all evidence-based).
- Variants in `data/processed/mimic_text_preprocessed_v1.0.csv`
  (git-ignored).
- Statistics and audit in `results/tables/preprocessing/`;
  `docs/TEXT_PREPROCESSING_REPORT.md`; log entry `INS-20261012-01`.
- Preprocessing effectiveness is NOT yet measured.

IMPLEMENTED in Task 7:
- One canonical, leakage-safe split (`src/data/splits.py`,
  `scripts/create_splits.py`): train 3,532 / validation 754 / test 758
  (70.02 / 14.95 / 15.03%), seed 42.
- Method: stratified-group (`StratifiedGroupKFold`) with near-duplicate,
  identical-image and identical-caption groups kept together; 11/11
  leakage checks pass.
- Canonical assignment `data/splits/split_assignments.csv` (tracked);
  `docs/DATA_SPLIT_REPORT.md`; log entry `INS-20261012-02`.
- **Test set locked**: `load_split("test")` requires `allow_test=True`.

IMPLEMENTED in Task 8:
- TF-IDF baselines B01–B06 (LR / Linear SVM × RAW / BASIC /
  HINGLISH_NORMALIZED), trained on train and compared on VALIDATION only.
- Best development baseline: **B03, LR + HINGLISH_NORMALIZED, macro-F1
  0.6671** (95% CI 0.633–0.701). All six are 0.650–0.667 with
  overlapping CIs. Normalization helped LR (+1.06 pp vs BASIC) but not the
  SVM (−0.26 pp); no significance is claimed.
- Shortcuts confirmed in model weights:
  - the watermark pattern survives mention masking (`user user`);
  - empty text is predicted misogynistic;
  - the OCR token `meः`.
- **Test set NOT evaluated.** `docs/TEXT_BASELINE_REPORT.md`.

DONE in Task 8.5 (feasibility study; nothing trained):
- The only content annotations are MIMIC `Misogyny` plus the sub-labels
  `Objectification`, `Prejudice` and `Humiliation`. These are
  multilabel and nested under misogyny.
- A general safe/harmful or multi-harm system (hate, abuse, threat,
  obscene, sarcasm) is **NOT supported** by any available annotation.
- PROPOSED, awaiting decision: a stage-2 multilabel misogyny
  sub-category task, and a narrowed RQ4.
- See `docs/MULTICATEGORY_FEASIBILITY_REPORT.md` and
  `config/categories.yaml`.

NOT STARTED: transformer/image/multimodal models, tuning, final test
evaluation, OCR, explainability, Streamlit UI, deployment, paper writing.

Next planned task: **TO BE DECIDED after review of
`docs/MULTICATEGORY_FEASIBILITY_REPORT.md`.**

### Academic evaluation milestones (for planning)
- **MSE 1** — Problem identification, dataset, preprocessing, EDA, model
  identification; paper Introduction + Literature Survey.
- **MSE 2** — Model training, hyperparameter tuning, result analysis;
  paper Methodology + Results.
- **ESE** — Working project (model behavior & evaluation, innovation &
  UX, deployment); paper Results & Discussion, comparison with existing
  systems, IEEE formatting + plagiarism report, paper communicated;
  viva.
