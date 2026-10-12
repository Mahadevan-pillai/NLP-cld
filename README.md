# Explainable Multimodal Hinglish Misogynistic Meme Detection Using Text and Vision-Language Models

*Short name: Hinglish Multimodal Meme Detection* — B.Tech Final-Year
Project + Research Paper (NLP · Computer Vision · Multimodal ML)

> **Status: project foundation only.** Model training has not started,
> dataset inspection has not been completed, and no experimental
> metrics exist yet. Everything described below as a model, pipeline or
> result is **PLANNED**. See [Current Project Status](#18-current-project-status).

---

## 1. Project Title
Explainable Multimodal Hinglish Misogynistic Meme Detection Using Text
and Vision-Language Models.

## 2. Project Overview
This project will build a research pipeline and a Streamlit application
that classify English/Hinglish text and memes as **Misogynistic** or
**Non-Misogynistic**. It compares text-only, image-only and multimodal
(image + text) models, studies Hinglish-specific normalization and OCR
noise, and provides explanations where technically valid. Analysis
history and feedback will be stored in MongoDB Atlas.

## 3. Problem Statement
Memes combine images and overlaid text, and misogyny may appear in the
text, in the image, or only in their combination. In Indian social
media much of this text is Hinglish (Hindi-English code-mixed, often
Romanized) with non-standard spelling. The project investigates how to
detect misogynistic content in such memes and text reliably.

## 4. Motivation
- Text-only or image-only systems can miss content whose meaning comes
  from the combination of both.
- Code-mixed Hinglish is a challenge for tools built for standard
  English.
- Real uploads require OCR, which introduces noise.
- Moderation tools benefit from honest, model-faithful explanations.

## 5. Application Modes (PLANNED)
| Mode | Input | Primary output |
|---|---|---|
| **Analyze Text** | English/Hinglish text | Text-model prediction |
| **Analyze Meme** | Meme image | Multimodal prediction (text-only and image-only shown for comparison) |

Planned pages: Home, Analyze Text, Analyze Meme, Analysis History,
Research Dashboard, About, plus user feedback.

## 6. Text Analysis Pipeline (PLANNED)
```
User Text -> Input Validation -> Text Preprocessing
          -> Hinglish Normalization -> Text Model -> Prediction
```
Shown to the user: original text, normalized text, predicted class,
confidence (where meaningful), explanation (where appropriate), model
version.

## 7. Meme Analysis Pipeline (PLANNED)
```
                       Uploaded Meme
                  +----------+----------+
                IMAGE                  OCR -> Preprocessing -> Normalization
                  v                     v
            Image Encoder          Text Encoder
                  +----------+----------+
                             v
                Multimodal Fusion -> Classifier -> Final Prediction
```
Shown to the user: meme, OCR text, normalized text, text-only /
image-only / multimodal predictions, final classification, confidence
and explanation where valid.

## 8. Proposed Research Questions
- **RQ1** — Does Hinglish-specific text normalization improve
  misogynistic content classification?
- **RQ2** — Does multimodal image + text modeling outperform text-only
  and image-only approaches?
- **RQ3** — How does OCR noise affect real-world meme classification?

These are proposed and may be refined after dataset inspection and
literature review. No novelty is claimed at this stage.

## 9. Planned Model Comparison
| ID | Modality | Hinglish normalization |
|---|---|---|
| E1 | Text only | Without |
| E2 | Text only | With |
| E3 | Image only | — |
| E4 | Image + Text | Without |
| E5 | Image + Text | With |

Plus an OCR experiment: reference text vs OCR text vs OCR + normalized
text.

Candidate model families (none finalized): TF-IDF + Logistic
Regression, TF-IDF + Linear SVM, a multilingual/Hinglish-capable
transformer encoder, a pretrained vision or vision-language encoder
(e.g. CLIP, to be justified), and a simple fusion classifier.

Planned metrics: Accuracy, Precision, Recall, F1, Macro-F1,
Weighted-F1, per-class metrics, confusion matrix.

## 10. Dataset Strategy
- **MIMIC** (Misogyny Identification in Multimodal Internet Content in
  Hindi-English Code-Mixed Language) is the **single primary dataset**:
  5,054 memes (image + OCR text), a balanced binary label
  (1 = misogynistic), licence CC-BY-4.0 (per its
  [dataset card](https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset)).
- **Cleaned metadata:** 5,044 samples. 10 duplicate or contradictory
  memes were excluded, with every decision logged. Raw files are never
  modified.
- No datasets are merged.
- Reports: [docs/DATASET_REPORT.md](docs/DATASET_REPORT.md) (inspection)
  and [docs/DATA_CLEANING_REPORT.md](docs/DATA_CLEANING_REPORT.md)
  (cleaning).
- Re-run with `python scripts/inspect_datasets.py` (read-only
  inspection) and `python scripts/clean_dataset.py` (deterministic
  cleaning).
- Raw datasets live in `data/raw/`, are never modified, and are never
  committed to Git.

## 11. Technology Stack
Python · scikit-learn · PyTorch · Hugging Face Transformers
(+ sentence-transformers) · Pillow · OpenCV · Streamlit · MongoDB Atlas
(PyMongo) · YAML config · `.env` for local secrets · Git + GitHub.
Deployment target (e.g. Hugging Face Spaces) is to be evaluated.
OCR engine is not yet selected.

## 12. Database Strategy (foundation IMPLEMENTED)
MongoDB Atlas stores **application data only** (never the training
datasets) in three collections: `analysis_history`, `user_feedback`,
`model_versions`. The database layer in `src/database/` (connection
manager, validation, repositories, indexes) is implemented and unit
tested; the app does not use it yet. **Image binaries are not stored in
documents**; the image storage approach is not yet decided. No
authentication. Full details: [docs/database.md](docs/database.md).

```bash
python scripts/check_mongodb.py
```
checks connectivity (needs `MONGODB_URI` in `.env`), and
`python scripts/init_database.py` creates the indexes.

## 13. Repository Structure
```
.
├── CLAUDE.md             # rules for Claude Code sessions
├── PROJECT_CONTEXT.md    # stable project context and decisions
├── RESEARCH_LOG.md       # experiment log + template
├── README.md
├── config.yaml           # central configuration (no secrets)
├── .env.example          # environment variable names (no values)
├── requirements.txt
├── pytest.ini
├── data/
│   ├── raw/              # original datasets (git-ignored)
│   ├── processed/        # derived data (git-ignored)
│   └── splits/           # canonical split_assignments.csv + manifest (tracked); per-split CSVs (git-ignored)
├── notebooks/            # 02_eda.ipynb (reproduces the EDA)
├── src/
│   ├── data/  preprocessing/  features/  models/  training/
│   ├── evaluation/  explainability/
│   ├── database/         # MongoDB connection, schemas, repositories, indexes
│   └── utils/            # config.py — config loader
├── scripts/              # check_mongodb.py, init_database.py, inspect_datasets.py, clean_dataset.py, run_eda.py, preprocess_text.py, create_splits.py, train_text_baselines.py
├── models/{text,image,multimodal}/   # model artifacts (git-ignored)
├── results/
│   ├── figures/  metrics/  tables/  paper/   # versioned small outputs
│   └── predictions/      # per-sample outputs (git-ignored)
├── app/                  # Streamlit app (later)
├── tests/                # pytest tests
└── docs/                 # database.md, DATASET_REPORT.md, DATA_CLEANING_REPORT.md, EDA_REPORT.md, TEXT_PREPROCESSING_REPORT.md, DATA_SPLIT_REPORT.md, TEXT_BASELINE_REPORT.md
```

## 14. Setup Instructions
Requires Python 3.10+ (developed on 3.10).

```bash
git clone <your-github-repo-url>
cd <repo-folder>
python -m venv .venv
```

Activate the virtual environment:
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (Git Bash): `source .venv/Scripts/activate`
- Linux/macOS: `source .venv/bin/activate`

Then install dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

For an NVIDIA GPU, install PyTorch first with the command from
<https://pytorch.org/get-started/locally/>, then install the rest.

## 15. Environment Variable Instructions
1. Copy `.env.example` to `.env` in the repository root.
2. Fill in your own values (`MONGODB_URI`, `MONGODB_DATABASE`,
   `HF_TOKEN`) in `.env` only.
3. `.env` is git-ignored — never commit it, paste it into chats, or
   share screenshots of it.
4. Code reads these values from environment variables (via
   `python-dotenv` locally). On a deployment platform, set them as
   platform secrets instead of uploading `.env`.

## 16. Research Reproducibility Principles
- Fixed, configurable random seed (`random_seed: 42` in `config.yaml`).
- Raw data preserved; preprocessing done by versioned code.
- Leakage-safe train / validation / test splits, with duplicate checks.
- Tuning on validation only; test set used once per final configuration.
- Every experiment recorded in `RESEARCH_LOG.md`.
- Same preprocessing code for training and inference.
- Exact dependency versions frozen once the environment is working.

## 17. Security Principles
- No secrets in code, config, logs, or Git history.
- Secrets only via environment variables; `.env.example` has names only.
- `.gitignore` excludes `.env`, credential files, datasets, model
  weights, caches and uploads.
- User uploads are not retained unnecessarily; raw user input is not
  logged unnecessarily.

## 18. Current Project Status
**Task 1 — Project Foundation: complete.**
**Task 2 — MongoDB Atlas Foundation: complete** (database layer + unit
tests; live Atlas connectivity depends on your `.env`).
**Task 3 — Dataset Inventory & Compatibility: complete.**
**Task 4 — Dataset Cleaning & Final Dataset Strategy: complete.**
**Task 5 — Exploratory Data Analysis: complete.**
**Task 6 — Text Preprocessing & Hinglish Normalization: complete.**
**Task 7 — Leakage-Safe Train/Validation/Test Split: complete.**
**Task 8 — Traditional Text Baselines: complete** (validation only).
- ✅ Repository structure, documentation, configuration, ignore rules,
  dependency list, config loader with tests.
- ✅ Database layer: connection manager, validation, repositories,
  indexes, health-check/init scripts, unit tests.
- ✅ Dataset inspection (read-only): MIMIC inventoried, report in
  `docs/DATASET_REPORT.md`. No data cleaned, split or modified.
- ✅ Dataset cleaning: 5,044 cleaned samples, every exclusion logged
  (`docs/DATA_CLEANING_REPORT.md`). Text left exactly as provided.
- ❌ Train/validation/test splitting — not done.
- ✅ EDA: 17 tables, 13 figures, notebook, `docs/EDA_REPORT.md`. It found shortcut risks (watermark
  handles, image resolutions, near-duplicate images) that later tasks must handle.
- ✅ Text preprocessing v1.0: three variants (RAW / BASIC / HINGLISH_NORMALIZED) via one shared
  `preprocess()` function (`docs/TEXT_PREPROCESSING_REPORT.md`). Their effect on models is not yet measured.
- ✅ Canonical leakage-safe split: train 3,532 / validation 754 / test 758 (seed 42); test set locked
  (`docs/DATA_SPLIT_REPORT.md`).
- ✅ TF-IDF text baselines (B01–B06): best **validation** macro-F1 0.6671 (LR + HINGLISH_NORMALIZED);
  `docs/TEXT_BASELINE_REPORT.md`. These are development results, not final test results.
- ❌ Transformer, image and multimodal models — not started.
- ❌ Final test-set results — **none exist** (test set locked).
- ❌ OCR, explainability, Streamlit UI, deployment — not started.

## 19. Planned Development Roadmap
1. ✅ Task 1 — Project foundation and repository initialization
2. ✅ Task 2 — MongoDB Atlas foundation and secure database layer
3. ✅ Task 3 — Dataset inventory, verification and compatibility analysis
4. ✅ Task 4 — Dataset cleaning and final dataset strategy
5. ✅ Task 5 — Exploratory data analysis
6. ✅ Task 6 — Text preprocessing and Hinglish normalization
7. ✅ Task 7 — Leakage-safe train/validation/test split
8. ✅ Task 8 — Text baselines (TF-IDF + LR / SVM), validation only
9. Transformer text model, image model, multimodal fusion
10. Hyperparameter tuning (validation only) and final test evaluation
11. OCR evaluation and OCR-noise experiment (to be reframed: dataset text is already OCR output)
12. Explainability
13. Streamlit application, history, dashboard, feedback
14. Deployment
15. Research paper (IEEE format)

The order after Task 8 is indicative and may change.

## 20. Disclaimer
This repository is an academic research project in progress.
**No model has been trained, the datasets have not yet been inspected,
and no experimental metrics exist.** Any future predictions will be
outputs of statistical models, which can be wrong and are not
definitive judgements about people or content. Explanations, when
added, are model attributions and not proof of how a model "reasons".
