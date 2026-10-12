# CLAUDE.md — Operating Guide for Claude Code Sessions

This file is permanent guidance for every Claude Code session in this
repository. **Read this file first, in full, before doing anything.**

Project: **Explainable Multimodal Hinglish Misogynistic Meme Detection
Using Text and Vision-Language Models** (B.Tech final-year project +
research paper). The developer is a beginner: explain important
engineering decisions, keep code understandable, and never hide
research decisions inside code.

---

## 1. Mandatory working rules

1. **Read `CLAUDE.md` first** at the start of every session.
2. **Read `PROJECT_CONTEXT.md` before any architectural change**
   (new module, new data flow, new dependency, new storage decision).
3. **Inspect the existing implementation before modifying code.** Read
   the files you are about to change and the code that calls them.
4. **Perform only the requested task.** Do not start future tasks, add
   unrequested features, or "improve" unrelated code.
5. **Do not rewrite working components unnecessarily.** Prefer small,
   targeted edits.
6. **Follow repository conventions** (directory layout below, naming,
   config in `config.yaml`, secrets in environment variables).
7. **Keep training and inference preprocessing identical.** The text
   cleaning / Hinglish normalization used by the Streamlit app must be
   the *same code* used to train the model — import it from
   `src/preprocessing/`, never re-implement it in `app/`.
8. **Protect against data leakage.** Split data before fitting anything
   (vectorizers, scalers, normalizers with learned state). Detect and
   remove duplicate / near-duplicate samples across splits where
   reasonably possible.
9. **Never tune on the test set.** Model selection and hyperparameter
   tuning use the validation split only. The test split is used once
   per finalized configuration.
10. **Never fabricate results** — metrics, tables, figures, dataset
    statistics, or "expected" numbers.
11. **Never fabricate citations** — papers, authors, DOIs, venues, years.
    If a reference cannot be verified, say so.
12. **Never expose credentials** — do not print, log, echo, or commit
    secrets, and do not put them in exception messages.
13. **Use environment variables for secrets** (loaded from a local,
    git-ignored `.env` during development). `.env.example` holds
    variable *names* only.
14. **Run appropriate tests after changes** (`python -m pytest`) and
    report the real outcome, including failures.
15. **Document meaningful architectural changes** in
    `PROJECT_CONTEXT.md` (and `README.md` if user-facing).
16. **Keep experiments reproducible** — fixed, configurable random seed
    (`random_seed` in `config.yaml`), versioned configs, recorded
    dataset versions and pretrained checkpoints.
17. **Record research experiments** in `RESEARCH_LOG.md` using the
    template there, with a unique experiment ID.
18. **Prefer simple, understandable solutions.** Avoid premature
    abstractions and frameworks the project does not need.
19. **Report failures honestly** — failed commands, failed tests,
    negative experiment results, and unverified assumptions.
20. **Stop when the requested task is complete.** Summarize what was
    done and what remains; do not continue into the next task.

Additional research-integrity rules are listed in `PROJECT_CONTEXT.md`
(Section 16). They are mandatory.

---

## 2. Project summary

### Objective
Classify English / Hinglish (Hindi-English code-mixed) text and memes as
**Misogynistic vs Non-Misogynistic**, compare text-only, image-only and
multimodal models, study the effect of Hinglish-specific normalization
and OCR noise, and provide explanations where technically valid.

### Two application modes (PLANNED)
- **Analyze Text** — user text → validation → preprocessing → Hinglish
  normalization → text model → prediction (+ confidence, explanation,
  model version where valid).
- **Analyze Meme** — uploaded image → (a) image encoder and
  (b) OCR → preprocessing → normalization → text encoder → fusion →
  classifier. The **multimodal prediction is the primary prediction**;
  text-only and image-only predictions are shown for comparison.

### Research pipeline (PLANNED)
Dataset inspection → raw data preserved → reproducible preprocessing →
leakage-safe train/validation/test splits → baselines (TF-IDF + LR /
Linear SVM) → transformer text model → image model → simple multimodal
fusion → validation-only tuning → single final test evaluation →
error analysis → explainability → paper.

Planned ablations E1–E5 and the OCR experiment are described in
`PROJECT_CONTEXT.md`. No experiment has been run yet.

---

## 3. Security expectations
- Secrets (MongoDB URI, HF token, any API key) come **only** from
  environment variables, read via `os.environ` / `python-dotenv`.
- Never hard-code, print, or log secrets. Never commit `.env`,
  `.streamlit/secrets.toml`, keys, or credential files.
- `config.yaml` stores the *names* of environment variables, never
  their values.
- Before finishing any task, check `git status` for files that should
  not be tracked.

## 4. Data handling expectations
- `data/raw/` is **read-only**. Never modify, clean, or overwrite raw
  data in place. Derived data goes to `data/processed/` / `data/splits/`.
- Raw/processed datasets, model weights, checkpoints, embedding caches
  and uploads are git-ignored. Do not force-add them.
- Do not assume any dataset schema, filename, label name, or class
  distribution — verify by inspection and document it.
- Do not merge datasets until label/task compatibility is demonstrated.
- Application: do not retain user uploads unnecessarily, do not store
  image binaries in normal MongoDB documents, avoid logging raw user
  input. The image storage strategy is **not yet decided**.

## 5. Directory conventions

| Path | Purpose |
|---|---|
| `data/raw/` | Original datasets, untouched (git-ignored) |
| `data/processed/` | Derived/cleaned data (git-ignored) |
| `data/splits/` | Train/validation/test split files (git-ignored) |
| `notebooks/` | Exploration and EDA notebooks (logic that matters moves to `src/`) |
| `src/data/` | Dataset loading |
| `src/preprocessing/` | Text cleaning, Hinglish normalization (shared by training and app) |
| `src/features/` | Feature extraction (TF-IDF, embeddings) |
| `src/models/` | Model definitions |
| `src/training/` | Training loops / fitting |
| `src/evaluation/` | Metrics, confusion matrices, reports |
| `src/database/` | MongoDB access layer |
| `src/explainability/` | SHAP / LIME / attribution code |
| `src/utils/` | Small shared helpers (e.g. config loading) |
| `scripts/` | Command-line entry points that call `src/` |
| `models/{text,image,multimodal}/` | Saved model artifacts (binaries git-ignored; small `.md` model cards allowed) |
| `results/metrics/`, `results/tables/` | Small CSV/JSON result files (trackable) |
| `results/figures/`, `results/paper/` | Plots and paper-ready assets (trackable) |
| `results/predictions/` | Per-sample predictions (git-ignored: large, contain dataset text) |
| `app/` | Streamlit application (UI only; logic lives in `src/`) |
| `tests/` | pytest tests |
| `docs/` | Additional documentation |

Configuration lives in `config.yaml` (loaded by
`src/utils/config.py`). Do not scatter hard-coded paths or
hyperparameters across modules.

## 6. Testing expectations
- Run tests with: `python -m pytest` from the repository root.
- Add or update tests for new non-trivial logic (preprocessing,
  splitting, metrics, database layer with mocks).
- Tests must **not** download models/datasets or connect to MongoDB
  Atlas; mock external services. Use the `fake_db` fixture in
  `tests/conftest.py` for repository tests; an autouse fixture there
  disables `.env` loading so a real `MONGODB_URI` is never used.
- Report test results honestly, including skipped or failing tests.

## 7. Current stage
Tasks 1–5 done:
- repository foundation;
- the MongoDB database layer (`src/database/`, see `docs/database.md`);
- the read-only dataset inventory (`docs/DATASET_REPORT.md`);
- deterministic cleaning (`src/data/cleaning.py`, see
  `docs/DATA_CLEANING_REPORT.md`);
- EDA (`src/data/eda.py`, `notebooks/02_eda.ipynb`, see
  `docs/EDA_REPORT.md`);
- text preprocessing v1.0 (`src/preprocessing/`, see
  `docs/TEXT_PREPROCESSING_REPORT.md`);
- the canonical leakage-safe split (`src/data/splits.py`, see
  `docs/DATA_SPLIT_REPORT.md`);
- TF-IDF text baselines B01–B06 (`src/models/text_baselines.py`,
  `src/training/text_baselines.py`, see `docs/TEXT_BASELINE_REPORT.md`),
  validation only.

**Model pipelines:**
- Models take RAW text and apply `TextPreprocessor(mode)` (the Task 6
  `preprocess()`) inside the saved pipeline; the app must not
  preprocess separately.
- For TF-IDF, use the Devanagari-safe `token_pattern` from `config.yaml`.
  sklearn's default splits Hindi words.
- Evaluate with `src/evaluation/metrics.py` (macro-F1 is the development
  metric).
- Joblib artifacts are pickles: only load files this project created.

**Splits (mandatory for every experiment):**
- Use only `data/splits/split_assignments.csv` via
  `src.data.splits.load_split()` / `load_split_assignments()`.
- Never create a new random split, and never regenerate the canonical one
  without explicit instruction.
- Fit anything learned (TF-IDF, scalers, class weights, calibration) on
  **train only**.
- Tune and select on **validation**.
- The **test set is locked**: `load_split("test", allow_test=True)` only
  for the final evaluation of a frozen configuration, recorded in
  `RESEARCH_LOG.md`.
- Report results with and without the `doublemean*` watermark samples.

Text for models comes from `data/processed/mimic_text_preprocessed_v1.0.csv`
(`text_raw` / `text_basic` / `text_hinglish_normalized`). Any new text
input (e.g. the app or OCR output) must go through
`src.preprocessing.preprocess()` with the same mode and version the
model was trained on. Changing the rules file or the preprocessing code
requires bumping `text_preprocessing.version`.

The single primary dataset is MIMIC. All later tasks read
`data/processed/mimic_cleaned.csv` (from `config.yaml`
`dataset.cleaned_file`), never the raw CSV directly.

Rules for later tasks:
- Database access goes through the repository classes only: no PyMongo
  calls in `app/` or ML code.
- **MIMIC file names (`_M_`/`_NM_`) encode the label.** Never use
  `image_reference`/`image_path` as model input or derive features from
  them.
- Leakage groups (identical and near-duplicate images, substantive
  identical captions) are already enforced by the canonical split
  (Task 7).
- Shortcut features found in EDA (`@doublemeaning`/`@doublemean`
  watermarks, class-correlated image resolutions) must be tested for in
  ablations and error analysis. They must never be silently exploited.
- Notebooks are committed **without outputs**, so no dataset text is
  stored in Git. Reusable logic belongs in `src/`, not in notebooks.
- `original_text` is raw OCR text; preprocessing writes new columns and
  never overwrites it.

No test-set evaluation has been done and no UI built. See
`PROJECT_CONTEXT.md` Section 17 for the up-to-date stage.
