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

## 7. Planned datasets

- **Primary candidate: MIMIC** — *Misogyny Identification in Multimodal
  Internet Content in Hindi-English Code-Mixed Language.* Provided by
  the developer. Schema, size, labels, splits and license are
  **TO BE VERIFIED**.
- Other datasets may be available locally. None has been inspected.

## 8. Dataset uncertainty / inspection requirement

No assumption is made about: schema, sample counts, label names,
class distribution, file names, image paths, text column names,
secondary labels, or licenses.

A dedicated future task will inspect each dataset for structure, task,
modalities, languages, labels, sample count, missing data, duplicates,
licensing and compatibility, and then assign roles:
PRIMARY / SECONDARY / EXTERNAL VALIDATION / MUST REMAIN SEPARATE.

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

## 11. Planned experiments (PLANNED — no results exist)

| ID | Modality | Hinglish normalization |
|---|---|---|
| E1 | Text only | Without |
| E2 | Text only | With |
| E3 | Image only | n/a |
| E4 | Image + Text | Without |
| E5 | Image + Text | With |

OCR experiment (PLANNED): reference dataset text vs OCR-extracted text
vs OCR-extracted + normalized text.

Every run is recorded in `RESEARCH_LOG.md`.

## 12. Evaluation strategy (PLANNED)

- Metrics: Accuracy, Precision, Recall, F1, Macro-F1, Weighted-F1,
  per-class metrics, confusion matrix; others where justified.
- Accuracy alone is not sufficient, especially with class imbalance.
  The primary selection metric is TO BE DECIDED after inspecting the
  class distribution.
- Train / validation / test are kept separate. Tuning uses validation
  only; the test set is not used repeatedly during development.
- Proposed default split if the dataset has no official split:
  70 / 15 / 15 with seed 42 (configurable in `config.yaml`). If an
  official split exists, using it is likely preferable (TO BE DECIDED).
- Statistical significance is only claimed if a suitable test is
  actually performed.

## 13. Database strategy (PLANNED)

- Technology: **MongoDB Atlas**, driver **PyMongo** (unless a later
  reason requires otherwise).
- Likely collections: `analysis_history`, `user_feedback`,
  `model_versions` (PROPOSED).
- Possible stored fields: analysis ID, analysis type (text/meme),
  timestamps, submitted text (where appropriate), OCR text, normalized
  text, image metadata/reference, text/image/multimodal/final
  predictions, model version, processing time, optional feedback.
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

**Task 1 — Project Foundation and Repository Initialization.**

IMPLEMENTED in Task 1: directory structure, documentation
(`CLAUDE.md`, `PROJECT_CONTEXT.md`, `RESEARCH_LOG.md`, `README.md`),
`config.yaml` with placeholders, `.env.example`, `.gitignore`,
`requirements.txt`, a minimal config loader (`src/utils/config.py`)
and its tests.

NOT STARTED: dataset inspection, preprocessing, EDA, splitting, model
training, tuning, OCR, explainability, database layer, Streamlit UI,
deployment, paper writing. No experimental results exist.

Next planned task: **Task 2 — MongoDB Atlas Foundation and Secure
Database Layer.**

### Academic evaluation milestones (for planning)
- **MSE 1** — Problem identification, dataset, preprocessing, EDA, model
  identification; paper Introduction + Literature Survey.
- **MSE 2** — Model training, hyperparameter tuning, result analysis;
  paper Methodology + Results.
- **ESE** — Working project (model behavior & evaluation, innovation &
  UX, deployment); paper Results & Discussion, comparison with existing
  systems, IEEE formatting + plagiarism report, paper communicated;
  viva.
