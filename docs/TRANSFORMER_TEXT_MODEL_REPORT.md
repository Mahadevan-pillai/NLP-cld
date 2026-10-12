# Transformer Text Model Report (Task 9: MuRIL, frozen embeddings and fine-tuning)

Date: 2026-10-13 · Transformer text models v1

Code:
- `src/models/transformer_text.py`: embeddings, fine-tuning loop, save/load
- `src/inference/text.py`: `predict_text(text, model_version)`
- `scripts/train_text_transformers.py`: `tokens`, `frozen`, `benchmark`, `finetune --exp T04|T05`
- `scripts/report_text_models.py`: comparison tables, figures, error analysis, model registry
- `tests/test_transformer_text.py`

Results:
- `results/tables/text_models/`
- `results/figures/text_models/`
- `results/metrics/text_models/` (per-experiment JSON, validation predictions without text; training logs are local only, git-ignored)
- `results/metrics/text_model_registry.json`

> **TEST SET NOT EVALUATED IN TASK 9.** The test split was never loaded.
> All numbers below are **validation** results (754 samples) used for
> development comparison. They are **not** final performance estimates.

Tags:
- **[MEASURED]**: produced by the code.
- **[INTERPRETATION]**: our reading of the measurements.
- **[HYPOTHESIS]**: a possible explanation, not tested.

---

## 1. Objective
Test whether a pretrained multilingual language model beats the Task 8
TF-IDF baselines on text-only misogyny classification, at two levels:

- **Level A:** a frozen encoder with a lightweight classifier.
- **Level B:** a fine-tuned encoder.

It also asks again (RQ1) whether Hinglish normalization still helps once
the model is pretrained. Development metric: **macro-F1** on validation.

## 2. Verified classification tasks
Source: `docs/MULTICATEGORY_FEASIBILITY_REPORT.md` and
`config/categories.yaml` (Task 8.5). Nothing was changed.

| Task | Status | Type | Used in Task 9 |
|---|---|---|---|
| `misogyny_detection` (MIMIC `Misogyny`: 1 = misogynistic, 0 = non-misogynistic) | SUPPORTED, enabled | binary | **yes** |
| `misogyny_subcategories` (`Objectification`, `Prejudice`, `Humiliation`) | SUPPORTED, **disabled** pending a decision | multilabel, conditional on misogyny = 1 | no model trained; used only descriptively in §13 |
| SAFE, HATE, ABUSE, THREAT, OBSCENE, SARCASM | **NOT SUPPORTED** (no annotation) | n/a | never trained, never created by relabelling |

- The developer chose **binary only** for Task 9.
- `Misogyny = 0` means "non-misogynistic", not "safe".
- The binary task has two mutually exclusive classes. A 2-logit head with
  softmax and cross-entropy is therefore correct (§9). The multilabel
  sub-label task, if enabled later, would need independent sigmoid
  outputs instead.

## 3. Model candidates considered

| Candidate | Size (weights) | Licence | Hindi / Romanized Hinglish | Notes |
|---|---|---|---|---|
| `google/muril-base-cased` (MuRIL) | about 953 MB | Apache-2.0 | Pretrained on 17 Indian languages **and their transliterated (romanized) forms** (model card) | BERT-base; vocabulary of 197,285 WordPiece tokens |
| `l3cube-pune/hing-mbert` (HingMBERT) | about 712 MB | CC-BY-4.0 | mBERT further pretrained on Hinglish | Less widely used |
| `bert-base-multilingual-cased` (mBERT) | about 714 MB | Apache-2.0 | Hindi in Devanagari; little romanized Hindi | Generic multilingual baseline |
| `xlm-roberta-base` (XLM-R) | about 1.1 GB | MIT | Hindi in Devanagari (and some romanized Hindi in CC-100) | Largest of the base models here |
| multilingual MiniLM (sentence-transformers) | about 120 MB | Apache-2.0 | Weakest on romanized Hindi | Fast on CPU |

Sizes are approximate download sizes from the Hugging Face model pages.
They were not re-measured, except for MuRIL (below).

## 4. Selected pretrained model
**MuRIL base cased** (`google/muril-base-cased`). The developer approved
this choice.

- **Revision pinned:** commit `afd9f36c7923d54e97903922ff1b260d091d202f`
  (`config.yaml` `text_model.transformer.revision`).
- **Weights:** SHA-256 `e65e3b7aaa3ce16d23fc4218783efd0417880655f8f398e6cb1a111a0de0c95f`,
  verified after download.
- **Loading:** every run sets `HF_HUB_OFFLINE=1` and loads this exact
  revision from the local Hugging Face cache.

## 5. Selection rationale
- **Language fit.** MIMIC text is code-mixed: mostly romanized Hinglish
  plus English, with some Devanagari (Task 5). MuRIL is the only
  candidate pretrained on **transliterated** Indian-language text as well
  as native script.
- **Prior use on this dataset.** The MIMIC paper (DOI
  10.1145/3656169) reports that its best multimodal model combined BiT
  (image) and MuRIL (text). This makes MuRIL a natural reference for this
  project.
  - That paper's numbers come from a different split and a multimodal
    model. **They are not comparable** with the validation numbers here.
- **Size.** MuRIL is a base model (about 237 M parameters), not the
  largest option. It fits the 16 GB-RAM, CPU-only laptop.
- **Licence and reproducibility.** Apache-2.0 licence; available on
  Hugging Face with a pinnable commit.
- **One architecture.** A single encoder is used for both levels (as the
  task asks), so level A and level B differ only in whether the encoder
  is trained.

## 6. Tokenization
- **Tokenizer:** MuRIL's own WordPiece tokenizer (cased), from the same
  pinned revision.
- **Max length 64 tokens.** The rule (fixed before any model was trained):
  use the smallest candidate in {32, 48, 64, 96, 128} that truncates
  ≤ 1% of **train** texts. Lengths were measured on train only.
- **Truncation:** `longest_first`, i.e. the end of long texts is cut.
- **Padding:** dynamic, to the longest text in each batch.
- **No words were removed** to shorten texts.

**[MEASURED]** Token lengths including `[CLS]`/`[SEP]` (`token_lengths.csv`):

| Variant | Split | Median | 95th pct | Max | Truncated at 32 | at 48 | **at 64** | at 96 |
|---|---|---|---|---|---|---|---|---|
| RAW | train | 19 | 39 | 87 | 10.70% | 1.53% | **0.28%** | 0% |
| RAW | validation | 19 | 41 | 83 | 12.33% | 1.86% | **0.40%** | 0% |
| BASIC | train | 19 | 39 | 87 | 10.50% | 1.47% | **0.28%** | 0% |
| HINGLISH_NORMALIZED | train | 19 | 39 | 87 | 10.33% | 1.44% | **0.28%** | 0% |

- 48 tokens would truncate 1.4–1.5% of train texts, above the 1% rule.
  64 truncates 10 of 3,532 train texts (0.28%) and 3 of 754 validation
  texts (0.40%).
- This agrees with the Task 5 finding that MIMIC captions are short
  (median 11 words).

## 7. Preprocessing variants
- The three Task 6 v1.0 variants are used **unchanged**: RAW, BASIC,
  HINGLISH_NORMALIZED.
- Training and inference both start from **raw text** and call the
  Task 6 `preprocess()` function (via `preprocess_all()`), with the mode
  recorded for each model (in the registry and in the fine-tuned
  checkpoint's `project_metadata.json`). `predict_text()` therefore uses
  the same code path as training (a test checks this).
- BASIC and HINGLISH_NORMALIZED both mask URLs and mentions
  (`<URL>`/`<USER>`). The tokenizer keeps them as ordinary word pieces.

**Experiment matrix (fixed before training, `config.yaml`):**

| ID | Level | Text variant | Decision rule |
|---|---|---|---|
| T01 | frozen MuRIL + LR | RAW | — |
| T02 | frozen MuRIL + LR | BASIC | — |
| T03 | frozen MuRIL + LR | HINGLISH_NORMALIZED | — |
| T04 | fine-tuned MuRIL | best of RAW / BASIC by **validation** macro-F1 of T01/T02 → **BASIC** | chosen automatically by the script |
| T05 | fine-tuned MuRIL | HINGLISH_NORMALIZED | — |

T04 against T05 is the fair normalization comparison at the fine-tuned
level: same model, same settings, different text variant.

## 8. Frozen embedding experiments (T01–T03)
- **Embedding:** the encoder runs in `eval()` mode under `no_grad`. The
  embedding is the **attention-mask mean** of the last hidden layer, so
  padding never changes it (a test checks this). Dimension: **768**.
- **Cache:** embeddings are cached per variant and split, outside the
  repository (`<NLP_CLD_MODEL_DIR>/../cache/embeddings/muril_afd9f36c_len64_pre1.0/{variant}_{split}.npy`).
  The cache key includes the model revision, max length and
  preprocessing version. Train and validation are cached separately.
- **Classifier:** `StandardScaler` + `LogisticRegression(C=1.0,
  max_iter=2000, class_weight=None, random_state=42)`, **fitted on
  train embeddings only**.
  - Logistic regression was chosen over an MLP because it is simple,
    deterministic and directly comparable with Task 8.
  - Scaling is used because embedding dimensions have different ranges.
- **Artifacts:** `models/text/T0{1,2,3}_muril_frozen_logreg_{variant}_v1.joblib`
  (25,937 bytes each, git-ignored). The encoder is not copied; it is
  loaded from the pinned revision.

**[MEASURED]**
- All three classifiers converged.
- Save → reload reproduces the validation predictions exactly.
- Embedding time on CPU: about 102–141 s for 3,532 train texts and
  23–25 s for 754 validation texts.

## 9. Fine-tuning methodology (T04, T05)
- **Model:** `AutoModelForSequenceClassification` with **2 output
  logits** on top of MuRIL. Loss is cross-entropy over 2 mutually
  exclusive classes. Softmax gives an **uncalibrated** score.
- **Training loop:** a plain PyTorch loop (`src/models/transformer_text.py`)
  rather than the Hugging Face `Trainer`. Every step is visible and no
  extra dependency (`accelerate`) is needed.
- **Data:** fit on **train only**; the order is shuffled each epoch with
  seed 42.
- **Validation:** after every epoch, the model predicts validation and
  records validation loss and macro-F1.
- **Checkpoint selection:** the epoch with the highest **validation
  macro-F1** is kept (rule fixed in `config.yaml` before training).
- **Early stopping:** stop when validation macro-F1 has not improved for
  1 epoch.
- **Word embeddings frozen.** MuRIL's 197,285 × 768 input-embedding
  matrix (about 151 M parameters, ~64% of the model) is not updated.
  - Why: AdamW keeps two extra copies of every trained parameter. Freezing
    the embedding matrix saves about 1.8 GB RAM and time on this CPU.
  - All 12 transformer layers, position/type embeddings, pooler and head
    are trained.
  - This is a recorded deviation from "full" fine-tuning (§18).
- **Class imbalance:** none to correct. The classes are 50.3 / 49.7%
  (Task 5); no weighting and no oversampling are used.
- **Save:** `save_pretrained` (safetensors) + tokenizer +
  `project_metadata.json`. The metadata holds the label map,
  preprocessing mode, version and options, max length, base revision,
  training settings and best epoch.
- **Reload check:** the saved folder is reloaded and must reproduce the
  validation logits within `atol = 1e-4`.

## 10. Training configuration
Fixed baseline settings, **not tuned** (`config.yaml`
`text_model.transformer.fine_tune`). Task 12 handles tuning.

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 2e-5, weight decay 0.01 (none on bias / LayerNorm) |
| Schedule | linear warmup over 10% of steps, then linear decay to 0 |
| Batch size | 16 |
| Max epochs | 4 |
| Early stopping | patience 1 on validation macro-F1 |
| Gradient clipping | max norm 1.0 |
| Max length | 64 |
| Seed | 42 |
| Device | CPU (Intel i5-13500H, 12 threads), PyTorch 2.14.1+cpu, transformers 5.19.0 |

## 11. Validation results
**[MEASURED]** Validation, n = 754 (379 non-misogynistic, 375
misogynistic). 95% CIs are bootstrap intervals for macro-F1 (1,000
resamples, seed 42). They are descriptive, not significance tests.

| ID | Model | Text variant | Macro-F1 (95% CI) | Accuracy | Precision* | Recall* | Weighted-F1 | Macro-F1 with text only | Macro-F1 without watermark |
|---|---|---|---|---|---|---|---|---|---|
| T01 | frozen MuRIL + LR | RAW | 0.6151 (0.580–0.651) | 0.6154 | 0.6184 | 0.5920 | 0.6152 | 0.6054 | 0.6118 |
| T02 | frozen MuRIL + LR | BASIC | 0.6298 (0.595–0.662) | 0.6300 | 0.6319 | 0.6133 | 0.6299 | 0.6207 | 0.6267 |
| T03 | frozen MuRIL + LR | HINGLISH_NORMALIZED | 0.6191 (0.585–0.652) | 0.6194 | 0.6222 | 0.5973 | 0.6192 | 0.6095 | 0.6159 |
| T04 | fine-tuned MuRIL | BASIC | 0.6751 (0.641–0.707) | 0.6751 | 0.6693 | 0.6853 | 0.6750 | 0.6679 | 0.6738 |
| **T05** | **fine-tuned MuRIL** | **HINGLISH_NORMALIZED** | **0.6869 (0.654–0.719)** | 0.6870 | 0.6759 | 0.7120 | 0.6868 | 0.6803 | 0.6858 |

\* Precision and recall of the **misogynistic** class.

**Per-class F1 [MEASURED]:**

| ID | F1 non-misogynistic | F1 misogynistic | TN | FP | FN | TP |
|---|---|---|---|---|---|---|
| T01 | 0.6253 | 0.6049 | 242 | 137 | 153 | 222 |
| T02 | 0.6372 | 0.6225 | 245 | 134 | 145 | 230 |
| T03 | 0.6287 | 0.6095 | 243 | 136 | 151 | 224 |
| T04 | 0.6729 | 0.6772 | 252 | 127 | 118 | 257 |
| T05 | 0.6802 | 0.6935 | 251 | 128 | 108 | 267 |

**Fine-tuning history [MEASURED]** (`T04_training_history.csv`,
`T05_training_history.csv`, `*_training_curves.png`):

| Epoch | T04 train loss | T04 val loss | T04 val macro-F1 | T05 train loss | T05 val loss | T05 val macro-F1 |
|---|---|---|---|---|---|---|
| 1 | 0.6924 | 0.6930 | 0.3322 | 0.6930 | 0.6915 | 0.6133 |
| 2 | 0.6761 | 0.6496 | 0.6483 | 0.6773 | 0.6512 | 0.6554 |
| 3 | 0.6098 | 0.6286 | 0.6721 | 0.6329 | 0.6247 | **0.6869** |
| 4 | 0.5612 | 0.6302 | **0.6751** | 0.5921 | 0.6217 | 0.6788 |

- **Selected checkpoints:** T04 epoch 4, T05 epoch 3 (highest
  validation macro-F1). T05's "early stopping" after epoch 4 is
  cosmetic: epoch 4 was the last allowed epoch.
- **T04 epoch 1 predicted almost only one class** (macro-F1 0.33 is the
  value for constant predictions). Training loss stayed at about ln 2 =
  0.693 for the first epoch, then the model started learning in epoch 2.
- **Save → reload** reproduces the validation predictions for all five
  models. Fine-tuned logits match within `1e-4` and the frozen
  classifiers match exactly.
- **End-to-end check:** `predict_text()` on 60 raw validation texts,
  through the registry, reproduces the stored predictions of T02, T04
  and T05 exactly (scores within 2e-6, the rounding of the CSV).

**[INTERPRETATION]**
- **Fine-tuning helps clearly; frozen embeddings do not.** On the same
  text variant, fine-tuning is 4.5 points above frozen (BASIC: T04 vs
  T02) and 6.8 points above (HINGLISH: T05 vs T03).
- **Both fine-tuned runs had not clearly converged.** Training loss was
  still falling at epoch 4. Validation loss was flat from epoch 3. More
  epochs or a different learning rate could change the result. That
  belongs to Task 12 tuning, not to this task.
- **Recall is higher than precision** for the misogynistic class in both
  fine-tuned models: they lean towards predicting misogyny.

## 12. Comparison with Task 8
`results/tables/text_models/text_model_comparison.csv` holds all 11
models. The Task 8 rows are copied read-only from
`results/tables/text_baselines/baseline_results.csv`, which was not
changed.

| ID | Model | Text variant | Validation macro-F1 (95% CI) | Training time | Validation inference | Model size |
|---|---|---|---|---|---|---|
| B03 (best Task 8) | TF-IDF + LR | HINGLISH_NORMALIZED | 0.6671 (0.633–0.701) | 0.25 s | 0.03 s | 0.18 MB |
| B01, B02, B04–B06 | TF-IDF + LR / SVM | various | 0.6499–0.6565 | 0.1–0.2 s | ≤ 0.04 s | 0.18 MB |
| T02 (best frozen) | frozen MuRIL + LR | BASIC | 0.6298 (0.595–0.662) | 130 s † | 0.01 s ‡ | 0.03 MB + 953 MB encoder |
| T04 | fine-tuned MuRIL | BASIC | 0.6751 (0.641–0.707) | 44.5 min | 24.8 s | 957 MB |
| **T05** | **fine-tuned MuRIL** | **HINGLISH_NORMALIZED** | **0.6869 (0.654–0.719)** | 33.1 min | 24.1 s | 957 MB |

† Mostly embedding extraction (train + validation); fitting the
classifier takes under 1 s.
‡ Classifier only, with embeddings precomputed. Embedding a new text
costs about the same as a fine-tuned forward pass (about 30 ms per text
on this CPU).

**[MEASURED]**
- **T05 − B03 = +0.0198 macro-F1** and T04 − B03 = +0.0080.
- **All three frozen models are below every TF-IDF baseline:** T02 −
  B03 = −0.0373.
- The 95% intervals of T05 and B03 overlap (0.654–0.719 vs 0.633–0.701).
- T05 has higher recall for the misogynistic class than B03 (0.712 vs
  0.659) at similar precision (0.676 vs 0.668).

**[INTERPRETATION]**
- Fine-tuned MuRIL on normalized text is the **best development text
  model so far**, but the gain over TF-IDF is small relative to the
  validation uncertainty. **No significance is claimed.**
- The cost is large: about 5,000× the model size and about 800× the
  inference time of TF-IDF.
- **Frozen mean-pooled MuRIL embeddings are a weak representation for
  this task.**
  - [HYPOTHESIS] MuRIL was pretrained with masked-language modelling,
    not for sentence similarity, so averaged token vectors are not tuned
    to separate sentence meaning.
  - [HYPOTHESIS] TF-IDF also benefits more directly from lexical
    shortcuts (watermark handles, OCR tokens; Task 8 §11).

## 13. Category-level results
Only the binary task is trained (§2). There is therefore no per-category
classifier.

1. **The two binary classes:** see the per-class F1 table in §11.
   - T05: F1 0.680 for non-misogynistic, 0.694 for misogynistic.
   - Support: 379 / 375.
2. **Descriptive breakdown by MIMIC sub-label.** This is the recall of
   the **binary** misogyny prediction inside each annotated sub-label
   group (`subcategory_recall.csv`).
   - Sub-labels exist only for misogynistic memes, so only recall is
     defined.
   - **No sub-label model was trained.**

**[MEASURED]**

| Validation group | n | B03 | T02 | T04 | T05 |
|---|---|---|---|---|---|
| any misogynistic | 375 | 0.659 | 0.613 | 0.685 | **0.712** |
| Objectification | 214 | 0.654 | 0.607 | 0.668 | **0.687** |
| Prejudice | 152 | 0.651 | 0.586 | 0.678 | **0.704** |
| Humiliation | 55 | 0.764 | 0.673 | 0.764 | **0.836** |
| more than one sub-label | 46 | **0.739** | 0.565 | 0.674 | 0.717 |

**[INTERPRETATION]**
- With T05, **Objectification memes are missed most often** (recall
  0.687) and Humiliation memes least often (0.836).
- The Humiliation and multi-label groups are small (55 and 46). Their
  recall values can move by several points with a handful of samples.
- [HYPOTHESIS] Objectification often lives in the image (bodies,
  appearance) rather than in the caption. If so, it is the category most
  likely to benefit from the image and multimodal models (Tasks 10–11).

## 14. Confusion matrices
`results/figures/text_models/T0{1..5}_confusion_matrix.png`:
- rows = true label, columns = predicted label, with actual class names;
- counts plus row percentages;
- two-line titles that are not clipped (checked visually).

**[MEASURED] Major confusion pattern:**
- The frozen models lean towards **non-misogynistic**: FN > FP, e.g.
  T02 has 145 FN / 134 FP.
- The fine-tuned models lean towards **misogynistic**: FP > FN, e.g.
  T05 has 128 FP / 108 FN.
- For T05, 33.8% of non-misogynistic memes are flagged misogynistic and
  28.8% of misogynistic memes are missed.

## 15. Error analysis (best model, T05)
Outputs:
- `validation_errors.csv`: IDs, labels, T05 prediction and uncalibrated
  score, whether B03 is also wrong, and computed flags. It contains
  **no text**.
- `error_category_rates.csv`.
- A git-ignored copy with text, for manual review:
  `validation_errors_with_text.csv`.
- **Labels were not changed.**

**[MEASURED]** T05 makes **236 errors**: 128 false positives and 108
false negatives.
- 125 of them are also B03 errors.
- **T05 fixes 126 of B03's 251 errors and makes 111 new ones.**
- T04 and T05 agree on 86.3% of validation samples; 189 samples are
  wrong in both.

| Category | n | B03 error rate | **T05 error rate** |
|---|---|---|---|
| All | 754 | 33.3% | **31.3%** |
| Very short (1–3 words) | 45 | 40.0% | 35.6% |
| Long (20+ words) | 79 | 29.1% | 30.4% |
| Contains negation | 132 | 30.3% | **39.4%** |
| Contains Devanagari | 217 | 39.6% | **31.8%** |
| Contains a Hinglish-rule variant | 93 | 26.9% | 34.4% |
| Contains @mention | 57 | 17.5% | 24.6% |
| Watermark | 6 | 0.0% | 16.7% (1 of 6) |
| No text | 22 | 9.1% | 9.1% |

**Empty text [MEASURED]:**
- T04 and T05 predict **misogynistic for all 22 empty validation texts**
  (scores 0.534 / 0.526).
- This is the same dataset-composition shortcut as in Task 8: 77 of 105
  empty training texts are misogynistic.

**Scores [MEASURED]:**
- T05's softmax scores lie between 0.328 and 0.677 on validation (median
  0.526).
- The mean distance from 0.5 is almost the same for errors (0.137) as
  for correct predictions (0.143).

**Manual review [INTERPRETATION].** About 30 sampled errors were read
locally (text is not reproduced here). Recurrent patterns:
- **Image-dependent meaning (most false negatives):** a caption that is
  neutral on its own (nostalgia, a family scene, a celebration) belongs
  to a meme labelled misogynistic. The text alone cannot reveal the
  label.
- **Gender-topic false positives:** captions that mention wives,
  husbands, sisters, aunties or mothers-in-law, or describe a domestic
  scene, are labelled non-misogynistic but predicted misogynistic.
  - [HYPOTHESIS] The model has partly learned "mentions women or family
    roles → misogynistic", a topic shortcut.
- **Content-free texts:** captions that are only a social-media handle,
  or are empty, are predicted misogynistic with mid-range scores.
- **OCR noise:** misspelled or broken words (character substitutions,
  merged words) appear in many errors on both sides.
- **Devanagari slang and insults:** a few misogynistic captions in
  Devanagari with slang terms are missed.
- **Sarcasm and jokes:** several errors are joke formats whose target
  depends on context. There is no sarcasm annotation, so this is not
  measurable.
- **Possible label ambiguity:** a few non-misogynistic-labelled captions
  contain body-shaming or stereotyped jokes about women. These are
  **flagged for manual review only**; labels are not changed.

**[INTERPRETATION]**
- MuRIL handles **Devanagari** better than TF-IDF (31.8% vs 39.6%
  error), consistent with its native-script pretraining.
- It is **worse on negation** (39.4% vs 30.3%) and on texts with
  mentions.
- [HYPOTHESIS] The higher mention and watermark error rates suggest
  T05 relies **less** on the handle/watermark shortcut than TF-IDF.
- [HYPOTHESIS] The negation result is not explained yet. The groups are
  small (132), and this needs targeted checks.

## 16. Hinglish-normalization findings (RQ1, validation only)
`normalization_comparison.csv`.

**[MEASURED]**

| Level | Fair comparison | Macro-F1 | Difference (HINGLISH − other) | Direction |
|---|---|---|---|---|
| Frozen MuRIL + LR | T03 vs T02 (BASIC) | 0.6191 vs 0.6298 | **−0.0107** | worse |
| Frozen MuRIL + LR | T03 vs T01 (RAW) | 0.6191 vs 0.6151 | +0.0040 | slightly better |
| Fine-tuned MuRIL | T05 vs T04 (BASIC) | 0.6869 vs 0.6751 | **+0.0118** | better |
| (Task 8) TF-IDF + LR | B03 vs B02 | 0.6671 vs 0.6565 | +0.0106 | better |
| (Task 8) TF-IDF + SVM | B06 vs B05 | 0.6499 vs 0.6525 | −0.0026 | worse |

At sub-label level (§13), T05 has higher recall than T04 in every
group: +1.9 pp Objectification, +2.6 pp Prejudice, +7.3 pp
Humiliation, +4.3 pp multi-label.

**[INTERPRETATION]**
- **Answer: mixed, and not established.** For the fine-tuned model (the
  level that matters for the final system), HINGLISH_NORMALIZED is 1.2
  points better than BASIC. For frozen embeddings it is 1.1 points worse.
- Every difference is far smaller than the CI widths (about ±3.3
  points), and each comparison is a **single run with one seed**.
- **No significance is claimed.** A fair claim would need repeated seeds
  and/or a paired test. That is planned evaluation work, not done here.
- [HYPOTHESIS] Normalization merges spelling variants (`kr`/`kar`).
  This may matter more when the encoder's weights adapt (fine-tuning)
  than when a fixed pretrained vocabulary is averaged (frozen).

## 17. Resource requirements
**[MEASURED]** Windows 11 laptop, Intel i5-13500H CPU (12 threads), 16 GB
RAM, **no GPU**.

| Item | Value |
|---|---|
| MuRIL download | about 953 MB (pinned revision, Hugging Face cache) |
| Embedding extraction (frozen) | 102–141 s for 3,532 train texts; 23–25 s for 754 validation texts |
| Embedding cache | 38 MB (outside the repository) |
| Fine-tuning benchmark (before training) | 2.38 s per batch of 16 → estimate 9.7 min per epoch (`finetune_benchmark.json`) |
| Fine-tuning, actual | T04 44.5 min (4 epochs), T05 33.1 min (4 epochs). Epochs took about 440 s with the machine otherwise idle; T04 epochs 1–2 took about 900 s while other jobs ran |
| Trainable parameters | 86.0 M of 237.6 M (word embeddings frozen) |
| Peak process memory (observed) | about 1.9 GB early in T04 training |
| Fine-tuned checkpoint | 957 MB each (safetensors + tokenizer + metadata), 1.8 GB for both |
| Inference | about 32–33 ms per text (754 validation texts in 24–25 s, batches of 32); first call also loads the model |

- **Large checkpoints are stored outside the repository and OneDrive**,
  in the folder named by the `NLP_CLD_MODEL_DIR` environment variable
  (developer choice). The registry records them as
  `{MODEL_DIR}/T0x_...`.
- The small frozen-LR classifiers live in `models/text/`. Everything is
  git-ignored.

## 18. Limitations
- **No tuning.** Learning rate, epochs and other settings are untuned
  defaults; both fine-tuned runs were still improving on training loss.
  The comparison with TF-IDF may change after tuning (Task 12,
  validation only).
- **Word embeddings frozen** during fine-tuning (RAM/CPU constraint).
  This is not full fine-tuning; full fine-tuning may score differently.
- **Single seed, single split.** Transformer fine-tuning can vary across
  seeds, and run-to-run variation was not measured. T04's slow
  first epoch suggests that sensitivity.
- **Small validation set (754).** CIs are about ±3.3 points, larger than
  most differences reported here.
- **Uncalibrated, compressed scores.** Softmax scores span only about
  0.33–0.68 and barely separate errors from correct predictions. They
  must **not** be shown as confidence percentages. Calibration, if
  needed, is a later task.
- **Shortcuts remain:**
  - empty text is predicted misogynistic;
  - a possible gender-topic shortcut;
  - watermark/handle effects (smaller than for TF-IDF, per the subset
    scores).
- **Text is OCR output**, so OCR errors are part of the input. The text
  alone often cannot determine the label (image-dependent memes).
- **Frozen level:** only mean pooling with logistic regression was
  tried. Other pooling choices or classifiers were out of scope.

## 19. Test-set protection
**TEST SET NOT EVALUATED IN TASK 9.**

- Both scripts load data only through `load_development_data()` (train +
  validation).
- A unit test checks that none of the Task 9 sources
  (`scripts/train_text_transformers.py`, `scripts/report_text_models.py`,
  `src/models/transformer_text.py`, `src/inference/text.py`) contains
  `load_split("test"` or `allow_test=True`.
- These decisions all used **validation** only:
  - max length (train lengths only);
  - choice of the T04 text variant;
  - checkpoint (epoch) selection;
  - early stopping;
  - the best Task 9 model.
- No test metric exists anywhere in the outputs.
- The canonical split file is unchanged (SHA-256 `47fb02a87b36…`).
