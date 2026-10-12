# Text Baseline Report (Task 8: TF-IDF + Logistic Regression + Linear SVM)

Date: 2026-10-12 · Baseline version 1.0

Code:
- `src/models/text_baselines.py`: pipelines
- `src/training/text_baselines.py`: training and diagnostics
- `src/evaluation/metrics.py`, `src/evaluation/plots.py`
- `scripts/train_text_baselines.py`

Results:
- `results/tables/text_baselines/`
- `results/figures/text_baselines/`
- `results/metrics/text_baselines_manifest.json`

> **TEST SET NOT EVALUATED IN TASK 8.** The test split was never loaded.
> All numbers below are **validation** results (754 samples) used for
> development comparison. They are **not** final performance estimates.

Tags:
- **[MEASURED]**: produced by the code.
- **[INTERPRETATION]**: our reading of the measurements.
- **[HYPOTHESIS]**: a possible explanation, not tested.

---

## 1. Objective
Establish reproducible traditional text-only baselines for misogyny
classification, and compare the three Task 6 text variants (RAW, BASIC,
HINGLISH_NORMALIZED) with two linear classifiers. This gives the first
evidence on RQ1. Development metric: **macro-F1**.

## 2. Experimental setup
| ID | Classifier | Text variant |
|---|---|---|
| B01 | Logistic Regression | RAW |
| B02 | Logistic Regression | BASIC |
| B03 | Logistic Regression | HINGLISH_NORMALIZED |
| B04 | Linear SVM | RAW |
| B05 | Linear SVM | BASIC |
| B06 | Linear SVM | HINGLISH_NORMALIZED |

- All six are trained on the same 3,532 train samples and compared on the
  same 754 validation samples.
- Seed 42. Python 3.10.0, scikit-learn 1.7.2.
- **Settings are fixed defaults, not tuned** (§4–6). Tuning is a later
  task.

## 3. Dataset split used
- **Split:** the canonical Task 7 split (assignment SHA-256
  `47fb02a87b36…`), loaded with `load_split("train")` and
  `load_split("validation")`.
- **Checks:** sizes match the manifest, there is no overlap, and labels
  are valid. The test split is never loaded (§16).
- **Empty texts:** all samples are used (train 105 empty, validation 22
  empty), so every experiment sees identical samples. An empty text
  becomes an all-zero feature vector.
  - This empty-text policy is the project decision first needed here.
  - Supplementary validation scores are reported on samples **with text**
    and **without watermark** (§8).

## 4. TF-IDF methodology
Each model is one scikit-learn `Pipeline`:

```
raw text -> TextPreprocessor(mode) -> TfidfVectorizer -> classifier
```

- **`TextPreprocessor`** calls the Task 6 `preprocess()` function.
  - It is stateless and learns nothing.
  - A check confirmed that its output **equals the stored Task 6 column
    exactly**, for every train and validation sample, in all six
    experiments.
  - The saved pipeline therefore takes raw text, the same way the app
    will.
- **TF-IDF is fitted on TRAIN only.** Validation is only transformed. A
  test checks that a word occurring only in validation never enters the
  vocabulary.

| Setting | Value | Reason |
|---|---|---|
| `token_pattern` | `(?u)[\wऀ-ॣॱ-ॿ]+` | **sklearn's default pattern splits Devanagari words at vowel signs** (`नहीं` → `नह`, `पता` → `पत`). This pattern keeps whole words in both scripts (measured on samples; covered by a test) |
| `lowercase` | true | Case is kept in preprocessing (Task 6) and lowercased here, identically for all variants |
| `ngram_range` | (1, 2) | Unigrams and bigrams: a common default |
| `min_df` / `max_df` | 2 / 1.0 | Drop words seen in only one training text (many are spelling/OCR singletons) |
| `sublinear_tf` | true | Common for short texts |
| `strip_accents` | none | Accent stripping must not touch Devanagari |

**[MEASURED]** Vocabulary sizes are 6,042 (RAW), 6,058 (BASIC) and
6,052 (HINGLISH_NORMALIZED) features.

## 5. Logistic Regression
- **Settings:** `C=1.0`, `solver="lbfgs"`, `max_iter=2000`,
  `class_weight=None`, `random_state=42`.
- **No class weighting:** the classes are balanced (Task 5: 50.3 / 49.7%).
- **Convergence:** all runs converged, with no warnings recorded.

## 6. Linear SVM
- **Settings:** `LinearSVC(C=1.0, loss="squared_hinge", dual="auto",
  max_iter=10000, class_weight=None, random_state=42)`.
- **Convergence:** all runs converged, with no warnings recorded.
- **Scores are not probabilities.** The SVM's `decision_function` values
  are signed margins. They are stored as `score_best` for diagnostics
  only and are **not** confidence probabilities. Calibration, if needed,
  is a later task.

## 7. Preprocessing variants
These are the Task 6 v1.0 variants, unchanged. No rule was modified after
seeing these results.

## 8. Validation metrics
**[MEASURED]** `baseline_results.csv`.
- Precision, recall and F1 are for the positive class (misogynistic).
- Macro scores are the unweighted mean over both classes.
- CI is the 95% bootstrap interval of macro-F1 (1,000 resamples). It
  describes sampling uncertainty and is **not a significance test**.

| ID | Model | Variant | Accuracy | Precision | Recall | **Macro-F1** | 95% CI | Weighted-F1 | Macro-F1 (with text, n=732) | Macro-F1 (no watermark, n=748) |
|---|---|---|---|---|---|---|---|---|---|---|
| B01 | LR | RAW | 0.6538 | 0.6541 | 0.6453 | **0.6538** | 0.619–0.685 | 0.6538 | 0.6456 | 0.6509 |
| B02 | LR | BASIC | 0.6565 | 0.6559 | 0.6507 | **0.6565** | 0.622–0.690 | 0.6565 | 0.6484 | 0.6536 |
| B03 | LR | HINGLISH | 0.6671 | 0.6676 | 0.6587 | **0.6671** | 0.633–0.701 | 0.6671 | 0.6593 | 0.6643 |
| B04 | SVM | RAW | 0.6512 | 0.6497 | 0.6480 | **0.6512** | 0.614–0.684 | 0.6512 | 0.6430 | 0.6483 |
| B05 | SVM | BASIC | 0.6525 | 0.6507 | 0.6507 | **0.6525** | 0.615–0.685 | 0.6525 | 0.6444 | 0.6497 |
| B06 | SVM | HINGLISH | 0.6499 | 0.6480 | 0.6480 | **0.6499** | 0.615–0.683 | 0.6499 | 0.6417 | 0.6470 |

Per-class results for B03 (`per_class_metrics.csv`):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| non-misogynistic | 0.6667 | 0.6755 | 0.6710 | 379 |
| misogynistic | 0.6676 | 0.6587 | 0.6631 | 375 |

**[INTERPRETATION]**
- **Accuracy ≈ macro-F1, legitimately.** Accuracy and macro-F1 agree to
  four decimals in every row (e.g. B01: 0.653846 vs 0.653797). This is
  not an error: the classes are balanced and false positives ≈ false
  negatives.
- **All baselines sit around 0.65–0.67 macro-F1**, well above the
  majority-class baseline of about 0.50 accuracy. Text alone is clearly
  informative but far from sufficient.

## 9. Confusion-matrix analysis
`results/figures/text_baselines/B0x_confusion_matrix.png`: rows = true
label, columns = predicted label. Colour shows the row percentage on a
fixed 0–100% scale for every figure.

**[MEASURED]**

| ID | TN | FP | FN | TP |
|---|---|---|---|---|
| B01 | 251 | 128 | 133 | 242 |
| B02 | 251 | 128 | 131 | 244 |
| B03 | 256 | 123 | 128 | 247 |
| B04 | 248 | 131 | 132 | 243 |
| B05 | 248 | 131 | 131 | 244 |
| B06 | 247 | 132 | 132 | 243 |

**[INTERPRETATION]** Errors are almost perfectly symmetric (FP ≈ FN).
No model is biased towards one class; each misclassifies about a third
of each class.

## 10. Preprocessing comparison (RQ1, validation only)
**[MEASURED]** `preprocessing_comparison.csv`, validation macro-F1:

| Model | RAW | BASIC | HINGLISH | BASIC − RAW | HINGLISH − BASIC | HINGLISH − RAW |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6538 | 0.6565 | 0.6671 | +0.0027 | **+0.0106** | +0.0133 |
| Linear SVM | 0.6512 | 0.6525 | 0.6499 | +0.0013 | **−0.0026** | −0.0013 |

1. **Did BASIC outperform RAW?** Numerically yes, for both models, but
   only by 0.1–0.3 points.
2. **Did HINGLISH_NORMALIZED outperform BASIC?** For Logistic Regression
   yes (+1.06 points). For Linear SVM **no** (−0.26 points).
3. **Did HINGLISH_NORMALIZED outperform RAW?** For Logistic Regression yes
   (+1.33 points). For Linear SVM no (−0.13 points).
4. **Highest validation macro-F1:** **B03, Logistic Regression +
   HINGLISH_NORMALIZED, 0.6671.**

**[INTERPRETATION]**
- **The differences are small and inconsistent across models.** All six
  bootstrap intervals overlap heavily (each is about ±3.4 points wide).
- **Normalization didn't help consistently:** it helped Logistic
  Regression but slightly hurt Linear SVM.
- **Conclusion for RQ1 at this stage:** there is **no evidence** that
  Hinglish normalization reliably improves TF-IDF baselines on this
  validation set. **No statistical significance is claimed.**
- A proper answer requires paired tests (e.g. McNemar) on the final
  test evaluation, and comparison across the transformer models.

**[HYPOTHESIS]** Normalization merges only about 2% of the vocabulary
(Task 6), and `min_df=2` already drops many rare variants. That leaves
little room for gains.

## 11. Feature inspection (diagnostics, not causal explanations)
`top_features.csv` lists the 25 strongest positive and negative weights
per model. **[MEASURED]**

- **Associated with misogynistic:** female and family referents, e.g.
  *she, her, girls, girl, didi, beti, behen, ladki, ladke, ladkiyan*.
- **Associated with non-misogynistic:** first-person and everyday
  topics, e.g. *me, bhai, friend, mom, teacher, students, parents, india,
  sir*.
- **Dataset artefacts among the strongest features:**
  - **Watermark:**
    - RAW models: `doublemean`, `doublemeaning`, `doublemeaning in` and
      `doublemean doublemeaning`; 4 of B01's top 25 features.
    - BASIC/HINGLISH models: the handles are masked, yet the
      **placeholder bigrams `user user` and `user in` rank 2nd–3rd**,
      and `doublemean` (written without @) still appears.
    - **Masking `@mentions` does not remove the watermark shortcut**,
      because the *pattern* of masked handles still identifies the source
      page.
  - **OCR artefact:** `meः` (colon misread as the visarga sign) is a
    strong non-misogynistic feature, probably from the "Me: …" meme
    format.
  - **Number-only tokens** (`8`, `१०`, `५`) carry weight in several
    models.

**[INTERPRETATION]** Part of the measured performance comes from
**source / format shortcuts**, not from understanding misogyny.

**[MEASURED]** The validation set contains only 6 watermark samples,
and removing them lowers macro-F1 by only about 0.3 points. The shortcut
is clearly learned, though; on a dataset with more watermarked memes it
could matter more.

## 12. Error analysis (best model, B03)
`validation_errors.csv` (IDs, labels, predictions of all six models and
computed flags; **no text**) and `error_category_rates.csv`. A
git-ignored copy with text, for manual review, is
`validation_errors_with_text.csv`.

**[MEASURED]** B03 makes 251 errors: 123 false positives and 128 false
negatives.

| Category | Validation samples | B03 error rate |
|---|---|---|
| All | 754 | 33.3% |
| Very short (1–3 words) | 45 | **40.0%** |
| Contains Devanagari | 217 | **39.6%** |
| Contains negation | 132 | 30.3% |
| Long (20+ words) | 79 | 29.1% |
| Contains a Hinglish-rule variant | 93 | 26.9% |
| Contains @mention | 57 | 17.5% |
| No text | 22 | 9.1% |
| Watermark | 6 | 0.0% |

- **207 of the 251 errors are made by all six models.** They are flagged
  for manual review as possible image-dependent meaning, ambiguity or
  label noise. **Labels were not changed.**
- **Empty text:** all six models predict **misogynistic for empty
  text** (positive bias, score about 0.18–0.20). That is because 77 of
  105 empty training texts are misogynistic. In validation 20 of 22
  empty texts are misogynistic, so this shortcut looks accurate (9%
  error), but it reflects the dataset's composition, not understanding.

**[INTERPRETATION]**
- **Devanagari and very short texts are hardest.** Devanagari texts are
  sparse in the vocabulary (OCR of Devanagari is noisy, and there is no
  Latin↔Devanagari bridging). Very short texts have too little signal.
- Texts with Hinglish variants are **not** harder than average.
- **Watermarked and mention-bearing texts are easier**, consistent with
  the shortcut in §11.

**[HYPOTHESIS]** Many consistent errors are memes whose misogyny is
expressed through the image or the image–text combination. This is what
the multimodal model (RQ2) should test.

## 13. Resource requirements
**[MEASURED]** On this laptop (CPU, no GPU):

| ID | Train time (s) | Validation prediction time (s) | Artifact (bytes) | TF-IDF features |
|---|---|---|---|---|
| B01–B06 | about 0.1–0.3 | about 0.015–0.05 | about 181,000 | 6,042–6,058 |

Exact per-run times are in `baseline_results.csv`; they vary slightly
between runs. Preprocessing is included in the timings. These models are
tiny and fast, a useful reference for the later transformer models.

## 14. Best validation baseline
**B03: TF-IDF (word 1–2-grams) + Logistic Regression on
HINGLISH_NORMALIZED text, validation macro-F1 = 0.6671 (95% CI
0.633–0.701).**

- Its interval overlaps those of all other baselines.
- It is the best **development** baseline, not a final result.
- It has not been tuned and has not been evaluated on test.

## 15. Limitations
- **No tuning.** Results may change with tuning (later task, validation
  only).
- **Small validation set (754).** The uncertainty (about ±3.4 points) is
  larger than the differences between variants.
- **Shortcuts.** Watermark patterns, empty text and OCR-format tokens are
  learned and inflate scores to an unknown degree.
- **Text is OCR output**, so OCR errors are part of the input.
- **Linear bag-of-words limits.** There is no context, no handling of
  negation scope, and no cross-script matching (`नहीं` ≠ `nahi`).
- **Shared rule caveat.** The Task 6 rule list was chosen from
  whole-corpus counts (Task 6 report §18).
- **Single split and seed.**

## 16. Test-set protection statement
**TEST SET NOT EVALUATED IN TASK 8.**

- `load_development_data()` loads only `train` and `validation`. A unit
  test checks that `test` is never requested.
- Another test checks that neither the script nor the training module
  calls `load_split("test")` or passes `allow_test=True`.
- No test metric exists anywhere in the outputs.
- Final test evaluation will happen once, for frozen configurations, in a
  later task.
