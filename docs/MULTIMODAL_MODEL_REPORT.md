# Multimodal Model Report (Task 11: image + text fusion)

Date: 2026-10-13 · Multimodal models v1.0

Code:
- `src/features/multimodal.py`: text features, sample-id keyed cache, `align()`
- `src/models/fusion.py`: fusion MLP, train-only standardisation, pickle-free save/load
- `src/inference/multimodal.py`: `predict_multimodal(image, text, model_version)`
- `scripts/train_multimodal.py`: `verify`, `features`, `train`
- `scripts/report_multimodal.py`: ablation, comparison, gains, disagreements, figures, registry
- `tests/test_multimodal.py`

Results:
- `results/tables/multimodal/`
- `results/figures/multimodal/`
- `results/metrics/multimodal/` (per-experiment JSON and validation predictions; no text)
- `results/metrics/multimodal_model_registry.json`

> **TEST SET NOT EVALUATED IN TASK 11.** All numbers are **validation**
> results (754 samples) for development comparison, **not** final
> performance estimates. No OCR was run: the captions are MIMIC's
> existing OCR text.

Tags:
- **[MEASURED]**: produced by the code.
- **[QUALITATIVE]**: manual inspection of a small sample.
- **[INTERPRETATION]**: our reading of the measurements.
- **[HYPOTHESIS]**: a possible explanation, not tested.

---

## 1. Objective
Build and evaluate a classifier that combines each meme's caption
(text) and picture (image), and compare it with the text-only (Tasks
8–9) and image-only (Task 10) models on the **same validation samples**.

## 2. Research question
**Does combining text and image information outperform text-only and
image-only models on the same validation samples?** (RQ2)

- Development metric: **macro-F1**.
- Fitting uses train; comparison uses validation; test is locked.
- No statistical significance is claimed.

## 3. Dataset and split
- **Data:** MIMIC, the canonical Task 7 split (assignment SHA-256
  `47fb02a87b36…`). It was verified, not regenerated.
- **Task:** binary `misogyny_detection` (1 = misogynistic). This is the
  only enabled task in `config/categories.yaml`.

**Input checks [MEASURED]** (`input_checks.csv`; 11 of 11 pass):

| Check | Result |
|---|---|
| Canonical split unchanged (manifest SHA-256), sizes 3,532 / 754 / 758 | ✅ |
| Train ∩ validation = ∅; no identical image across splits; no near-duplicate pair across splits | ✅ |
| T05 checkpoint, T03 classifier and I01/I02 classifiers exist | ✅ |
| Image cache covers exactly the canonical train / validation samples (IDs **and** image SHA-256 fingerprint) | ✅ |
| Validation labels identical to the T05 and I01 prediction files (same IDs, same labels) | ✅ |

**Usable samples [MEASURED]** (`usable_samples.csv`):

| Split | Samples | With image | Caption non-empty | Caption empty | Used |
|---|---|---|---|---|---|
| train | 3,532 | 3,532 | 3,427 | 105 | 3,532 |
| validation | 754 | 754 | 732 | 22 | 754 |
| test | 758 | not opened | not loaded | not loaded | 0 |

- **Every sample has both modalities.** An *empty caption* is a valid
  observation: OCR found no text. It is kept, as in Tasks 8–10, and the
  text representation of an empty string is used.
- This keeps the populations identical across Tasks 8–11. Results are
  **also** reported on the 732 validation samples with a non-empty
  caption (§11).
- The final empty-text policy is still an open project decision.

## 4. Text representation
Selected from **Task 9 validation evidence**, without retraining:
- **`finetuned_pooled`** (used by M01 and M02): the **T05** fine-tuned
  MuRIL pooled `[CLS]` vector (768-d).
  - It is exactly the input of T05's classification head.
  - Checkpoint: T05 (`google/muril-base-cased@afd9f36c`, fine-tuned),
    on HINGLISH_NORMALIZED text, Task 6 preprocessing v1.0, max length
    64.
  - Why: T05 is the best text model (validation macro-F1 0.6869).
- **`frozen_mean`** (used by the M03 control): frozen MuRIL, attention-mask
  mean pooling (768-d). This is the T03 representation (frozen +
  HINGLISH_NORMALIZED, 0.6191).

**Why a control.** T05 was fine-tuned on the same training samples.
- [MEASURED] T05 scores macro-F1 **0.7477 on train vs 0.6869 on
  validation**.
- Its training-set features therefore fit the labels better than unseen
  data does. A fusion classifier trained on them may over-trust text.
- The frozen features never saw a label. M03 measures how much this
  matters. The developer chose this design.

**Sanity checks [MEASURED]** (`text_feature_sanity.json`): see §10.

## 5. Image representation
- **Source:** Task 10 SigLIP base (`google/siglip-base-patch16-224@7fd15f06`),
  frozen vision tower, `pooler_output` (768-d).
- **Preprocessing:** `load_image()` v1.0, then SigLIP's PIL processor
  (224×224 stretch).
- **Reused, not recomputed:** the cached train / validation embeddings
  were not re-extracted. Their stored sample-ID list and (sample ID,
  image SHA-256) fingerprint are checked against the canonical split
  before use.
- Why: I01 (SigLIP + LR) is the best image model (0.7586).

## 6. Alignment methodology
- **Every feature file stores its own `sample_id` list.**
  - Text features are cached as `.npz` (IDs + features) plus `.json`
    metadata, including a fingerprint of the (sample ID, raw caption)
    pairs.
  - A changed caption, order or encoder makes the cache stale, and it is
    recomputed.
- **`align()` joins each modality onto the labelled samples by
  `sample_id`, never by row order.** It raises an error on duplicate
  IDs, missing labelled samples or non-finite values, so nothing is lost
  silently.
- **Audit:** `alignment_audit.csv`, see §10.

## 7. Fusion architecture
```
M01  [text 768 ; image 768] -> StandardScaler -> LogisticRegression

M02 / M03
  text  768 -> standardise -> Linear(768->256) -> LayerNorm --+
                                                             concat 512 -> Dropout(0.3) -> Linear(512->256)
  image 768 -> standardise -> Linear(768->256) -> LayerNorm --+             -> GELU -> Dropout(0.3) -> Linear(256->2)
```
- **Standardisation** (per feature) uses **train** statistics only. They
  are stored with the model.
- **LayerNorm after each projection** gives both modalities the same
  scale, so neither dominates by raw magnitude.
- **Output:** 2 logits, cross-entropy. Softmax gives an **uncalibrated**
  score.
- **Size:** M02/M03 have **526,594** trainable parameters (measured). The encoders
  are frozen and not part of the fusion model.

## 8. Experiments
Fixed before training (`config.yaml` `multimodal_model.experiments`):

| ID | Fusion | Text representation | Image | Role |
|---|---|---|---|---|
| M01 | early concatenation + LR | T05 pooled | SigLIP | multimodal **baseline** |
| M02 | projection + MLP | T05 pooled | SigLIP | **main** multimodal model (declared in advance) |
| M03 | projection + MLP | frozen MuRIL mean | SigLIP | **control** for the T05 feature bias |

- **M03 replaces the task's optional "Hinglish-normalized" variant.**
  Both M02 and M03 already use HINGLISH_NORMALIZED text, and Task 9
  found no established normalization effect. The measured T05
  train/validation gap is the more important question to control.
- No other variants were run. Tuning belongs to Task 12.

## 9. Training methodology
- **M01:**
  - `StandardScaler` + `LogisticRegression(C=1.0, lbfgs, max_iter=3000,
    class_weight=None, random_state=42)`;
  - fitted on train.
- **M02 / M03:**
  - AdamW, learning rate 1e-3, weight decay 1e-2, batch size 64;
  - up to 50 epochs, early stopping with patience 5 on **validation
    macro-F1**;
  - the best epoch is restored;
  - cross-entropy loss, seed 42, CPU.
- **Class imbalance:** none to correct (50.3 / 49.7%). No weighting and
  no oversampling.
- **No validation statistics are used for fitting.** Standardisation
  and the LR scaler are fitted on train; validation only selects the
  epoch.

## 10. Validation results
**Pre-checks [MEASURED]:**
- **Text features** (`text_feature_sanity.json`):
  - T05's own classification head applied to the extracted pooled
    features reproduces T05's stored validation predictions **754/754**;
  - the recomputed frozen features are **identical** to the Task 9 cache
    (max difference 0.0).
- **Alignment** (`alignment_audit.csv`): for train and validation and
  all three feature sets (fine-tuned text, frozen text, image):
  - 0 duplicate IDs, 0 missing samples, 0 extra IDs;
  - all values finite, dimension 768.

**Main results [MEASURED]:** validation, n = 754. The 95% CIs are
bootstrap intervals; they are descriptive.

| ID | Model | Macro-F1 (95% CI) | Accuracy | Precision* | Recall* | Weighted-F1 | Train macro-F1 |
|---|---|---|---|---|---|---|---|
| M01 | concat + LR | 0.7679 (0.736–0.798) | 0.7679 | 0.7688 | 0.7627 | 0.7679 | 0.9465 |
| **M02** | **fusion MLP, T05 text (main)** | **0.8076 (0.779–0.834)** | 0.8077 | 0.8194 | 0.7867 | 0.8076 | 0.8915 |
| M03 | fusion MLP, frozen text (control) | 0.8077 (0.780–0.836) | 0.8077 | 0.8026 | 0.8133 | 0.8077 | 0.9918 |

\* Precision and recall of the **misogynistic** class.

**Per-class and confusion counts [MEASURED]:**

| ID | F1 non-misogynistic | F1 misogynistic | TN | FP | FN | TP |
|---|---|---|---|---|---|---|
| M01 | 0.7700 | 0.7657 | 293 | 86 | 89 | 286 |
| M02 | 0.8124 | 0.8027 | 314 | 65 | 80 | 295 |
| M03 | 0.8074 | 0.8079 | 304 | 75 | 70 | 305 |

**Checks [MEASURED]:**
- **Save → reload** reproduces the validation predictions:
  - M01 (joblib) exactly;
  - M02 and M03 (safetensors + JSON) within 1e-6.
- **End-to-end:** `predict_multimodal()` on all 754 raw validation
  (image file, caption) pairs: it reproduces the stored predictions of **M01, M02 and M03: 754/754 each** (scores within 4e-6, the CSV rounding). All 22 empty captions are flagged `processed_text_empty`.

**Selection optimism [MEASURED]:**
- The MLP epoch was chosen on validation, so the selected score is the
  **peak** of a noisy curve.
- Across *all* epochs run:
  - M02 validation macro-F1 ranged **0.784–0.808** (mean 0.794 over 7
    epochs);
  - M03 ranged **0.772–0.808** (mean 0.794 over 13 epochs).
- So the selected values are about 1.4 points above the epoch mean.
  **Even the worst epoch of either MLP is above the best unimodal model**
  (I01, 0.7586).

**[INTERPRETATION]**
- The neural fusion (M02, M03) is clearly better than the linear
  concatenation (M01).
  - M01 overfits strongly (train 0.9465): 1,536 standardised dimensions
    with C = 1.0.
- **M02 and M03 are indistinguishable** (0.8076 vs 0.8077).
  - Fine-tuned T05 text features give **no** measurable fusion advantage
    over frozen MuRIL features, even though alone they are much better
    (0.687 vs 0.619).
  - [HYPOTHESIS] The fusion gets most of its text benefit from
    information that frozen MuRIL already encodes, and T05's gain as a
    standalone classifier does not add on top of the image.
- **The T05 feature bias did not produce visible harm.** M02 stopped at
  epoch 2 with a train macro-F1 of 0.8915, lower than M03's 0.9918, and
  generalised as well.

## 11. Text vs image vs multimodal comparison
`main_comparison.csv` and `modality_ablation.csv`. **All rows use the
identical validation samples**, joined on `sample_id`.

| Role | ID | Modality | Macro-F1, all 754 | Caption not empty (732) | Without watermark (748) | Training cost | Inference |
|---|---|---|---|---|---|---|---|
| best traditional text | B03 | text | 0.6671 | 0.6593 | 0.6643 | 0.25 s | 0.03 s / 754 texts |
| frozen-text reference | T03 | text | 0.6191 | 0.6095 | 0.6159 | 128 s (embedding) | — |
| best pretrained text | T05 | text | 0.6869 | 0.6803 | 0.6858 | 33 min fine-tuning | 24 s / 754 texts |
| best image-only | I01 | image | 0.7586 | 0.7581 | 0.7567 | 11.9 min embedding + 0.3 s | ~0.21 s / image |
| multimodal baseline | M01 | image+text | 0.7679 | 0.7675 | 0.7660 | 3.4 s (on cached features) | 0.04 s classifier |
| **MAIN multimodal** | **M02** | image+text | **0.8076** | 0.8027 | 0.8059 | 8.8 s (on cached features) | 0.02 s classifier |
| multimodal control | M03 | image+text | 0.8077 | **0.8045** | **0.8061** | 16.1 s (on cached features) | 0.03 s classifier |

- The multimodal "training cost" covers only the fusion classifier.
- Its features need the T05 text encoder and the SigLIP image encoder at
  inference: about 0.6–1.0 s per meme measured end-to-end on this busy CPU (§19).

**[MEASURED]**
- The ranking is the same in all three populations:
  **multimodal MLP > M01 ≈ I01 > text models**.
- Removing the 22 empty-caption samples or the 6 watermark samples
  changes no conclusion.

## 12. Multimodal gain / loss
`multimodal_gain.csv`. These are absolute macro-F1 differences on the
same 754 samples. **No significance is claimed.**

| Multimodal | Macro-F1 | − best text (T05) | − best image (I01) | − matched text representation |
|---|---|---|---|---|
| M01 | 0.7679 | +0.0810 | **+0.0093** | +0.0810 (T05) |
| **M02 (main)** | 0.8076 | **+0.1207** | **+0.0490** | +0.1207 (T05) |
| M03 | 0.8077 | +0.1208 | +0.0491 | +0.1886 (T03) |

**[INTERPRETATION]**
- On validation, **combining text and image is better than either
  alone** for this task.
- The gain over the image-only model is modest (+4.9 points for the
  MLPs). For M01 it is almost nil (+0.9).
- The gain over text-only is large (+12.1), mainly because the image is
  the stronger modality on MIMIC.
- Even the worst-epoch MLP scores are above I01 (§10), so the direction
  of the result does not depend on the epoch choice. It is still one
  split and one seed, and **the final answer to RQ2 requires the
  single, locked test evaluation (later task).**

## 13. Confusion matrices
`results/figures/multimodal/M0{1,2,3}_confusion_matrix.png`:
- rows = true label, columns = predicted label, with class names;
- counts and row percentages;
- two-line titles, checked visually, nothing clipped.

**[MEASURED]**
- M02 misses 21.3% of misogynistic memes and flags 17.2% of
  non-misogynistic ones.
- It is the most precision-oriented model (precision 0.819, recall
  0.787). M03 is balanced (0.803 / 0.813).

**Modality reliance [MEASURED]**, a diagnostic: one modality's input is
replaced by its training-set mean on validation. The result is crude
(out-of-distribution) but shows direction:

| Model | Normal | Text → mean | Image → mean |
|---|---|---|---|
| M02 | 0.8076 | 0.7861 | 0.6816 |
| M03 | 0.8077 | 0.7597 | 0.4548 |

- [INTERPRETATION] Both fusion models **rely mainly on the image**, and
  M03 collapses without it.
- Text adds a smaller but real correction: 2–5 points.
- M03's frozen text matters **more** inside its fusion than T05's
  features do inside M02.

**Sub-label recall** of the binary prediction (descriptive;
`subcategory_recall.csv`):

| Group | n | T05 (text) | I01 (image) | M02 | M03 |
|---|---|---|---|---|---|
| Objectification | 214 | 0.687 | 0.855 | 0.860 | **0.888** |
| Prejudice | 152 | 0.704 | 0.638 | 0.678 | **0.717** |
| Humiliation | 55 | **0.836** | 0.727 | 0.782 | 0.800 |

- [INTERPRETATION] Fusion keeps the image's strength on objectification
  and recovers much of the text's strength on prejudice and humiliation.
  - M03 exceeds T05 on prejudice (0.717 vs 0.704).
  - Neither MLP reaches T05 on humiliation (n = 55, a small group).

## 14. Disagreement analysis
`modality_disagreements.csv` lists, per validation sample, the
predictions of T05 (text), I01 (image), M02 (main) and M01/M03, the
category flags and `caption_empty`. It contains **no text**.
`disagreement_summary.csv` holds the counts.

| Category | Samples | Share |
|---|---|---|
| A. text correct, image wrong, multimodal correct | 59 | 7.8% |
| B. image correct, text wrong, multimodal correct | 132 | 17.5% |
| C. text wrong, image wrong, multimodal correct | 15 | 2.0% |
| D. text correct, image correct, multimodal wrong | 10 | 1.3% |
| E. text and image predictions disagree | 264 | 35.0% |
| F. all three wrong | 62 | 8.2% |

**[MEASURED]**
- When text and image **disagree** (264), M02 is right in **191**
  (72.3%).
- In those 264 cases it sides with the image in 178 and with the text in
  86.
- It rescues 15 samples that both unimodal models get wrong, and breaks
  only 10 that both get right.

**[INTERPRETATION]** The fusion works mostly as an **image-led model
corrected by text**. It fixes 59 image errors with text evidence, and
uses the image to overturn 132 text errors.

## 15. Qualitative image–text interaction analysis
**[QUALITATIVE]** 40 validation memes were viewed locally, with their
captions: 8 each from categories A, B, C, D and F, randomly sampled.
Nothing was saved or committed. Captions are paraphrased, not quoted,
and real people are not named. This is a small sample; the observations
are illustrative, not measured.

- **The image changes the meaning of the text** (category B):
  - an innocent food caption over a photo whose framing sexualises a
    woman;
  - innuendo captions ("take me to a hotel", "attack from behind")
    whose sexual meaning is fixed by the picture.

  The text model misses these; image and fusion catch them.
- **The same template, a different label depending on who is pictured.**
  A "THE mistake → THE result" template (condom image → photo) is
  labelled **misogynistic when the "result" photo shows a woman** and
  **non-misogynistic when it shows a man**. The caption is identical in
  substance, so only the image–text combination determines the label.
- **The text changes the meaning of the image** (category A):
  - a glamour group photo of young women becomes a jab once the caption
    says their wealth is inherited;
  - a cartoon becomes a stereotype of the "quarrelsome woman" through its
    caption.
- **The image alone rescues false text alarms:** captions about marriage
  or poverty over cat, dog or cartoon pictures are flagged by the text
  model but correctly cleared by image and fusion.
- **An apparent text success can be a shortcut.** In one category-A
  sample the OCR caption is **empty** (the picture's Devanagari caption
  was not extracted). The text model is "correct" only through its
  empty-text → misogynistic shortcut (Task 8–9).
- **Failures of all models** (category F):
  - **Cultural references:** a famous Bollywood film line over a film
    still (labelled misogynistic) is missed.
  - **Dialogue and sarcasm formats** ("She: … Dad: …") about daughters
    or girlfriends are labelled non-misogynistic but flagged. This fits
    the **gender-topic / gender-presence shortcuts** seen in Tasks 9–10.
  - **Screenshots** of social-media posts are missed.
- **Possible label ambiguity** (flagged for manual review; **labels not
  changed**):
  - some memes mocking men as feminine are labelled misogynistic;
  - some body-related jokes about girlfriends are labelled
    non-misogynistic.

  Annotator intent cannot be verified here.

## 16. Training curves
`results/figures/multimodal/M02_training_curves.png` and
`M03_training_curves.png`; histories in `M0{2,3}_training_history.csv`.

**[MEASURED]**
- **M02:**
  - validation loss is lowest at epoch 2 (0.433) and then rises;
  - training loss falls from 0.454 to 0.158;
  - early stopping at epoch 7; epoch 2 restored.
- **M03:**
  - validation loss is lowest at epoch 1 (0.445) and reaches 0.80 at the
    selected epoch 8;
  - early stopping at epoch 13.

**[INTERPRETATION]**
- **Both MLPs overfit within a few epochs.** Selection by macro-F1 can
  pick an epoch whose loss is already poor; M03 is the clear case.
- M03's scores at the selected epoch are therefore likely
  **overconfident**.
- Stronger regularisation, or selection by loss, is a Task 12 question.

## 17. Multicategory status
- **The sub-labels exist for this data.** Task 8.5 verified
  `Objectification`, `Prejudice` and `Humiliation`. They are multilabel,
  nested under misogyny, and both modalities exist for all of these
  samples.
- **The architecture could support them.** The same fusion inputs could
  feed a 3-output sigmoid (multilabel) head trained on the misogynistic
  subset.
- **Status: SUPPORTED FOR FUTURE / NOT TRAINED.** The stage-2
  sub-category task is **disabled** in `config/categories.yaml` pending a
  project decision. Task 11 therefore reports only the descriptive
  sub-label recall of the binary prediction (§13). No six-class or other
  invented labels were created.

## 18. Limitations
- **Validation-only, one split, one seed.**
  - The MLP epoch is selected on the same validation set, so the scores
    are optimistic by about 1.4 points versus the epoch mean (§10).
  - CIs are about ±3 points.
- **No tuning:** fixed baseline settings. The MLPs overfit quickly
  (§16).
- **T05 features were fitted on the training labels.** The M03 control
  shows no measurable harm here, but out-of-fold text features would be
  the clean solution.
- **Image-led model:** it inherits the image model's suspected
  **gender-presence shortcut** (Task 10) and the text models'
  **empty-caption** and **gender-topic** shortcuts.
- **The captions are MIMIC's OCR text.**
  - The app will need its own OCR (later task), whose errors may differ.
  - Some captions are missing or broken (e.g. Devanagari not extracted).
- **Scores are uncalibrated.** M03's selected epoch has a high
  validation loss (0.80), i.e. it is overconfident.
- **The modality-reliance diagnostic** (mean replacement) is a crude,
  out-of-distribution probe.

## 19. Runtime / resources
**[MEASURED]** CPU only (i5-13500H, no GPU).

| Step | Time |
|---|---|
| Text features, T05 pooled | train 389 s + validation 24 s |
| Text features, frozen MuRIL | train 267 s + validation 26 s |
| Image features | reused from the Task 10 cache (originally 713 s) |
| M01 fit | 3.4 s |
| M02 / M03 training (CPU) | 8.8 s (7 epochs) / 16.1 s (13 epochs) |
| Classifier inference, 754 samples | 0.02–0.04 s |
| `predict_multimodal()` end-to-end | 0.63–1.05 s per meme (M01 631 ms, M02 1,048 ms, M03 704 ms), measured while other jobs shared the CPU, so the figures are approximate |

- **Artifacts:** M01 50,513 B (joblib); M02 and M03 about 2.12 MB each
  (safetensors + JSON).
- **Model size:** M02/M03 have 526,594 parameters.
- **Feature caches:** text features are stored outside the repository
  (`<dir of $NLP_CLD_MODEL_DIR>/../cache/multimodal_text/`).

## 20. Test-set protection
**TEST SET NOT EVALUATED IN TASK 11.**
- Only `load_development_data()` (train + validation) is used.
- No test caption is preprocessed, no test image is opened, and no test
  feature exists.
- A unit test checks that none of the Task 11 sources contains
  `load_split("test"` or `allow_test=True`.
- **Validation only selected things; it never fitted anything:**
  - all fitted statistics (standardisers, LR scaler, MLP weights) use
    train only;
  - validation was used only for early stopping, epoch selection and
    comparison.
- **The main model (M02) was declared before training**, not picked by
  score.
- The canonical split is unchanged (SHA-256 `47fb02a87b36…`).
- Task 8–10 result files were read and not modified (§21 verification).

## 21. Implications for Task 12
1. **Regularise the fusion MLP.** It overfits within 1–2 epochs. Tune
   dropout, weight decay, projection size and learning rate on
   validation, and consider **selection by validation loss** or
   averaging over seeds, given the noisy macro-F1 peak.
2. **Run several seeds** for M02/M03 to measure run-to-run variation
   before claiming any difference between them.
3. **Fine-tuned vs frozen text inside fusion** showed no difference.
   Task 12 can keep the cheaper frozen representation unless tuning
   changes this. Out-of-fold T05 features are the principled alternative
   if fine-tuned text is kept.
4. **Calibration** (on validation) is needed before any score is shown in
   the app.
5. **Shortcut checks** (gender presence, empty caption, watermark,
   resolution) must be repeated for the tuned model.
6. **Final RQ2 answer:** after tuning is frozen, evaluate the chosen
   text-only, image-only and multimodal configurations **once** on the
   locked test set.
