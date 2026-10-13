# OCR Pipeline & OCR Robustness Report (Task 14)

Date: 2026-10-13 · OCR pipeline v1.0

Code:
- `src/ocr/engine.py`: `extract_text_from_image()`
- `src/ocr/pipeline.py`: `analyze_meme_input()` and the fallback `route()`
- `src/ocr/metrics.py`: CER/WER
- `src/ocr/cache.py`
- `scripts/ocr_experiment.py`: `select`, `ocr-test`, `evaluate`
- `scripts/ocr_evaluate.py`
- `scripts/report_ocr.py`
- `tests/test_ocr.py`

Config: `config/ocr.yaml`.

Results:
- `results/tables/ocr/`, `results/figures/ocr/`
- `results/metrics/ocr/test_plan.json`
- per-sample OCR text: `results/predictions/ocr/` (git-ignored, dataset-derived text)

> **Frozen models only.** No model was trained, tuned or changed. The Task 13 final models and TEST results are untouched; their checksums were re-verified before every TEST step. The TEST split was read only through the Task 13 access point.

Tags:
- **[OCR QUALITY]**: OCR text compared with the dataset caption.
- **[MODEL PERFORMANCE]**: frozen classifiers on TEST.
- **[QUALITATIVE]**: manual inspection of a small sample.
- **[INTERPRETATION]**: our reading of the measurements.

---

## 1. Objective
- **What:** build a reusable OCR step for meme images (image → text), and measure how OCR text changes the frozen classifiers' TEST performance.
- **Conditions compared:**
  - REFERENCE: the dataset caption;
  - RAW / BASIC / HINGLISH-normalised OCR text.
- **Question:** how does OCR-extracted text affect real-world meme classification compared with the dataset-provided text? Does the image make the multimodal model more robust to OCR errors?

## 2. OCR engine candidates
| Engine | Languages / scripts | Meme suitability | Install on this Windows CPU machine | Licence | Confidence output |
|---|---|---|---|---|---|
| **EasyOCR 1.7.2** | English + Hindi/Devanagari recognisers (`en`, `hi`); CRAFT text detector | Handles coloured backgrounds and multi-line captions; region boxes with rotation | pip only (`easyocr`, `torchvision` CPU, small deps); weights downloaded on first use | Apache-2.0 | Per-region confidence + polygon |
| Tesseract 5 | English + `hin` traineddata | Weaker on stylised meme fonts and text over images without heavy pre-processing | System program (winget) + language data + `pytesseract` | Apache-2.0 | Per-word confidence |
| PaddleOCR | Multilingual incl. Devanagari | Strong detector / recogniser | Needs `paddlepaddle` (several hundred MB); harder on Windows | Apache-2.0 | Per-line confidence |

## 3. Selected OCR engine
**EasyOCR 1.7.2**, languages `["en", "hi"]`, CPU. The developer approved this choice.

**Installed:**
- `easyocr` 1.7.2 and `torchvision` 0.29.1+cpu (CPU index, installed without dependencies so that `torch` stayed at 2.14.1+cpu);
- `scikit-image`, `shapely`, `pyclipper`, `python-bidi`, `ninja`.

**Weights** (`~/.EasyOCR/model`, 285 MB):
- CRAFT detector `craft_mlt_25k.pth`, 83 MB;
- Devanagari recogniser `devanagari.pth`, 215 MB. It also reads Latin script, so it covers English and romanised Hinglish.

**Safety check after installation [MEASURED]:**
- SigLIP still loads the pinned **PIL** processor (`SiglipImageProcessorPil`) now that `torchvision` is present. The Task 10 `backend="pil"` pin worked.
- All 19 frozen models still reproduce their validation predictions exactly.

## 4. Selection rationale
- **It matches the training text.** MIMIC's own `ExtractedText` was produced with EasyOCR (dataset card: "EasyORC"). Using the same engine keeps the text the application sends to the frozen models close to what they were trained on.
- **It covers the scripts in MIMIC:** English, romanised Hinglish and Devanagari, via the `hi` recogniser that includes Latin characters.
- **It is easy to deploy:** pip-only, Apache-2.0, CPU-capable, and it reports per-region confidence and boxes, which the fallback policy needs.
- **Costs:** CPU speed (§17) and a 285 MB weight download.

## 5. OCR preprocessing
- **Kept separate from the Task 10 vision preprocessing.** Images are decoded and validated with the shared `load_image()`, then prepared for OCR only.
- **Original files are never modified** (a test checks this).
- **Two pre-declared variants:**
  - `none`: RGB as decoded; EasyOCR grays internally.
  - `grayscale_autocontrast`: grayscale + autocontrast, for low-contrast captions.
- **No blind stacking** of thresholding, resizing or other filters.
- **Selection on VALIDATION only:**
  - a stratified subset of 150 validation memes (75 per class, seed 42);
  - pre-declared rule: lowest corpus CER of HINGLISH-normalised OCR vs HINGLISH-normalised caption; ties go to `none`.
  - The choice was written to `config/ocr.yaml` and `results/metrics/ocr/test_plan.json` **before** any TEST meme was OCR'd. Result: §8.
- **OCR text then goes through the unchanged Task 6 `preprocess()`**, giving `ocr_text_raw`, `ocr_text_basic` and `ocr_text_hinglish_normalized`. There is no separate normalisation code.

- **For the multimodal model**, the OCR text goes through T05's HINGLISH preprocessing (M04's text branch is T05), exactly as the dataset caption did in Task 13.

## 6. Evaluation subset
**System verified first (14.1):**
- final models T05 / I03 / M04 (plus B07) as in `results/final/final_model_registry.json`; `verify_freeze()` passed (34 checksums) before every TEST step;
- Task 6 preprocessing v1.0; label mapping `target_label` 1 = misogynistic, 0 = non-misogynistic;
- canonical split (SHA-256 `47fb02a87b36…`);
- reference text = `text_raw` (MIMIC `ExtractedText`); images via `image_path`, which is used only to open the file, never as a feature.

| Use | Split | Samples | Classes (non-mis. / mis.) | Selection |
|---|---|---|---|---|
| Choose the OCR preparation | VALIDATION | 150 | 75 / 75 | stratified random, seed 42 (ID-list SHA-256 `2e23047f…` in the test plan) |
| OCR quality + robustness | TEST | **758 (all)** | 382 / 376 | none: every TEST meme, so no easy-example filtering |
| Check that captions match the picture | TEST | 16 | – | random, seed 2026; images and captions only |

- **All 758 TEST images were readable;** none were excluded.
- **21 TEST memes have an empty caption.** CER/WER are undefined for them, so OCR quality uses **737** memes. Classification uses all 758.
- **Why TEST, given that Task 13 already used it** (developer's choice, "validation tune, full TEST eval"):
  - every setting was fixed on VALIDATION, or declared in the test plan, before any TEST meme was OCR'd;
  - the TEST results change nothing: no model, threshold, preprocessing rule or OCR setting.

  This is a **post-hoc robustness measurement** on frozen models.
- **The 16-meme caption check:**
  - no model prediction and no OCR output was looked at;
  - it informs only how CER is interpreted (§7);
  - result `results/tables/ocr/reference_text_check.csv` [QUALITATIVE]: **8 close, 8 partial**. Partial means a caption line or region was missing, the reading order differed, or words were garbled.

## 7. CER/WER methodology
- **Definitions:**
  - CER = character edit distance (Levenshtein, over Unicode code points) ÷ reference characters;
  - WER = the same over whitespace-separated words.
- **Light normalisation of both strings** (`compare_form`): Unicode NFC, lower case, whitespace collapsed.
- **Same preprocessing level on both sides:** raw OCR vs raw caption, BASIC vs BASIC, HINGLISH vs HINGLISH. Normalised OCR is never compared with an un-normalised caption (14.8).
- **Two summaries:**
  - **corpus (micro)**: total edits ÷ total reference characters or words;
  - the **per-meme median**.
- **CER can exceed 1** when the OCR text is longer than the caption.
- **What CER measures here** [INTERPRETATION]:
  - **The reference is not ground truth.** MIMIC's caption is the dataset authors' EasyOCR output, so CER measures *agreement with the dataset's OCR*, not accuracy against what a human would read.
  - **The two can disagree for harmless reasons:** different versions, settings, reading order or region merging.
  - **Very high CER often means the caption is incomplete.** The caption check (§6) found half the captions partial, and the worst-CER review (§9) found 8 of 16 cases where the caption kept only a fragment while our OCR read the full visible text.
  - **The median is therefore the more robust summary.**

## 8. OCR quality results
**[OCR QUALITY] TEST, 758 memes** (`results/tables/ocr/ocr_quality_summary.csv`)

| Measure | Value |
|---|---|
| OCR success (text found) | **757 / 758 (99.87%)** |
| No text detected / empty OCR | 1 (0.13%); that meme's caption is also empty |
| OCR engine errors / invalid images | 0 / 0 |
| OCR found text where the caption is empty | 20 of 21 |
| Mean / median OCR confidence (mean of region confidences) | 0.5746 / 0.5967 |
| Memes with confidence < 0.30 | 48 |

| Level (both sides) | Corpus CER | Corpus WER | Median CER | Median WER |
|---|---|---|---|---|
| RAW | 0.3307 | 0.4399 | 0.2456 | 0.3333 |
| BASIC | 0.3299 | 0.4378 | 0.2444 | 0.3333 |
| HINGLISH-normalised | 0.3297 | 0.4378 | 0.2439 | 0.3333 |

- **Distribution** (HINGLISH level, 737 memes):
  - 377 have CER ≤ 0.25;
  - 171 have 0.25 < CER ≤ 0.5;
  - 189 have CER > 0.5.

  Figures: `results/figures/ocr/cer_distribution.png`, `wer_distribution.png`.
- **Normalised vs raw OCR (14.8):**
  - CER is practically unchanged across levels (0.3307 → 0.3297). Task 6 normalisation is applied to both sides, so it neither creates nor removes much disagreement.
  - Individual memes move both ways; examples in `ocr_examples.csv`: "normalization improves / worsens CER".
- **Validation selection** (150 memes; `preparation_selection_validation.csv`):
  - `none`: corpus CER 0.2805;
  - `grayscale_autocontrast`: 0.2856.

  ⇒ **`none` selected** by the pre-declared rule. Both variants had an empty-OCR rate of 0.0067 (1 of 150).

**By group** (`ocr_quality_by_group.csv`; flags are automatic heuristics):

| Group | Memes | Median CER | Mean CER |
|---|---|---|---|
| all | 758 | 0.2439 | 0.5702 |
| Devanagari in caption | 212 | 0.2663 | 0.6489 |
| Hinglish spelling variant in caption | 98 | 0.1881 | 0.3830 |
| many text regions (> 6) | 192 | 0.3349 | 0.6986 |
| rotated text (> 10°) | 78 | 0.3622 | 0.7824 |
| low resolution (short side < 400 px) | 14 | 0.4186 | 0.9091 |
| very small text (median box < 2.5% of height) | 16 | 0.8077 | 1.8055 |
| low contrast (gray std < 40) | 2 | 0.1220 | 0.1220 |
| **OCR confidence < 0.30** | 48 | **1.3814** | 2.5837 |

- **Emoji:** the group is empty (0 memes) because neither the dataset OCR nor ours outputs emoji.
- **Confidence is informative** [INTERPRETATION]: memes with confidence < 0.30 disagree with the caption far more (median CER 1.38 vs 0.24). This supports using confidence in the fallback (§15).

## 9. OCR error analysis
- **`results/tables/ocr/ocr_errors.csv` has 189 TEST memes** (99 non-misogynistic, 90 misogynistic). They are every meme with HINGLISH-level CER > 0.5. There were no empty-OCR or engine-error cases with a non-empty caption.
- **What each row holds:** status, confidence, CER, WER, automatic flags, and the manual category where reviewed. **No meme text** is stored in the tracked table.
- **Automatic flags among the 189:**
  - many regions 65; Devanagari 59; rotated 25; Hinglish variant 16; very small text 14; low resolution 3;
  - none 71;
  - confidence < 0.30: 32.

**Manual review [QUALITATIVE]:** the 16 worst-CER memes (CER about 4–30) were inspected visually. The notes are paraphrased; no harmful text is reproduced.

| Category | Memes | What happened |
|---|---|---|
| other: **reference caption incomplete** | 8 | The caption kept one word or one line; our OCR read the whole visible text, sometimes with character errors. High CER here is mostly **not** an OCR failure. |
| stylized font | 3 | Decorative or italic captions, logos, and emoji read as symbols or digits |
| multiple text regions | 2 | Repeated watermark handles or several captions merged across regions |
| screenshot/UI content | 2 | Chat bubbles or video-app interface text (titles, counters) read as caption |
| Devanagari issue | 1 | Long Devanagari caption over a photo, with character substitutions |

- **Categories not seen** among the reviewed memes: low contrast, no text, and a romanised-Hinglish-specific failure.
- **Not reviewed:** the remaining 173 rows are marked "not manually reviewed". The categories describe the extreme tail, not all errors.
- Figure: `ocr_failure_categories.png`.

**[INTERPRETATION]**
- **Two different things raise CER:**
  1. genuine OCR errors (stylised fonts, small or rotated text, Devanagari substitutions);
  2. captions that cover less text than the image shows.
- **The second kind does not hurt the models.** On memes with CER > 0.5, OCR text did not lower macro-F1 (§13).
- **Watermarks:** our OCR also reads watermark handles. The watermark shortcut found in EDA therefore remains present in a deployed OCR pipeline.

## 10. Reference-text classification
- **REFERENCE = the Task 13 TEST predictions**, reused as they are (`results/predictions/final/*_test_predictions.csv`). Nothing was re-run (14.10).
- **Code-path check [MEASURED]:** the OCR experiment's "forced preprocessing" path, given the *reference* caption, reproduces B07's Task 13 predictions on 758 of 758 memes.

**[MODEL PERFORMANCE] TEST, n = 758, macro-F1 (95% CI)**

| Model | Input | Macro-F1 |
|---|---|---|
| B01 TF-IDF LR, RAW | caption | 0.6319 (0.598–0.666) |
| B02 TF-IDF LR, BASIC | caption | 0.6346 (0.600–0.669) |
| B03 TF-IDF LR, HINGLISH | caption | 0.6359 (0.601–0.670) |
| **B07** TF-IDF (tuned), HINGLISH | caption | **0.6359** (0.602–0.670) |
| T04 fine-tuned MuRIL, BASIC | caption | 0.6736 (0.6395–0.7069) |
| **T05** fine-tuned MuRIL, HINGLISH (FINAL_TEXT_MODEL) | caption | **0.6476** (0.613–0.682) |
| **M04** fusion (FINAL_MULTIMODAL_MODEL) | image + caption | **0.8031** (0.774–0.831) |
| **I03** SigLIP + LR (FINAL_IMAGE_MODEL) | image only | **0.8047** (0.776–0.832) |

These equal the Task 13 results; they are not new measurements.

## 11. Raw-OCR classification
OCR text replaces the caption. The image, weights, labels and decision rule (argmax) are the same, and nothing is retrained (14.11). Full metrics are in `results/tables/ocr/ocr_robustness_results.csv`.

- **Compatible use of raw OCR text:**
  - B01 was trained on RAW text, so it receives raw OCR text through its own pipeline;
  - result: **0.6278** (reference 0.6319).
- **Mismatch ablation (`OCR_RAW_ONLY`)** (documented, not a deployable configuration):
  - the HINGLISH-trained finals receive raw OCR text with their preprocessing **skipped**;
  - B07 **0.6304**, T05 **0.6385**, M04 **0.8031**.

  It answers "what if the app forgot to preprocess?".

## 12. Normalized-OCR classification
- **`OCR_NATIVE`, each model's own frozen preprocessing applied to the OCR text.** This is the deployable pipeline:
  - HINGLISH models: B03 **0.6289**, B07 **0.6263**, T05 **0.6372**, M04 **0.8018**;
  - BASIC models: B02 **0.6291**, T04 **0.6623**.
- **`OCR_BASIC_ONLY` (ablation):** HINGLISH-trained finals receive BASIC OCR text, with the Hinglish step skipped:
  - B07 **0.6276**, T05 **0.6438**, M04 **0.8018**.
- **Compatibility:**
  - `OCR_NATIVE` is fully compatible;
  - `OCR_RAW_ONLY` / `OCR_BASIC_ONLY` deliberately feed a level the model was not trained on, and are labelled as ablations everywhere.

## 13. Performance degradation
**OCR drop = reference macro-F1 − OCR macro-F1** (positive = OCR is worse). 95% CI from a paired bootstrap (10,000 resamples, seed 42) on the same memes. "Changed" = memes whose predicted label differs from the reference condition.

| Model | Condition | Macro-F1 | Drop | Drop 95% CI | Changed (of 758) |
|---|---|---|---|---|---|
| B01 | OCR native (RAW) | 0.6278 | +0.0041 | −0.023 to 0.031 | 109 |
| B02 | OCR native (BASIC) | 0.6291 | +0.0055 | −0.021 to 0.032 | 104 |
| B03 | OCR native (HINGLISH) | 0.6289 | +0.0069 | −0.020 to 0.033 | 105 |
| T04 | OCR native (BASIC) | 0.6623 | +0.0113 | −0.016 to 0.040 | 115 |
| **B07** | OCR native | 0.6263 | **+0.0096** | −0.018 to 0.037 | 111 |
| **T05** | OCR native | 0.6372 | **+0.0104** | −0.015 to 0.036 | 92 |
| **M04** | OCR native | 0.8018 | **+0.0013** | −0.013 to 0.016 | 31 |
| B07 / T05 / M04 | OCR raw only | 0.6304 / 0.6385 / 0.8031 | +0.0055 / +0.0091 / +0.0001 | all include 0 | 112 / 99 / 30 |
| B07 / T05 / M04 | OCR basic only | 0.6276 / 0.6438 / 0.8018 | +0.0083 / +0.0038 / +0.0013 | all include 0 | 116 / 91 / 29 |
| M04 | OCR native + fallback (§15) | 0.8031 | 0.0000 | −0.015 to 0.015 | 32 |

Figure: `reference_vs_ocr_macro_f1.png`.

**[MODEL PERFORMANCE]**
- **Every drop is small** (≤ 0.011 macro-F1), and **every 95% CI includes 0**. No measurable macro-F1 loss from replacing the dataset caption with our OCR text.
- **Individual predictions do change:**
  - text models: 12–15% of memes (92–116);
  - M04: about 4% (29–32).
- **The changes roughly cancel in macro-F1.** Aggregate stability hides per-meme instability.

**By population** (`results/tables/ocr/ocr_robustness_by_subset.csv`; OCR native):

| Population | n | B07 drop | T05 drop | M04 drop |
|---|---|---|---|---|
| all TEST | 758 | +0.0096 | +0.0104 | +0.0013 |
| caption not empty | 737 | +0.0124 | +0.0122 | +0.0013 |
| without `doublemean` watermark | 744 | +0.0101 | +0.0108 | +0.0014 |
| caption empty | 21 | −0.2227 | −0.1714 | −0.0053 |
| CER ≤ 0.25 | 377 | +0.0161 | +0.0130 | 0.0000 |
| 0.25 < CER ≤ 0.5 | 171 | +0.0292 | +0.0351 | +0.0294 |
| CER > 0.5 | 189 | −0.0108 | −0.0131 | −0.0214 |

**[INTERPRETATION]** (descriptive; no CIs computed for the subsets; small n):
- **Watermark:** results with and without the watermark memes are nearly identical.
- **Empty captions:** for the 21 empty-caption memes, OCR text *raised* the text models' macro-F1 from 0.30 to about 0.50. OCR supplies text the dataset lacks, and avoids the empty-caption → misogynistic shortcut. With n = 21, this is anecdotal.
- **Largest drop in the middle bin** (0.25 < CER ≤ 0.5): genuine OCR noise there seems to hurt the most.
- **The high-CER bin shows no drop**, consistent with §9: there OCR often reads *more* of the caption than the reference.

**Normalisation recovery (14.13)** (`normalization_recovery.csv`, `raw_vs_normalized_ocr.png`): change from raw-only to HINGLISH OCR text.

| Model | Reference | OCR raw only | OCR basic only | OCR HINGLISH (native) | Recovered by normalisation | 95% CI |
|---|---|---|---|---|---|---|
| B07 | 0.6359 | 0.6304 | 0.6276 | 0.6263 | −0.0041 | −0.016 to 0.008 |
| T05 | 0.6476 | 0.6385 | 0.6438 | 0.6372 | −0.0013 | −0.012 to 0.010 |
| M04 | 0.8031 | 0.8031 | 0.8018 | 0.8018 | −0.0013 | −0.007 to 0.004 |

- **[MODEL PERFORMANCE] Normalisation recovered nothing.** The raw-to-HINGLISH change is slightly negative for all three models, and all CIs include 0.
- **This matches Task 13 RQ1**, where normalisation was inconclusive and leaned negative.
- **No preprocessing change follows from this.** It would be a post-hoc change made on TEST (14.21).

## 14. Text vs multimodal robustness
**[MODEL PERFORMANCE]** (`text_vs_multimodal_robustness.csv`, `text_vs_multimodal_degradation.png`)

| | T05 text-only | M04 image + text |
|---|---|---|
| Reference macro-F1 | 0.6476 | 0.8031 |
| OCR macro-F1 (native) | 0.6372 | 0.8018 |
| **Drop** | **+0.0104** (CI −0.015 to 0.036) | **+0.0013** (CI −0.013 to 0.016) |
| Predictions changed | 92 (12.1%) | 31 (4.1%) |

- **The multimodal model degrades less.** Difference of drops (T05 − M04) = **+0.0091**, paired-bootstrap 95% CI **−0.0128 to 0.0309**; M04 degrades less in 78.5% of resamples.

**[INTERPRETATION]**
- **Direction:** M04 degraded less than T05, and its predictions changed about three times less often.
- **Not statistically supported:** the difference in macro-F1 degradation has a CI that includes 0, and neither model's own drop is distinguishable from 0. **We therefore do not claim** that the image modality makes the model significantly more robust to OCR errors in macro-F1.
- **Supported descriptively:** M04's *predictions are more stable* under OCR noise.
- **Why this is expected:** Task 13 showed M04 relies mostly on the image (it reproduces T05 when the image is removed). The image-only model I03 (0.8047) is unaffected by OCR by construction.

## 15. Fallback design
**Policy:** in `config/ocr.yaml` `fallback`, implemented by `src/ocr/pipeline.route()`.

- **Fixed in advance:** the policy was fixed in the test plan **before** the TEST OCR ran, and the 0.30 confidence threshold was **not** tuned on TEST.
- **Never sends empty text to the fusion model:** `analyze_meme_input()` returns the OCR text and a routing decision, but never passes empty OCR text to the multimodal model.

| OCR outcome | Route | What the application should show |
|---|---|---|
| Invalid / corrupt image | `reject` | "This image could not be read; please upload another file." No analysis. |
| OCR engine error | `image_only` | Image-only prediction (I03) + "text could not be read; image-only analysis" |
| No text detected, or text empty after preprocessing | `image_only` | Image-only prediction + "no text detected" |
| Mean OCR confidence < 0.30 | `image_only_and_human_review` | Image-only prediction + **HUMAN REVIEW** flag ("text unreliable") |
| Otherwise | `multimodal` | Multimodal prediction (M04); text-only and image-only shown for comparison |

**What the policy does on TEST [MEASURED]** (`fallback_routes_test.csv`):
- 709 memes → multimodal;
- 48 → image-only + human review (low confidence);
- 1 → image-only (no text);
- 0 rejected.

M04 with fallback scores macro-F1 **0.8031** (drop 0.0000 vs reference; 32 predictions changed). On the 49 routed memes alone, macro-F1 was:

| Condition | Macro-F1 (n = 49) |
|---|---|
| M04 with the reference caption | 0.8151 |
| M04 with OCR text | 0.8367 |
| Fallback (I03) | 0.8571 |

n = 49 is far too small to draw conclusions. About 6.5% of memes would carry a HUMAN REVIEW flag.

**Rationale:**
- **Image-only is never worse here.** It is as accurate as the multimodal model on TEST (Task 13: 0.8047 vs 0.8031), so falling back to image-only costs nothing measurable.
- **Empty captions were a shortcut.** The text models learned empty caption → misogynistic (Tasks 8–9). Routing empty OCR away from the text path avoids exploiting that shortcut.
- **The threshold is provisional.** 0.30 was chosen as a conservative "very low" level before seeing TEST. Low-confidence memes do disagree much more with the caption (§8), but calibrating the threshold on validation is future work.
- **Moderation wording** stays ALLOW / WARNING / HUMAN REVIEW (project policy; no automatic BLOCK).
- **Scores are uncalibrated.** OCR confidence and model scores must not be shown as probabilities.

## 16. Privacy considerations
- **Uploads are processed in memory.** `analyze_meme_input()` keeps the image and its OCR text only for the duration of the call and returns `"stored": False`. Nothing is written to disk.
- **The research cache covers dataset images only.** `config/ocr.yaml` sets `cache.cache_user_uploads: false`. The application must call the OCR functions without the cache.
- **Planned application behaviour** (Task 16):

  ```
  user meme -> temporary processing -> OCR + prediction -> discard the image
  ```

  - Do **not** retain the raw image unless the user explicitly opts in.
  - MongoDB may later store an analysis ID, the prediction and score type, model versions, the OCR text **only if the privacy policy allows it**, and image metadata (size, format, hash), never the binary.
  - Do not log raw OCR text by default.
- **Repository:** per-sample OCR text of MIMIC memes is dataset-derived text and stays in git-ignored folders (`results/predictions/ocr/`, `*_with_text.csv`). Committed tables hold IDs, flags and numbers.
- **Cache contents** [MEASURED]:
  - 1,058 records, 5.0 MB, outside the repository and OneDrive (`<NLP_CLD_MODEL_DIR parent>/cache/ocr`);
  - each record holds a sample ID, image SHA-256, engine and version, preparation, pipeline version and the OCR result;
  - a scan for credential patterns found only the word "Password" printed inside one meme picture (dataset content, not a secret).

## 17. Runtime/resource requirements
[MEASURED] on CPU (i5-13500H, 12 threads, about 15.7 GB RAM, no GPU).

| Item | Value |
|---|---|
| EasyOCR time per meme, TEST (669 fresh calls; 89 cache hits excluded) | mean **4.84 s**, median 4.56 s, max 16.65 s |
| EasyOCR time per meme, VALIDATION subset (`none` / `grayscale_autocontrast`) | 4.89 s / 4.61 s |
| Fresh OCR compute for TEST | about 54 min (3,237 s) |
| Cache hit | about 0.01 s |
| Weights (`~/.EasyOCR/model`) | 285 MB (CRAFT 83 MB + Devanagari 215 MB) |
| Extra Python packages | `easyocr`, `torchvision` (CPU), `scikit-image`, `shapely`, `pyclipper`, `python-bidi`, `ninja` |
| OCR cache | 1,058 records, 5.0 MB |

- **Not measured:** peak memory and reader start-up time.
- **[INTERPRETATION]** About 5 s of OCR per meme is acceptable for an interactive single-user demo, but slow for batch use. The reader should be loaded once and reused (`get_reader()` is cached). GPU speed was not measured.

## 18. Limitations
- **The reference is EasyOCR output, not ground truth.**
  - CER/WER measure agreement with the dataset's OCR.
  - Using the same engine family probably makes agreement higher than an independent engine would.
  - The experiment compares *dataset OCR vs our OCR*, not *clean text vs OCR*: MIMIC captions, and therefore the models' training text, are already OCR output. A larger drop is possible for a model trained on clean, human-transcribed text; that was not tested.
- **Incomplete captions inflate CER.** 8 of 16 checked captions were partial, and 8 of the 16 worst-CER memes were truncated captions.
- **One engine, one configuration, CPU only.** No second engine was tested. Only two preparations were compared, on 150 validation memes.
- **TEST had already been used (Task 13).** This is a post-hoc robustness measurement. Nothing was changed because of it, and it adds no tuning.
- **Small subsets** (21 empty captions, 49 fallback-routed) and the CER bins are descriptive, with no CIs.
- **The fallback threshold (0.30) is uncalibrated,** and OCR confidence is the mean of EasyOCR region scores, not a probability.
- **The error categories are thin.** Manual categories cover only the 16 worst memes, and the automatic flags are heuristics. Emoji are invisible to both OCR outputs.
- **The watermark shortcut remains:** OCR reads `@doublemeaning` handles too.
- **Multicategory (14.22):** stage-2 sub-category models are not trained (Task 8.5), so there is no auxiliary text classifier to test. Only the binary misogyny models were evaluated.
- **Config location (deviation from 14.26):** `config.yaml` is checksum-frozen by the Task 13 freeze. Editing it would break `verify_freeze()`, so the OCR settings live in `config/ocr.yaml`. The same rigidity is why the OCR code is in `src/ocr/`, not `src/preprocessing/`.

## 19. Research conclusions
1. **[OCR QUALITY]** EasyOCR (en + hi, no extra preparation) found text in 757 of 758 TEST memes. Agreement with the dataset caption was median CER 0.244 and WER 0.333 (corpus 0.330 / 0.438). Much of the high-CER tail reflects incomplete dataset captions rather than OCR failure.
2. **[MODEL PERFORMANCE]** Replacing the caption with our OCR text lowered macro-F1 by at most 0.011 for every frozen model:
   - T05 −0.0104, B07 −0.0096, M04 −0.0013;
   - all 95% CIs include 0.
   - The deployable OCR pipeline therefore performs close to the reported TEST results (M04 0.8018, T05 0.6372).
3. **[MODEL PERFORMANCE]** Hinglish normalisation did **not** recover any OCR loss (−0.001 to −0.004, CIs include 0). This is consistent with Task 13's RQ1 finding.
4. **[INTERPRETATION]** The multimodal model was more *stable*: 4% of its predictions changed vs 12% for T05, and its macro-F1 drop was smaller. The difference in macro-F1 degradation is **not statistically supported** (CI −0.013 to 0.031), so robustness is not claimed.
5. **Design:**
   - the fallback (image-only, or image-only + HUMAN REVIEW at confidence < 0.30) kept M04's TEST macro-F1 at 0.8031;
   - uploads are processed in memory and never cached.

   Both are ready for Task 16 UI integration.
6. **No model, threshold, preprocessing rule or split was changed.** The Task 13 results stand.
