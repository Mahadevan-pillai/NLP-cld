# Explainability (XAI) & Advanced Error Analysis Report (Task 15)

Date: 2026-10-13 · explainability config v1.0

Code:
- `src/explainability/text.py`: `explain_text()`, Integrated Gradients / exact linear contributions
- `src/explainability/image.py`: `explain_image()`, occlusion
- `src/explainability/multimodal.py`: `explain_multimodal()`, unimodal comparison + modality ablation
- `src/explainability/moderation.py`: `recommend()`
- `src/explainability/explanation.py`: `explain_meme()` / `explain_text_input()`, the UI object
- `src/explainability/errors.py`: error taxonomy
- `scripts/xai_experiment.py` (`plan`, `compute`, `runtime`)
- `scripts/report_xai.py`
- `tests/test_explainability.py`

Config: `config/explainability.yaml`.

Results:
- `results/tables/explainability/`, `results/figures/explainability/`
- `results/metrics/explainability/xai_plan.json`
- per-meme explanations: `results/predictions/explainability/` (git-ignored, dataset-derived text)

> **Attribution is not proof of causal reasoning.**
> - Every explanation here is **MODEL ATTRIBUTION**: it shows which inputs moved a frozen model's score, under a specific method and baseline.
> - It does not show why a human would find a meme harmful, what anyone intended, or that a highlighted word or region alone "caused" a label.
> - The model classifies content, not people.

> **Frozen models only.** No model was trained, tuned or changed.
> - The Task 13 freeze checksums were re-verified before every TEST step, and TEST was read only through the Task 13 access point.
> - The analysis plan (`xai_plan.json`) was written before any TEST explanation was computed.
> - The Task 13 final metrics are untouched.

Tags:
- **[MEASURED]**: computed from the frozen models.
- **[QUALITATIVE]**: manual inspection by a single reviewer (the AI assistant in this session).
- **[INTERPRETATION]**: our reading.
- **[SYNTHETIC]**: probe sentences written for this task, not dataset content.

---

## 1. Objective
- **Explain** the three frozen final models: text, image and multimodal.
- **Advanced error analysis** of their TEST errors (Task 13) and OCR-induced changes (Task 14).
- **Design for the future application (Task 16):** the explanation object, responsible wording and a moderation recommendation.
- **What a future user should be able to see:**
  - which words moved the text score;
  - which image regions moved the image score;
  - whether the text, image and multimodal models agree;
  - what the explanation cannot tell them.

## 2. Final models explained
Verified (15.1):
- `verify_freeze()` passed: all 34 Task 13 checksums.
- The XAI code path reproduces the Task 13 TEST predictions of all four models exactly (`consistency_checks.csv`, §18).

| Role | Model | Artifact SHA-256 (prefix) | Preprocessing | Score |
|---|---|---|---|---|
| FINAL_TEXT_MODEL | **T05**, fine-tuned MuRIL (`google/muril-base-cased@afd9f36c`), max 64 tokens | `697692cc1d9e` | Task 6 v1.0, HINGLISH_NORMALIZED | uncalibrated softmax |
| FINAL_IMAGE_MODEL | **I03**, frozen SigLIP-B/16 (`@7fd15f06`, PIL processor) + LR (C = 0.001) | `73c5497e95da` | Task 10 v1.0 (stretch to 224 x 224) | uncalibrated LR probability |
| FINAL_MULTIMODAL_MODEL | **M04**, fusion MLP over T05 pooled [CLS] + SigLIP pooled embedding | `5b132e127981` | as T05 + as I03 | uncalibrated softmax |
| traditional baseline | **B07**, TF-IDF (1–3-grams) + LR | `a2aaae391288` | Task 6 v1.0, HINGLISH_NORMALIZED | uncalibrated LR probability |

- **Inputs:**
  - label map: 0 = non-misogynistic, 1 = misogynistic;
  - TEST predictions: `results/predictions/final/*_test_predictions.csv` (Task 13);
  - OCR artifacts: `results/predictions/ocr/` (Task 14; EasyOCR 1.7.2, preparation `none`).
- **Attribution units:** all attributions are in **log-odds of "misogynistic"** (logit difference). Positive = towards misogynistic.
- **T05's scores are compressed:**
  - on TEST they lie between 0.33 and 0.68 (log-odds −0.72 to +0.74);
  - they cluster near both ends, so T05 behaves almost like a two-state detector;
  - its word attributions therefore sum to at most about ±1 log-odds (largest |f(input) − f(baseline)| on TEST: 1.02).

## 3. Text explanation method
**T05: Integrated Gradients (IG) on the input word embeddings**
- **Path:** 32 midpoint steps from a baseline in which every non-special token is replaced by the [PAD] embedding. [CLS] and [SEP] are kept.
- **Target:** the logit difference.
- **WordPiece pieces are summed per word.** Words follow the tokenizer's pre-tokenisation: punctuation is separate and English contractions are split.
- **`[UNK]` pieces and truncation are flagged.** Words beyond 64 tokens were never seen and get no attribution.
- **Completeness check:** IG's attributions should sum to f(input) − f(baseline). The remaining gap is returned with every explanation (`completeness_error`). On the synthetic smoke test it was 0.002 log-odds on a 0.74 output.
- **Same code path as inference:** the text goes through T05's own Task 6 `preprocess()`. The explained output is the same forward pass the app uses; the consistency test checks the predicted score against `predict_text()`.

**B07: exact linear decomposition**
- **Contribution of an n-gram** = LR coefficient × its TF-IDF value. Sum + intercept = the decision function exactly; the maximum error on TEST is reported in §18.
- **Word highlights:** each n-gram's contribution is split equally over its words. This is a display approximation, labelled as such.
- **Words outside the vocabulary** contribute 0 and are marked `not_in_vocabulary`. Nothing is attributed to words the model cannot see.

**UI highlighting format** (`tokens` in the explanation object):

```json
[{"token": "...", "position": 0, "contribution": 0.42, "direction": "towards misogynistic", "unknown_token": false}, ...]
```

## 4. Image explanation method
**I03: occlusion sensitivity**
- **Procedure:**
  - a mid-grey square (56 × 56 px of the 224 × 224 model input; 0 in SigLIP-normalised space) slides with stride 28;
  - that gives 49 occluded copies;
  - each copy goes through the real pinned processor → frozen SigLIP → I03 classifier.
- **Value of a region:** the drop in log-odds when it is hidden. Each of the 8 × 8 cells averages the windows covering it.
  - Positive = the region supported "misogynistic";
  - negative = it supported "non-misogynistic".
- **Mapping to the original image:** the model input is the image stretched to a square, so a cell maps to the same fraction of the original image.
- **Overlays** (`overlay()`) are new images; the original files are never modified (a test checks this).
- **Research overlays** of dataset memes are git-ignored (`*_with_content.png`).

**Validity checks:**
- the occluded and full scores come from the real model;
- model-randomisation sanity check: random linear classifiers on the same occluded embeddings (§9).

**Faces.** Automatic face detection was **not** done.
- OpenCV 5.0 (installed) no longer ships the Haar cascade detector, and a separate face model would be an extra download.
- People and faces were assessed only in the manual heatmap review.
- Text regions were measured automatically, using the Task 14 OCR boxes.

## 5. Multimodal explanation method
**M04:** modality comparison + modality ablation (15.8–15.9).
1. **Single-modality comparison.** T05 (text only), I03 (image only) and M04 labels and scores, from the inference functions the app uses.
2. **Train-mean ablation inside M04.** One modality's features are replaced by their TRAIN mean, which M04's own standardiser maps to exactly 0 ("an average training meme" for that modality).
   - It is the same pre-specified neutralisation as Task 13's reliance diagnostic, not an arbitrary zero vector in raw feature space.
   - Reported: the label and log-odds with each modality neutralised, and a label-flip summary: depends on the text / on the image / needs both / unchanged.
3. **Agreement.** The number of single-modality models (T05, I03) whose label differs from M04's:

   | Disagreeing models | Category |
   |---|---|
   | 0 | **AGREE** |
   | 1 | **PARTIAL DISAGREEMENT** |
   | 2 | **DISAGREE** |
4. **Attribution through the fusion model** (research and sample set):
   - IG over the T05 encoder into M04, with M04's image features fixed;
   - occlusion through SigLIP into M04, with its text features fixed.

**No "percentage contribution" of a modality is reported.** The fusion MLP is non-linear, the two ablation effects do not add up, and none of the methods supports such a split.

## 6. Method-selection rationale

| Model | Chosen | Not used, and why |
|---|---|---|
| T05 (fine-tuned transformer) | **Integrated Gradients** (Sundararajan, Taly & Yan, ICML 2017, [PMLR 70](https://proceedings.mlr.press/v70/sundararajan17a.html)): gradient-based, needs no model change, has a checkable completeness property, about 1–2 s on CPU | SHAP / LIME: not installed; sampling-based and slower, with no exactness guarantee for a transformer. Attention weights: not attributions of the output. |
| B07 (TF-IDF + LR) | **coefficient × TF-IDF**: exact for a linear model; milliseconds | SHAP would reproduce the same linear decomposition at extra cost |
| I03 (ViT + LR) | **Occlusion** (Zeiler & Fergus, ECCV 2014, [PDF](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)): model-agnostic, measures the real model output | Grad-CAM: needs convolutional feature maps; SigLIP is a ViT. Attention rollout: SigLIP pools with an attention head; attention is not a faithful attribution of the classifier output. |
| M04 (fusion MLP) | **Unimodal comparison + train-mean ablation** (pre-specified in Task 13); IG and occlusion through the fusion model | Zero vectors in raw feature space: off-distribution. Percentage splits: not supported by any method used. |

- **Faithfulness sanity checks** follow the idea of Adebayo et al., "Sanity Checks for Saliency Maps" (NeurIPS 2018, [proceedings](https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps)): an attribution should change when the model's weights are randomised.
- **The citations were verified online in this session.**

## 7. Representative explanations
**Explanation sample set (15.10)** (`explanation_sample_set.csv`; IDs, labels and predictions only)
- **30 TEST memes, 3 per category**, drawn deterministically (seed 42, each meme used once):
  - true positive, true negative, false positive, false negative (M04);
  - text/image disagreement (T05 ≠ I03);
  - multimodal correction (I03 wrong, M04 right);
  - multimodal failure (M04 wrong while T05 or I03 is right);
  - OCR failure (Task 14: the M04 or T05 label changes with OCR text);
  - Hinglish example (Task 6 variant rule);
  - negation example (Task 6 negation word).
- **What each sample-set meme has:**
  - T05 IG and B07 linear contributions;
  - I03 and M04 occlusion maps;
  - M04 ablation;
  - IG through M04.
- **The rest of TEST also has explanations:**
  - all 758 TEST captions: T05 IG + B07 linear;
  - 351 memes: image occlusion (200 uniform random + every I03/M04 error + the sample set).

**Where the content is (git-ignored):**
- per-meme explanations: `results/predictions/explainability/*.jsonl`;
- overlays: `results/figures/explainability/image_attribution_examples_with_content.png`.

**Tracked examples use synthetic inputs:**
- `text_attribution_example.png`: probe sentences;
- `results/metrics/explainability/explanation_object_example.json`: the full UI object for a generated image and invented sentences.

**Examples [MEASURED; QUALITATIVE description, no meme text reproduced]:**
- **Neutral probe, female vs male subject [SYNTHETIC].** "she is going to the office" gets T05 log-odds +0.72 and label 1; "she" carries +1.03. With "he", it is −0.58 and label 0. `text_attribution_example.png`.
- **Couple meme, "husband / wife" layout (sample-set false positive).** I03's map is positive on the woman's face and negative on the man's face, and the two nearly cancel (log-odds −0.00).
- **Swimsuit post (sample-set false positive).** The strongest positive cell is the swimsuit body region. The meme is labelled non-misogynistic.

## 8. Hinglish/negation analysis
**[SYNTHETIC] probe sentences** (`xai_probes.csv`, `xai_probe_summary.csv`): 14 minimal pairs written for this task. They are illustrative, not a sample.

| Probe type | Pairs | T05 label changes | B07 label changes | T05 mean log-odds change |
|---|---|---|---|---|
| negation added (e.g. "… hoti hain" / "… nahi hoti hain"; "she is smart" / "she is not smart") | 6 | **0** | 1 | 0.002 |
| female → male subject in a neutral sentence | 6 | **4** | 4 | 0.87 |
| repeated letters / abbreviations ("kr rhi" for "kar rahi") | 2 | 0 | 1 | −0.001 |
| 5 spellings of the negation (nahi / nhi / nahin / nai / ni) | 5 | 0 | 0 | max − min = 0.002 |

- **T05 labels every probe containing a female-reference word "misogynistic"** (log-odds about +0.72), whatever the sentence says.
- **Negation hardly matters:**
  - adding "nahi" / "not" changes T05's log-odds by 0.002 on average;
  - in "she is not smart", "not" itself pushes towards misogynistic (+0.20).
- **The text normalisation works as designed:**
  - "nahi / nhi / nahin" give identical T05 scores (Task 6 maps them to the same form), and so do "kr rhi" and "kar rahi";
  - repeated letters are cut to two by Task 6 ("ladkiii" becomes "ladkii").
  - T05 still flags "ladkii", but B07's label changes because "ladkii" is not in its vocabulary.

**[MEASURED] on TEST**

T05 Integrated Gradients, all 758 captions (`lexical_cue_attribution.csv`):

| Word group (lexicons fixed in the plan) | Occurrences (memes) | Mean attribution | … in non-misogynistic memes | Share positive | In the top-3 words of its memes |
|---|---|---|---|---|---|
| female-reference words | 175 (129) | **+0.281** | **+0.245** | 86% | **74%** |
| male-reference words | 146 (128) | −0.108 | −0.161 | 32% | 49% |
| negation words (Task 6 list) | 169 (140) | −0.045 | −0.053 | 35% | 21% |
| watermark handle | 4 (4) | +0.138 | – | 75% | 50% |
| all other words | 8,880 | +0.015 | +0.001 | 54% | – |

- A female-reference word is T05's strongest positive word in 75 of its positive predictions.

Macro-F1 by caption group (`negation_hinglish_groups.csv`):

| Group | n | T05 | B07 | M04 |
|---|---|---|---|---|
| contains a negation word | 141 | 0.598 | 0.632 | 0.808 |
| no negation word | 617 | 0.653 | 0.635 | 0.801 |
| Hinglish spelling variant | 98 | 0.600 | 0.656 | 0.827 |
| no Hinglish variant | 660 | 0.650 | 0.630 | 0.797 |

**[INTERPRETATION] T05 appears to depend heavily on individual lexical cues:**
- female-reference words push towards "misogynistic" even in non-misogynistic memes;
- male-reference words push the other way;
- negation is largely ignored.

T05 is weaker on captions with negation or Hinglish variants (descriptive; no CIs). M04 is not, because it relies mainly on the image (§10).

## 9. Image attribution analysis
**Validity checks [MEASURED]** (`image_attribution_summary.csv`):
- **The maps reflect the trained classifier, not just the encoder.**
  - Five random linear classifiers on the same occluded embeddings give maps unrelated to I03's: median Spearman ρ = **0.01** (uniform random sample, n = 200).
  - This passes the model-randomisation check.
- **The explanation agrees with inference.** The full-input score reproduces I03's and M04's TEST labels for all 351 explained memes.
- **M04's image attribution** (occlusion through the fusion model, text fixed) closely resembles I03's map: median ρ = **0.87**.

**Where the attribution falls (uniform random sample, n = 200; 198 with OCR text boxes):**

| Measure | Value |
|---|---|
| Share of \|attribution\| in the top 10% of cells (uniform would be 0.10) | median **0.25** (moderately concentrated) |
| OCR-text-region enrichment (attribution share towards the predicted class ÷ area share) | median **0.99**; above 1 in 49% of memes |
| … I03 false positives / true negatives (research set) | 0.90 / 1.16 |

**[INTERPRETATION]**
- On average, I03 does **not** concentrate its attribution on rendered text: text regions receive about their area's share.
- It does respond to text regions in individual memes; one sample-set meme's strongest cells are its caption. So **the image attribution must not be described as purely non-textual visual reasoning**.

**Manual heatmap review [QUALITATIVE]** (`image_heatmap_review.csv`): the 30 sample-set memes, single reviewer.
- **Main focus:**
  - person/face: 16;
  - diffuse (no clear focus): 7;
  - body/clothing: 4;
  - object (a car): 1;
  - background: 1;
  - text region: 1.
- **Pictured women and the strongest positive cell:**
  - I03 predicted "misogynistic" for 12 memes; in **10** of them the strongest positive cell lies on a pictured woman.
  - Of the 18 predicted "non-misogynistic", 12 show no woman; men's and children's faces usually push towards non-misogynistic.
- This supports the Task 10 hypothesis of a **gender-presence shortcut** (§13).
- Faces were judged only by eye: no automatic face detector was available (§4).

## 10. Modality contribution/disagreement
**[MEASURED] all 758 TEST memes** (`modality_agreement_summary.csv`, `modality_disagreement.png`)

| Agreement with M04 | Memes | Share | M04 accuracy |
|---|---|---|---|
| AGREE (T05, I03 and M04 same label) | 472 | 62.3% | **0.864** |
| PARTIAL DISAGREEMENT (one single-modality model differs) | 281 | 37.1% | 0.705 |
| DISAGREE (both differ from M04) | 5 | 0.7% | 0.600 |

| M04 label when a modality is set to its train mean | Memes | M04 accuracy |
|---|---|---|
| unchanged either way | 465 (61.3%) | 0.865 |
| depends on the **image** | 208 (27.4%) | 0.789 |
| depends on the **text** | 83 (10.9%) | **0.518** |
| needs both | 2 | 0 of 2 |

**[INTERPRETATION]**
- **The image dominates M04.** Neutralising the image changes 27.7% of its labels; neutralising the text changes 11.2%. This matches Task 13's within-model reliance (macro-F1 0.648 without the image vs 0.802 without the text).
- **Text-dependent decisions are near chance.** When M04's label depends on the text, it is right only 52% of the time, so text-driven decisions are its weak point.
- **M04 uses the text much as T05 does.** Its text attribution (IG through the fusion model) ranks words like T05's (median ρ = 0.91, 51 memes) and is about 1.9× larger in log-odds. The fused model inherits T05's lexical cues when the text decides.
- **Agreement is informative but not proof:** AGREE memes are 86% correct, PARTIAL 70%, and all three models were wrong together on 64 memes (§11).

## 11. Error taxonomy
**Population:** `advanced_error_analysis.csv`, **391 TEST memes**, i.e. every meme where T05, I03 or M04 is wrong, or where the M04/T05 label changes with OCR text.

**Columns:**
- labels, the three predictions and FP/FN types;
- OCR status and route, modality agreement, M04 dependence;
- primary `error_category` / `error_subcategory` and its source;
- automatic flags;
- paraphrased manual notes (no meme text).

**Method (deterministic, `src/explainability/errors.py`):**
1. **Automatic flags.** Several can apply to one meme.
   - OCR: empty caption = *missing text*; label changes with OCR text; CER > 0.5 (Task 14); confidence < 0.30.
   - MULTIMODAL: *fusion failure* (both single-modality models right); *followed the wrong modality*; *all models wrong*.
   - LINGUISTIC: negation, Hinglish variant, 1–3 words, Devanagari.
   - VISUAL: short side < 400 px; text-heavy (≥ 20 words or > 6 OCR regions).
   - DATA: watermark; duplicate/template group within TEST.
2. **Manual review** of **60 of the 149 M04 errors**:
   - 24 from Task 13;
   - 36 new, a stratified random sample (18 FP / 18 FN, seed 42) viewed as image + caption.

   A manual label (meaning: slang, cultural reference, image–text dependency, annotation ambiguity, gender cues …) overrides the rules. One reviewed meme with "no clear content cause" falls back to the rule.
3. **Otherwise, the first automatic rule in a documented priority order is used.** This is a convention, not a measured cause.

**Categories used (only where supported):**

| Category | Subcategories |
|---|---|
| LINGUISTIC | Hinglish spelling variant, negation, slang, sarcasm, ambiguity, very short text, gender-term false alarm |
| VISUAL | text-heavy meme, gender-presence false alarm |
| MULTIMODAL | image–text dependency, followed the wrong modality, fusion failure, all models wrong, gender cues in both modalities |
| OCR | missing text, text-region failure, OCR text changes the prediction |
| CONTEXTUAL | cultural reference, implicit meaning |
| DATA | possible annotation ambiguity |

**Not used** (no supporting evidence):
- *class imbalance*: TEST is balanced, 382 / 376;
- *world knowledge* and *conversational context*: merged into cultural reference / implicit meaning;
- *low resolution* as a primary category: it never came first; it remains a flag.

## 12. Error distribution
`error_distribution.csv`, `error_category_distribution.png`

**Errors by model:**

| Model | Errors | False positives | False negatives |
|---|---|---|---|
| T05 | 267 | 144 | 123 |
| I03 | 148 | 70 | 78 |
| M04 | 149 | 63 | 86 |

**Primary category of the 149 M04 errors** (manual label for 60, rule otherwise):

| Category / subcategory | Errors | % | FP | FN |
|---|---|---|---|---|
| MULTIMODAL / followed the wrong modality (rule) | 41 | 27.5 | 10 | 31 |
| MULTIMODAL / all models wrong (rule) | 17 | 11.4 | 8 | 9 |
| MULTIMODAL / image–text dependency (manual) | 15 | 10.1 | 0 | 15 |
| OCR / OCR text changes the prediction (rule) | 14 | 9.4 | 6 | 8 |
| LINGUISTIC / slang (manual) | 8 | 5.4 | 2 | 6 |
| MULTIMODAL / gender cues in both modalities (manual) | 6 | 4.0 | 6 | 0 |
| DATA / possible annotation ambiguity (manual) | 6 | 4.0 | 4 | 2 |
| CONTEXTUAL / cultural reference (manual) | 6 | 4.0 | 2 | 4 |
| VISUAL / gender-presence false alarm (manual) | 5 | 3.4 | 5 | 0 |
| LINGUISTIC / ambiguity (manual) | 5 | 3.4 | 5 | 0 |
| other (each ≤ 4) | 26 | 17.4 | 15 | 11 |

**The 60 manually reviewed M04 errors only** (closest to a *content* view):

| Manual category | n | % |
|---|---|---|
| image–text dependency | 15 | 25 |
| slang | 8 | 13 |
| cultural reference | 6 | 10 |
| possible annotation ambiguity | 6 | 10 |
| gender cues in both modalities | 6 | 10 |
| ambiguity | 5 | 8 |
| gender-presence false alarm | 5 | 8 |
| other | 9 | 15 |

**Pattern:**
- 27 of the 30 manually reviewed false negatives were an image–text dependency, slang, implicit meaning or a cultural reference: the meaning was carried by words or context that the image model cannot see. The other 3 were annotation ambiguity (2) and sarcasm (1).
- Every *gender-cue* category consists of false positives.

**Most frequent automatic flags among M04 errors (multi-label):**
- followed the wrong modality: 83;
- all models wrong: 64;
- Devanagari: 44;
- text-heavy: 42;
- high OCR disagreement: 42;
- negation: 26;
- OCR changes a label: 26.

**Caveat:** manual categories are one reviewer's judgement on a 40% sample, with no agreement statistic. Rule categories describe correlates, not causes.

## 13. Shortcut analysis
**TEXT: female-reference words [MEASURED]** (`gender_term_shortcut.csv`): false-positive rate on the 382 non-misogynistic TEST memes, by whether the caption contains a female-reference word (47 do).

| Model | FP rate with | FP rate without | Fisher p (Holm) |
|---|---|---|---|
| T05 | **70.2%** | 33.1% | **1.3e-5** |
| M04 | **38.3%** | 13.4% | **6.3e-4** |
| I03 (image only; control) | 34.0% | 16.1% | 0.038 |
| B07 | 42.6% | 35.5% | 0.79 |

- Male-reference words show no such increase.
- **The image-only control is confounded:** I03 cannot read the caption, yet it also flags these memes more often. Captions about women tend to accompany pictured women, so the T05/M04 gap is not caused by the words alone.
- **For T05 the probes and attributions point to the words:**
  - female-reference words have +0.24 mean attribution in non-misogynistic memes;
  - one is the top positive word in about 20% of T05's positive predictions, both true and false positives (`text_correct_vs_incorrect.csv`), but almost never in its negatives;
  - the gender-swap probes flip 4 of 6 labels.
- **[INTERPRETATION]** T05 has learned a **female-reference-word shortcut**.

**TEXT: other artefacts**
- **Empty caption.** All 21 empty TEST captions are predicted misogynistic by T05; only 9 are. This is the known empty-input shortcut, now explicitly handled in the UI design by the Task 14 route and moderation rule T0.
- **Watermark handle.** Present in only 4 captions, with +0.14 mean attribution: too rare to judge.

**IMAGE: gender presence [QUALITATIVE + MEASURED]**
- In the sample set, 10 of 12 I03-positive memes have their strongest positive cell on a pictured woman.
- Five manually reviewed M04 false positives are benign photos of women ("gender-presence false alarm"), and six more combine gender cues in both modalities.
- Rendered text is not systematically over-weighted (enrichment ≈ 1).
- The resolution shortcut found in EDA cannot be seen by occlusion: the encoder sees a stretched 224 px image. It was tested separately in Task 10.

**MULTIMODAL: one modality dominates.**
- The image decides M04's label far more often than the text (§10).
- When the text decides, M04 inherits T05's lexical cues: IG ranks words the same way through M04 (ρ = 0.91).

**Limit of this evidence.** It shows *association with model behaviour* under fixed methods; it does not prove what the model "believes".

## 14. OCR-induced explanation changes
**[MEASURED]** T05 IG on our OCR text vs the dataset caption (`ocr_explanation_changes.csv`, `ocr_explanation_changes.png`):
- the 92 TEST memes whose T05 label changes with OCR text (Task 14);
- 92 random memes whose label does not change (control).

| | Label changed (92) | Control (92) |
|---|---|---|
| T05 correct before → after OCR | 50 → 42 | 67 → 67 |
| Median Jaccard of the top-3 words (each explanation's predicted class) | **0.00** | 0.50 |
| Decisive OCR word (strongest towards the new label) absent from the reference caption | **58.7%** | 28.3% |
| Decisive reference word lost in the OCR text | **40.0%** | 20.0% |
| Median share of words shared | 0.46 | 0.70 |

Mann–Whitney p for the Jaccard difference: 2.4e-23.

**[INTERPRETATION]**
- When OCR changes T05's label, the most influential words change completely.
- In most cases the decisive word is new: text the dataset caption did not contain, a mis-read word, or a word from a region the caption missed.
- In 40% of cases, the reference caption's decisive word was lost.

**OCR thus adds or removes influential tokens; it does not just perturb the same ones.** This is a concrete mechanism behind Task 14's per-meme instability (12% of T05 labels changed) despite a stable macro-F1. Per-meme tokens: `ocr_explanation_changes_with_text.csv` (git-ignored).

## 15. Correct vs incorrect explanation patterns
**Image (I03, research set: 203 correct / 148 incorrect;** `image_correct_vs_incorrect.csv`**)**

| Measure | Correct | Incorrect | Holm p |
|---|---|---|---|
| Concentration (top-10% share) | 0.247 | 0.252 | 0.91 |
| Text-region enrichment | 1.02 | 1.08 | 0.80 |
| Largest cell change | 0.62 | 0.62 | 0.80 |

- **Errors do not show more diffuse maps or more text focus.** The maps of wrong predictions look like those of right ones.
- **[INTERPRETATION]** Errors come from *what* the model attends to (for example, a pictured woman in a benign meme), not from unfocused attention.

**Text (T05):**
- A female-reference word is the strongest positive word in 20% of true positives and 19% of false positives, i.e. equally in right and wrong positives, and almost never in negatives (TN 0.4%, FN 1.6%).
- Positive predictions concentrate more on one word: median top-1 share 0.31 vs 0.23 for negatives. Errors therefore correspond to **misleading lexical cues** rather than to diffuse attribution.

**Multimodal:**
- Errors cluster in modality conflict: 83 of 149 M04 errors are PARTIAL DISAGREEMENT, where M04 followed the wrong single-modality model.
- In text-dependent decisions, accuracy is 52%.

**OCR:** 14 M04 errors are primarily OCR-driven (label changes with OCR text).

**[QUALITATIVE] Ambiguous context:** cultural references, implicit meaning and debatable labels make up about a quarter of the reviewed errors.

## 16. Multicategory explainability status
- **Not applicable.** Only the binary misogyny models are trained.
- The MIMIC sub-labels (Objectification, Prejudice, Humiliation) have **no trained model** (Task 8.5 stage 2 is disabled), so no sub-category explanation was created.
- No SAFE / HATE / ABUSE / other category is modelled or explained.

## 17. Moderation recommendation design
**Design** (`config/explainability.yaml` `moderation`, `src/explainability/moderation.py`):
- deterministic and documented;
- configurable: rule order, outcomes and reasons live in the config;
- uses only predicted labels, agreement and the Task 14 OCR route;
- **no confidence thresholds**, because the scores are uncalibrated;
- **no BLOCK outcome exists**, and the validator rejects one.

| Mode | Rule (first match wins) | Outcome |
|---|---|---|
| meme | M0: image unreadable | no recommendation; ask for another file |
| meme | M1: OCR confidence < 0.30 (Task 14 route) | HUMAN_REVIEW |
| meme | M2 / M3: no text or OCR error → image-only, image model flags / does not flag | HUMAN_REVIEW / ALLOW |
| meme | M4: T05, I03 and M04 all flag | WARNING |
| meme | M5: no model flags | ALLOW |
| meme | M6: otherwise (models disagree) | HUMAN_REVIEW |
| text | T0: processed text empty | HUMAN_REVIEW |
| text | T1 / T2: T05 and B07 both flag / neither flags | WARNING / ALLOW |
| text | T3: otherwise | HUMAN_REVIEW |

**Wording (15.20):**

| Outcome | Wording |
|---|---|
| ALLOW | "Not flagged by the models (model prediction)." |
| WARNING | "Potentially misogynistic content (model prediction). Shown with a warning; open to review." |
| HUMAN_REVIEW | "Flagged for human review: the model outputs are uncertain or disagree." |

- Always paired with the disclaimer: *"Explanations show model attribution and may not reflect human reasoning. The model classifies content, not people."*
- Never use "this proves…", "this user is misogynistic" or "definitely offensive".

**Descriptive TEST simulation [MEASURED]** (`moderation_test_simulation.csv`): the rules were fixed in the plan before this run and were not adjusted afterwards.

| Scenario | ALLOW (truly misogynistic) | WARNING (truly misogynistic) | HUMAN_REVIEW |
|---|---|---|---|
| meme, dataset captions | 242 (11.6%) | 225 (**85.8%**) | 291 (38.4%) |
| meme, our OCR + Task 14 fallback | 228 (11.8%) | 206 (**84.5%**) | 324 (42.7%) |
| text mode, T05 + B07 | 243 (26.7%) | 239 (72.0%) | 276 (36.4%) |

**[INTERPRETATION]**
- **The design is conservative:** about 4 in 10 memes go to a human.
- **What ALLOW still lets through:** about 7% of all misogynistic TEST memes (27–28 of 376) are allowed in meme mode, and 17% (65) in text mode.
- **These rates come from TEST and are descriptive:** changing the rules to improve them would be test-set tuning.
- **The rules inherit the models' shortcuts.** In `explanation_object_example.json`, the neutral [SYNTHETIC] sentence "she is going to the office" receives **WARNING** in text mode, because T05 and B07 both flag it. The UI must therefore show text-mode WARNINGs with the shortcut limitation, and a human should be able to overrule them.

## 18. Runtime
**[MEASURED]** Public functions, warm models (loading excluded), CPU i5-13500H, 5 TEST memes each (`explanation_runtime.csv`):

| Explanation | Mean s | Max s | Use |
|---|---|---|---|
| text: T05 Integrated Gradients (32 steps) | **1.40** | 2.28 | real-time UI |
| text: B07 exact linear | 0.01 | 0.01 | real-time UI (fast alternative) |
| image: I03 occlusion, research setting (56/28 px, 50 forward passes) | 12.99 | 15.90 | **research only** |
| image: I03 occlusion, UI setting (112/56 px, 10 passes, 4 × 4 map) | **3.21** | 4.55 | real-time UI |
| multimodal: comparison + ablation | 1.31 | 1.36 | real-time UI |
| multimodal: + IG and occlusion through M04 | 15.52 | 16.07 | **research only** |
| full UI object (`explain_meme`, caption given, UI image map) | **5.93** | 6.98 | real-time UI (+ about 5 s OCR when the caption comes from Task 14 OCR) |

**RESEARCH vs REAL-TIME (15.22):**
- **Research only:** the 50-pass occlusion map and attribution through M04.
- **The UI uses:**
  - T05 IG (or B07 contributions as a fast fallback);
  - the coarse 9-copy occlusion map;
  - the unimodal comparison + ablation.

  That is about 6 s for a meme with a caption, plus about 5 s when OCR runs.
- **Peak memory was not measured.** The batch computation for TEST took about 1 h (occlusion about 8 s per image when batched).

**Consistency checks [MEASURED]** (`consistency_checks.csv`):
- the XAI code path reproduces the Task 13 TEST predictions of T05, I03, M04 and B07 on **758/758** memes each;
- the M04 ablation's full label matches on 758/758;
- the B07 decomposition error is 0, with matching signs on 758/758;
- the T05 IG forward sign matches the label on 758/758 reference captions and 184/184 OCR texts.

**IG completeness (32 steps, 758 captions):**
- the median gap is 0.002 log-odds (0.4% relative); the 95th percentile is 18% relative.
- For the 20 worst cases, 128 steps reduce the gap to a mean of 0.004, and the word ranking stays the same (Spearman 0.99 with the 32-step result; `ig_convergence_check.csv`).
- 32 steps are therefore adequate for ranking words; magnitudes in the tail are approximate.

**Text faithfulness** (deletion test, 183 uniform-random memes with ≥ 4 words; `explanation_faithfulness_checks.csv`):
- deleting the top-1 IG word lowers the predicted-class log-odds by median **0.066**, vs 0.008 for a random word;
- for the top-3 words, 0.275 vs 0.081;
- the top-attributed deletion beats random in 77–78% of memes (p < 1e-7).

**Model-randomisation checks for T05 IG** (`text_randomisation_checks.csv`): 50 memes, Spearman ρ with the real attributions.

| Re-initialised | Median ρ | Median \|ρ\| | Share \|ρ\| < 0.5 |
|---|---|---|---|
| classifier head only (planned check) | 0.92 | 0.92 | 2% |
| head + last encoder layer + pooler (added after the plan) | −0.43 | 0.43 | 56% |
| all weights: random BERT (added after the plan) | −0.11 | 0.32 | 76% |

**Why the head-only check is not discriminative here:** T05's pooled TEST features are essentially one-dimensional. The first principal component holds **98.0%** of their variance; the first three, 98.9%. Any linear head therefore reads the same direction.

Randomising the encoder destroys the attribution pattern, so **T05's explanations depend on its trained weights**. Combined with the deletion test, the IG explanations are faithful to the model.

**[INTERPRETATION]** The one-dimensional representation also means M04 receives what amounts to a single "text score" from T05. This fits T05's compressed output (§2) and its near-binary, cue-driven behaviour (§8).

**Image** (§9): five random linear classifiers on the same occluded embeddings give maps with median ρ = 0.01 to I03's map, so the I03 maps depend on the trained classifier.

**Cache (15.23):**
- research explanations are cached outside the repository: `<parent of $NLP_CLD_MODEL_DIR>/cache/xai`;
- IG results are keyed by text SHA-256 + model artifact hash + method + steps + config hash;
- occluded SigLIP embeddings are keyed by image SHA-256 + encoder revision + window/stride/fill.
- **User inputs are never cached** (`cache.cache_user_uploads: false`). `explain_*()` and the UI functions keep everything in memory and return `"stored": False`.

## 19. Limitations
- **Attribution ≠ causation.**
  - IG depends on its baseline ([PAD] embeddings); occlusion on its grey fill and window size.
  - Both treat words and regions one at a time, and interactions are not separated.
- **T05's output is compressed** (log-odds ±0.74), so its attributions are small in absolute terms.
- **Randomisation checks:** the planned head-only check is not discriminative for T05, whose pooled representation is about 98% one-dimensional. Deeper checks, added after the plan, show that the explanations depend on the trained encoder (§18).
- **Faces were not detected automatically** (OpenCV 5 has no Haar detector). The gender-presence evidence is a single reviewer's look at 30 heatmaps plus FP-rate statistics, and the image FP statistics are confounded with captions about women.
- **Manual error categories:** one reviewer (the AI assistant), 60 of 149 M04 errors, no inter-annotator agreement, judged from image + caption only.
- **Lexicons are hand-written** (declared in the plan) and incomplete. English contractions are split by the tokenizer, so "don't" is not matched as negation.
- **Probes are 14 invented sentences:** illustrative, not a sample.
- **TEST was reused for explanation (post-hoc).** No model, threshold, preprocessing rule or split was changed. The moderation rates are descriptive.
- **Small subsets** (21 empty captions, 5 DISAGREE memes, 30 heatmaps) are not reliable estimates.
- **Plan deviations:**
  - automatic face detection dropped (unavailable);
  - cascading randomisation checks, an IG convergence check and the T05 correct-vs-incorrect table added after the planned run (motivated by the head-only result).
- **Config location:** `config.yaml` is checksum-frozen, so explainability settings live in `config/explainability.yaml` (as `config/ocr.yaml` in Task 14).

## 20. Responsible interpretation
- **Explanations are model attribution, not proof of causal reasoning.**
  - A highlighted word or region moved *this model's* score under *this method*.
  - It does not explain why content is harmful, what its author intended, or how a person would judge it.
- **The models classify content, not people.** The UI must never describe a user or author as misogynistic.
- **The main research finding is a warning:**
  - the text model reacts to the *presence of women* in words (female-reference terms) and largely ignores negation;
  - the image model appears to react to pictured women.

  These are fairness risks: benign content about or showing women is over-flagged. They are reported, **not fixed**: fixing them would require new training, outside Task 15.
- **The future UI** (`explain_meme()` / `explain_text_input()`; example `explanation_object_example.json`) should show:

  ```
  WHY THIS WAS FLAGGED   (model attribution; may not reflect human reasoning)
  Text signals:          top words with direction (towards / against "misogynistic")
  Image regions:         heat-map cells (red = towards, blue = against)
  Model comparison:      text-only / image-only / multimodal labels (uncalibrated scores) + AGREE / PARTIAL / DISAGREE
  Recommendation:        ALLOW / WARNING / HUMAN REVIEW with its rule and reason
  Limitations:           the method limitations listed in the object
  ```
