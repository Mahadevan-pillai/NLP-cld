# Image-Only Model Report (Task 10: frozen SigLIP + linear classifiers)

Date: 2026-10-13 · Image models v1.0

Code:
- `src/preprocessing/image.py`: `load_image()` (validation, decoding, EXIF orientation, RGB)
- `src/models/image_encoder.py`: frozen encoder, embedding cache, classifiers
- `src/inference/image.py`: `predict_image(image, model_version)`
- `scripts/train_image_models.py`: `verify`, `benchmark`, `embed`, `train`
- `scripts/report_image_models.py`: figures, comparisons, error analysis, diagnostics, registry
- `tests/test_image_models.py`

Results:
- `results/tables/image_models/`
- `results/figures/image_models/`
- `results/metrics/image_models/` (per-experiment JSON and validation predictions; no text)
- `results/tables/model_comparison_unimodal.csv`
- `results/metrics/image_model_registry.json`

> **TEST SET NOT EVALUATED IN TASK 10.** No test image was opened or
> embedded. All numbers below are **validation** results (754 samples)
> for development comparison, **not** final performance estimates.
>
> **NO TEXT FEATURES.** The models receive only the meme image. No OCR
> was run and no caption column is model input. Captions are used only
> to *describe* errors (e.g. "caption empty") in the error analysis.

Tags:
- **[MEASURED]**: produced by the code.
- **[INTERPRETATION]**: our reading of the measurements.
- **[HYPOTHESIS]**: a possible explanation, not tested.

---

## 1. Objective
Build the **image-only** baseline for the planned comparison

> text only (Tasks 8–9) vs **image only (Task 10)** vs multimodal (Task 11),

on the **same canonical samples and split**. Development metric:
**macro-F1** on validation.

## 2. Classification task
- **Task:** `misogyny_detection`, the binary MIMIC `Misogyny` label
  (1 = misogynistic, 0 = non-misogynistic), the only enabled task in
  `config/categories.yaml` (Task 8.5).
- **Sub-labels not trained.** The sub-labels (Objectification,
  Prejudice, Humiliation) remain disabled pending a project decision, so
  no image sub-label model was trained.
- **No new categories.** None were created (no SAFE / HATE / OBSCENE,
  etc.).
- **Model head:** two mutually exclusive classes, so a binary linear
  classifier is the correct head.

**Data verification (10.1)** [MEASURED]. `train_image_models.py verify`
writes `image_data_checks.csv` and `image_data_summary.json`. All 9
checks pass:

| Check | Result |
|---|---|
| Canonical split counts (verified against the manifest SHA-256) | train 3,532 / validation 754 / test 758 ✅ |
| Every `sample_id` assigned once; train ∩ validation = ∅ | ✅ |
| Labels valid (0/1) | ✅ |
| All 4,286 train + validation images exist | ✅ |
| All decode with `load_image()` | ✅ 4,286 / 4,286 |
| Images unchanged since Task 4 (file SHA-256 = recorded `image_sha256`) | ✅ 4,286 / 4,286 |
| No identical image (SHA-256) in both train and validation | ✅ 0 |
| No Task 5 near-duplicate image pair crosses splits | ✅ 0 of 55 |

- All images are RGB JPEGs, none needs EXIF rotation, and the smallest
  side is 107 px.
- Test sample IDs were **counted** from the split file. Test images were
  **not opened**.

## 3. Hardware
**[MEASURED]** (detected at the start of the task):

| Item | Value |
|---|---|
| CPU | Intel Core i5-13500H, 12 PyTorch threads |
| RAM | 15.7 GB total (about 6.4 GB free at the start) |
| GPU | Intel Iris Xe (integrated, not CUDA); `torch.cuda.is_available() = False` |
| PyTorch | 2.14.1+cpu (CPU-only build) |
| Device used | CPU |

**Consequence:** following the task rules, there is no multi-hour
vision fine-tuning on CPU. Fine-tuning is **DEFERRED TO GPU** (§9).

## 4. Vision-model candidates
Sizes and licences come from the Hugging Face API (file metadata) and
model cards, checked on 2026-10-13.

| Candidate | Weights file (size) | Licence / use terms | Notes |
|---|---|---|---|
| **`google/siglip-base-patch16-224`** (SigLIP) | `model.safetensors`, 812.7 MB (image + text towers) | **Apache-2.0**; no deployment restriction in the model card | ViT-B/16 image tower (92.9 M parameters), 224×224; trained on English WebLI image–text pairs |
| `openai/clip-vit-base-patch32` (CLIP B/32) | `pytorch_model.bin`, 605.2 MB | No licence tag on Hugging Face. The model card says **"Any deployed use case of the model … is currently out of scope"** | Fastest on CPU; pickle-format file only |
| `openai/clip-vit-base-patch16` (CLIP B/16) | `pytorch_model.bin`, 598.6 MB | Same model-card restriction as B/32 | About 4× the compute of B/32 |
| `google/vit-base-patch16-224-in21k` (ViT) | `model.safetensors`, 345.6 MB | Apache-2.0 | ImageNet-21k classification pretraining only (no image–text alignment) |
| `microsoft/resnet-50` (ResNet) | `model.safetensors`, 102.5 MB | Apache-2.0 | ImageNet-1k CNN; smallest; no image–text alignment |

## 5. Selected encoder and rationale
**SigLIP base patch16-224**, pinned at commit
`7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`. The developer chose it from
the three image–text candidates.

- **Weights:** `model.safetensors` SHA-256
  `2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8`.
  This equals the content hash under which Hugging Face stores the file,
  so the download is intact.
- **Only the vision tower is used** (`SiglipVisionModel`). The text-tower
  weights are reported as "unexpected" by the loader and ignored. No
  vision weight is missing.
- **Embedding:** `pooler_output`, SigLIP's attention-pooling head,
  **768-d**. This is the same vector SigLIP uses as its image embedding.

Why SigLIP:
1. **Image–text pretraining.** Like CLIP, SigLIP learned images jointly
   with text. Its embeddings are therefore organised around the kind of
   semantic content a caption describes, which suits memes better than
   ImageNet-only encoders (ViT-in21k, ResNet). It can also take part in
   the Task 11 fusion.
2. **Licence and deployment.** Apache-2.0, without the "any deployed use
   is out of scope" clause of the OpenAI CLIP model card. This matters
   because the project plans a deployed application.
3. **Safe, pinned file format.** It ships `safetensors` (no pickle) and
   is pinned to one commit.
4. **Manageable size.** It is a base model, not the largest option. The
   CPU benchmark measured 156 ms per image (§15).

## 6. Image preprocessing
Two stages, both deterministic, with nothing learned:

1. **`load_image()` (project code, loader v1.0, shared with inference):**
   - accepts a path, raw bytes or a PIL image;
   - fully decodes it (corrupt or truncated files raise `ImageLoadError`);
   - applies the EXIF orientation;
   - converts to RGB;
   - rejects images with a side < 16 px.
2. **SigLIP's own processor** (`SiglipImageProcessorPil` from the pinned
   revision, `backend="pil"` pinned):
   - resize to **224×224, bicubic, without crop**;
   - rescale ×1/255;
   - normalise with mean 0.5 / std 0.5 (values in [−1, 1]).

Notes:
- **The complete meme is used.** It is stretched rather than cropped, so
  captions and borders near the edges are kept. The aspect ratio is
  **not** preserved (a tall 720×900 meme is squeezed to a square).
- **No OCR, no text removal, no file changes.** Original files are never
  modified (a test checks this), and no resized copy is saved.
- **Backend pinned.** transformers 5 would switch to a torchvision-based
  resize if torchvision were installed. That can shift pixel values
  slightly, so the PIL backend is pinned in code and config.
- **Limitation: the encoder can see rendered text.** Meme captions are
  drawn **inside** the image. A vision encoder trained on web image–text
  pairs may respond to large rendered words (mostly Latin-script
  English), although **no extracted text is given to it**. So
  "image-only" means *no OCR / caption input*, not "no textual
  information at all" (§14).

## 7. Frozen embedding methodology
- **Encoder frozen:** `eval()` mode under `torch.no_grad()`. A test
  checks that the weights are unchanged after embedding.
- **Batches:** 32 images. Images are loaded lazily per batch, so memory
  stays bounded.
- **Splits:** train and validation are embedded **separately**. Test is
  not embedded.
- **Cache** (outside the repository):
  `<dir of $NLP_CLD_MODEL_DIR>/../cache/image_embeddings/siglip-base-patch16-224_7fd15f06_img1.0/{train,validation}.{npy,json}`.
  - **Metadata:** split, `sample_id` list in order, encoder checkpoint and
    revision, image-preprocessing version, embedding dimension and a
    fingerprint of (sample ID, image SHA-256) pairs.
  - **Stale-cache detection:** the cache is reused only if all of these
    match. A different order, a changed image, another encoder revision
    or another split forces recomputation (tested).
  - **Alignment:** readers check the stored sample order against the
    requested order before use (`load_cached_embeddings`).

## 8. Classifiers
Both are fitted on **train embeddings only**, with fixed defaults that
were **not tuned**, the same family as Tasks 8–9.

| ID | Pipeline | Settings |
|---|---|---|
| I01 | `StandardScaler` + `LogisticRegression` | C = 1.0, lbfgs, max_iter 2000, class_weight None, seed 42 |
| I02 | `StandardScaler` + `LinearSVC` | C = 1.0, squared hinge, dual auto, max_iter 10000, class_weight None, seed 42 |

- **No MLP (I03).** The task makes it optional "only if justified". With
  3,532 training samples and untuned linear models as the reference, an
  MLP adds tuning choices without a clear need, so it was not trained.
- **Class imbalance:** none to correct (50.3 / 49.7%, Task 5). No class
  weights and no oversampling; no images were synthesised.
- **Scores:**
  - I01 gives an **uncalibrated** LR probability.
  - I02 gives a **signed SVM margin, which is not a probability**.

**D01: shortcut diagnostic (not an image model).** Task 5 found that
exact image resolutions are class-correlated. D01 measures how much the
**resolution alone** predicts:
- input: a one-hot of the exact `width×height`, read from the file
  header (resolutions seen < 5 times in train are grouped together);
- classifier: logistic regression, fitted on train;
- it uses no pixels, no file name and no text.

D01 is reported for interpretation only and is not a candidate model.

## 9. Optional fine-tuning status
**DEFERRED TO GPU / NOT REQUIRED FOR BASELINE.**

- No CUDA GPU is available (§3).
- The task forbids launching multi-hour vision fine-tuning on CPU, and
  frozen features are the required baseline.
- Fine-tuning SigLIP, if wanted later, should run on a GPU (e.g. Colab)
  as one controlled experiment with its own ID. It should follow the
  same train/validation protocol: checkpoint chosen on validation, test
  untouched.

## 10. Validation results
**[MEASURED]** Validation, n = 754 (379 non-misogynistic / 375
misogynistic). The 95% CIs are bootstrap intervals for macro-F1 (1,000
resamples). They describe uncertainty and are not significance tests.
Sources: `image_model_results.csv` and `results/metrics/image_models/*.json`.

| ID | Model | Macro-F1 (95% CI) | Accuracy | Precision* | Recall* | Weighted-F1 | Macro-F1 with caption only | Macro-F1 without watermark |
|---|---|---|---|---|---|---|---|---|
| **I01** | **frozen SigLIP + LR** | **0.7586 (0.731–0.789)** | 0.7586 | 0.7533 | 0.7653 | 0.7586 | 0.7581 | 0.7567 |
| I02 | frozen SigLIP + Linear SVM | 0.7507 (0.720–0.781) | 0.7507 | 0.7493 | 0.7493 | 0.7507 | 0.7498 | 0.7486 |
| *D01* | *diagnostic: exact resolution only* | *0.5153 (0.479–0.550)* | *0.5703* | *0.7040* | *0.2347* | — | — | — |

\* Precision and recall of the **misogynistic** class. "Caption only"
and "without watermark" are the same reporting subsets as in Tasks 8–9.
They are defined from the captions for reporting only and are not model
input.

**Per-class [MEASURED]:**

| ID | Precision / recall / F1, non-misogynistic | Precision / recall / F1, misogynistic | TN | FP | FN | TP |
|---|---|---|---|---|---|---|
| I01 | 0.764 / 0.752 / 0.758 | 0.753 / 0.765 / 0.759 | 285 | 94 | 88 | 287 |
| I02 | 0.752 / 0.752 / 0.752 | 0.749 / 0.749 / 0.749 | 285 | 94 | 94 | 281 |

**Checks [MEASURED]:**
- Both classifiers converged.
- Save → reload reproduces the validation predictions exactly.
- **End-to-end:** `predict_image()` on all 754 validation image files
  reproduces the stored predictions of I01 and I02 (754/754 each; scores
  within 7e-6, the CSV rounding).
- **Preprocessing backend:** the pinned PIL backend gives embeddings
  **identical** (max difference 0.0) to the cached ones.

**Scores [MEASURED]:**
- I01's LR probabilities are **extremely polarised**. Half of all
  validation scores are < 0.023 or > 0.96, yet the accuracy of those
  confident predictions is only about 82–84%.
- 50.5% of I01's errors have a score < 0.1 or > 0.9.
- These scores are uncalibrated and **overconfident**. They must not be
  shown as confidence.

**[INTERPRETATION]**
- **I01 is the best image-only model.** LR and SVM differ by 0.8 points,
  well inside the CIs.
- **The classes are balanced in errors:** 94 FP and 88 FN.
- [HYPOTHESIS] The overconfidence comes from an untuned C = 1.0 on 768
  standardised dimensions. Calibration or regularisation would be
  Task 12 work, on validation only.

## 11. Confusion matrices
`results/figures/image_models/I01_confusion_matrix.png` and
`I02_confusion_matrix.png`:
- rows = true label, columns = predicted label, with class names;
- counts plus row percentages;
- two-line titles, checked visually, with nothing clipped.

**[MEASURED]**
- I01 flags 24.8% of non-misogynistic memes as misogynistic and misses
  23.5% of misogynistic memes.
- Unlike the text models, which lean towards one class, the image
  models' errors are nearly symmetric.

**Embedding inspection** (`validation_embeddings_pca.png`, PCA fitted on
train, validation projected):
- The first two components explain only 5.4% and 4.2% of the training
  variance.
- Misogynistic memes concentrate towards one region (low PC2), but the
  classes overlap heavily in 2-D.
- [INTERPRETATION] This is descriptive only. The linear classifier works
  in all 768 dimensions.

## 12. Comparison with text-only models
`results/tables/model_comparison_unimodal.csv`. Validation only; this is
**descriptive, not a test of multimodal benefit**.

| Model | Modality | Validation macro-F1 (95% CI) | Accuracy | Training cost | Inference | Size |
|---|---|---|---|---|---|---|
| B03 TF-IDF + LR | text (OCR caption) | 0.6671 (0.633–0.701) | 0.6671 | 0.25 s | about 0.04 ms per text | 0.18 MB |
| T05 fine-tuned MuRIL | text (OCR caption) | 0.6869 (0.654–0.719) | 0.6870 | 33 min (CPU) | about 32 ms per text | 957 MB |
| **I01 frozen SigLIP + LR** | **image** | **0.7586 (0.731–0.789)** | 0.7586 | 11.9 min embedding + 0.3 s fit | about 210 ms per image end-to-end (156 ms embedding) | 0.03 MB + 813 MB encoder |

**[MEASURED]**
- I01 is **+0.072 macro-F1 above T05** and **+0.092 above B03**. Its CI
  (0.731–0.789) does not overlap T05's (0.654–0.719).
- **Image and text get different samples right**
  (`image_text_complementarity.csv`, I01 vs T05):

  | | T05 correct | T05 wrong |
  |---|---|---|
  | **I01 correct** | 413 | 159 |
  | **I01 wrong** | 105 | 77 |

  - Prediction agreement is 65.0%; the error correlation is weak (φ =
    0.134).
  - At least one of the two is right for 677 of 754 samples (89.8%). This
    is an **upper bound** that no real fusion is guaranteed to reach.
- **Sub-label recall** of the binary prediction (descriptive;
  `image_models/subcategory_recall.csv`):

  | Group | n | I01 (image) | T05 (text) |
  |---|---|---|---|
  | Objectification | 214 | **0.855** | 0.687 |
  | Prejudice | 152 | 0.638 | **0.704** |
  | Humiliation | 55 | 0.727 | **0.836** |
  | any misogynistic | 375 | **0.765** | 0.712 |

**[INTERPRETATION]**
- On MIMIC, **the image carries more label signal than the OCR caption**
  for this binary task, at the validation stage.
- The two modalities are **complementary**:
  - images catch **objectification** (visual sexualisation) far better;
  - captions catch **prejudice and humiliation** (meaning carried by
    words) better.
- This supports the Task 9 hypothesis, and it is the main motivation
  for fusion in Task 11. **Multimodal superiority is not claimed**; it
  must be measured.
- [HYPOTHESIS] Part of the image advantage may be the shortcut in §13
  (pictured women) rather than an understanding of misogyny. The
  comparison is therefore between two *differently biased* models, not
  between a "good" and a "bad" one.

## 13. Error analysis
Outputs:
- `validation_errors.csv`: I01's 182 errors, with IDs, labels, prediction,
  score and computed flags. It contains **no text and no image**.
- `error_category_rates.csv`.
- Labels were **not** changed.

**[MEASURED]** Error rates by flag (I01):

| Category | n | Error rate |
|---|---|---|
| All | 754 | 24.1% |
| Class-correlated "source" resolution (defined on train*) | 53 | 17.0% |
| Low resolution (shorter side < 400 px) | 15 | 26.7% |
| Extreme aspect ratio (< 0.5 or > 2) | 2 | 0 of 2 |
| Caption empty (image with no OCR text) | 22 | 22.7% |
| Caption ≥ 20 words (text-heavy meme) | 79 | **30.4%** |
| Caption contains the `doublemean*` watermark | 6 | 0 of 6 |
| Member of a multi-sample leakage group (template/near-duplicate) | 8 | 2 of 8 |

\* `source_resolutions_train.csv`: resolutions with ≥ 20 train images and
≥ 80% one class. Six qualify: 1440×1800, 1080×1350 and 736×920 (mostly
misogynistic); 1080×1078, 1080×1079 and 945×944 (mostly not). The rule
was fixed before looking at the validation errors.

**Shortcut checks [MEASURED]:**
1. **Resolution** (`resolution_shortcut_check.csv`). Resolution alone
   (D01) reaches macro-F1 0.903 inside the 53 source-resolution
   validation images but 0.489 on the other 701.
   - I01 scores 0.795 inside and **0.753 outside** the source
     resolutions.
   - **I01's performance therefore does not depend on the resolution
     shortcut.** The shortcut is real but affects only about 7% of
     validation.
2. **Hidden near-duplicates** (`nearest_neighbour_diagnostic.csv`).
   - **No** validation image has a train image with cosine similarity
     ≥ 0.95. Only 3 have ≥ 0.90.
   - The nearest train image shares the label for about 74–75% of
     validation images in both main similarity bins (< 0.80 and
     0.80–0.90).
   - So the score is not explained by near-copies of training memes that
     the dHash check missed.
3. **Watermark subset.** Removing the 6 validation memes whose caption
   contains the watermark changes macro-F1 from 0.7586 to 0.7567.

**Manual review [INTERPRETATION].** 48 validation images were viewed
locally: 12 each of TP, TN, FP and FN, randomly sampled. Nothing was
saved or committed. A small sample; counts are approximate:
- **Pictured women drive positive predictions.** In about 18 of 24
  predicted-misogynistic images (TP + FP), a woman is the visual focus:
  actress or glamour photos, body-focused shots. In the TN sample a woman
  is prominent in only about 2 of 12.
- **False positives** are mostly ordinary, non-sexualised photos of
  women (portraits, couples, family photos) in memes labelled
  non-misogynistic.
  - [HYPOTHESIS] The model has learned **"a woman is pictured →
    misogynistic"**. This is a **gender-presence shortcut**, a fairness
    risk: harmless images of women would be flagged.
- **False negatives** are mostly:
  - cartoons and illustrations;
  - tweet or chat **screenshots** and other text-heavy layouts (error
    rate 30.4% for captions of 20+ words);
  - memes whose visual content is harmless while the **caption or the
    caption–image combination** carries the misogyny (e.g. object
    photos paired with women's names, or a neutral portrait with a
    vulgar caption).
- **Template reuse:** a few images with no woman at all (animal photos in
  "papa ki pari"-style memes) receive very high misogynistic scores.
  - [HYPOTHESIS] The model also recognises recurring meme formats. Task 7
    found reused templates carrying both labels (4 cross-class groups),
    where a format prior would mislead.
- **Rendered text and watermarks:** some correctly classified
  misogynistic images show page logos or large captions.
  - [HYPOTHESIS] The encoder may partly respond to page branding or
    large English words rendered in the picture. This was not measured.

## 14. Image-only limitations
- **Image–text interaction is invisible to an image-only model.** Two
  failure types appear in the errors:
  1. *harmless image + harmful caption*: a neutral photo or cartoon
     carries a misogynistic caption, causing false negatives;
  2. *individually harmless image and caption whose combination is
     harmful*: e.g. object images captioned with women's names
     (sexual innuendo), which neither modality shows alone.

  Conversely, an image of a woman with a harmless caption is often
  flagged (false positives). Only a model that sees **both** can resolve
  these cases, which motivates Task 11.
- **"Image-only" does not mean "no textual information".** Captions are
  rendered in the image. SigLIP saw them as pixels (no OCR input); how
  much it uses them is unknown, and it is likely limited to large Latin
  script.
- **Gender-presence shortcut** (§13): a serious fairness concern for any
  deployment of an image-only model.
- **Overconfident scores** (§10).
- **Stretching to 224×224** distorts aspect ratios and makes small text
  unreadable.
- **Frozen encoder, untuned classifier, single split and seed.** The
  validation set is small (CI about ±3 points).
- **The resolution shortcut** exists in the data (D01 0.903 inside the
  source resolutions), though I01 does not depend on it overall.

## 15. Runtime / resource requirements
**[MEASURED]** CPU only (§3).

| Item | Value |
|---|---|
| SigLIP download | 812.7 MB (`model.safetensors`, pinned revision, Hugging Face cache) |
| CPU benchmark (before extraction) | 156 ms per image → estimate 11.1 min for 4,286 images |
| Embedding extraction, actual | train 593 s (3,532 images), validation 120 s (754 images): 713 s total |
| Embedding cache | 13.2 MB (`.npy` + `.json`, outside the repository) |
| Classifier training | I01 0.27 s, I02 2.38 s |
| Classifier inference (precomputed embeddings) | about 5 ms for 754 images |
| `predict_image()` end-to-end | about 210 ms per image (load + preprocess + encoder + classifier) |
| Process memory (observed) | about 1.1 GB during embedding extraction |
| Classifier artifacts | 25,937 B (I01), 25,789 B (I02) |

## 16. Artifacts
- **Classifiers:** `models/image/I01_siglip_logistic_regression_v1.0.joblib`
  and `models/image/I02_siglip_linear_svm_v1.0.joblib`. These are
  git-ignored joblib pickles; load only files this project created.
- **Encoder:** not duplicated. It is loaded from the Hugging Face cache at
  the pinned revision.
- **Registry:** `results/metrics/image_model_registry.json`, used by
  `predict_image()`. Each model entry records:
  - kind, artifact and score semantics;
  - encoder checkpoint and revision;
  - embedding type and dimension (768);
  - image preprocessing (version and settings);
  - label map: 0 = non-misogynistic, 1 = misogynistic.
- **Config:** `config.yaml` `image_model` (encoder, preprocessing,
  classifiers, experiments, fine-tune status).
- **Embedding cache:** outside the repository (§7), git-ignored by design.

## 17. Test-set protection
**TEST SET NOT EVALUATED IN TASK 10.**
- The scripts load data only through `load_development_data()` (train +
  validation).
- The data check counts test **IDs** from the canonical assignment file
  but **opens no test image**. No test embedding exists.
- A unit test checks that no Task 10 source contains
  `load_split("test"` or `allow_test=True`.
- Choices made on validation only:
  - the classifier comparison (I01 vs I02);
  - the "best image model";
  - every diagnostic.
- The canonical split file is unchanged (SHA-256 `47fb02a87b36…`).
- **No text features entered the image models.** A unit test checks that
  the image-model modules never reference caption columns or file names.

## 18. Implications for multimodal modelling (Task 11)
**[INTERPRETATION]**
1. **Fusion has real room.** Errors are weakly correlated (φ = 0.13);
   in 264 validation samples exactly one modality is right.
2. **The modalities cover different misogyny types.** The image is
   stronger on Objectification; text on Prejudice and Humiliation.
3. **Inputs are ready:**
   - cached 768-d SigLIP embeddings (train and validation);
   - MuRIL text representations from Task 9.
   - A simple late or embedding-concatenation fusion can be built on the
     same canonical split without new feature extraction for train and
     validation.
4. **Fusion must be checked against the shortcuts:** the
   gender-presence pattern (§13), the resolution groups, and the empty
   or watermark captions. A fused model could otherwise combine the two
   modalities' shortcuts rather than their understanding.
5. **Score calibration** will matter if the app shows any score. Image
   LR scores are overconfident; MuRIL softmax scores are compressed.

[HYPOTHESIS] A multimodal model will beat I01. **This has not been
tested.**
