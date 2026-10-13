# EDA Report (Task 5: Exploratory Data Analysis)

EDA date: 2026-10-10 · Code: `src/data/eda.py` (statistics),
`src/data/eda_plots.py` (figures), `scripts/run_eda.py` (pipeline) ·
Notebook: `notebooks/02_eda.ipynb` · Numbers:
`results/metrics/eda_summary.json` and `results/tables/eda/*.csv` ·
Figures: `results/figures/eda/*.png`

Each major finding is split into three parts:

- **MEASURED**: computed from the data by the code above.
- **INTERPRETATION**: what it probably means. A hypothesis, not proof.
- **IMPLICATION**: what later tasks should consider.

Nothing here is causal. No statistical significance tests were run;
differences between classes are **descriptive**.

Method notes:

- **EDA-only tokenization.** Text was lowercased and split into
  Latin-letter or Devanagari runs **only to count words**. This is not
  the project's preprocessing (Task 6), and nothing was written back to
  the data.
- **Heuristic language labels.** Script and language indicators come
  from character ranges and small marker-word lists. They are not
  validated language identification.

---

## 1. Dataset used

- **MIMIC**, cleaned metadata from Task 4:
  `data/processed/mimic_cleaned.csv` (SHA-256 `1ff17281346f…`).
- Before analysis, the code checked that the file's checksum and row
  count match the Task 4 fingerprint, that all required columns exist,
  that sample IDs are unique, that labels are valid, and that every image
  exists. All checks passed. [MEASURED]
- The cleaned file and the raw data were byte-identical before and after
  the EDA, and the notebook run left repository files unchanged.
  [MEASURED]

## 2. Sample count

**MEASURED** 5,044 samples. All 5,044 have a readable image; 4,896 have
non-empty text; 148 have no text.

## 3. Target distribution (Figure 01)

**MEASURED**

| Label | Count | % |
|---|---|---|
| 0 non-misogynistic | 2,537 | 50.30 |
| 1 misogynistic | 2,507 | 49.70 |

Majority:minority ratio **1.012**.

**INTERPRETATION** The classes are practically balanced.

**IMPLICATION** Class imbalance is **not** a meaningful problem for the
primary task:
- Accuracy is not distorted by a dominant class, and a majority-class
  baseline scores about 50.3%.
- Precision, recall and F1 are not skewed by base rates.
- Macro-F1 ≈ weighted-F1 should hold, which is a useful sanity check.
- Rebalancing (class weights, resampling) is unlikely to be necessary.
  It can still be tested as a hyperparameter.

Report macro-F1 plus per-class precision and recall anyway: they reveal
whether a model favours one class.

## 4. Secondary-label findings (Figure 12)

**MEASURED** These counts are within the 2,507 misogynistic samples. No
non-misogynistic sample has a sub-label.

| Sub-label | Count | % of misogynistic |
|---|---|---|
| Objectification | 1,488 | 59.35 |
| Prejudice | 916 | 36.54 |
| Humiliation | 392 | 15.64 |

- Labels per misogynistic sample: 1 label = 2,219; 2 = 284; 3 = 3;
  0 = 1.
- Co-occurrence (both labels on one sample): Objectification+Prejudice
  165, Prejudice+Humiliation 78, Objectification+Humiliation 50.

**INTERPRETATION** The sub-labels are mostly mutually exclusive, and
Humiliation is rare.

**IMPLICATION** A multi-label sub-task would be imbalanced (Humiliation
is about 7.8% of all samples) and would need per-label metrics. It stays
an **optional secondary analysis**; the primary task is binary.

## 5. Text statistics (Figures 02–04)

**MEASURED** (non-empty texts; words = whitespace-separated chunks)

| | n | Mean | Std | Median | p25 | p75 | p90 | p95 | Max |
|---|---|---|---|---|---|---|---|---|---|
| Characters, all | 4,896 | 59.6 | 31.6 | 55 | 37.75 | 77 | 101 | 117 | 283 |
| Characters, non-mis. | 2,495 | 62.0 | 30.7 | 59 | 40 | 81 | 104 | 117.3 | 193 |
| Characters, mis. | 2,401 | 57.1 | 32.2 | 52 | 35 | 74 | 99 | 116 | 283 |
| Words, all | 4,896 | 11.6 | 6.3 | 11 | 7 | 15 | 20 | 23 | 64 |
| Words, non-mis. | 2,495 | 12.0 | 6.0 | 11 | 8 | 16 | 20 | 23 | 40 |
| Words, mis. | 2,401 | 11.2 | 6.6 | 10 | 7 | 14 | 19 | 23 | 64 |

Minimum: 1 character / 1 word.

**INTERPRETATION**
- Meme captions are short (median 11 words).
- Misogynistic texts are slightly shorter on average but have a longer
  tail. The distributions overlap heavily (Figure 03).

**IMPLICATION**
- Length alone is a weak signal.
- 95% of texts have 23 or fewer whitespace words. Subword tokenizers
  produce more tokens than words, so a transformer max-length of about
  64–128 tokens would probably cover almost all texts. This must be
  **checked against the chosen tokenizer** in a later task.

## 6. Vocabulary observations (Figures 05–06)

**MEASURED** (EDA-only tokens)

| Measure | Value |
|---|---|
| Tokens | 55,403 |
| Vocabulary | 14,244 types (11,051 Latin, 3,193 Devanagari) |
| Hapax legomena (words appearing once) | 9,241 (65% of types) |

- **Most frequent terms** are function words in Romanized Hindi and
  English: *hai* 970, *me* 832, *to* 458, *ho* 353, *in* 341, *ki* 326.
- **Most frequent Devanagari terms:** है 284, के 182, की 173, तो 158.
- **Most class-associated terms** (weighted log-odds z-score with an
  informative Dirichlet prior; Monroe, Colaresi & Quinn, 2008):
  - misogynistic: *doublemean* (5.2), *doublemeaning* (5.0), *she*
    (4.8), *girls*, *her*, *beti*, *didi*, *ladki*, *ladke*;
  - non-misogynistic: *me* (−5.4), *bhai* (−3.8), *my*, *friend*,
    *sir*, *mom*, *teacher*, *class*, *students*, *exam*.
- **Most frequent bigram:** "doublemean doublemeaning" (62).

**INTERPRETATION**
- Many class-associated words are plausible topical cues: female
  referents in misogynistic memes; family, school and first-person
  topics in the others.
- The **top two** cues, however, are page handles (§12), not content.
- The many words seen only once show heavy spelling variation in
  Romanized Hindi (e.g. *nahi* / *nhi*) plus OCR noise.

**IMPLICATION**
- A bag-of-words model can exploit both genuine cues and artefacts.
- Character n-grams or subword models may handle the spelling variation
  better than word unigrams. That is something to test, not assume.
- Raw frequency alone is uninformative here, because stopwords dominate.

## 7. Hinglish / script observations (Figure 10, HEURISTIC)

**MEASURED** (all 5,044 samples)

| Indicator | % non-mis. | % mis. | % all |
|---|---|---|---|
| Devanagari only | 8.1 | 9.7 | 8.9 |
| Mixed Devanagari + Latin | 22.5 | 17.4 | 19.9 |
| Latin, Hindi markers only | 21.1 | 22.5 | 21.8 |
| Latin, Hindi + English markers | 16.7 | 14.1 | 15.4 |
| Latin, English markers only | 18.7 | 18.6 | 18.6 |
| Latin, no marker words | 11.2 | 13.3 | 12.2 |
| No letters | 0.1 | 0.3 | 0.2 |
| Empty | 1.7 | 4.2 | 2.9 |

Devanagari share of letters (non-empty texts): median 0, mean 0.151,
p90 0.948.

**INTERPRETATION**
- The text mixes Romanized Hindi, Devanagari Hindi and English, often
  within one meme.
- Most texts are Latin script. A minority (about 29%) contains
  Devanagari, usually either only a little or nearly all of the text.
- Class differences in script use are small.

**IMPLICATION**
- Hinglish normalization (RQ1) must handle two scripts. One option to
  test is transliteration between them; this is a design choice, not a
  measured necessity.
- The text encoder must support Devanagari **and** Romanized Hindi.

## 8. Social-media characteristics (Figure 11)

**MEASURED** (share of non-empty texts per class)

| Feature | Non-mis. | Mis. |
|---|---|---|
| URL | 0% | 0% |
| Emoji | 0% | 0% |
| @mention | 9.9% | 8.4% |
| Hashtag | 0.7% | 0.5% |
| Repeated punctuation (`!!`, `??`, `..`) | 10.1% | 8.8% |
| Repeated letters (3+, e.g. *soooo*) | 1.9% | 1.3% |
| Uppercase-heavy (≥70% capitals, ≥5 letters) | 2.9% | **8.1%** |
| Contains a digit | 24.8% | 19.8% |

- The frequent bigram "me ः" (30) pairs a word with a lone Devanagari
  visarga sign.

**INTERPRETATION**
- No URLs and no emojis is consistent with OCR text: OCR does not
  output emojis.
- Uppercase-heavy text is about 2.8× more common in misogynistic memes.
  This may reflect particular meme templates or pages rather than the
  content itself.
- *Hypothesis:* the "ः" is OCR misreading a colon (e.g. "me:" as
  "me ः"), i.e. measurable OCR noise.

**IMPLICATION**
- URL and emoji handling is **not needed** for this dataset (zero
  occurrences). It may still matter for real-world inputs in the app.
- Casing carries a class-correlated signal. Whether to lowercase should
  be decided by ablation, not by default.

## 9. Image statistics (Figures 07–09)

**MEASURED**
- 5,044 JPEG images, all RGB.
- Median width 768 px (non-mis. 945, mis. 720) and median height 899.
- Aspect ratio: median 1.00 (non-mis. 1.00, mis. 0.95), range
  0.44–3.00.
- Widths cluster at 720 and 1080 px. Median file size 76.9 KB, maximum
  626.5 KB.
- **Exact resolutions are class-correlated:**

  | Resolution | Samples | Share misogynistic |
  |---|---|---|
  | 1440×1800 | 57 | **100%** |
  | 720×900 | 61 | 80% |
  | 1080×1079 | 103 | 17% |
  | 1080×1078 | 102 | 16% |
  | 945×944 | 85 | 14% |
  | 720×718 | 57 | 17.5% |

**INTERPRETATION** Images seem to come from different sources (pages or
export tools) whose fixed sizes differ by class. Resolution is therefore
a **source signal**, not a content signal.

**IMPLICATION**
- Image encoders resize inputs to a fixed size, but aspect-ratio
  handling (stretch vs. pad), compression and borders may still carry
  source information.
- The image-only model should be checked for this shortcut, e.g. by
  error analysis per resolution group.

## 10. Text/image availability

**MEASURED**

| Experiment type | Samples |
|---|---|
| Image-only | 5,044 |
| Multimodal | 5,044 |
| Text-only, non-empty text | 4,896 |

The 148 samples without text are 106 misogynistic and 42
non-misogynistic.

**IMPLICATION**
- For a fair comparison, all three model types should be evaluated on
  the **same split**.
- Results should be reported both on all 5,044 samples (the text model
  receives an empty string) and on the 4,896 samples with text.
  Otherwise the text model's "empty means misogynistic" shortcut cannot
  be distinguished from real understanding.

## 11. Outliers (recorded, not removed)

**MEASURED** (`results/tables/eda/outlier_samples.csv`)

| Category | Rule | Samples (non-mis. / mis.) |
|---|---|---|
| Very long text | > 135.9 chars (Q3 + 1.5 IQR) | 107 (49 / 58) |
| Very short text | ≤ 3 chars, non-empty | 20 (4 / 16) |
| Small image | shorter side < 200 px | 9 (2 / 7) |
| Extreme aspect ratio | < 0.5 or > 2.0 | 22 (9 / 13) |
| Large file | > 266,713 bytes (Q3 + 3 IQR) | 55 (12 / 43) |

**INTERPRETATION**
- Long texts are plausible (dense meme text).
- Very short texts are probably OCR fragments.
- Large files are mostly misogynistic, consistent with the
  source-resolution pattern in §9.

**IMPLICATION** Keep all of them. Truncation length (text) and resizing
(images) will handle the extremes. The very short texts may be worth
reviewing as OCR failures.

## 12. Potential leakage concerns

**MEASURED**

1. **Page handles reveal the class.** `@doublemeaning` (59 samples) and
   `@doublemean` (56) occur **only** in misogynistic samples, and they
   are the top two class-associated terms. `@kyahogyafatima` is 85%
   non-misogynistic (20 samples). (Figure 13)
2. **File names encode the label**: the `_M_` / `_NM_` pattern predicts
   it perfectly (Tasks 3–4).
3. **Near-duplicate images** (dHash on an in-memory 9×8 grayscale copy;
   no image changed):
   - 55 candidate pairs at ≤ 4 of 64 bits, 6 of them at distance 0;
   - **48 are not covered by Task 4's duplicate groups**;
   - **4 pairs are cross-class**, with different text: the same or a
     near-same template with different captions.
4. **Identical captions across classes:** 2 Task 4 duplicate-text
   groups (5 samples) mix both labels.
5. **Class-correlated image resolutions** (§9).

**INTERPRETATION** Points 1, 2 and 5 are **shortcut features**. A model
could score well by recognising the page or source instead of misogyny.
Point 3 means visually near-identical memes could fall into both train
and test.

**IMPLICATION**
- The **splitting task** should group near-duplicate image pairs (after
  manual confirmation) **and** the Task 4 `duplicate_group`s into the
  same split.
- Preprocessing should **test masking @handles**, i.e. an ablation that
  reports results with and without them.
- Error analysis should check performance on samples **with vs. without**
  these handles and per resolution group.
- File names must never be features.

## 13. Class-imbalance concerns

**MEASURED** Primary ratio 1.012; Humiliation is 7.8% of all samples.

**IMPLICATION** There is no rebalancing need for the primary task.
Per-label metrics are needed only if the sub-label task is attempted.

## 14. Findings relevant to preprocessing (Task 6)

**DEFINITELY REQUIRED** (by the data or the methodology):
- Keep `original_text` untouched and write preprocessed text to **new**
  columns (reproducibility, RQ1).
- Apply **identical** preprocessing in training and inference.
- Handle **empty text** explicitly (148 samples).
- Support **both** Devanagari and Latin script without corrupting
  either.

**RECOMMENDED FOR TESTING** (let ablations decide):

| Operation | Reason (measured) |
|---|---|
| @handle masking or removal | Class-exclusive watermarks (§12) |
| Lowercasing vs. keeping case | Uppercase-heavy text differs by class (8.1% vs 2.9%) |
| Repeated-character normalization (*soooo* → *soo*) | Present but rare (1.3–1.9%) |
| Repeated-punctuation normalization | About 9–10% of texts |
| Romanized-Hindi spelling normalization (*nhi*/*nahi*) | 65% of word types appear only once |
| Script handling / transliteration | About 29% of texts contain Devanagari |
| OCR-artefact handling (e.g. "ः") | Probable OCR noise (§8) |
| Digit handling | 20–25% of texts contain digits |

**NOT NEEDED for this dataset:** URL handling and emoji handling (zero
occurrences). Both may be added for live app input.

## 15. Findings relevant to model selection (MSE-1 Model Identification)

These are reasons to **evaluate** each family, not predictions of which
will perform best.

| Model family | Why it is reasonable to evaluate (from EDA) | Watch-outs |
|---|---|---|
| **TF-IDF + Logistic Regression** | Short texts, balanced classes; strong, interpretable baseline; class-associated terms exist (§6) | Sparse vocabulary (65% single-use words); learns handle shortcuts |
| **TF-IDF + Linear SVM** | Same as above; robust in high dimensions | No native probabilities (calibration needed for "confidence") |
| **TF-IDF with character n-grams** | Handles spelling variation and OCR noise | Same shortcut risks |
| **Pretrained multilingual / Hinglish-capable text encoder** | Mixed Devanagari, Romanized Hindi and English (§7); short inputs fit small max-lengths | Must handle Romanized Hindi, not only Devanagari; check tokenizer coverage |
| **Pretrained vision or vision-language encoder** (e.g. CLIP, to be justified) | Every sample has a readable image; some memes carry meaning only through the image | Source/resolution shortcut (§9); near-duplicates (§12) |
| **Simple multimodal fusion** (concatenated image + text embeddings → classifier) | Same 5,044 samples for all modalities; identical captions with different labels (§12) can only be separated using the image | Must use the same split and empty-text policy as the unimodal models |

## 16. Limitations

- **Language indicators are heuristic.** They use character ranges plus
  about 45 Romanized-Hindi and about 30 English marker words. They are
  not language identification and miss many words.
- **EDA tokenization is crude.** It splits apostrophes ("don't" becomes
  "don" + "t") and has no Devanagari morphology.
- **Term charts show Latin-script terms only**, because matplotlib cannot
  shape Devanagari. Devanagari terms appear in the tables.
- **Near-duplicates are candidates.** dHash finds them at a chosen
  threshold (≤ 4 bits); they need manual confirmation. Paraphrased
  captions are not detected.
- **No significance tests** were run. Class differences are descriptive
  and may partly reflect chance.
- **Text is OCR output.** All text findings include OCR errors, and the
  dataset contains no clean transcription.
- **The watermark explanation is a hypothesis.** Handles were not
  checked visually against the images.
