# Data Cleaning Report (Task 4: Cleaning & Final Dataset Strategy)

Cleaning date: 2026-10-10 · Code: `src/data/cleaning.py` (version 1.0)
+ `scripts/clean_dataset.py` · Fingerprint:
`results/metrics/mimic_cleaning_fingerprint.json`

All counts below come from the pipeline outputs:
`results/tables/cleaning_summary.csv`, `excluded_samples.csv` and
`duplicate_decisions.csv`.

Tags used:

- **[MEASURED]**: computed from the files.
- **[DOCUMENTED]**: from the dataset card or paper.
- **[DECISION]**: a choice made in this task, with its reason.

---

## 1. Final dataset strategy

**[DECISION]** One dataset, **MIMIC**, is used for every experiment
(E1–E5 and the OCR experiment). No datasets are merged.

The project must compare text-only, image-only and multimodal models.
That comparison is only fair if all three see **the same samples with
the same labels**. MIMIC provides both modalities for every sample,
under one documented misogyny definition.

## 2. Primary dataset

**MIMIC**: *Misogyny Identification in Multimodal Internet Content in
Hindi-English Code-Mixed Language*.

- **Paper [DOCUMENTED]:** Singh, A., Sharma, D., & Singh, V. K. (2024),
  *ACM TALLIP*, <https://doi.org/10.1145/3656169>.
- **Dataset card [DOCUMENTED]:**
  <https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset>.
- **Role:** PRIMARY TRAINING DATASET.

Facts resolved since Task 3 **[DOCUMENTED, dataset card]**:

| Item | Finding |
|---|---|
| Label meaning | `Misogyny`: **1 = Misogynistic, 0 = Non Misogynistic** |
| Licence | **CC-BY-4.0** (card licence tag) |
| Text origin | `ExtractedText` was produced by **OCR** ("EasyORC" on the card, presumably EasyOCR) |
| Official split | **None**; the release is a single 5,054-row set |

Notes on these facts:
- The card's file structure (`MIMIC2024` CSV, `Files/` folder, the same
  6 columns, 5.05k rows) matches the local copy. [MEASURED]
- The card's uploader (`Aakash941`) is not confirmed to be an author, so
  confirm the licence against your own download source.
- The 4,044/1,010 split seen in Task 3 came from a third-party paper, not
  from the release. **[DECISION]** The project will create its own
  duplicate-aware split in a later task.

## 3. Role of other datasets

| Dataset | Role | Reason |
|---|---|---|
| MIMIC | PRIMARY TRAINING DATASET | Matches task, language and modality |
| `Downloads/DATASET 00.csv`, `DATASET 01.csv`, `DATASET 02.xlsx` | **NOT USED** | Not selected by the developer; never inspected |
| General offensive / hate / abuse data | **SEPARATE AUXILIARY TASK** (none present) | Different label meaning; "offensive" ≠ "misogynistic" |
| Text-only misogyny data | Possible **EXTERNAL VALIDATION** (text model only), none present | Different annotation guidelines; no images |

## 4. Raw-data problems discovered

From Task 3 `docs/DATASET_REPORT.md`, re-detected here by the cleaning
code **[MEASURED]**:

| Problem | Count |
|---|---|
| Byte-identical image **and** identical text, same primary label | 6 groups |
| Byte-identical image and text, **conflicting** primary labels | 2 groups (4 rows) |
| Same text on **different** images | 10 groups (21 rows), 2 of them with conflicting labels (5 rows) |
| Identical pair whose sub-labels differ | 1 (`1530_M_pic_1` / `2437_M_pic_2`) |
| Misogynistic sample with no sub-label | 1 (`704_M_pic_0.jpg`) |
| Empty text | 148 |
| File names encode the label (`_M_` / `_NM_`) | all rows |
| Malformed records, missing/duplicate IDs, missing/invalid labels, missing/corrupt images | **0** |

## 5. Cleaning rules

**[DECISION]** These are defined in `POLICY` in `src/data/cleaning.py`
and applied in the order below. Every exclusion is logged; nothing is
removed silently. `data/raw/` is only read: a file fingerprint was
identical before and after every run, and a test checks this.

| Issue | Action | Why | Triggered in MIMIC |
|---|---|---|---|
| Malformed record (wrong field count) | REMOVE | Values cannot be assigned to columns reliably | 0 |
| Missing ID | REMOVE | Cannot link to an image or trace the sample | 0 |
| Duplicate row (exact repeat) | REMOVE later copies | Same sample twice | 0 |
| Same ID, different content | REMOVE all + FLAG | Unknowable which row is correct | 0 |
| Missing primary label | REMOVE | Supervised learning needs the label | 0 |
| Invalid primary label (not `0`/`1`) | REMOVE | Outside documented values | 0 |
| Invalid sub-label | KEEP + FLAG (value set empty) | Primary task unaffected | 0 |
| Missing/empty text | **KEEP** (`has_text=False`) | A meme can have no text; image and multimodal models still use it | 148 |
| Image path differs only by folder/case | REPAIR METADATA + FLAG | Store the real file path | 0 |
| Missing image | REMOVE | The same samples must serve all three model types | 0 |
| Corrupt/unreadable image | REMOVE | Unusable by image models | 0 |
| Identical image + identical text, same label | REMOVE later copies | Same sample twice; leakage risk | **6** |
| Identical image + identical text, conflicting labels | REMOVE all + FLAG | True label unknown | **4** |
| Identical image, different text | KEEP + group + FLAG | Different multimodal input | 0 |
| Same text, different images | **KEEP** + group | Can be a valid, different multimodal sample | 21 rows |
| …and labels differ | KEEP + FLAG | Plausible (the image changes meaning); review | 5 rows |
| Kept copy whose duplicate had other sub-labels | KEEP + FLAG | Exclude it from sub-label analysis | 1 |
| Positive label without sub-label | KEEP + FLAG | Primary task unaffected | 1 |

## 6. Duplicate policy

**[DECISION]** The key distinction is **what the model would see**.

- **Same image bytes and same text:** this is the *same input*. Keeping
  both copies double-counts it, and if the copies fall into different
  splits, the test set leaks into training.
  - Same label: keep the first occurrence (lowest row number, so the
    choice is deterministic).
  - Different labels: the input is identical but annotated both ways, so
    neither label can be trusted and **both copies are removed** and
    flagged for review. We do not relabel data ourselves.
- **Same text on different images:** this is *not* a duplicate for a
  multimodal model, because the image can change the meaning. **All
  copies are kept.** That some carry different labels (2 groups) is
  consistent with the project's multimodal hypothesis (RQ2), so they are
  flagged but not removed.
  - Each group gets a `duplicate_group` ID so that a later split keeps
    the copies together. Otherwise a text-only model would see the exact
    test caption during training.
- Every group and decision is listed in
  `results/tables/duplicate_decisions.csv`: 18 groups, 8 identical-image
  and 10 duplicate-text. [MEASURED]

Not detected: near-duplicate images (re-encoded or resized) and
paraphrased text (§16).

## 7. Missing-data policy

**[DECISION]** The 148 samples with empty text are **kept**, with
`has_text=False`. Removing them would bias the image-only and multimodal
evaluation, because those samples are real memes.

⚠️ **[MEASURED]** Empty text is unevenly distributed: **106 are
misogynistic and 42 non-misogynistic**. A text-only model can partly
learn "empty text means misogynistic".

Two approaches are possible, and the later experiment task must choose
one and document it:
- (a) evaluate all models on all 5,044 samples, with the text model
  receiving an empty string;
- (b) also report results on the 4,896 samples that have text.

Both are possible with the cleaned file. Nothing was imputed.

## 8. Image validation policy

**[DECISION]** Every referenced image is checked with Pillow
(`verify()`, then reopened to read its size) and fingerprinted with
SHA-256. Images are **never copied, resized or modified**. The cleaned
metadata stores a **relative path** (`data/raw/mimic/Files/<name>`) and
the SHA-256, so later tasks can confirm the image is unchanged.

**[MEASURED]** 5,054/5,054 images were readable. After cleaning, the
5,044 retained images are unique by SHA-256 and all readable
(re-checked after writing).

## 9. Label handling

- **[DOCUMENTED]** `Misogyny` 1 = misogynistic, 0 = non-misogynistic.
- **[DECISION]** `target_label` is an explicit identity mapping
  (`"0"→0`, `"1"→1`, `config.yaml` `dataset.label_mapping`). The raw
  value is kept alongside it as `original_label`.
- No other label concept is mapped onto misogyny.
- The sub-labels (Objectification, Prejudice, Humiliation) keep their
  original values as integers. They support an optional secondary
  analysis only.

**File-name leakage [DECISION]:**
- `sample_id` is a hash of the raw file name (`mimic_<12 hex>`). It is
  stable (independent of row order) and does **not** reveal `_M_`/`_NM_`.
- `image_reference` and `image_path` still contain the label-revealing
  name, because they are needed to load the image. They must **never**
  be used as model features.

## 10. Samples before cleaning

**[MEASURED]** 5,054 records: `Misogyny` 0 = 2,539 (50.24%),
1 = 2,515 (49.76%).

## 11. Samples after cleaning

**[MEASURED]** **5,044 retained**: 4,896 with text, 148 without.

| Experiment type | Usable samples |
|---|---|
| Image-only | 5,044 |
| Multimodal | 5,044 |
| Text-only | 5,044 (4,896 with non-empty text) |

All three use the same samples and labels.

- 23 retained samples carry review flags.
- 21 retained samples belong to 10 duplicate-text groups.

## 12. Exclusion reasons

**[MEASURED]** 10 samples were excluded. The full list, without any text,
is in `results/tables/excluded_samples.csv`.

| Reason | Count | Samples |
|---|---|---|
| `duplicate_image_and_text` (later copy of an identical meme) | 6 | `44_M_pic_2`, `2423_M_pic_1`, `1530_M_pic_1`, `2422_M_pic_1`, `1561_M_pic_3`, `2550_M_pic_1` |
| `conflicting_labels_identical_sample` | 4 | `2402_NM_pic` + `282_M_pic_12`; `1827_NM_pic` + `938_M_pic_2` |

## 13. Class distribution before/after

**[MEASURED]**

| `Misogyny` | Before | After | Removed |
|---|---|---|---|
| 0 (non-misogynistic) | 2,539 (50.24%) | 2,537 (50.30%) | 2 |
| 1 (misogynistic) | 2,515 (49.76%) | 2,507 (49.70%) | 8 |
| Total | 5,054 | 5,044 | 10 |

| Sub-label = 1 | Before | After |
|---|---|---|
| Objectification | 1,493 | 1,488 |
| Prejudice | 919 | 916 |
| Humiliation | 393 | 392 |

The class balance is essentially unchanged.

## 14. Cleaned dataset schema

`data/processed/mimic_cleaned.csv` (5,044 rows, UTF-8). It is
**git-ignored** because it contains dataset text.

| Column | Meaning |
|---|---|
| `sample_id` | Stable ID, `mimic_` + first 12 hex of SHA-256(raw file name) |
| `source_row` | 1-based row in the raw CSV (provenance) |
| `image_reference` | Raw `FileName` value (**not a feature**: encodes the label) |
| `image_path` | Relative path to the raw image (**not a feature**) |
| `image_sha256` | SHA-256 of the image file |
| `original_text` | Raw `ExtractedText`, byte-for-byte (OCR output; no preprocessing) |
| `has_text` | `True` if `original_text` has non-whitespace characters |
| `original_label` | Raw `Misogyny` value (`"0"`/`"1"`) |
| `target_label` | Internal label (0/1) via the documented mapping |
| `Objectification`, `Prejudice`, `Humiliation` | Original sub-labels (0/1) |
| `duplicate_group` | Shared ID for same-text groups (keep together when splitting); empty otherwise |
| `review_flags` | `;`-separated flags (e.g. `duplicate_text`, `text_label_conflict`) |

## 15. Reproducibility information

From `results/metrics/mimic_cleaning_fingerprint.json` **[MEASURED]**:

| Field | Value |
|---|---|
| Input file | `data/raw/mimic/MIMIC2024.csv`, SHA-256 `98dd8843ef81…` |
| Cleaning code | version 1.0, `cleaning.py` SHA-256 `068f1dd8bb1b…` |
| Output file | `mimic_cleaned.csv`, SHA-256 `1ff17281346f…`, 5,044 rows |
| Retained-images digest | `b824fa0eaebf…` |
| Random seed | **not used** (no random operations) |

The pipeline is deterministic: two consecutive runs produced
byte-identical output files and tables.

- Re-run with `python scripts/clean_dataset.py`.
- If the dataset is absent it exits with an error code and writes
  nothing.

## 16. Limitations

- **Duplicate detection:** only exact images (SHA-256) and text equal up
  to case/whitespace are caught. Re-encoded or resized images and
  paraphrased captions are not.
- **Excluded label conflicts:** the 2 conflicting identical memes were
  excluded, not resolved. The 3 annotators' individual votes are not in
  the release, so the true label cannot be recovered.
- **Keep-first rule:** "keep first occurrence" is a deterministic but
  arbitrary choice of representative. For the one pair whose sub-labels
  differ, the retained sub-labels are flagged as unreliable.
- **OCR text:** `original_text` is OCR output, so OCR errors are already
  in the "reference" text. ⚠️ This changes RQ3: no human transcription
  exists to compare OCR against. The OCR experiment must be reframed
  before it is designed, e.g. dataset OCR (EasyOCR) vs. the project's
  own OCR pipeline, or a small manually transcribed subset.
- **Label noise:** remaining label noise beyond exact duplicates cannot
  be measured from this data.
