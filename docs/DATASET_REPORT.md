# Dataset Report (Task 3: Inventory, Verification & Compatibility)

Inspection date: 2026-10-10 · Inspection code: `scripts/inspect_datasets.py`
(+ `src/data/inspection.py`, `src/data/datasets.py`) · Full measurements:
`results/metrics/dataset_inspection_mimic.json` · Tables:
`results/tables/dataset_inventory.csv`, `results/tables/dataset_compatibility.csv`

Every statement is tagged:

- **[MEASURED]**: computed from the local files by the inspection code.
- **[DOCUMENTED]**: stated in a published source (cited).
- **[TO BE VERIFIED]**: not confirmed; must be checked before relying on it.
- **[ANALYSIS]**: our interpretation/recommendation.

> **Update (Task 4, 2026-10-10).** The MIMIC dataset card
> (<https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset>)
> resolves several TO-BE-VERIFIED items below **[DOCUMENTED]**:
>
> - `Misogyny` **1 = Misogynistic, 0 = Non Misogynistic**;
> - licence **CC-BY-4.0**;
> - `ExtractedText` was produced by **OCR** ("EasyORC", presumably
>   EasyOCR);
> - the release has **no official split** (the 4,044/1,010 split is a
>   third-party choice).
>
> The measurements in this report are unchanged. Cleaning decisions are
> in `docs/DATA_CLEANING_REPORT.md`.

Raw data was only read, never modified: a fingerprint of every file
(path, size, modification time) under `data/raw/` was identical before
and after running the inspection. [MEASURED]

---

## 1. Datasets discovered

| Dataset | Location | Status |
|---|---|---|
| MIMIC | `data/raw/mimic/` | Fully inspected |

Only one dataset is present in `data/raw/`. Other CSV/XLSX files exist on
the developer's machine (`Downloads/DATASET 00.csv`, `DATASET 01.csv`,
`DATASET 02.xlsx`), but the developer did not select them for this
project, so they were **not opened or inspected**. [MEASURED: presence
only]

## 2. Purpose

**[DOCUMENTED]** MIMIC, *"MIMIC: Misogyny Identification in Multimodal
Internet Content in Hindi-English Code-Mixed Language"*, Singh, A.,
Sharma, D., & Singh, V. K. (2024), *ACM Transactions on Asian and
Low-Resource Language Information Processing*,
<https://doi.org/10.1145/3656169>. It defines two sub-tasks:

1. binary misogyny identification for memes;
2. multi-label categorisation of misogynistic memes into
   **Objectification, Prejudice, Humiliation**.

Task scope is **misogyny specifically**. It is not general
offensiveness, hate, abuse, sarcasm or sentiment, and must not be treated
as equivalent to them. [ANALYSIS]

## 3. Languages

- **[DOCUMENTED]** Hindi-English code-mixed.
- **[MEASURED, heuristic]** Script and marker-word indicator over the
  4,906 non-empty texts. Marker words are a small list of common
  Romanized-Hindi function words (*hai, nahi, kya, …*) and English
  stopwords (*the, is, you, …*):

| Indicator | Samples | % of non-empty |
|---|---|---|
| Latin script, Hindi markers only | 1,101 | 22.4 |
| Mixed Devanagari + Latin script | 1,008 | 20.5 |
| Latin script, English markers only | 942 | 19.2 |
| Latin script, both Hindi and English markers | 778 | 15.9 |
| Latin script, no marker words | 616 | 12.6 |
| Devanagari only | 452 | 9.2 |
| No letters | 9 | 0.2 |

This is **not** validated language identification. It shows that the
text contains Romanized Hindi, Devanagari Hindi, English and mixtures:
about 30% of texts contain Devanagari script. This is relevant to the
Hinglish normalization design (RQ1). [ANALYSIS]

## 4. Modalities

Image + text. Each record has one meme image (`FileName`) and one text
field (`ExtractedText`). [MEASURED]

## 5. File structure

```
data/raw/mimic/                  (438,172,926 bytes total)
├── MIMIC2024.csv                annotation file, UTF-8 (with BOM), comma-separated
└── Files/                       5,054 .jpg images (flat folder)
```

- No README, licence or split files are present. [MEASURED]
- Annotation file SHA-256 prefix `98dd8843ef81`, identical to the source
  copy in `Downloads/`. [MEASURED]

## 6. Sample counts

**[MEASURED]**
- 5,054 records.
- All have the same 6 fields; no malformed rows.
- 5,054 unique image references and 5,054 image files.

**[DOCUMENTED]** The M3Hop-CoT paper (arXiv:2410.09220, Table 1) reports
MIMIC as **4,044 train / 1,010 test**. That totals 5,054, which matches
the local count. The local file contains **no split column**, so that
split cannot be reproduced from it. [MEASURED]

## 7. Labels (schema)

**[MEASURED]**

| Column | Role | Type | Missing | Unique |
|---|---|---|---|---|
| `FileName` | ID + image reference | text | 0 | 5,054 |
| `ExtractedText` | text | text | 148 (empty) | 4,888 |
| `Misogyny` | primary label | binary `0`/`1` | 0 | 2 |
| `Objectification` | secondary label | binary `0`/`1` | 0 | 2 |
| `Prejudice` | secondary label | binary `0`/`1` | 0 | 2 |
| `Humiliation` | secondary label | binary `0`/`1` | 0 | 2 |

**Label meaning [TO BE VERIFIED].** The column names and the paper's task
description strongly suggest that `Misogyny = 1` means "misogynistic".
The official documentation was not accessible (the ACM page blocks
automated access), so this mapping is **not yet confirmed**.
`config.yaml` keeps `positive_label: null` until it is.

## 8. Class distribution

**[MEASURED]** Primary label, all 5,054 records:

| `Misogyny` | Count | % |
|---|---|---|
| 0 | 2,539 | 50.24 |
| 1 | 2,515 | 49.76 |

The classes are almost perfectly balanced. Accuracy is therefore less
misleading than usual, but macro-F1 remains the safer primary metric.
[ANALYSIS]

**[MEASURED]** Secondary labels:

| Label | `1` count | % of all rows | % of `Misogyny=1` rows |
|---|---|---|---|
| Objectification | 1,493 | 29.54 | 59.36 |
| Prejudice | 919 | 18.18 | 36.54 |
| Humiliation | 393 | 7.78 | 15.63 |

**[MEASURED]** Sub-label consistency:

- No row has `Misogyny=0` together with any sub-label `=1`.
- 288 rows have more than one sub-label.
- **1 row** has `Misogyny=1` but no sub-label (`704_M_pic_0.jpg`).
- Combinations among the 2,515 `Misogyny=1` rows:

  | Combination | Rows |
  |---|---|
  | Objectification only | 1,280 |
  | Prejudice only | 678 |
  | Humiliation only | 268 |
  | Objectification + Prejudice | 163 |
  | Prejudice + Humiliation | 75 |
  | Objectification + Humiliation | 47 |
  | All three | 3 |
  | None | 1 |

Humiliation is rare (393 positives), so the multi-label sub-task would be
strongly imbalanced. [ANALYSIS]

## 9. Text availability

**[MEASURED]**
- 4,906 non-empty texts; **148 empty** (2.93%).
- **The empty texts are unevenly split: 106 in `Misogyny=1`, 42 in
  `Misogyny=0`.**
- Character length: min 1, median 55, mean 59.6, p95 117, max 283.
- Word count: min 1, median 11, mean 11.6, p95 23, max 64.
- 0 URLs, 0 emojis, 0 line breaks.
- **448 texts contain an `@mention`**; 30 contain a hashtag.

**[ANALYSIS / hypothesis]**
- The @mentions may be page or watermark handles that were captured with
  the meme text. If particular handles are associated with one class,
  they could act as a shortcut. Check this in Task 4 or EDA.
- An empty text carries label information (106 vs 42), so the text-only
  model's handling of empty texts must be decided deliberately.

**Text origin [TO BE VERIFIED].** The column name `ExtractedText`
suggests the text was extracted from the image, but whether by OCR,
manually, or OCR plus correction is not documented locally. This matters
for **RQ3**: if `ExtractedText` is itself OCR output, the planned
comparison of "reference text vs OCR text" must be reframed.

## 10. Image availability

**[MEASURED]**

| Check | Result |
|---|---|
| Referenced / found / missing | 5,054 / 5,054 / 0 |
| Unreferenced image files | 0 |
| Corrupt or unreadable (Pillow `verify` + reopen) | 0 |
| Format | JPEG 5,054 (no extension/format mismatch) |
| Width | min 107, median 768, mean 875, p95 1,101, max 2,048 px |
| Height | min 108, median 899, mean 922, p95 1,361, max 2,048 px |
| Aspect ratio (w/h) | min 0.44, median 1.00, p95 1.33, max 3.00 |

Image sizes vary widely, so a consistent resize policy is needed later
(Task 4+). [ANALYSIS]

## 11. Missing data

**[MEASURED]**
- 0 missing IDs, 0 missing labels, 0 missing images.
- 148 empty texts (see §9).

## 12. Duplicate findings

**[MEASURED]**

| Check | Result |
|---|---|
| Duplicate full rows | 0 |
| Duplicate IDs / image references | 0 |
| Exact duplicate texts | 17 groups, 35 rows |
| Duplicate texts ignoring case/whitespace | 18 groups, 37 rows |
| **Duplicate-text groups with conflicting `Misogyny` labels** | **4 groups, 9 rows** |
| Byte-identical images (SHA-256) | 8 groups, 16 images |
| **Identical-image groups with conflicting `Misogyny` labels** | **2 groups** |

Identical-image groups (all have the same text within the group):

| Images | Labels (M/O/P/H) | Note |
|---|---|---|
| `1304_M_pic_2`, `44_M_pic_2` | 1010, 1010 | consistent |
| `1472_M_pic_1`, `2422_M_pic_1` | 1100, 1100 | consistent |
| `1475_M_pic_1`, `2423_M_pic_1` | 1100, 1100 | consistent |
| `1530_M_pic_1`, `2437_M_pic_2` | 1100, 1010 | sub-labels differ |
| `1561_M_pic_3`, `935_M_pic_3` | 1001, 1001 | consistent |
| `1827_NM_pic`, `938_M_pic_2` | 0000, 1010 | **primary label conflict** |
| `2402_NM_pic`, `282_M_pic_12` | 0000, 1110 | **primary label conflict** |
| `2550_M_pic_1`, `303_M_pic_1` | 1100, 1100 | consistent |

Duplicate-text groups with conflicting primary labels: {`1438_NM`,
`1574_M_pic_2`}, {`1443_M_pic_1`, `1785_NM`, `2127_M_pic_1`},
{`2402_NM`, `282_M_pic_12`}, {`1827_NM`, `938_M_pic_2`}. The last two are
also identical images, i.e. the **same meme annotated both ways**.

**Leakage relevance [ANALYSIS]:**
- If duplicates fall into different splits, test performance is inflated.
- Conflicting duplicates are label noise.
- Task 4 must decide how to handle them (e.g. keep each duplicate group
  within one split, and review the conflicting groups manually).
- Exact-hash checking does not catch near-duplicates (re-encoded or
  resized copies); see §19.

## 13. Integrity issues

**[MEASURED]**

1. **File names reveal the label.** All 2,515 `Misogyny=1` files match
   `<n>_M_pic_<n>.jpg`, and all 2,539 `Misogyny=0` files match
   `<n>_NM_pic.jpg`, so the pattern predicts the label perfectly.
   - The file name must **never** be used as a feature or shown to a
     model.
   - Pipeline code must not derive anything from it beyond loading the
     image.
2. 2 identical memes have opposite primary labels; 4 duplicate-text
   groups have conflicting labels (§12).
3. 1 misogynistic row has no sub-label (`704_M_pic_0.jpg`).
4. 148 empty texts, unevenly split across labels (§9).
5. No official split in the local file (§6).

Problems are reported only; **nothing was fixed** (out of scope).

## 14. Dataset comparison

Only one dataset is available, so there is nothing to compare it with.
`results/tables/dataset_inventory.csv` records its measured profile.

## 15. Compatibility analysis

See `results/tables/dataset_compatibility.csv`. **[ANALYSIS]**

| Model family | MIMIC suitability |
|---|---|
| Text-only | Suitable (4,906 non-empty texts; empty-text policy needed) |
| Image-only | Suitable (5,054 readable images) |
| Multimodal | Suitable: the same samples and labels serve all three models |

Distinctions to preserve when evaluating any future dataset:

- **Misogyny** is hostility or prejudice directed at women.
- **Offensiveness, hate, abuse** are broader and target any group or
  person; a meme can be offensive without being misogynistic.
- **Sarcasm and sentiment** are different tasks altogether.

Their labels are not interchangeable. "Offensive" must never be mapped
to "misogynistic" without documentation that supports the mapping.

## 16. Recommended primary dataset

**[ANALYSIS] MIMIC → PRIMARY TRAINING DATASET.**

- It is the only dataset whose task (misogyny), language (Hindi-English
  code-mixed) and modality (memes: image + text) match the project goal.
- It provides text and images for every sample, so the text-only,
  image-only and multimodal models (E1–E5) can be compared on the **same
  examples with the same labels**, which is the scientifically cleanest
  comparison.

## 17. Recommended secondary / external dataset

**None available locally.**

- **External validation** needs a separately collected dataset with a
  compatible misogyny definition. If one is later obtained, it should be
  used only for evaluation, never merged into training.
- A possible text-only candidate appears in the literature: *"Curated
  Hinglish Dataset for Deep Learning-Based Misogyny Detection"*
  ([River Publishers](https://journals.riverpublishers.com/index.php/JRSS/article/view/30547/23603)),
  described as 17,234 social-media comments. Its availability, licence
  and label definition are **[TO BE VERIFIED]**. Being text-only, it
  could at most externally validate the text model.
- MIMIC's reported **official test split** (1,010) is a held-out test set
  if its membership can be obtained. It is not external validation.

## 18. Datasets that should remain separate

- **General offensive / hate / abuse datasets** should not be merged with
  MIMIC; if used at all, they belong to a separate auxiliary task.
  Their labels answer a different question.
- **Text-only misogyny datasets** should not be merged into the
  multimodal training set: they lack images and were annotated under
  different guidelines.
- The unselected `Downloads/DATASET 0x` files were not inspected and
  have no role.

Answers to the required questions **[ANALYSIS]**:

1. **Primary:** MIMIC.
2. **Combine datasets?** No. Only one suitable dataset exists, and
   combining would mix label definitions.
3. **External validation?** None available now. Possible later, for
   evaluation only, after its label definitions are verified.
4. **Offensive-content data:** keep it a separate task, if used at all.
5. **Label compatibility:** no second dataset to compare. Within MIMIC,
   sub-labels are consistent with the primary label, apart from 1 row.
6. **Would combining reduce validity?** Yes. Different definitions,
   annotators, domains and modalities would make RQ1–RQ3 results
   uninterpretable.
7. **Simplest strong strategy:**
   - Use MIMIC alone for E1–E5 and the OCR experiment.
   - Split once, duplicate-aware (keep duplicate groups together),
     stratified by `Misogyny`, seed 42. Use the official split instead if
     its membership can be verified.
   - Use the identical split for all models.
   - Apply one documented empty-text policy to every model.
   - Never use file names as features.
   - Treat the multi-label sub-task as optional, secondary analysis.

## 19. Limitations

- Language figures come from a heuristic, not validated language
  identification.
- Duplicate detection covers exact text, case/whitespace-normalized text
  and byte-identical images only. Near-duplicate images (re-encoded or
  resized) and paraphrased text are **not** detected; perceptual hashing
  could be added later if justified.
- Label meanings, text origin, licence and the official split could not
  be verified from local files (§20).
- This is an inventory, not full EDA: no per-class text analysis, no
  image-content analysis.

## 20. Items requiring manual verification

| # | Item | How |
|---|---|---|
| 1 | Licence / terms of use of MIMIC | Check where the dataset was downloaded (dataset page, GitHub, or authors' request form) and the paper's data-availability statement |
| 2 | Meaning of label values (`Misogyny=1` = misogynistic?) | Paper (DOI above), dataset README |
| 3 | Origin of `ExtractedText` (OCR? manual?) | Paper's data-collection section (affects RQ3) |
| 4 | Official 4,044 / 1,010 split membership | Dataset release / authors |
| 5 | Exact download source and version | Your download record |
| 6 | The 4 conflicting duplicate-text groups and 2 conflicting identical images (§12), and `704_M_pic_0.jpg` | Manual review in Task 4 |
| 7 | Whether @mention handles leak the class (§9) | Task 4 / EDA |
