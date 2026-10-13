# Text Preprocessing Report (Task 6: Preprocessing & Hinglish Normalization)

Date: 2026-10-12 · Preprocessing version **1.0** · Rules version
**1.0**

Code:
- `src/preprocessing/text.py`: modes and the `preprocess()` API
- `src/preprocessing/hinglish.py`: rules engine
- `src/preprocessing/resources/hinglish_normalization.yaml`: rules
- `src/preprocessing/dataset.py`: statistics and audit
- `scripts/preprocess_text.py`: pipeline

Outputs:
- `data/processed/mimic_text_preprocessed_v1.0.csv` (git-ignored)
- `results/tables/preprocessing/`
- `results/metrics/mimic_preprocessing_fingerprint.json`

> **Preprocessing effectiveness has NOT been established.** No model
> has been trained. This task only creates three controlled text
> variants. Experiments E1 vs E2 (text-only) and E4 vs E5 (multimodal)
> will compare them. Nothing below claims that any variant improves
> classification.

All numbers below come from the pipeline outputs. Examples in this
report are **synthetic** sentences, not dataset text.

---

## 1. Objective
Produce three reproducible versions of every sample's text (RAW, BASIC,
HINGLISH_NORMALIZED) so that RQ1 can be tested:

> *Does Hinglish-specific normalization improve misogynistic content
> classification compared with raw/basic preprocessing?*

## 2. Why preprocessing matters for Hinglish
The EDA (Task 5) measured several properties that make this text hard to
model:

- **Spelling variation.** The text is code-mixed, with no standard
  spelling for Romanized Hindi: the same word appears as *nahi*/*nhi*,
  *raha*/*rha*, *bahut*/*bohot*/*bahot*/*bhot*/*bht*.
- **Many rare words.** 65% of word types occur only once, so variant
  spellings split the evidence a model can learn from.
- **Two scripts.** About 29% of samples contain Devanagari, and Unicode
  can encode the same Devanagari character in different ways (159
  samples change under NFC normalization).
- **Social-media artefacts.** There are @handles (523 occurrences,
  including class-exclusive watermarks), hashtags, elongations
  (*yaaar*), double spaces and repeated punctuation.

Normalization *may* make variants share evidence, but it can also erase
signal or change meaning. That is why the three modes are compared
experimentally rather than assumed.

## 3. RAW
The text exactly as stored in `original_text` (Task 4 cleaned file),
byte-for-byte. It is the **experimental control**. The pipeline checks
`text_raw == original_text` for every sample.

## 4. BASIC
Conservative, generic social-media cleaning, applied in this order:

| # | Operation | Rationale / evidence |
|---|---|---|
| 1 | Unicode **NFC** | Composes equivalent Devanagari encodings (159 samples). NFKC was tested and changed nothing more, so the gentler NFC is used |
| 2 | HTML entities unescaped (`&amp;` → `&`) | 0 in MIMIC; protects app input |
| 3 | Zero-width space / BOM removed; NBSP → space | 0 in MIMIC; protects app input. ZWJ/ZWNJ are **kept** because they affect Devanagari rendering |
| 4 | Lowercasing: **off** by default (configurable) | See §12 note |
| 5 | URLs → `<URL>` | §6 |
| 6 | @mentions → `<USER>` | §7 |
| 7 | Hashtags: `#` removed, word kept | §8 |
| 8 | Whitespace collapsed and trimmed | 74 samples had double spaces |

BASIC does **not** translate, stem, lemmatize, remove stopwords, remove
punctuation or numbers, remove emojis, or normalize Hinglish words.

## 5. HINGLISH_NORMALIZED
This mode is BASIC plus two additional steps:

| # | Operation | Detail |
|---|---|---|
| 9 | Repeated-character normalization | §11 |
| 10 | Romanized-Hindi spelling normalization | §14 (rules file) |

BASIC and HINGLISH_NORMALIZED differ **only** in steps 9–10. That makes
the comparison between them (E1 vs E2, E4 vs E5) the clean test of
Hinglish normalization.

RAW vs BASIC additionally includes mention masking, which partly removes
the watermark shortcut (§7). Gains from BASIC over RAW must therefore
**not** be attributed to Hinglish handling.

## 6. URL strategy
URLs (`http(s)://…`, `www.…`) are replaced by `<URL>`. This is
configurable (`url_handling: placeholder | keep | remove`). MIMIC
contains **0 URLs**. The rule exists for consistent handling of app
input, so that individual URLs cannot dominate the vocabulary.

## 7. Mention strategy
@mentions are replaced by `<USER>`. This is configurable
(`placeholder | keep | remove`).

- **Effect:** 523 mentions in 448 samples.
- **Why it matters:** this masks the class-exclusive watermark handles
  found in the EDA (`@doublemeaning`, `@doublemean`; 128 occurrences
  with "@").
- **Not fully removed:** 18 occurrences of these names **without** "@"
  remain in the text. Removing them would need handle-specific rules,
  which is a possible future ablation, not done here.
- **Email addresses** are not treated as mentions (`a@b.com` is kept).

## 8. Hashtag strategy
The `#` symbol is removed and the word kept (`#WomenCanDrive` →
`WomenCanDrive`), so hashtags never become empty. This is configurable
(`strip_symbol | keep`). There are 35 hashtags in 30 samples.

No hashtag segmentation (`WomenCanDrive` → `Women Can Drive`) is done:
there are too few hashtags to justify it.

## 9. Emoji strategy
Emojis are **kept unchanged**; `keep` is the only allowed value in v1.0.

- MIMIC contains **0 emojis**, because the text comes from OCR.
- In app input emojis can carry meaning, so they are never silently
  removed.
- No emoji library was added: nothing in the dataset would use it.

## 10. Negation strategy
Negations can flip meaning ("*ladki nahi*" vs "*ladki*"), so they are
protected in three ways:

1. **Never removed.** No step deletes words. Stopword removal is
   disabled by design (§13).
2. **Safe normalization.** A negation variant may only map to a negation
   word, and a non-negation word may never map to a negation. This is
   enforced when the rules file is loaded (`validate_rules`). In v1.0
   the only negation rules are `nhi` → `nahi` and `nahin` → `nahi`.
3. **Audited.** For each sample the audit counts negation words (English,
   Romanized Hindi and Devanagari forms) in RAW vs HINGLISH_NORMALIZED.
   One sample changed: its handle `@ni` was masked as a mention, and
   `ni` can also be a negation. No Hinglish rule changed a negation.

Ambiguous negation-like forms (`ni`, `nai`, `na`) are **not normalized**.

## 11. Repeated-character strategy
In HINGLISH_NORMALIZED only (configurable), a Latin letter repeated
**3+** times is shortened to **2**: `bhaiiii` → `bhaii`, `nooooo` →
`noo`.

- Keeping two letters avoids damaging legitimate double letters (*good*,
  *accha*, *cool*).
- An elongation can then match a rule (`yaaar` → `yaar`).
- Effect: 81 elongations in 69 samples (EDA: 85 elongated words).
- Devanagari repeats (18) are not touched.

## 12. Punctuation strategy (and casing)
- **Punctuation is kept**, including repeated `!!`, `??`, `..`, `...`
  (`..` appears 274 times, `!!` 81 times). It may carry tone, and some
  tokenizers use it. No punctuation is removed in any mode.
- **Numbers are kept** (Latin and Devanagari digits); 20–25% of texts
  contain digits (EDA).
- **Casing is kept by default.** Uppercase-heavy text is
  class-correlated (EDA: 8.1% vs 2.9%), which may be signal or a
  template shortcut.
  - Lowercasing is left to the model side: `TfidfVectorizer(lowercase=…)`
    or a cased/uncased tokenizer, so it can be tested as a
    hyperparameter.
  - Hinglish rules match case-insensitively and preserve the casing
    pattern (`NHI` → `NAHI`, `Nhi` → `Nahi`).

## 13. Stopword decision
**No stopword removal.** Stopword lists contain negations (*not, no,
nahi, mat*) and words that carry meaning in abusive expressions.
`remove_stopwords: false` is fixed in v1.0; setting it to `true` raises
an error rather than silently doing something unvalidated.

## 14. Hinglish normalization rules
File: `src/preprocessing/resources/hinglish_normalization.yaml` (v1.0).
It is plain YAML: inspectable, versioned and easy to edit.

**Selection method:**
- Counted whole-word, case-insensitive Latin tokens in the cleaned
  corpus.
- Included a variant only if it occurs **≥ 2 times** and has **no
  plausible second meaning**.
- The canonical form is a common spelling that also occurs in the
  corpus.
- The evidence counts are stored next to every rule.

**32 rules, 40 variants:**

| Canonical | Variants (corpus count) |
|---|---|
| nahi (negation) | nhi (137), nahin (11) |
| kar / karna / karo / karte / karke / karta / karti | kr (39) / krna (6) / kro (17) / krte (12) / krke (6) / krta (6) / krti (11) |
| raha / rahi / rahe | rha (37) / rhi (21) / rhe (15) |
| gaya / gayi | gya (31) / gyi (17) |
| hoon | hu (152) |
| kyun | kyu (52) |
| bahut | bohot (22), bahot (11), bhot (7), bht (4) |
| accha | acha (20), achha (10) |
| kaise / mujhe / kuch / pata / theek | kese (13) / muje (6) / kuchh (2) / pta (14) / thik (14) |
| matlab / jab / sath / yaar / please | mtlb (3) / jb (4) / saath (15) / yar (6), yarr (2) / plz (4) |
| ladki / ladka / ladkiyan | larki (4) / larka (2) / ladkiya (17), ladkiyaan (3) |
| wala / wali | vala (4) / vali (3), waali (3) |

**Deliberately excluded** (listed with reasons in the YAML):

| Excluded | Reason |
|---|---|
| `h`, `hy`, `he` (→ hai) | Letter, "hey" or English *he* |
| `ni`, `nai` | Particle/name, and "new" |
| `fir` | FIR, the police report |
| `bhut` | *bhoot*, ghost |
| `gai` | Cow |
| `samaj` | Society |
| `kha` | Eat |
| `kia`, `ky` | Maps to several different words |
| `hun` | English endearment |
| `mai`/`main`/`mein`/`me`/`m` | "I" vs "in" vs English *me* |
| `or`, `to`, `tho`, `k`, `sb`, `yr`, `tm`, `hm`, `tb`, `bt`, `bat` | English words or abbreviations |
| Variants occurring only once | Below the evidence threshold |

**Spot check:** short word windows around several rules (`hu`, `acha`,
`saath`, `yar`, `jb`, `vala`, `thik`, `plz`, `kese`) were reviewed. The
replaced words were used in the intended sense (e.g. *gaya hu*, *jaata
hu* → *hoon*). No misuse was seen in these samples, but this is not an
exhaustive check.

## 15. Transformation statistics
From `results/tables/preprocessing/preprocessing_summary.csv` and
`operation_counts.csv`:

| Mode | Changed vs RAW | % | Mean chars | Mean tokens | Vocabulary (EDA tokens) | Vocabulary (case-sensitive words) | Newly empty |
|---|---|---|---|---|---|---|---|
| RAW | 0 | 0.00 | 57.857 | 11.258 | 14,244 | 20,235 | – |
| BASIC | 671 | 13.30 | 57.477 | 11.258 | 14,022 | 19,919 | **0** |
| HINGLISH_NORMALIZED | 1,255 | 24.88 | 57.615 | 11.258 | 13,962 | 19,810 | **0** |

- HINGLISH_NORMALIZED differs from BASIC in **655** samples.
- The 148 empty texts stay empty in every mode; no text becomes empty
  through preprocessing.

| Operation | Occurrences | Samples |
|---|---|---|
| Mention → `<USER>` | 523 | 448 |
| Unicode NFC | 159 | 159 |
| Whitespace collapse | 74 | 74 |
| Hashtag `#` removed | 35 | 30 |
| Elongation 3+ → 2 (Hinglish mode) | 81 | 69 |
| Hinglish rules (Hinglish mode) | **761** | **597** |

The most frequent rules are `hu`→hoon (152), `nhi`→nahi (137), `kyu`→kyun
(52), `kr`→kar (39) and `rha`→raha (37). All 40 variants fired at least
once. Rule counts match the YAML evidence counts, except `acha` and
`ladkiya` (one fewer each, probably inside masked mentions or hashtags).
Full table: `hinglish_rule_frequencies.csv`.

The vocabulary reduction from RAW to HINGLISH_NORMALIZED is small:
- −2.1% case-sensitive words (20,235 → 19,810);
- −2.0% lowercased EDA tokens (14,244 → 13,962).

Whether it matters for classification is **unknown**.

## 16. Before/after examples
`results/tables/preprocessing/preprocessing_examples.csv` holds 33 rows:
up to 3 per category, taken in source-file order and **not
cherry-picked**. Audit-flagged rows are included.

- **Categories:** mention, hashtag, whitespace, unicode_nfc,
  repeated_characters, romanized_hindi_rule, negation_rule,
  negation_present_unchanged, unchanged, audit_flagged; URL, HTML and
  emoji are marked "NONE IN DATASET".
- **Kept out of Git:** the file contains dataset text, so it is
  git-ignored, consistent with the project rule of no dataset text in
  Git.

Synthetic illustrations of the same behaviour:

| RAW | BASIC | HINGLISH_NORMALIZED |
|---|---|---|
| `Ye  NHI hoga yaaar!!! @page #WomenCanDrive` | `Ye NHI hoga yaaar!!! <USER> WomenCanDrive` | `Ye NAHI hoga yaar!!! <USER> WomenCanDrive` |
| `Main kr rha hu, bhaiiii 100%` | (unchanged) | `Main kar raha hoon, bhaii 100%` |
| `pta nhi kyu` | (unchanged) | `pata nahi kyun` |
| `FIR filed, gai, h, me main` | (unchanged) | (unchanged: ambiguous words excluded) |
| `see https://a.co &amp; more` | `see <URL> & more` | `see <URL> & more` |

## 17. Risky transformations (audit)
`results/tables/preprocessing/normalization_audit.csv` contains sample
IDs and numbers only. **48 samples** are flagged for manual review, and
none are removed:

| Flag | Samples | Finding |
|---|---|---|
| `many_rule_replacements` (≥ 3 rule hits) | 25 | Hinglish-heavy texts. Expected, but the most changed |
| `large_length_change` (> 30% characters) | 12 | 11 are explained by mention masking (watermark-heavy texts); 1 is a 3-character text expanded by a rule |
| `high_replacement_share` (≥ 30% of tokens changed) | 12 | Short texts where most words were normalized |
| `negation_count_changed` | 1 | Handle `@ni` masked as `<USER>`; `ni` is also a negation form. A mention-policy effect, not a rule error |
| `became_empty` | **0** | – |
| `token_count_changed` | **0** | Rules replace one word with one word |

## 18. Limitations
- **Rules are corpus-specific.** They were derived from MIMIC itself
  (5,044 OCR texts); spellings outside the corpus are not covered.
  Applying rules derived from the full dataset to every split is
  standard for deterministic rule-based normalization, because no
  statistics are learned. Still, the rule choice used the whole corpus,
  test samples included. This should be acknowledged in the paper; a
  stricter version could re-derive the rules from the training split
  only.
- **No context.** Rules are word-by-word. An ambiguous word that slipped
  through could still be wrong in some sentences; the spot check was not
  exhaustive.
- **Devanagari** is only NFC-normalized. There is no transliteration
  between the scripts, so `नहीं` and `nahi` remain different tokens.
  Transliteration is a candidate for a later version.
- **OCR artefacts are not repaired.** For example, the visarga `ः` after
  a Latin word (90 occurrences, probably a misread `:`). It is a
  candidate rule that needs review.
- **Watermarks persist.** Names without "@" (18 occurrences) remain.
- **Elongation** shortens to 2 letters, so `noo` and `no` stay distinct.

## 19. Reuse during inference (the app)
- The Streamlit app must call **the same function** for both modes of
  analysis: `from src.preprocessing import preprocess` and
  `preprocess(text, mode=..., config=PreprocessingConfig.from_config(load_config(), REPO_ROOT))`.
  This covers Analyze Text and the OCR text in Analyze Meme.
- **No preprocessing code may be written in `app/`.**
- Every trained model must record the `text_preprocessing.version` (here
  `1.0`), the mode and the rules version it was trained with. Task 2's
  `model_versions.preprocessing_version` field exists for this.
- The app must use the mode the deployed model was trained on.
- Any change to the rules file or the code requires a version bump and
  re-running `scripts/preprocess_text.py`. The fingerprint records the
  code and rules checksums so mismatches can be detected.
