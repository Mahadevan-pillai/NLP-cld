# Data Split Report (Task 7: Leakage-Safe Train / Validation / Test Split)

Date: 2026-10-12 · Code: `src/data/splits.py`, `scripts/create_splits.py`

Artifacts:
- `data/splits/split_assignments.csv` (**canonical**, tracked)
- `data/splits/split_manifest.json` (tracked)
- `data/splits/{train,validation,test}.csv` (git-ignored: contain text)
- `results/tables/splits/` (`split_distribution.csv`, `leakage_audit.csv`,
  `leakage_groups.csv`)

> **The same split assignment is used for text-only, image-only and
> multimodal experiments to enable fair comparison.**
>
> Splitting is done by `sample_id`. Every text variant (`text_raw`,
> `text_basic`, `text_hinglish_normalized`), the image and the
> image + text pair of a sample are always in the same split.

## 1. Dataset used
- **MIMIC**, cleaned (Task 4) and preprocessed (Task 6):
  `data/processed/mimic_text_preprocessed_v1.0.csv`.
- Before splitting, the input was checked against its Task 6 fingerprint:
  - 5,044 unique sample IDs, identical in order to Task 4;
  - valid labels;
  - all three text variants present;
  - all images present.
- All 5,044 image files were re-hashed and match their Task 4 SHA-256
  values.

## 2. Dataset version
| Item | Value |
|---|---|
| Raw annotation file | `MIMIC2024.csv`, SHA-256 `98dd8843ef81…` |
| Preprocessed file | SHA-256 `51faa312cd1b…`, preprocessing v1.0 |
| Split assignment | SHA-256 `47fb02a87b36…` (recorded in the manifest) |

## 3. Total samples
5,044. Every sample is assigned to exactly one split; nothing was dropped.

## 4. Split strategy
MIMIC has **no official split** (Task 4), so one canonical split is
created. It has to do two things at once:

1. Keep **leakage groups** together (§8).
2. Preserve the **class balance** and **70 / 15 / 15** ratios.

**Algorithm:**
1. Build leakage groups with union-find (§8).
2. Run scikit-learn `StratifiedGroupKFold(n_splits=20, shuffle=True,
   random_state=42)` with `y = target_label` and `groups = leakage
   group`. This produces 20 folds of about 5% each. A whole group always
   goes to a single fold, and folds are balanced by class.
3. Map folds to splits: folds 0–2 → **test**, 3–5 → **validation**,
   6–19 → **train** (3 / 3 / 14 folds = 15 / 15 / 70%).
4. Run the leakage audit (§12). Nothing is written if any blocking check
   fails.

## 5. Ratios
| Split | Target | Actual | Samples |
|---|---|---|---|
| Train | 70% | **70.02%** | 3,532 |
| Validation | 15% | **14.95%** | 754 |
| Test | 15% | **15.03%** | 758 |

The small deviations come from keeping groups whole. Leakage protection
was not traded for exact percentages.

## 6. Random seed
`random_seed: 42`, the global seed in `config.yaml`.

## 7. Stratification method
Stratification is by the **primary label `target_label` only**.

- The secondary labels (Objectification / Prejudice / Humiliation) were
  not used. They exist only for misogynistic samples and are secondary
  to the research questions.
- They nevertheless ended up proportionally distributed:

| Sub-label | Train | Validation | Test |
|---|---|---|---|
| Objectification | 1,049 | 214 | 225 |
| Prejudice | 642 | 152 | 122 |
| Humiliation | 262 | 55 | 75 |

## 8. Duplicate / group strategy
Exact duplicate memes were already removed in Task 4. Task 7 therefore
links **samples that are different files but the same, or almost the
same, underlying meme**:

| Link | Source | Used? | Why |
|---|---|---|---|
| Identical image file (SHA-256) | Task 4 hashes, re-verified | Yes | Same image must never be in two splits. None remain after Task 4 |
| Near-duplicate image pair (dHash ≤ 4 of 64 bits) | Task 5, 55 pairs | Yes | Re-encoded or resized copies of the same meme. Not manually reviewed, so grouping is the **conservative** choice: a false positive only constrains randomization, while a missed true duplicate leaks |
| Same caption, different image file, **≥ 3 words** | Task 4 `duplicate_group` (case/whitespace-insensitive) | Yes (7 groups) | A text-only model would otherwise see the exact test caption during training |
| Same caption, **< 3 words** | Task 4 `duplicate_group` | **No** (3 groups) | Trivial generic captions (1, 1 and 2 words), as the task instructions require |
| Identical *normalized* text | Task 6 | **No** | Normalization could merge unrelated samples; reported only (§12) |

**Result (`results/tables/splits/leakage_groups.csv`):**
- **43 multi-sample groups covering 93 samples**: 40 pairs, 2 triples
  and 1 group of 7. The group of 7 is a chain of near-duplicate images,
  probably one meme template, and all 7 are non-misogynistic.
- 4 groups contain both labels: reused templates whose captions differ.
- 6 of the 7 caption groups are **also** near-duplicate images, i.e. the
  same meme saved as different files. The two signals confirm each
  other.
- 4,994 groups in total; most samples are singletons.

## 9. Train distribution
3,532 samples: 1,776 non-misogynistic (50.28%), 1,756 misogynistic
(49.72%). 3,427 have non-empty text.

## 10. Validation distribution
754 samples: 379 non-misogynistic (50.27%), 375 misogynistic (49.73%).
732 have non-empty text.

## 11. Test distribution
758 samples: 382 non-misogynistic (50.40%), 376 misogynistic (49.60%).
737 have non-empty text.

**Comparison with the full dataset** (50.30% / 49.70%): every split is
within **0.13 percentage points**. Empty texts are spread 105 / 22 / 21,
which is proportional.

## 12. Leakage checks
`results/tables/splits/leakage_audit.csv`:

| Check | Result |
|---|---|
| Only valid split names (`train`, `validation`, `test`) | ✅ |
| No `sample_id` duplicated or in more than one split | ✅ 0 |
| Splits mutually exclusive | ✅ 0 overlaps |
| Union of splits = dataset (5,044) | ✅ |
| No sample lost or duplicated | ✅ 5,044 assigned once |
| No leakage group crosses splits | ✅ 0 of 43 |
| No identical image (SHA-256) crosses splits | ✅ 0 |
| No near-duplicate image pair crosses splits | ✅ 0 of 55 |
| No identical record (image + text) crosses splits | ✅ 0 |
| Reproducible re-run identical | ✅ |
| *Info:* Task 4 caption groups crossing | 1 (a trivial < 3-word caption, allowed by design) |
| *Info:* identical non-empty raw text crossing | 1 (the same trivial caption) |
| *Info:* identical non-empty normalized text crossing | 1 (not used for grouping) |

**Known shortcut, NOT leakage:** samples containing the watermark
`doublemean*` are spread **train 70 / validation 6 / test 14**.

- A model that learns the watermark can score on the test set without
  having seen those exact memes. Task 5 identified this shortcut.
- Model evaluation must therefore also report results **with and without
  watermark samples**.

## 13. Reproducibility check
- The split is generated **twice inside the script** and compared before
  anything is written: identical.
- Re-running the whole script produced a byte-identical
  `split_assignments.csv` (SHA-256 `47fb02a87b36…`).
- Automated tests check determinism (same seed gives the same split), seed
  dependence, and independence from the text variant.
- **Version freeze:** the assignment file is **tracked in Git** and
  checked against the manifest checksum whenever it is loaded
  (`load_split_assignments`). A future scikit-learn version that shuffled
  differently therefore cannot silently change the split. The manifest
  records Python 3.10.0, scikit-learn 1.7.2 and pandas 2.3.3.
- **This risk is real (measured while setting up CI, 2026-10-12).**
  Regenerating the split under Python 3.14 / scikit-learn 1.9.1 / pandas
  3.0.6 produced a **different** split: 2,279 of 5,044 samples would
  move.
  - `scripts/create_splits.py` therefore **aborts and writes nothing**
    when the regenerated split differs from the existing canonical file,
    unless `--regenerate` is passed deliberately.
  - Always load the frozen file. Never regenerate it to "reproduce" the
    split.

## 14. Test-set usage policy
The test set is **LOCKED** from now on.

| Split | Use |
|---|---|
| **Train** | Fitting models and everything learned from data |
| **Validation** | Model selection, hyperparameter tuning, preprocessing-variant comparison (E1 vs E2, E4 vs E5), development error analysis |
| **Test** | **Only** the final evaluation of configurations frozen on validation. Report it once per final model, and never use it to choose between options |

Enforcement (simple, not a security mechanism):
- `src.data.splits.load_split("test")` raises an error unless
  `allow_test=True` is passed explicitly.
- Future training scripts should evaluate the test set only behind an
  explicit flag (e.g. `--final-test`).
- Every test evaluation must be recorded in `RESEARCH_LOG.md`.

**Preprocessing leakage rule:**
- Any operation that **learns from corpus statistics** must be fitted on
  **train only**. This includes TF-IDF vocabulary and IDF weights,
  feature scaling, learned vocabularies, data-driven thresholds, class
  weights, and calibration.
- It is then *applied* unchanged to validation and test.
- Validation and test data must never influence a fitted component.
- The Task 6 normalization is deterministic (fixed rules, no fitting), so
  it is applied identically to all splits.
- Caveat (Task 6 report §18): its rule list was chosen by looking at
  word counts in the whole corpus, test included. That is acknowledged as
  a limitation. The rules must **not** be changed in response to
  validation or test results.

## 15. Limitations
- **Near-duplicate grouping** uses a perceptual hash threshold and was
  not manually confirmed. Some grouped pairs may not be true duplicates
  (the cost is negligible), and near-duplicates beyond the threshold
  (heavier edits, crops) are not detected.
- **Caption grouping** relies on Task 4's case/whitespace-insensitive
  equality. Paraphrased captions and the 3-word cut-off are judgment
  calls.
- **Shortcut features** (watermark handles, class-correlated image
  resolutions) are present in all splits. Splitting cannot remove them;
  evaluation must analyse them.
- **One split, one seed.** Variance across random splits is not measured.
  If time allows, later experiments could repeat model training over
  multiple seeds or folds, keeping this split as the canonical reported
  one.
- **Small test set.** About 758 samples gives limited precision for
  comparing close models. Differences must not be called significant
  without an appropriate statistical test.
