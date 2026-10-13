# Multi-Category & Hierarchical Label Feasibility Report (Task 8.5)

Date: 2026-10-12

Code: `scripts/analyze_category_feasibility.py`, `src/data/categories.py`
· Tables: `results/tables/category_feasibility/` · Proposed config:
`config/categories.yaml`

This is a **feasibility and design study only**. Nothing was trained,
relabelled or merged; Task 8 (B01–B06) is untouched; no test-set
evaluation took place. Label counts per split are descriptive
statistics.

Tags:

| Tag | Meaning |
|---|---|
| **[VERIFIED DATASET LABEL]** | An annotation actually present in the data |
| **[PROPOSED MAPPING]** | Our interpretation of how a label relates to a concept |
| **[UNSUPPORTED CATEGORY]** | No annotation exists; must not be trained or displayed as a prediction |

---

## 1. Current binary task
Task 8 established text baselines for **misogynistic vs
non-misogynistic** (MIMIC `Misogyny` label; best validation macro-F1
0.6671, B03). This remains the project's primary task and is unchanged.

## 2. Reason for investigating category expansion
The planned app could present a richer moderation view (safe / harmful,
harm type). Before any design commits to that, we must know whether the
data can actually support it. Training on categories nobody annotated
would produce outputs with no scientific basis.

## 3. Available dataset labels
`all_dataset_labels.csv`.

**MIMIC, the only content dataset [VERIFIED DATASET LABEL]:**

| Column | Role | Values (cleaned, n = 5,044) | Missing | Meaning |
|---|---|---|---|---|
| `Misogyny` | primary, binary | 0: 2,537 · 1: 2,507 | 0 | 1 = Misogynistic, 0 = Non Misogynistic (dataset card) |
| `Objectification` | sub-category | 0: 3,556 · 1: 1,488 | 0 | category of a misogynistic meme (dataset card, sub-task 2) |
| `Prejudice` | sub-category | 0: 4,128 · 1: 916 | 0 | same |
| `Humiliation` | sub-category | 0: 4,652 · 1: 392 | 0 | same |

The raw release has the same four columns (5,054 rows); Task 4 removed
10 duplicate or conflicting memes.

**Other files on the machine.** The three un-selected files in
`Downloads` were inspected read-only on 2026-10-12, with your approval.
**None is a content-moderation dataset:**

| File | Content | Label-like column |
|---|---|---|
| `DATASET 00.csv` | agriculture: crop, soil, moisture, temperature | `result` 0/1/2 |
| `DATASET 01.csv` | mortality statistics by country, age and sex | none |
| `DATASET 02.xlsx` | FIFA World Cup match results | none |

## 4. Candidate category mapping
`label_mapping_analysis.csv` scores every label × concept pair. Rows not
listed below are **NOT_SUPPORTED** (no annotation exists).

| Label | Concept | Status | Why / semantic difference |
|---|---|---|---|
| `Misogyny = 1` | MISOGYNY | **SUPPORTED** | Exactly what it annotates |
| `Misogyny = 0` | SAFE | **NOT_SUPPORTED** | Means *non-misogynistic*. Other harms (hate against other groups, insults, threats, obscenity) were never annotated, so "0" is **not** "safe" |
| `Prejudice` | HATE | PARTIALLY_SUPPORTED (low) | Gender prejudice is one form of identity hostility, but only against women and only within misogynistic memes. It cannot represent hate in general |
| `Humiliation` | ABUSE | PARTIALLY_SUPPORTED (low) | Humiliation of women only, within misogynistic memes. Not general personal abuse |
| `Objectification` | OBSCENE | AMBIGUOUS (low) | Objectification is not the same as sexual or obscene content, and no explicitness was annotated |
| any label | THREAT, SARCASM | NOT_SUPPORTED | Never annotated |

All three sub-labels map to MISOGYNY (SUPPORTED) as **sub-categories of
it**, not as separate harm categories. **[PROPOSED MAPPING]** No forced
mappings: "offensive → abuse", "hate → misogyny" and "sarcasm → harmful"
were not applied.

## 5. MIMIC subcategories
**[VERIFIED DATASET LABEL]** MIMIC genuinely contains
**Objectification, Prejudice and Humiliation**. These are the categories
of its sub-task 2 (dataset card). No other sub-categories exist.

| Property | Measurement |
|---|---|
| Relationship to the main label | **Strictly nested:** 0 sub-label positives among the 2,537 non-misogynistic memes |
| Mutually exclusive? | **No.** 287 of 2,507 misogynistic memes (11.45%) have 2 or 3 sub-labels |
| Overlap between labels (Jaccard, within misogynistic) | Objectification–Prejudice 0.074 · Prejudice–Humiliation 0.063 · Objectification–Humiliation 0.027, so the labels are largely distinct |
| Missing annotations | 0 missing values; 1 misogynistic meme has no sub-label (`704_M_pic_0`) |

Label combinations (`subcategory_structure.csv`):

| Combination | Memes | % of misogynistic |
|---|---|---|
| Objectification only | 1,276 | 50.90 |
| Prejudice only | 676 | 26.96 |
| Humiliation only | 267 | 10.65 |
| Objectification + Prejudice | 162 | 6.46 |
| Prejudice + Humiliation | 75 | 2.99 |
| Objectification + Humiliation | 47 | 1.87 |
| All three | 3 | 0.12 |
| None | 1 | 0.04 |

**Conclusion: the correct formulation is MULTILABEL**, three independent
yes/no labels, **conditional on misogyny = 1**. A multiclass formulation
would force one label per meme and discard the 287 multi-category memes.

## 6. Secondary dataset labels
None are available. The project has no secondary content dataset; the
three other local files are unrelated domains (§3). So nothing can
support separate hate, abuse, threat, obscenity, sarcasm, sentiment or
target-group classifiers, and **nothing should be merged**.

## 7. Sample sufficiency
`category_sufficiency.csv`, counted on the canonical Task 7 split.

**Tier criteria** (judged on train and validation positives):

| Tier | Train positives | Validation positives |
|---|---|---|
| STRONG | ≥ 1,000 | ≥ 150 |
| MODERATE | ≥ 500 | ≥ 100 |
| WEAK | ≥ 200 | ≥ 50 |
| INSUFFICIENT | below that, or no annotation | |

*Rationale:*
- Train positives drive learning.
- Validation positives set how precisely a per-category score can be
  measured. With about 55 positives, a recall estimate around 0.6 has a
  95% interval of roughly ±13 points; with about 150 positives, roughly
  ±8 points.
- These thresholds are a documented project judgment, not a standard.

| Category | Population | Positives | % positive in population | Negative:positive ratio | Train / val / test positives | Tier |
|---|---|---|---|---|---|---|
| MISOGYNY | all 5,044 | 2,507 | 49.70 | 1.01 | 1,756 / 375 / 376 | **STRONG** |
| OBJECTIFICATION | misogynistic 2,507 | 1,488 | 59.35 | 0.68 | 1,049 / 214 / 225 | **STRONG** |
| PREJUDICE | misogynistic 2,507 | 916 | 36.54 | 1.74 | 642 / 152 / 122 | **MODERATE** |
| HUMILIATION | misogynistic 2,507 | 392 | 15.64 | 5.40 | 262 / 55 / 75 | **WEAK** |
| SAFE, HATE, ABUSE, THREAT, OBSCENE, SARCASM | – | 0 | – | – | – | **INSUFFICIENT (no annotation)** |

**Limitations to state with every sub-category result:**
- **Humiliation** has few validation positives (55), so its scores will
  be noisy.
- **Prejudice** has fewer test than validation positives (122 vs 152).

## 8. Multiclass vs multilabel analysis
| Design | Matches the annotations? | Reason |
|---|---|---|
| **A: single multiclass** (e.g. none / objectification / prejudice / humiliation) | **No** | 11.45% of misogynistic memes have several categories; multiclass cannot represent them |
| **B: hierarchical** | **Yes, limited to misogyny** | The annotation is itself hierarchical (sub-labels exist only under misogyny = 1) |
| **C: multilabel** | **Yes, for the sub-categories** | Independent yes/no per sub-category, with co-occurrence allowed |
| **D: multiple specialized harm classifiers** | **No** | Needs hate, abuse, threat and obscenity datasets that do not exist here |

## 9. Hierarchical classification feasibility
The preferred conceptual architecture (*Harm detection → Safe / Harmful
→ Misogyny / Hate / Abuse / Threat / Obscene*) is **NOT supportable**:

- There is no "harmful vs safe" label. `Misogyny = 0` is not "safe".
- 4 of the 5 harm branches (Hate, Abuse, Threat, Obscene) have no data.
- The sarcasm branch has no data.

**What the data supports [PROPOSED]** is a two-stage hierarchy *within
misogyny*:

```
INPUT (text / meme)
   |
   v
STAGE 1  Misogyny detection (binary)          [existing task, Task 8 baselines]
   |
   +-- non-misogynistic  -> stop  ("no misogyny detected", NOT "safe")
   |
   +-- misogynistic
          |
          v
STAGE 2  Misogyny sub-categories (multilabel, conditional)
          [ ] Objectification   (STRONG)
          [ ] Prejudice         (MODERATE)
          [ ] Humiliation       (WEAK)
```

Notes on training and evaluating this design (later tasks):

- **Training data.** Stage 2 is trained only on misogynistic memes from
  the train split, which is how the labels were annotated.
- **Evaluation.** Stage 2 is evaluated in two ways:
  - with **gold** stage-1 labels: sub-category quality alone;
  - **end-to-end**: errors in stage 1 propagate.
- **Joint alternative.** A single multilabel model with a consistency
  rule (sub-labels only when misogyny = 1) is an equivalent option.

## 10. Text / image / multimodal support
`modality_support.csv`:

- Every MIMIC sample has an image and an OCR text field.
- MISOGYNY and all three sub-categories can therefore be trained as
  **text-only, image-only and multimodal**.
- Text is empty for 148 memes (106 of them inside the misogynistic
  population used by stage 2).
- No unsupported category can be trained in any modality.

## 11. Recommended categories
| Category | Use | Status in `config/categories.yaml` |
|---|---|---|
| MISOGYNY (binary) | primary task | enabled |
| OBJECTIFICATION | stage-2 multilabel head | disabled until approved; STRONG |
| PREJUDICE | stage-2 multilabel head | disabled until approved; MODERATE |
| HUMILIATION | stage-2 multilabel head | disabled until approved; WEAK, results to be reported with caution |

## 12. Categories NOT scientifically supportable
**[UNSUPPORTED CATEGORY]** These must not be trained, displayed as model
predictions, or produced by relabelling MIMIC:

- **SAFE / NON-HARMFUL**
- **HATE / IDENTITY ATTACK**
- **PERSONAL INSULT / ABUSE**
- **THREAT / VIOLENCE**
- **SEXUAL / OBSCENE**
- **SARCASM / MOCKERY**

Adding any of them would require new, separately annotated datasets
with compatible definitions, evaluated as **separate tasks**. That is
out of scope for the current data.

## 13. Recommended final architecture
**Two-stage hierarchical misogyny classifier: Design B, whose stage 2
is a multilabel classifier (Design C).**

- Stage 1 (binary misogyny) remains the primary task for RQ1–RQ3.
- Stage 2 (Objectification / Prejudice / Humiliation, multilabel) is an
  **optional secondary task**, enabled only after your decision.
- Each stage can be built as text-only, image-only and multimodal, so
  RQ2 can also be studied per sub-category.

## 14. Moderation recommendation design (not implemented)
The layer turns model signals into a **recommendation**, never an
automatic block.

```
Prediction (per modality) -> calibrated score -> rules -> ALLOW | WARNING | HUMAN REVIEW
```

| Outcome | Proposed condition (to be validated) | User-facing meaning |
|---|---|---|
| **ALLOW** | Misogyny score below a low threshold `t_low`, and no review trigger | "No misogyny detected by the model." **Not** "safe": other harms are not assessed |
| **WARNING** | Score ≥ a high threshold `t_high` and modalities agree | "Likely misogynistic content", plus detected sub-categories |
| **HUMAN REVIEW** | `t_low` ≤ score < `t_high`, **or** text/image/multimodal disagree, **or** text empty or OCR unreliable, **or** a known shortcut signal is present (e.g. the watermark pattern) | "Uncertain: needs a human decision" |

There is **no BLOCK**: ML predictions at the measured accuracy (about
0.65–0.67 validation macro-F1 for text baselines) cannot justify
automatic removal.

### Proposed UI output (Analyze Text / Analyze Meme; not implemented)
```
CONTENT ANALYSIS (misogyny detection; other harm types are NOT assessed)

Misogyny:            Detected / Not detected
Sub-categories:      Objectification / Prejudice / Humiliation
                     (only if misogyny detected, only if stage 2 is enabled;
                      several can apply)
Sarcasm:             not available (no annotated data)

Text model:          prediction + score type
Image model:         prediction + score type          (meme mode only)
Multimodal model:    prediction + score type          (meme mode only; primary)
  score type = "calibrated probability" only if calibration was validated,
               otherwise "decision score (not a probability)"

Moderation recommendation:  ALLOW / WARNING / HUMAN REVIEW (+ reason)
Model & preprocessing versions;
note: "automated estimate; may be wrong; not a judgement about any person"
```

- No confidence value is shown until calibration has been validated.
- "Safety: Safe / Harmful" and "Primary category: Hate / Abuse / Threat /
  Obscene" are **not shown**, because no model for them can exist with
  this data.

**Must be validated later, on validation data only:**
1. **Calibration.** LR probabilities need checking; SVM margins are not
   probabilities and need Platt or isotonic calibration. Report
   reliability diagrams and ECE.
2. **Thresholds.** Choose `t_low` and `t_high` from validation
   precision/recall trade-offs, with an explicit cost assumption.
3. **Review load.** The share of inputs routed to HUMAN REVIEW.
4. **Subgroup errors.** Error rates for Devanagari, short text, empty text
   and watermark subgroups.
5. **Final check.** One frozen evaluation on test.

## 15. Research implications
- **RQ1–RQ3:** keep them **specific to misogyny**. Broadening RQ1 to
  "harmful content" is **not supported**, because only misogyny is
  annotated.
- **RQ4 (PROPOSED, supported in narrowed form):** *"How effectively can
  hierarchical multilabel models identify misogyny sub-categories
  (objectification, prejudice, humiliation) in Hindi-English memes, and
  does multimodality help per sub-category?"*
  - The dataset was designed for this (sub-task 2).
  - Humiliation is WEAK, so results must carry that caveat.
  - Adding RQ4 is **your decision**: it adds scope (more models,
    per-label metrics).

## 16. Limitations
- **One dataset.** Every conclusion is limited to MIMIC's annotation
  scheme. Annotation guidelines and inter-annotator agreement per
  sub-category are **TO BE VERIFIED** in the paper (the paper reports
  Krippendorff's alpha; values not checked here).
- **Inferred value meaning.** The meaning of the sub-label values (1 =
  present) is inferred from the dataset card's task description and the
  perfect nesting under `Misogyny = 1`. The card does not state the value
  format explicitly.
- **Judgment calls.** The sufficiency tiers are a documented project
  judgment.
- **No evidence yet that stage 2 is learnable.** That needs a later
  validation-only experiment.

## 17. Final recommendation
1. **Keep the binary misogyny task as the primary task.** Task 8 results
   are unchanged.
2. **Do not build** a general safe/harmful or multi-harm system: the
   annotations do not exist.
3. **Optionally add stage 2** as a **multilabel** misogyny sub-category
   task (Objectification, Prejudice, Humiliation), with Humiliation
   flagged WEAK. This also enables a narrowed **RQ4**.
4. Show users **"misogyny detected / not detected"**, never "safe".
   Moderation outputs are limited to ALLOW / WARNING / HUMAN REVIEW,
   after calibration and threshold validation.

**NEXT TASK: TO BE DECIDED AFTER REVIEW OF MULTICATEGORY_FEASIBILITY_REPORT.md**
