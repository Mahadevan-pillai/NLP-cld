# Hyperparameter Tuning Report (Task 12)

Date: 2026-10-13 · Tuning v1.0

Code:
- `src/training/tuning.py`: grid / random search, trial log, selection, seed summary
- `scripts/tune_models.py`: `summary`, `text`, `image`, `multimodal`, `report`
- `scripts/train_text_transformers.py finetune --run-id … --learning-rate … --out tuning`: MuRIL learning-rate sweep
- `scripts/report_tuning.py`: merged trial log, best configurations, before/after table, figures, registries, test-access audit
- `tests/test_tuning.py`

Results:
- `results/tables/tuning/`
- `results/figures/tuning/`
- `results/metrics/tuning/` (tuned-model JSON and validation predictions; no text)

> **TEST SET NOT EVALUATED IN TASK 12.** Every search fits on **train** and
> selects on **validation** (754 samples). All numbers are validation
> results. Because many configurations were compared on the same
> validation set, the selected scores are **optimistic**; only the later,
> single test evaluation is an unbiased estimate.

Tags:
- **[BASELINE]**: the Task 8–11 configuration.
- **[TUNED RESULT]**: measured after the search.
- **[INTERPRETATION]**: our reading of the measurements.

---

## 1. Objective
Show a defensible, compute-aware tuning methodology (MSE-2 "Hyperparameter
Tuning"):
- tune only the models that matter for the final comparison;
- use a fixed protocol: train → fit, validation → select, test → locked;
- record every trial;
- check whether gains survive random-seed variation.

Primary metric: **validation macro-F1**.

## 2. Models considered
`results/tables/tuning/pre_tuning_summary.csv` gathers the existing
validation results; nothing was re-trained for it.

| Family | Best before tuning | Validation macro-F1 | Others |
|---|---|---|---|
| Traditional text | **B03** TF-IDF + LR, HINGLISH_NORMALIZED | 0.6671 | B01–B06: 0.650–0.667 |
| Pretrained text | **T05** fine-tuned MuRIL, HINGLISH_NORMALIZED | 0.6869 | T01–T03 frozen 0.615–0.630; T04 0.6751 |
| Image-only | **I01** frozen SigLIP + LR | 0.7586 | I02 SVM 0.7507 |
| Multimodal | **M02** fusion MLP (main model, declared in Task 11) | 0.8076 | M03 0.8077 (control); M01 0.7679 |

M03 is 0.0001 above M02, which is meaningless. M02 remains the main model,
as fixed in Task 11.

## 3. Models selected for tuning
| Target | Model | Tuned | Why |
|---|---|---|---|
| A | B03 TF-IDF + LR | TF-IDF settings + C (grid) | Best traditional baseline; cheap |
| B | T05 fine-tuned MuRIL | **Learning rate only** (2 runs) | CPU fine-tuning costs about 30–45 min per run. The developer chose a 2-run learning-rate sweep; T04/T05 learned slowly in epoch 1 at 2e-5 |
| C | I01 SigLIP + LR | Classifier type + C on **cached** embeddings | Cheap; I01 was overconfident at C = 1 (Task 10) |
| D | **M02 fusion MLP** | 6 MLP hyperparameters (random search) + seeds | **Primary target**: the main research model |

**Not tuned** (intentionally):
- B01, B02, B04–B06, T01–T04, I02 as separate families, M01 and M03: they
  are not the best of their family.
  - I02's classifier (Linear SVM) is still searched inside target C.
  - The M03 frozen-text variant is re-checked with the tuned configuration
    (§12).
- **Encoders:** neither MuRIL's tokenizer or architecture nor SigLIP
  fine-tuning was tuned. SigLIP fine-tuning was deferred to a GPU in
  Task 10.
- **Preprocessing:** fixed at v1.0, as the task requires.

## 4. Compute budget
**Hardware [MEASURED]:**
- Intel i5-13500H, 12 PyTorch threads, **no CUDA GPU**, 15.7 GB RAM.
- Only about 2.8 GB of RAM was free at the start, because a VirtualBox VM
  was running. The developer closed it during the sweep (§16).

Per-trial cost was measured **before** the searches:

| Target | Cost per trial | Trials planned | Estimate | Actual (sum of trial times) |
|---|---|---|---|---|
| A TF-IDF + LR | 0.25–0.53 s | 90 + baseline | < 1 min | 72.8 s |
| C image classifier | 0.26–0.88 s | 16 + baseline | < 1 min | 49.8 s |
| D fusion MLP | about 5–20 s | 30 + baseline + 7 seed runs | about 5–10 min | 272.8 s (4 threads, while B was running) |
| B MuRIL fine-tuning | about 30–45 min (Task 9, idle CPU) | 2 runs | about 1–1.5 h | T06 29.8 min (2 epochs), T07 25.2 min (3 epochs); both stopped early |

No search was launched that was expected to take many hours. The only
expensive target (B) was limited to two runs.

## 5. Search strategy
- **A and C: small grid search.** Every combination of a compact,
  pre-declared space.
- **D: randomized search.** **30** configurations sampled without
  replacement, with seed 42, from a 486-point grid, plus the baseline
  configuration. Then **seed-robustness runs** (§12).
- **B: one-factor sweep.** The learning rate is the single changed
  setting; everything else is as in T05.
- **Fixed conditions (12.8)**, identical for all trials:
  - dataset, canonical split (SHA-256 `47fb02a87b36…`) and labels;
  - text preprocessing v1.0 (HINGLISH_NORMALIZED for A, B and D);
  - MuRIL `afd9f36c` and SigLIP `7fd15f06` encoders;
  - cached features;
  - metric (validation macro-F1);
  - primary seed 42.
- **Baseline reproduction check.** Each search first re-ran its
  **baseline** configuration through the tuning harness, and stopped if
  the stored score was not reproduced **exactly**:
  - TT-000 = B03 0.6671 ✅
  - TI-000 = I01 0.7586 ✅
  - TM-000 = M02 0.8076 ✅

## 6. Search spaces
All declared in `config.yaml` `tuning` **before** running.

| Target | Parameter | Values |
|---|---|---|
| A | `ngram_range` | (1,1), (1,2), (1,3) |
| A | `min_df` | 1, 2, 3 |
| A | `sublinear_tf` | true, false |
| A | LR `C` | 0.1, 0.3, 1, 3, 10 |
| A | fixed | `max_df` 1.0, `max_features` none, `class_weight` none (classes balanced), Devanagari-safe token pattern |
| B | learning rate | **3e-5 (T06)**, **5e-5 (T07)** vs baseline 2e-5 (T05); batch 16, max 4 epochs, patience 1, warmup 10%, word embeddings frozen |
| C | classifier | logistic regression `C` ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3}; linear SVM `C` ∈ {0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1} |
| D | learning rate | 3e-4, 1e-3, 3e-3 |
| D | dropout | 0.1, 0.3, 0.5 |
| D | hidden dim | 128, 256, 512 |
| D | projection dim | 128, 256 |
| D | weight decay | 1e-3, 1e-2, 1e-1 |
| D | batch size | 32, 64, 128 |

**Early stopping (D), defined before training:**
- monitor validation macro-F1, patience 5, minimum improvement 0.0,
  maximum 50 epochs;
- identical to Task 11, so baseline and tuned runs are comparable;
- the best epoch is restored and recorded.

## 7. Selection metric
- **Validation macro-F1.** Ties go to the earlier trial (deterministic).
  Only COMPLETED or EARLY_STOPPED trials can be selected.
- **Multimodal adoption rule** (declared in `config.yaml`): adopt the
  best seed-42 configuration only if its **mean over seeds 42, 123 and
  2026** exceeds the baseline configuration's mean.
- Seeds were **never** used to pick a lucky run. The saved model always
  uses the primary seed 42.

## 8. Trial results
`results/tables/tuning/hyperparameter_trials.csv` holds all trials, with
ID, search, model, modality, parameters, seed, epochs, best epoch,
validation accuracy / macro-F1 / weighted-F1, train macro-F1, training
time, status and note.
- Per-search logs: `trials_traditional_text.csv`, `trials_image.csv`,
  `trials_multimodal.csv`.
- Fusion histories for every trial: `multimodal_trial_histories.csv`.

**[TUNED RESULT]**

| Search | Trials | Status | Validation macro-F1, min / median / max |
|---|---|---|---|
| A TF-IDF + LR | 91 (incl. baseline) | 91 COMPLETED | 0.5962 / 0.6512 / 0.6684 |
| C image classifier | 17 (incl. baseline) | 17 COMPLETED | 0.7507 / 0.7652 / 0.7997 |
| D fusion MLP | 31 (incl. baseline) | 31 EARLY_STOPPED | 0.8050 / 0.8101 / 0.8183 |
| D seed robustness | 7 | 7 EARLY_STOPPED | §12 |
| B MuRIL learning rate | 2 (+ T05 reference) | 2 EARLY_STOPPED, **both degenerate** | T06 0.3322, T07 0.3345 (T05 reference 0.6869) |

- No trial failed.
- A **FAILED** status would have been logged, not dropped (tested in
  `tests/test_tuning.py`).

**Figures** (`results/figures/tuning/`):
- `param_vs_macro_f1.png`: A and C;
- `multimodal_trials.png`: D, all trials sorted;
- `finetune_lr_sweep.png`: B;
- `seed_robustness.png`;
- `M04_training_curves.png`;
- `before_after_tuning.png`.

## 9. Best parameters
`results/tables/tuning/best_hyperparameters.csv`

| Family | [BASELINE] configuration → macro-F1 | [TUNED RESULT] best configuration → macro-F1 | Change |
|---|---|---|---|
| Traditional text (B03 → **B07**) | 1–2-grams, min_df 2, sublinear, C 1 → 0.6671 | **1–3-grams**, min_df 2, sublinear, C 1 (TT-073) → **0.6684** | +0.0013 (+0.19%) |
| Pretrained text (T05 → best sweep run T07) | lr 2e-5 → 0.6869 | lr 5e-5 (T07) → 0.3345; lr 3e-5 (T06) → 0.3322: **both degenerate** | −0.3524: **no usable improvement; T05 kept** |
| Image (I01 → **I03**) | LR C 1.0 → 0.7586 | **LR C 0.001** (TI-001) → **0.7997** | **+0.0411 (+5.4%)** |
| Multimodal (M02 → **M04**) | lr 1e-3, dropout 0.3, hidden 256, proj 256, wd 1e-2, batch 64 → 0.8076 | **lr 3e-4, dropout 0.1, hidden 512**, proj 256, **wd 0.1**, batch 64 (TM-007) → **0.8183** (seed 42) | +0.0107 at seed 42; **+0.0022 by 3-seed mean** |

## 10. Before / after comparison
`results/tables/tuning/before_after_tuning.csv` (validation, seed 42; the
multimodal row also gives the seed means).

| Family | Model | Macro-F1 | Accuracy | Precision* | Recall* | Training time |
|---|---|---|---|---|---|---|
| Traditional text | B03 [BASELINE] | 0.6671 | 0.6671 | 0.6676 | 0.6587 | 0.25 s |
| | **B07** [TUNED] | 0.6684 | 0.6684 | 0.6694 | 0.6587 | 1.23 s (incl. preprocessing) |
| Pretrained text | T05 [BASELINE] | 0.6869 | 0.6870 | 0.6759 | 0.7120 | 33.1 min |
| | T06 [TUNING RUN] lr 3e-5 | 0.3322 | 0.4973 | 0.4973 | 1.0000 | 29.8 min: predicts **every** meme misogynistic |
| | T07 [TUNING RUN] lr 5e-5 | 0.3345 | 0.5027 | 0.0000 | 0.0000 | 25.2 min: predicts **every** meme non-misogynistic |
| Image | I01 [BASELINE] | 0.7586 | 0.7586 | 0.7533 | 0.7653 | 0.27 s (+ 11.9 min embedding) |
| | **I03** [TUNED] | **0.7997** | 0.7997 | 0.8043 | 0.7893 | 0.31 s (same embeddings) |
| Multimodal | M02 [BASELINE] | 0.8076 (3-seed mean 0.8059) | 0.8077 | 0.8194 | 0.7867 | 8.8 s |
| | **M04** [TUNED] | 0.8183 (3-seed mean 0.8081) | 0.8183 | 0.8199 | 0.8133 | 9.1 s |

\* Precision and recall of the misogynistic class.

**[INTERPRETATION]**
- **Traditional text: tuning did not improve B03 in any meaningful way.**
  - The +0.0013 is **one** more correct validation sample (one fewer
    false positive: TN 257 vs 256).
  - The Task 8 defaults were already at the top of a broad plateau: the
    median trial scores 0.651.
- **Pretrained text: tuning failed.** Both higher learning rates produced
  single-class models, so T05 (lr 2e-5) remains the best (§11, §13).
- **Image: a clear gain**, from much stronger regularisation (§13).
- **Multimodal: no meaningful gain.** See §12.

## 11. Neural learning curves
- **M04** (`M04_training_curves.png`, `M04_training_history.csv`):
  - validation loss is lowest at epoch 2 (0.426);
  - the selected epoch is 3 (macro-F1 0.8183, validation loss 0.442);
  - early stopping at epoch 8;
  - training loss falls from 0.435 to 0.125.
- **All fusion trials** (`multimodal_trial_histories.csv`): every one of
  the 31 search trials stopped early, after 7–12 epochs, with best epochs
  2–7.
- **MuRIL** (`finetune_lr_sweep.png`; histories in
  `T06_training_history.csv`, `T07_training_history.csv`):
  - T05 (lr 2e-5) learns from epoch 1, from 0.613 to 0.687 at epoch 3.
  - **T06 (3e-5) and T07 (5e-5) never leave the one-class plateau.**
    Training loss stays at about 0.693 (= ln 2, chance level), and
    validation macro-F1 stays at 0.332–0.335.
  - With patience 1 they were stopped after epochs 2 and 3.
  - T04 (Task 9, lr 2e-5) showed the same plateau in epoch 1 but escaped it
    in epoch 2.

## 12. Seed robustness
`seed_robustness.csv`, `multimodal_seed_robustness.json`,
`seed_robustness.png`. The same configuration is retrained with seeds
42, 123 and 2026. The seed changes weight initialisation and batch order
only.

| Configuration | Seed 42 | Seed 123 | Seed 2026 | Mean ± std |
|---|---|---|---|---|
| Baseline M02 configuration | 0.8076 | 0.7997 | 0.8103 | **0.8059 ± 0.0055** |
| Tuned TM-007 configuration (M04) | **0.8183** | 0.8129 | 0.7931 | **0.8081 ± 0.0133** |
| TM-007 configuration with **frozen** MuRIL text (M03 features) | 0.8050 | 0.8011 | 0.8024 | 0.8028 ± 0.0020 |

**[INTERPRETATION]**
- **Most of the seed-42 gain is selection optimism.** The best of 31
  configurations at one seed scored +0.0107 over the baseline, but over
  three seeds the difference is only **+0.0022**. That is far smaller
  than the tuned configuration's own seed standard deviation (0.0133).
- **Adopted, but not a real improvement.** The pre-declared rule (higher
  mean) adopts the tuned configuration. The improvement is **not
  meaningful**, and the tuned configuration is **less stable** across
  seeds than the baseline (std 0.0133 vs 0.0055).
- **The search landscape is flat.** All 31 configurations lie within
  0.805–0.818 at seed 42, a range comparable to seed noise. Only the
  learning rate shows a mild pattern: trials at 3e-4 average 0.8123,
  those at 1e-3 and 3e-3 about 0.809.
- **Frozen text is a stable alternative.** With the tuned configuration,
  frozen MuRIL features give a slightly lower mean (0.8028) with the
  lowest spread (0.0020).

## 13. Overfitting analysis
| Model | Train macro-F1 | Validation macro-F1 | Gap |
|---|---|---|---|
| B03 [BASELINE] (TT-000) | 0.8867 | 0.6671 | 0.22 |
| B07 [TUNED] (TT-073) | 0.8884 | 0.6684 | 0.22 |
| TF-IDF with min_df 1 and C ≥ 1 (18 trials) | 0.92–0.99 | ≤ 0.664 | ≥ 0.26 |
| I01 [BASELINE] C = 1 | 0.8992 | 0.7586 | 0.14 |
| **I03 [TUNED] C = 0.001** | 0.8494 | **0.7997** | **0.05** |
| M02 [BASELINE] (TM-000) | 0.8915 | 0.8076 | 0.08 |
| M04 [TUNED] (TM-007) | 0.9119 | 0.8183 | 0.09 |
| T05 [BASELINE] | 0.7477 | 0.6869 | 0.06 |
| T06 / T07 (degenerate) | 0.3327 / 0.3346 | 0.3322 / 0.3345 | ≈ 0 (**underfitting**: nothing learned) |

**[INTERPRETATION]**
- **Image:** the 768-d frozen embeddings with only 3,532 training images
  were **overfitted at C = 1**. Strong L2 regularisation trades training
  fit for generalisation; validation rises steadily as C decreases.
  - **The best value (0.001) is the edge of the pre-declared grid**, for
    both logistic regression and the SVM (SVM best 0.0003: 0.7984).
    Smaller C was **not** tested, to keep the protocol fixed. This is
    noted for the future.
- **TF-IDF:** with `min_df = 1` (keeping words seen only once, often OCR
  noise) and weak regularisation, train macro-F1 reaches 0.92–0.99.
  Validation does not improve: the mean is 0.6515 for min_df 1 vs 0.6522
  for min_df 2. The extra fit is memorisation, not generalisation.
- **MuRIL at higher learning rates: underfitting / collapse,** not
  overfitting. The train and validation scores are equally at chance.
  - [HYPOTHESIS] With word embeddings frozen, 10% warm-up and a 2-class
    head, a larger learning rate drives the model to a constant prediction
    from which it does not recover within the epochs allowed. This is a
    known kind of fine-tuning instability.
  - Untested here: other seeds, longer patience, or more warm-up.
  - **Patience 1** (the Task 9 setting) is fragile for this behaviour; it
    can stop a run during a plateau that it might later escape, as T04
    did.
- **Fusion MLP:** it still **overfits within 2–3 epochs** in every
  configuration (validation loss rises after epoch 2). The search did not
  change this: regularisation by dropout or weight decay within the
  searched ranges made no systematic difference.

## 14. Final development-model selection
Selection uses validation macro-F1, with stability, runtime, complexity
and reproducibility as secondary criteria. **None was evaluated on
test.**

| Role | Selected | Validation macro-F1 | Why | Alternative kept on record |
|---|---|---|---|---|
| **FINAL_TEXT_MODEL** | **T05** fine-tuned MuRIL, HINGLISH_NORMALIZED, lr 2e-5 (Task 9; unchanged) | 0.6869 | Best text model. The learning-rate sweep produced only degenerate runs, and tuned TF-IDF B07 is 0.0185 lower. T05 is also the text encoder inside the multimodal model | **B07** (0.6684): about 4,700× smaller (204 KB vs 957 MB) and about 800× faster, a reasonable fallback for the app |
| **FINAL_IMAGE_MODEL** | **I03** frozen SigLIP + LR, C = 0.001 | 0.7997 | Clear gain (+0.041) at identical cost and complexity to I01; less overfitted (train–validation gap 0.05). Deterministic: no seed variance | I01 (0.7586) |
| **FINAL_MULTIMODAL_MODEL** | **M04** fusion MLP, TM-007 configuration, seed 42 | 0.8183 at seed 42; **3-seed mean 0.8081 ± 0.0133** | Adopted by the **pre-declared** rule (mean > baseline mean 0.8059). The gain is **not meaningful** and the configuration is less stable; it is recorded rather than overridden after the fact | **M02** (baseline configuration, 3-seed mean 0.8059 ± 0.0055) is equivalent within noise |

**Serialization check (12.20) [MEASURED]:**
- Each selected model, plus B07, was reloaded from disk through its own
  inference interface (`predict_text`, `predict_image`,
  `predict_multimodal`), starting from raw validation captions and image
  files.
- Each reproduces its stored validation predictions **754/754**:

  | Model | Score difference | Time per sample |
  |---|---|---|
  | T05 | ≤ 1e-6 | 41 ms |
  | B07 | 0 | 1 ms |
  | I03 | ≤ 1e-6 | 251 ms |
  | M04 | ≤ 2e-6 | 290 ms |

- The required metadata travels with each artifact: model version,
  configuration, preprocessing version, encoder revisions, label mapping,
  seed and validation result. It is stored in the registries,
  `results/metrics/tuning/*.json` and the M04 `fusion_config.json`.

**[INTERPRETATION] for the final comparison (RQ2):**
- **Tuning the image baseline shrinks the multimodal advantage.**
  - Before tuning: fusion was +0.049 over image-only (M02 vs I01).
  - After tuning: the fusion's 3-seed mean (0.8081) is only **+0.008**
    above the tuned image model I03 (0.7997, deterministic). M04's seed-42
    score is +0.019 above I03.
- **Text-only remains far behind.** T05 is at 0.6869.
- **The RQ2 claim must therefore be stated cautiously** and settled by the
  single test evaluation in Task 13.

**Recommendation for Task 13** (not done here): evaluate each final model
**once** on test.
- For the multimodal model, consider reporting the TM-007 configuration at
  all three seeds as one pre-declared configuration (mean ± std), since
  single-seed results differ by up to 2.5 points.

## 15. Multicategory tuning status
- Task 8.5 kept the stage-2 sub-category task (Objectification /
  Prejudice / Humiliation) **disabled**. No sub-category model exists,
  so nothing auxiliary was tuned.
- **Status: NOT TRAINED, NOT TUNED.** This does not delay the main
  binary pipeline.

## 16. Compute limitations
- **CPU only.** Only the MuRIL learning rate was tuned, over 2 runs.
  - Batch size, epochs, weight decay, warmup and full (unfrozen-embedding)
    fine-tuning were not explored.
  - SigLIP fine-tuning remains deferred to a GPU.
- **A VirtualBox VM shared the machine** at the start of the MuRIL sweep
  (about 3 GB RAM, much of the CPU). T06's first epoch took 1,292 s
  instead of about 440 s. The developer then closed the VM.
  With the VM closed, MuRIL epochs took about 8 min: T06 needed 29.8 min
  for 2 epochs, T07 25.2 min for 3.
- **The fusion search ran with 4 threads in parallel with the MuRIL
  sweep.** Its trial times are therefore not comparable with Task 11's
  timings. The results are unaffected: the TM-000 baseline reproduced
  M02 exactly.

## 17. Test-set protection
**TEST SET NOT EVALUATED IN TASK 12.**
`results/tables/tuning/test_access_audit.csv` records each check:
- the Task 12 sources (`tune_models.py`, `report_tuning.py`,
  `tuning.py`) never contain `load_split("test"` or `allow_test=True`;
- no Task 12 output contains a test metric (`test_accuracy`,
  `test_macro_f1`, …);
- every tuned-model result says `test_set: NOT EVALUATED`;
- the canonical split is unchanged (manifest SHA-256 verified);
- no test-based threshold, early stopping or model selection: only train
  and validation are ever loaded.

The MuRIL sweep reuses the Task 9 training code, which is covered by its
own test-protection unit test.

## 18. Limitations
- **Selection optimism.** Choosing the best of many configurations on the
  same 754 validation samples inflates the selected scores; §12 shows
  this directly for the fusion model. The final test evaluation (Task 13)
  is the unbiased check.
- **Single split.** Validation CIs are about ±3 points, larger than most
  tuning differences.
- **Compact spaces.** The image optimum lies at the grid boundary, and the
  MuRIL sweep covers one parameter with two values. Both of those values
  failed to learn, with one seed and patience 1, so the sweep says little
  about the true learning-rate sensitivity of MuRIL.
- **Seed robustness** was measured for the fusion MLP only. TF-IDF and LR
  are deterministic given the data. The MuRIL runs use one seed.
- **Tuning does not remove the known shortcuts** (Tasks 9–11: empty
  caption, gender presence, watermark, resolution). They must be
  re-checked on the final models in Task 13.
