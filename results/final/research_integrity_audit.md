# Research Integrity Audit (Task 13, final TEST evaluation)

Date: 2026-10-13. Every check below was verified against files and git history, not asserted from memory.

| # | Requirement | Result | Evidence |
|---|---|---|---|
| 1 | Models frozen **before** TEST | ✅ PASS | `results/final/model_freeze_manifest.json` was created at 2026-10-13T09:35:40Z and committed in **`ed0de53`** ("Task 13 FREEZE", 09:37:29Z). The first TEST metric is timestamped 11:13:03Z. The manifest holds SHA-256 checksums of all 19 model artifacts, both encoders' weights, config files, the Hinglish rule file, preprocessing/model/feature/evaluation code and the split files |
| 2 | Evaluation code verified before TEST | ✅ PASS | Before freezing, the evaluation code re-scored VALIDATION and reproduced every stored validation result exactly: 19/19 metrics, and per-sample predictions where stored |
| 3 | TEST not used for tuning | ✅ PASS | No training, tuning or selection code ran after the freeze. `git diff ed0de53 -- src config.yaml config results/metrics data/splits scripts/final_evaluation.py` is **empty** |
| 4 | No architecture change based on TEST | ✅ PASS | Fusion / model code (`src/models`, `src/features`) is unchanged since the freeze (checksums re-verified by `verify_freeze` before `integrity`, `evaluate` and the report) |
| 5 | No preprocessing change based on TEST | ✅ PASS | `src/preprocessing` and the Hinglish rule file are unchanged since the freeze (checksums) |
| 6 | No threshold change based on TEST | ✅ PASS | The decision rule was frozen as argmax (LR probability > 0.5, SVM margin > 0, softmax argmax); no threshold appears anywhere in the evaluation code |
| 7 | Model selection not influenced by TEST | ✅ PASS | The finals (T05 / I03 / M04) were selected in Task 12 on validation and recorded in the manifest. They were **kept** although TEST shows T04 > T05 and I03 ≥ M04 |
| 8 | All reported numbers trace to prediction files | ✅ PASS | `results/tables/final/consistency_checks.csv`: for all 19 models, metrics recomputed from `results/predictions/final/*_test_predictions.csv` equal the metric JSONs; confusion totals = 758 = sample count. Unit tests check tables against the metric files |
| 9 | Validation and TEST clearly separated | ✅ PASS | VALIDATION results: `validation_before_after_tuning.csv` (`result_type` = VALIDATION DEVELOPMENT RESULT). TEST results: `main_results.csv` (`result_type` = FINAL TEST RESULT). The tuning-generalisation table keeps validation and test in separately named columns |
| 10 | Negative results reported | ✅ PASS | Reported, not hidden: the multimodal model does **not** beat image-only (RQ3 NOT SUPPORTED); Hinglish normalization does not help (RQ1 INCONCLUSIVE, 3 of 4 contrasts negative); the final text model T05 scores below T04 on TEST; TF-IDF tuning gave no TEST gain |
| 11 | No fabricated literature comparison | ✅ PASS | `existing_system_comparison_template.csv` contains **no** literature value: `reported_result` = TO_BE_VERIFIED, `comparison_valid` = NO |
| 12 | Test split opened only by the final evaluation | ✅ PASS | Only `scripts/final_evaluation.py` calls `load_split("test", allow_test=True)` (unit test). Task 7's `create_splits.py` creates the split assignment but evaluates no model |
| 13 | Statistical claims justified | ✅ PASS | "Significant" is used only for the two pre-specified primary comparisons, under the frozen rule (Holm-adjusted exact McNemar p < 0.05 **and** a paired bootstrap 95% CI excluding 0; 10,000 resamples, seed 42) |

## Disclosed deviations
1. **The evaluation command ran twice.** The first `evaluate` call was interrupted by the developer while running. It had written 12 prediction files (B01–T05) before a second, complete run was started.
   - Its partial predictions were set aside and compared: all 12 are **byte-identical** to the complete run.
   - All 19 metric files come from the single complete run (same timestamp 11:13:03Z).
   - The models are deterministic; no result was selected between runs.
2. **Figure-helper copy.** The TEST confusion-matrix figures use a copy of the frozen plotting helper with a TEST footnote. The frozen helper hard-codes "Validation set", and `src/evaluation` is checksum-frozen. The figures were changed; no numbers were.
3. **Reporting-code fixes after TEST.** Changes were made only to `scripts/report_final_evaluation.py`:
   - colours, labels and footnotes of figures;
   - computing the gain table from unrounded metrics (−0.0016 → −0.0015, now equal to the statistics table).

   No model, prediction or metric file was edited by hand.

## Not verifiable here
- Literature values for an external comparison (left TO_BE_VERIFIED).
- Annotation quality: possible label ambiguities were flagged in `error_analysis.csv`; labels were **not** changed.
