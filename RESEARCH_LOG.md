# RESEARCH_LOG.md — Experiment Record

This log is the single source of truth for every meaningful experiment
in this project. It exists so results can be reproduced, compared
honestly, and defended in the viva and the paper.

## How to use this log

1. **Every meaningful experiment gets a unique Experiment ID** before
   it is run. Suggested format: `EXP-YYYYMMDD-NN`
   (e.g. the first run on a given day is `-01`). Planned ablations can
   also reference their design ID (E1–E5, OCR) in the
   "Research Question" field.
2. Copy the template below, fill it in, and append it to the
   **Experiment Entries** section (newest at the bottom).
3. **Status** is one of: `PLANNED`, `RUNNING`, `COMPLETED`, `FAILED`,
   `ABANDONED`. Failed and negative experiments are recorded too.
4. **Only write numbers that were actually measured.** Copy metrics from
   the files the code produced (e.g. `results/metrics/...`) and link
   them under "Artifacts". Never estimate, round favourably, or fill in
   "expected" values.
5. Keep **Validation Metrics** and **Test Metrics** separate. Test
   metrics are only filled in for a final, frozen configuration; write
   `NOT EVALUATED` otherwise.
6. Write observations as observations ("macro-F1 on validation was
   lower than E1") and hypotheses as hypotheses ("this *may* be because
   ..."). Do not claim statistical significance without a test.
7. Do not edit a completed entry's results. If something was wrong,
   add a correction note with the date.

---

## Template

```markdown
### Experiment ID:

Date:

Status:

Research Question:

Dataset:

Dataset Version:

Dataset Source:

Train Split:

Validation Split:

Test Split:

Input Modality:

Preprocessing:

Feature Representation:

Model:

Model Version / Pretrained Checkpoint:

Hyperparameters:

Random Seed:

Training Environment:

Training Details:

Validation Metrics:

Test Metrics:

Runtime / Resource Notes:

Artifacts:

Observations:

Error Analysis Notes:

Limitations:

Conclusion:

Next Experiment:
```

---

## Experiment Entries

_No experiments have been run yet._
