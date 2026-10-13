# Application Integration Report (Task 16)

Date: 2026-10-13 · application v1.0.0

**Run locally:**

```bash
python -m streamlit run app/app.py
```

The app opens at `http://localhost:8501`.

**Code:**
- `app/` (Streamlit UI only);
- `src/application/`: `AnalysisService`, `HistoryService`, validation, artifact audit, research loaders, safe logging;
- `scripts/app_e2e.py`;
- `tests/test_application.py`.

**Configuration:**
- `config/app.yaml`, plus `config/ocr.yaml` and `config/explainability.yaml` (unchanged);
- `.streamlit/config.toml`;
- `.gitattributes`.

> **Frozen models only.** No model was trained, tuned or changed; the Task 13 TEST metrics are untouched.
> - The app loads the final models named in `config.yaml` `final_models`, and checks their SHA-256 against the Task 13 freeze manifest at start-up.
> - `verify_freeze()` passes (34 checksums).
> - The end-to-end tests use **synthetic, safe inputs** (an invented sentence and a generated meme): this is application testing, not research evaluation.

---

## 1. Architecture

```
Streamlit (app/)                       UI only: pages, components, cached access to the services
  app.py + pages/ + components/ui.py   no model, OCR or PyMongo code
  services/state.py                    st.cache_resource: audit, AnalysisService (models loaded once), HistoryService
        │
src/application/                       service layer (reusable by a future API)
  service.py      AnalysisService.analyze_text / analyze_meme / explain_text / explain_meme
  history.py      HistoryService: consent-based history + feedback via the Task 2 repositories
  validation.py   text + upload validation      artifacts.py  integration audit (no substitution)
  research.py     stored results for the dashboard   logging_setup.py  safe logging
        │
existing, validated components (unchanged)
  src/preprocessing (Task 6, frozen)   src/inference (Tasks 9-11)   src/ocr (Task 14)
  src/explainability (Task 15)         src/database (Task 2; two optional fields added)
```

**Integration matrix (16.1)**

The start-up audit `src.application.artifacts.audit()` produces this matrix. It is also shown in the app under Home → System status.

| Component | Implementation | Artifact / version | Status | Dependency | Fallback |
|---|---|---|---|---|---|
| FINAL_TEXT_MODEL | `src.inference.text.predict_text` | T05 fine-tuned MuRIL (sha256 `697692cc1d9e`) | available | checkpoint in `$NLP_CLD_MODEL_DIR` | none: Analyze Text disabled |
| FINAL_IMAGE_MODEL | `src.inference.image.predict_image` | I03 SigLIP + LR (`73c5497e95da`) | available | SigLIP weights (HF cache) | none: Analyze Meme disabled |
| FINAL_MULTIMODAL_MODEL | `src.inference.multimodal.predict_multimodal` | M04 fusion MLP (`5b132e127981`) | available | T05 + SigLIP | none: Analyze Meme disabled |
| second-opinion text model | `predict_text` | B07 TF-IDF + LR (`a2aaae391288`) | available | `models/text` | none |
| SigLIP encoder weights | `load_encoder(backend="pil")` | `google/siglip-base-patch16-224@7fd15f06` | available | HF cache (offline) | none |
| label mapping | 0 non-misogynistic / 1 misogynistic | freeze manifest | available | manifest | none |
| Task 6 preprocessing | `src.preprocessing.preprocess` | v1.0 (HINGLISH_NORMALIZED for T05/M04) | available | frozen | none |
| OCR pipeline | `src.ocr.pipeline.analyze_meme_input` | EasyOCR 1.7.2, `["en", "hi"]`, preparation `none` | available | `~/.EasyOCR/model` (285 MB) | Analyze Meme disabled |
| XAI pipeline | `src.explainability` | IG / occlusion / ablation (config v1.0) | available | final models | prediction shown without explanation |
| moderation | `src.explainability.moderation.recommend` | rules in `config/explainability.yaml` | available | labels | none (deterministic) |
| MongoDB repositories | `src.database.repositories` via `HistoryService` | `MONGODB_URI` set | available | Atlas (network, IP allow-list) | analysis works; history / feedback disabled |
| research results | `src.application.research` | 32 / 32 stored tables / figures | available | `results/` | missing items shown as "unavailable" |

**Missing-component fixes (minimal):**
1. **History schema.** Four optional, validated fields (`moderation`, `ocr`, `app_version`, `source`), plus `final_label` / `since` filters in the repository. Older documents stay valid.
2. **Application config.** `config.yaml` is checksum-frozen, so the app settings live in `config/app.yaml`. The stale `application:` section of `config.yaml` (`ocr_engine: null`) cannot be edited; `config/app.yaml` is authoritative.
3. **`.gitattributes`.** Pins line endings so checkouts reproduce the frozen bytes (§16 and §22).

## 2. Application pages
1. **Home:** problem statement, both modes, architecture diagram, supported categories (from `config/categories.yaml`), responsible use, system status.
2. **Analyze Text:** described in §5.
3. **Analyze Meme:** described in §6.
4. **Analysis History:** saved analyses only (with consent); filters by type, final prediction and date; details view; feedback.
5. **Research Dashboard:** six tabs:
   - dataset;
   - development (VALIDATION);
   - final comparison (TEST) with confusion matrices;
   - ablations;
   - OCR robustness;
   - error analysis & XAI.

   Every number is read from `results/` (32 stored files); nothing is typed by hand.
6. **Feedback / About:**
   - how feedback works;
   - objective, architecture and models (read from config);
   - dataset;
   - research questions (from `research_question_answers.csv`);
   - methodology, limitations, responsible use.

The sidebar shows whether each mode is ready and whether MongoDB is connected, plus a reminder that scores are uncalibrated.

## 3. Unified inference service
`AnalysisService` (`src/application/service.py`) orchestrates the existing code and returns one plain, JSON-serialisable result object (schema 1.0). It is shared by Streamlit and suitable for a future API.

```
analysis_type, status (ok | limited | error), error {category, message}, app_version,
input {original_text / normalized_text | image_metadata + ocr {status, text, normalized_text, confidence, route, ...}},
prediction {model_version, display, label, score, score_type, calibrated: false, source},
category {task, label, subcategories}, models {text, image, multimodal, second_opinion},
modality {agreement, disagreeing, dependence, ...}, moderation {recommendation, rule_id, reason, wording},
explanation {text | image}, warnings [], timings_ms {}, model_versions {}, disclaimer
```

- **No model logic is re-implemented.** The text goes through each model's own frozen preprocessing inside `predict_text()` / `predict_multimodal()`. The *displayed* normalised text is produced by the same `preprocess_all()` call with the same mode; tests check it equals `preprocess(..., "hinglish_normalized")`.

## 4. Model loading
- **Loaded once.** `services/state.get_service()` is an `st.cache_resource`. It builds `AnalysisService` once per server process and calls `warm_up()`, which loads T05 + B07, I03 + SigLIP, M04 and the EasyOCR reader. The inference modules keep the objects cached (`lru_cache`), so later clicks never reload a model.
- **Checked against the freeze.** `artifacts.audit()` compares each final artifact's SHA-256 with `results/final/model_freeze_manifest.json`.
- **Never substituted.** A missing or changed artifact makes `require(mode)` raise `ArtifactUnavailableError`. The page shows a safe error naming the missing components, and *no other model is substituted* (tested).

## 5. Text workflow
```
input validation -> Task 6 preprocessing (T05's mode) -> T05 (+ B07 second opinion)
  -> Integrated Gradients word attributions -> moderation (text-mode rules) -> optional history (consent)
```

**Validation** (`validate_text`):
- **Rejected with short messages:**
  - empty or whitespace-only input;
  - more than 2,000 characters;
  - no Latin or Devanagari letter (emoji, digits or symbols only);
  - non-string input.
- **Cleaned:** control characters are removed and Unicode is NFC-normalised.
- **Accepted:** Hindi, Hinglish, emojis and URLs. The frozen preprocessing handles them (URL → placeholder).
- **Warnings:**
  - text in another script (results may be unreliable);
  - text longer than T05's 64-token window (the rest is not analysed).

**Display:**
- original and normalised text;
- prediction ("Potentially misogynistic" / "Not flagged as misogynistic") with the category and the uncalibrated score;
- moderation, highlighted words, model versions and timings.

## 6. Meme workflow
```
upload (JPG/PNG <= 10 MB, in memory) -> validate (decode, format, size, pixels) -> OCR (Task 14)
  -> fallback route -> T05 on OCR text | I03 on the image | M04 on both  -> agreement + ablation
  -> moderation (meme-mode rules) -> [Explain prediction] -> optional history (consent)
```

- **The multimodal prediction is the main result.**
- **Model comparison:** the three outputs are shown side by side, labelled with their input:
  - text model: "OCR text only";
  - image model: "image only (no text)";
  - multimodal model: "image + OCR text".
- **OCR display:** OCR status, confidence, route, raw and normalised OCR text.

## 7. OCR integration
The Task 14 pipeline is reused unchanged: EasyOCR 1.7.2, preparation `none`, and the pre-declared fallback:

| OCR outcome | What the app does |
|---|---|
| text found, confidence ≥ 0.30 | all three models; multimodal result |
| no text / empty after preprocessing / OCR exception | **image-only** fallback (I03); status `limited`; "the caption could not be analysed" |
| mean confidence < 0.30 | image-only + **HUMAN REVIEW** |
| unreadable image | rejected with "please upload another file" |

**Empty or unreliable text is never sent to the multimodal model, and no text is ever invented.** All four branches are tested.

## 8. XAI integration
- **Text:** T05 Integrated Gradients (32 steps), computed automatically (about 1 s). Words are coloured red (towards "potentially misogynistic") or blue (against). Values appear on hover; the strongest words and the completeness error are listed.
- **Image:** I03 occlusion on a coarse 4 × 4 grid (9 occluded copies), behind an **Explain prediction** button, shown as an overlay. The overlay caption says the image model can also react to text printed in the picture.
- **Multimodal:**
  - text / image / multimodal comparison;
  - modality agreement (AGREE / PARTIAL / DISAGREE);
  - a train-mean ablation summary ("label depends on the image / text").

  No percentage contribution is shown.
- **Disclaimer on every result page:** "Explanations show model attribution and may not reflect human reasoning or causality."
- **Explanation failures are non-fatal:** the prediction is still shown (tested).

## 9. Multicategory behavior
Categories are derived from `config/categories.yaml`, never hard-coded:
- **Shown:** **Misogyny** (binary, enabled).
- **Shown as "not available: no trained model":** Objectification, Prejudice and Humiliation. They are annotated in the dataset, but stage 2 is disabled.
- **Listed under "not detected":** SAFE, HATE, ABUSE, THREAT, OBSCENE and SARCASM, each with its reason.
- **Label 0** is displayed as "not flagged as misogynistic", never "safe".

## 10. Moderation recommendations
The Task 15 design is used without change (`recommend()`, rules in `config/explainability.yaml`):
- outputs ALLOW / WARNING / HUMAN REVIEW; there is **no automatic block**;
- labels, agreement and the OCR route only; **no confidence thresholds**.

| Mode | Rules |
|---|---|
| meme | T05, I03 and M04 all flag → WARNING; none flags → ALLOW; any disagreement → HUMAN REVIEW; low OCR confidence → HUMAN REVIEW; image-only fallback: flagged → HUMAN REVIEW, not flagged → ALLOW |
| text | T05 and B07 both flag → WARNING; neither flags → ALLOW; disagreement or empty text → HUMAN REVIEW |

**Wording:**
- "Potentially misogynistic content (model prediction)…";
- "Flagged for human review…";
- "No content is blocked automatically."

## 11. MongoDB integration
- **Through the Task 2 layer only.** `HistoryService` uses the Task 2 repositories; `app/` contains no PyMongo calls.
- **Lazy connection:** the connection is created only when needed, from `MONGODB_URI` / `MONGODB_DATABASE` (`.env`). The status is cached for 60 s for the sidebar.
- **Stored (with consent only):**
  - `analysis_id`, `analysis_type`, `created_at`, `processing_time_ms`;
  - input: text and normalised text, or OCR text, normalised OCR text and image metadata;
  - predictions `{label, score}` for each model and the final one;
  - `model_versions`;
  - `moderation`;
  - meme `ocr` status;
  - `app_version`, `source`.
- **Never stored:** image binaries, `image_reference`, explanations or tensors.
- **Database failures** (connection, timeout, validation) return `{"ok": False, "message": <safe text>}`. The prediction stays on screen with "…The prediction above is still valid." Messages never contain the URI, hosts or credentials (tested with a fake URI inside an exception).

## 12. History
- **What is listed:** the last 50 analyses (newest first) as a table of time, type, prediction, category, moderation, model versions and source, with no content.
- **Filters:** type, final prediction and date (from).
- **Details:** per-model predictions with uncalibrated scores, moderation, OCR status and processing time. Stored text sits behind a "may contain offensive content" expander.
- **Without MongoDB:** the page shows "History is unavailable" and does not crash (tested).

## 13. Feedback
- **Form:** Correct / Incorrect plus an optional comment (≤ 500 characters). It appears after a saved analysis and on the History details view.
- **Stored** in `user_feedback`: `feedback_id`, `analysis_id`, `rating`, `comment`, `created_at`. Feedback for a missing analysis is refused.
- **Feedback never retrains anything;** future retraining stays a separate, controlled research process.

## 14. Research Dashboard
- **Read-only:** `src.application.research.ARTIFACTS` lists 32 stored tables and figures (Tasks 5–15).
- **Each section labels VALIDATION or TEST** and cites its source file.
- **Aggregates only:** files with dataset text are not listed (e.g. EDA term lists, `*_with_text.csv`).
- **Missing files** show "result file unavailable" (tested); nothing is recomputed.

## 15. Privacy
- **Uploads stay in memory.** `UploadedFile.getvalue()` bytes are kept only in the browser session's state, for the on-demand explanation. No temporary file is written, nothing is stored in MongoDB, and the OCR research cache is not used for uploads.
- **Text and OCR text** are stored only with explicit per-analysis consent (off by default).
- **Images are never stored:** only width, height, format, mode and size; EXIF is not stored.
- **Logs** (`logs/app.log`, git-ignored) contain only whitelisted fields: event, analysis type, model versions, route, moderation rule, duration and error *category*. User text, OCR text, images and exception messages are never logged (tested).
- **Usage statistics are disabled:** `browser.gatherUsageStats = false`.

## 16. Security
- **Secrets only from the environment** (`.env`, git-ignored; `.env.example` holds names only). No secret appears in the code or configs; a secret scan of the staged files is clean.
- **Safe errors:** `client.showErrorDetails = "type"`, so no stack traces reach the browser. Service errors use fixed messages plus an exception *type*.
- **Uploads:**
  - the format is decided by decoding (JPEG/PNG only), never by the file name;
  - the name is never used as a path;
  - 10 MB limit (Streamlit `server.maxUploadSize` + service check);
  - 40 MP pixel limit; PIL's decompression-bomb guard stays active;
  - corrupt or truncated files are rejected.
- **XSS protection:** highlighted words are HTML-escaped; user text is rendered with `st.text` (no Markdown interpretation). XSRF protection is on.
- **Integrity:**
  - the app verifies the frozen artifacts' checksums at start-up;
  - `.gitattributes` keeps the frozen files' exact bytes on every checkout (§22).

## 17. Performance
**[MEASURED]**
- **Hardware:** Intel Core i5-13500H (12 threads), 15.7 GB RAM, CPU only, Windows 11, Python 3.10.0, torch 2.14.1+cpu.
- **Power:** the laptop was **on battery** during the repeated measurements (Windows battery status "discharging"), and the CPU is slower then. The first end-to-end run was faster (OCR 2.5 s, meme analysis 3.3 s), so expect roughly 2–4× better times on mains power.
- **Sources:** `results/tables/app/app_performance.csv` (end-to-end run) and `app_latency_repeated.csv` (repeats, no database writes).

| Measure | Value |
|---|---|
| import of the application modules | 4.0 s |
| start-up audit incl. SHA-256 of every final artifact (about 2 GB) | 7.7 s |
| model loading, once per process: T05 + B07 8.6 s, I03 + SigLIP 1.9 s, M04 5.0 s, EasyOCR reader 7.0 s | 22.4 s |
| **cold start (first page fully ready)** | **about 34–40 s** |
| text analysis (T05 + B07 + moderation), warm | **0.07 s** median (0.06–0.16, 6 runs) |
| text explanation (T05 IG, 32 steps) | 2.2 s median (1.8–3.4) |
| **meme analysis total** (validation + OCR + T05 / I03 / M04 + ablation + moderation) | **12.2 s** median (11.6–12.6, 6 runs) |
| … of which OCR (EasyOCR) | 9.7 s median (8.9–10.0) |
| meme explanation on demand (occlusion 9 copies + IG on the OCR text) | 7.9 s median (7.6–8.2, 4 runs) |
| Streamlit page render after models are loaded | under 5 s (includes the cached audit) |

**Caching:**
- models, the audit and the history service are cached per process (`st.cache_resource`, `lru_cache`);
- the database status is cached for 60 s;
- no model is reloaded on a click.

**Which explanations are real-time:**
- the text explanation is fast enough to compute automatically;
- the image explanation is on demand, using the coarse 9-copy occlusion map instead of the 50-copy research map from Task 15;
- attribution *through* the fusion model (Task 15) is not used in the UI, because it is too slow on CPU.

## 18. Failure handling

| Failure | Behaviour (all tested) |
|---|---|
| missing / changed model artifact | the affected mode is disabled with a safe message; no substitution |
| MongoDB unavailable | analysis works; "history could not be saved"; History page shows a warning |
| OCR exception / no text / low confidence | image-only fallback (limited) or HUMAN REVIEW |
| corrupt / unsupported / oversized image | "please upload a JPG or PNG …" (no crash) |
| empty / too long / symbol-only text | short validation message |
| inference exception | "The model failed (RuntimeError). No prediction is shown." (no internals) |
| XAI failure | prediction shown; "explanation could not be computed" |
| missing research file | dashboard item shows "unavailable" |

## 19. Testing
- **`tests/test_application.py`: 37 tests.** Models are mocked (no download), OCR uses a fake reader, MongoDB uses `fake_db`; the real preprocessing, validation, moderation and schemas run. They cover:
  - text / meme service, preprocessing reuse, OCR integration and every fallback route;
  - text / image / multimodal paths, moderation, persistence with and without consent, MongoDB failure, feedback;
  - invalid uploads, empty input, result schema and model-version reporting;
  - missing artifacts and safe logging;
  - **headless Streamlit tests** (`streamlit.testing.v1.AppTest`): every page renders; Analyze Text with consent and feedback; Analyze Meme upload → analyze → explain; dashboard with a missing file; History with the database down.
- **Supporting tests:** `test_db_schemas.py` / `test_db_repositories.py` cover the new fields; `requirements-ci.txt` gains `streamlit` for CI.
- **End-to-end:** `scripts/app_e2e.py`, with real frozen models and real Atlas: **27/27 checks passed** (`results/tables/app/app_e2e_checks.csv`).
  - **Integrity:** checksums match the Task 13 freeze; `verify_freeze()` passes; MongoDB is reachable.
  - **TEXT:** schema complete; T05 used; normalised text equals Task 6 `preprocess()`; IG computed; moderation valid; saved, read back, feedback stored; nothing saved without consent.
  - **MEME:**
    - OCR read the printed caption;
    - T05 / I03 / M04 all ran, with M04 as the primary result;
    - the image model used no text;
    - moderation and agreement present; explanation computed;
    - saved with image metadata only, plus OCR status and moderation; feedback stored.
  - **Edge cases:**
    - a meme without text → image-only fallback;
    - a corrupt upload is rejected.
  - **Streamlit with the real services:**
    - all six pages render;
    - Analyze Text shows the FINAL ANALYSIS;
    - Analyze Meme shows the OCR text, three models and moderation;
    - Explain prediction renders the heatmap.
  - **Records:** the test records are tagged `source = "e2e_test"`. Two text and two meme records came from the last two runs; the first run stopped at the freeze check before writing anything (§22).
- **Browser smoke test:** Home, Dashboard and History render with no exceptions. Mouse clicks could not be automated in the built-in browser pane, which did not draw; interaction was therefore tested headlessly instead.

## 20. Known limitations
- **The models' known shortcuts reach the app** (Task 15):
  - the text model over-reacts to words referring to women and to feminine verb forms;
  - the image model reacts to pictured women.
  - Example: the harmless synthetic sentence "Yaar ye movie bahut acchi thi …" was flagged *Potentially misogynistic* (WARNING), because both text models react to it. The UI states this limitation; nothing is changed in the models.
- **Scores are uncalibrated;** the image model is overconfident.
- **Only binary misogyny is supported;** there are no sub-categories.
- **CPU-only:**
  - first start takes about 35–40 s (imports + checksum audit + model loading);
  - a meme analysis takes about 12 s on battery power, mostly OCR (≈ 10 s). On mains power, the first run measured 2.5 s of OCR and Task 14 measured 4.8 s per dataset meme. OCR is the bottleneck.
- **History** has no retention policy or authentication yet (Task 17 decisions); history is local to one MongoDB database.
- **Single-user local app:** models are shared per process, with no request queue or rate limiting.
- **Images can't be re-explained later** (they are never stored); the explanation works only within the session.

## 21. Deployment readiness
- **Ready for local demonstration and examiner use.** All components are integrated, verified against the freeze and tested.
- **Not yet ready for public deployment (Task 17):**
  - a hosting target with ≥ 4–6 GB RAM for the models (memory was not measured precisely);
  - model artifacts must be provisioned outside git (MuRIL checkpoint about 950 MB, SigLIP about 800 MB, EasyOCR 285 MB);
  - secrets management on the host (no `.env`);
  - a retention policy, authentication / rate limiting if public, HTTPS;
  - a production logging target;
  - a cold-start strategy;
  - optionally a GPU or a smaller OCR / image model for latency.

## 22. Integrity incident during Task 16 (resolved)
**What happened:**
- After the PR merges, branch switches rewrote tracked files with CRLF line endings, because of git's `core.autocrlf=true` on this machine.
- The Task 13 `verify_freeze()` then failed, since it hashes exact file bytes.

**Investigation:**
- **No content changed.** `git diff ed0de53 HEAD` over every frozen path is empty, and the stored split file hashes to the canonical `47fb02a87b36…`.
- **The freeze had mixed line endings.** By matching the manifest hashes, the original bytes were identified: six files were CRLF at freeze time, written by Python on Windows (`config.yaml`, the split manifest, the three registries and `src/models/image_encoder.py`); every other file was LF.

**Fix:**
- `.gitattributes` uses LF by default and pins those six files to CRLF.
- The working tree was restored. `verify_freeze()` passes again (34/34), and git records no content change.

**Lesson for future tasks:** never re-save frozen files. Python's `write_text` on Windows writes CRLF.
