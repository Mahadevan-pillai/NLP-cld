"""Start-up audit of every component the application needs (Task 16 integration matrix).

Each row: component, implementation, artifact / version, status, dependency, fallback.
Status: "available", "missing", "checksum mismatch" or "warning". A required component that is not
"available" disables the affected mode with a safe message; the app NEVER substitutes another model.
Checksums are compared with the Task 13 freeze manifest (results/final/model_freeze_manifest.json).
"""

import json
from pathlib import Path

import yaml

from src.application.settings import final_models, load_app_config
from src.utils.config import REPO_ROOT, load_config

MANIFEST = REPO_ROOT / "results" / "final" / "model_freeze_manifest.json"
REGISTRIES = {kind: REPO_ROOT / "results" / "metrics" / f"{kind}_model_registry.json" for kind in ("text", "image", "multimodal")}
EASYOCR_WEIGHTS = ("craft_mlt_25k.pth", "devanagari.pth")
MODE_REQUIREMENTS = {"text": ("FINAL_TEXT_MODEL", "second-opinion text model", "Task 6 preprocessing", "label mapping"),
                     "meme": ("FINAL_TEXT_MODEL", "FINAL_IMAGE_MODEL", "FINAL_MULTIMODAL_MODEL", "Task 6 preprocessing",
                              "label mapping", "SigLIP encoder weights", "OCR pipeline (EasyOCR)")}


class ArtifactUnavailableError(RuntimeError):
    """A required model or component is missing; str(error) is safe to show to the user."""


def _registry_entry(kind: str, model_id: str) -> dict | None:
    try:
        return json.loads(REGISTRIES[kind].read_text(encoding="utf-8"))["models"].get(model_id)
    except (OSError, ValueError, KeyError):
        return None


def _artifact_path(entry: dict) -> Path:
    from src.inference.text import resolve
    return resolve(entry["artifact"])


def _hf_cached(repo_id: str, revision: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(repo_id, "model.safetensors", revision=revision)
        return isinstance(hit, str)
    except Exception:
        return False


def audit(verify_checksums: bool | None = None) -> list[dict]:
    """Integration matrix rows (no model is loaded, nothing is changed)."""
    from src.evaluation.final import sha256_path

    app = load_app_config()
    verify = app["verify_artifact_checksums"] if verify_checksums is None else verify_checksums
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"checksums": {}}
    models = final_models()
    rows = []

    def model_row(component, kind, mid, impl, dependency, fallback):
        entry = _registry_entry(kind, mid)
        if entry is None:
            rows.append({"component": component, "implementation": impl, "artifact": f"{mid} (not in registry)",
                         "status": "missing", "dependency": dependency, "fallback": fallback})
            return
        path = _artifact_path(entry) if entry["artifact"].startswith("{MODEL_DIR}") else REPO_ROOT / entry["artifact"]
        status, version = ("available" if path.exists() else "missing"), mid
        expected = manifest["checksums"].get(f"artifact:{mid}")
        if status == "available" and expected and verify:
            actual = sha256_path(path)
            status = "available" if actual == expected else "checksum mismatch"
            version = f"{mid} (sha256 {actual[:12]})"
        elif status == "available" and expected:
            version = f"{mid} (frozen sha256 {expected[:12]}; not re-hashed)"
        rows.append({"component": component, "implementation": impl, "artifact": version, "status": status,
                     "dependency": dependency, "fallback": fallback,
                     "preprocessing": entry.get("preprocessing_mode") or entry.get("text_preprocessing_mode")})

    model_row("FINAL_TEXT_MODEL", "text", models["text"], "src.inference.text.predict_text (fine-tuned MuRIL)",
              "checkpoint in $NLP_CLD_MODEL_DIR", "none: Analyze Text disabled; memes cannot use M04")
    model_row("FINAL_IMAGE_MODEL", "image", models["image"], "src.inference.image.predict_image (SigLIP + LR)",
              "SigLIP weights (Hugging Face cache)", "none: Analyze Meme disabled")
    model_row("FINAL_MULTIMODAL_MODEL", "multimodal", models["multimodal"], "src.inference.multimodal.predict_multimodal",
              "T05 checkpoint + SigLIP weights", "none: Analyze Meme disabled")
    model_row("second-opinion text model", "text", models["second_opinion_text"], "src.inference.text.predict_text (TF-IDF + LR)",
              "joblib artifact in models/text", "none: text-mode moderation rule unavailable")

    cfg = load_config()
    enc = cfg["image_model"]["encoder"]
    rows.append({"component": "SigLIP encoder weights", "implementation": "src.models.image_encoder.load_encoder (backend='pil')",
                 "artifact": f"{enc['checkpoint']}@{enc['revision'][:8]}",
                 "status": "available" if _hf_cached(enc["checkpoint"], enc["revision"]) else "missing",
                 "dependency": "Hugging Face cache (offline)", "fallback": "none: Analyze Meme disabled"})
    labels = manifest.get("label_map", {})
    from src.inference.text import LABEL_NAMES
    rows.append({"component": "label mapping", "implementation": "0 = non_misogynistic, 1 = misogynistic",
                 "artifact": "freeze manifest label_map", "status": "available" if {int(k): v for k, v in labels.items()} == LABEL_NAMES
                 else "checksum mismatch", "dependency": "results/final/model_freeze_manifest.json", "fallback": "none"})
    tp = cfg["text_preprocessing"]["version"]
    t_entry = _registry_entry("text", models["text"]) or {}
    rows.append({"component": "Task 6 preprocessing", "implementation": "src.preprocessing.preprocess (inside each model's path)",
                 "artifact": f"v{tp}", "status": "available" if t_entry.get("preprocessing_version") == tp else "checksum mismatch",
                 "dependency": "src/preprocessing (frozen)", "fallback": "none"})
    ocr_cfg = yaml.safe_load((REPO_ROOT / "config" / "ocr.yaml").read_text(encoding="utf-8"))
    weights = Path.home() / ".EasyOCR" / "model"
    try:
        import easyocr
        ocr_ok = all((weights / w).exists() for w in EASYOCR_WEIGHTS)
        ocr_version = easyocr.__version__
    except Exception:
        ocr_ok, ocr_version = False, "not installed"
    rows.append({"component": "OCR pipeline (EasyOCR)", "implementation": "src.ocr.pipeline.analyze_meme_input (Task 14)",
                 "artifact": f"easyocr {ocr_version}, languages {ocr_cfg['languages']}, preparation {ocr_cfg['preparation']}",
                 "status": "available" if ocr_ok else "missing", "dependency": "~/.EasyOCR/model weights (285 MB)",
                 "fallback": "Analyze Meme disabled (OCR is required to read the caption)"})
    from src.explainability.moderation import load_policy, validate_policy
    try:
        validate_policy(load_policy())
        mod_status = "available"
    except Exception:
        mod_status = "missing"
    rows.append({"component": "XAI pipeline", "implementation": "src.explainability (IG, occlusion, ablation; Task 15)",
                 "artifact": "config/explainability.yaml v1.0", "status": "available", "dependency": "the final models",
                 "fallback": "prediction shown without explanation"})
    rows.append({"component": "moderation recommendation", "implementation": "src.explainability.moderation.recommend",
                 "artifact": "rules in config/explainability.yaml", "status": mod_status, "dependency": "model labels",
                 "fallback": "none (rules are deterministic)"})
    import os
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
    uri_var = cfg["database"]["uri_env_var"]
    rows.append({"component": "MongoDB repositories", "implementation": "src.database.repositories (Task 2) via HistoryService",
                 "artifact": f"{uri_var} {'set' if os.environ.get(uri_var, '').strip() else 'NOT set'}",
                 "status": "available" if os.environ.get(uri_var, "").strip() else "warning",
                 "dependency": "MongoDB Atlas (network + IP allow-list)", "fallback": "analysis works; history / feedback disabled"})
    from src.application.research import ARTIFACTS
    present = sum((REPO_ROOT / a["path"]).exists() for a in ARTIFACTS.values())
    rows.append({"component": "research results (dashboard)", "implementation": "src.application.research",
                 "artifact": f"{present}/{len(ARTIFACTS)} stored tables / figures", "status": "available" if present == len(ARTIFACTS)
                 else "warning", "dependency": "results/ (Tasks 5-15)", "fallback": "missing sections are shown as unavailable"})
    return rows


def mode_availability(rows: list[dict]) -> dict:
    """{"text": (ok, [blocking components]), "meme": (...)} from audit rows."""
    status = {r["component"]: r["status"] for r in rows}
    out = {}
    for mode, needed in MODE_REQUIREMENTS.items():
        blocking = [c for c in needed if status.get(c) != "available"]
        out[mode] = (not blocking, blocking)
    return out


def require(mode: str, rows: list[dict]) -> None:
    ok, blocking = mode_availability(rows)[mode]
    if not ok:
        raise ArtifactUnavailableError(f"This analysis is unavailable because required components are missing or "
                                       f"changed: {', '.join(blocking)}. No other model is substituted.")
