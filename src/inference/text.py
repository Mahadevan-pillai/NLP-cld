"""Text-model inference interface (Task 9). Not connected to Streamlit or MongoDB yet.

    from src.inference.text import predict_text
    predict_text("ye kya bakwas hai", model_version="T05")

Every model applies the SAME Task 6 preprocessing used in training:
- TF-IDF baselines: the saved sklearn Pipeline contains TextPreprocessor;
- frozen / fine-tuned transformers: preprocess(mode, version) from their metadata.

Model versions are listed in the registry
(results/metrics/text_model_registry.json). Large checkpoints live in the
folder named by the NLP_CLD_MODEL_DIR environment variable.

SCORES ARE NOT CALIBRATED CONFIDENCES. score_type says what the number is.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.utils.config import REPO_ROOT, load_config

REGISTRY_PATH = REPO_ROOT / "results" / "metrics" / "text_model_registry.json"
LABEL_NAMES = {0: "non_misogynistic", 1: "misogynistic"}
SCORE_TYPES = {
    "tfidf_logreg": "uncalibrated logistic-regression probability of 'misogynistic'",
    "tfidf_svm": "decision score (signed margin, NOT a probability)",
    "frozen_logreg": "uncalibrated logistic-regression probability of 'misogynistic'",
    "fine_tuned": "uncalibrated softmax probability of 'misogynistic'",
}


def model_dir(config: dict | None = None) -> Path:
    """Folder for large checkpoints: $NLP_CLD_MODEL_DIR, else models/text in the repo."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    tcfg = (config or load_config())["text_model"]["transformer"]
    value = os.environ.get(tcfg["artifact_dir_env"], "").strip()
    return Path(value) if value else REPO_ROOT / tcfg["artifact_dir_default"]


def resolve(path_template: str) -> Path:
    """Registry paths are repo-relative, or start with {MODEL_DIR} for large checkpoints."""
    if path_template.startswith("{MODEL_DIR}"):
        return model_dir() / path_template.replace("{MODEL_DIR}", "").lstrip("/\\")
    return REPO_ROOT / path_template


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _load(model_version: str, registry_path: str = str(REGISTRY_PATH)):
    entry = load_registry(Path(registry_path))["models"][model_version]
    kind = entry["kind"]
    if kind in ("tfidf_logreg", "tfidf_svm"):
        import joblib

        return kind, entry, joblib.load(resolve(entry["artifact"]))
    if kind == "fine_tuned":
        from src.models.transformer_text import load_fine_tuned

        return kind, entry, load_fine_tuned(resolve(entry["artifact"]))
    if kind == "frozen_logreg":
        import joblib
        from transformers import AutoModel, AutoTokenizer

        clf = joblib.load(resolve(entry["artifact"]))
        name, rev = entry["encoder"], entry["encoder_revision"]      # pinned pretrained encoder
        return kind, entry, (clf, AutoTokenizer.from_pretrained(name, revision=rev),
                             AutoModel.from_pretrained(name, revision=rev))
    raise ValueError(f"unknown model kind {kind!r}")


def predict_text(text, model_version: str, registry_path: Path = REGISTRY_PATH) -> dict:
    """Predict misogynistic / non-misogynistic for one text. Returns a structured dict."""
    kind, entry, obj = _load(model_version, str(registry_path))
    if kind in ("tfidf_logreg", "tfidf_svm"):
        pipeline = obj
        processed = pipeline.named_steps["preprocess"].transform(["" if text is None else text])[0]
        label = int(pipeline.predict([text])[0])
        score = (float(pipeline.predict_proba([text])[0, 1]) if kind == "tfidf_logreg"
                 else float(pipeline.decision_function([text])[0]))
    else:
        from src.models.transformer_text import mean_pooled_embeddings, predict_logits, preprocess_all, softmax

        processed = preprocess_all([text], entry["preprocessing_mode"], entry.get("preprocessing_options"))[0]
        if kind == "fine_tuned":
            model, tokenizer, meta = obj
            probs = softmax(predict_logits(model, tokenizer, [processed], entry["max_length"]))[0]
        else:
            clf, tokenizer, encoder = obj
            emb = mean_pooled_embeddings([processed], tokenizer, encoder, entry["max_length"])
            probs = clf.predict_proba(emb)[0]
        label, score = int(np.argmax(probs)), float(probs[1])
    return {
        "model_version": model_version,
        "task": "misogyny_detection",
        "predicted_label": label,
        "predicted_name": LABEL_NAMES[label],
        "score": round(score, 6),
        "score_type": SCORE_TYPES[kind],
        "calibrated": False,
        "preprocessing_mode": entry["preprocessing_mode"],
        "preprocessing_version": entry["preprocessing_version"],
        "processed_text_empty": not processed.strip(),
    }
