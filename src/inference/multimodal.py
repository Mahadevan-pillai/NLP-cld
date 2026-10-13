"""Multimodal inference interface (Task 11). Not connected to Streamlit or MongoDB yet.

    from src.inference.multimodal import predict_multimodal
    predict_multimodal("meme.jpg", "caption text", model_version="M02")

Steps (identical to training):
1. validate inputs: a missing image or missing text (None) raises MissingModalityError,
   a corrupt image raises ImageLoadError. No fake embedding is ever substituted.
   An EMPTY caption is allowed (models were trained with empty OCR captions) and is flagged.
2. text: Task 6 preprocess() with the model's recorded mode -> MuRIL representation
   (fine-tuned T05 pooled [CLS] or frozen mean pooling, as recorded in the registry);
3. image: Task 10 load_image() + pinned SigLIP processor -> frozen pooled embedding;
4. fusion classifier (M01 logistic regression or M02/M03 fusion MLP).

No OCR is run here: the caption must be supplied (OCR arrives in a later task).
Model versions are listed in results/metrics/multimodal_model_registry.json.
SCORES ARE NOT CALIBRATED CONFIDENCES. score_type says what the number is.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.models.image_encoder import LABEL_NAMES, image_embeddings
from src.preprocessing.image import IMAGE_PREPROCESSING_VERSION, load_image
from src.utils.config import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "results" / "metrics" / "multimodal_model_registry.json"
SCORE_TYPES = {
    "concat_logreg": "uncalibrated logistic-regression probability of 'misogynistic'",
    "fusion_mlp": "uncalibrated softmax probability of 'misogynistic'",
}


class MissingModalityError(ValueError):
    """The image or the text input is missing; the multimodal model needs both."""


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _text_encoder(representation: str, source: str, revision: str):
    """Fine-tuned T05 (folder path) or the frozen base MuRIL (checkpoint name at a pinned revision)."""
    if representation == "finetuned_pooled":
        from src.inference.text import resolve
        from src.models.transformer_text import load_fine_tuned

        model, tokenizer, _ = load_fine_tuned(resolve(source))
        return model, tokenizer
    from transformers import AutoModel, AutoTokenizer

    return AutoModel.from_pretrained(source, revision=revision), AutoTokenizer.from_pretrained(source, revision=revision)


@lru_cache(maxsize=4)
def _image_encoder(checkpoint: str, revision: str):
    from src.models.image_encoder import load_encoder

    return load_encoder(checkpoint, revision)


@lru_cache(maxsize=8)
def _classifier(model_version: str, registry_path: str):
    entry = load_registry(Path(registry_path))["models"][model_version]
    if entry["kind"] not in SCORE_TYPES:
        raise ValueError(f"unknown model kind {entry['kind']!r}")
    if entry["image_preprocessing_version"] != IMAGE_PREPROCESSING_VERSION:
        raise RuntimeError(f"{model_version} expects image preprocessing {entry['image_preprocessing_version']}, "
                           f"code is {IMAGE_PREPROCESSING_VERSION}")
    path = REPO_ROOT / entry["artifact"]
    if entry["kind"] == "concat_logreg":
        import joblib

        return entry, joblib.load(path)
    from src.models.fusion import load_fusion

    return entry, load_fusion(path)


def text_features(text: str, entry: dict) -> tuple[np.ndarray, bool]:
    from src.features.multimodal import finetuned_pooled_features
    from src.models.transformer_text import mean_pooled_embeddings, preprocess_all

    processed = preprocess_all([text], entry["text_preprocessing_mode"], entry.get("text_preprocessing_options"))[0]
    model, tokenizer = _text_encoder(entry["text_representation"], entry["text_source"], entry["text_encoder_revision"])
    if entry["text_representation"] == "finetuned_pooled":
        feats = finetuned_pooled_features([processed], model, tokenizer, entry["max_length"])
    else:
        feats = mean_pooled_embeddings([processed], tokenizer, model, entry["max_length"])
    return feats, not processed.strip()


def predict_multimodal(image, text, model_version: str, registry_path: Path = REGISTRY_PATH) -> dict:
    """Predict misogynistic / non-misogynistic from an image AND its caption. Returns a structured dict."""
    if image is None:
        raise MissingModalityError("image is missing; the multimodal model needs both image and text")
    if text is None:
        raise MissingModalityError("text is missing; supply the caption (an empty string is allowed)")
    if not isinstance(text, str):
        raise MissingModalityError(f"text must be a string, got {type(text).__name__}")
    entry, clf = _classifier(model_version, str(registry_path))
    img = load_image(image)                                   # raises ImageLoadError for corrupt input
    t_feat, text_empty = text_features(text, entry)
    processor, encoder = _image_encoder(entry["image_encoder"], entry["image_encoder_revision"])
    i_feat = image_embeddings([img], processor, encoder, batch_size=1)
    if entry["kind"] == "concat_logreg":
        x = np.hstack([t_feat, i_feat])
        label, score = int(clf.predict(x)[0]), float(clf.predict_proba(x)[0, 1])
    else:
        from src.models.fusion import predict_proba

        model, standardizers, _ = clf
        proba = predict_proba(model, standardizers, t_feat, i_feat)[0]
        label, score = int(np.argmax(proba)), float(proba[1])
    return {
        "model_version": model_version,
        "task": "misogyny_detection",
        "modality": "image+text",
        "predicted_label": label,
        "predicted_name": LABEL_NAMES[label],
        "score": round(score, 6),
        "score_type": SCORE_TYPES[entry["kind"]],
        "calibrated": False,
        "text_representation": entry["text_representation"],
        "text_preprocessing_mode": entry["text_preprocessing_mode"],
        "text_preprocessing_version": entry["text_preprocessing_version"],
        "image_preprocessing_version": entry["image_preprocessing_version"],
        "processed_text_empty": text_empty,
        "ocr_used": False,
    }
