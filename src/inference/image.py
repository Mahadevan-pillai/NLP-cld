"""Image-model inference interface (Task 10). Not connected to Streamlit or MongoDB yet.

    from src.inference.image import predict_image
    predict_image("meme.jpg", model_version="I01")      # path, raw bytes or PIL image

Steps (identical to training):
1. load_image(): validate, decode, EXIF orientation, RGB      (src/preprocessing/image.py)
2. the pinned encoder's own image processor (resize, rescale, normalise)
3. frozen vision tower -> pooled embedding
4. the saved classifier (fitted on TRAIN embeddings)

The image ONLY is used: no OCR, no text. Model versions are listed in
results/metrics/image_model_registry.json.

SCORES ARE NOT CALIBRATED CONFIDENCES. score_type says what the number is.
"""

import json
from functools import lru_cache
from pathlib import Path

from src.models.image_encoder import LABEL_NAMES, classifier_scores, image_embeddings, load_encoder
from src.preprocessing.image import IMAGE_PREPROCESSING_VERSION, load_image
from src.utils.config import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "results" / "metrics" / "image_model_registry.json"
SCORE_TYPES = {
    "frozen_logreg": "uncalibrated logistic-regression probability of 'misogynistic'",
    "frozen_svm": "decision score (signed margin, NOT a probability)",
}


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _encoder(checkpoint: str, revision: str):
    return load_encoder(checkpoint, revision)


@lru_cache(maxsize=8)
def _load(model_version: str, registry_path: str = str(REGISTRY_PATH)):
    import joblib

    entry = load_registry(Path(registry_path))["models"][model_version]
    if entry["kind"] not in SCORE_TYPES:
        raise ValueError(f"unknown model kind {entry['kind']!r}")
    if entry["image_preprocessing_version"] != IMAGE_PREPROCESSING_VERSION:
        raise RuntimeError(f"{model_version} was trained with image preprocessing "
                           f"{entry['image_preprocessing_version']}, code is {IMAGE_PREPROCESSING_VERSION}")
    return entry, joblib.load(REPO_ROOT / entry["artifact"])


def predict_image(image, model_version: str, registry_path: Path = REGISTRY_PATH) -> dict:
    """Predict misogynistic / non-misogynistic from the image alone. Raises ImageLoadError for bad input."""
    entry, clf = _load(model_version, str(registry_path))
    img = load_image(image)                                    # validates the input first
    processor, encoder = _encoder(entry["encoder"], entry["encoder_revision"])
    emb = image_embeddings([img], processor, encoder, batch_size=1)
    label = int(clf.predict(emb)[0])
    score = float(classifier_scores(clf, emb)[0])
    return {
        "model_version": model_version,
        "task": "misogyny_detection",
        "modality": "image",
        "predicted_label": label,
        "predicted_name": LABEL_NAMES[label],
        "score": round(score, 6),
        "score_type": SCORE_TYPES[entry["kind"]],
        "calibrated": False,
        "encoder": f"{entry['encoder']}@{entry['encoder_revision'][:8]}",
        "image_preprocessing_version": entry["image_preprocessing_version"],
        "input_size": list(img.size),
        "text_used": False,
    }
