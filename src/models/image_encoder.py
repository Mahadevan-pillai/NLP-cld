"""Frozen pretrained image encoder + lightweight classifiers (Task 10).

    image -> load_image() (RGB, EXIF orientation)            src/preprocessing/image.py
          -> encoder's own image processor (SigLIP: resize 224x224, rescale, normalise)
          -> frozen vision tower -> pooled embedding (768-d for SigLIP base)
          -> StandardScaler + LogisticRegression / LinearSVC  (fitted on TRAIN only)

Only the VISION tower is loaded; the text tower is never used, and no OCR or
extracted text enters the image model. The encoder may still "see" text that is
rendered inside the meme picture (a documented limitation, not a text feature).
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.preprocessing.image import IMAGE_PREPROCESSING_VERSION, load_image
from src.utils.config import REPO_ROOT

LABEL_NAMES = {0: "non_misogynistic", 1: "misogynistic"}


# ------------------------------------------------------------------
# Encoder
# ------------------------------------------------------------------

def load_encoder(checkpoint: str, revision: str | None = None):
    """Return (image_processor, vision_model) for a pinned checkpoint, in eval mode."""
    from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

    # backend="pil" pins the PIL implementation of resizing. Without it, transformers would switch to the
    # torchvision implementation whenever torchvision happens to be installed, which changes pixel values slightly.
    processor = AutoImageProcessor.from_pretrained(checkpoint, revision=revision, backend="pil")
    if "siglip" in checkpoint.lower():
        model = SiglipVisionModel.from_pretrained(checkpoint, revision=revision)   # vision tower only
    else:
        model = AutoModel.from_pretrained(checkpoint, revision=revision)
    model.eval()
    return processor, model


def _batches(n: int, size: int):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


@torch.no_grad()
def image_embeddings(images, processor, model, batch_size: int = 32, log=None) -> np.ndarray:
    """Pooled embeddings for a list of image sources (paths, bytes or PIL images).

    Images are loaded lazily per batch (bounded memory). The encoder is frozen:
    eval mode and no gradients, so its weights never change.
    """
    model.eval()
    out, start = [], time.perf_counter()
    for i, sl in enumerate(_batches(len(images), batch_size)):
        pixel_values = processor(images=[load_image(x) for x in images[sl]], return_tensors="pt")["pixel_values"]
        out.append(model(pixel_values=pixel_values).pooler_output.cpu().numpy())
        if log and i % 20 == 0:
            log(f"    {sl.stop}/{len(images)} images ({time.perf_counter() - start:.0f}s)")
    return np.concatenate(out).astype(np.float32)


# ------------------------------------------------------------------
# Embedding cache (one .npy + one .json per split; never mixed)
# ------------------------------------------------------------------

def embedding_cache_dir(config: dict) -> Path:
    """<dir of $NLP_CLD_MODEL_DIR>/../cache/image_embeddings/<encoder>_<rev>_img<version> (outside the repo).

    Falls back to models/image/cache (git-ignored) when the variable is unset.
    """
    from dotenv import load_dotenv

    icfg, enc = config["image_model"], config["image_model"]["encoder"]
    load_dotenv(REPO_ROOT / ".env", override=False)
    value = os.environ.get(icfg["cache_dir_env"], "").strip()
    root = Path(value).parent / "cache" / "image_embeddings" if value else REPO_ROOT / icfg["cache_dir_default"]
    return root / f"{enc['checkpoint'].split('/')[-1]}_{enc['revision'][:8]}_img{IMAGE_PREPROCESSING_VERSION}"


def load_cached_embeddings(config: dict, split: str, sample_ids) -> np.ndarray:
    """Cached embeddings for a split, checked to be in exactly this sample order."""
    cdir = embedding_cache_dir(config)
    meta = json.loads((cdir / f"{split}.json").read_text(encoding="utf-8"))
    if meta["sample_ids"] != list(sample_ids):
        raise RuntimeError(f"cached {split} embeddings do not match the sample order; re-run the embed step")
    return np.load(cdir / f"{split}.npy")


def fingerprint(sample_ids, image_sha256) -> str:
    """Identifies the exact ordered list of samples and image contents."""
    h = hashlib.sha256()
    for sid, sha in zip(sample_ids, image_sha256):
        h.update(f"{sid}:{sha}\n".encode())
    return h.hexdigest()


def cache_metadata(split: str, sample_ids, image_sha256, checkpoint: str, revision: str, dim: int | None = None) -> dict:
    return {"split": split, "n": len(sample_ids), "encoder_checkpoint": checkpoint, "encoder_revision": revision,
            "image_preprocessing_version": IMAGE_PREPROCESSING_VERSION, "embedding_dim": dim,
            "samples_fingerprint": fingerprint(sample_ids, image_sha256), "sample_ids": list(sample_ids)}


def cache_is_valid(npy: Path, meta_path: Path, expected: dict) -> bool:
    """True only if the cached file was made for exactly these samples, images, encoder and preprocessing."""
    if not (npy.exists() and meta_path.exists()):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    keys = ("split", "n", "encoder_checkpoint", "encoder_revision", "image_preprocessing_version",
            "samples_fingerprint", "sample_ids")
    if any(meta.get(k) != expected[k] for k in keys):
        return False
    arr = np.load(npy, mmap_mode="r")
    return arr.shape[0] == expected["n"] and arr.shape[1] == meta.get("embedding_dim")


def embed_cached(paths, expected_meta: dict, cache_dir: Path, processor, model, batch_size: int = 32, log=print):
    """Load embeddings from cache if valid, otherwise compute and store them. Returns (array, seconds, cached)."""
    cache_dir = Path(cache_dir)
    npy, meta_path = cache_dir / f"{expected_meta['split']}.npy", cache_dir / f"{expected_meta['split']}.json"
    if cache_is_valid(npy, meta_path, expected_meta):
        return np.load(npy), 0.0, True
    if npy.exists():
        log(f"    cache for {expected_meta['split']} is stale or incomplete; recomputing")
    start = time.perf_counter()
    emb = image_embeddings(list(paths), processor, model, batch_size, log=log)
    seconds = time.perf_counter() - start
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, emb)
    meta_path.write_text(json.dumps({**expected_meta, "embedding_dim": int(emb.shape[1]),
                                     "seconds": round(seconds, 1)}, indent=1), encoding="utf-8")
    return emb, seconds, False


# ------------------------------------------------------------------
# Classifiers (fitted on TRAIN embeddings only)
# ------------------------------------------------------------------

def build_classifier(kind: str, settings: dict, seed: int):
    """StandardScaler + LR or LinearSVC. Scaling statistics are learned from TRAIN only (inside fit)."""
    if kind == "logistic_regression":
        clf = LogisticRegression(random_state=seed, **settings)
    elif kind == "linear_svm":
        clf = LinearSVC(random_state=seed, **settings)
    else:
        raise ValueError(f"unknown classifier {kind!r}")
    return make_pipeline(StandardScaler(), clf)


def classifier_scores(pipeline, x: np.ndarray) -> np.ndarray:
    """LR: uncalibrated probability of class 1. SVM: signed decision score (NOT a probability)."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x)[:, 1]
    return pipeline.decision_function(x)
