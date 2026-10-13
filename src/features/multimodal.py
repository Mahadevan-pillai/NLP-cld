"""Multimodal features (Task 11): text features, a sample-id keyed cache, and modality alignment.

Text features (from Task 9, no retraining):
- "finetuned_pooled": the fine-tuned MuRIL (T05) pooled [CLS] vector, i.e. exactly the
  768-d input of T05's classification head;
- "frozen_mean": frozen MuRIL attention-mask mean pooling (the T01-T03 representation).

Image features: the Task 10 SigLIP cache (src/models/image_encoder.py).

Every cached array is stored WITH its sample_ids, and modalities are joined on
sample_id only (never on row order). align() returns an audit of the join.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------------
# Text features
# ------------------------------------------------------------------

@torch.no_grad()
def finetuned_pooled_features(texts: list[str], model, tokenizer, max_length: int, batch_size: int = 32) -> np.ndarray:
    """Pooled [CLS] output of a fine-tuned BertForSequenceClassification (input of its classifier head)."""
    model.eval()
    out = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt")
        out.append(model.base_model(**enc).pooler_output.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ------------------------------------------------------------------
# Sample-id keyed feature cache (.npz = ids + features; .json = metadata)
# ------------------------------------------------------------------

def inputs_fingerprint(sample_ids, inputs) -> str:
    """Identifies the ordered (sample_id, input) pairs, e.g. the raw caption or the image SHA-256."""
    h = hashlib.sha256()
    for sid, x in zip(sample_ids, inputs):
        h.update(f"{sid}\x1f{x}\x1e".encode("utf-8"))
    return h.hexdigest()


def save_features(path: Path, sample_ids, features: np.ndarray, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(sample_ids) != len(features):
        raise ValueError("sample_ids and features differ in length")
    np.savez(path.with_suffix(".npz"), sample_ids=np.asarray(sample_ids, dtype=str), features=features)
    path.with_suffix(".json").write_text(json.dumps({**metadata, "n": len(sample_ids),
                                                     "dim": int(features.shape[1])}, indent=1), encoding="utf-8")


def load_features(path: Path, expected_metadata: dict | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (sample_ids, features), or None if missing or stale (metadata differs from expected)."""
    path = Path(path)
    if not (path.with_suffix(".npz").exists() and path.with_suffix(".json").exists()):
        return None
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if expected_metadata and any(meta.get(k) != v for k, v in expected_metadata.items()):
        return None
    data = np.load(path.with_suffix(".npz"))
    return data["sample_ids"].astype(str), data["features"]


# ------------------------------------------------------------------
# Alignment (join on sample_id only)
# ------------------------------------------------------------------

def align(labels: pd.DataFrame, modalities: dict[str, tuple[np.ndarray, np.ndarray]]):
    """Join each modality's (sample_ids, features) onto `labels` (sample_id, target_label) by sample_id.

    Returns (labels_aligned, {name: features in labels order}, audit rows). Raises if any
    modality has duplicate ids, is missing a labelled sample, or has a non-finite value.
    Samples missing from a modality are never silently dropped.
    """
    base_ids = labels["sample_id"].astype(str).to_numpy()
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("duplicate sample_id in labels")
    aligned, audit = {}, []
    for name, (ids, feats) in modalities.items():
        ids = np.asarray(ids).astype(str)
        index = pd.Index(ids)
        duplicates = int(index.duplicated().sum())
        missing = sorted(set(base_ids) - set(ids))
        extra = len(set(ids) - set(base_ids))
        finite = bool(np.isfinite(feats).all())
        audit.append({"modality": name, "labelled_samples": len(base_ids), "feature_rows": len(ids),
                      "duplicate_ids": duplicates, "missing_from_modality": len(missing),
                      "extra_ids_not_in_labels": extra, "dim": int(feats.shape[1]), "all_finite": finite,
                      "row_order_already_identical": bool(len(ids) == len(base_ids) and (ids == base_ids).all())})
        if duplicates or missing or not finite:
            raise ValueError(f"{name}: {duplicates} duplicate ids, {len(missing)} labelled samples missing, "
                             f"finite={finite}")
        aligned[name] = feats[index.get_indexer(base_ids)]
    return labels.reset_index(drop=True), aligned, audit
