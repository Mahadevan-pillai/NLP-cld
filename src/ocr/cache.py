"""Research OCR cache (Task 14): avoids re-running OCR on the same DATASET image.

One JSON file per key. Key = SHA-256 of (image file SHA-256, engine, engine version, preparation, OCR
pipeline version), so a changed image, engine or preparation never reuses a stale result. Stored:
sample_id, image_sha256, engine/version, preparation, pipeline version and the OCR result (text,
confidence, regions). No credentials or user data.

For research only: the application must NOT cache user uploads (config/ocr.yaml cache.cache_user_uploads).
"""

import hashlib
import json
from pathlib import Path


def image_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_key(img_sha: str, engine: str, engine_version: str, preparation: str, pipeline_version: str) -> str:
    return hashlib.sha256(f"{img_sha}|{engine}|{engine_version}|{preparation}|{pipeline_version}".encode()).hexdigest()


class OCRCache:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, key: str, record: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def cached_ocr(path: Path, sample_id: str, preparation: str, cache: OCRCache, run_ocr) -> tuple[dict, bool]:
    """Return (OCR result, from_cache). run_ocr(path, preparation) produces a fresh result on a miss."""
    from src.ocr.engine import OCR_PIPELINE_VERSION, engine_version

    sha = image_sha256(path)
    key = cache_key(sha, "easyocr", engine_version(), preparation, OCR_PIPELINE_VERSION)
    hit = cache.get(key)
    if hit is not None:
        return hit["result"], True
    result = run_ocr(path, preparation)
    if result["status"] in ("ok", "no_text"):          # failures are not cached (they may be transient)
        cache.put(key, {"sample_id": sample_id, "image_sha256": sha, "engine": result["engine"],
                        "engine_version": result["engine_version"], "preparation": preparation,
                        "pipeline_version": OCR_PIPELINE_VERSION, "result": result})
    return result, False
