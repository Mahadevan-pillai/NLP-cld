"""OCR engine wrapper: extract_text_from_image(image) -> structured result (never raises on normal failures).

Engine: EasyOCR (Apache-2.0), languages ["en", "hi"] (English + Devanagari), CPU. MIMIC's own captions
were produced with EasyOCR (dataset card), so this keeps application text close to the training text.

OCR-specific image preparation is separate from the Task 10 vision preprocessing and never modifies
the original file. Variants (one is selected on VALIDATION, see config.yaml `ocr.preparation`):
    "none"                   decoded RGB image as is (EasyOCR converts to grayscale internally)
    "grayscale_autocontrast" grayscale + autocontrast (stretches low-contrast text)

Returned fields (only what EasyOCR actually provides):
    status            "ok" | "no_text" | "invalid_image" | "ocr_error"
    text              detected strings joined by single spaces, in EasyOCR's reading order ("" if none)
    confidence        mean recognition confidence of the regions (None if no region)
    detected_regions  [{"text", "confidence", "bbox"}]  (bbox = 4 corner points, pixels)
    engine, engine_version, languages, preparation, pipeline_version, error
"""

from functools import lru_cache

import numpy as np
from PIL import ImageOps

from src.preprocessing.image import ImageLoadError, load_image

OCR_PIPELINE_VERSION = "1.0"
LANGUAGES = ("en", "hi")
PREPARATIONS = ("none", "grayscale_autocontrast")


@lru_cache(maxsize=2)
def get_reader(languages: tuple = LANGUAGES):
    """EasyOCR reader (loaded once; weights in the EasyOCR model folder, downloaded on first use)."""
    import easyocr

    return easyocr.Reader(list(languages), gpu=False, verbose=False)


def engine_version() -> str:
    try:
        import easyocr

        return easyocr.__version__
    except Exception:                      # engine not installed: reported, not raised
        return "unavailable"


def prepare(img, preparation: str):
    """Return a numpy array for the OCR engine. Never modifies or saves the original image."""
    if preparation == "none":
        return np.asarray(img)
    if preparation == "grayscale_autocontrast":
        return np.asarray(ImageOps.autocontrast(img.convert("L")))
    raise ValueError(f"unknown OCR preparation {preparation!r}; choose from {PREPARATIONS}")


def _result(status, preparation, text="", confidence=None, regions=None, error=None) -> dict:
    return {"status": status, "text": text, "confidence": confidence, "detected_regions": regions or [],
            "engine": "easyocr", "engine_version": engine_version(), "languages": list(LANGUAGES),
            "preparation": preparation, "pipeline_version": OCR_PIPELINE_VERSION, "error": error}


def extract_text_from_image(image, preparation: str = "none", reader=None) -> dict:
    """OCR one image (file path, raw bytes or PIL image). Failures are returned as a status, not raised."""
    try:
        img = load_image(image)                                  # Task 10 validation: decode, EXIF, RGB
    except ImageLoadError as exc:
        return _result("invalid_image", preparation, error=str(exc))
    try:
        array = prepare(img, preparation)
        raw = (reader or get_reader()).readtext(array)          # [(bbox, text, confidence), ...]
    except ValueError:
        raise                                                    # programming error (bad preparation name)
    except Exception as exc:                                     # engine failure must not crash the app
        return _result("ocr_error", preparation, error=f"{type(exc).__name__}: {str(exc)[:200]}")
    regions = [{"text": str(t), "confidence": round(float(c), 4), "bbox": [[float(x), float(y)] for x, y in b]}
               for b, t, c in raw if str(t).strip()]
    if not regions:
        return _result("no_text", preparation)
    text = " ".join(r["text"].strip() for r in regions)
    confidence = round(float(np.mean([r["confidence"] for r in regions])), 4)
    return _result("ok", preparation, text=text, confidence=confidence, regions=regions)
