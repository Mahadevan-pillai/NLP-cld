"""OCR for meme images (Task 14).

Kept OUTSIDE src/preprocessing, src/models, src/features and src/evaluation on purpose: those folders
are checksum-frozen by the Task 13 final evaluation (adding files would change their hashes). OCR text
is preprocessed with the unchanged Task 6 `src.preprocessing.preprocess()`.
"""

from src.ocr.engine import OCR_PIPELINE_VERSION, extract_text_from_image

__all__ = ["OCR_PIPELINE_VERSION", "extract_text_from_image"]
