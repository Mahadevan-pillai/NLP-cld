"""Input validation for the application (Task 16). Messages are short, user-facing and safe.

Text: the frozen Task 6 preprocessing handles Unicode, Hindi, Hinglish, emojis and URLs; validation only
rejects input the models cannot analyse (empty / too long / no letters) and reports truncation.
Uploads: the format is decided by DECODING the bytes (never by the file name, which is not trusted and is
never used as a path); size and pixel limits protect memory; the bytes stay in memory.
"""

import io
import re
import unicodedata

from PIL import Image, UnidentifiedImageError

from src.preprocessing.image import ImageLoadError, load_image

LETTER_RE = re.compile(r"[A-Za-zऀ-ॿ]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
OTHER_SCRIPT_RE = re.compile(r"[^\W\d_A-Za-zÀ-ɏऀ-ॿ]")   # letters outside Latin / Devanagari


class InputError(ValueError):
    """The input cannot be analysed; str(error) is safe to show to the user."""


def validate_text(text, max_chars: int = 2000, min_letters: int = 1) -> tuple[str, list[str]]:
    """Return (cleaned text, warnings) or raise InputError."""
    if text is None:
        raise InputError("Please enter some text to analyse.")
    if not isinstance(text, str):
        raise InputError("Text input must be plain text.")
    cleaned = CONTROL_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()
    if not cleaned:
        raise InputError("Please enter some text to analyse (the input is empty or only spaces).")
    if len(cleaned) > max_chars:
        raise InputError(f"The text is too long ({len(cleaned):,} characters). Please shorten it to at most "
                         f"{max_chars:,} characters.")
    if len(LETTER_RE.findall(cleaned)) < min_letters:
        raise InputError("The text contains no Latin or Devanagari letters, so the models cannot analyse it "
                         "(emojis, numbers or symbols alone are not enough).")
    warnings = []
    if OTHER_SCRIPT_RE.search(cleaned):
        warnings.append("Parts of the text use a script other than Latin or Devanagari; the models were trained "
                        "on English / Hinglish / Hindi only, so results may be unreliable.")
    return cleaned, warnings


def token_warning(tokenizer, processed: str, max_length: int) -> str | None:
    """Warn when the processed text is longer than the model's input window (it is truncated)."""
    n = len(tokenizer([processed], truncation=False)["input_ids"][0])
    if n > max_length:
        return (f"The text is long ({n} tokens); the text model reads only the first {max_length} tokens, "
                "so the end of the text was not analysed.")
    return None


def validate_upload(data: bytes, allowed_formats=("JPEG", "PNG"), max_upload_mb: float = 10,
                    max_pixels: int = 40_000_000):
    """Return (RGB PIL image, safe metadata) or raise InputError. Nothing is written to disk."""
    if data is None or len(data) == 0:
        raise InputError("Please upload an image file.")
    if not isinstance(data, (bytes, bytearray)):
        raise InputError("The upload could not be read.")
    size = len(data)
    if size > max_upload_mb * 1024 * 1024:
        raise InputError(f"The image is too large ({size / 1024 / 1024:.1f} MB). The limit is {max_upload_mb:g} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:            # header only: format and size, no full decode
            fmt, (w, h), mode = probe.format, probe.size, probe.mode
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise InputError("The file is not a readable image. Please upload a JPG or PNG meme.") from None
    if fmt not in allowed_formats:
        raise InputError(f"Unsupported image format ({fmt}). Please upload a JPG or PNG image.")
    if w * h > max_pixels:
        raise InputError(f"The image has too many pixels ({w} x {h}). Please upload a smaller image.")
    try:
        img = load_image(bytes(data))                          # full decode + Task 10 checks (corrupt / tiny)
    except ImageLoadError:
        raise InputError("The image is corrupt, truncated or too small to analyse.") from None
    return img, {"width": int(w), "height": int(h), "format": str(fmt), "mode": str(mode), "size_bytes": int(size)}
