"""Image loading shared by training and inference (Task 10).

Version 1.0 steps (deterministic; nothing is learned; nothing is saved):
1. open the image (file path, raw bytes, or an already-open PIL image);
2. fully decode it, so corrupt / truncated files fail here with ImageLoadError;
3. apply the EXIF orientation tag (phone photos may be stored rotated);
4. convert to 3-channel RGB (drops alpha, expands grayscale / palette images);
5. reject degenerate images (a side shorter than MIN_SIDE pixels).

Resizing and normalisation are NOT done here: they belong to the pretrained
encoder's own image processor (see src/models/image_encoder.py), so the exact
preprocessing the encoder was trained with is used.

The original image file is never modified. No OCR is run and no text region
is removed: the model receives the complete meme image.
"""

import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_PREPROCESSING_VERSION = "1.0"
MIN_SIDE = 16           # pixels; smaller images carry no usable content (MIMIC minimum is far larger)


class ImageLoadError(ValueError):
    """The input is missing, unreadable, corrupt or not a usable image."""


def load_image(source) -> Image.Image:
    """Return a fully decoded RGB PIL image, or raise ImageLoadError."""
    if source is None:
        raise ImageLoadError("no image given")
    try:
        if isinstance(source, Image.Image):
            img = source.copy()
        elif isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise ImageLoadError(f"image file not found: {path.name}")
            with Image.open(path) as opened:
                opened.load()                       # decode now; the file handle is closed afterwards
                img = opened.copy()
        else:
            raise ImageLoadError(f"unsupported image input type: {type(source).__name__}")
        img.load()
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageLoadError(f"unreadable or corrupt image ({type(exc).__name__})") from exc
    if min(img.size) < MIN_SIDE:
        raise ImageLoadError(f"image too small: {img.size[0]}x{img.size[1]} pixels")
    return img
