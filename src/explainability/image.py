"""Image explanations (Task 15): occlusion sensitivity for the frozen SigLIP image models.

    from src.explainability.image import explain_image
    explain_image("meme.jpg", model_version="I03")

Method (chosen to match the ACTUAL model):
- SigLIP is a Vision Transformer (no convolutional feature maps), so Grad-CAM does not apply, and
  attention maps of its attention-pooling head are not a faithful attribution of the classifier output.
- **Occlusion**: a grey square (window x window pixels of the 224 x 224 model input) is slid over the
  image with a fixed stride; each occluded copy goes through the SAME frozen pipeline (pinned processor
  -> vision tower -> classifier). The drop in the classifier's log-odds when a region is hidden is that
  region's attribution. Model-agnostic and faithful by construction (it measures the real model's
  output), at the cost of one forward pass per window.
- Fill value 0.0 in SigLIP-normalised space = mid-grey (pixel value 0.5).
- Cell map: the 224 x 224 input is divided into (224 / stride)^2 cells; a cell's value is the mean drop
  of all windows that cover it. Positive = hiding the region LOWERS the misogyny log-odds (the region
  supported "misogynistic"); negative = the region supported "non-misogynistic".

The model input is the image stretched to a square (Task 10 preprocessing), so a cell maps to the same
fraction of the original image. The vision encoder also "sees" text rendered in the picture: a hot cell
over a caption means the model responded to that region, which may be the text itself.
Attribution is MODEL ATTRIBUTION, not proof of visual reasoning.
"""

import numpy as np
import torch
from PIL import Image

from src.inference import image as image_inference
from src.inference.image import SCORE_TYPES
from src.models.image_encoder import LABEL_NAMES
from src.preprocessing.image import load_image

METHOD = "occlusion"
INPUT_SIZE = 224
LIMITATIONS = [
    "Model attribution, not proof of visual reasoning, intent or harm.",
    "Occlusion with a grey square creates an unnatural input; the score change partly reflects that artefact.",
    "Resolution is coarse (one value per cell); small details inside a cell are not separated.",
    "Regions interact: hiding two regions together can change the score differently from the sum of the two.",
    "The image model can respond to text rendered inside the picture; a highlighted region may be the "
    "meme's text, not a purely visual feature.",
    "The score is uncalibrated (an overconfident LR probability), not a confidence percentage.",
]


def window_positions(size: int = INPUT_SIZE, window: int = 56, stride: int = 28) -> list[tuple[int, int]]:
    if window > size or stride <= 0 or (size - window) % stride or size % stride:
        raise ValueError(f"window {window} / stride {stride} must tile a {size} px input exactly")
    return [(y, x) for y in range(0, size - window + 1, stride) for x in range(0, size - window + 1, stride)]


def occluded_copies(pixel_values: torch.Tensor, positions, window: int, fill: float = 0.0) -> torch.Tensor:
    """(len(positions), C, H, W) copies of a (1, C, H, W) input, each with one square set to `fill`."""
    batch = pixel_values.repeat(len(positions), 1, 1, 1)
    for i, (y, x) in enumerate(positions):
        batch[i, :, y:y + window, x:x + window] = fill
    return batch


def windows_to_cells(drops: np.ndarray, positions, size: int, window: int, stride: int) -> np.ndarray:
    """Mean drop of every window covering each stride x stride cell -> (size/stride, size/stride) map."""
    g, k = size // stride, window // stride
    total, count = np.zeros((g, g)), np.zeros((g, g))
    for d, (y, x) in zip(drops, positions):
        cy, cx = y // stride, x // stride
        total[cy:cy + k, cx:cx + k] += d
        count[cy:cy + k, cx:cx + k] += 1
    return total / np.maximum(count, 1)


@torch.no_grad()
def occlusion_embeddings(pixel_values: torch.Tensor, encoder, window: int = 56, stride: int = 28, fill: float = 0.0,
                         batch_size: int = 16) -> tuple[np.ndarray, np.ndarray, list]:
    """Pooled embeddings of the full input and of every occluded copy (computed once, reusable by any head)."""
    positions = window_positions(pixel_values.shape[-1], window, stride)
    full = encoder(pixel_values=pixel_values).pooler_output.numpy()
    copies = occluded_copies(pixel_values, positions, window, fill)
    occluded = np.concatenate([encoder(pixel_values=copies[i:i + batch_size]).pooler_output.numpy()
                               for i in range(0, len(copies), batch_size)])
    return full.astype(np.float32), occluded.astype(np.float32), positions


def occlusion_map(full_score: float, occluded_scores: np.ndarray, positions, size: int = INPUT_SIZE,
                  window: int = 56, stride: int = 28) -> np.ndarray:
    return windows_to_cells(full_score - np.asarray(occluded_scores, dtype=float), positions, size, window, stride)


def log_odds(classifier, x: np.ndarray) -> np.ndarray:
    """Classifier log-odds of 'misogynistic' (LR) or signed margin (SVM): the decision function."""
    return np.asarray(classifier.decision_function(x), dtype=float)


def pixel_values_for(img, processor) -> torch.Tensor:
    return processor(images=[img], return_tensors="pt")["pixel_values"]


def top_regions(cell_map: np.ndarray, k: int = 3, sign: int = 1) -> list[dict]:
    """Strongest cells in one direction, with their box as FRACTIONS of the image (x0, y0, x1, y1)."""
    g = cell_map.shape[0]
    order = np.argsort(-(sign * cell_map), axis=None)[:k]
    out = []
    for flat in order:
        r, c = divmod(int(flat), g)
        if sign * cell_map[r, c] <= 0:
            break
        out.append({"cell": [r, c], "box_fraction": [round(c / g, 4), round(r / g, 4), round((c + 1) / g, 4),
                                                       round((r + 1) / g, 4)],
                    "contribution": round(float(cell_map[r, c]), 6)})
    return out


def explain_image(image, model_version: str = "I03", window: int = 56, stride: int = 28, fill: float = 0.0,
                  registry_path=None) -> dict:
    """Prediction + occlusion map for one image, from the frozen image model `model_version`."""
    registry_path = registry_path or image_inference.REGISTRY_PATH
    if image is None:
        raise ValueError("image is missing")
    entry, clf = image_inference._load(model_version, str(registry_path))
    img = load_image(image)                                         # raises ImageLoadError for bad input
    prediction = image_inference.predict_image(img, model_version, registry_path)   # same path as the app
    processor, encoder = image_inference._encoder(entry["encoder"], entry["encoder_revision"])
    pv = pixel_values_for(img, processor)
    full, occluded, positions = occlusion_embeddings(pv, encoder, window, stride, fill)
    s_full = float(log_odds(clf, full)[0])
    cell_map = occlusion_map(s_full, log_odds(clf, occluded), positions, pv.shape[-1], window, stride)
    return {
        "model_version": model_version,
        "task": "misogyny_detection",
        "modality": "image",
        "predicted_label": prediction["predicted_label"],
        "predicted_name": LABEL_NAMES[prediction["predicted_label"]],
        "score": prediction["score"],
        "score_type": SCORE_TYPES[entry["kind"]],
        "calibrated": False,
        "logit_difference": round(s_full, 6),
        "method": {"name": METHOD, "window_px": window, "stride_px": stride, "fill": "mid-grey (0 after normalisation)",
                   "input": f"{pv.shape[-1]} x {pv.shape[-1]} (image stretched to a square, Task 10 preprocessing)",
                   "target": "drop in log-odds of 'misogynistic' when the region is hidden",
                   "forward_passes": len(positions) + 1},
        "heatmap": [[round(float(v), 6) for v in row] for row in cell_map],
        "heatmap_shape": list(cell_map.shape),
        "top_regions_towards_misogynistic": top_regions(cell_map, 3, +1),
        "top_regions_towards_non_misogynistic": top_regions(cell_map, 3, -1),
        "input_size": list(img.size),
        "text_used": False,
        "limitations": LIMITATIONS,
    }


# ------------------------------------------------------------------ research helpers (figures / statistics)

def upsample(cell_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Cell map -> (height, width) array by nearest-cell repetition (no smoothing, no invented detail)."""
    w, h = size
    g = cell_map.shape[0]
    rows = np.minimum((np.arange(h) * g) // h, g - 1)
    cols = np.minimum((np.arange(w) * g) // w, g - 1)
    return cell_map[rows][:, cols]


def overlay(image, cell_map: np.ndarray, alpha: float = 0.45, vmax: float | None = None) -> Image.Image:
    """New RGB image: the original with a red (towards misogynistic) / blue (towards non) overlay.
    The input image is never modified."""
    import matplotlib

    img = load_image(image).copy()
    vmax = vmax or float(np.abs(cell_map).max()) or 1.0
    rgba = matplotlib.colormaps["coolwarm"]((upsample(cell_map, img.size) / vmax + 1) / 2)
    heat = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    return Image.blend(img, heat, alpha)


def region_share(cell_map: np.ndarray, boxes_fraction, resolution: int = INPUT_SIZE) -> dict:
    """Share of the POSITIVE attribution mass inside boxes vs the share of the area they cover.
    enrichment > 1: attribution concentrates on the boxes more than their size alone would give."""
    grid = upsample(cell_map, (resolution, resolution))
    mask = np.zeros((resolution, resolution), bool)
    for x0, y0, x1, y1 in boxes_fraction:
        mask[int(max(0, y0) * resolution):int(np.ceil(min(1, y1) * resolution)),
             int(max(0, x0) * resolution):int(np.ceil(min(1, x1) * resolution))] = True
    pos = np.clip(grid, 0, None)
    area = float(mask.mean())
    mass = float(pos[mask].sum() / pos.sum()) if pos.sum() > 0 else None
    return {"area_share": round(area, 4), "positive_mass_share": None if mass is None else round(mass, 4),
            "enrichment": None if mass is None or area == 0 else round(mass / area, 4)}


def concentration(cell_map: np.ndarray, top_fraction: float = 0.1) -> float | None:
    """Share of the total |attribution| held by the top `top_fraction` of cells (1 = all in a few cells)."""
    a = np.sort(np.abs(cell_map).ravel())[::-1]
    if a.sum() == 0:
        return None
    k = max(1, int(round(top_fraction * a.size)))
    return round(float(a[:k].sum() / a.sum()), 4)
