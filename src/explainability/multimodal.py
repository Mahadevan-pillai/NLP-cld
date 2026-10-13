"""Multimodal explanations (Task 15): how the text and the image relate to the fused prediction.

    from src.explainability.multimodal import explain_multimodal
    explain_multimodal("meme.jpg", "caption text", model_version="M04")

Three pieces, all from the already trained, frozen models (nothing is re-fitted):

1. **Unimodal comparison**: text-only T05, image-only I03 and multimodal M04 labels + scores, from the
   same inference functions the application uses.
2. **Modality ablation inside M04**: one modality's features are replaced by their TRAIN mean, which
   the fusion model's own standardiser maps to exactly 0 ("an average training meme" for that
   modality). This is the same pre-specified neutralisation used in Task 13 (M04 reliance diagnostic);
   it is NOT an arbitrary zero vector in the raw feature space.
   Reported: the label and log-odds with each modality neutralised, and whether the label changes.
3. **Agreement**: how many of the two single-modality models (T05, I03) disagree with M04's label:
   0 -> AGREE, 1 -> PARTIAL DISAGREEMENT, 2 -> DISAGREE.

No percentage "contribution" of a modality is reported: the fusion MLP is non-linear, the two ablation
effects do not add up, and no method used here supports such a split.
Optional attributions THROUGH the fusion model: Integrated Gradients over the T05 encoder with M04's
image features held fixed, and occlusion over the image with M04's text features held fixed.
"""

import numpy as np
import torch

from src.inference import image as image_inference
from src.inference import multimodal as mm_inference
from src.inference import text as text_inference
from src.inference.multimodal import SCORE_TYPES, MissingModalityError
from src.models.image_encoder import LABEL_NAMES
from src.preprocessing.image import load_image

AGREEMENT = {0: "AGREE", 1: "PARTIAL DISAGREEMENT", 2: "DISAGREE"}
LIMITATIONS = [
    "Model attribution, not proof of how the models reason or of what makes a meme harmful.",
    "Neutralising a modality with its TRAIN mean shows how much M04 depends on that input for THIS meme; "
    "it is not a percentage contribution and the two effects do not add up.",
    "T05, I03 and M04 scores are uncalibrated; comparing their sizes across models is not meaningful.",
    "Agreement between models is not evidence of correctness: all three can be wrong together.",
]


def modality_agreement(text_label: int, image_label: int, multimodal_label: int) -> str:
    """0, 1 or 2 single-modality models disagree with the multimodal label -> AGREE / PARTIAL / DISAGREE."""
    return AGREEMENT[int(text_label != multimodal_label) + int(image_label != multimodal_label)]


def dependence(full_label: int, label_without_text: int, label_without_image: int) -> str:
    """Which neutralisation changes M04's label (a label-flip summary, not a contribution split)."""
    t, i = label_without_text != full_label, label_without_image != full_label
    return {(False, False): "label unchanged when either modality is neutralised",
            (True, False): "label depends on the text", (False, True): "label depends on the image",
            (True, True): "label needs both modalities"}[(t, i)]


def fusion_logit_difference(model, standardizers, text_feat: np.ndarray, image_feat: np.ndarray) -> np.ndarray:
    """M04 log-odds of 'misogynistic' for raw (unstandardised) feature rows."""
    t_std, i_std = standardizers
    with torch.no_grad():
        logits = model(torch.from_numpy(t_std.transform(text_feat)), torch.from_numpy(i_std.transform(image_feat))).numpy()
    return logits[:, 1] - logits[:, 0]


def ablation(model, standardizers, text_feat: np.ndarray, image_feat: np.ndarray) -> dict:
    """Full / text-neutralised / image-neutralised M04 outputs for ONE meme (train-mean replacement)."""
    t_mean, i_mean = standardizers[0].mean[None], standardizers[1].mean[None]
    rows = {"full": (text_feat, image_feat), "text_neutralised": (t_mean, image_feat),
            "image_neutralised": (text_feat, i_mean)}
    out = {}
    for name, (t, i) in rows.items():
        lo = float(fusion_logit_difference(model, standardizers, t.astype(np.float32), i.astype(np.float32))[0])
        out[name] = {"predicted_label": int(lo > 0), "logit_difference": round(lo, 6),
                     "score": round(float(1 / (1 + np.exp(-lo))), 6)}
    out["label_changes_without_text"] = out["text_neutralised"]["predicted_label"] != out["full"]["predicted_label"]
    out["label_changes_without_image"] = out["image_neutralised"]["predicted_label"] != out["full"]["predicted_label"]
    out["dependence"] = dependence(out["full"]["predicted_label"], out["text_neutralised"]["predicted_label"],
                                   out["image_neutralised"]["predicted_label"])
    return out


def fusion_text_forward(t05_model, model, standardizers, image_feat: np.ndarray):
    """forward(inputs_embeds, mask) for IG: T05 encoder -> pooled [CLS] -> M04 with the image fixed."""
    t_std, i_std = standardizers
    mean, std = torch.from_numpy(t_std.mean), torch.from_numpy(t_std.std)
    image = torch.from_numpy(i_std.transform(image_feat.astype(np.float32)))

    def forward(inputs_embeds, attention_mask):
        pooled = t05_model.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).pooler_output
        logits = model((pooled - mean) / std, image.expand(len(pooled), -1))
        return logits[:, 1] - logits[:, 0]
    return forward


def explain_multimodal(image, text, model_version: str = "M04", text_model: str = "T05", image_model: str = "I03",
                       attributions: bool = False, ig_steps: int = 32, window: int = 56, stride: int = 28,
                       registry_path=None, text_registry=None, image_registry=None) -> dict:
    """Unimodal comparison + modality ablation (+ optional attributions through M04) for one meme."""
    registry_path = registry_path or mm_inference.REGISTRY_PATH
    text_registry = text_registry or text_inference.REGISTRY_PATH
    image_registry = image_registry or image_inference.REGISTRY_PATH
    if image is None:
        raise MissingModalityError("image is missing; the multimodal explanation needs both image and text")
    if text is None or not isinstance(text, str):
        raise MissingModalityError("text is missing; supply the caption or OCR text (an empty string is allowed)")
    entry, clf = mm_inference._classifier(model_version, str(registry_path))
    if entry["kind"] != "fusion_mlp":
        raise ValueError(f"multimodal explanations are implemented for fusion MLPs, not {entry['kind']!r}")
    model, standardizers, _ = clf
    img = load_image(image)
    preds = {"text": text_inference.predict_text(text, text_model, text_registry),
             "image": image_inference.predict_image(img, image_model, image_registry),
             "multimodal": mm_inference.predict_multimodal(img, text, model_version, registry_path)}
    from src.models.image_encoder import image_embeddings

    t_feat, text_empty = mm_inference.text_features(text, entry)
    processor, encoder = mm_inference._image_encoder(entry["image_encoder"], entry["image_encoder_revision"])
    i_feat = image_embeddings([img], processor, encoder, batch_size=1)
    abl = ablation(model, standardizers, t_feat, i_feat)
    labels = {k: v["predicted_label"] for k, v in preds.items()}
    result = {
        "model_version": model_version,
        "task": "misogyny_detection",
        "predictions": {k: {"model_version": v["model_version"], "predicted_label": v["predicted_label"],
                            "predicted_name": v["predicted_name"], "score": v["score"], "score_type": v["score_type"],
                            "calibrated": False} for k, v in preds.items()},
        "modality_agreement": modality_agreement(labels["text"], labels["image"], labels["multimodal"]),
        "single_modality_models_disagreeing": [k for k in ("text", "image") if labels[k] != labels["multimodal"]],
        "ablation": {"method": "modality features replaced by their TRAIN mean (0 after M04's standardiser)", **abl},
        "processed_text_empty": text_empty,
        "consistent_with_inference": abl["full"]["predicted_label"] == labels["multimodal"],
        "limitations": LIMITATIONS,
    }
    if attributions:
        from src.explainability.image import occlusion_embeddings, occlusion_map, pixel_values_for, top_regions
        from src.explainability.text import transformer_attribution
        from src.models.transformer_text import preprocess_all

        t05, tokenizer = mm_inference._text_encoder(entry["text_representation"], entry["text_source"],
                                                    entry["text_encoder_revision"])
        processed = preprocess_all([text], entry["text_preprocessing_mode"], entry.get("text_preprocessing_options"))[0]
        ig = transformer_attribution(t05, tokenizer, processed, entry["max_length"], ig_steps,
                                     forward=fusion_text_forward(t05, model, standardizers, i_feat))
        pv = pixel_values_for(img, processor)
        full, occluded, positions = occlusion_embeddings(pv, encoder, window, stride)
        s_full = float(fusion_logit_difference(model, standardizers, t_feat, full)[0])
        s_occ = fusion_logit_difference(model, standardizers, np.repeat(t_feat, len(occluded), 0), occluded)
        cell_map = occlusion_map(s_full, s_occ, positions, pv.shape[-1], window, stride)
        result["attributions_through_fusion"] = {
            "text": {"method": "integrated_gradients through T05 encoder -> M04 (image features fixed)", **ig},
            "image": {"method": "occlusion through SigLIP -> M04 (text features fixed)", "window_px": window,
                      "stride_px": stride, "heatmap": [[round(float(v), 6) for v in r] for r in cell_map],
                      "top_regions_towards_misogynistic": top_regions(cell_map, 3, +1)},
        }
    return result


__all__ = ["explain_multimodal", "modality_agreement", "dependence", "ablation", "fusion_logit_difference",
           "fusion_text_forward", "AGREEMENT", "LABEL_NAMES", "SCORE_TYPES"]
