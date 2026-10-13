"""Advanced error taxonomy (Task 15): deterministic flags + documented primary category.

Two layers, kept separate on purpose:
1. **Automatic flags** (`auto_flags`): every rule-based (category, subcategory) that applies to an
   error. Several can apply to one meme, so flag counts do not add up to 100%.
2. **Primary category** (`primary_category`): ONE category per M04 error, for the distribution table.
   A manual review label wins when it exists (it describes the meaning, which rules cannot see).
   Otherwise the first matching rule in PRIORITY is used. The priority is a documented convention,
   not a measured cause.

Manual labels are a single reviewer's judgement on a sample of errors; they are not objective ground
truth. Categories without any supporting rule or manual label (e.g. "class imbalance": the TEST set is
balanced) are not used.
"""

TAXONOMY = {
    "LINGUISTIC": ["Hinglish spelling variant", "negation", "slang", "sarcasm", "ambiguity", "very short text",
                   "Devanagari script", "gender-term false alarm"],
    "VISUAL": ["low resolution", "text-heavy meme", "gender-presence false alarm"],
    "MULTIMODAL": ["image-text dependency", "followed the wrong modality", "fusion failure", "all models wrong",
                   "gender cues in both modalities", "corrected by fusion (single-modality error only)"],
    "OCR": ["missing text", "text-region failure", "OCR text changes the prediction", "high OCR disagreement",
            "low OCR confidence"],
    "CONTEXTUAL": ["cultural reference", "implicit meaning"],
    "DATA": ["possible annotation ambiguity", "watermark artifact", "duplicate/template group"],
}

# manual review labels (Task 13 + Task 15) -> taxonomy
MANUAL_MAP = {
    "slang": ("LINGUISTIC", "slang"),
    "sarcasm": ("LINGUISTIC", "sarcasm"),
    "ambiguous content": ("LINGUISTIC", "ambiguity"),
    "short text": ("OCR", "missing text"),
    "cultural/contextual reference": ("CONTEXTUAL", "cultural reference"),
    "implicit meaning": ("CONTEXTUAL", "implicit meaning"),
    "image-text interaction": ("MULTIMODAL", "image-text dependency"),
    "text-dependent meaning": ("MULTIMODAL", "image-text dependency"),
    "possible annotation ambiguity": ("DATA", "possible annotation ambiguity"),
    "other: woman pictured (gender-presence false alarm)": ("VISUAL", "gender-presence false alarm"),
    "gender-presence false alarm": ("VISUAL", "gender-presence false alarm"),
    "text-heavy meme": ("VISUAL", "text-heavy meme"),
    "OCR damage": ("OCR", "OCR text changes the prediction"),
    "Hinglish spelling variant": ("LINGUISTIC", "Hinglish spelling variant"),
    "negation": ("LINGUISTIC", "negation"),
    "text gender-term false alarm": ("LINGUISTIC", "gender-term false alarm"),
    "gender cues in both modalities": ("MULTIMODAL", "gender cues in both modalities"),
    "incomplete caption": ("OCR", "text-region failure"),
    "no clear content cause": None,          # reviewed, but no content-based cause: the automatic rule decides
}


def wrong(row: dict, model: str) -> bool:
    return int(row[f"{model}_prediction"]) != int(row["true_label"])


def auto_flags(row: dict) -> list[tuple[str, str]]:
    """All rule-based (category, subcategory) pairs for one sample (see the module docstring)."""
    t, i, m = (wrong(row, k) for k in ("text", "image", "multimodal"))
    f = []
    if row.get("caption_empty"):
        f.append(("OCR", "missing text"))
    if row.get("ocr_changes_prediction"):
        f.append(("OCR", "OCR text changes the prediction"))
    if row.get("ocr_high_cer"):
        f.append(("OCR", "high OCR disagreement"))
    if row.get("ocr_low_confidence"):
        f.append(("OCR", "low OCR confidence"))
    if m and not t and not i:
        f.append(("MULTIMODAL", "fusion failure"))
    elif m and t != i:
        f.append(("MULTIMODAL", "followed the wrong modality"))
    elif m:
        f.append(("MULTIMODAL", "all models wrong"))
    elif t or i:
        f.append(("MULTIMODAL", "corrected by fusion (single-modality error only)"))
    for key, sub in (("has_negation", "negation"), ("has_hinglish_variant", "Hinglish spelling variant"),
                     ("very_short_text", "very short text"), ("has_devanagari", "Devanagari script")):
        if row.get(key):
            f.append(("LINGUISTIC", sub))
    for key, sub in (("low_resolution", "low resolution"), ("text_heavy", "text-heavy meme")):
        if row.get(key):
            f.append(("VISUAL", sub))
    for key, sub in (("has_watermark", "watermark artifact"), ("template_group", "duplicate/template group")):
        if row.get(key):
            f.append(("DATA", sub))
    return f


# first match wins for samples WITHOUT a manual label (documented convention)
PRIORITY = [("OCR", "missing text"), ("MULTIMODAL", "fusion failure"), ("OCR", "OCR text changes the prediction"),
            ("MULTIMODAL", "followed the wrong modality"), ("LINGUISTIC", "negation"),
            ("LINGUISTIC", "Hinglish spelling variant"), ("LINGUISTIC", "very short text"),
            ("VISUAL", "low resolution"), ("VISUAL", "text-heavy meme"), ("DATA", "watermark artifact"),
            ("DATA", "duplicate/template group"), ("MULTIMODAL", "all models wrong"),
            ("MULTIMODAL", "corrected by fusion (single-modality error only)")]


def primary_category(row: dict, flags: list | None = None, manual_label: str | None = None) -> dict:
    """One (category, subcategory, source) per sample: manual label if mapped, else the first PRIORITY rule."""
    if manual_label:
        if manual_label not in MANUAL_MAP:
            raise ValueError(f"unknown manual label {manual_label!r}; add it to MANUAL_MAP")
        if MANUAL_MAP[manual_label] is not None:
            c, s = MANUAL_MAP[manual_label]
            return {"error_category": c, "error_subcategory": s, "category_source": "manual review"}
    flags = auto_flags(row) if flags is None else flags
    for pair in PRIORITY:
        if pair in flags:
            return {"error_category": pair[0], "error_subcategory": pair[1],
                    "category_source": "automatic rule" + (" (manual review: no clear content cause)" if manual_label else "")}
    return {"error_category": "UNCATEGORISED", "error_subcategory": "no rule applies", "category_source": "automatic rule"}


def manual_review_sample(sample_ids, error_types, already_reviewed, n_total: int, seed: int) -> list[str]:
    """Stratified random sample (half FP, half FN) of errors not reviewed before; deterministic for a seed."""
    import numpy as np

    rng = np.random.default_rng(seed)
    chosen = []
    for kind in ("false_positive", "false_negative"):
        pool = sorted(s for s, t in zip(sample_ids, error_types) if t == kind and s not in set(already_reviewed))
        chosen += list(rng.choice(pool, size=min(n_total // 2, len(pool)), replace=False))
    return chosen


def error_type(row: dict, model: str) -> str | None:
    """FP / FN / None for one model on one sample."""
    if not wrong(row, model):
        return None
    return "false_positive" if int(row[f"{model}_prediction"]) == 1 else "false_negative"
