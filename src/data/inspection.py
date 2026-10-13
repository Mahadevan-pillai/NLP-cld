"""Read-only dataset inspection functions.

Everything here only READS files: nothing under data/raw is modified,
moved or rewritten. The functions measure structure, labels, text,
images and duplicates so that dataset statistics are never assumed.

Text/language indicators are simple heuristics (character scripts and
small marker-word lists). They give an approximate picture of the data,
not a validated language identification.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
# Common emoji / pictograph blocks (approximate, "where practical").
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF]"
)
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[a-z]+")

# Heuristic marker words. Romanized-Hindi words are frequent function
# words that are rare in English; English markers are common stopwords.
ROMANIZED_HINDI_MARKERS = {
    "hai", "hain", "nahi", "nhi", "kya", "kyu", "kyun", "mein", "mai", "main",
    "aur", "bhi", "yeh", "ye", "woh", "wo", "toh", "kar", "karo", "kuch",
    "mera", "meri", "mere", "tera", "teri", "tum", "hum", "aap", "ap", "ka",
    "ki", "ke", "ko", "se", "tha", "thi", "hota", "hoti", "wala", "wali",
    "ladki", "ladka", "jab", "sab", "bas", "abhi",
}
ENGLISH_MARKERS = {
    "the", "is", "are", "and", "you", "to", "of", "in", "for", "with", "my",
    "me", "it", "this", "that", "when", "what", "your", "be", "on", "was",
    "not", "have", "do", "they", "she", "he", "her", "his", "we", "will",
}


# ------------------------------------------------------------------
# Files
# ------------------------------------------------------------------

def file_inventory(root: Path) -> dict:
    """Files under a dataset directory with sizes and extensions."""
    files = [p for p in Path(root).rglob("*") if p.is_file() and p.name != ".gitkeep"]
    by_ext = Counter(p.suffix.lower() or "(none)" for p in files)
    return {
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "extensions": dict(by_ext.most_common()),
        "non_image_files": sorted(
            str(p.relative_to(root)).replace("\\", "/")
            for p in files if p.suffix.lower() not in IMAGE_EXTENSIONS
        ),
        "image_directories": sorted({
            str(p.parent.relative_to(root)).replace("\\", "/")
            for p in files if p.suffix.lower() in IMAGE_EXTENSIONS
        }),
    }


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/Excel file with every cell as a raw string.

    dtype=str and keep_default_na=False keep values exactly as written
    (e.g. "0" stays "0", empty stays ""), so nothing is silently converted.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


# ------------------------------------------------------------------
# Schema and labels
# ------------------------------------------------------------------

def schema_summary(df: pd.DataFrame) -> dict:
    columns = {}
    for col in df.columns:
        values = df[col]
        missing = _is_missing(values)
        present = values[~missing]
        numeric = pd.to_numeric(present, errors="coerce").notna().all() if len(present) else False
        columns[col] = {
            "inferred_type": "numeric-like" if numeric else "text",
            "missing": int(missing.sum()),
            "unique_values": int(present.nunique()),
        }
    return {"rows": len(df), "column_count": df.shape[1], "columns": columns}


def label_summary(series: pd.Series) -> dict:
    missing = _is_missing(series)
    counts = series[~missing].value_counts().sort_index()
    total = int((~missing).sum())
    return {
        "unique_values": [str(v) for v in counts.index],
        "counts": {str(k): int(v) for k, v in counts.items()},
        "percentages": {str(k): round(100 * v / total, 2) for k, v in counts.items()} if total else {},
        "missing": int(missing.sum()),
        "type": "binary" if len(counts) == 2 else ("single-value" if len(counts) == 1 else "multiclass"),
    }


def sublabel_consistency(df: pd.DataFrame, primary: str, sublabels: list[str],
                         positive: str, id_col: str) -> dict:
    """How sub-labels relate to the primary label (no meaning is assumed).

    `positive` is only the raw value whose rows are compared; it does not
    claim what that value means.
    """
    sub = df[sublabels]
    any_sub = (sub == positive).any(axis=1)
    is_pos = df[primary] == positive
    combos = Counter(
        "+".join(c for c in sublabels if row[c] == positive) or "(none)"
        for _, row in sub[is_pos].iterrows()
    )
    return {
        "rows_primary_not_positive_but_sublabel_positive": int((~is_pos & any_sub).sum()),
        "ids_primary_not_positive_but_sublabel_positive": sorted(df.loc[~is_pos & any_sub, id_col]),
        "rows_primary_positive_with_no_sublabel": int((is_pos & ~any_sub).sum()),
        "ids_primary_positive_with_no_sublabel": sorted(df.loc[is_pos & ~any_sub, id_col]),
        "rows_with_multiple_sublabels": int(((sub == positive).sum(axis=1) > 1).sum()),
        "sublabel_combinations_when_primary_positive": dict(combos.most_common()),
    }


# ------------------------------------------------------------------
# Text
# ------------------------------------------------------------------

def _describe(values: pd.Series) -> dict:
    if values.empty:
        return {}
    q = values.quantile([0.25, 0.5, 0.75, 0.95])
    stats = {
        "min": values.min(), "p25": q[0.25], "median": q[0.5], "mean": values.mean(),
        "p75": q[0.75], "p95": q[0.95], "max": values.max(),
    }
    return {k: round(float(v), 3) for k, v in stats.items()}


def _language_indicator(text: str) -> str:
    has_dev = bool(DEVANAGARI_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))
    if has_dev and has_lat:
        return "mixed_script"
    if has_dev:
        return "devanagari_only"
    if not has_lat:
        return "no_letters"
    tokens = set(TOKEN_RE.findall(text.lower()))
    hindi, english = bool(tokens & ROMANIZED_HINDI_MARKERS), bool(tokens & ENGLISH_MARKERS)
    if hindi and english:
        return "latin_hindi_and_english_markers"
    if hindi:
        return "latin_hindi_markers_only"
    if english:
        return "latin_english_markers_only"
    return "latin_no_markers"


def text_summary(series: pd.Series) -> dict:
    missing = _is_missing(series)
    text = series[~missing].astype(str)
    return {
        "non_empty": int((~missing).sum()),
        "missing_or_empty": int(missing.sum()),
        "char_length": _describe(text.str.len()),
        "word_count": _describe(text.str.split().str.len()),
        "samples_with_url": int(text.str.contains(URL_RE).sum()),
        "samples_with_mention": int(text.str.contains(MENTION_RE).sum()),
        "samples_with_hashtag": int(text.str.contains(HASHTAG_RE).sum()),
        "samples_with_emoji": int(text.str.contains(EMOJI_RE).sum()),
        "samples_with_newline": int(text.str.contains("\n").sum()),
        "language_indicator_heuristic": dict(Counter(text.map(_language_indicator)).most_common()),
    }


# ------------------------------------------------------------------
# Duplicates and integrity
# ------------------------------------------------------------------

def dup_key(text: str) -> str:
    """Comparison key for duplicate detection only (data is not changed)."""
    return " ".join(str(text).casefold().split())


_dup_key = dup_key  # backwards-compatible alias


def check_image(path: Path) -> dict:
    """Read-only check of one image: SHA-256, readability, format and size."""
    from PIL import Image

    data = Path(path).read_bytes()
    result = {"sha256": hashlib.sha256(data).hexdigest(), "readable": False,
              "format": None, "width": None, "height": None}
    try:
        with Image.open(path) as img:
            img.verify()  # integrity check; image must be reopened afterwards
        with Image.open(path) as img:
            result.update(readable=True, format=img.format, width=img.size[0], height=img.size[1])
    except Exception:
        pass
    return result


def duplicate_summary(df: pd.DataFrame, id_col: str, text_col: str | None, label_col: str | None) -> dict:
    out = {
        "duplicate_full_rows": int(df.duplicated().sum()),
        "missing_ids": int(_is_missing(df[id_col]).sum()),
        "duplicate_ids": int(df[id_col].duplicated().sum()),
        "duplicate_id_values": sorted(df.loc[df[id_col].duplicated(keep=False), id_col].unique().tolist()),
    }
    if text_col:
        present = df[~_is_missing(df[text_col])]
        exact = present[text_col].duplicated(keep=False)
        keys = present[text_col].map(_dup_key)
        norm = keys.duplicated(keep=False)
        groups = present[norm].groupby(keys[norm])
        conflicting = [g for _, g in groups if label_col and g[label_col].nunique() > 1]
        out.update({
            "rows_in_exact_duplicate_text_groups": int(exact.sum()),
            "exact_duplicate_text_groups": int(present[exact][text_col].nunique()),
            "rows_in_case_whitespace_duplicate_groups": int(norm.sum()),
            "case_whitespace_duplicate_groups": int(keys[norm].nunique()),
            "duplicate_text_groups_with_conflicting_labels": len(conflicting),
            "rows_in_conflicting_label_groups": int(sum(len(g) for g in conflicting)),
            # IDs only (no text), so the report can be shared/committed.
            "conflicting_label_group_ids": [sorted(g[id_col].tolist()) for g in conflicting],
        })
    return out


def filename_label_association(names: pd.Series, labels: pd.Series) -> dict:
    """Does the file-name pattern (digits replaced by '#') reveal the label?"""
    patterns = names.astype(str).str.replace(r"\d+", "#", regex=True)
    table = pd.crosstab(patterns, labels)
    perfectly_predictive = bool((table.gt(0).sum(axis=1) == 1).all()) and table.shape[1] > 1
    return {
        "pattern_label_counts": {p: {str(k): int(v) for k, v in row.items() if v} for p, row in table.iterrows()},
        "pattern_perfectly_predicts_label": perfectly_predictive,
    }


# ------------------------------------------------------------------
# Images
# ------------------------------------------------------------------

def image_summary(root: Path, referenced: pd.Series) -> dict:
    """Match referenced image names to files under root and check them.

    Opens each image read-only with Pillow (verify + size); computes
    SHA-256 of file bytes to find exact duplicates. No image is changed.
    """
    root = Path(root)
    on_disk = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            on_disk.setdefault(p.name, []).append(p)

    refs = referenced[~_is_missing(referenced)].astype(str)
    unique_refs = set(refs)
    found = sorted(unique_refs & set(on_disk))
    result = {
        "referenced_rows": int(len(refs)),
        "unique_referenced_names": len(unique_refs),
        "image_files_on_disk": sum(len(v) for v in on_disk.values()),
        "same_filename_in_multiple_folders": sorted(n for n, v in on_disk.items() if len(v) > 1),
        "found": len(found),
        "missing": len(unique_refs - set(on_disk)),
        "unreferenced_files_on_disk": len(set(on_disk) - unique_refs),
    }
    if not found:
        result["status"] = "NO_IMAGES_FOUND" if not on_disk else "NO_REFERENCED_IMAGES_FOUND"
        return result

    formats, widths, heights, ratios, corrupt, ext_mismatch = Counter(), [], [], [], [], []
    hashes = {}
    for name in found:
        path = on_disk[name][0]
        info = check_image(path)
        hashes.setdefault(info["sha256"], []).append(name)
        if not info["readable"]:
            corrupt.append(name)
            continue
        fmt, w, h = info["format"], info["width"], info["height"]
        formats[fmt] += 1
        widths.append(w)
        heights.append(h)
        ratios.append(w / h if h else 0)
        expected = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}, "GIF": {".gif"},
                    "WEBP": {".webp"}, "BMP": {".bmp"}, "TIFF": {".tif", ".tiff"}}.get(fmt)
        if expected and path.suffix.lower() not in expected:
            ext_mismatch.append(name)

    dup_groups = [sorted(v) for v in hashes.values() if len(v) > 1]
    result.update({
        "status": "INSPECTED",
        "corrupt_or_unreadable": len(corrupt),
        "corrupt_files": sorted(corrupt),
        "formats": dict(formats.most_common()),
        "extension_format_mismatch": len(ext_mismatch),
        "width": _describe(pd.Series(widths)),
        "height": _describe(pd.Series(heights)),
        "aspect_ratio_w_over_h": {k: round(v, 3) for k, v in _describe(pd.Series(ratios)).items()} if ratios else {},
        "exact_duplicate_image_groups": len(dup_groups),
        "images_in_exact_duplicate_groups": sum(len(g) for g in dup_groups),
        "exact_duplicate_image_names": dup_groups,
    })
    return result
