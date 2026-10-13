"""Exploratory data analysis statistics for the cleaned primary dataset.

Read-only: every function computes statistics from the cleaned
metadata (and the referenced raw images) and returns new objects. The
cleaned CSV and the images are never modified.

EDA-ONLY TOKENIZATION: `eda_tokens()` lowercases and splits text into
Latin-letter and Devanagari runs purely to COUNT words. It is not the
project's preprocessing (that is a later task, decided by ablation
experiments) and its output is never written back to the dataset.

Language/script indicators are HEURISTIC (character ranges and small
marker-word lists), not validated language identification.
"""

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inspection import (
    DEVANAGARI_RE,
    EMOJI_RE,
    ENGLISH_MARKERS,
    HASHTAG_RE,
    LATIN_RE,
    MENTION_RE,
    ROMANIZED_HINDI_MARKERS,
    URL_RE,
    _language_indicator,
    filename_label_association,
)
from src.utils.config import REPO_ROOT, load_config

PERCENTILES = [0.25, 0.5, 0.75, 0.90, 0.95]

# EDA-only token pattern: runs of Latin letters, or runs of Devanagari
# letters/vowel signs (U+0900-U+0963, U+0971-U+097F; excludes danda and
# Devanagari digits). Digits and punctuation are not tokens.
EDA_TOKEN_RE = re.compile(r"[a-z]+|[ऀ-ॣॱ-ॿ]+")
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1+")
REPEATED_CHAR_RE = re.compile(r"([^\W\d_])\1{2,}")      # same letter 3+ times
DIGIT_RE = re.compile(r"[0-9०-९]")
HANDLE_RE = re.compile(r"(?<!\w)@(\w+)")


# ------------------------------------------------------------------
# Loading and input verification
# ------------------------------------------------------------------

def load_cleaned(config: dict | None = None) -> pd.DataFrame:
    """Load the Task 4 cleaned metadata with proper types (file is not changed)."""
    config = config or load_config()
    ds = config["dataset"]
    df = pd.read_csv(REPO_ROOT / ds["cleaned_file"], dtype=str, keep_default_na=False,
                     encoding="utf-8")
    df["target_label"] = df["target_label"].astype(int)
    df["has_text"] = df["has_text"].map({"True": True, "False": False})
    for col in ds["secondary_label_columns"]:
        df[col] = pd.to_numeric(df[col].replace("", pd.NA)).astype("Int64")
    return df


def class_names(config: dict) -> dict[int, str]:
    ds = config["dataset"]
    return {int(ds["label_mapping"][raw]): name for raw, name in ds["labels"].items()}


def verify_cleaned(df: pd.DataFrame, config: dict, fingerprint: dict) -> dict:
    """Check that the EDA input is exactly the Task 4 output."""
    import hashlib

    ds = config["dataset"]
    path = REPO_ROOT / ds["cleaned_file"]
    required = ["sample_id", "image_path", "original_text", "has_text", "original_label",
                "target_label", "duplicate_group", "review_flags", *ds["secondary_label_columns"]]
    return {
        "file_sha256_matches_task4_fingerprint":
            hashlib.sha256(path.read_bytes()).hexdigest() == fingerprint["output"]["cleaned_sha256"],
        "row_count_matches_task4": len(df) == fingerprint["output"]["rows"],
        "required_columns_present": set(required) <= set(df.columns),
        "sample_ids_unique": bool(df["sample_id"].is_unique),
        "target_labels_valid": bool(df["target_label"].isin(class_names(config)).all()),
        "all_image_files_exist": bool(df["image_path"].map(lambda p: (REPO_ROOT / p).is_file()).all()),
    }


# ------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------

def class_distribution(df: pd.DataFrame, names: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    counts = df["target_label"].value_counts().sort_index()
    table = pd.DataFrame({
        "target_label": counts.index,
        "label_name": [names[i] for i in counts.index],
        "count": counts.values,
        "percent": (100 * counts / counts.sum()).round(2).values,
    })
    ratio = {
        "total": int(counts.sum()),
        "majority_class": names[int(counts.idxmax())],
        "minority_class": names[int(counts.idxmin())],
        "imbalance_ratio_majority_to_minority": round(counts.max() / counts.min(), 4),
    }
    return table, ratio


def secondary_label_stats(df: pd.DataFrame, columns: list[str], positive: int = 1) -> dict:
    pos = df[df["target_label"] == positive]
    per_label = pd.DataFrame([{
        "label": c,
        "count_all_samples": int((df[c] == 1).sum()),
        "percent_all_samples": round(100 * (df[c] == 1).mean(), 2),
        "count_within_positive_class": int((pos[c] == 1).sum()),
        "percent_within_positive_class": round(100 * (pos[c] == 1).mean(), 2),
        "count_in_negative_class": int((df.loc[df["target_label"] != positive, c] == 1).sum()),
    } for c in columns])
    combos = Counter("+".join(c for c in columns if row[c] == 1) or "(none)"
                     for _, row in pos[columns].iterrows())
    co = pd.DataFrame(0, index=columns, columns=columns)
    for a in columns:
        for b in columns:
            co.loc[a, b] = int(((pos[a] == 1) & (pos[b] == 1)).sum())
    n_labels = (pos[columns] == 1).sum(axis=1)
    return {
        "per_label": per_label,
        "combinations": pd.DataFrame(combos.most_common(), columns=["combination", "count"]),
        "cooccurrence_within_positive": co,
        "labels_per_positive_sample": n_labels.value_counts().sort_index().to_dict(),
    }


# ------------------------------------------------------------------
# Text features (computed on the fly from original_text)
# ------------------------------------------------------------------

def eda_tokens(text: str) -> list[str]:
    """EDA-only tokenization for counting (see module docstring)."""
    return EDA_TOKEN_RE.findall(str(text).lower())


def _uppercase_heavy(text: str) -> bool:
    letters = re.findall(r"[A-Za-z]", text)
    return len(letters) >= 5 and sum(c.isupper() for c in letters) / len(letters) >= 0.7


def _devanagari_letter_share(text: str) -> float:
    dev = len(DEVANAGARI_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    return dev / (dev + lat) if dev + lat else math.nan


def text_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sample with simple, documented text measurements."""
    text = df["original_text"].astype(str)
    tokens = text.map(lambda t: set(eda_tokens(t)))
    return pd.DataFrame({
        "sample_id": df["sample_id"],
        "target_label": df["target_label"],
        "has_text": df["has_text"],
        "char_count": text.str.len(),                      # raw characters, as stored
        "word_count": text.str.split().str.len(),          # whitespace-separated chunks
        "has_url": text.str.contains(URL_RE),
        "has_mention": text.str.contains(MENTION_RE),
        "has_hashtag": text.str.contains(HASHTAG_RE),
        "has_emoji": text.str.contains(EMOJI_RE),
        "has_repeated_punctuation": text.map(lambda t: bool(REPEATED_PUNCT_RE.search(t))),
        "has_repeated_characters": text.map(lambda t: bool(REPEATED_CHAR_RE.search(t))),
        "is_uppercase_heavy": text.map(_uppercase_heavy),
        "has_digit": text.str.contains(DIGIT_RE),
        "script_category": [(_language_indicator(t) if h else "empty") for t, h in zip(text, df["has_text"])],
        "devanagari_letter_share": text.map(_devanagari_letter_share),
        "has_romanized_hindi_marker": tokens.map(lambda s: bool(s & ROMANIZED_HINDI_MARKERS)),
        "has_english_marker": tokens.map(lambda s: bool(s & ENGLISH_MARKERS)),
    })


def describe(values: pd.Series) -> dict:
    values = pd.Series(values).dropna().astype(float)
    if values.empty:
        return {"count": 0}
    q = values.quantile(PERCENTILES)
    return {
        "count": int(len(values)), "mean": round(values.mean(), 3), "std": round(values.std(), 3),
        "min": float(values.min()), "p25": float(q[0.25]), "median": float(q[0.5]),
        "p75": float(q[0.75]), "p90": float(q[0.90]), "p95": float(q[0.95]), "max": float(values.max()),
    }


def length_statistics(features: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Character/word statistics on NON-EMPTY texts, overall and per class."""
    rows = []
    present = features[features["has_text"]]
    for metric in ("char_count", "word_count"):
        rows.append({"metric": metric, "group": "all", **describe(present[metric])})
        for label, name in names.items():
            rows.append({"metric": metric, "group": name,
                         **describe(present.loc[present["target_label"] == label, metric])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------------

def ngram_counts(texts, n: int = 1) -> Counter:
    counts = Counter()
    for text in texts:
        toks = eda_tokens(text)
        counts.update(toks if n == 1 else (" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)))
    return counts


def script_of(token: str) -> str:
    return "devanagari" if DEVANAGARI_RE.search(token) else "latin"


def top_terms(df: pd.DataFrame, names: dict[int, str], n: int = 1, top: int = 30) -> pd.DataFrame:
    """Raw frequency (overall and per class) with document frequency."""
    rows = []
    groups = [("all", df)] + [(name, df[df["target_label"] == label]) for label, name in names.items()]
    for group, part in groups:
        counts = ngram_counts(part["original_text"], n)
        doc_freq = Counter()
        for t in part["original_text"]:
            toks = eda_tokens(t)
            grams = toks if n == 1 else [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
            doc_freq.update(set(grams))
        total = sum(counts.values())
        for rank, (term, c) in enumerate(counts.most_common(top), start=1):
            rows.append({"group": group, "ngram": n, "rank": rank, "term": term, "script": script_of(term),
                         "count": c, "per_1000_tokens": round(1000 * c / total, 3),
                         "samples_containing": doc_freq[term]})
    return pd.DataFrame(rows)


def distinctive_terms(df: pd.DataFrame, names: dict[int, str], positive: int = 1,
                      min_count: int = 10, top: int = 20) -> pd.DataFrame:
    """Weighted log-odds ratio with an informative Dirichlet prior.

    Compares term usage between the two classes while accounting for
    term frequency, so very rare or very common words do not dominate
    (Monroe, Colaresi & Quinn, 2008, "Fightin' Words", Political
    Analysis). The prior is the pooled counts of both classes. The
    z-scores are DESCRIPTIVE: no significance threshold or
    multiple-comparison correction is applied.
    """
    a = ngram_counts(df.loc[df["target_label"] == positive, "original_text"])
    b = ngram_counts(df.loc[df["target_label"] != positive, "original_text"])
    prior = a + b
    na, nb, a0 = sum(a.values()), sum(b.values()), sum(prior.values())
    rows = []
    for term, alpha in prior.items():
        ya, yb = a[term], b[term]
        if ya + yb < min_count:
            continue
        delta = (math.log((ya + alpha) / (na + a0 - ya - alpha))
                 - math.log((yb + alpha) / (nb + a0 - yb - alpha)))
        z = delta / math.sqrt(1 / (ya + alpha) + 1 / (yb + alpha))
        rows.append({"term": term, "script": script_of(term), "z_score": round(z, 3),
                     f"count_{names[positive]}": ya,
                     f"count_{[v for k, v in names.items() if k != positive][0]}": yb})
    table = pd.DataFrame(rows).sort_values("z_score", ascending=False)
    pos_name = names[positive]
    neg_name = [v for k, v in names.items() if k != positive][0]
    return pd.concat([
        table.head(top).assign(associated_with=pos_name),
        table.tail(top).iloc[::-1].assign(associated_with=neg_name),
    ], ignore_index=True)


def vocabulary_summary(df: pd.DataFrame, names: dict[int, str]) -> dict:
    out = {}
    for group, part in [("all", df)] + [(n, df[df["target_label"] == l]) for l, n in names.items()]:
        counts = ngram_counts(part["original_text"])
        out[group] = {
            "tokens": int(sum(counts.values())),
            "vocabulary_size": len(counts),
            "latin_types": sum(1 for t in counts if script_of(t) == "latin"),
            "devanagari_types": sum(1 for t in counts if script_of(t) == "devanagari"),
            "hapax_legomena": sum(1 for c in counts.values() if c == 1),
        }
    return out


# ------------------------------------------------------------------
# Class comparisons
# ------------------------------------------------------------------

BOOLEAN_FEATURES = ["has_url", "has_mention", "has_hashtag", "has_emoji", "has_repeated_punctuation",
                    "has_repeated_characters", "is_uppercase_heavy", "has_digit",
                    "has_romanized_hindi_marker", "has_english_marker"]


def feature_by_class(features: pd.DataFrame, names: dict[int, str], columns=BOOLEAN_FEATURES,
                     only_with_text: bool = True) -> pd.DataFrame:
    """Count and percentage of samples with each boolean feature, per class."""
    data = features[features["has_text"]] if only_with_text else features
    rows = []
    for col in columns:
        row = {"feature": col}
        for label, name in names.items():
            part = data[data["target_label"] == label]
            row[f"count_{name}"] = int(part[col].sum())
            row[f"percent_{name}"] = round(100 * part[col].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


SCRIPT_ORDER = ["devanagari_only", "mixed_script", "latin_hindi_markers_only", "latin_hindi_and_english_markers",
                "latin_english_markers_only", "latin_no_markers", "no_letters", "empty"]


def script_by_class(features: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    rows = []
    present = set(features["script_category"])
    for cat in [c for c in SCRIPT_ORDER if c in present] + sorted(present - set(SCRIPT_ORDER)):
        row = {"script_category": cat}
        for label, name in names.items():
            part = features[features["target_label"] == label]
            row[f"count_{name}"] = int((part["script_category"] == cat).sum())
            row[f"percent_{name}"] = round(100 * (part["script_category"] == cat).mean(), 2)
        row["count_all"] = int((features["script_category"] == cat).sum())
        row["percent_all"] = round(100 * (features["script_category"] == cat).mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Images (read-only; header access, no image is altered or saved)
# ------------------------------------------------------------------

def image_metadata(df: pd.DataFrame, repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    from PIL import Image

    rows = []
    for sid, rel, label in zip(df["sample_id"], df["image_path"], df["target_label"]):
        path = repo_root / rel
        with Image.open(path) as img:
            w, h = img.size
            rows.append({"sample_id": sid, "target_label": label, "width": w, "height": h,
                         "aspect_ratio": w / h, "format": img.format, "mode": img.mode,
                         "file_bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def image_statistics(meta: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    rows = []
    for metric in ("width", "height", "aspect_ratio", "file_bytes"):
        rows.append({"metric": metric, "group": "all", **describe(meta[metric])})
        for label, name in names.items():
            rows.append({"metric": metric, "group": name,
                         **describe(meta.loc[meta["target_label"] == label, metric])})
    return pd.DataFrame(rows)


def common_resolutions(meta: pd.DataFrame, names: dict[int, str], top: int = 10) -> pd.DataFrame:
    """Most frequent exact (width x height) sizes and how they split by class."""
    res = meta["width"].astype(str) + "x" + meta["height"].astype(str)
    table = pd.crosstab(res, meta["target_label"]).rename(columns=names)
    table["samples"] = table.sum(axis=1)
    table = table.sort_values(["samples"], ascending=False).head(top)
    for name in names.values():
        table[f"share_{name}"] = (table[name] / table["samples"]).round(3)
    return table.reset_index(names="resolution")


def dhash(path: Path, size: int = 8) -> int:
    """Difference hash computed IN MEMORY on a small grayscale copy.

    Used only to find near-duplicate images (same template, re-encoded,
    resized). The image file is not modified and no resized image is saved.
    """
    from PIL import Image

    with Image.open(path) as img:
        small = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def image_dhashes(df: pd.DataFrame, repo_root: Path = REPO_ROOT) -> np.ndarray:
    return np.array([dhash(repo_root / p) for p in df["image_path"]], dtype=np.uint64)


def near_duplicate_images(df: pd.DataFrame, max_distance: int = 4, repo_root: Path = REPO_ROOT,
                          hashes: np.ndarray | None = None) -> pd.DataFrame:
    """Pairs of images whose 64-bit dHashes differ in <= max_distance bits.

    These are CANDIDATE near-duplicates (same template, re-encoded or
    resized copies); they need manual confirmation.
    """
    hashes = image_dhashes(df, repo_root) if hashes is None else hashes
    bits = np.unpackbits(hashes.view(np.uint8).reshape(-1, 8), axis=1)
    pairs = []
    for i in range(len(bits) - 1):
        dist = np.count_nonzero(bits[i + 1:] != bits[i], axis=1)
        for j in np.nonzero(dist <= max_distance)[0]:
            k = i + 1 + int(j)
            pairs.append({"sample_id_a": df["sample_id"].iloc[i], "sample_id_b": df["sample_id"].iloc[k],
                          "hamming_distance": int(dist[j]),
                          "label_a": int(df["target_label"].iloc[i]), "label_b": int(df["target_label"].iloc[k])})
    out = pd.DataFrame(pairs, columns=["sample_id_a", "sample_id_b", "hamming_distance", "label_a", "label_b"])
    out["cross_class"] = out["label_a"] != out["label_b"]
    # Same caption too? (template reused with identical text vs. different text)
    key = dict(zip(df["sample_id"], df["original_text"].map(lambda t: " ".join(str(t).casefold().split()))))
    out["same_text"] = [key[a] == key[b] for a, b in zip(out["sample_id_a"], out["sample_id_b"])]
    group = dict(zip(df["sample_id"], df["duplicate_group"]))
    out["already_grouped_in_task4"] = [bool(group[a]) and group[a] == group[b]
                                       for a, b in zip(out["sample_id_a"], out["sample_id_b"])]
    return out


# ------------------------------------------------------------------
# Leakage signals and outliers
# ------------------------------------------------------------------

def mention_handles(df: pd.DataFrame, names: dict[int, str], min_samples: int = 5) -> pd.DataFrame:
    """@handles (often page watermarks) and how their samples split by class."""
    rows = Counter()
    for text, label in zip(df["original_text"], df["target_label"]):
        for h in set(m.lower() for m in HANDLE_RE.findall(text)):
            rows[(h, label)] += 1
    handles = sorted({h for h, _ in rows})
    table = []
    for h in handles:
        counts = {name: rows[(h, label)] for label, name in names.items()}
        total = sum(counts.values())
        if total >= min_samples:
            table.append({"handle": "@" + h, "samples": total, **{f"count_{k}": v for k, v in counts.items()},
                          "max_class_share": round(max(counts.values()) / total, 3)})
    return pd.DataFrame(table).sort_values(["samples", "handle"], ascending=[False, True]) \
        if table else pd.DataFrame(columns=["handle", "samples", "max_class_share"])


def id_pattern_association(df: pd.DataFrame) -> dict:
    return filename_label_association(df["image_reference"], df["target_label"])


def iqr_upper(values: pd.Series, k: float = 1.5) -> float:
    q1, q3 = values.quantile([0.25, 0.75])
    return float(q3 + k * (q3 - q1))


def outliers(features: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Samples that are unusual on one measure. Recorded for review, never removed."""
    present = features[features["has_text"]]
    rules = []
    long_cut = iqr_upper(present["char_count"])
    rules.append(("text_very_long", present, "char_count", lambda v: v > long_cut, f"> {long_cut:g} chars (Q3 + 1.5 IQR)"))
    rules.append(("text_very_short", present, "char_count", lambda v: v <= 3, "<= 3 chars (non-empty)"))
    rules.append(("image_small", meta.assign(min_side=meta[["width", "height"]].min(axis=1)), "min_side",
                  lambda v: v < 200, "shorter side < 200 px"))
    rules.append(("aspect_ratio_extreme", meta, "aspect_ratio", lambda v: (v < 0.5) | (v > 2.0),
                  "w/h < 0.5 or > 2.0"))
    big_cut = iqr_upper(meta["file_bytes"], k=3.0)
    rules.append(("image_file_large", meta, "file_bytes", lambda v: v > big_cut, f"> {big_cut:,.0f} bytes (Q3 + 3 IQR)"))
    rows = []
    for name, data, col, test, rule in rules:
        hit = data[test(data[col])]
        for sid, value, label in zip(hit["sample_id"], hit[col], hit["target_label"]):
            rows.append({"category": name, "sample_id": sid, "target_label": int(label),
                         "value": round(float(value), 4), "rule": rule})
    return pd.DataFrame(rows, columns=["category", "sample_id", "target_label", "value", "rule"])


def availability(df: pd.DataFrame) -> dict:
    image_ok = df["image_path"].map(lambda p: (REPO_ROOT / p).is_file())
    return {
        "samples": int(len(df)),
        "with_usable_text": int(df["has_text"].sum()),
        "with_usable_image": int(image_ok.sum()),
        "with_text_and_image": int((df["has_text"] & image_ok).sum()),
        "image_only_experiment_samples": int(image_ok.sum()),
        "multimodal_experiment_samples": int(image_ok.sum()),
        "text_only_experiment_samples_nonempty_text": int(df["has_text"].sum()),
    }
