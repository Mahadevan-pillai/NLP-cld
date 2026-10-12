"""Apply the three preprocessing modes to a dataset and measure the effects.

Read-only with respect to the input DataFrame: a new DataFrame is
returned and `original_text` is copied, never modified.
"""

from collections import Counter

import pandas as pd

from src.data.eda import eda_tokens
from src.data.inspection import EMOJI_RE
from src.preprocessing import hinglish
from src.preprocessing.text import MODES, PreprocessingConfig, preprocess_with_details

TEXT_COLUMNS = {mode: f"text_{mode}" for mode in MODES}
CARRIED_COLUMNS = ["sample_id", "source_row", "image_reference", "image_path", "has_text",
                   "original_label", "target_label", "duplicate_group", "review_flags"]

# Audit thresholds (documented in docs/TEXT_PREPROCESSING_REPORT.md)
LENGTH_CHANGE_RATIO = 0.30
MANY_RULE_HITS = 3
REPLACEMENT_SHARE = 0.30


def build_variants(df: pd.DataFrame, cfg: PreprocessingConfig, secondary_labels: list[str],
                   text_column: str = "original_text") -> tuple[pd.DataFrame, dict]:
    """Return (new DataFrame with text_raw/text_basic/text_hinglish_normalized, operations).

    operations[mode] is a list (one Counter per row) of applied operations.
    """
    out = df[[c for c in CARRIED_COLUMNS if c in df.columns] + secondary_labels].copy()
    operations = {}
    for mode, col in TEXT_COLUMNS.items():
        results = [preprocess_with_details(t, mode, cfg) for t in df[text_column]]
        out[col] = [r[0] for r in results]
        operations[mode] = [r[1] for r in results]
    out["preprocessing_version"] = cfg.version
    return out, operations


def _vocab(texts) -> int:
    return len({t for text in texts for t in eda_tokens(text)})


def summary_by_mode(out: pd.DataFrame) -> pd.DataFrame:
    raw = out[TEXT_COLUMNS["raw"]]
    rows, previous = [], None
    for mode, col in TEXT_COLUMNS.items():
        text = out[col]
        changed = text != raw
        rows.append({
            "mode": mode,
            "samples": len(text),
            "changed_vs_raw": int(changed.sum()),
            "percent_changed_vs_raw": round(100 * changed.mean(), 2),
            "changed_vs_previous_mode": int((text != out[previous]).sum()) if previous else 0,
            "mean_chars": round(text.str.len().mean(), 3),
            "mean_whitespace_tokens": round(text.str.split().str.len().mean(), 3),
            "vocabulary_eda_tokens": _vocab(text),
            "vocabulary_whitespace_tokens_case_sensitive": len({w for t in text for w in t.split()}),
            "empty_texts": int((text.str.strip() == "").sum()),
            "newly_empty_vs_raw": int(((text.str.strip() == "") & (raw.str.strip() != "")).sum()),
        })
        previous = col
    return pd.DataFrame(rows)


def operation_counts(operations: dict) -> pd.DataFrame:
    rows = []
    for mode, per_row in operations.items():
        totals, samples = Counter(), Counter()
        for c in per_row:
            totals.update(c)
            samples.update(c.keys())
        for op in sorted(totals):
            if op.startswith("rule:"):
                continue
            rows.append({"mode": mode, "operation": op, "occurrences": totals[op], "samples_affected": samples[op]})
        rule_total = sum(v for k, v in totals.items() if k.startswith("rule:"))
        if rule_total:
            rows.append({"mode": mode, "operation": "hinglish_rules_total", "occurrences": rule_total,
                         "samples_affected": sum(1 for c in per_row if any(k.startswith("rule:") for k in c))})
    return pd.DataFrame(rows, columns=["mode", "operation", "occurrences", "samples_affected"])


def rule_frequencies(operations: list[Counter], rules: hinglish.RuleSet) -> pd.DataFrame:
    totals, samples = Counter(), Counter()
    for c in operations:
        for k, v in c.items():
            if k.startswith("rule:"):
                totals[k] += v
                samples[k] += 1
    rows = []
    for variant, canonical in sorted(rules.mapping.items(), key=lambda kv: (kv[1], kv[0])):
        key = f"rule:{variant}->{canonical}"
        rows.append({"canonical": canonical, "variant": variant, "occurrences": totals[key],
                     "samples_affected": samples[key], "negation": variant in rules.negation_variants})
    return pd.DataFrame(rows).sort_values(["occurrences", "variant"], ascending=[False, True]).reset_index(drop=True)


def audit(out: pd.DataFrame, operations: dict) -> pd.DataFrame:
    """Per-sample risk flags for manual review (IDs and numbers only, no text)."""
    raw, basic, hin = (out[TEXT_COLUMNS[m]] for m in MODES)
    rows = []
    for i, sid in enumerate(out["sample_id"]):
        ops = operations["hinglish_normalized"][i]
        rule_hits = sum(v for k, v in ops.items() if k.startswith("rule:"))
        n_tokens = max(len(basic.iloc[i].split()), 1)
        len_raw, len_hin = len(raw.iloc[i]), len(hin.iloc[i])
        neg_raw, neg_hin = hinglish.count_negations(raw.iloc[i]), hinglish.count_negations(hin.iloc[i])
        flags = {
            "negation_count_changed": neg_raw != neg_hin,
            "became_empty": bool(raw.iloc[i].strip()) and not hin.iloc[i].strip(),
            "large_length_change": abs(len_hin - len_raw) / max(len_raw, 1) > LENGTH_CHANGE_RATIO,
            "many_rule_replacements": rule_hits >= MANY_RULE_HITS,
            "high_replacement_share": (rule_hits + ops.get("elongation", 0)) / n_tokens >= REPLACEMENT_SHARE,
            "token_count_changed": len(raw.iloc[i].split()) != len(hin.iloc[i].split()),
        }
        if any(flags.values()):
            rows.append({"sample_id": sid, "target_label": int(out["target_label"].iloc[i]), **flags,
                         "rule_replacements": rule_hits, "elongations": ops.get("elongation", 0),
                         "mentions_masked": ops.get("mention", 0),
                         "chars_raw": len_raw, "chars_hinglish": len_hin,
                         "negations_raw": neg_raw, "negations_hinglish": neg_hin,
                         "reasons": ";".join(k for k, v in flags.items() if v)})
    return pd.DataFrame(rows)


def examples(out: pd.DataFrame, operations: dict, audit_table: pd.DataFrame, per_category: int = 3) -> pd.DataFrame:
    """Deterministic before/after examples per category (first rows in file order).

    Not cherry-picked: rows are taken in source order, and audit-flagged
    rows are included. Contains dataset text, so the file is git-ignored.
    """
    hin_ops = operations["hinglish_normalized"]
    categories = {
        "mention": lambda i, o: "mention" in o,
        "hashtag": lambda i, o: "hashtag" in o,
        "url": lambda i, o: "url" in o,
        "emoji_kept": lambda i, o: bool(EMOJI_RE.search(out["text_raw"].iloc[i])),
        "html": lambda i, o: "html_unescape" in o,
        "whitespace": lambda i, o: "whitespace" in o,
        "unicode_nfc": lambda i, o: "unicode_nfc" in o,
        "repeated_characters": lambda i, o: "elongation" in o,
        "romanized_hindi_rule": lambda i, o: any(k.startswith("rule:") and "->nahi" not in k for k in o),
        "negation_rule": lambda i, o: any(k.endswith("->nahi") for k in o),
        "negation_present_unchanged": lambda i, o: hinglish.count_negations(out["text_raw"].iloc[i]) > 0 and not o,
        "unchanged": lambda i, o: not o and bool(out["text_raw"].iloc[i].strip()),
        "audit_flagged": lambda i, o: out["sample_id"].iloc[i] in flagged,
    }
    flagged = set(audit_table["sample_id"]) if len(audit_table) else set()
    order = out["source_row"].astype(int).argsort().tolist() if "source_row" in out else range(len(out))
    rows = []
    for name, test in categories.items():
        picked = [i for i in order if test(i, hin_ops[i])][:per_category]
        for i in picked:
            rows.append({"category": name, "sample_id": out["sample_id"].iloc[i],
                         "raw": out["text_raw"].iloc[i], "basic": out["text_basic"].iloc[i],
                         "hinglish_normalized": out["text_hinglish_normalized"].iloc[i],
                         "operations": ";".join(f"{k}x{v}" for k, v in sorted(hin_ops[i].items()))})
        if not picked:
            rows.append({"category": name, "sample_id": "", "raw": "", "basic": "", "hinglish_normalized": "",
                         "operations": "NONE IN DATASET"})
    return pd.DataFrame(rows)
