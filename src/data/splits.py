"""Leakage-safe train/validation/test splitting and canonical split access.

ONE split assignment (data/splits/split_assignments.csv) is used by every
later experiment: text-only, image-only and multimodal models and all
preprocessing variants. Splitting is by sample_id, so the text variant
a model uses cannot change which split a sample is in.

Algorithm (documented in docs/DATA_SPLIT_REPORT.md):
1. Build leakage groups with union-find. Samples are joined if they
   - share an identical image file (SHA-256),
   - form a candidate near-duplicate image pair (Task 5 dHash <= 4), or
   - share a substantive identical caption (Task 4 duplicate_group with
     >= MIN_CAPTION_WORDS words). Trivial shared captions are not grouped.
2. scikit-learn StratifiedGroupKFold (n_folds, shuffle, seed) stratified
   by target_label. Whole groups go to one fold.
3. Folds are mapped to splits by ratio (e.g. 20 folds: 14 train /
   3 validation / 3 test = 70/15/15).

TEST SET POLICY: load_split("test") requires allow_test=True. The test
split is for the final, one-time evaluation of frozen configurations.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import REPO_ROOT, load_config

SPLITS = ("train", "validation", "test")
MIN_CAPTION_WORDS = 3


# ------------------------------------------------------------------
# Leakage groups
# ------------------------------------------------------------------

class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: smaller id wins
            self.parent[max(ra, rb)] = min(ra, rb)


def build_groups(df: pd.DataFrame, near_duplicate_pairs: pd.DataFrame | None = None,
                 min_caption_words: int = MIN_CAPTION_WORDS) -> tuple[pd.Series, pd.DataFrame]:
    """Return (group_id per sample, table of links that formed the groups).

    df needs: sample_id, image_sha256, duplicate_group, text_raw.
    """
    uf = _UnionFind(df["sample_id"])
    links = []

    def link(a, b, reason):
        uf.union(a, b)
        links.append({"sample_id_a": a, "sample_id_b": b, "reason": reason})

    for _, part in df.groupby("image_sha256", sort=True):
        ids = sorted(part["sample_id"])
        for other in ids[1:]:
            link(ids[0], other, "identical_image_sha256")

    if near_duplicate_pairs is not None and len(near_duplicate_pairs):
        known = set(df["sample_id"])
        for a, b in sorted(zip(near_duplicate_pairs["sample_id_a"], near_duplicate_pairs["sample_id_b"])):
            if a in known and b in known:
                link(a, b, "near_duplicate_image_dhash")

    grouped = df[df["duplicate_group"] != ""]
    for gid, part in grouped.groupby("duplicate_group", sort=True):
        ids = sorted(part["sample_id"])
        words = len(str(part["text_raw"].iloc[0]).split())
        if words >= min_caption_words:
            for other in ids[1:]:
                link(ids[0], other, "identical_caption_task4_group")
        else:
            for other in ids[1:]:
                links.append({"sample_id_a": ids[0], "sample_id_b": other,
                              "reason": f"NOT_GROUPED_trivial_caption_{words}_words"})

    roots = {sid: uf.find(sid) for sid in df["sample_id"]}
    # readable, deterministic group ids: hash of the root sample_id
    group_ids = pd.Series({sid: "grp_" + hashlib.sha256(r.encode()).hexdigest()[:10] for sid, r in roots.items()})
    return group_ids.reindex(df["sample_id"]).reset_index(drop=True), pd.DataFrame(
        links, columns=["sample_id_a", "sample_id_b", "reason"])


# ------------------------------------------------------------------
# Splitting
# ------------------------------------------------------------------

def folds_per_split(ratios: dict, n_folds: int) -> dict:
    counts = {s: ratios[s] * n_folds for s in SPLITS}
    if any(abs(c - round(c)) > 1e-9 for c in counts.values()) or round(sum(counts.values())) != n_folds:
        raise ValueError(f"ratios {ratios} cannot be expressed in {n_folds} equal folds")
    return {s: int(round(c)) for s, c in counts.items()}


def assign_splits(labels: pd.Series, groups: pd.Series, ratios: dict, seed: int,
                  n_folds: int = 20) -> pd.Series:
    """Group-aware stratified assignment. Deterministic for the same inputs and seed."""
    per_split = folds_per_split(ratios, n_folds)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=labels.index)
    X = pd.DataFrame(index=labels.index)   # features are irrelevant for splitting
    for k, (_, test_idx) in enumerate(sgkf.split(X, labels.to_numpy(), groups.to_numpy())):
        fold.iloc[test_idx] = k
    # fold k -> split: first test folds, then validation, rest train
    order = ["test"] * per_split["test"] + ["validation"] * per_split["validation"] + ["train"] * per_split["train"]
    return fold.map(dict(enumerate(order)))


# ------------------------------------------------------------------
# Distribution and leakage audit
# ------------------------------------------------------------------

def distribution(df: pd.DataFrame, split: pd.Series, names: dict[int, str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for s in ("all",) + SPLITS:
        part = df if s == "all" else df[split == s]
        row = {"split": s, "samples": len(part), "percent_of_total": round(100 * len(part) / total, 2)}
        for label, name in names.items():
            n = int((part["target_label"].astype(int) == label).sum())
            row[f"count_{name}"] = n
            row[f"percent_{name}"] = round(100 * n / len(part), 2) if len(part) else 0.0
        row["samples_with_text"] = int((part["text_raw"].str.strip() != "").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _crossing(values: pd.Series, split: pd.Series, ignore_empty: bool = False) -> int:
    frame = pd.DataFrame({"v": values.values, "s": split.values})
    if ignore_empty:
        frame = frame[frame["v"].astype(str).str.strip() != ""]
    return int((frame.groupby("v")["s"].nunique() > 1).sum())


def leakage_audit(df: pd.DataFrame, split: pd.Series, groups: pd.Series,
                  near_duplicate_pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Explicit checks. 'blocking' checks must pass; 'info' rows are reported only."""
    ids = df["sample_id"]
    sid_split = dict(zip(ids, split))
    checks = []

    def add(name, passed, detail, kind="blocking"):
        checks.append({"check": name, "type": kind, "passed": bool(passed), "detail": detail})

    add("valid_split_names", split.isin(SPLITS).all(), f"values: {sorted(map(str, split.unique()))}")
    add("sample_id_unique", ids.is_unique, f"{int(ids.duplicated().sum())} duplicated ids")
    add("every_sample_assigned", split.notna().all() and len(split) == len(df), f"{len(split)} of {len(df)}")
    sets = {s: set(ids[split == s]) for s in SPLITS}
    overlap = sum(len(sets[a] & sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
    add("splits_mutually_exclusive", overlap == 0, f"{overlap} ids in more than one split")
    union = set().union(*sets.values())
    add("union_equals_dataset", union == set(ids), f"union {len(union)} vs dataset {len(set(ids))}")
    add("no_sample_lost_or_duplicated", sum(len(v) for v in sets.values()) == len(df),
        f"{sum(len(v) for v in sets.values())} assigned for {len(df)} samples")
    add("leakage_groups_not_crossing", _crossing(groups, split) == 0, f"{_crossing(groups, split)} groups cross splits")
    add("identical_images_not_crossing", _crossing(df["image_sha256"], split) == 0,
        f"{_crossing(df['image_sha256'], split)} image hashes in >1 split")
    if near_duplicate_pairs is not None and len(near_duplicate_pairs):
        crossing = sum(sid_split.get(a) != sid_split.get(b) for a, b in
                       zip(near_duplicate_pairs["sample_id_a"], near_duplicate_pairs["sample_id_b"]))
        add("near_duplicate_pairs_not_crossing", crossing == 0, f"{crossing} of {len(near_duplicate_pairs)} pairs cross")
    record = df["image_sha256"] + "|" + df["text_raw"]
    add("identical_records_not_crossing", _crossing(record, split) == 0, f"{_crossing(record, split)} records cross")
    grouped = df["duplicate_group"] != ""
    add("task4_duplicate_groups_crossing", True,
        f"{_crossing(df.loc[grouped, 'duplicate_group'], split[grouped])} Task 4 groups cross "
        "(only trivial <3-word captions are allowed to)", kind="info")
    add("identical_raw_text_crossing", True,
        f"{_crossing(df['text_raw'], split, ignore_empty=True)} distinct non-empty raw texts occur in >1 split",
        kind="info")
    add("identical_normalized_text_crossing", True,
        f"{_crossing(df['text_hinglish_normalized'], split, ignore_empty=True)} distinct non-empty normalized "
        "texts occur in >1 split (not used for grouping by design)", kind="info")
    return pd.DataFrame(checks)


# ------------------------------------------------------------------
# Canonical split access for later tasks
# ------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_split_assignments(config: dict | None = None, verify: bool = True) -> pd.DataFrame:
    """Return the canonical sample_id -> split table, checked against the manifest."""
    sp = (config or load_config())["splitting"]
    path = REPO_ROOT / sp["assignments_file"]
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if verify:
        manifest = json.loads((REPO_ROOT / sp["manifest_file"]).read_text(encoding="utf-8"))
        if sha256_file(path) != manifest["assignments_sha256"]:
            raise RuntimeError("split_assignments.csv does not match split_manifest.json; "
                               "the canonical split was changed. Regenerate deliberately or restore it.")
    return table


def compare_with_canonical(new_assignments: pd.DataFrame, path: Path) -> dict:
    """Compare a freshly generated assignment with the existing canonical file.

    Different library versions (e.g. scikit-learn 1.9 vs 1.7) can shuffle
    differently, so regeneration elsewhere may produce a DIFFERENT split.
    The canonical file must then not be overwritten silently.
    """
    path = Path(path)
    if not path.exists():
        return {"exists": False, "identical": False, "samples_moved": None}
    old = pd.read_csv(path, dtype=str, keep_default_na=False)
    merged = old[["sample_id", "split"]].merge(new_assignments[["sample_id", "split"]], on="sample_id",
                                               how="outer", suffixes=("_old", "_new"))
    moved = int((merged["split_old"] != merged["split_new"]).sum())
    return {"exists": True, "identical": moved == 0, "samples_moved": moved}


def load_split(name: str, allow_test: bool = False, config: dict | None = None) -> pd.DataFrame:
    """Load one split's metadata (data/splits/<name>.csv).

    The test split is LOCKED: pass allow_test=True only for the final,
    pre-registered evaluation of a frozen model configuration.
    """
    if name not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if name == "test" and not allow_test:
        raise PermissionError("The test split is locked for final evaluation only. "
                              "Use 'validation' during development; pass allow_test=True "
                              "only for the final evaluation (see docs/DATA_SPLIT_REPORT.md).")
    sp = (config or load_config())["splitting"]
    assignments = load_split_assignments(config)
    data = pd.read_csv(REPO_ROOT / sp["output_dir"] / f"{name}.csv", dtype=str, keep_default_na=False)
    expected = set(assignments.loc[assignments["split"] == name, "sample_id"])
    if set(data["sample_id"]) != expected:
        raise RuntimeError(f"{name}.csv does not match the canonical split assignment")
    return data
