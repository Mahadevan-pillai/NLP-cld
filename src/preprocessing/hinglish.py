"""Rule-based Hinglish (Romanized Hindi) normalization.

Rules live in resources/hinglish_normalization.yaml (inspectable,
versioned). This module loads and validates them and applies them as
whole-word, case-insensitive, case-preserving replacements on
Latin-script words. Devanagari text is never touched.
"""

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

RULES_PATH = Path(__file__).resolve().parent / "resources" / "hinglish_normalization.yaml"

# Negation words (lowercase). Used to protect rules and to audit output.
# Includes ambiguous forms (e.g. "na", "nai") on purpose: if any rule
# ever touched them, the audit would notice.
NEGATION_WORDS = frozenset({
    # English
    "no", "not", "never", "nope", "nothing", "nobody", "none", "neither", "nor",
    "dont", "don't", "doesnt", "doesn't", "didnt", "didn't", "isnt", "isn't",
    "cant", "can't", "cannot", "wont", "won't", "aint", "ain't",
    # Romanized Hindi
    "nahi", "nhi", "nahin", "nhin", "nahii", "nhii", "na", "naa", "ni", "nai", "mat",
    # Devanagari
    "नहीं", "नही", "ना", "न", "मत",
})
NEGATION_TOKEN_RE = re.compile(r"[A-Za-z']+|[ऀ-ॣॱ-ॿ]+")
ELONGATION_RE = re.compile(r"([A-Za-z])\1{2,}")   # same Latin letter 3+ times


def count_negations(text: str) -> int:
    return sum(1 for t in NEGATION_TOKEN_RE.findall(text) if t.lower() in NEGATION_WORDS)


@dataclass(frozen=True)
class RuleSet:
    version: str
    mapping: dict          # variant (lowercase) -> canonical (lowercase)
    negation_variants: frozenset
    pattern: re.Pattern


def validate_rules(rules: list[dict]) -> dict:
    """Return {variant: canonical}; raise ValueError on unsafe rules."""
    mapping = {}
    for rule in rules:
        canonical = str(rule.get("canonical", "")).strip().lower()
        variants = [str(v).strip().lower() for v in rule.get("variants", [])]
        if not canonical or not re.fullmatch(r"[a-z]+", canonical):
            raise ValueError(f"Invalid canonical form: {rule.get('canonical')!r}")
        if not variants:
            raise ValueError(f"Rule for {canonical!r} has no variants")
        for v in variants:
            if not re.fullmatch(r"[a-z]+", v):
                raise ValueError(f"Invalid variant {v!r} (rules must map a word to a word)")
            if v == canonical:
                raise ValueError(f"Variant equals canonical: {v!r}")
            if v in mapping:
                raise ValueError(f"Variant {v!r} appears in more than one rule")
            # Negation protection: never turn a negation into a non-negation
            # or a non-negation into a negation.
            if (v in NEGATION_WORDS) != (canonical in NEGATION_WORDS):
                raise ValueError(f"Rule {v!r} -> {canonical!r} would change negation")
            mapping[v] = canonical
    chained = set(mapping) & set(mapping.values())
    if chained:
        raise ValueError(f"Canonical forms must not also be variants: {sorted(chained)}")
    return mapping


@lru_cache(maxsize=4)
def load_rules(path: str | Path = RULES_PATH) -> RuleSet:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    mapping = validate_rules(data["rules"])
    # Longest variants first so e.g. "bahot" wins over any shorter overlap.
    alternation = "|".join(sorted(map(re.escape, mapping), key=len, reverse=True))
    pattern = re.compile(rf"(?<![A-Za-z'])({alternation})(?![A-Za-z'])", re.IGNORECASE)
    negs = frozenset(v for v in mapping if v in NEGATION_WORDS)
    return RuleSet(str(data["version"]), mapping, negs, pattern)


def _match_case(source: str, target: str) -> str:
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[0].isupper():
        return target[0].upper() + target[1:]
    return target


def apply_rules(text: str, rules: RuleSet, counts: Counter | None = None) -> str:
    """Replace whole-word variants with canonical forms (case-preserving)."""
    def repl(m):
        word = m.group(0)
        canonical = rules.mapping[word.lower()]
        if counts is not None:
            counts[f"rule:{word.lower()}->{canonical}"] += 1
        return _match_case(word, canonical)
    return rules.pattern.sub(repl, text)


def collapse_elongation(text: str, counts: Counter | None = None) -> str:
    """Shorten a Latin letter repeated 3+ times to 2 ("bhaiiii" -> "bhaii").

    Keeping two letters avoids damaging legitimate doubles (good, too,
    accha) while still merging elongated variants of the same word.
    """
    new, n = ELONGATION_RE.subn(r"\1\1", text)
    if counts is not None and n:
        counts["elongation"] += n
    return new
