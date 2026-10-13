"""Text preprocessing shared by training AND inference (Streamlit app).

Three modes, compared later in experiments (RQ1):

    raw                  text exactly as stored (experimental control)
    basic                conservative, generic social-media cleaning
    hinglish_normalized  basic + Romanized-Hindi normalization

Usage:
    from src.preprocessing import preprocess
    preprocess("Ye kya hai @page #WomenCanDrive", mode="basic")

The app must call this same function (never re-implement it), so that
a model is always fed text processed exactly like its training data.
Effectiveness of each mode has NOT been established yet.

What each mode does (defaults from config.yaml `text_preprocessing`):

basic
  1. Unicode NFC normalization (composes Devanagari characters; NFKC
     was not needed: it changed nothing more in the corpus)
  2. HTML entities unescaped (&amp; -> &)
  3. zero-width space / BOM removed, non-breaking space -> space
  4. optional lowercasing (OFF by default: casing is kept so the
     model/vectorizer decides; see docs/TEXT_PREPROCESSING_REPORT.md)
  5. URLs -> <URL>, @mentions -> <USER> (configurable)
  6. hashtags: '#' removed, word kept (#WomenCanDrive -> WomenCanDrive)
  7. whitespace collapsed and trimmed
  Kept unchanged: punctuation (incl. !!! ??? ...), numbers, emojis,
  stopwords, negation, Devanagari text.

hinglish_normalized
  basic, then
  8. Latin letter repeated 3+ times -> 2 (bhaiiii -> bhaii)
  9. Romanized-Hindi spelling variants -> canonical form
     (rules file: resources/hinglish_normalization.yaml)
"""

import html
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

from src.preprocessing import hinglish

MODES = ("raw", "basic", "hinglish_normalized")

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
ZERO_WIDTH_RE = re.compile("[​﻿]")   # ZWJ/ZWNJ are kept: they matter in Devanagari
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessingConfig:
    version: str = "1.0"
    lowercase: bool = False
    url_handling: str = "placeholder"        # placeholder | keep | remove
    url_placeholder: str = "<URL>"
    mention_handling: str = "placeholder"    # placeholder | keep | remove
    mention_placeholder: str = "<USER>"
    hashtag_handling: str = "strip_symbol"   # strip_symbol | keep
    emoji_handling: str = "keep"             # keep (only strategy; emojis carry meaning)
    repeated_char_normalization: bool = True  # hinglish_normalized mode only
    remove_stopwords: bool = False           # not implemented by design (see report)
    normalization_rules: str = str(hinglish.RULES_PATH)

    def __post_init__(self):
        choices = {"url_handling": ("placeholder", "keep", "remove"),
                   "mention_handling": ("placeholder", "keep", "remove"),
                   "hashtag_handling": ("strip_symbol", "keep"),
                   "emoji_handling": ("keep",)}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.remove_stopwords:
            raise ValueError("remove_stopwords is disabled by design: stopwords include negations "
                             "and words that change meaning (see docs/TEXT_PREPROCESSING_REPORT.md)")

    @classmethod
    def from_config(cls, config: dict, repo_root: Path | None = None) -> "PreprocessingConfig":
        """Build from config.yaml's `text_preprocessing` section."""
        section = dict(config["text_preprocessing"])
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in section.items() if k in known}
        if "normalization_rules" in kwargs and repo_root is not None:
            kwargs["normalization_rules"] = str(Path(repo_root) / kwargs["normalization_rules"])
        return cls(**kwargs)


DEFAULT_CONFIG = PreprocessingConfig()


def _to_text(text) -> str:
    """None / NaN -> "" ; other values -> str. Never raises on missing data."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return str(text)


def _sub(pattern: re.Pattern, repl, text: str, counts: Counter, name: str) -> str:
    new, n = pattern.subn(repl, text)
    if n:
        counts[name] += n
    return new


def _basic(text: str, cfg: PreprocessingConfig, counts: Counter) -> str:
    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        counts["unicode_nfc"] += 1
    text = nfc
    unescaped = html.unescape(text)
    if unescaped != text:
        counts["html_unescape"] += 1
    text = unescaped
    text = _sub(ZERO_WIDTH_RE, "", text, counts, "zero_width_removed")
    text = text.replace(" ", " ")
    if cfg.lowercase:
        lowered = text.lower()
        if lowered != text:
            counts["lowercase"] += 1
        text = lowered
    if cfg.url_handling != "keep":
        text = _sub(URL_RE, cfg.url_placeholder if cfg.url_handling == "placeholder" else " ",
                    text, counts, "url")
    if cfg.mention_handling != "keep":
        text = _sub(MENTION_RE, cfg.mention_placeholder if cfg.mention_handling == "placeholder" else " ",
                    text, counts, "mention")
    if cfg.hashtag_handling == "strip_symbol":
        text = _sub(HASHTAG_RE, r"\1", text, counts, "hashtag")
    collapsed = WHITESPACE_RE.sub(" ", text).strip()
    if collapsed != text:
        counts["whitespace"] += 1
    return collapsed


def preprocess_with_details(text, mode: str = "raw",
                            config: PreprocessingConfig = DEFAULT_CONFIG) -> tuple[str, Counter]:
    """Like preprocess(), but also returns a Counter of applied operations."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    text = _to_text(text)
    counts = Counter()
    if mode == "raw":
        return text, counts
    text = _basic(text, config, counts)
    if mode == "hinglish_normalized":
        if config.repeated_char_normalization:
            text = hinglish.collapse_elongation(text, counts)
        text = hinglish.apply_rules(text, hinglish.load_rules(config.normalization_rules), counts)
    return text, counts


def preprocess(text, mode: str = "raw", config: PreprocessingConfig = DEFAULT_CONFIG) -> str:
    """Return the preprocessed text for one input. Deterministic; input is not modified."""
    return preprocess_with_details(text, mode, config)[0]
