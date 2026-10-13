"""Tests for text preprocessing (src/preprocessing). Synthetic examples only."""

import copy
import math
import unicodedata

import pandas as pd
import pytest

from src.preprocessing import MODES, PreprocessingConfig, preprocess, preprocess_with_details
from src.preprocessing import dataset as prep
from src.preprocessing import hinglish
from src.utils.config import REPO_ROOT, load_config

B, H = "basic", "hinglish_normalized"


# ---------------- RAW ----------------

@pytest.mark.parametrize("text", ["  Ye  NHI hai!!! @user #Tag  ", "नमस्ते", "", "kr rha hu"])
def test_raw_returns_text_unchanged(text):
    assert preprocess(text, "raw") == text


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        preprocess("x", mode="aggressive")


# ---------------- null / empty / types ----------------

@pytest.mark.parametrize("value", [None, float("nan")])
@pytest.mark.parametrize("mode", MODES)
def test_null_input_returns_empty_string(value, mode):
    assert preprocess(value, mode) == ""


@pytest.mark.parametrize("mode", MODES)
def test_empty_and_whitespace_input(mode):
    assert preprocess("", mode) == ""
    assert preprocess("   ", mode) == ("   " if mode == "raw" else "")


def test_non_string_converted():
    assert preprocess(123, B) == "123"


# ---------------- whitespace / unicode / html ----------------

def test_whitespace_collapsed_and_trimmed():
    assert preprocess("  a \t b\n\nc  ", B) == "a b c"


def test_unicode_nfc_composes_devanagari():
    decomposed = "क़"            # क + nukta
    assert preprocess(decomposed, B) == unicodedata.normalize("NFC", decomposed)
    assert preprocess("मैं नहीं जाऊँगा", B) == "मैं नहीं जाऊँगा"   # Devanagari preserved


def test_zero_width_and_nbsp_cleaned_but_zwj_kept():
    assert preprocess("a​b c", B) == "ab c"
    assert "‍" in preprocess("क्‍ष", B)


def test_html_entities_unescaped():
    assert preprocess("Tom &amp; Jerry &lt;3", B) == "Tom & Jerry <3"


# ---------------- URLs / mentions / hashtags ----------------

def test_urls_replaced_with_placeholder():
    assert preprocess("see https://example.com/x?a=1 and www.test.org", B) == "see <URL> and <URL>"


def test_mentions_replaced_and_email_untouched():
    assert preprocess("@user1 ye dekho, mail a@b.com", B) == "<USER> ye dekho, mail a@b.com"


def test_url_and_mention_handling_configurable():
    keep = PreprocessingConfig(url_handling="keep", mention_handling="keep")
    assert preprocess("@u https://x.io", B, keep) == "@u https://x.io"
    remove = PreprocessingConfig(url_handling="remove", mention_handling="remove")
    assert preprocess("hi @u https://x.io bye", B, remove) == "hi bye"


def test_hashtag_symbol_stripped_word_kept():
    assert preprocess("#WomenCanDrive yes", B) == "WomenCanDrive yes"
    assert preprocess("#WomenCanDrive", B, PreprocessingConfig(hashtag_handling="keep")) == "#WomenCanDrive"


# ---------------- emojis / punctuation / numbers / case / stopwords ----------------

@pytest.mark.parametrize("mode", [B, H])
def test_emojis_kept(mode):
    assert preprocess("so funny 😂🔥", mode) == "so funny 😂🔥"


@pytest.mark.parametrize("mode", [B, H])
def test_punctuation_and_repeated_punctuation_kept(mode):
    assert preprocess("Kya??? Haan!!! Theek...", mode).endswith("Haan!!! Theek...")


@pytest.mark.parametrize("mode", [B, H])
def test_numbers_kept(mode):
    assert preprocess("100% sach, 2021 me, ३ baar", mode) == "100% sach, 2021 me, ३ baar"


def test_case_kept_by_default_and_lowercase_configurable():
    assert preprocess("LOOK Here", B) == "LOOK Here"
    assert preprocess("LOOK Here @User", B, PreprocessingConfig(lowercase=True)) == "look here <USER>"


@pytest.mark.parametrize("mode", [B, H])
def test_stopwords_not_removed(mode):
    assert preprocess("this is not the end", mode) == "this is not the end"


def test_stopword_removal_disabled_by_design():
    with pytest.raises(ValueError, match="disabled by design"):
        PreprocessingConfig(remove_stopwords=True)


def test_invalid_option_rejected():
    with pytest.raises(ValueError):
        PreprocessingConfig(emoji_handling="remove")


# ---------------- Hinglish normalization ----------------

@pytest.mark.parametrize("src, expected", [
    ("main kr rha hu", "main kar raha hoon"),
    ("pta nhi kyu", "pata nahi kyun"),
    ("bohot acha laga", "bahut accha laga"),
    ("vo larki gyi", "vo ladki gayi"),
])
def test_romanized_hindi_rules(src, expected):
    assert preprocess(src, H) == expected


def test_rules_are_not_applied_in_basic_mode():
    assert preprocess("main kr rha hu", B) == "main kr rha hu"


def test_rules_preserve_case():
    assert preprocess("NHI Nhi nhi", H) == "NAHI Nahi nahi"


def test_rules_match_whole_words_only():
    assert preprocess("kro krona krow hu human", H) == "karo krona krow hoon human"


def test_ambiguous_words_not_normalized():
    # excluded on purpose: FIR (police report), gai (cow), h (letter), me/main (I vs in)
    assert preprocess("FIR filed, gai, h, me main", H) == "FIR filed, gai, h, me main"


def test_devanagari_untouched_by_rules():
    assert preprocess("नहीं पता", H) == "नहीं पता"


# ---------------- repeated characters ----------------

def test_elongation_collapsed_to_two_in_hinglish_mode_only():
    assert preprocess("bhaiiii nooooo stupidddd", H) == "bhaii noo stupidd"
    assert preprocess("bhaiiii", B) == "bhaiiii"


def test_legit_double_letters_unchanged():
    assert preprocess("good accha cool yaar", H) == "good accha cool yaar"


def test_elongation_then_rule():
    assert preprocess("yaaar", H) == "yaar"


def test_repeated_char_normalization_configurable():
    assert preprocess("bhaiiii", H, PreprocessingConfig(repeated_char_normalization=False)) == "bhaiiii"


# ---------------- negation protection ----------------

@pytest.mark.parametrize("text", ["ye nahi hai", "ye nhi hai", "not good", "never again", "mat karo",
                                  "na na", "don't", "no way", "नहीं", "ye nahin chalega"])
def test_negation_never_removed(text):
    for mode in MODES:
        assert hinglish.count_negations(preprocess(text, mode)) == hinglish.count_negations(text)


def test_negation_variants_map_to_negation():
    rules = hinglish.load_rules()
    for variant in rules.negation_variants:
        assert rules.mapping[variant] in hinglish.NEGATION_WORDS


@pytest.mark.parametrize("bad_rules, message", [
    ([{"canonical": "haan", "variants": ["nhi"]}], "negation"),
    ([{"canonical": "nahi", "variants": ["hai"]}], "negation"),
    ([{"canonical": "", "variants": ["x"]}], "canonical"),
    ([{"canonical": "kar", "variants": ["kr"]}, {"canonical": "karo", "variants": ["kr"]}], "more than one"),
    ([{"canonical": "kar", "variants": ["kr"]}, {"canonical": "kr2", "variants": ["kar"]}], "Invalid canonical"),
    ([{"canonical": "kar", "variants": ["kr"]}, {"canonical": "x", "variants": ["kar"]}], "also be variants"),
    ([{"canonical": "kar", "variants": ["k r"]}], "Invalid variant"),
])
def test_unsafe_rules_rejected(bad_rules, message):
    with pytest.raises(ValueError, match=message):
        hinglish.validate_rules(bad_rules)


def test_shipped_rules_file_is_valid_and_versioned():
    rules = hinglish.load_rules()
    assert rules.version and len(rules.mapping) >= 10
    assert "h" not in rules.mapping and "fir" not in rules.mapping and "me" not in rules.mapping


# ---------------- determinism / no mutation / config ----------------

def test_deterministic_output():
    text = "Ye  NHI hoga yaaar!!! @page #Tag https://a.b kr rha"
    for mode in MODES:
        assert len({preprocess(text, mode) for _ in range(5)}) == 1


def test_details_report_operations():
    _, ops = preprocess_with_details("@u #t kr  rha", H)
    assert ops["mention"] == 1 and ops["hashtag"] == 1 and ops["whitespace"] == 1
    assert ops["rule:kr->kar"] == 1 and ops["rule:rha->raha"] == 1


def test_config_yaml_builds_valid_config():
    config = load_config()
    cfg = PreprocessingConfig.from_config(config, repo_root=REPO_ROOT)
    assert cfg.version == config["text_preprocessing"]["version"]
    assert cfg.remove_stopwords is False and cfg.lowercase is False
    assert hinglish.load_rules(cfg.normalization_rules).mapping


@pytest.fixture
def frame():
    return pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "source_row": ["1", "2", "3", "4"],
        "image_reference": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "image_path": ["p/a.jpg", "p/b.jpg", "p/c.jpg", "p/d.jpg"],
        "original_text": ["kr rha hu", "@ni dekho", "", "kr rha hu nhi pta kyu"],
        "has_text": ["True", "True", "False", "True"],
        "original_label": ["1", "0", "1", "0"],
        "target_label": ["1", "0", "1", "0"],
        "Objectification": ["1", "0", "0", "0"],
        "duplicate_group": ["", "", "", ""],
        "review_flags": ["", "", "", ""],
    })


def test_build_variants_does_not_mutate_input(frame):
    before = copy.deepcopy(frame)
    out, _ = prep.build_variants(frame, PreprocessingConfig(), ["Objectification"])
    pd.testing.assert_frame_equal(frame, before)
    assert out["text_raw"].tolist() == frame["original_text"].tolist()
    assert out["text_hinglish_normalized"].iloc[0] == "kar raha hoon"
    assert (out["sample_id"] == frame["sample_id"]).all() and (out["target_label"] == frame["target_label"]).all()
    assert set(out["preprocessing_version"]) == {"1.0"}


def test_summary_audit_and_examples(frame):
    out, ops = prep.build_variants(frame, PreprocessingConfig(), ["Objectification"])
    summary = prep.summary_by_mode(out).set_index("mode")
    assert summary.loc["raw", "changed_vs_raw"] == 0
    assert summary.loc["hinglish_normalized", "newly_empty_vs_raw"] == 0
    audit = prep.audit(out, ops).set_index("sample_id")
    assert audit.loc["s2", "negation_count_changed"]          # '@ni' masked as a mention
    assert audit.loc["s4", "many_rule_replacements"]
    examples = prep.examples(out, ops, audit.reset_index())
    assert "NONE IN DATASET" in examples.loc[examples.category == "url", "operations"].tolist()
    # s1 "kr rha hu" has 3 rule replacements -> flagged too
    assert audit.loc["s1", "many_rule_replacements"]
    assert set(examples.loc[examples.category == "audit_flagged", "sample_id"]) == {"s1", "s2", "s4"}
    freq = prep.rule_frequencies(ops["hinglish_normalized"], hinglish.load_rules()).set_index("variant")
    assert freq.loc["kr", "occurrences"] == 2 and freq.loc["nhi", "negation"]
