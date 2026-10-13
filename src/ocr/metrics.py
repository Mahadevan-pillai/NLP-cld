"""OCR quality metrics (Task 14): character and word error rates.

CER = edit distance between character sequences / number of reference characters
WER = edit distance between word sequences      / number of reference words
(edit distance = minimum insertions + deletions + substitutions; Levenshtein).

Before comparing, both strings get the SAME light normalisation (compare_form): Unicode NFC,
lower case, whitespace collapsed. When "normalized OCR" is evaluated, the reference is passed
through the SAME Task 6 preprocessing mode first (done by the caller), so normalization is never
penalised for differing from un-normalised reference text.

An EMPTY reference has no defined error rate: the functions return None and callers report such
samples separately (OCR found text where the dataset has none, or both are empty).
"""

import unicodedata


def compare_form(text: str | None) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.lower().split())


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or lists), O(len(a) * len(b)) time, O(len(b)) memory."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def cer(reference: str | None, hypothesis: str | None) -> float | None:
    ref, hyp = compare_form(reference), compare_form(hypothesis)
    if not ref:
        return None
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str | None, hypothesis: str | None) -> float | None:
    ref, hyp = compare_form(reference).split(), compare_form(hypothesis).split()
    if not ref:
        return None
    return edit_distance(ref, hyp) / len(ref)


def corpus_rates(pairs) -> dict:
    """Micro-averaged CER/WER over (reference, hypothesis) pairs with non-empty references."""
    ce = cn = we = wn = 0
    for ref, hyp in pairs:
        r, h = compare_form(ref), compare_form(hyp)
        if not r:
            continue
        ce += edit_distance(r, h)
        cn += len(r)
        we += edit_distance(r.split(), h.split())
        wn += len(r.split())
    return {"cer": ce / cn if cn else None, "wer": we / wn if wn else None, "reference_chars": cn, "reference_words": wn}
