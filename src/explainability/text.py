"""Text explanations (Task 15): which words pushed a frozen text model towards each label.

    from src.explainability.text import explain_text
    explain_text("ye kya bakwas hai", model_version="T05")

Two methods, chosen to match the ACTUAL model type:

- Fine-tuned transformer (T05, kind "fine_tuned"): **Integrated Gradients** (IG) on the input word
  embeddings. Baseline = every non-special token replaced by the [PAD] embedding ([CLS]/[SEP] kept).
  Target = logit difference  logit(misogynistic) - logit(non-misogynistic)  (the log-odds of the
  softmax). IG's completeness property says the attributions sum to f(input) - f(baseline); the
  remaining gap is returned as `completeness_error` so every explanation can be checked.
  WordPiece attributions are summed per word ("##" continuation pieces join the previous piece).
- TF-IDF + logistic regression / linear SVM (kind "tfidf_logreg" / "tfidf_svm"): the decision function
  is linear, so contribution(n-gram) = coefficient x TF-IDF value is an EXACT decomposition
  (sum of contributions + intercept = decision function).

The text goes through the model's own Task 6 preprocessing (the same code the model was trained and
evaluated with); nothing is re-implemented here.

Attributions are MODEL ATTRIBUTION in log-odds units: they show what moved this model's score, not
why a human would find a text harmful, and not that a highlighted word alone "caused" the label.
"""

import numpy as np
import torch

from src.inference import text as text_inference
from src.inference.text import LABEL_NAMES, SCORE_TYPES

METHOD_IG = "integrated_gradients"
METHOD_LINEAR = "coefficient_times_tfidf"
TARGET = "logit difference: log-odds of 'misogynistic' (positive = towards misogynistic)"
LIMITATIONS_IG = [
    "Model attribution, not proof of reasoning, intent or harm.",
    "Integrated Gradients depends on the chosen baseline ([PAD] embeddings); another baseline gives other values.",
    "Attributions of single words ignore how words act together; removing a highlighted word is not "
    "guaranteed to change the label.",
    "Words are the tokenizer's pre-tokenised units (punctuation is separate); [UNK] pieces are flagged.",
    "Words beyond the model's maximum length were never seen and have no attribution.",
    "The score is an uncalibrated softmax probability, not a confidence percentage.",
]
LIMITATIONS_LINEAR = [
    "Model attribution, not proof of reasoning, intent or harm.",
    "Exact for this linear model, but n-gram weights reflect training-data correlations (shortcuts included).",
    "Word highlights split each n-gram's contribution equally over its words (display approximation); "
    "the exact values are per n-gram feature.",
    "Words not in the TF-IDF vocabulary contribute nothing and are marked not_in_vocabulary.",
]


def direction(value: float, eps: float = 1e-9) -> str:
    return "towards misogynistic" if value > eps else "towards non-misogynistic" if value < -eps else "neutral"


# ------------------------------------------------------------------ Integrated Gradients (generic)

def integrated_gradients(forward, embeddings: torch.Tensor, baseline: torch.Tensor, attention_mask: torch.Tensor,
                         steps: int = 32) -> tuple[np.ndarray, float, float]:
    """IG attributions per token position.

    forward(inputs_embeds, attention_mask) -> one scalar per batch row (here: a logit difference).
    embeddings / baseline: (1, T, H). Midpoint Riemann sum with `steps` points, computed in one batch.
    Returns (attributions (T,), f(input), f(baseline)).
    """
    alphas = (torch.arange(steps, dtype=embeddings.dtype) + 0.5) / steps
    path = (baseline + alphas[:, None, None] * (embeddings - baseline)).detach().requires_grad_(True)
    out = forward(path, attention_mask.expand(steps, -1))
    grads, = torch.autograd.grad(out.sum(), path)                 # gradients w.r.t. the inputs only
    attributions = ((embeddings - baseline)[0] * grads.mean(0)).sum(-1)
    with torch.no_grad():
        fx, fb = float(forward(embeddings, attention_mask)[0]), float(forward(baseline, attention_mask)[0])
    return attributions.detach().numpy().astype(float), fx, fb


def encode_with_baseline(tokenizer, embedding_layer, processed: str, max_length: int):
    """Token ids, attention mask, input embeddings, [PAD] baseline, and whether the text was truncated."""
    enc = tokenizer([processed], truncation=True, max_length=max_length, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    full_length = len(tokenizer([processed], truncation=False)["input_ids"][0])
    with torch.no_grad():
        x = embedding_layer(ids)
        pad = embedding_layer(torch.full_like(ids, tokenizer.pad_token_id))
    special = torch.tensor([[t in (tokenizer.cls_token_id, tokenizer.sep_token_id) for t in ids[0].tolist()]])
    baseline = torch.where(special[..., None], x, pad)
    return ids, mask, x, baseline, full_length > ids.shape[1]


def merge_wordpieces(pieces: list[str], attributions: np.ndarray, unk_token: str) -> list[dict]:
    """Sum WordPiece attributions per word, skipping [CLS]/[SEP]; returns words in text order."""
    words = []
    for piece, a in zip(pieces, attributions):
        if piece in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if piece.startswith("##") and words:
            words[-1]["pieces"].append(piece)
            words[-1]["contribution"] += float(a)
            words[-1]["unknown_token"] |= piece == unk_token
        else:
            words.append({"pieces": [piece], "contribution": float(a), "unknown_token": piece == unk_token})
    out = []
    for i, w in enumerate(words):
        token = w["pieces"][0] + "".join(p[2:] for p in w["pieces"][1:])
        out.append({"token": token, "position": i, "contribution": round(w["contribution"], 6),
                    "direction": direction(w["contribution"]), "unknown_token": w["unknown_token"],
                    "wordpieces": len(w["pieces"])})
    return out


def logit_difference_forward(model):
    """forward() for a fine-tuned BertForSequenceClassification: logit(1) - logit(0)."""
    def forward(inputs_embeds, attention_mask):
        logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        return logits[:, 1] - logits[:, 0]
    return forward


def transformer_attribution(model, tokenizer, processed: str, max_length: int, steps: int = 32, forward=None) -> dict:
    """IG word attributions for any forward() built on this encoder's input embeddings (T05 head by default)."""
    embedding_layer = model.base_model.embeddings.word_embeddings
    ids, mask, x, baseline, truncated = encode_with_baseline(tokenizer, embedding_layer, processed, max_length)
    attr, fx, fb = integrated_gradients(forward or logit_difference_forward(model), x, baseline, mask, steps)
    tokens = merge_wordpieces(tokenizer.convert_ids_to_tokens(ids[0].tolist()), attr, tokenizer.unk_token)
    words_in_text = len(processed.split())
    return {"tokens": tokens, "logit_difference": round(fx, 6), "baseline_logit_difference": round(fb, 6),
            "completeness_error": round(float(attr.sum()) - (fx - fb), 6), "truncated": bool(truncated),
            "model_tokens": int(ids.shape[1]), "words_in_processed_text": words_in_text}


# ------------------------------------------------------------------ linear TF-IDF models (exact)

def linear_attribution(pipeline, text: str) -> dict:
    """Exact per-n-gram contributions of a Pipeline(preprocess, tfidf, clf) + display word highlights."""
    pre, vec, clf = pipeline.named_steps["preprocess"], pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    processed = pre.transform([text])[0]
    x = vec.transform([processed])
    names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    contrib = x.data * coef[x.indices]
    intercept = float(clf.intercept_[0])
    decision = float(contrib.sum() + intercept)
    features = sorted(({"feature": str(names[i]), "contribution": round(float(c), 6), "tfidf": round(float(v), 6),
                        "direction": direction(float(c))} for i, c, v in zip(x.indices, contrib, x.data)),
                      key=lambda f: -abs(f["contribution"]))
    words = vec.build_tokenizer()(vec.build_preprocessor()(processed))
    share = np.zeros(len(words))
    for f in features:                                   # split each n-gram equally over every occurrence's words
        gram = f["feature"].split(" ")
        n = len(gram)
        hits = [i for i in range(len(words) - n + 1) if words[i:i + n] == gram]
        for i in hits:
            share[i:i + n] += f["contribution"] / (n * len(hits))
    vocab = vec.vocabulary_
    tokens = [{"token": w, "position": i, "contribution": round(float(share[i]), 6), "direction": direction(share[i]),
               "not_in_vocabulary": w not in vocab} for i, w in enumerate(words)]
    return {"processed": processed, "features": features, "tokens": tokens, "intercept": round(intercept, 6),
            "decision_function": round(decision, 6),
            "decomposition_error": round(decision - float(clf.decision_function(x)[0]), 9)}


# ------------------------------------------------------------------ public interface

def _top(tokens: list[dict], k: int, sign: int) -> list[dict]:
    chosen = [t for t in tokens if sign * t["contribution"] > 0]
    return sorted(chosen, key=lambda t: -abs(t["contribution"]))[:k]


def explain_text(text, model_version: str = "T05", steps: int = 32, top_k: int = 5,
                 registry_path=None) -> dict:
    """Prediction + word attributions for one text, from the frozen model `model_version`."""
    registry_path = registry_path or text_inference.REGISTRY_PATH
    if text is None:
        raise ValueError("text is missing")
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    kind, entry, obj = text_inference._load(model_version, str(registry_path))
    prediction = text_inference.predict_text(text, model_version, registry_path)            # the same inference path as the app
    if kind == "fine_tuned":
        from src.models.transformer_text import preprocess_all

        model, tokenizer, _ = obj
        processed = preprocess_all([text], entry["preprocessing_mode"], entry.get("preprocessing_options"))[0]
        detail = transformer_attribution(model, tokenizer, processed, entry["max_length"], steps)
        method = {"name": METHOD_IG, "steps": steps, "baseline": "[PAD] embedding for every non-special token",
                  "target": TARGET, "aggregation": "sum of WordPiece attributions per word"}
        limitations = LIMITATIONS_IG
    elif kind in ("tfidf_logreg", "tfidf_svm"):
        detail = linear_attribution(obj, text)
        processed = detail.pop("processed")
        detail["logit_difference"] = detail["decision_function"]
        method = {"name": METHOD_LINEAR, "target": ("log-odds of 'misogynistic'" if kind == "tfidf_logreg"
                                                    else "SVM margin (NOT log-odds)"),
                  "aggregation": "exact per n-gram; word highlights split n-gram contributions equally"}
        limitations = LIMITATIONS_LINEAR
    else:
        raise ValueError(f"explanations are implemented for fine-tuned and TF-IDF models, not {kind!r}")
    tokens = detail["tokens"]
    label = prediction["predicted_label"]
    return {
        "model_version": model_version,
        "task": "misogyny_detection",
        "predicted_label": label,
        "predicted_name": LABEL_NAMES[label],
        "score": prediction["score"],
        "score_type": SCORE_TYPES[kind],
        "calibrated": False,
        "method": method,
        "preprocessing_mode": entry["preprocessing_mode"],
        "preprocessing_version": entry["preprocessing_version"],
        "processed_text_empty": not processed.strip(),
        **detail,
        "top_positive": _top(tokens, top_k, +1),
        "top_negative": _top(tokens, top_k, -1),
        "limitations": limitations + (["The processed text is empty: the label comes from the model's behaviour on "
                                       "empty input (a known shortcut), and no word can be highlighted."]
                                      if not processed.strip() else []),
    }
