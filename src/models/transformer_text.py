"""Pretrained transformer text models (Task 9): frozen embeddings and fine-tuning.

Two levels, one encoder (MuRIL, see config.yaml text_model.transformer):

LEVEL A - frozen encoder:
    raw text -> preprocess(mode) -> tokenizer -> encoder (no gradient)
             -> attention-mask MEAN of the last hidden layer -> embedding
             -> StandardScaler + LogisticRegression (fitted on TRAIN only)

LEVEL B - fine-tuning:
    raw text -> preprocess(mode) -> tokenizer -> encoder + 2-class head
    Cross-entropy loss (binary task as 2 mutually exclusive classes, softmax).
    Best epoch chosen on VALIDATION macro-F1; TEST is never used.

Scores: softmax / LR probabilities are UNCALIBRATED model scores, not
calibrated confidences. Text is always preprocessed with the Task 6
preprocess() function (same code as the app will use).
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.preprocessing import preprocess
from src.preprocessing.text import PreprocessingConfig

LABEL_NAMES = {0: "non_misogynistic", 1: "misogynistic"}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_all(texts, mode: str, options: dict | None = None) -> list[str]:
    cfg = PreprocessingConfig(**(options or {}))
    return [preprocess(t, mode, cfg) for t in texts]


# ------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------

def token_lengths(tokenizer, texts) -> np.ndarray:
    """Number of tokens per text INCLUDING special tokens, without truncation."""
    enc = tokenizer(list(texts), add_special_tokens=True, truncation=False)
    return np.array([len(ids) for ids in enc["input_ids"]])


def truncation_rate(lengths: np.ndarray, max_length: int) -> float:
    return float((lengths > max_length).mean())


def _batches(n: int, size: int):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


# ------------------------------------------------------------------
# Level A: frozen embeddings
# ------------------------------------------------------------------

@torch.no_grad()
def mean_pooled_embeddings(texts: list[str], tokenizer, encoder, max_length: int,
                           batch_size: int = 32) -> np.ndarray:
    """Attention-mask mean of the last hidden layer (encoder weights unchanged)."""
    encoder.eval()
    out = []
    for sl in _batches(len(texts), batch_size):
        enc = tokenizer(texts[sl], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        hidden = encoder(**enc).last_hidden_state                     # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ------------------------------------------------------------------
# Level B: fine-tuning
# ------------------------------------------------------------------

@dataclass
class FineTuneConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    max_epochs: int = 4
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 1
    freeze_word_embeddings: bool = True
    max_length: int = 64
    seed: int = 42
    max_batches_per_epoch: int | None = None   # only for smoke tests


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    learning_rate_end: float
    seconds: float


@dataclass
class FineTuneResult:
    history: list = field(default_factory=list)
    best_epoch: int = 0
    best_val_macro_f1: float = -1.0
    best_state: dict | None = None
    train_seconds: float = 0.0
    trainable_parameters: int = 0
    total_parameters: int = 0


@torch.no_grad()
def predict_logits(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for sl in _batches(len(texts), batch_size):
        enc = tokenizer(texts[sl], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        out.append(model(**enc).logits.cpu().numpy())
    return np.concatenate(out)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fine_tune(model, tokenizer, train_texts: list[str], y_train, val_texts: list[str], y_val,
              cfg: FineTuneConfig, log=print) -> FineTuneResult:
    """Plain PyTorch loop. Fits on TRAIN; selects the best epoch by VALIDATION macro-F1."""
    set_seed(cfg.seed)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    if cfg.freeze_word_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate)
    steps_per_epoch = math.ceil(len(train_texts) / cfg.batch_size)
    if cfg.max_batches_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_batches_per_epoch)
    total = steps_per_epoch * cfg.max_epochs
    warmup = int(cfg.warmup_ratio * total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / max(warmup, 1) if s < warmup else max(0.0, (total - s) / max(total - warmup, 1)))
    loss_fn = torch.nn.CrossEntropyLoss()
    result = FineTuneResult(trainable_parameters=sum(p.numel() for p in params),
                            total_parameters=sum(p.numel() for p in model.parameters()))
    rng = np.random.default_rng(cfg.seed)
    stale, start_all = 0, time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        start, order, losses = time.perf_counter(), rng.permutation(len(train_texts)), []
        for step, sl in enumerate(_batches(len(order), cfg.batch_size)):
            if cfg.max_batches_per_epoch and step >= cfg.max_batches_per_epoch:
                break
            idx = order[sl]
            enc = tokenizer([train_texts[i] for i in idx], padding=True, truncation=True,
                            max_length=cfg.max_length, return_tensors="pt")
            loss = loss_fn(model(**enc).logits, torch.as_tensor(y_train[idx], dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step % 50 == 0:
                log(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {loss.item():.4f}")
        logits = predict_logits(model, tokenizer, val_texts, cfg.max_length)
        val_loss = float(loss_fn(torch.as_tensor(logits), torch.as_tensor(y_val, dtype=torch.long)))
        val_f1 = float(f1_score(y_val, logits.argmax(1), average="macro"))
        rec = EpochRecord(epoch, float(np.mean(losses)), val_loss, val_f1, scheduler.get_last_lr()[0],
                          time.perf_counter() - start)
        result.history.append(rec)
        log(f"epoch {epoch}: train_loss {rec.train_loss:.4f} val_loss {val_loss:.4f} val_macro_f1 {val_f1:.4f} "
            f"({rec.seconds:.0f}s)")
        if val_f1 > result.best_val_macro_f1:
            result.best_val_macro_f1, result.best_epoch, stale = val_f1, epoch, 0
            result.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.early_stopping_patience:     # same definition as Hugging Face / Keras
                log(f"early stopping after epoch {epoch} (no improvement for {stale} epochs)")
                break
    result.train_seconds = time.perf_counter() - start_all
    if result.best_state is not None:
        model.load_state_dict(result.best_state)
    return result


# ------------------------------------------------------------------
# Save / load (everything the app needs)
# ------------------------------------------------------------------

def save_fine_tuned(model, tokenizer, out_dir: Path, metadata: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "project_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def load_fine_tuned(path: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(path)
    metadata = json.loads((path / "project_metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return model, tokenizer, metadata
