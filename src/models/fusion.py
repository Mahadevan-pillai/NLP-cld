"""Multimodal fusion classifiers (Task 11).

M01 (early fusion baseline):
    [text features ; image features] -> StandardScaler -> LogisticRegression

M02 / M03 (small neural fusion, this module):
    text  (standardised) -> Linear(d_text, p) -> LayerNorm --+
                                                            concat (2p) -> Dropout -> Linear(2p, h) -> GELU
    image (standardised) -> Linear(d_img,  p) -> LayerNorm --+               -> Dropout -> Linear(h, 2)

- Standardisation statistics are computed on TRAIN only and stored with the model.
- The LayerNorm after each projection puts both modalities on a comparable scale,
  so neither dominates just because of its raw magnitude.
- Binary task, two mutually exclusive classes: cross-entropy over 2 logits, softmax score
  (UNCALIBRATED).
- The best epoch is chosen on VALIDATION macro-F1 (early stopping). TEST is never used.

Artifacts are pickle-free: weights + standardiser statistics in one safetensors file,
configuration and metadata in JSON.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import f1_score
from torch import nn


@dataclass
class FusionConfig:
    projection_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 5
    seed: int = 42


class Standardizer:
    """Per-feature mean/std fitted on TRAIN only (zero std -> 1, so constant features stay finite)."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        self.std = np.where(std > 0, std, 1.0).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


class FusionMLP(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, cfg: FusionConfig, n_classes: int = 2):
        super().__init__()
        p, h = cfg.projection_dim, cfg.hidden_dim
        self.text_proj = nn.Sequential(nn.Linear(text_dim, p), nn.LayerNorm(p))
        self.image_proj = nn.Sequential(nn.Linear(image_dim, p), nn.LayerNorm(p))
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(2 * p, h), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(h, n_classes))

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.text_proj(text), self.image_proj(image)], dim=1))


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    seconds: float


@dataclass
class FusionResult:
    history: list = field(default_factory=list)
    best_epoch: int = 0
    best_val_macro_f1: float = -1.0
    train_seconds: float = 0.0
    parameters: int = 0


@torch.no_grad()
def predict_logits(model: FusionMLP, text: np.ndarray, image: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(text[i:i + batch_size]), torch.from_numpy(image[i:i + batch_size])).numpy()
           for i in range(0, len(text), batch_size)]
    return np.concatenate(out)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def train_fusion(text_tr, image_tr, y_tr, text_va, image_va, y_va, cfg: FusionConfig, log=print):
    """Fit standardisers + FusionMLP on TRAIN; pick the epoch with the best VALIDATION macro-F1.

    Inputs are raw (unstandardised) feature arrays aligned by sample_id. Returns
    (model, (text_std, image_std), FusionResult).
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    t_std, i_std = Standardizer().fit(text_tr), Standardizer().fit(image_tr)       # TRAIN statistics only
    xt_tr, xi_tr = t_std.transform(text_tr), i_std.transform(image_tr)
    xt_va, xi_va = t_std.transform(text_va), i_std.transform(image_va)
    y_tr_t = torch.as_tensor(np.asarray(y_tr), dtype=torch.long)
    y_va = np.asarray(y_va)
    model = FusionMLP(text_tr.shape[1], image_tr.shape[1], cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    result = FusionResult(parameters=sum(p.numel() for p in model.parameters()))
    best_state, stale, start_all = None, 0, time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        start, losses = time.perf_counter(), []
        model.train()
        order = rng.permutation(len(xt_tr))
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            loss = loss_fn(model(torch.from_numpy(xt_tr[idx]), torch.from_numpy(xi_tr[idx])), y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logits = predict_logits(model, xt_va, xi_va)
        val_loss = float(loss_fn(torch.from_numpy(logits), torch.as_tensor(y_va, dtype=torch.long)))
        val_f1 = float(f1_score(y_va, logits.argmax(1), average="macro"))
        result.history.append(EpochRecord(epoch, float(np.mean(losses)), val_loss, val_f1, time.perf_counter() - start))
        if val_f1 > result.best_val_macro_f1:
            result.best_val_macro_f1, result.best_epoch, stale = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.early_stopping_patience:
                log(f"  early stopping after epoch {epoch} (best epoch {result.best_epoch})")
                break
    result.train_seconds = time.perf_counter() - start_all
    model.load_state_dict(best_state)
    model.eval()
    return model, (t_std, i_std), result


def predict_proba(model: FusionMLP, standardizers, text: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Uncalibrated softmax scores [P(non-misogynistic), P(misogynistic)] for raw (unstandardised) features."""
    t_std, i_std = standardizers
    return softmax(predict_logits(model, t_std.transform(text), i_std.transform(image)))


# ------------------------------------------------------------------
# Save / load (pickle-free)
# ------------------------------------------------------------------

def save_fusion(model: FusionMLP, standardizers, cfg: FusionConfig, out_dir: Path, metadata: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_std, i_std = standardizers
    tensors = {f"model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    tensors.update({"text_mean": torch.from_numpy(t_std.mean), "text_std": torch.from_numpy(t_std.std),
                    "image_mean": torch.from_numpy(i_std.mean), "image_std": torch.from_numpy(i_std.std)})
    save_file(tensors, str(out_dir / "fusion.safetensors"))
    (out_dir / "fusion_config.json").write_text(json.dumps({"fusion_config": asdict(cfg),
                                                             "text_dim": int(t_std.mean.shape[0]),
                                                             "image_dim": int(i_std.mean.shape[0]),
                                                             **metadata}, indent=2), encoding="utf-8")
    return out_dir


def load_fusion(path: Path):
    """Return (model, standardizers, metadata) from a folder written by save_fusion."""
    path = Path(path)
    meta = json.loads((path / "fusion_config.json").read_text(encoding="utf-8"))
    tensors = load_file(str(path / "fusion.safetensors"))
    cfg = FusionConfig(**meta["fusion_config"])
    model = FusionMLP(meta["text_dim"], meta["image_dim"], cfg)
    model.load_state_dict({k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")})
    model.eval()
    standardizers = (Standardizer(tensors["text_mean"].numpy(), tensors["text_std"].numpy()),
                     Standardizer(tensors["image_mean"].numpy(), tensors["image_std"].numpy()))
    return model, standardizers, meta
