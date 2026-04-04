#!/usr/bin/env python3
"""
domain_adaptation.py — Inference-time and Training-time Domain Adaptation
==========================================================================

Implements two complementary DA strategies on top of the HAR model zoo:

1. Test-Time Batch Normalisation (TTBN)
   ─────────────────────────────────────
   During inference, replace each BN layer's stored running statistics
   (μ_src, σ²_src) with statistics computed from the TARGET batch.
   This corrects the input-feature amplitude shift with ZERO retraining.

   Why it works: BN normalises to N(0,1) per channel. After source training
   the running stats encode source-domain moments. On target data those stats
   are wrong, causing covariate shift at every BN layer. Replacing them with
   target moments makes BN act as if it had been trained on target data.

   Implementation detail:
     • Temporarily switch BN layers to train() mode during inference
       (this makes them recompute batch stats instead of using running stats)
     • Run one forward pass over the target batch with torch.no_grad()
     • After the pass the running stats are updated; switch back to eval()
     • All subsequent target inferences use the adapted stats

2. Domain-Adversarial Neural Network (DANN)
   ─────────────────────────────────────────
   Adds a domain discriminator on top of the feature extractor.
   The gradient reversal layer (GRL) negates gradients during backprop,
   forcing the feature extractor to produce domain-invariant representations.

   Training procedure (each iteration):
     a) Source batch (Xs, ys): CE loss on task labels
     b) Source + Target batch unlabelled: adversarial loss on domain labels
     c) Total loss = λ·task_loss + (1-λ)·domain_loss   (λ annealed)
        with λ = 2/(1 + exp(-10·p)) - 1,  p ∈ [0,1] training progress

   The GRL multiplier α follows the same schedule, ensuring the domain
   discriminator is not overwhelmed at the start of training.

   Unlabelled target data: WISDM-Val (subjects 1-30, 24,599 windows).
   No target labels used during DANN training.

References
----------
   TTBN:  Schneider et al. "Improving robustness against common corruptions
          by covariate shift adaptation." NeurIPS 2020.
   DANN:  Ganin et al. "Domain-adversarial training of neural networks."
          JMLR 2016.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. Test-Time Batch Normalisation (TTBN)
# ─────────────────────────────────────────────────────────────────────────────

def apply_ttbn(
    model:      nn.Module,
    target_X:   np.ndarray,
    device:     torch.device,
    batch_size: int = 512,
    n_passes:   int = 1,
) -> nn.Module:
    """
    Adapt BatchNorm running statistics to the target distribution.

    The model is modified IN-PLACE.  Call this AFTER loading a trained
    checkpoint and BEFORE evaluating on target data.

    Parameters
    ----------
    model       : trained model with BN layers
    target_X    : (N, T, C) target domain windows (numpy)
    device      : compute device
    batch_size  : mini-batch size for the adaptation pass
    n_passes    : number of passes over target data (>1 gives smoother stats)

    Returns
    -------
    model  (same object, adapted in-place)
    """
    model = model.to(device)

    # Switch BN layers to train mode (they recompute batch stats)
    # Keep all other layers in eval mode (dropout=0, etc.)
    _set_bn_train(model)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(target_X).float()),
        batch_size=batch_size, shuffle=True,
    )

    with torch.no_grad():
        for _ in range(n_passes):
            for (X_b,) in loader:
                model(X_b.to(device))   # forward pass updates running stats

    model.eval()   # freeze everything (BN now uses adapted stats)
    return model


def _set_bn_train(model: nn.Module) -> None:
    """Set only BatchNorm layers to train(), everything else to eval()."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()


def clone_and_ttbn(
    model:     nn.Module,
    target_X:  np.ndarray,
    device:    torch.device,
    **kwargs,
) -> nn.Module:
    """Return a TTBN-adapted clone (original model is untouched)."""
    import copy
    adapted = copy.deepcopy(model)
    return apply_ttbn(adapted, target_X, device, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DANN components
# ─────────────────────────────────────────────────────────────────────────────

class _GradientReversalFn(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin et al., 2016).

    Forward pass  : identity
    Backward pass : negate gradients and scale by α
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (alpha,) = ctx.saved_tensors
        return -alpha * grad_output, None


class GradientReversal(nn.Module):
    """Wrapper module for the GRL (alpha is a learnable schedule parameter)."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    """
    Two-layer MLP that classifies source (0) vs target (1) domains.
    Receives feature vectors from the backbone's penultimate layer.
    """
    def __init__(self, feat_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),   # binary: source vs target
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


class DANNWrapper(nn.Module):
    """
    Wraps any model that implements `forward_features()` and `head` with
    a domain discriminator and gradient reversal for DANN training.

    Usage
    -----
        dann = DANNWrapper(model)
        feat = dann.features(x)
        task_logits   = dann.classify(feat)
        domain_logits = dann.discriminate(feat)  # uses GRL internally
    """
    def __init__(self, base_model: nn.Module, disc_hidden: int = 256):
        super().__init__()
        self.backbone    = base_model
        self.head        = base_model.head   # shared reference
        feat_dim         = base_model.feat_dim
        self.grl         = GradientReversal(alpha=1.0)
        self.discriminator = DomainDiscriminator(feat_dim, hidden=disc_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard inference (no GRL, no domain head)."""
        return self.backbone(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def classify(self, feat: torch.Tensor) -> torch.Tensor:
        return self.backbone.head(feat)

    def discriminate(self, feat: torch.Tensor) -> torch.Tensor:
        """Domain prediction with gradient reversal."""
        return self.discriminator(self.grl(feat))

    def set_alpha(self, alpha: float) -> None:
        self.grl.alpha = alpha


# ─────────────────────────────────────────────────────────────────────────────
# 3. DANN training loop
# ─────────────────────────────────────────────────────────────────────────────

def _dann_alpha_schedule(p: float) -> float:
    """
    Annealed GRL multiplier.  p ∈ [0,1] = fraction of training complete.
    α → 0 at start (domain discriminator builds up),  α → 1 at end.
    """
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def train_dann(
    base_model:    nn.Module,
    splits:        dict,
    device:        torch.device,
    epochs:        int       = 80,
    patience:      int       = 15,
    lr:            float     = 1e-3,
    weight_decay:  float     = 1e-4,
    batch_size:    int       = 128,
    label_smoothing: float   = 0.1,
    dann_weight:   float     = 0.5,   # weight on domain loss
    seed:          int       = 42,
    verbose:       bool      = True,
) -> nn.Module:
    """
    Full DANN training procedure.

    Parameters
    ----------
    base_model    : an architecture from architectures.py (with forward_features)
    splits        : dict with keys aug_train, aug_val, wisdm_val, wisdm_test
    device        : compute device
    epochs        : max training epochs
    patience      : early stopping based on WISDM-val F1
    lr            : learning rate
    weight_decay  : L2 regularisation
    batch_size    : mini-batch size
    label_smoothing : label smoothing ε for task CE loss
    dann_weight   : λ ∈ [0,1] for domain loss (0 = pure supervised)
    seed          : random seed

    Returns
    -------
    Trained base_model (best weights by WISDM-val F1 restored)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dann = DANNWrapper(base_model).to(device)

    optimizer = torch.optim.Adam(
        dann.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    task_crit   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    domain_crit = nn.BCEWithLogitsLoss()

    # Source loader (labelled aug-UCI)
    Xs, ys = splits["aug_train"]
    src_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xs).float(), torch.from_numpy(ys).long()),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    # Target loader (unlabelled WISDM-val — domain adaptation signal only)
    Xt = splits["wisdm_val"][0]
    tgt_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xt).float()),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )

    best_val_f1  = -1.0
    best_state   = None
    patience_cnt = 0
    total_steps  = epochs * len(src_loader)
    step         = 0

    for epoch in range(1, epochs + 1):
        dann.train()
        tgt_iter = _inf_iter(tgt_loader)

        for X_src, y_src in src_loader:
            X_src, y_src = X_src.to(device), y_src.to(device)
            (X_tgt,)     = next(tgt_iter)
            X_tgt        = X_tgt.to(device)

            # Anneal GRL alpha
            p     = step / total_steps
            alpha = _dann_alpha_schedule(p)
            dann.set_alpha(alpha)
            step += 1

            # ── Forward ──────────────────────────────────────────────────
            feat_src = dann.features(X_src)
            feat_tgt = dann.features(X_tgt)

            task_logits  = dann.classify(feat_src)
            dom_src      = dann.discriminate(feat_src)  # GRL applied
            dom_tgt      = dann.discriminate(feat_tgt)  # GRL applied

            # ── Losses ───────────────────────────────────────────────────
            task_loss = task_crit(task_logits, y_src)

            # Domain labels: source=0, target=1
            dom_labels_src = torch.zeros(len(X_src), device=device)
            dom_labels_tgt = torch.ones(len(X_tgt),  device=device)
            dom_loss = domain_crit(
                torch.cat([dom_src, dom_tgt]),
                torch.cat([dom_labels_src, dom_labels_tgt]),
            )

            total_loss = (1 - dann_weight) * task_loss + dann_weight * dom_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(dann.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # ── Early stopping on WISDM-val F1 ────────────────────────────
        val_f1 = _quick_eval(dann.backbone, *splits["wisdm_val"], device)
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = {k: v.cpu().clone()
                            for k, v in dann.backbone.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  [DANN ep {epoch:3d}]  α={alpha:.3f}  "
                  f"task={task_loss.item():.4f}  dom={dom_loss.item():.4f}  "
                  f"wisdm_val_f1={val_f1:.4f}  best={best_val_f1:.4f}")

        if patience_cnt >= patience:
            if verbose:
                print(f"  Early stop at ep {epoch}  (best_val_f1={best_val_f1:.4f})")
            break

    # Restore best weights into base_model
    base_model.load_state_dict(best_state)
    return base_model


def _inf_iter(loader: DataLoader) -> Iterator:
    """Infinite iterator over a DataLoader."""
    while True:
        yield from loader


# ─────────────────────────────────────────────────────────────────────────────
# 4. Shared evaluation utility
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _quick_eval(
    model:      nn.Module,
    X:          np.ndarray,
    y:          np.ndarray,
    device:     torch.device,
    batch_size: int = 512,
) -> float:
    """Return macro F1 on (X, y)."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float()),
        batch_size=batch_size, shuffle=False,
    )
    preds = []
    for (X_b,) in loader:
        preds.append(model(X_b.to(device)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    return float(f1_score(y, preds, average="macro", zero_division=0))


@torch.no_grad()
def evaluate_full(
    model:      nn.Module,
    X:          np.ndarray,
    y:          np.ndarray,
    device:     torch.device,
    batch_size: int = 512,
) -> dict:
    """Return full metrics dict (acc, macro_f1, per_class_f1)."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float()),
        batch_size=batch_size, shuffle=False,
    )
    preds = []
    for (X_b,) in loader:
        preds.append(model(X_b.to(device)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    return {
        "acc":          float(accuracy_score(y, preds)),
        "macro_f1":     float(f1_score(y, preds, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y, preds, average=None, zero_division=0).tolist(),
    }
