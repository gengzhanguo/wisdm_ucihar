"""
architectures.py — HAR Model Zoo for Domain Shift Analysis
===========================================================
Five models for input (B, 51, 3), each ~80–105K parameters:

    FFN          — Flat MLP baseline (no temporal structure)
    CNN1D        — Three-block 1-D CNN (local temporal patterns)
    BiGRU        — Two-layer bidirectional GRU (sequential context)
    TCN          — Dilated causal CNN with residuals (multi-scale temporal)
    HARTransformer — Self-attention encoder + sinusoidal PE (global context)

All models output raw logits (B, num_classes); use nn.CrossEntropyLoss.

Design principle
----------------
Similar parameter counts (~80–105 K) for a fair comparison of how each
architectural inductive bias interacts with covariate shift between UCI HAR
and WISDM.

Domain-shift intuition per model
---------------------------------
FFN
    Flattens (B,51,3)→(B,153) — every sample position treated independently.
    Most sensitive to orientation shifts: axis X in position 0…50, Y in 51…101.
    Establishes a lower bound on shift robustness.

CNN1D
    Kernel slides over T, so axis ordering is fixed but position within T has
    local translation equivariance. Sensitive to global amplitude shift; less
    sensitive to within-window timing jitter.

BiGRU
    Processes the sequence recurrently; the final state is a summary of the
    whole window. Better handles variable-speed activities. Sensitive to
    accumulated noise (each step affects hidden state).

TCN
    Dilated causal convolutions look at multiple temporal scales simultaneously.
    Residuals keep gradients stable even with high-frequency noise injection.
    Receptive field spans the entire 2.56-s window with only 4 blocks.

HARTransformer
    Self-attention is permutation-equivariant (absent positional encoding),
    potentially more robust to phase shifts in periodic activities. Pre-LN
    variant is empirically more stable with small datasets.

Factory
-------
    get_model(name, num_classes=5, **kwargs) → nn.Module
    count_params(model) → int
    model_summary() → None

Usage
-----
    from architectures import get_model
    model = get_model("tcn")
    logits = model(x)          # x: (B, 51, 3)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────

T_LEN       = 51           # window length (samples @ 20 Hz)
N_AXES      = 3            # accelerometer channels: X, Y, Z
N_FLAT      = T_LEN * N_AXES   # = 153
NUM_CLASSES = 5            # Walking, Upstairs, Downstairs, Sitting, Standing


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — FFN  (Flat MLP Baseline)
# ─────────────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    """
    Flat Multi-Layer Perceptron baseline.

    Flattens (B, T, C) → (B, 153), then applies three fully-connected layers
    with BatchNorm and Dropout.  No temporal structure whatsoever.

    This model will suffer most from orientation shift (Sec 4.2) because each
    axis occupies fixed positions in the flattened vector.

    Architecture
    ------------
    Flatten → [Linear(153,256) → BN → ReLU → Drop] ×2 → Linear(256,128) →
    BN → ReLU → Linear(128, num_classes)

    ~82 K parameters
    """

    def __init__(
        self,
        num_classes:  int        = NUM_CLASSES,
        hidden_dims:  tuple[int] = (256, 256, 128),
        dropout:      float      = 0.3,
    ):
        super().__init__()
        dims   = [N_FLAT] + list(hidden_dims)
        layers = [nn.Flatten()]
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
            ]
            if i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.feat_dim = dims[-1]
        self.head     = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, C)  →  logits : (B, num_classes)"""
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — CNN1D  (Local Temporal Patterns)
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float = 0.2):
        pad = kernel // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


class CNN1D(nn.Module):
    """
    Three-block 1-D convolutional network.

    Uses progressively larger channels and a mix of kernel sizes to capture
    local temporal patterns at multiple scales.  Global average pooling
    removes the dependence on window length, making the model robust to
    minor timing variations.

    Architecture
    ------------
    [Conv(3→64,7) → BN → ReLU → Drop → MaxPool(2)] ×1
    [Conv(64→128,5) → BN → ReLU → Drop → MaxPool(2)] ×1
    [Conv(128→128,3) → BN → ReLU → Drop] ×1
    → GlobalAvgPool → Linear(128,64) → ReLU → Linear(64, num_classes)

    ~101 K parameters
    """

    def __init__(
        self,
        num_classes: int             = NUM_CLASSES,
        channels:    tuple[int, ...] = (64, 128, 128),
        dropout:     float           = 0.2,
    ):
        super().__init__()
        ch = list(channels)
        blocks = [_ConvBnRelu(N_AXES, ch[0], 7, dropout), nn.MaxPool1d(2)]
        for i in range(1, len(ch)):
            blocks.append(_ConvBnRelu(ch[i-1], ch[i], 5 if i == 1 else 3, dropout))
            if i < len(ch) - 1:
                blocks.append(nn.MaxPool1d(2))
        self.features = nn.Sequential(*blocks)
        self.pool     = nn.AdaptiveAvgPool1d(1)
        head_in       = ch[-1]
        self.feat_dim = head_in
        self.head     = nn.Sequential(
            nn.Linear(head_in, max(head_in // 2, 32)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(head_in // 2, 32), num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.features(x)
        return self.pool(x).squeeze(-1)   # (B, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, C)  →  logits : (B, num_classes)"""
        return self.head(self.forward_features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — BiGRU  (Sequential Context)
# ─────────────────────────────────────────────────────────────────────────────

class BiGRU(nn.Module):
    """
    Two-layer bidirectional GRU.

    Both forward and backward passes produce a hidden state that summarises
    the sequence in each direction.  The final layer's forward and backward
    hidden states are concatenated into a fixed-size context vector.

    Bidirectionality is especially useful for activities whose signatures
    appear at both the start and end of a window (e.g., transitions between
    sitting and standing).

    Architecture
    ------------
    BiGRU(input=3, hidden=64, layers=2) → concat(h_fwd, h_bwd) (128,)
    → Linear(128,64) → ReLU → Dropout → Linear(64, num_classes)

    ~101 K parameters
    """

    def __init__(
        self,
        num_classes: int   = NUM_CLASSES,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size    = N_AXES,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.feat_dim = hidden_size * 2
        self.head     = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, 2H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, C)  →  logits : (B, num_classes)"""
        return self.head(self.forward_features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Model 4 — TCN  (Dilated Causal CNN + Residuals)
# ─────────────────────────────────────────────────────────────────────────────

class _Chomp1d(nn.Module):
    """Remove the right-hand padding added by causal convolution."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class _TCNResBlock(nn.Module):
    """
    One TCN residual block.

        Conv1d(pad=dilation*(k-1)) → Chomp → BN → ReLU → Drop
        Conv1d(pad=dilation*(k-1)) → Chomp → BN → ReLU → Drop
        + residual (1×1 Conv if channels differ, else identity)
        → ReLU
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int,
        dropout:      float = 0.2,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation, bias=False),
            _Chomp1d(pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation, bias=False),
            _Chomp1d(pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


class TCN(nn.Module):
    """
    Temporal Convolutional Network with dilated causal convolutions.

    Receptive field = 1 + (kernel_size − 1) × Σ(dilations)
    Default: RF = 1 + 3 × (1+2+4+8) = 46 samples (2.3 s @ 20 Hz).
    This covers the entire 2.56-s window with margin.

    Why TCN for noisy HAR data:
      - No vanishing gradients (residual shortcuts)
      - Parallelisable across T (unlike RNN)
      - Dilation exposes the model to both fine-grained (cycle-level) and
        coarse (activity-level) temporal patterns simultaneously
      - BatchNorm inside each block partially normalises amplitude shifts
        between source and target domain

    Architecture
    ------------
    4 × _TCNResBlock(dilations=[1,2,4,8], kernel=4, ch=64)
    → AdaptiveAvgPool → Linear(64, num_classes)

    ~88 K parameters
    """

    def __init__(
        self,
        num_classes: int       = NUM_CLASSES,
        n_channels:  int       = 64,
        kernel_size: int       = 4,
        dilations:   list[int] = None,
        dropout:     float     = 0.2,
    ):
        super().__init__()
        dilations = dilations or [1, 2, 4, 8]

        blocks = []
        in_ch  = N_AXES
        for d in dilations:
            blocks.append(_TCNResBlock(in_ch, n_channels, kernel_size, d, dropout))
            in_ch = n_channels

        self.network  = nn.Sequential(*blocks)
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.feat_dim = n_channels
        self.head     = nn.Linear(n_channels, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.network(x)
        return self.pool(x).squeeze(-1)   # (B, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, C)  →  logits : (B, num_classes)"""
        return self.head(self.forward_features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Model 5 — HARTransformer  (Global Self-Attention)
# ─────────────────────────────────────────────────────────────────────────────

class _SinusoidalPE(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    Fixed (non-learnable) and generalises better than learned PE for
    short, fixed-length sequences like HAR windows.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model)"""
        return self.dropout(x + self.pe[:, : x.size(1)])


class HARTransformer(nn.Module):
    """
    Transformer Encoder for HAR.

    Pre-LayerNorm variant (norm_first=True) is empirically more stable
    with small datasets and avoids the well-known degradation of vanilla
    Post-LN transformers in low-data regimes.

    Global mean pooling over the T dimension aggregates all timesteps
    equally — unlike the CLS-token approach, this doesn't require the
    model to learn to route information to a special token.

    Architecture
    ------------
    Linear(3 → d_model)
    + SinusoidalPE
    → 2 × TransformerEncoderLayer(d=64, nhead=4, ff=256, Pre-LN)
    → mean(T) → Linear(64, num_classes)

    ~100 K parameters
    """

    def __init__(
        self,
        num_classes:     int   = NUM_CLASSES,
        d_model:         int   = 64,
        nhead:           int   = 4,
        num_layers:      int   = 2,
        dim_feedforward: int   = 256,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(N_AXES, d_model)
        self.pos_enc    = _SinusoidalPE(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,     # Pre-LN for training stability
        )
        self.encoder  = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.feat_dim = d_model
        self.head     = nn.Linear(d_model, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x.mean(dim=1)      # (B, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, C)  →  logits : (B, num_classes)"""
        return self.head(self.forward_features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Factory & utilities
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[nn.Module]] = {
    "ffn":         FFN,
    "cnn1d":       CNN1D,
    "bigru":       BiGRU,
    "tcn":         TCN,
    "transformer": HARTransformer,
}


def get_model(
    name:        str,
    num_classes: int = NUM_CLASSES,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a registered model by name.

    Parameters
    ----------
    name : str
        One of: "ffn", "cnn1d", "bigru", "tcn", "transformer"
    num_classes : int, default 5
    **kwargs
        Forwarded to the model constructor (e.g. dropout=0.4).

    Returns
    -------
    nn.Module  (weights randomly initialised)
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](num_classes=num_classes, **kwargs)


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the total number of (trainable) parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(num_classes: int = NUM_CLASSES) -> None:
    """
    Print parameter counts and a forward-pass shape check for all models.
    """
    import time
    dummy = torch.zeros(8, T_LEN, N_AXES)

    header = f"{'Model':<14} {'Params':>9}  {'Output shape':>14}  {'ms/batch':>9}"
    print(header)
    print("─" * len(header))

    for name, cls in _REGISTRY.items():
        m = cls(num_classes=num_classes).eval()
        n = count_params(m)
        with torch.no_grad():
            t0  = time.perf_counter()
            out = m(dummy)
            ms  = (time.perf_counter() - t0) * 1000
        print(f"{name:<14} {n:>9,}  {str(tuple(out.shape)):>14}  {ms:>8.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test  (python architectures.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  architectures.py — model summary + forward-pass check")
    print("=" * 60)
    print()
    model_summary()

    print()
    print("get_model() factory test:")
    for name in _REGISTRY:
        m = get_model(name, dropout=0.1)
        x = torch.randn(4, T_LEN, N_AXES)
        out = m(x)
        assert out.shape == (4, NUM_CLASSES), f"{name}: unexpected output shape {out.shape}"
        print(f"  {name:<14} ✓  out={tuple(out.shape)}")

    print()
    print("Gradient flow check:")
    for name in _REGISTRY:
        m = get_model(name)
        x = torch.randn(4, T_LEN, N_AXES)
        loss = get_model(name)(x).sum()
        loss.backward()
        has_grad = all(
            p.grad is not None
            for p in get_model(name).parameters()
            if p.requires_grad
        )
        # Just verify no NaN in first model's grads
        m2 = get_model(name)
        out2 = m2(x)
        out2.sum().backward()
        nan_grads = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in m2.parameters()
        )
        print(f"  {name:<14} {'NaN grads! ✗' if nan_grads else 'no NaN grads ✓'}")

    print()
    print("Done ✓")
