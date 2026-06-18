"""
models/transformer.py
---------------------
Temporal Transformer Encoder for gait occlusion detection.

Design decisions:
-----------------
1. Input projection: CNN features (dim D_cnn) are first projected to a
   smaller hidden_dim (e.g. 256) before the Transformer to reduce
   parameter count and avoid attention collapse on very high-dim inputs.

2. Positional encoding: a learnable positional embedding is added to the
   sequence dimension so the Transformer knows *which frame* each token
   corresponds to.  Sinusoidal encoding is also supported via a flag.

3. CLS token: optionally prepend a [CLS] token whose output is used for
   sequence-level tasks.  Here we return per-frame outputs, so no CLS is
   needed — both heads operate frame-by-frame over the temporal axis.

4. Standard nn.TransformerEncoder is used (not Flash Attention) for
   maximum compatibility; upgrading to xFormers / Flash Attention is
   trivial since the interface is the same.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable per-position embedding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout  = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        T = x.size(1)
        positions = torch.arange(T, device=x.device)          # (T,)
        pos_enc   = self.embedding(positions).unsqueeze(0)     # (1, T, D)
        return self.dropout(x + pos_enc)


class TemporalTransformer(nn.Module):
    """
    Projects CNN features to hidden_dim, adds positional encoding, then
    passes through a standard TransformerEncoder.

    Args:
        input_dim:       Dimension of CNN feature vectors (e.g. 512).
        hidden_dim:      Transformer model dimension d_model.
        num_layers:      Number of TransformerEncoderLayer stacks.
        num_heads:       Number of attention heads (must divide hidden_dim).
        dropout:         Dropout probability in attention and FFN.
        dim_feedforward: FFN hidden dimension inside each layer.
        max_seq_len:     Maximum sequence length (for positional encoding).
        pos_encoding:    'learnable' | 'sinusoidal'

    Forward:
        Input:  (B, T, input_dim)
        Output: (B, T, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int  = 8,
        dropout: float  = 0.1,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        pos_encoding: str = "learnable",
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if pos_encoding == "sinusoidal":
            self.pos_enc: nn.Module = SinusoidalPositionalEncoding(
                hidden_dim, max_seq_len, dropout
            )
        else:
            self.pos_enc = LearnablePositionalEncoding(
                hidden_dim, max_seq_len, dropout
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # (B, T, D) convention
            norm_first=True,     # Pre-LN for training stability (Wang et al.)
        )
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)   — batch of frame feature sequences

        Returns:
            out: (B, T, hidden_dim) — contextualised per-frame features
        """
        x = self.input_proj(x)      # (B, T, hidden_dim)
        x = self.pos_enc(x)         # add positional information
        x = self.transformer(x)     # self-attention over T frames
        return x
