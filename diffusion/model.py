"""
Transformer model for Masked Diffusion on Sokoban puzzles.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out

        # Feed-forward with pre-norm
        x = x + self.ff(self.norm2(x))
        return x


class SokobanTransformer(nn.Module):
    """
    Transformer for masked diffusion on Sokoban grids.

    Takes in:
        - x: [B, seq_len] token IDs (some masked)
        - t: [B] timesteps

    Returns:
        - logits: [B, seq_len, vocab_size] predictions for all positions
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (vocab includes mask token)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # 2D position embeddings for grid
        self.row_emb = nn.Embedding(config.grid_height, config.d_model)
        self.col_emb = nn.Embedding(config.grid_width, config.d_model)

        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_layers)
        ])

        # Final norm and output projection
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size - 1)  # Don't predict mask

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        device = x.device

        # Token embeddings
        h = self.token_emb(x)  # [B, L, d_model]

        # Add 2D position embeddings
        rows = torch.arange(self.config.grid_height, device=device)
        cols = torch.arange(self.config.grid_width, device=device)
        row_pos = rows.repeat_interleave(self.config.grid_width)  # [L]
        col_pos = cols.repeat(self.config.grid_height)  # [L]

        h = h + self.row_emb(row_pos) + self.col_emb(col_pos)

        # Add timestep embedding (broadcast to all positions)
        t_emb = self.time_emb(t)  # [B, d_model]
        h = h + t_emb[:, None, :]

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        # Output
        h = self.final_norm(h)
        logits = self.output_proj(h)  # [B, L, vocab_size-1]

        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    config = ModelConfig()
    model = SokobanTransformer(config)
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randint(0, config.vocab_size, (4, config.seq_len))
    t = torch.randint(0, config.num_timesteps, (4,))
    logits = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
