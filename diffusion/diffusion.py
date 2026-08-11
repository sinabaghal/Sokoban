"""
Masked Diffusion for discrete sequences.

Forward process: progressively mask tokens according to schedule α_t = 1 - t/T
Reverse process: predict and unmask tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from config import ModelConfig


class MaskedDiffusion:
    """
    Masked Diffusion for discrete tokens.

    Forward process: q(x_t | x_0) masks each token independently with prob 1 - α_t
    where α_t = 1 - t/T (linear schedule)

    At t=0: α_0 = 1, no masking (original data)
    At t=T: α_T = 0, fully masked
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps
        self.mask_token_id = config.mask_token_id
        self.vocab_size = config.vocab_size
        self.w_max = config.w_max

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Get α_t = 1 - t/T (probability of keeping token unmasked)."""
        return 1.0 - t.float() / self.num_timesteps

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0).
        Each token is independently masked with probability 1 - α_t.

        Args:
            x_0: [B, L] original tokens
            t: [B] timesteps

        Returns:
            x_t: [B, L] noised tokens (some replaced with mask)
        """
        B, L = x_0.shape
        device = x_0.device

        alpha_t = self.get_alpha(t)  # [B]

        # Sample mask: True means KEEP original, False means MASK
        keep_prob = alpha_t[:, None].expand(B, L)  # [B, L]
        keep_mask = torch.rand(B, L, device=device) < keep_prob

        # Apply masking
        x_t = torch.where(keep_mask, x_0, self.mask_token_id)

        return x_t

    def compute_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute masked diffusion loss.

        Loss = E_t,x_0,mask [ w(t) * CE(model(x_t, t), x_0) ]

        Only compute loss on masked positions.
        Weight w(t) = min(1/(1-α_t), w_max) to handle small t.

        Args:
            model: transformer that predicts logits
            x_0: [B, L] original tokens
            t: [B] timesteps (optional, sampled if None)

        Returns:
            loss: scalar
        """
        B, L = x_0.shape
        device = x_0.device

        # Sample timesteps if not provided
        if t is None:
            # Sample t uniformly from [1, T] (not 0, as α_0=1 means no masking)
            t = torch.randint(1, self.num_timesteps + 1, (B,), device=device)

        # Get noised input
        x_t = self.q_sample(x_0, t)

        # Get model predictions
        logits = model(x_t, t)  # [B, L, vocab_size-1]

        # Mask for loss computation (only masked positions)
        loss_mask = (x_t == self.mask_token_id)  # [B, L]

        if not loss_mask.any():
            # Edge case: no masked tokens (shouldn't happen with t >= 1)
            return torch.tensor(0.0, device=device)

        # Compute cross-entropy loss
        # logits: [B, L, vocab_size-1], targets: [B, L]
        # Flatten for loss computation
        logits_flat = logits.view(-1, self.vocab_size - 1)  # [B*L, vocab_size-1]
        targets_flat = x_0.view(-1)  # [B*L]
        loss_mask_flat = loss_mask.view(-1)  # [B*L]

        # Cross-entropy on all positions
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # [B*L]

        # Apply mask and compute mean
        masked_loss = ce_loss * loss_mask_flat.float()

        # Weight by w(t) = min(1/(1-α_t), w_max)
        alpha_t = self.get_alpha(t)  # [B]
        weights = torch.clamp(1.0 / (1.0 - alpha_t + 1e-8), max=self.w_max)  # [B]

        # Expand weights to match positions
        weights_expanded = weights[:, None].expand(B, L).reshape(-1)  # [B*L]

        weighted_loss = masked_loss * weights_expanded

        # Mean over masked positions
        loss = weighted_loss.sum() / (loss_mask_flat.sum() + 1e-8)

        return loss

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        strategy: str = 'random',
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate samples by iteratively unmasking.

        Args:
            model: trained transformer
            batch_size: number of samples to generate
            device: torch device
            strategy: 'confidence' or 'random' for reveal order
            temperature: softmax temperature for stochastic token sampling.
                Tokens are sampled from the predicted distribution (MaskGIT-style)
                rather than taken via argmax, so identical inputs (e.g. the shared
                fully-masked start state) diverge into distinct samples across a
                batch. Lower values behave more like greedy argmax.

        Returns:
            samples: [B, L] generated token sequences
        """
        L = self.config.seq_len

        # Start fully masked
        x = torch.full((batch_size, L), self.mask_token_id, device=device, dtype=torch.long)

        # Number of tokens to unmask per step
        # We'll unmask over num_timesteps steps
        steps = min(self.num_timesteps, L)

        for step in range(steps):
            # Current "time" (going backwards from T to ~0), scaled proportionally
            # across the full [1, num_timesteps] schedule the model was trained on
            # -- steps < num_timesteps here (100 vs 1000), so a flat -1 decrement
            # per step would only ever cover t in [num_timesteps - steps, num_timesteps].
            t_val = self.num_timesteps - round(step * self.num_timesteps / steps)
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            # Get predictions
            logits = model(x, t)  # [B, L, vocab_size-1]
            probs = F.softmax(logits / temperature, dim=-1)  # [B, L, vocab_size-1]

            # Find masked positions
            masked = (x == self.mask_token_id)  # [B, L]

            if not masked.any():
                break

            # Sample tokens stochastically (MaskGIT-style) instead of greedy
            # argmax, so batch elements with identical context (e.g. the shared
            # all-masked start) diverge. Each sampled token's own probability is
            # used as its confidence score for reveal ordering.
            V = probs.shape[-1]
            pred_tokens = torch.multinomial(probs.view(-1, V), 1).view(batch_size, L)  # [B, L]
            confidence = probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L]

            # Determine how many tokens to unmask this step
            # Linear schedule: unmask (L / steps) tokens per step on average
            num_masked = masked.sum(dim=1)  # [B]
            num_to_unmask = torch.clamp(
                (num_masked.float() / (steps - step)).ceil().long(),
                min=1
            )
            num_to_unmask = torch.minimum(num_to_unmask, num_masked)  # never exceed what's available

            # Rank every position within its row by a priority key (confidence or
            # random), fully vectorized across the batch -- no per-sample Python
            # loop. Non-masked positions get the lowest-possible priority so they
            # always sort after every masked position and are never selected.
            if strategy == 'confidence':
                priority = confidence.clone()
                priority[~masked] = -float('inf')
            else:  # random
                priority = torch.rand(batch_size, L, device=device)
                priority[~masked] = -1.0  # torch.rand is in [0, 1), so -1 always ranks last

            order = priority.argsort(dim=1, descending=True)
            rank = order.argsort(dim=1)  # rank[b, i] = position of column i in row b's sorted order
            select = masked & (rank < num_to_unmask.unsqueeze(1))
            x = torch.where(select, pred_tokens, x)

        # Final pass: unmask any remaining
        masked = (x == self.mask_token_id)
        if masked.any():
            t = torch.ones(batch_size, device=device, dtype=torch.long)
            logits = model(x, t)
            probs = F.softmax(logits / temperature, dim=-1)
            V = probs.shape[-1]
            pred_tokens = torch.multinomial(probs.view(-1, V), 1).view(batch_size, L)
            x = torch.where(masked, pred_tokens, x)

        return x


if __name__ == '__main__':
    from model import SokobanTransformer

    config = ModelConfig()
    model = SokobanTransformer(config)
    diffusion = MaskedDiffusion(config)

    # Test forward process
    x_0 = torch.randint(0, config.vocab_size - 1, (4, config.seq_len))
    t = torch.tensor([100, 500, 800, 999])
    x_t = diffusion.q_sample(x_0, t)

    print(f"Original tokens: {x_0[0, :10]}")
    print(f"Noised at t=100: {x_t[0, :10]}")
    print(f"Mask count at t=100: {(x_t[0] == config.mask_token_id).sum()}")
    print(f"Mask count at t=999: {(x_t[3] == config.mask_token_id).sum()}")

    # Test loss
    loss = diffusion.compute_loss(model, x_0)
    print(f"Loss: {loss.item():.4f}")
