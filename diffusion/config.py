"""
Configuration for Sokoban Masked Diffusion Model.
"""

import os
from dataclasses import dataclass

# Default data paths (relative to this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BOXOBAN_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'boxoban-levels'))


@dataclass
class ModelConfig:
    # Sokoban grid
    grid_height: int = 10
    grid_width: int = 10
    seq_len: int = 100  # 10 * 10

    # Vocabulary: #, space, @, $, ., *, +, [MASK]
    vocab_size: int = 8
    mask_token_id: int = 7

    # Token mapping
    # 0: '#' (wall)
    # 1: ' ' (floor)
    # 2: '@' (player)
    # 3: '$' (box)
    # 4: '.' (goal)
    # 5: '*' (box on goal)
    # 6: '+' (player on goal)
    # 7: [MASK]

    # Transformer architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1

    # Diffusion
    num_timesteps: int = 1000
    w_max: float = 10.0  # Weight capping for loss


@dataclass
class TrainConfig:
    batch_size: int = 1536
    learning_rate: float = 2.45e-4
    weight_decay: float = 0.01
    num_epochs: int = 1000
    warmup_steps: int = 125
    grad_clip: float = 1.0

    # Data
    data_path: str = os.path.join(_BOXOBAN_DIR, 'medium', 'train')
    val_path: str = os.path.join(_BOXOBAN_DIR, 'medium', 'valid')

    # Stop after this many optimizer steps (0 = run all num_epochs).
    # Needed to compare runs on differently-sized corpora: a quartile has 1/4 the
    # data, so equal epochs would mean 1/4 the steps. Matched STEPS is the fair
    # comparison, and this also sets the cosine schedule horizon.
    max_steps: int = 0

    # Logging
    log_every: int = 100
    save_every: int = 5000
    eval_every: int = 1000
    checkpoint_dir: str = './checkpoints'
    samples_dir: str = './samples'
    samples_per_checkpoint: int = 1000

    # Device
    device: str = 'cuda'
