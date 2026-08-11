"""
Training script for Sokoban Masked Diffusion Model.
"""

import functools
import glob
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from config import ModelConfig, TrainConfig
from model import SokobanTransformer, count_parameters
from diffusion import MaskedDiffusion
from dataset import create_dataloaders, tensor_to_puzzle
from generate import validate_puzzle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
METRICS_DIR = os.path.join(SOURCE_DIR, 'metrics')
sys.path.insert(0, SOURCE_DIR)
sys.path.insert(0, METRICS_DIR)

from test_solver import boxoban_to_matrix, solve_push
from tile_pattern import puzzle_to_grid, extract_patterns
from kl_divergence import js_divergence

TRAIN_DIST_PATH = os.path.join(METRICS_DIR, 'distributions', 'tile_pattern_n3.json')


class Tee:
    """Writes to multiple streams at once, so prints show up on the console and in a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def train(model_config: ModelConfig, train_config: TrainConfig, use_bf16: bool = False):
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    run_id = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(train_config.checkpoint_dir, f'train_{run_id}.txt')
    log_file = open(log_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    print(f"Logging to {log_path}")

    # Device
    device = torch.device(train_config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model
    model = SokobanTransformer(model_config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Create diffusion
    diffusion = MaskedDiffusion(model_config)

    # Create dataloaders
    print(f"\nLoading data from {train_config.data_path}")
    train_loader, val_loader = create_dataloaders(
        train_config.data_path,
        train_config.val_path,
        train_config.batch_size,
        num_workers=4
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    # max_steps, when set, is the authoritative horizon: it caps training AND the
    # cosine decay. Without this a smaller corpus would anneal the LR to eta_min
    # over 1/4 as many steps, so runs on different-sized corpora would not be
    # comparable even at matched step counts.
    total_steps = len(train_loader) * train_config.num_epochs
    if train_config.max_steps:
        total_steps = train_config.max_steps
        # num_epochs would otherwise cap the run before max_steps is reached: a
        # 42k-puzzle corpus is only ~27 steps/epoch, so the default 1000 epochs
        # tops out at ~27k steps. Raise the epoch count so max_steps is genuinely
        # the binding limit.
        needed = math.ceil(train_config.max_steps / len(train_loader))
        if needed > train_config.num_epochs:
            train_config.num_epochs = needed
        print(f"[max_steps] {train_config.max_steps:,} steps over "
              f"{len(train_loader)} steps/epoch -> {needed:,} epochs "
              f"({train_config.max_steps * train_config.batch_size / len(train_loader.dataset):.0f} "
              f"passes over the data)")
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Mixed precision. bf16 has fp32's exponent range so it can't underflow --
    # no loss scaling needed, so the GradScaler is disabled (a no-op passthrough)
    # and the scaler.scale/unscale/step/update calls below stay a single code path.
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=not use_bf16)
    print(f"Mixed precision: {'bf16 (no GradScaler)' if use_bf16 else 'fp16 (GradScaler on)'}")

    # Training-set tile-pattern distribution, loaded once and reused for every
    # checkpoint's JSD comparison rather than re-reading it from disk each time.
    train_dist_counts = load_train_distribution(TRAIN_DIST_PATH)

    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    start_epoch = 0

    # Resume from the latest checkpoint, if one exists
    resume_path = find_latest_checkpoint(train_config.checkpoint_dir)
    if resume_path is not None:
        # Reconstruct the scheduler fresh instead of restoring its state dict --
        # CosineAnnealingLR.get_lr() nudges the optimizer's *current* lr by a
        # ratio rather than recomputing it from last_epoch, so loading old
        # state (or passing last_epoch= to the constructor) just carries
        # forward whatever lr had already decayed to under the old, shorter
        # T_max. The closed-form lr for the new horizon is computed explicitly
        # below instead.
        global_step, _ = load_checkpoint(resume_path, model, optimizer)
        eta_min = 1e-6
        for group in optimizer.param_groups:
            group['initial_lr'] = train_config.learning_rate
            group['lr'] = eta_min + (train_config.learning_rate - eta_min) * (
                1 + math.cos(math.pi * global_step / total_steps)) / 2
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min, last_epoch=global_step)
        start_epoch = global_step // len(train_loader)
        print(f"Resumed from {resume_path} at step {global_step} (epoch {start_epoch + 1})")

        ensure_samples_exist(global_step, model, diffusion, device, train_config, train_dist_counts)

        best_path = os.path.join(train_config.checkpoint_dir, 'best.pt')
        if os.path.exists(best_path):
            best_checkpoint = torch.load(best_path, map_location=device)
            best_val_loss = best_checkpoint['loss']
            print(f"Restored best_val_loss: {best_val_loss:.4f}")

    print(f"\nStarting training for {train_config.num_epochs} epochs")
    print(f"Total steps: {total_steps:,}")
    print("-" * 60)

    for epoch in range(start_epoch, train_config.num_epochs):
        model.train()
        epoch_loss = torch.zeros((), device=device)  # accumulate on-GPU; avoid a sync every step
        epoch_start = time.time()

        for batch_idx, x_0 in enumerate(train_loader):
            x_0 = x_0.to(device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(dtype=amp_dtype):
                loss = diffusion.compute_loss(model, x_0)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.detach()
            global_step += 1

            # Logging
            if global_step % train_config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e}")

            # Evaluation
            if global_step % train_config.eval_every == 0:
                val_loss = evaluate(model, diffusion, val_loader, device)
                print(f"Step {global_step:6d} | Val Loss: {val_loss:.4f}")

                avg_walls = compute_avg_wall_count(model, diffusion, device, num_samples=100)
                print(f"Step {global_step:6d} | Avg wall count (100 samples): {avg_walls:.1f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, val_loss,
                        os.path.join(train_config.checkpoint_dir, 'best.pt')
                    )
                    print(f"  -> New best model saved!")

                model.train()

            # Save checkpoint
            if global_step % train_config.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, loss.item(),
                    os.path.join(train_config.checkpoint_dir, f'step_{global_step}.pt')
                )

                samples_path = os.path.join(
                    train_config.samples_dir, f'ckp_{global_step}', 'samples.txt'
                )
                puzzles = generate_and_save_samples(
                    model, diffusion, device, train_config.samples_per_checkpoint, samples_path
                )
                print(f"  Saved {train_config.samples_per_checkpoint} samples to {samples_path}")

                perf_path = os.path.join(train_config.samples_dir, 'perf.txt')
                log_perf(global_step, puzzles, train_dist_counts, perf_path)

                model.train()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = (epoch_loss / len(train_loader)).item()
        print(f"\nEpoch {epoch+1}/{train_config.num_epochs} | "
              f"Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

        # Generate samples at end of epoch
        print("\nGenerating samples...")
        generate_samples(model, diffusion, device, num_samples=3)
        print("-" * 60)

        if train_config.max_steps and global_step >= train_config.max_steps:
            print(f"\nReached max_steps ({train_config.max_steps}); stopping.")
            save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss,
                            os.path.join(train_config.checkpoint_dir,
                                         f'step_{global_step}.pt'))
            break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    sys.stdout = original_stdout
    log_file.close()


@torch.no_grad()
def evaluate(model, diffusion, val_loader, device, max_batches=50):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x_0 in val_loader:
        x_0 = x_0.to(device)
        loss = diffusion.compute_loss(model, x_0)
        total_loss += loss.item()
        num_batches += 1

        if num_batches >= max_batches:
            break

    return total_loss / num_batches


@torch.no_grad()
def generate_samples(model, diffusion, device, num_samples=3):
    """Generate and display sample puzzles."""
    model.eval()
    samples = diffusion.sample(model, num_samples, device, strategy='random')

    for i, sample in enumerate(samples):
        puzzle = tensor_to_puzzle(sample)
        print(f"\nSample {i+1}:")
        print(puzzle)


@torch.no_grad()
def compute_avg_wall_count(model, diffusion, device, num_samples=100):
    """Generate samples and return their average wall ('#') count."""
    model.eval()
    samples = diffusion.sample(model, num_samples, device, strategy='random')
    wall_counts = [tensor_to_puzzle(sample).count('#') for sample in samples]
    return sum(wall_counts) / len(wall_counts)


@torch.no_grad()
def generate_and_save_samples(model, diffusion, device, num_samples, output_path):
    """Generate num_samples puzzles and write them as a Boxoban-format .txt file."""
    model.eval()
    samples = diffusion.sample(model, num_samples, device, strategy='random')

    puzzles = [tensor_to_puzzle(sample) for sample in samples]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for i, puzzle in enumerate(puzzles):
            f.write(f"; {i}\n")
            f.write(puzzle + '\n')

    return puzzles


def _is_solvable(puzzle: str, max_states: int) -> bool:
    """Run in a worker process: structural validity + BFS solvability for one puzzle."""
    result = validate_puzzle(puzzle)
    if not result['valid']:
        return False

    matrix, player_pos = boxoban_to_matrix(puzzle)
    if player_pos is None:
        return False

    solution, _ = solve_push(matrix, player_pos, max_states=max_states)
    return solution is not None


def ensure_samples_exist(step, model, diffusion, device, train_config, train_dist_counts):
    """If this checkpoint's samples/perf.txt entry was never produced (e.g. training
    was interrupted right after the checkpoint save but before sample generation
    finished), generate it now before continuing training."""
    samples_path = os.path.join(train_config.samples_dir, f'ckp_{step}', 'samples.txt')
    if os.path.exists(samples_path):
        return

    print(f"Samples for step {step} were never generated (interrupted?) -- generating now...")
    puzzles = generate_and_save_samples(
        model, diffusion, device, train_config.samples_per_checkpoint, samples_path
    )
    print(f"  Saved {train_config.samples_per_checkpoint} samples to {samples_path}")

    perf_path = os.path.join(train_config.samples_dir, 'perf.txt')
    log_perf(step, puzzles, train_dist_counts, perf_path)

    model.train()


def load_train_distribution(path):
    """Load the training-set tile-pattern counts (N=3) for JSD comparison, or None if missing."""
    if not os.path.exists(path):
        print(f"Warning: training distribution not found at {path} -- skipping JSD logging")
        return None
    with open(path) as f:
        return json.load(f)['counts']


def log_perf(step, puzzles, train_dist_counts, perf_path, max_states=200000, workers=None):
    """Append one line to perf.txt: solvability rate and N=3 tile-pattern JSD vs. training."""
    workers = workers or os.cpu_count()
    worker = functools.partial(_is_solvable, max_states=max_states)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        solvable_flags = list(executor.map(worker, puzzles, chunksize=8))
    solvable_count = sum(solvable_flags)

    line = f"step {step:>7} | solvable: {solvable_count:4d}/{len(puzzles)} ({100 * solvable_count / len(puzzles):5.1f}%)"

    if train_dist_counts is not None:
        gen_counts = Counter()
        for puzzle in puzzles:
            grid = puzzle_to_grid(puzzle)
            gen_counts.update(extract_patterns(grid, 3))
        jsd = js_divergence(train_dist_counts, gen_counts)
        line += f" | N=3 JSD: {jsd:.4f}"

    os.makedirs(os.path.dirname(perf_path), exist_ok=True)
    with open(perf_path, 'a') as f:
        f.write(line + '\n')

    print(f"  {line}")


def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    """Save model checkpoint."""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['step'], checkpoint['loss']


def find_latest_checkpoint(checkpoint_dir):
    """Return the path to the highest-step periodic checkpoint (step_*.pt), or None.

    Deliberately excludes best.pt -- that file tracks best validation loss, not
    training progression, so it can lag behind or otherwise not reflect the
    furthest point actually reached.
    """
    paths = glob.glob(os.path.join(checkpoint_dir, 'step_*.pt'))
    if not paths:
        return None

    latest_step = -1
    latest_path = None
    for path in paths:
        checkpoint = torch.load(path, map_location='cpu')
        if checkpoint['step'] > latest_step:
            latest_step = checkpoint['step']
            latest_path = path

    return latest_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the Sokoban masked-diffusion model')
    parser.add_argument('--w-max', type=float, default=None,
                        help='Override ModelConfig.w_max (loss-weight cap). Leaves config.py untouched.')
    parser.add_argument('--num-timesteps', type=int, default=None,
                        help='Override ModelConfig.num_timesteps (# noise levels, L). Leaves config.py untouched.')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Training corpus (folder of Boxoban .txt, or a single .txt). '
                             'Overrides TrainConfig.data_path -- used to train on a '
                             'difficulty-stratified subset.')
    parser.add_argument('--val-path', type=str, default=None,
                        help='Validation corpus. Overrides TrainConfig.val_path.')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Stop after N optimizer steps and set the cosine horizon to N. '
                             'Required for fair comparison across corpora of different '
                             'sizes, where equal epochs != equal steps.')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Override checkpoint output dir (e.g. ./checkpoints_100 for an isolated run)')
    parser.add_argument('--samples-dir', type=str, default=None,
                        help='Override per-checkpoint samples/perf output dir (e.g. ./samples_100)')
    parser.add_argument('--bf16', action='store_true',
                        help='Train in bfloat16 mixed precision (no GradScaler) instead of fp16')
    parser.add_argument('--test', action='store_true', help='Quick smoke-test run (1 epoch, frequent eval/save)')
    args = parser.parse_args()

    model_config = ModelConfig()
    train_config = TrainConfig()

    # CLI overrides -- specified on the command line so config.py stays the
    # canonical baseline and isolated experiments don't mutate shared state.
    if args.w_max is not None:
        model_config.w_max = args.w_max
        print(f"[override] w_max = {model_config.w_max}")
    if args.num_timesteps is not None:
        model_config.num_timesteps = args.num_timesteps
        print(f"[override] num_timesteps (L) = {model_config.num_timesteps}")
    if args.data_path is not None:
        train_config.data_path = args.data_path
        print(f"[override] data_path = {train_config.data_path}")
    if args.val_path is not None:
        train_config.val_path = args.val_path
        print(f"[override] val_path = {train_config.val_path}")
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
        print(f"[override] max_steps = {train_config.max_steps}")
    if args.checkpoint_dir is not None:
        train_config.checkpoint_dir = args.checkpoint_dir
        print(f"[override] checkpoint_dir = {train_config.checkpoint_dir}")
    if args.samples_dir is not None:
        train_config.samples_dir = args.samples_dir
        print(f"[override] samples_dir = {train_config.samples_dir}")

    if args.test:
        train_config.num_epochs = 1
        train_config.log_every = 10
        train_config.eval_every = 50
        train_config.save_every = 100

    train(model_config, train_config, use_bf16=args.bf16)
