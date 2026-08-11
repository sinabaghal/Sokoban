"""
N x N tile-pattern distribution over the Sokoban training dataset.

Slides an N x N window (stride 1) over every puzzle in the training set and
tallies how often each distinct N x N tile pattern occurs, producing an
empirical distribution over local tile structure. Unlike a single global
scalar (e.g. wall count), this captures local arrangements -- corners,
corridors, dead ends, box-against-wall -- that determine whether a puzzle
looks structurally plausible.

Usage:
    python tile_pattern.py --tile-size 3
    python tile_pattern.py --tile-size 2 --data-path ../../boxoban-levels/medium/valid
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
DIFFUSION_DIR = os.path.join(SOURCE_DIR, 'diffusion')
sys.path.insert(0, DIFFUSION_DIR)

from config import TrainConfig
from dataset import parse_boxoban_file  # imports torch; must load before matplotlib

import matplotlib.pyplot as plt

DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'distributions')
DEFAULT_PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots')


def puzzle_to_grid(puzzle: str, height: int = 10, width: int = 10):
    """Pad/truncate a raw puzzle string to a height x width list of char rows."""
    lines = puzzle.strip('\n').split('\n')
    grid = []
    for i in range(height):
        row = lines[i] if i < len(lines) else ''
        row = row.ljust(width)[:width]
        grid.append(row)
    return grid


def extract_patterns(grid, n: int):
    """Yield every N x N tile pattern (flattened row-major to a string) via a stride-1 slide."""
    height = len(grid)
    width = len(grid[0])
    for i in range(height - n + 1):
        for j in range(width - n + 1):
            yield ''.join(grid[i + di][j:j + n] for di in range(n))


def build_distribution(data_path: str, n: int) -> Counter:
    files = sorted(glob.glob(os.path.join(data_path, '*.txt')))
    counts = Counter()

    for i, filepath in enumerate(files):
        for puzzle in parse_boxoban_file(filepath):
            grid = puzzle_to_grid(puzzle)
            counts.update(extract_patterns(grid, n))

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(files)} files...")

    return counts


def entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def plot_distribution(counts: Counter, tile_size: int, output_path: str, label: str = 'Boxoban Training Set'):
    """Log-log rank-frequency plot -- shows the full long-tail shape, not just the head."""
    frequencies = sorted(counts.values(), reverse=True)
    ranks = range(1, len(frequencies) + 1)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')

    ax.plot(ranks, frequencies, color='#2a78d6', linewidth=1.5, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Pattern rank (by frequency)', color='#0b0b0b')
    ax.set_ylabel('Occurrence count', color='#0b0b0b')
    ax.set_title(f'{tile_size}x{tile_size} Tile-Pattern Rank-Frequency — {label}',
                 color='#0b0b0b')

    ax.grid(True, which='both', color='#e1e0d9', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#52514e')
    for spine in ax.spines.values():
        spine.set_color('#c3c2b7')

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Build an N x N tile-pattern distribution over the training dataset')
    parser.add_argument('--tile-size', '-n', type=int, required=True,
                         help='Tile window size N (e.g. 2 for 2x2, 3 for 3x3)')
    parser.add_argument('--data-path', type=str, default=TrainConfig.data_path,
                         help='Folder of Boxoban .txt files (default: training split)')
    parser.add_argument('--output', type=str, default=None,
                         help='Output JSON path (default: distributions/tile_pattern_n{N}.json)')
    parser.add_argument('--plot-output', type=str, default=None,
                         help='Output PNG path (default: plots/tile_pattern_n{N}.png)')
    parser.add_argument('--top-k', type=int, default=10,
                         help='Number of most common patterns to print')
    parser.add_argument('--label', type=str, default='Boxoban Training Set',
                         help='Dataset label used in the plot title')
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"Error: data path not found: {args.data_path}")
        sys.exit(1)

    if not (1 <= args.tile_size <= 10):
        print(f"Error: --tile-size must be between 1 and 10 (got {args.tile_size})")
        sys.exit(1)

    print(f"Scanning puzzles in {args.data_path} with {args.tile_size}x{args.tile_size} tiles...")
    counts = build_distribution(args.data_path, args.tile_size)

    total_windows = sum(counts.values())
    unique_patterns = len(counts)
    dist_entropy = entropy_bits(counts)
    max_entropy = math.log2(unique_patterns) if unique_patterns > 1 else 0.0

    print(f"\nTotal windows counted: {total_windows:,}")
    print(f"Unique patterns seen:  {unique_patterns:,}")
    print(f"Distribution entropy:  {dist_entropy:.3f} bits (max given support: {max_entropy:.3f})")

    print(f"\nTop {args.top_k} most common {args.tile_size}x{args.tile_size} patterns:")
    for pattern, count in counts.most_common(args.top_k):
        rows = [pattern[i:i + args.tile_size] for i in range(0, len(pattern), args.tile_size)]
        display = ' / '.join(rows)
        print(f"  [{display}] count={count:,} ({100 * count / total_windows:.2f}%)")

    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f'tile_pattern_n{args.tile_size}.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'tile_size': args.tile_size,
            'data_path': args.data_path,
            'total_windows': total_windows,
            'unique_patterns': unique_patterns,
            'entropy_bits': dist_entropy,
            'counts': dict(counts),
        }, f, indent=2)

    print(f"\nSaved distribution: {output_path}")

    plot_output_path = args.plot_output or os.path.join(DEFAULT_PLOT_DIR, f'tile_pattern_n{args.tile_size}.png')
    plot_distribution(counts, args.tile_size, plot_output_path, label=args.label)
    print(f"Saved plot:         {plot_output_path}")


if __name__ == '__main__':
    main()
