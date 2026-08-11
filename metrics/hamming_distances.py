"""
Nearest-neighbour Hamming distance to the training corpus, for generated
puzzles and for real held-out puzzles measured identically.

The point of the comparison. A distance is meaningless on its own -- puzzles
are 100 cells over 7 tile types, so "13 cells differ" carries no intuition
about whether that is close or far. Measuring genuine held-out puzzles the
same way supplies the reference: whatever they score is what a generator that
had memorised nothing, drawn from the true distribution, would score.

Distance is computed by one-hot encoding each puzzle to 700 dims, so an inner
product counts agreeing cells and distance is 100 minus it. Reuses
compare/identity.py's chunked implementation rather than reimplementing it.

Writes a histogram (one row per integer distance) to viz/csv/hamming.csv.

    python hamming_distances.py
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
S_DIR = os.path.dirname(SOURCE_DIR)
sys.path.insert(0, os.path.join(SOURCE_DIR, 'diffusion'))
sys.path.insert(0, os.path.join(S_DIR, 'compare'))

from config import TrainConfig
from identity import load_puzzles, nearest_neighbors

OUT_CSV = os.path.join(SOURCE_DIR, 'viz', 'csv', 'hamming.csv')
DEFAULT_GEN = os.path.join(SCRIPT_DIR, 'bulk_samples_50k.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated', default=DEFAULT_GEN)
    ap.add_argument('--train-path', default=TrainConfig.data_path)
    ap.add_argument('--val-path', default=TrainConfig.val_path)
    ap.add_argument('--chunk', type=int, default=10000)
    ap.add_argument('--raw', action='store_true',
                    help='compare grids literally, counting a moved player as a '
                         'difference. Default canonicalises the player away.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=OUT_CSV)
    args = ap.parse_args()

    def strip_player(a):
        """Player position is not part of a puzzle's identity: the worker walks
        freely within its reachable region, and the push solver canonicalises it
        away for exactly that reason. Two grids differing only in where the
        worker stands are the same puzzle, so counting that as a difference
        would undercount duplicates."""
        b = a.copy()
        b[b == 2] = 1      # player          -> floor
        b[b == 6] = 4      # player-on-goal  -> goal
        return b

    device = torch.device(args.device)
    train = load_puzzles(args.train_path)
    if not args.raw:
        train = strip_player(train)
    print(f'training corpus: {train.shape[0]:,} puzzles')

    series = {}
    for name, path in (('generated', args.generated), ('held_out', args.val_path)):
        q = load_puzzles(path)
        if not args.raw:
            q = strip_player(q)
        agree, _ = nearest_neighbors(q, train, device, args.chunk)
        dist = 100 - agree.astype(np.int32)
        series[name] = dist
        print(f'{name:<10} n={len(dist):,}  identical={int((dist==0).sum())}  '
              f'median={np.median(dist):.0f}  mean={dist.mean():.2f}  '
              f'<=2={int((dist<=2).sum())}  <=5={int((dist<=5).sum())}')

    hi = max(d.max() for d in series.values())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['distance', 'generated_pct', 'held_out_pct',
                    'generated_n', 'held_out_n'])
        for d in range(hi + 1):
            g = int((series['generated'] == d).sum())
            h = int((series['held_out'] == d).sum())
            w.writerow([d,
                        f"{100*g/len(series['generated']):.5f}",
                        f"{100*h/len(series['held_out']):.5f}",
                        g, h])
    print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()
