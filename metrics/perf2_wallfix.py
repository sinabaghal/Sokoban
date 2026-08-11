"""
Extended per-checkpoint metrics for the w_max=10 10K-sample sets (samples_2/):
raw solvability PLUS "wall-fix" rates -- how many unsolvable puzzles are
near-misses that become solvable by removing a few interior walls.
Writes samples_2/perf_2.txt.

Method:
  - solvable rate: EXACT, from solving all 10K puzzles (push solver).
  - +1wall / +2wall rates: ESTIMATED from a random SAMPLE of the unsolvable
    puzzles (default 100). Because it's only ~100 puzzles, the wall search uses
    fully-accurate settings (every interior wall, full 200k state cap).

Reported per checkpoint (cumulative = effective solvability after a k-wall repair):
  solvable = exact push-solvable fraction
  +1wall   = solvable + (unsolvable fraction) * (sampled frac fixable with <=1 wall)
  +2wall   = solvable + (unsolvable fraction) * (sampled frac fixable with <=2 walls)

Resumable: checkpoints already in perf_2.txt are skipped.

Usage:
    python perf2_wallfix.py                # 100-sample estimate
    python perf2_wallfix.py --sample 200
"""

import argparse
import glob
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
DIFFUSION_DIR = os.path.join(SOURCE_DIR, 'diffusion')
sys.path.insert(0, SOURCE_DIR)
sys.path.insert(0, DIFFUSION_DIR)

from dataset import parse_boxoban_file
from test_solver import boxoban_to_matrix, solve_push

SAMPLES_DIR = os.path.join(DIFFUSION_DIR, 'samples_2')
MAX_STATES = 200000


def _solvable(puzzle):
    matrix, player_pos = boxoban_to_matrix(puzzle)
    if player_pos is None:
        return None  # structurally invalid
    sol, _ = solve_push(matrix, player_pos, max_states=MAX_STATES)
    return sol is not None


def _solvable_after(lines, remove):
    grid = [list(row) for row in lines]
    for (r, c) in remove:
        grid[r][c] = ' '
    return _solvable('\n'.join(''.join(row) for row in grid)) is True


def min_fix(args):
    """For an unsolvable puzzle: min interior walls (<=max_remove) to make solvable, else max_remove+1.
    Accurate: considers every interior wall."""
    puzzle, max_remove = args
    lines = puzzle.split('\n')
    H, W = len(lines), len(lines[0])
    walls = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1) if lines[r][c] == '#']
    for k in range(1, max_remove + 1):
        for combo in combinations(walls, k):
            if _solvable_after(lines, combo):
                return k
    return max_remove + 1


def done_steps(perf_path):
    done = set()
    if os.path.exists(perf_path):
        with open(perf_path) as f:
            for line in f:
                m = re.match(r'\s*step\s+(\d+)', line)
                if m:
                    done.add(int(m.group(1)))
    return done


def load_solvable_rates(perf_path):
    """Parse exact solvable counts from the existing samples_2/perf.txt: {step: (solvable, total)}."""
    rates = {}
    if os.path.exists(perf_path):
        with open(perf_path) as f:
            for line in f:
                m = re.search(r'step\s+(\d+)\s*\|\s*solvable:\s*(\d+)/(\d+)', line)
                if m:
                    rates[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return rates


def collect_unsolvable(puzzles, want, seed, workers):
    """Scan puzzles in shuffled order, solving in parallel batches, until `want` unsolvable are found."""
    idx = list(range(len(puzzles)))
    random.Random(seed).shuffle(idx)
    found = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        batch = max(want, workers * 4)
        pos = 0
        while pos < len(idx) and len(found) < want:
            chunk = [puzzles[i] for i in idx[pos:pos + batch]]
            for p, s in zip(chunk, ex.map(_solvable, chunk, chunksize=8)):
                if s is not True:
                    found.append(p)
                    if len(found) >= want:
                        break
            pos += batch
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', default=SAMPLES_DIR)
    parser.add_argument('--sample', type=int, default=100, help='Unsolvable puzzles to sample for fix-rate estimate')
    parser.add_argument('--max-remove', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    perf_path = os.path.join(args.samples_dir, 'perf_2.txt')
    rates = load_solvable_rates(os.path.join(args.samples_dir, 'perf.txt'))
    done = done_steps(perf_path)

    ckp_dirs = glob.glob(os.path.join(args.samples_dir, 'ckp_*'))
    steps = sorted((int(re.search(r'ckp_(\d+)', d).group(1)), d) for d in ckp_dirs)
    steps = [(s, d) for s, d in steps if s not in done]

    print(f"perf_2 wall-fix (exact solvable + {args.sample}-sample fix estimate, cap k={args.max_remove})")
    print(f"Checkpoints to run: {[s for s, _ in steps]}\n", flush=True)

    if not os.path.exists(perf_path):
        with open(perf_path, 'w') as f:
            f.write(f"# solvable = exact push-solvable (10K). +1wall/+2wall = cumulative effective "
                    f"solvability after removing <=1/<=2 interior walls, ESTIMATED from a "
                    f"{args.sample}-puzzle random sample of the unsolvable set.\n")

    for step, d in steps:
        if step not in rates:
            print(f"step {step}: no solvable rate in perf.txt -- skipping")
            continue
        solvable, n = rates[step]
        solv_frac = solvable / n
        unsolv_frac = 1 - solv_frac

        puzzles = parse_boxoban_file(os.path.join(d, 'samples.txt'))
        print(f"step {step}: solvable {solvable}/{n} ({100*solv_frac:.1f}%, from perf.txt); "
              f"scanning for {args.sample} unsolvable to sample...", flush=True)

        sample = collect_unsolvable(puzzles, args.sample, args.seed, args.workers)
        print(f"  collected {len(sample)} unsolvable; estimating fix-rates (accurate wall search)...", flush=True)

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fixes = list(ex.map(min_fix, [(p, args.max_remove) for p in sample], chunksize=4))

        m = len(sample)
        f1 = sum(1 for k in fixes if k <= 1) / m if m else 0.0   # frac of unsolvable fixable with <=1 wall
        f2 = sum(1 for k in fixes if k <= 2) / m if m else 0.0
        within1 = solv_frac + unsolv_frac * f1
        within2 = solv_frac + unsolv_frac * f2

        line = (f"step {step:>7} | solvable: {100*solv_frac:5.1f}% (exact)"
                f" | +1wall: {100*within1:5.1f}% | +2wall: {100*within2:5.1f}%"
                f"   [est n={m}: {100*f1:.0f}% of unsolv fixable by 1, {100*f2:.0f}% by <=2]")
        with open(perf_path, 'a') as f:
            f.write(line + '\n')
        print(f"  {line}\n", flush=True)

    print(f"Done. {perf_path}")


if __name__ == '__main__':
    main()
