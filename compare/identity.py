"""
Memorization / novelty check: are generated puzzles actually NEW?

Stacks the full training set into one (K, 100) tensor and a set of generated
puzzles into an (M, 100) tensor, then compares every generated puzzle against
every training puzzle via an inner product.

Why one-hot. A plain inner product of token-ID vectors is NOT an identity test:
<a, b> over integers 0..6 says nothing about whether a == b. One-hot encoding
each puzzle to a 700-dim binary vector (100 cells x 7 tile types) makes the
inner product exactly the number of cells on which two puzzles AGREE, so

    <a, b> == 100  <=>  the puzzles are identical
    hamming(a, b)  ==  100 - <a, b>

which turns the whole comparison into one (M, 700) @ (700, K) matmul -- tensor
cores, and the near-miss distances come out for free alongside exact matches.

Scale note: the full (M, K) matrix is not materialized. At M=10,000 / K=450,000
that would be 4.5 GB for a result we only ever reduce over. Instead K is chunked
and a running max-inner-product (== running min Hamming distance) plus its
argmax is kept per query, so memory is bounded by one (M, chunk) block.

The exact-duplicate count is cross-checked against a hash-based set intersection,
which is O(M+K) and independent of the matmul path.

Usage:
    python identity.py --query ../source/diffusion/samples_2/ckp_225000/samples.txt
    python identity.py --query ... --baseline      # add held-out + train-vs-train nulls
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
S_DIR = os.path.dirname(SCRIPT_DIR)
SOURCE_DIR = os.path.join(S_DIR, 'source')
DIFFUSION_DIR = os.path.join(SOURCE_DIR, 'diffusion')
sys.path.insert(0, DIFFUSION_DIR)

from config import TrainConfig
from dataset import CHAR_TO_ID

GRID = 10
L = GRID * GRID
V = 7  # tile types, excluding [MASK] (never appears in a finished puzzle)

DEFAULT_QUERY = os.path.join(DIFFUSION_DIR, 'samples_2', 'ckp_225000', 'samples.txt')

# char -> token id lookup over the full byte range, so parsing is a vectorized
# table index rather than a per-cell dict lookup.
_LUT = np.full(256, CHAR_TO_ID[' '], dtype=np.uint8)  # unknown chars -> floor, as dataset.py does
for _c, _i in CHAR_TO_ID.items():
    _LUT[ord(_c)] = _i


def load_puzzles(path: str) -> np.ndarray:
    """Parse Boxoban-format .txt file(s) into a single (N, 100) uint8 array.

    Accepts a file or a folder of .txt files. Puzzle blocks are ';'-delimited
    exactly as in dataset.parse_boxoban_file; this is the vectorized equivalent.
    """
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, '*.txt')))
    if not files:
        raise SystemExit(f"no .txt files found at {path}")

    grids = []
    for filepath in files:
        with open(filepath) as f:
            rows = [ln.rstrip('\n') for ln in f if ln[:1] != ';' and ln.strip('\n')]
        # Every puzzle is a fixed 10x10 block, so the surviving lines are just
        # the grid rows back to back. Pad/truncate defensively to width 10.
        if len(rows) % GRID != 0:
            raise SystemExit(f"{filepath}: {len(rows)} grid rows is not a multiple of {GRID}")
        packed = ''.join(r.ljust(GRID)[:GRID] for r in rows)
        arr = _LUT[np.frombuffer(packed.encode('latin-1'), dtype=np.uint8)]
        grids.append(arr.reshape(-1, L))

    return np.concatenate(grids, axis=0)


def one_hot(tokens: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N, 100) token ids -> (N, 700) fp16 one-hot, on device.

    Values are 0/1 and inner products top out at 100, all exactly representable
    in fp16 (integers are exact below 2048), so the matmul is exact.
    """
    t = torch.from_numpy(tokens.astype(np.int64)).to(device)
    oh = torch.zeros((t.shape[0], L, V), dtype=torch.float16, device=device)
    oh.scatter_(2, t.unsqueeze(-1), 1.0)
    return oh.reshape(t.shape[0], L * V)


def nearest_neighbors(query: np.ndarray, ref: np.ndarray, device: torch.device,
                      chunk: int = 50000, self_index: np.ndarray = None):
    """Per-query best match against ref, via chunked one-hot inner product.

    Returns (best_match_count, best_index) as (M,) int16 / int64 numpy arrays,
    where best_match_count is the number of agreeing cells (100 == identical).

    self_index gives, for each query row, the row of ref that IS that same puzzle
    (or -1 if absent). Those entries are masked so a puzzle never matches itself
    -- required whenever query is drawn from ref, otherwise every result is a
    trivial perfect self-match. Pass np.arange(M) when query IS ref.
    """
    M, K = query.shape[0], ref.shape[0]
    q = one_hot(query, device)                       # (M, 700)

    best = torch.full((M,), -1, dtype=torch.float16, device=device)
    best_idx = torch.zeros((M,), dtype=torch.long, device=device)

    si = None if self_index is None else torch.from_numpy(np.asarray(self_index)).to(device)

    for start in range(0, K, chunk):
        end = min(start + chunk, K)
        r = one_hot(ref[start:end], device)          # (|b|, 700)
        block = q @ r.T                              # (M, |b|) agreeing-cell counts

        if si is not None:
            # mask the (query row, its own ref row) entry for rows whose twin
            # falls inside this chunk
            hit = (si >= start) & (si < end)
            if hit.any():
                rows = torch.nonzero(hit, as_tuple=True)[0]
                block[rows, si[rows] - start] = -1

        blk_best, blk_arg = block.max(dim=1)
        improved = blk_best > best
        best = torch.where(improved, blk_best, best)
        best_idx = torch.where(improved, blk_arg + start, best_idx)

        del r, block

    return best.to(torch.int16).cpu().numpy(), best_idx.cpu().numpy()


def hash_duplicates(query: np.ndarray, ref: np.ndarray):
    """Exact-duplicate count via set intersection of raw row bytes -- an O(M+K)
    cross-check on the matmul path, sharing none of its code."""
    qb = [row.tobytes() for row in query]
    rb = {row.tobytes() for row in ref}
    hits = [i for i, b in enumerate(qb) if b in rb]
    return hits, len(set(qb))


def describe(name: str, agree: np.ndarray) -> dict:
    """Summarize a nearest-neighbor result. agree == matching cells out of 100."""
    dist = 100 - agree.astype(np.int32)
    return {
        'set': name,
        'n': len(dist),
        'identical': int((dist == 0).sum()),
        'dist<=2': int((dist <= 2).sum()),
        'dist<=5': int((dist <= 5).sum()),
        'min': int(dist.min()),
        'p1': float(np.percentile(dist, 1)),
        'median': float(np.median(dist)),
        'mean': float(dist.mean()),
        'max': int(dist.max()),
    }


def print_table(rows):
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))
                                 for r in rows)) for c in cols}
    print('  '.join(c.rjust(widths[c]) for c in cols))
    print('  '.join('-' * widths[c] for c in cols))
    for r in rows:
        print('  '.join((f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])).rjust(widths[c])
                        for c in cols))


def main():
    p = argparse.ArgumentParser(description='Exact-duplicate / novelty check via one-hot inner product')
    p.add_argument('--query', type=str, action='append', default=None,
                   help='samples.txt of generated puzzles (repeatable). Default: baseline 10K set.')
    p.add_argument('--train-path', type=str, default=TrainConfig.data_path)
    p.add_argument('--val-path', type=str, default=TrainConfig.val_path)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--chunk', type=int, default=50000, help='training puzzles per matmul block')
    p.add_argument('--baseline', action='store_true',
                   help='also compute held-out-vs-train and train-vs-train nulls (interpretation baselines)')
    p.add_argument('--baseline-n', type=int, default=10000,
                   help='puzzles to subsample for the baseline runs')
    args = p.parse_args()

    queries = args.query or [DEFAULT_QUERY]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    t0 = time.time()
    train = load_puzzles(args.train_path)
    print(f"training set: {train.shape[0]:,} puzzles  ({time.time() - t0:.1f}s)")

    rows = []
    for qpath in queries:
        q = load_puzzles(qpath)
        label = os.path.relpath(qpath, S_DIR)
        print(f"\n=== {label}: {q.shape[0]:,} puzzles ===")

        t0 = time.time()
        agree, idx = nearest_neighbors(q, train, device, args.chunk)
        print(f"  vs train: {time.time() - t0:.1f}s")

        hits, n_unique = hash_duplicates(q, train)
        matmul_hits = set(np.flatnonzero(agree == 100).tolist())
        assert matmul_hits == set(hits), f"matmul/hash disagree: {matmul_hits ^ set(hits)}"
        print(f"  cross-check OK: {len(hits)} exact duplicate(s) of training data (both methods agree)")
        print(f"  distinct puzzles within the set: {n_unique:,}/{q.shape[0]:,} "
              f"({q.shape[0] - n_unique:,} internal duplicate(s))")

        rows.append(describe(f"{label} vs train", agree))

        # self-comparison: is the model repeating ITSELF (mode collapse)?
        agree_self, _ = nearest_neighbors(q, q, device, args.chunk,
                                          self_index=np.arange(q.shape[0]))
        rows.append(describe(f"{label} vs itself", agree_self))

        if len(hits) <= 20 and hits:
            print(f"  duplicate query ids: {hits}")

    if args.baseline:
        rng = np.random.default_rng(0)
        val = load_puzzles(args.val_path)
        vsub = val[rng.choice(len(val), min(args.baseline_n, len(val)), replace=False)]
        print(f"\n=== baselines ===")
        agree_v, _ = nearest_neighbors(vsub, train, device, args.chunk)
        rows.append(describe("held-out real vs train", agree_v))

        # train-vs-train: subsample query rows but compare against all K, masking
        # each row's own position in the reference set
        sel = rng.choice(len(train), min(args.baseline_n, len(train)), replace=False)
        agree_t, _ = nearest_neighbors(train[sel], train, device, args.chunk, self_index=sel)
        rows.append(describe("train vs train (excl. self)", agree_t))

    print(f"\n{'=' * 78}")
    print("Nearest-neighbor Hamming distance (0 == identical puzzle, out of 100 cells)")
    print('=' * 78)
    print_table(rows)


if __name__ == '__main__':
    main()
