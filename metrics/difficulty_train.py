"""Per-puzzle push-solver difficulty of the Boxoban corpus, persisted for reuse.

The project records solution length (pushes) and search effort (push-nodes
expanded) for GENERATED puzzles only. There was no corpus-side reference, so
"the model generates puzzles of median 14 pushes" had nothing to compare
against. This computes that baseline -- and, unlike a summary-statistics run,
saves ONE ROW PER PUZZLE so the results can be reused to build difficulty-
stratified training corpora without ever re-solving.

Solving the full 450k corpus costs hours, so the output is designed to be
computed once and never again:

  * one row per puzzle, keyed by a stable `puzzle_id`
  * the grid itself is stored, so partitions can be written out from this file
    alone -- no dependence on re-parsing the corpus in the same order
  * `walls` is stored because difficulty is confounded with wall density
    (r = -0.494); building an unconfounded partition requires matching on it
  * written incrementally as Parquet row groups and RESUMABLE, so an
    interrupted multi-hour run resumes instead of restarting

Ordering is deterministic: files in sorted(glob) order, puzzles in file order,
`puzzle_id` a running counter from 0.

Usage:
    python difficulty_train.py --num 10000          # sample (writes sampled rows)
    python difficulty_train.py                      # FULL corpus (hours; resumable)
    python difficulty_train.py --split valid
    python difficulty_train.py --summary-only       # re-print stats from existing file
"""

import argparse
import functools
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SOURCE_DIR, 'diffusion'))
sys.path.insert(0, SOURCE_DIR)

from config import TrainConfig
from dataset import parse_boxoban_file
from test_solver import boxoban_to_matrix, solve_push

SCHEMA = pa.schema([
    ('puzzle_id', pa.int32()),          # stable index into the deterministic enumeration
    ('source_file', pa.string()),       # provenance
    ('idx_in_file', pa.int32()),
    ('grid', pa.string()),              # 100 chars, row-major -- makes this file self-contained
    ('solved', pa.bool_()),
    ('status', pa.string()),            # solvable / unsolvable / unknown_capped / invalid
    ('pushes', pa.int32()),             # -1 if not solved
    ('states_expanded', pa.int32()),    # search effort; the validated difficulty proxy
    ('walls', pa.int16()),              # confound covariate -- needed for matched partitioning
    ('boxes', pa.int16()),
    ('goals', pa.int16()),
])


def solve_one(args, max_states):
    """(solved, status, pushes, states) for one puzzle. Mirrors run_eval.py's call."""
    puzzle = args
    try:
        matrix, pos = boxoban_to_matrix(puzzle)
        if pos is None:
            return (False, 'invalid', -1, 0)
        sol, _n, stats = solve_push(matrix, pos, max_states=max_states, return_stats=True)
        states = int(stats['states_expanded'])
        if sol is not None:
            return (True, 'solvable', int(stats['pushes']), states)
        # a capped run ends AT the cap; a proven-unsolvable one exits below it
        return (False, 'unknown_capped' if states >= max_states else 'unsolvable', -1, states)
    except Exception:
        return (False, 'invalid', -1, 0)


def enumerate_corpus(path):
    """Deterministic (puzzle_id, source_file, idx_in_file, puzzle) enumeration."""
    out = []
    pid = 0
    for fp in sorted(glob.glob(os.path.join(path, '*.txt'))):
        base = os.path.basename(fp)
        for j, pz in enumerate(parse_boxoban_file(fp)):
            out.append((pid, base, j, pz))
            pid += 1
    return out


def counts(puzzle):
    return (puzzle.count('#'),
            puzzle.count('$') + puzzle.count('*'),
            puzzle.count('.') + puzzle.count('*') + puzzle.count('+'))


def summarize(path):
    t = pq.read_table(path)
    solved = t.column('solved').to_numpy(zero_copy_only=False)
    p = t.column('pushes').to_numpy()[solved]
    s = t.column('states_expanded').to_numpy()[solved]
    out = {
        'rows': t.num_rows, 'solved': int(solved.sum()),
        'solved_pct': float(100 * solved.mean()),
        'pushes': {k: float(v) for k, v in zip(
            ('p10', 'median', 'mean', 'p90', 'max'),
            (np.percentile(p, 10), np.median(p), p.mean(), np.percentile(p, 90), p.max()))},
        'states': {k: float(v) for k, v in zip(
            ('p10', 'median', 'mean', 'p90', 'max'),
            (np.percentile(s, 10), np.median(s), s.mean(), np.percentile(s, 90), s.max()))},
    }
    print(f"rows {out['rows']:,} | solved {out['solved']:,} ({out['solved_pct']:.2f}%)")
    print(f"pushes         p10 {out['pushes']['p10']:.0f}  median {out['pushes']['median']:.1f}"
          f"  mean {out['pushes']['mean']:.2f}  p90 {out['pushes']['p90']:.0f}  max {out['pushes']['max']:.0f}")
    print(f"search effort  p10 {out['states']['p10']:.0f}  median {out['states']['median']:.0f}"
          f"  mean {out['states']['mean']:.0f}  p90 {out['states']['p90']:.0f}  max {out['states']['max']:.0f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num', type=int, default=0, help='0 = full corpus; else a seeded random sample')
    ap.add_argument('--split', choices=['train', 'valid'], default='train')
    ap.add_argument('--max-states', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=5000, help='rows per Parquet row group / resume granularity')
    ap.add_argument('--out', default=None)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    tag = args.split if not args.num else f'{args.split}_{args.num}'
    out_path = args.out or os.path.join(SCRIPT_DIR, f'difficulty_{tag}.parquet')

    if args.summary_only:
        if not os.path.exists(out_path):
            sys.exit(f'no such file: {out_path}')
        summarize(out_path)
        return

    path = TrainConfig.data_path if args.split == 'train' else TrainConfig.val_path
    corpus = enumerate_corpus(path)
    print(f'{args.split}: {len(corpus):,} puzzles enumerated')

    if args.num:
        sel = np.random.default_rng(args.seed).choice(len(corpus), min(args.num, len(corpus)),
                                                      replace=False)
        corpus = [corpus[i] for i in sorted(sel)]
        print(f'  sampled {len(corpus):,} (seed {args.seed}); puzzle_id preserves corpus position')

    # resume: count rows already written and skip that many
    done = 0
    if os.path.exists(out_path):
        try:
            done = pq.ParquetFile(out_path).metadata.num_rows
        except Exception:
            done = 0
        if done >= len(corpus):
            print(f'already complete ({done:,} rows) -> {out_path}')
            summarize(out_path)
            return
        if done:
            # Parquet cannot be appended to in place; rewrite what exists, then continue.
            print(f'resuming: {done:,} rows already present')
            existing = pq.read_table(out_path)
            tmp = out_path + '.tmp'
            writer = pq.ParquetWriter(tmp, SCHEMA, compression='zstd')
            writer.write_table(existing)
    if not done:
        tmp = out_path + '.tmp'
        writer = pq.ParquetWriter(tmp, SCHEMA, compression='zstd')

    worker = functools.partial(solve_one, max_states=args.max_states)
    todo = corpus[done:]
    import time
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        for start in range(0, len(todo), args.chunk):
            block = todo[start:start + args.chunk]
            res = list(ex.map(worker, [b[3] for b in block], chunksize=16))
            cnt = [counts(b[3]) for b in block]
            writer.write_table(pa.Table.from_pydict({
                'puzzle_id':      pa.array([b[0] for b in block], pa.int32()),
                'source_file':    pa.array([b[1] for b in block], pa.string()),
                'idx_in_file':    pa.array([b[2] for b in block], pa.int32()),
                'grid':           pa.array([b[3].replace('\n', '') for b in block], pa.string()),
                'solved':         pa.array([r[0] for r in res], pa.bool_()),
                'status':         pa.array([r[1] for r in res], pa.string()),
                'pushes':         pa.array([r[2] for r in res], pa.int32()),
                'states_expanded': pa.array([r[3] for r in res], pa.int32()),
                'walls':          pa.array([c[0] for c in cnt], pa.int16()),
                'boxes':          pa.array([c[1] for c in cnt], pa.int16()),
                'goals':          pa.array([c[2] for c in cnt], pa.int16()),
            }, schema=SCHEMA))
            n = done + start + len(block)
            el = time.time() - t0
            print(f'  {n:,}/{len(corpus):,}  ({el:.0f}s, {el / max(start + len(block), 1) * len(todo):.0f}s projected)',
                  flush=True)
    writer.close()
    os.replace(tmp, out_path)
    print(f'\nwrote {out_path}')
    stats = summarize(out_path)
    json.dump(stats, open(os.path.join(SCRIPT_DIR, f'difficulty_{tag}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
