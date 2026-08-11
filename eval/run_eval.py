"""
Checkpoint evaluation pipeline (implements eval.md).

Inference-only. For every K-th checkpoint in an input folder:
  * generate N puzzles,
  * record the per-token probability trajectory across denoising iterations
    (+ the reveal step of each token) into 7 CSVs (one per tile type),
  * run the push-based solver and record solvability, push-move count, and
    search effort (states expanded) per puzzle,
  * write a one-row-per-checkpoint summary.

All outputs go under this folder (source/eval/), default source/eval/output/.
Shared model/solver code is IMPORTED from its canonical locations
(source/diffusion, source, source/metrics) rather than duplicated, so the model
architecture can never drift from the training code that produced the weights.

Usage (run from source/eval):
    python run_eval.py --checkpoint-dir ../diffusion/checkpoints_T100 \
        --num-timesteps 100 --every 5 --num 1000 --prob-samples 10
"""

import argparse
import csv
import functools
import glob
import math
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # source/eval
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)                         # source
DIFFUSION_DIR = os.path.join(SOURCE_DIR, 'diffusion')
METRICS_DIR = os.path.join(SOURCE_DIR, 'metrics')
sys.path.insert(0, DIFFUSION_DIR)
sys.path.insert(0, SOURCE_DIR)
sys.path.insert(0, METRICS_DIR)

from config import ModelConfig
from model import SokobanTransformer
from dataset import tensor_to_puzzle
from generate import validate_puzzle
from test_solver import boxoban_to_matrix, solve_push

# class index == token id (see CHAR_TO_ID): 0 '#', 1 ' ', 2 '@', 3 '$', 4 '.', 5 '*', 6 '+'
TILE_NAMES = ['wall', 'floor', 'player', 'box', 'goal', 'box_on_goal', 'player_on_goal']
MASK_ID = 7
GRID = 10
L = GRID * GRID


@torch.no_grad()
def sample_instrumented(model, device, num_timesteps, batch_size, temperature, record_probs):
    """Reverse diffusion with recording. Mirrors MaskedDiffusion.sample (random reveal).

    Returns:
        tokens        [B, L]            final generated grids (token ids)
        revealed_at   [B, L]            iteration each position was committed
        commit_class  [B, L]            token id committed at that position
        commit_prob   [B, L]            prob of the committed token, at commit time
        max_prob      [B, L]            prob of the model's TOP choice, at commit time
        prob_trace    [steps, R, L, 7]  softmax probs for the first R=record_probs rows
                                        (None if record_probs == 0)

    commit_prob vs max_prob separates two failure modes: if max_prob is high but
    commit_prob is low, the model's top choice was something else and the
    multinomial draw took the tail (a sampling slip, fixable with temperature);
    if both are low, the model was genuinely uncertain.
    """
    x = torch.full((batch_size, L), MASK_ID, device=device, dtype=torch.long)
    revealed_at = torch.full((batch_size, L), -1, device=device, dtype=torch.long)
    commit_class = torch.full((batch_size, L), -1, device=device, dtype=torch.long)
    commit_prob = torch.zeros(batch_size, L, device=device)
    max_prob = torch.zeros(batch_size, L, device=device)
    steps = min(num_timesteps, L)
    rp = min(record_probs, batch_size)
    prob_trace = [] if rp > 0 else None

    def step_probs(logits):
        return F.softmax(logits / temperature, dim=-1)

    for step in range(steps):
        t_val = num_timesteps - round(step * num_timesteps / steps)
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        probs = step_probs(model(x, t))                                  # [B, L, 7]
        if prob_trace is not None:
            prob_trace.append(probs[:rp].detach().cpu())

        masked = (x == MASK_ID)
        if not masked.any():
            break

        V = probs.shape[-1]
        pred = torch.multinomial(probs.view(-1, V), 1).view(batch_size, L)
        num_masked = masked.sum(dim=1)
        num_to_unmask = torch.clamp((num_masked.float() / (steps - step)).ceil().long(), min=1)
        num_to_unmask = torch.minimum(num_to_unmask, num_masked)

        priority = torch.rand(batch_size, L, device=device)
        priority[~masked] = -1.0
        rank = priority.argsort(dim=1, descending=True).argsort(dim=1)
        select = masked & (rank < num_to_unmask.unsqueeze(1))

        # `select` requires `masked`, so a selected cell is by definition not yet
        # committed -- record its commit-time stats here, once.
        p_sel = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)         # [B, L]
        p_max = probs.max(dim=-1).values                                 # [B, L]
        revealed_at = torch.where(select, torch.tensor(step, device=device), revealed_at)
        commit_class = torch.where(select, pred, commit_class)
        commit_prob = torch.where(select, p_sel, commit_prob)
        max_prob = torch.where(select, p_max, max_prob)
        x = torch.where(select, pred, x)

    # final pass: commit anything still masked
    masked = (x == MASK_ID)
    if masked.any():
        t = torch.ones(batch_size, device=device, dtype=torch.long)
        probs = step_probs(model(x, t))
        if prob_trace is not None:
            prob_trace.append(probs[:rp].detach().cpu())
        V = probs.shape[-1]
        pred = torch.multinomial(probs.view(-1, V), 1).view(batch_size, L)
        p_sel = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        p_max = probs.max(dim=-1).values
        revealed_at = torch.where(masked, torch.tensor(steps, device=device), revealed_at)
        commit_class = torch.where(masked, pred, commit_class)
        commit_prob = torch.where(masked, p_sel, commit_prob)
        max_prob = torch.where(masked, p_max, max_prob)
        x = torch.where(masked, pred, x)

    trace = torch.stack(prob_trace, dim=0) if prob_trace else None       # [steps(+1), R, L, 7]
    return x, revealed_at, commit_class, commit_prob, max_prob, trace


class ProbTraceWriter:
    """Streams per-iteration probability traces into ONE Parquet file (a row group per chunk).

    Replaces the old 7 CSVs. Measured on a 25-sample trace (250k rows):
        7x CSV                        37.5 MB   1.9s
        1x Parquet f32 + zstd          9.7 MB   0.2s
        1x Parquet f16 + zstd          1.5 MB   0.2s   <- default
    float16 costs at most ~1e-4 absolute error, which is finer than the old CSV's
    %.5f for small probabilities, so nothing analytically useful is lost.

    `revealed_at` is deliberately NOT stored here: it is a per-(sample, token)
    property that the old format repeated once per iteration. It lives in
    reveal.csv -- join on (sample_id, row_id, col_id).
    """

    COLS = ['p_' + n for n in TILE_NAMES]

    def __init__(self, path, float16=True):
        self.path = path
        self.dtype = np.float16 if float16 else np.float32
        self.writer = None
        # position columns are identical for every chunk of a given shape; cache them
        self._cache = {}

    def _positions(self, I, B):
        key = (I, B)
        if key not in self._cache:
            tok = np.arange(L)
            self._cache[key] = (
                np.tile(np.repeat(np.arange(I, dtype=np.int16), L), B),
                np.tile((tok // GRID).astype(np.int8), B * I),
                np.tile((tok % GRID).astype(np.int8), B * I),
            )
        return self._cache[key]

    def add(self, prob_trace, sample_offset):
        """prob_trace: [I, B, L, 7] CPU tensor. Rows are ordered (sample, iteration, token)."""
        if prob_trace is None:
            return
        tr = prob_trace.numpy()
        I, B = tr.shape[0], tr.shape[1]
        it_col, r_col, c_col = self._positions(I, B)
        s_col = np.repeat(np.arange(sample_offset, sample_offset + B, dtype=np.int32), I * L)
        probs = np.transpose(tr, (1, 0, 2, 3)).reshape(-1, 7).astype(self.dtype)

        data = {'sample_id': s_col, 'iteration': it_col, 'row_id': r_col, 'col_id': c_col}
        for k, col in enumerate(self.COLS):
            data[col] = probs[:, k]
        table = pa.table(data)

        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression='zstd')
        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _solve_worker(puzzle, max_states):
    """(valid, solvable, pushes, states_expanded, status) for one puzzle.

    status distinguishes three genuinely different outcomes:
      'solvable'       -- a solution was found
      'unsolvable'     -- PROVEN: the search exhausted the whole push-state space
      'unknown_capped' -- the solver hit max_states and gave up; solvability is UNKNOWN
      'invalid'        -- structurally malformed (bad piece counts / no player)

    solve_push's loop is `while q and states < max_states`, so a capped run ends with
    states == max_states while a proven-unsolvable run exits with the queue empty at
    states < max_states. That lets us tell them apart without touching the shared solver.
    """
    if not validate_puzzle(puzzle)['valid']:
        return (0, 0, -1, 0, 'invalid')
    matrix, player_pos = boxoban_to_matrix(puzzle)
    if player_pos is None:
        return (0, 0, -1, 0, 'invalid')
    sol, _, st = solve_push(matrix, player_pos, max_states=max_states, return_stats=True)
    states = st['states_expanded']
    if sol is not None:
        status = 'solvable'
    elif states >= max_states:
        status = 'unknown_capped'
    else:
        status = 'unsolvable'
    return (1, 1 if sol is not None else 0, st['pushes'], states, status)


# ---------------------------------------------------------------------------
# Wall-fix analysis (eval.md section 5)
# ---------------------------------------------------------------------------

def _is_solvable_grid(lines, max_states):
    puzzle = '\n'.join(''.join(row) if isinstance(row, list) else row for row in lines)
    matrix, player_pos = boxoban_to_matrix(puzzle)
    if player_pos is None:
        return False
    sol, _ = solve_push(matrix, player_pos, max_states=max_states)
    return sol is not None


def _solvable_without(lines, remove, max_states):
    """Is the puzzle solvable with the given wall cells flipped to floor?"""
    grid = [list(row) for row in lines]
    for (r, c) in remove:
        grid[r][c] = ' '
    return _is_solvable_grid(grid, max_states)


def _wall_candidates(lines):
    """Interior walls with >=1 non-wall neighbour.

    A wall whose four neighbours are all walls can never be reached by the player
    or a box, so removing it provably cannot change solvability -- safe to skip.
    (A reachability-based filter would be smaller but is NOT safe: reachability
    changes as boxes move, so it can miss genuine fixes.)
    """
    H, W = len(lines), len(lines[0])
    out = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if lines[r][c] != '#':
                continue
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and lines[nr][nc] != '#':
                    out.append((r, c))
                    break
    return out


def _wallfix_worker(args):
    """Classify one UNSOLVABLE puzzle. args = (puzzle, max_states). Returns (category, fixes).

    fixes is a list of tuples of removed wall positions:
      '1wall'             -> EVERY single wall whose removal fixes it (full enumeration)
      '2wall'             -> the first fixing pair found (pairs are expensive; one suffices)
      'really_unsolvable' -> [] (not fixable by removing 1 or 2 interior walls)
    """
    puzzle, max_states = args
    lines = puzzle.split('\n')
    cands = _wall_candidates(lines)

    # k=1: enumerate the FULL loop, collecting every hit (no early exit).
    singles = [w for w in cands if _solvable_without(lines, (w,), max_states)]
    if singles:
        return ('1wall', [(w,) for w in singles])

    # k=2: first hit only.
    for combo in combinations(cands, 2):
        if _solvable_without(lines, combo, max_states):
            return ('2wall', [combo])

    return ('really_unsolvable', [])


def write_reveal_csv(path, revealed_at, commit_class, commit_prob, max_prob):
    """Per-token commit record for ALL generated samples (N x 100 rows).

    Lean counterpart to the 7 full-trace prob CSVs: it carries only the commit
    moment, which is what the reveal-step and commit-confidence analyses need, so
    it can cover every sample instead of just --prob-samples of them.
    """
    ra = revealed_at.tolist()
    cc = commit_class.tolist()
    cp = commit_prob.tolist()
    mp = max_prob.tolist()
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'row_id', 'col_id', 'revealed_at',
                    'committed_class', 'commit_prob', 'max_prob'])
        for s in range(len(ra)):
            for tok in range(L):
                r, c = divmod(tok, GRID)
                w.writerow([s, r, c, ra[s][tok], cc[s][tok],
                            f'{cp[s][tok]:.5f}', f'{mp[s][tok]:.5f}'])


def evaluate_checkpoint(path, args, device, out_root, perf_writer):
    config = ModelConfig()
    config.num_timesteps = args.num_timesteps
    model = SokobanTransformer(config).to(device)
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    step = ck['step']
    print(f"step {step}: generating {args.num} samples (full recording for all)...", flush=True)

    ckp_dir = os.path.join(out_root, f'ckp_{step}')
    os.makedirs(ckp_dir, exist_ok=True)

    # PIPELINED: generate on the GPU in chunks and submit each chunk's puzzles to the
    # worker pool immediately, so CPU solving of chunk n overlaps GPU generation of
    # chunk n+1 (solving is ~85% of the wall-clock, generation ~15%, and they use
    # different hardware). Per-puzzle submit also gives dynamic scheduling, which
    # matters because solve cost is heavily skewed (p50 ~86ms, max ~4s).
    # The prob-logged batch goes first (it gets the full 7-class traces).
    # Everything is recorded for every one of the N samples -- no sub-sampling knobs.
    # Generation runs in --chunk sized batches (VRAM bound) and each chunk's traces are
    # written as one Parquet row group, with sample_ids offset to stay globally correct.
    solve_fn = functools.partial(_solve_worker, max_states=args.max_states)
    parts, puzzles, futs = [], [], []
    trace_writer = ProbTraceWriter(os.path.join(ckp_dir, 'prob_trace.parquet'),
                                   float16=not args.prob_f32)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        while done < args.num:
            b = min(args.chunk, args.num - done)
            x, ra, cc, cp, mp, trace = sample_instrumented(
                model, device, args.num_timesteps, b, args.temperature, record_probs=b)
            parts.append((x.cpu(), ra.cpu(), cc.cpu(), cp.cpu(), mp.cpu()))

            chunk_puzzles = [tensor_to_puzzle(t) for t in x.cpu()]
            puzzles.extend(chunk_puzzles)
            futs.extend(ex.submit(solve_fn, p) for p in chunk_puzzles)   # workers start now

            trace_writer.add(trace, sample_offset=done)
            done += b
            print(f"  generated {done}/{args.num} (solving in background)...", flush=True)

        trace_writer.close()

        tokens, revealed_at, commit_class, commit_prob, max_prob = (
            torch.cat([p[i] for p in parts], dim=0) for i in range(5))
        write_reveal_csv(os.path.join(ckp_dir, 'reveal.csv'),
                         revealed_at, commit_class, commit_prob, max_prob)

        print(f"  waiting on {len(futs)} solves...", flush=True)
        results = [f.result() for f in futs]

        with open(os.path.join(ckp_dir, 'solvability.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'valid', 'solvable', 'pushes', 'states_expanded', 'status'])
            for i, (v, s, p, st, status) in enumerate(results):
                w.writerow([i, v, s, p, st, status])

    # save the generated grids, labeled with solvability. Boxoban format, so it's
    # playable with source/UI/play.py, and the header sample_id joins solvability.csv.
    with open(os.path.join(ckp_dir, 'samples.txt'), 'w') as f:
        for i, (puzzle, res) in enumerate(zip(puzzles, results)):
            f.write(f"; {i} {res[4]}\n{puzzle}\n")   # status: solvable/unsolvable/unknown_capped/invalid

    n = len(results)
    solvable = sum(r[1] for r in results)
    capped = sum(1 for r in results if r[4] == 'unknown_capped')
    solved_pushes = sorted(r[2] for r in results if r[1])
    solved_states = sorted(r[3] for r in results if r[1])

    def med(v):
        return v[len(v) // 2] if v else -1

    # Wall-fix analysis (eval.md section 5). Runs on every proven-unsolvable puzzle by
    # default; --wallfix-sample N restricts it to a random N of them, which is much
    # faster but makes the reported rates estimates rather than exact.
    # Capped puzzles are excluded either way: one might actually be solvable, so "how
    # many walls must be removed to fix it" is not a meaningful question for it.
    wf_cols = ['', '', '']
    unsolv_ids = [i for i, r in enumerate(results) if r[4] == 'unsolvable']
    sampled = 0 < args.wallfix_sample < len(unsolv_ids)
    sel = (random.Random(args.wallfix_seed).sample(unsolv_ids, args.wallfix_sample)
           if sampled else unsolv_ids)
    if sel:
        scope = (f"a random {len(sel)} of {len(unsolv_ids)}" if sampled
                 else f"all {len(sel)}")
        print(f"  wall-fix analysis on {scope} proven-unsolvable "
              f"(the slowest phase; ~seconds each)...", flush=True)
        # chunksize=1: wall-fix cost is highly variable (p50 ~0.6s, tail >15s), so hand
        # work out one item at a time rather than in fixed blocks to avoid a straggler tail.
        with ProcessPoolExecutor(max_workers=args.workers) as ex2:
            wf = list(ex2.map(_wallfix_worker,
                              [(puzzles[i], args.max_states) for i in sel], chunksize=1))

        with open(os.path.join(ckp_dir, 'wallfix.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'category', 'w1_row', 'w1_col', 'w2_row', 'w2_col'])
            for sid, (cat, fixes) in zip(sel, wf):
                if not fixes:                      # really_unsolvable
                    w.writerow([sid, cat, -1, -1, -1, -1])
                    continue
                for fix in fixes:                  # one row per fixing solution
                    (r1, c1) = fix[0]
                    (r2, c2) = fix[1] if len(fix) > 1 else (-1, -1)
                    w.writerow([sid, cat, r1, c1, r2, c2])

        m = len(wf)
        cats = [c for c, _ in wf]
        wf_cols = [f'{100*cats.count(k)/m:.1f}'
                   for k in ('1wall', '2wall', 'really_unsolvable')]
        note = f"n={m}, ESTIMATE of {len(unsolv_ids)}" if sampled else f"n={m}, exact"
        print(f"    1wall {wf_cols[0]}% | 2wall {wf_cols[1]}% | "
              f"really_unsolvable {wf_cols[2]}%  ({note})", flush=True)

    perf_writer.writerow([step, n, f'{100*solvable/n:.1f}', med(solved_pushes),
                          f'{sum(solved_pushes)/len(solved_pushes):.1f}' if solved_pushes else -1,
                          med(solved_states),
                          f'{sum(solved_states)/len(solved_states):.0f}' if solved_states else -1,
                          f'{100*capped/n:.1f}', *wf_cols])
    print(f"  step {step}: solvable {solvable}/{n} ({100*solvable/n:.1f}%), "
          f"capped/unknown {capped} ({100*capped/n:.1f}%), "
          f"median pushes {med(solved_pushes)}, median states {med(solved_states)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Checkpoint evaluation pipeline (see eval.md)')
    parser.add_argument('--checkpoint-dir', required=True, help='Folder of step_*.pt checkpoints')
    parser.add_argument('--num-timesteps', type=int, required=True, help='Noise levels T the run was trained with')
    parser.add_argument('--every', type=int, default=1, help='Evaluate every K-th checkpoint (by step)')
    parser.add_argument('--num', type=int, default=1000, help='Puzzles generated per checkpoint')
    parser.add_argument('--prob-f32', action='store_true',
                        help='Store trace probabilities as float32 instead of float16 '
                             '(~6.5x larger; float16 error is <=1e-4, finer than the old CSV format)')
    parser.add_argument('--wallfix-sample', type=int, default=0,
                        help='Run the wall-fix analysis on a random N of the proven-unsolvable '
                             'puzzles instead of all of them. 0 (default) = all, exact but slow '
                             '(~25 min/checkpoint at --num 5000). 100 makes it ~15s but the reported '
                             'rates become estimates (+/-9pp at n=100). Everything else is still '
                             'recorded for all --num samples.')
    parser.add_argument('--wallfix-seed', type=int, default=0,
                        help='Seed for --wallfix-sample selection (reproducible subsets)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--chunk', type=int, default=2500, help='Generation batch size (VRAM bound)')
    parser.add_argument('--max-states', type=int, default=200000,
                        help='Push-solver state cap. Lowering it does not speed things up (puzzles '
                             'exhaust the search space rather than hit the cap); below ~50k it starts '
                             'reporting false unsolvables. Left at the safe default.')
    parser.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'output'))
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    paths = glob.glob(os.path.join(args.checkpoint_dir, 'step_*.pt'))
    steps = sorted((int(re.search(r'step_(\d+)\.pt', p).group(1)), p) for p in paths)
    selected = steps[::args.every]
    print(f"Device {device} | T={args.num_timesteps} | {len(selected)} checkpoints "
          f"(every {args.every} of {len(steps)}) | out={args.out}", flush=True)

    perf_path = os.path.join(args.out, 'perf.csv')
    done = set()
    if os.path.exists(perf_path):
        with open(perf_path) as f:
            for row in csv.reader(f):
                if row and row[0].isdigit():
                    done.add(int(row[0]))
    new_file = not os.path.exists(perf_path)
    with open(perf_path, 'a', newline='') as pf:
        pw = csv.writer(pf)
        if new_file:
            pw.writerow(['step', 'num', 'solvable_pct', 'median_pushes', 'mean_pushes',
                         'median_states', 'mean_states', 'capped_unknown_pct',
                         'wallfix_1wall_pct', 'wallfix_2wall_pct', 'wallfix_really_unsolvable_pct'])
        for step, path in selected:
            if step in done:
                continue
            evaluate_checkpoint(path, args, device, args.out, pw)
            pf.flush()

    print(f"Done. Summary: {perf_path}")


if __name__ == '__main__':
    main()
