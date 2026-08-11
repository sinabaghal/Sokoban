"""Animate the reverse diffusion process: masked grid -> finished Sokoban puzzle.

Produces the README animations for the paper. Two outputs:

  --mode panel   an NxM board of puzzles denoising in lockstep (the headline GIF)
  --mode hero    one puzzle, large, with the per-commit confidence readout

Both are driven by the SAME sampler the evaluation uses, so the animation shows
the real generation process rather than a reconstruction of it. Reveal order is
uniformly random by construction (`strategy='random'`), so the picture arrives as
a random dissolve -- the model chooses *what* goes in each cell, never *which*
cell comes next. Saying otherwise would misread the animation.

Puzzles are solved before rendering and, by default, only solvable ones are
shown, with their push count captioned. Use --no-filter to draw an unfiltered
sample instead.

Usage:
    python denoise_gif.py                          # 3x3 panel, default checkpoint
    python denoise_gif.py --mode hero --seed 7
    python denoise_gif.py --rows 2 --cols 4 --cell 22 --stride 2
"""

import argparse
import functools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
S_DIR = os.path.dirname(SOURCE_DIR)
sys.path.insert(0, os.path.join(SOURCE_DIR, 'diffusion'))
sys.path.insert(0, os.path.join(SOURCE_DIR, 'metrics'))
sys.path.insert(0, os.path.join(S_DIR, 'compare'))
sys.path.insert(0, os.path.join(SOURCE_DIR, 'eval'))

from config import ModelConfig, TrainConfig
from model import SokobanTransformer
from dataset import tensor_to_puzzle
from difficulty_train import solve_one
from identity import load_puzzles, nearest_neighbors

import labels
import playback
import render as R
from render import (C, SS, GRID, font, frame_tokens, progress_bar, render_panel,
                    write, write_split)

L = GRID * GRID
# All drawn text lives in labels.py -- edit it there.
LBL_GEN, LBL_HERO, LBL_WALL = labels.GENERATE, labels.HERO, labels.WALLFIX
DEFAULT_CKPT = os.path.join(SOURCE_DIR, 'diffusion', 'checkpoints_T100', 'step_290000.pt')


@torch.no_grad()
def sample_traced(model, device, num_timesteps, batch_size, temperature):
    """Reverse diffusion, recording when each cell was committed and to what.

    Mirrors MaskedDiffusion.sample (random reveal) and eval/run_eval.py's
    instrumented copy. Every intermediate frame is recoverable from
    (revealed_at, commit_class) alone -- frame k shows commit_class wherever
    revealed_at < k and [MASK] elsewhere -- so no per-step grid is stored.

    Returns tokens, revealed_at, commit_class, commit_prob, max_prob  (all [B, L]).
    """
    x = torch.full((batch_size, L), R.MASK_ID, device=device, dtype=torch.long)
    revealed_at = torch.full((batch_size, L), -1, device=device, dtype=torch.long)
    commit_class = torch.full((batch_size, L), -1, device=device, dtype=torch.long)
    commit_prob = torch.zeros(batch_size, L, device=device)
    max_prob = torch.zeros(batch_size, L, device=device)
    steps = min(num_timesteps, L)

    for step in range(steps):
        t_val = num_timesteps - round(step * num_timesteps / steps)
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        probs = F.softmax(model(x, t) / temperature, dim=-1)

        masked = (x == R.MASK_ID)
        if not masked.any():
            break

        V = probs.shape[-1]
        pred = torch.multinomial(probs.view(-1, V), 1).view(batch_size, L)
        num_masked = masked.sum(dim=1)
        n_un = torch.clamp((num_masked.float() / (steps - step)).ceil().long(), min=1)
        n_un = torch.minimum(n_un, num_masked)

        priority = torch.rand(batch_size, L, device=device)
        priority[~masked] = -1.0
        rank = priority.argsort(dim=1, descending=True).argsort(dim=1)
        select = masked & (rank < n_un.unsqueeze(1))

        p_sel = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        revealed_at = torch.where(select, torch.tensor(step, device=device), revealed_at)
        commit_class = torch.where(select, pred, commit_class)
        commit_prob = torch.where(select, p_sel, commit_prob)
        max_prob = torch.where(select, probs.max(dim=-1).values, max_prob)
        x = torch.where(select, pred, x)

    masked = (x == R.MASK_ID)
    if masked.any():
        t = torch.ones(batch_size, device=device, dtype=torch.long)
        probs = F.softmax(model(x, t) / temperature, dim=-1)
        V = probs.shape[-1]
        pred = torch.multinomial(probs.view(-1, V), 1).view(batch_size, L)
        p_sel = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        revealed_at = torch.where(masked, torch.tensor(steps, device=device), revealed_at)
        commit_class = torch.where(masked, pred, commit_class)
        commit_prob = torch.where(masked, p_sel, commit_prob)
        max_prob = torch.where(masked, probs.max(dim=-1).values, max_prob)
        x = torch.where(masked, pred, x)

    return (x.cpu().numpy(), revealed_at.cpu().numpy(), commit_class.cpu().numpy(),
            commit_prob.cpu().numpy(), max_prob.cpu().numpy())


def nn_distance(tokens, train_path, device):
    """Nearest-neighbour Hamming distance from each sample to the whole training set.

    Reuses compare/identity.py rather than reimplementing: one-hot to 700 dims so
    the inner product counts agreeing cells exactly, distance = 100 - agreement.
    0 would mean the model reproduced a training level verbatim.
    """
    train = load_puzzles(train_path)
    agree, _ = nearest_neighbors(tokens.astype(np.uint8), train, device)
    return 100 - agree.astype(np.int32), len(train)


def render_hero(grid, title, counter, frac, lines, confidence, cell):
    """One frame of the single-puzzle animation.

    `lines` is a list of (text, colour) drawn under the grid; `confidence`, if
    given, adds the commit-probability readout used during the denoise phase.
    """
    s, pad = cell * SS, 18 * SS
    gw = GRID * s
    head, foot = 56 * SS, 46 * SS
    W = 2 * pad + gw
    H = head + gw + foot

    im = Image.new('RGB', (W, H), C['page'])
    d = ImageDraw.Draw(im)
    f_title = font(16 * SS, bold=True)
    f_small = font(11 * SS)
    f_mono = font(11 * SS, mono=True)

    d.text((pad, pad - 4 * SS), title, font=f_title, fill=C['text'])
    d.text((W - pad - d.textlength(counter, font=f_mono), pad - 1 * SS),
           counter, font=f_mono, fill=C['dim'])
    progress_bar(d, pad, pad + 22 * SS, W - 2 * pad, 5 * SS, frac)

    d.rectangle([pad - 2 * SS, head - 2 * SS, pad + gw + 2 * SS, head + gw + 2 * SS],
                fill=C['panel'], outline=C['edge'], width=SS)
    R.draw_grid(d, grid, pad, head, s)

    y = head + gw + 12 * SS
    for j, (text, colour) in enumerate(lines):
        d.text((pad, y + j * 15 * SS), text, font=f_small, fill=colour)

    if confidence is not None:
        d.text((pad, y), LBL_HERO['confidence'], font=f_small, fill=C['dim'])
        w = d.textlength(LBL_HERO['confidence'], font=f_small)
        d.text((pad + w + 5 * SS, y), f'{confidence:.2f}', font=f_mono,
               fill=C['good'] if confidence > 0.7 else C['bad'])
        bx = pad + w + 46 * SS
        progress_bar(d, bx, y + 4 * SS, W - pad - bx, 5 * SS, confidence)

    return im.resize((W // SS, H // SS), R.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['panel', 'hero', 'both', 'wallfix'], default='both')
    ap.add_argument('--blink', type=int, default=5, help='blink cycles on the culprit wall')
    ap.add_argument('--blink-ms', type=int, default=190, help='ms per blink half-cycle')
    ap.add_argument('--hold-fix', type=int, default=8,
                    help='frames held on the repaired level before it plays')
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--num-timesteps', type=int, default=100,
                    help='must match training; checkpoints do not store it')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--rows', type=int, default=3)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--cell', type=int, default=20)
    ap.add_argument('--hero-cell', type=int, default=38)
    ap.add_argument('--pool', type=int, default=64, help='candidates to sample from')
    ap.add_argument('--stride', type=int, default=1, help='reveal steps per frame')
    ap.add_argument('--ms', type=int, default=70, help='ms per denoising frame')
    ap.add_argument('--hold-start', type=int, default=6, help='frames held fully masked')
    ap.add_argument('--hold-ms', type=int, default=2500, help='pause on the last frame')
    ap.add_argument('--pause-ms', type=int, default=1600,
                    help='beat on the finished levels before the solution plays')
    ap.add_argument('--move-ms', type=int, default=110, help='ms per solution move')
    ap.add_argument('--move-stride', type=int, default=1, help='moves per solve frame')
    ap.add_argument('--split', action='store_true',
                    help='write each animation as two GIFs, cut at its phase boundary')
    ap.add_argument('--no-solve', action='store_true',
                    help='stop at the finished level, no solution playback')
    ap.add_argument('--no-filter', action='store_true', help='do not require solvable')
    ap.add_argument('--no-hamming', action='store_true',
                    help='skip the nearest-training-level distance (saves loading 450k puzzles)')
    ap.add_argument('--train-path', default=TrainConfig.data_path)
    ap.add_argument('--max-states', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'out'))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    cfg = ModelConfig()
    cfg.num_timesteps = args.num_timesteps
    model = SokobanTransformer(cfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f"[viz] {os.path.basename(args.checkpoint)}  step {ck['step']:,}  "
          f"T={args.num_timesteps}  temp={args.temperature}  device={device}")

    t0 = time.time()
    tokens, reveal, commit, prob, _pmax = sample_traced(
        model, device, args.num_timesteps, args.pool, args.temperature)
    steps = min(args.num_timesteps, L)
    print(f"  sampled {args.pool} candidates in {time.time()-t0:.1f}s")

    puzzles = [tensor_to_puzzle(torch.from_numpy(t.astype(np.int64))) for t in tokens]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(functools.partial(solve_one, max_states=200000),
                              puzzles, chunksize=4))
    ok = [i for i, r in enumerate(results) if r[0]]
    print(f"  solvable {len(ok)}/{args.pool} ({100*len(ok)/args.pool:.0f}%)")

    nn = None
    if not args.no_hamming:
        t0 = time.time()
        nn, n_train = nn_distance(tokens, args.train_path, device)
        print(f"  nearest training level over {n_train:,} puzzles "
              f"({time.time()-t0:.0f}s): min {nn.min()} cells, median {np.median(nn):.0f}, "
              f"{int((nn == 0).sum())} exact reproduction(s)")

    pick = list(range(args.pool)) if args.no_filter else (ok or list(range(args.pool)))
    if not args.no_filter and not ok:
        print('  ! no solvable candidate in the pool -- showing unfiltered')

    ks = list(range(0, steps, args.stride)) + [steps]

    if args.mode == 'wallfix':
        n = args.rows * args.cols
        bad = [i for i, r in enumerate(results) if not r[0] and r[1] == 'unsolvable']
        print(f"  {len(bad)} proven-unsolvable; searching for single-wall fixes...")
        t0 = time.time()
        fixes = find_wallfixes(bad, puzzles, args.max_states)
        print(f"  {len(fixes)}/{len(bad)} are 1-wall fixable ({time.time()-t0:.0f}s)")
        if len(fixes) < n:
            sys.exit(f'need {n} one-wall-fixable levels, found {len(fixes)}; raise --pool')
        keys = sorted(fixes)
        sel = [keys[i] for i in np.linspace(0, len(keys) - 1, n).astype(int)]
        frames, durs, cut, blink_end = build_wallfix(sel, fixes, tokens, reveal,
                                                     commit, puzzles, steps, ks, args)
        stills = [(cut, 'wallfix_sampled_broken.png'),
                  (cut + 1, 'wallfix_sampled_marked.png'),
                  (len(frames) - 1, 'wallfix_sampled_solved.png')]
        if args.split:
            write_split(frames, durs, blink_end, args.out,
                        ('wallfix_sampled_break.gif', 'wallfix_sampled_repair.gif'),
                        args.hold_ms, stills)
        else:
            write(frames, durs, os.path.join(args.out, 'wallfix_sampled_panel.gif'), stills)
        return

    if args.mode in ('panel', 'both'):
        n = args.rows * args.cols
        if len(pick) < n:
            sys.exit(f'need {n} puzzles, pool yielded {len(pick)}; raise --pool')
        # spread the selection over the pool rather than taking the first n, so
        # the board is not biased toward whatever the low indices happen to be
        sel = [pick[i] for i in np.linspace(0, len(pick) - 1, n).astype(int)]
        frames, durs, cut = build_panel(sel, tokens, reveal, commit, results, nn,
                                        puzzles, steps, ks, args)
        stills = [(cut, 'panel_levels.png'), (len(frames) - 1, 'panel_solved.png')]
        if args.split:
            write_split(frames, durs, cut, args.out,
                        ('generate_panel.gif', 'solve_panel.gif'),
                        args.hold_ms, stills)
        else:
            write(frames, durs, os.path.join(args.out, 'denoise_panel.gif'), stills)

    if args.mode in ('hero', 'both'):
        frames, durs, cut = build_hero(pick[0], tokens, reveal, commit, prob, results,
                                       nn, puzzles, steps, ks, args)
        write(frames, durs, os.path.join(args.out, 'denoise_hero.gif'),
              [(cut, 'hero_level.png'), (len(frames) - 1, 'hero_solved.png')])


def build_panel(sel, tokens, reveal, commit, results, nn, puzzles, steps, ks, args):
    """Denoise frames, then solve frames, for the multi-puzzle animation."""
    rv, cm = reveal[sel], commit[sel]
    n = len(sel)
    pushes = [results[i][2] for i in sel]
    effort = [results[i][3] for i in sel]
    bands = [R.difficulty_band(e) for e in effort]

    def caps(show_verdict):
        out = []
        for j, i in enumerate(sel):
            if not show_verdict:
                out.append(None)
                continue
            ok = results[i][0]
            row = [(LBL_GEN['solvable'].format(pushes=pushes[j]) if ok else LBL_GEN['unsolvable'],
                    C['good'] if ok else C['bad'])]
            if ok:
                row.append((LBL_GEN['effort'].format(effort=effort[j]), C['dim']))
            elif nn is not None:
                row.append((LBL_GEN['nn'].format(nn=nn[i]), C['dim']))
            out.append(row)
        return out

    # No title on the generation frames -- the counter rides the difficulty-key
    # row instead (see render_panel's `compact` mode), saving one header row.
    frames, durs = [], []

    f0 = render_panel([frame_tokens(rv[j], cm[j], 0) for j in range(n)], caps(False),
                      '', LBL_GEN['counter'].format(k=0, steps=steps), 0,
                      args.cell, args.rows, args.cols, bands=bands)
    frames += [f0] * args.hold_start
    durs += [args.ms] * args.hold_start

    for k in ks[1:]:
        frames.append(render_panel([frame_tokens(rv[j], cm[j], k) for j in range(n)],
                                   caps(k >= steps), '',
                                   LBL_GEN['counter'].format(k=k, steps=steps), k / steps,
                                   args.cell, args.rows, args.cols, bands=bands))
        durs.append(args.ms)
    durs[-1] = args.pause_ms  # beat on the finished levels before play starts
    cut = len(frames) - 1     # last denoise frame: the levels as generated

    if args.no_solve:
        durs[-1] = args.hold_ms
        return frames, durs, cut

    # ---- solve phase: replay each level's solution, all boards in parallel ----
    plays = [playback.replay(tokens[i], playback.solution_moves(puzzles[i]) or '')
             for i in sel]
    longest = max(len(p) for p in plays)
    print(f"  solving: {longest - 1} moves in the longest of {n} solutions")

    for m in range(1, longest, args.move_stride):
        grids = [p[min(m, len(p) - 1)] for p in plays]
        done = sum(playback.solved(g) for g in grids)
        rows = []
        for j, g in enumerate(grids):
            if playback.solved(g):
                rows.append([(LBL_GEN['solved'].format(pushes=pushes[j]), C['good']),
                             (LBL_GEN['effort'].format(effort=effort[j]), C['dim'])])
            else:
                rows.append([(LBL_GEN['move'].format(m=min(m, len(plays[j]) - 1)), C['dim'])])
        frames.append(render_panel(grids, rows, '',
                                   LBL_GEN['solve_counter'].format(done=done, n=n), done / n,
                                   args.cell, args.rows, args.cols, bands=bands))
        durs.append(args.move_ms)

    durs[-1] = args.hold_ms
    return frames, durs, cut


def find_wallfixes(indices, puzzles, max_states, workers=None):
    """For each unsolvable level, every single interior wall whose removal fixes it.

    Reuses eval/run_eval.py's search rather than reimplementing: it enumerates
    interior walls with at least one non-wall neighbour (a wall boxed in by four
    walls is unreachable, so removing it provably cannot matter) and tries each.
    Returns {index: [cell_index, ...]} for the levels that are 1-wall fixable.
    """
    from run_eval import _wallfix_worker

    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        cats = list(ex.map(_wallfix_worker,
                           [(puzzles[i], max_states) for i in indices], chunksize=1))
    for i, (cat, fixes) in zip(indices, cats):
        if cat == '1wall':
            out[i] = [w[0][0] * GRID + w[0][1] for w in fixes]
    return out


def build_wallfix(sel, fixes, tokens, reveal, commit, puzzles, steps, ks, args):
    """Denoise to an UNSOLVABLE level, point at the culprit wall, delete it, solve.

    This is the paper's wall-fix result as an animation: the model's failures are
    overwhelmingly one cell away from correct, and removing that one cell yields a
    level that plays to completion.

    Caveat carried in the caption text, not just the docs: 'removing wall X fixes
    it' establishes that X PARTICIPATES in the deadlock, not that committing X was
    the model's mistake -- the real error may have been a box placed later.
    """
    n = len(sel)
    rv, cm = reveal[sel], commit[sel]
    frames, durs = [], []

    def blank():
        return [None] * n

    title_gen = LBL_WALL['gen_title']
    f0 = render_panel([frame_tokens(rv[j], cm[j], 0) for j in range(n)], blank(),
                      title_gen, LBL_WALL['counter'].format(k=0, steps=steps), 0,
                      args.cell, args.rows, args.cols)
    frames += [f0] * args.hold_start
    durs += [args.ms] * args.hold_start

    for k in ks[1:]:
        final = k >= steps
        caps = [[(LBL_WALL['unsolvable'], C['bad'])] for _ in range(n)] if final else blank()
        frames.append(render_panel([frame_tokens(rv[j], cm[j], k) for j in range(n)],
                                   caps, title_gen,
                                   LBL_WALL['counter'].format(k=k, steps=steps), k / steps,
                                   args.cell, args.rows, args.cols))
        durs.append(args.ms)
    durs[-1] = args.pause_ms
    cut = len(frames) - 1

    broken = [tokens[i].copy() for i in sel]
    culprit = [fixes[i][0] for i in sel]
    n_fixes = [len(fixes[i]) for i in sel]

    # ---- blink the culprit wall ----
    for b in range(args.blink * 2):
        colour = C['mark'] if b % 2 == 0 else C['mark_dim']
        caps = [[(LBL_WALL['blink_caption'], C['mark'])] for _ in range(n)]
        frames.append(render_panel(broken, caps, LBL_WALL['blink_title'],
                                   LBL_WALL['blink_counter'], 1.0,
                                   args.cell, args.rows, args.cols,
                                   marks=[(culprit[j], colour) for j in range(n)]))
        durs.append(args.blink_ms)

    # ---- delete it ----
    fixed = []
    for j in range(n):
        g = broken[j].copy()
        g[culprit[j]] = 1  # wall -> floor
        fixed.append(g)
    blink_end = len(frames) - 1     # split point: everything after this is repair
    caps = [[(LBL_WALL['fixed_caption'], C['good']),
             (LBL_WALL['fixed_alt'].format(n_fixes=n_fixes[j]) if n_fixes[j] > 1 else '',
              C['dim'])]
            for j in range(n)]
    for _ in range(args.hold_fix):
        frames.append(render_panel(fixed, caps, LBL_WALL['fixed_title'],
                                   LBL_WALL['fixed_counter'], 1.0,
                                   args.cell, args.rows, args.cols,
                                   marks=[(culprit[j], C['mark_dim']) for j in range(n)]))
        durs.append(args.ms)
    durs[-1] = args.pause_ms

    if args.no_solve:
        durs[-1] = args.hold_ms
        return frames, durs, cut, blink_end

    # ---- and play the repaired levels ----
    repaired = [tensor_to_puzzle(torch.from_numpy(g.astype(np.int64))) for g in fixed]
    rres = [solve_one(p, max_states=args.max_states) for p in repaired]
    pushes = [r[2] for r in rres]
    effort = [r[3] for r in rres]
    bands = [R.difficulty_band(e) for e in effort]
    plays = [playback.replay(fixed[j], playback.solution_moves(repaired[j]) or '')
             for j in range(n)]
    longest = max(len(p) for p in plays)
    print(f"  solving repaired: {longest - 1} moves in the longest of {n}")

    for m in range(1, longest, args.move_stride):
        grids = [p[min(m, len(p) - 1)] for p in plays]
        done = sum(playback.solved(g) for g in grids)
        caps = []
        for j, g in enumerate(grids):
            if playback.solved(g):
                caps.append([(LBL_WALL['solved'].format(pushes=pushes[j]), C['good']),
                             (LBL_WALL['effort'].format(effort=effort[j]), C['dim'])])
            else:
                caps.append([(LBL_WALL['move'].format(m=min(m, len(plays[j]) - 1)), C['dim'])])
        frames.append(render_panel(grids, caps, LBL_WALL['solve_title'],
                                   LBL_WALL['solve_counter'].format(done=done, n=n), done / n,
                                   args.cell, args.rows, args.cols, bands=bands))
        durs.append(args.move_ms)

    durs[-1] = args.hold_ms
    return frames, durs, cut, blink_end


def build_hero(i, tokens, reveal, commit, prob, results, nn, puzzles, steps, ks, args):
    """Denoise frames, then solve frames, for the single-level animation."""
    ok, _st, pushes, states = results[i]
    rv, cm, pr = reveal[i], commit[i], prob[i]
    frames, durs = [], []

    def conf(k):
        cells = np.flatnonzero(rv == k - 1)
        return float(pr[cells].mean()) if len(cells) else None

    f0 = render_hero(frame_tokens(rv, cm, 0), LBL_HERO['title'],
                     LBL_HERO['counter'].format(k=0, steps=steps), 0, [], None,
                     args.hero_cell)
    frames += [f0] * args.hold_start
    durs += [args.ms] * args.hold_start

    verdict = [((LBL_HERO['solvable'].format(pushes=pushes, states=states)
                 if ok else LBL_HERO['unsolvable']), C['good'] if ok else C['bad'])]
    if nn is not None:
        verdict.append((LBL_HERO['nn'].format(nn=int(nn[i])), C['dim']))

    for k in ks[1:]:
        final = k >= steps
        frames.append(render_hero(frame_tokens(rv, cm, k),
                                  LBL_HERO['title'], LBL_HERO['counter'].format(k=k, steps=steps),
                                  k / steps, verdict if final else [],
                                  None if final else conf(k), args.hero_cell))
        durs.append(args.ms)
    durs[-1] = args.pause_ms
    cut = len(frames) - 1

    if args.no_solve or not ok:
        durs[-1] = args.hold_ms
        return frames, durs, cut

    moves = playback.solution_moves(puzzles[i]) or ''
    play = playback.replay(tokens[i], moves)
    for m in range(1, len(play), args.move_stride):
        g = play[m]
        fin = playback.solved(g)
        lines = [(LBL_HERO['move'].format(m=m, total=len(play) - 1, pushes=pushes), C['dim'])]
        if fin:
            lines = [(LBL_HERO['solved'], C['good'])]
        frames.append(render_hero(g, LBL_HERO['solve_title'],
                                  f'{m:>3}/{len(play)-1}', m / (len(play) - 1),
                                  lines, None, args.hero_cell))
        durs.append(args.move_ms)

    durs[-1] = args.hold_ms
    return frames, durs, cut


if __name__ == '__main__':
    main()
