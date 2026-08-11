"""Wall-fix animation replayed from RECORDED evaluation data -- no inference.

Everything this needs was already written by `eval/run_eval.py` and is what the
paper's numbers were computed from:

    reveal.csv       per cell: revealed_at, committed_class, commit_prob, max_prob
    wallfix.csv      per unsolvable level: every single wall whose removal fixes it
    solvability.csv  per level: solvable, pushes, states_expanded
    samples.txt      the grids

Because `reveal.csv` carries `revealed_at` and `committed_class` for every cell,
the entire denoising animation is recoverable exactly as it happened -- frame k
shows `committed_class` wherever `revealed_at < k`. Re-running the sampler would
produce *different* levels; this replays the ones that were actually measured.

What that buys, beyond fidelity: `commit_prob` is the probability the model
assigned to the tile it committed, at the moment it committed it. So the culprit
wall can be shown with the confidence the model had when it placed it -- and
against the confidence it had on that same level's other interior walls. The
answer is the finding: the model was markedly less sure about the cell that broke
the puzzle.

Border cells are excluded from the baseline. The outer ring is 100% wall,
committed at p = 1.0000, and including it would inflate the comparison.

Usage:
    python wallfix_replay.py                       # ckp_255000, 3x3
    python wallfix_replay.py --ckp ckp_205000 --rows 2 --cols 3
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SOURCE_DIR, 'diffusion'))

import playback
import labels
import render as R

T = labels.WALLFIX_REPLAY
from render import C, GRID, render_panel, frame_tokens
from render import write, write_split

L = GRID * GRID
ID_TO_CHAR = {0: '#', 1: ' ', 2: '@', 3: '$', 4: '.', 5: '*', 6: '+'}


def to_puzzle(tokens):
    """(100,) token ids -> Boxoban text, without going through torch."""
    return '\n'.join(''.join(ID_TO_CHAR[int(t)] for t in tokens[r * GRID:(r + 1) * GRID])
                     for r in range(GRID))


def load_reveal(path, wanted):
    """Per-sample (revealed_at, committed_class, commit_prob), for `wanted` ids only.

    reveal.csv is 500k rows; filtering while streaming keeps this to one pass and
    a few MB rather than loading the whole table.
    """
    out = {s: (np.zeros(L, np.int16), np.zeros(L, np.uint8), np.zeros(L, np.float32))
           for s in wanted}
    with open(path) as f:
        for row in csv.reader(f):
            if row[0] == 'sample_id':
                continue
            sid = int(row[0])
            if sid not in out:
                continue
            i = int(row[1]) * GRID + int(row[2])
            ra, cc, cp = out[sid]
            ra[i], cc[i], cp[i] = int(row[3]), int(row[4]), float(row[5])
    return out


def interior_wall_baseline(commit_class, commit_prob):
    """Median commit_prob over this level's own INTERIOR walls.

    A within-puzzle baseline controls for level difficulty automatically, which a
    corpus-wide average does not.
    """
    idx = [i for i in range(L)
           if commit_class[i] == 0 and 0 < i // GRID < GRID - 1 and 0 < i % GRID < GRID - 1]
    return float(np.median(commit_prob[idx])) if idx else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckp', default='ckp_255000')
    ap.add_argument('--out-root', default=os.path.join(SOURCE_DIR, 'eval', 'output'))
    ap.add_argument('--rows', type=int, default=3)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--cell', type=int, default=20)
    ap.add_argument('--ms', type=int, default=70)
    ap.add_argument('--move-ms', type=int, default=110)
    ap.add_argument('--move-stride', type=int, default=1)
    ap.add_argument('--hold-start', type=int, default=6)
    ap.add_argument('--hold-ms', type=int, default=2500)
    ap.add_argument('--pause-ms', type=int, default=1600)
    ap.add_argument('--blink', type=int, default=5)
    ap.add_argument('--blink-ms', type=int, default=190)
    ap.add_argument('--hold-fix', type=int, default=8)
    ap.add_argument('--split', action='store_true', default=True)
    ap.add_argument('--still-only', action='store_true',
                    help='write only wallfix_culprits.png, skip the animations')
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'out'))
    args = ap.parse_args()

    cdir = os.path.join(SCRIPT_DIR, 'csv')
    print(f'[replay] {cdir}  (recorded data, no inference)')

    rev = {}
    for r in csv.DictReader(open(os.path.join(cdir, 'wallfix_cells.csv'))):
        sid = int(r['sample_id'])
        if sid not in rev:
            rev[sid] = (np.zeros(L, np.int16), np.zeros(L, np.uint8),
                        np.zeros(L, np.float32))
        i = int(r['row']) * GRID + int(r['col'])
        rev[sid][0][i] = int(r['revealed_at'])
        rev[sid][1][i] = int(r['committed_class'])
        rev[sid][2][i] = float(r['commit_prob'])

    sel, culprit, cprob, base, nfix = [], [], [], [], []
    for r in csv.DictReader(open(os.path.join(cdir, 'wallfix_culprits.csv'))):
        sel.append(int(r['sample_id']))
        culprit.append(int(r['culprit_row']) * GRID + int(r['culprit_col']))
        cprob.append(float(r['culprit_prob']))
        base.append(float(r['other_walls_median']))
        nfix.append(int(r['n_fixing_walls']))
    n = len(sel)
    broken = [rev[s][1].copy() for s in sel]

    print(f'\n  {"sample":>7} {"culprit p":>10} {"other walls":>12} {"fixes":>6}')
    for i, s in enumerate(sel):
        print(f'  {s:>7} {cprob[i]:>10.3f} {base[i]:>12.3f} {nfix[i]:>6}')
    print(f'\n  median culprit p {np.median(cprob):.3f} vs '
          f'{np.median(base):.3f} on the same levels\' other interior walls')

    # ---- standalone still, no progress bar and its own title ----
    ST = labels.WALLFIX_STILL
    still = render_panel(
        broken,
        [[(ST['caption'].format(p=cprob[j]),
           C['mark'] if cprob[j] < 0.7 else C['text']),
          (ST['baseline'].format(b=base[j]), C['dim'])] for j in range(n)],
        ST['title'], ST['subtitle'], None, args.cell, args.rows, args.cols,
        marks=[(culprit[j], C['mark'], ST['chip'].format(p=cprob[j]))
               for j in range(n)])
    sp = os.path.join(args.out, 'wallfix_culprits.png')
    still.save(sp)
    print(f'  -> {sp}')
    if args.still_only:
        return

    steps = 100
    ks = list(range(0, steps)) + [steps]
    ra = np.stack([rev[s][0] for s in sel])
    cc = np.stack([rev[s][1] for s in sel])

    frames, durs = [], []
    title = T['gen_title']
    f0 = render_panel([frame_tokens(ra[j], cc[j], 0) for j in range(n)],
                      [None] * n, title, T['counter'].format(k=0, steps=steps), 0,
                      args.cell, args.rows, args.cols)
    frames += [f0] * args.hold_start
    durs += [args.ms] * args.hold_start

    for k in ks[1:]:
        final = k >= steps
        caps = ([[(T['unsolvable'], C['bad'])] for _ in range(n)]
                if final else [None] * n)
        frames.append(render_panel([frame_tokens(ra[j], cc[j], k) for j in range(n)],
                                   caps, title, T['counter'].format(k=k, steps=steps),
                                   k / steps, args.cell, args.rows, args.cols))
        durs.append(args.ms)
    durs[-1] = args.pause_ms

    # ---- the culprit wall, with the confidence the model committed it at ----
    for b in range(args.blink * 2):
        colour = C['mark'] if b % 2 == 0 else C['mark_dim']
        caps = [[(T['blink_caption'].format(p=cprob[j]),
                  C['mark'] if cprob[j] < 0.7 else C['text']),
                 (T['blink_baseline'].format(b=base[j]), C['dim'])] for j in range(n)]
        frames.append(render_panel(broken, caps, T['blink_title'],
                                   T['blink_counter'],
                                   1.0, args.cell, args.rows, args.cols,
                                   marks=[(culprit[j], colour,
                                           T['blink_chip'].format(p=cprob[j]))
                                          for j in range(n)]))
        durs.append(args.blink_ms)
    durs[-1] = args.pause_ms
    blink_end = len(frames) - 1

    fixed = []
    for j in range(n):
        g = broken[j].copy()
        g[culprit[j]] = 1
        fixed.append(g)

    puzzles = [to_puzzle(g) for g in fixed]
    plays = [playback.replay(fixed[j], playback.solution_moves(puzzles[j]) or '')
             for j in range(n)]
    caps = [[(T['fixed_caption'], C['good']),
             (T['fixed_alt'].format(n_fixes=nfix[j]) if nfix[j] > 1 else '', C['dim'])]
            for j in range(n)]
    for _ in range(args.hold_fix):
        frames.append(render_panel(fixed, caps, T['fixed_title'], T['fixed_counter'],
                                   1.0, args.cell, args.rows, args.cols,
                                   marks=[(culprit[j], C['mark_dim']) for j in range(n)]))
        durs.append(args.ms)

    longest = max(len(p) for p in plays)
    print(f'  playing back: {longest - 1} moves in the longest of {n}')
    for m in range(1, longest, args.move_stride):
        grids = [p[min(m, len(p) - 1)] for p in plays]
        done = sum(playback.solved(g) for g in grids)
        caps = [[(T['solved'], C['good'])] if playback.solved(g)
                else [(T['move'].format(m=min(m, len(plays[j]) - 1)), C['dim'])]
                for j, g in enumerate(grids)]
        frames.append(render_panel(grids, caps, T['solve_title'],
                                   T['solve_counter'].format(done=done, n=n), done / n,
                                   args.cell, args.rows, args.cols))
        durs.append(args.move_ms)
    durs[-1] = args.hold_ms

    stills = [(blink_end - 1, 'wallfix_marked.png'), (len(frames) - 1, 'wallfix_solved.png')]
    if args.split:
        write_split(frames, durs, blink_end, args.out,
                    ('wallfix_break.gif', 'wallfix_repair.gif'), args.hold_ms, stills)
    else:
        write(frames, durs, os.path.join(args.out, 'wallfix_panel.gif'), stills)


if __name__ == '__main__':
    main()
