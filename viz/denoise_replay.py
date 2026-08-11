"""Generation-panel GIF replayed from RECORDED evaluation data -- no inference.

Same idea as wallfix_replay.py: source/eval/run_eval.py already wrote everything
this needs when it evaluated ckp_255000 (the paper's reference checkpoint) --

    reveal.csv       per cell: revealed_at, committed_class, commit_prob
    solvability.csv  per puzzle: solvable, pushes, states_expanded, status
    samples.txt      the grids

Because reveal.csv carries revealed_at/committed_class for every cell, the
denoising animation is recoverable exactly as it happened -- frame k shows
committed_class wherever revealed_at < k -- without re-running the model.
This is what you want when the puzzles themselves don't need to change and
only the rendering does (layout, labels, etc.): it reuses the exact puzzles
already on record instead of sampling a fresh batch.

Reuses build_panel() from denoise_gif.py unchanged, so any rendering fix made
there (e.g. the compact no-title header) applies here automatically.

Usage:
    python denoise_replay.py                     # ckp_255000, 3x3, solvable only
    python denoise_replay.py --ckp ckp_205000 --rows 2 --cols 4
"""

import argparse
import csv
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SOURCE_DIR, 'diffusion'))

# denoise_gif (and therefore torch) must import before anything that pulls in
# PIL (render, playback) -- on Windows, importing Pillow before torch can wedge
# torch's own DLL loading (c10.dll) later in the same process.
from denoise_gif import build_panel
import playback
from render import GRID, write, write_split

L = GRID * GRID


def load_solvability(path):
    """sample_id -> (solved:bool, status:str, pushes:int, states:int), matching
    difficulty_train.solve_one's tuple format exactly (build_panel expects it)."""
    out = {}
    for r in csv.DictReader(open(path)):
        sid = int(r['sample_id'])
        solved = r['solvable'] == '1'
        out[sid] = (solved, r['status'], int(r['pushes']), int(r['states_expanded']))
    return out


def load_reveal(path, wanted):
    """Per selected sample: (revealed_at, committed_class) arrays, streamed from
    the (large) CSV in one pass rather than loading all 5000 samples' worth."""
    out = {s: (np.zeros(L, np.int16), np.zeros(L, np.uint8)) for s in wanted}
    with open(path) as f:
        for row in csv.reader(f):
            if row[0] == 'sample_id':
                continue
            sid = int(row[0])
            if sid not in out:
                continue
            i = int(row[1]) * GRID + int(row[2])
            ra, cc = out[sid]
            ra[i], cc[i] = int(row[3]), int(row[4])
    return out


def load_puzzles(path, wanted):
    """sample_id -> puzzle text, from the ';  <id>  <status>' blocks run_eval.py wrote."""
    out = {}
    lines = [l.rstrip('\n') for l in open(path)]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(';'):
            sid = int(line.split()[1])
            if sid in wanted:
                out[sid] = '\n'.join(lines[i + 1:i + 1 + GRID])
            i += 1 + GRID
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckp', default='ckp_255000')
    ap.add_argument('--eval-root', default=os.path.join(SOURCE_DIR, 'eval', 'output'))
    ap.add_argument('--rows', type=int, default=3)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--cell', type=int, default=20)
    ap.add_argument('--stride', type=int, default=1, help='reveal steps per frame')
    ap.add_argument('--ms', type=int, default=70, help='ms per denoising frame')
    ap.add_argument('--hold-start', type=int, default=6, help='frames held fully masked')
    ap.add_argument('--pause-ms', type=int, default=1600)
    ap.add_argument('--move-ms', type=int, default=110)
    ap.add_argument('--move-stride', type=int, default=1)
    ap.add_argument('--hold-ms', type=int, default=2500)
    ap.add_argument('--no-solve', action='store_true')
    ap.add_argument('--no-filter', action='store_true', help='do not require solvable')
    ap.add_argument('--split', action='store_true', default=True)
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'out'))
    args = ap.parse_args()

    cdir = os.path.join(args.eval_root, args.ckp)
    print(f'[replay] {cdir}  (recorded data, no inference)')

    solv = load_solvability(os.path.join(cdir, 'solvability.csv'))

    # spread the selection across the recorded pool's own difficulty range,
    # sorted by search effort, so the panel actually shows easy-to-hard variety
    # rather than whatever order the samples happened to be generated in
    n = args.rows * args.cols
    pool = [sid for sid, r in solv.items() if args.no_filter or r[0]]
    if len(pool) < n:
        sys.exit(f'need {n} puzzles, recorded pool yielded {len(pool)}; try a different --ckp')
    pool.sort(key=lambda sid: solv[sid][3])  # ascending states_expanded
    sel_ids = [pool[i] for i in np.linspace(0, len(pool) - 1, n).astype(int)]

    reveal_map = load_reveal(os.path.join(cdir, 'reveal.csv'), set(sel_ids))
    puzzle_map = load_puzzles(os.path.join(cdir, 'samples.txt'), set(sel_ids))

    # build_panel does numpy fancy-indexing (reveal[sel], commit[sel]), so
    # these have to be stacked arrays, not plain lists
    reveal = np.stack([reveal_map[sid][0] for sid in sel_ids])
    commit = np.stack([reveal_map[sid][1] for sid in sel_ids])
    tokens = commit  # after every reveal, committed_class IS the final grid
    puzzles = [puzzle_map[sid] for sid in sel_ids]
    results = [solv[sid] for sid in sel_ids]
    sel = list(range(n))
    steps = 100
    ks = list(range(0, steps, args.stride)) + [steps]

    print(f'  {n} puzzles from {len(pool)} solvable-and-recorded candidates, '
          f'states_expanded {solv[sel_ids[0]][3]}..{solv[sel_ids[-1]][3]}')

    frames, durs, cut = build_panel(sel, tokens, reveal, commit, results, None,
                                    puzzles, steps, ks, args)
    stills = [(cut, 'panel_levels.png'), (len(frames) - 1, 'panel_solved.png')]
    if args.split:
        write_split(frames, durs, cut, args.out,
                    ('generate_panel.gif', 'solve_panel.gif'), args.hold_ms, stills)
    else:
        write(frames, durs, os.path.join(args.out, 'denoise_panel.gif'), stills)


if __name__ == '__main__':
    main()
