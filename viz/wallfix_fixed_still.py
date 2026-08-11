"""One still PNG: the same 9 unsolvable levels wallfix_replay.py uses, but shown
AFTER the single-wall repair -- the culprit cell already turned to floor and
ringed in red. No caption, no solve playback, just the fixed grids.

Reuses the same recorded data (viz/csv/wallfix_cells.csv, wallfix_culprits.csv)
as wallfix_replay.py -- no inference, no re-solving.

    python wallfix_fixed_still.py
"""

import csv
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SOURCE_DIR, 'diffusion'))

from render import C, GRID, render_panel

L = GRID * GRID
FLOOR_ID = 1
# render_panel's header height with no title and no bands (60*SS supersampled,
# which is 60px once render_panel downscales by SS) -- cropped off below since
# an empty title still reserves a blank header row otherwise
HEAD_PX = 60


def main():
    cdir = os.path.join(SCRIPT_DIR, 'csv')
    out = os.path.join(SCRIPT_DIR, 'out')
    os.makedirs(out, exist_ok=True)

    rev = {}
    for r in csv.DictReader(open(os.path.join(cdir, 'wallfix_cells.csv'))):
        sid = int(r['sample_id'])
        if sid not in rev:
            rev[sid] = np.zeros(L, np.uint8)
        rev[sid][int(r['row']) * GRID + int(r['col'])] = int(r['committed_class'])

    sel, culprit, cprob = [], [], []
    for r in csv.DictReader(open(os.path.join(cdir, 'wallfix_culprits.csv'))):
        sel.append(int(r['sample_id']))
        culprit.append(int(r['culprit_row']) * GRID + int(r['culprit_col']))
        cprob.append(float(r['culprit_prob']))
    n = len(sel)

    fixed = []
    for sid, cell in zip(sel, culprit):
        g = rev[sid].copy()
        g[cell] = FLOOR_ID
        fixed.append(g)

    # the chip carries the probability the model assigned that wall when it
    # committed it -- pinned to the cell rather than left to a caption the
    # reader would have to match up by position
    im = render_panel(fixed, [None] * n, '', '', None, 20, 3, 3,
                      marks=[(cell, C['mark'], f'p={p:.2f}')
                             for cell, p in zip(culprit, cprob)])
    im = im.crop((0, HEAD_PX, im.width, im.height))
    p = os.path.join(out, 'wallfix_fixed.png')
    im.save(p)
    print(f'-> {p}  {im.size}')


if __name__ == '__main__':
    main()
