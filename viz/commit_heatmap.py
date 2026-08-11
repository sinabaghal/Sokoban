"""Per-cell commitment confidence, and how it changes as the grid fills in.

Reads the recorded evaluation run -- no inference. `reveal.csv` stores, for every
cell of every sample, the iteration it was committed at and the probability the
model assigned the tile it chose. Two questions follow:

  1. WHERE does the uncertainty sit?  -> the 3x3 heat maps
  2. WHEN does it resolve?            -> the curve underneath

The second tests the obvious prediction about masked diffusion: a cell committed
into an empty grid has almost no context, so it should be a coin flip; a cell
committed at the end is surrounded by 99 decided neighbours and should be nearly
forced. If that is right, confidence rises monotonically with reveal step.

Border cells are plotted separately, not folded in. The outer ring is 36 of 100
cells, is 100% wall, and commits at p = 1.0000 throughout -- averaging it into
the interior would flatten the very trend being measured.

Data: viz/csv/commit_cells.csv and commit_curve.csv, written by
make_figure_data.py. Regenerate with:

    python make_figure_data.py --only commit
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

import labels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
GRID, L = 10, 100
CSV_DIR = os.path.join(SCRIPT_DIR, 'csv')

# All drawn text lives in labels.py -- edit it there.
T = labels.HEATMAP 

# Sequential blue, steps 100->700 of the validated ramp: one hue, monotone
# light->dark. A rainbow here would invent ordering that the data does not have.
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
        '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
CMAP = LinearSegmentedColormap.from_list('conf', RAMP)

SURFACE, INK, INK2, GRIDC = '#fcfcfb', '#0b0b0b', '#52514e', '#e2e1dd'
S1, S2 = '#2a78d6', '#eb6834'          # validated categorical slots 1 and 2
GLYPH = {0: '#', 1: '', 2: '@', 3: '$', 4: '.', 5: '*', 6: '+'}


def load_cells():
    """Per-cell records for the panels, keyed by sample id."""
    per, solv = {}, {}
    for r in csv.DictReader(open(os.path.join(CSV_DIR, 'commit_cells.csv'))):
        sid = int(r['sample_id'])
        if sid not in per:
            per[sid] = (np.zeros(L, np.int16), np.zeros(L, np.uint8),
                        np.zeros(L, np.float32))
        i = int(r['row']) * GRID + int(r['col'])
        ra, cc, cp = per[sid]
        ra[i] = int(r['revealed_at'])
        cc[i] = int(r['committed_class'])
        cp[i] = float(r['commit_prob'])
        solv[sid] = r['solvable'] == '1'
    return per, solv


def load_curve():
    """Median commit probability by reveal step, per cell group."""
    out = {'interior': ([], []), 'border': ([], [])}
    for r in csv.DictReader(open(os.path.join(CSV_DIR, 'commit_curve.csv'))):
        out[r['group']][0].append(int(r['reveal_step']))
        out[r['group']][1].append(float(r['median_p']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'out', 'commit_heatmap.png'))
    args = ap.parse_args()

    per, sol = load_cells()
    curve = load_curve()
    sel = sorted(per)
    print(f'[heatmap] {CSV_DIR}  samples {sel}')

    fig = plt.figure(figsize=(9.6, 12.4), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1.15],
                  hspace=0.30, wspace=0.10, left=0.055, right=0.945,
                  top=0.855, bottom=0.055)

    fig.text(0.055, 0.985, T['title'], fontsize=16, fontweight='bold', color=INK, va='top')
    fig.text(0.055, 0.960, T['subtitle'], fontsize=12, color=INK2, va='top')
    fig.text(0.055, 0.936, T['cbar'], fontsize=10, color=INK2, va='top')

    for k, s in enumerate(sel):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ra, cc, cp = per[s]
        im = ax.imshow(cp.reshape(GRID, GRID), cmap=CMAP, vmin=0, vmax=1)
        for i in range(L):
            g = GLYPH[int(cc[i])]
            if g:
                ax.text(i % GRID, i // GRID, g, ha='center', va='center',
                        fontsize=7.5, family='monospace',
                        color='white' if cp[i] > 0.55 else '#2b2b2b')
        ok = sol[s]
        med = np.median(cp[[i for i in range(L)
                            if 0 < i // GRID < 9 and 0 < i % GRID < 9]])
        ax.set_title(T['panel'].format(sid=s, med=med,
                                      status='solvable' if ok else 'unsolvable'),
                     fontsize=9, color=INK if ok else '#b3453f', pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRIDC)

    cax = fig.add_axes([0.055, 0.900, 0.89, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=9.5, colors=INK2)
    cb.outline.set_edgecolor(GRIDC)

    # ---- when does confidence resolve? ----
    ax = fig.add_subplot(gs[3, :])
    ax.set_facecolor(SURFACE)
    # both curves converge on 1.0, so end-labels would collide -- label each
    # along its own path instead, where they are unambiguous
    for key, col, lab, lx, dy in (('border', S2, T['border'], 34, 0.035),
                                  ('interior', S1, T['interior'], 34, 0.045)):
        steps, med = curve[key]
        ax.plot(steps, med, color=col, lw=2, zorder=3)
        ax.plot([lx], [med[lx]], 'o', ms=7, color=col, mfc=SURFACE, mew=2, zorder=4)
        ax.annotate(lab, (lx + 2.5, med[lx] + dy), fontsize=12, color=INK,
                    va='bottom', ha='left')

    ax.set_xlabel(T['curve_x'], fontsize=11, color=INK2)
    ax.set_ylabel(T['curve_y'], fontsize=11, color=INK2)
    ax.set_title(T['curve_title'], fontsize=12, color=INK, loc='left', pad=8)
    ax.set_ylim(0, 1.12); ax.set_xlim(-2, 101)
    ax.grid(axis='y', color=GRIDC, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRIDC)
    ax.tick_params(colors=INK2, labelsize=10.5)

    fig.savefig(args.out, facecolor=SURFACE)
    print(f'  -> {args.out}')

    mi = curve['interior'][1]
    print(f'  interior, first 20 reveal steps : median p {np.median(mi[:20]):.3f}')
    print(f'  interior, last 20 reveal steps  : median p {np.median(mi[-20:]):.3f}')
    print(f"  border ring: median p {np.median(curve['border'][1]):.4f}")


if __name__ == '__main__':
    main()
