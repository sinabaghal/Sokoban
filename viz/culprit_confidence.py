"""Commitment confidence: the walls that break a level vs the walls that don't.

Both groups are drawn from the SAME puzzles -- the 96 unsolvable levels that a
single interior-wall deletion repairs. A within-puzzle comparison controls for
how hard each level was to build; a corpus-wide average would not.

Interior cells only. The border ring is 36 of 100 cells, is 100% wall, and
commits at p = 1.0000 -- including it would drag the "innocent" group upward.

(Reveal step is also matched, 51 vs 49, but that is guaranteed rather than
observed: reveal order is uniformly random by construction, so it could not have
come out otherwise. Still printed to stdout as a sanity check; not worth a line
on the figure.)

Data: viz/csv/culprit_confidence.csv and culprit_summary.csv, written by
make_figure_data.py. Regenerate with:

    python make_figure_data.py --only culprit
"""

import argparse
import csv
import os

import numpy as np

import labels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
CSV_DIR = os.path.join(SCRIPT_DIR, 'csv')

# All drawn text lives in labels.py -- edit it there.
T = labels.CULPRIT

SURFACE, INK, INK2, GRIDC = '#fcfcfb', '#0b0b0b', '#52514e', '#e2e1dd'
S_OTHER, S_CULPRIT = '#2a78d6', '#eb6834'   # validated categorical slots 1 and 2
THRESH = 0.7


def load():
    hist = list(csv.DictReader(open(os.path.join(CSV_DIR, 'culprit_confidence.csv'))))
    summ = {r['group']: r for r in
            csv.DictReader(open(os.path.join(CSV_DIR, 'culprit_summary.csv')))}
    lo = np.array([float(r['bin_lo']) for r in hist])
    return (lo,
            np.array([float(r['culprit_pct']) for r in hist]),
            np.array([float(r['other_pct']) for r in hist]),
            summ['culprit'], summ['other_interior'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'out', 'culprit_confidence.png'))
    args = ap.parse_args()

    lo, hc, ho, sc, so = load()
    print(f'[culprit] {CSV_DIR}')
    for r in (sc, so):
        print(f"  {r['group']:>14} walls  n={int(r['n']):5d}  "
              f"median p {float(r['median_p']):.3f}  <{THRESH} {float(r['pct_below_0.7']):5.1f}%")
    bins = np.append(lo, 1.0)

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    w = 0.040
    centres = bins[:-1] + 0.05
    ax.bar(centres - w / 2 - 0.002, hc, width=w, color=S_CULPRIT, zorder=3)
    ax.bar(centres + w / 2 + 0.002, ho, width=w, color=S_OTHER, zorder=3)

    ax.axvline(THRESH, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.annotate(f'p = {THRESH}', (THRESH - 0.012, 62), rotation=90, fontsize=10.5,
                color=INK2, ha='right', va='top')

    # inline key -- identity carried by a chip beside text, never by text colour
    for i, (col, lab) in enumerate(((S_CULPRIT, T['culprit'].format(n=int(sc['n']))),
                                    (S_OTHER, T['other'].format(n=int(so['n']))))):
        y = 60 - i * 5.6
        ax.add_patch(plt.Rectangle((0.035, y), 0.028, 3.2, color=col, zorder=5))
        ax.text(0.072, y + 1.6, lab, fontsize=11, color=INK, va='center')

    ax.annotate(T['note'].format(a=float(sc['pct_below_0.7']), b=float(so['pct_below_0.7'])),
                (0.33, 40), fontsize=10.5, color=INK2, ha='center', linespacing=1.5)

    fig.text(0.075, 0.965, T['title'], fontsize=15, fontweight='bold', color=INK, va='top')
    fig.text(0.075, 0.908, T['subtitle'].format(n_levels=int(sc['n_levels'])),
             fontsize=10.5, color=INK2, va='top')

    ax.set_xlabel(T['x'], fontsize=11, color=INK2)
    ax.set_ylabel(T['y'], fontsize=11, color=INK2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 68)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.grid(axis='y', color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRIDC)
    ax.tick_params(colors=INK2, labelsize=10.5)

    fig.subplots_adjust(left=0.075, right=0.975, top=0.815, bottom=0.13)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f'  -> {args.out}')


if __name__ == '__main__':
    main()
