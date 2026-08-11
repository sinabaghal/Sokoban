"""Solvability across training, exact and after a single-wall repair.

The figure the paper's Section 6 argues for: the pass/fail rate climbs steadily,
but the *effective* rate -- counting failures that one interior wall deletion
repairs -- climbs faster and further, and the gap between the two lines is the
depth of the model's remaining mistakes.

Data: viz/csv/solvability.csv, written by make_figure_data.py. Nothing here reads
the raw evaluation outputs, so editing a title cannot change a number and
re-rendering is instant. Regenerate the data with:

    python make_figure_data.py --only solvability

Writes light and dark versions (the README is dark, the paper is not) plus a PDF
for LaTeX inclusion.
"""

import csv
import os

import numpy as np

import labels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(SCRIPT_DIR, 'csv', 'solvability.csv')
OUT = os.path.join(SCRIPT_DIR, 'out')

# Output scale. Applied to dpi rather than figsize so the whole figure scales
# uniformly -- shrinking figsize instead would leave the text at its old point
# size and make it proportionally larger.
SCALE = 0.6
BASE_DPI = 130

# All drawn text lives in labels.py -- edit it there.
T = labels.SOLVABILITY

# Categorical slots 1 and 2 of the validated default palette. Both modes pass
# every check of the palette validator (worst CVD dE 24.7 light / 26.8 dark,
# well clear of the >=8 target).
THEME = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e',
                  grid='#e2e1dd', s1='#2a78d6', s2='#eb6834'),
    'dark':  dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7',
                  grid='#333331', s1='#3987e5', s2='#d95926'),
}


def load():
    rows = list(csv.DictReader(open(CSV)))
    return (np.array([int(r['step']) for r in rows]),
            np.array([float(r['solvable_pct']) for r in rows]) / 100,
            np.array([float(r['effective_pct']) for r in rows]) / 100,
            np.array([float(r['solvable_ci95']) for r in rows]) / 100,
            np.array([float(r['effective_ci95']) for r in rows]) / 100)


def draw(mode, step, s, eff, ci_s, ci_e):
    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=BASE_DPI * SCALE)
    fig.patch.set_facecolor(t['surface'])
    ax.set_facecolor(t['surface'])
    x = step / 1000

    # the gap between the curves IS the finding -- give it a light fill rather
    # than leaving the reader to measure it by eye, but keep it recessive enough
    # that it reads as a gap and not as a third mark
    ax.fill_between(x, 100 * s, 100 * eff, color=t['s2'], alpha=0.055, lw=0)

    handles = []
    for y, ci, col, lab in ((eff, ci_e, t['s2'], T['effective']),
                            (s, ci_s, t['s1'], T['exact'])):
        ax.fill_between(x, 100 * (y - ci), 100 * (y + ci), color=col, alpha=0.18, lw=0)
        ln, = ax.plot(x, 100 * y, color=col, lw=2, marker='o', ms=5.5,
                      mfc=t['surface'], mew=2, zorder=3, label=lab)
        handles.append(ln)

    # a plain legend instead of direct end-labels and a title block: this
    # figure is captioned where it is embedded, so repeating it here is noise
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=10.5,
              labelcolor=t['ink'], handlelength=1.8, borderaxespad=1.0)

    ax.set_xlabel(T['x'], fontsize=11, color=t['ink2'])
    ax.set_ylabel(T['y'], fontsize=11, color=t['ink2'])

    ax.set_ylim(35, 103)
    ax.set_xlim(-4, 262)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_xticks([0, 50, 100, 150, 200, 255])
    ax.grid(axis='y', color=t['grid'], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['grid'])
    ax.tick_params(colors=t['ink2'], labelsize=10.5)

    # no title block and no end-labels, so the axes can use the full canvas
    fig.subplots_adjust(left=0.095, right=0.975, top=0.965, bottom=0.135)
    suffix = '' if mode == 'light' else '_dark'
    for ext in ('png', 'pdf') if mode == 'light' else ('png',):
        p = os.path.join(OUT, f'solvability{suffix}.{ext}')
        fig.savefig(p, facecolor=t['surface'])
        print(f'  -> {p}')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    step, s, eff, ci_s, ci_e = load()
    print(f'[solvability] {CSV}')
    print('step      solvable        effective (<=1 wall)')
    for i in range(len(step)):
        print(f'{step[i]:>7,}   {100*s[i]:5.1f} +-{100*ci_s[i]:.1f}   '
              f'{100*eff[i]:5.1f} +-{100*ci_e[i]:.1f}')
    for mode in ('light', 'dark'):
        draw(mode, step, s, eff, ci_s, ci_e)


if __name__ == '__main__':
    main()
