"""Solvability and average wall count vs sampling temperature.

Data: source/eval/output/temperature_sweep_10k.csv, written by
source/eval/temperature_sweep.py (10,000 samples per temperature, T=100
baseline checkpoint step 290000, push-based solver). Nothing here re-solves
anything -- editing a title cannot change a number.

Two y-axes because the two series live on unrelated scales: solvability is a
percentage, wall count is a raw average. Same categorical order as every
other figure in this project -- slot 1 (blue) first, slot 2 (orange) second.

    python temperature_plot.py
    python temperature_plot.py --csv path/to/other_sweep.csv
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
DEFAULT_CSV = os.path.join(SCRIPT_DIR, 'csv', 'temperature_sweep.csv')
OUT = os.path.join(SCRIPT_DIR, 'out')

# Applied to dpi, not figsize, so the figure scales uniformly instead of
# leaving the text at its old point size. Matches solvability_plot.py.
SCALE = 0.6
BASE_DPI = 130

T = labels.TEMPERATURE_SWEEP

# Same validated palette every other figure in this project uses. Both modes
# pass every check of the palette validator.
THEME = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e',
                  grid='#e2e1dd', s1='#2a78d6', s2='#eb6834'),
    'dark':  dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7',
                  grid='#333331', s1='#3987e5', s2='#d95926'),
}


def load(path):
    rows = list(csv.DictReader(open(path)))
    return (np.array([float(r['temperature']) for r in rows]),
            np.array([float(r['solvable_pct']) for r in rows]),
            np.array([float(r['solvable_ci95']) for r in rows]),
            np.array([float(r['avg_walls']) for r in rows]),
            np.array([float(r['avg_walls_ci95']) for r in rows]),
            int(rows[0]['n']))


def draw(mode, tau, solvable, s_ci, walls, w_ci, n, out_path):
    t = THEME[mode]
    fig, ax1 = plt.subplots(figsize=(7.6, 4.3), dpi=BASE_DPI * SCALE)
    fig.patch.set_facecolor(t['surface'])
    ax1.set_facecolor(t['surface'])
    ax2 = ax1.twinx()

    # 95% CI: binomial for the solvable proportion, SE of the mean for walls.
    # Both are drawn as bands rather than caps -- at this figure size a capped
    # errorbar on every point turns into visual noise.
    ax1.fill_between(tau, solvable - s_ci, solvable + s_ci,
                     color=t['s1'], alpha=0.20, lw=0, zorder=2)
    ax2.fill_between(tau, walls - w_ci, walls + w_ci,
                     color=t['s2'], alpha=0.20, lw=0, zorder=2)

    l1, = ax1.plot(tau, solvable, color=t['s1'], lw=2, marker='o', ms=5.5,
                   mfc=t['surface'], mew=2, zorder=3, label=T['solvable'])
    l2, = ax2.plot(tau, walls, color=t['s2'], lw=2, marker='o', ms=5.5,
                   mfc=t['surface'], mew=2, zorder=3, label=T['walls'])

    # a legend, not direct end-labels -- the two series move together and
    # converge at the low-temperature end, where end-of-line labels collide
    leg = ax1.legend([l1, l2], [T['solvable'], T['walls']], loc='upper right',
                     frameon=False, fontsize=10.5, labelcolor=t['ink'],
                     handlelength=1.6, bbox_to_anchor=(1.0, 1.02))

    ax1.set_xlabel(T['x'], fontsize=11, color=t['ink2'])
    ax1.set_ylabel(T['y_solvable'], fontsize=11, color=t['s1'])
    ax2.set_ylabel(T['y_walls'], fontsize=11, color=t['s2'])

    ax1.set_xlim(tau.min() - 0.04, tau.max() + 0.04)
    ax1.set_xticks(tau)
    ax1.grid(axis='y', color=t['grid'], lw=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for ax in (ax1, ax2):
        for side in ('top',):
            ax.spines[side].set_visible(False)
    ax1.spines['left'].set_color(t['s1'])
    ax1.spines['bottom'].set_color(t['grid'])
    ax2.spines['right'].set_color(t['s2'])
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(axis='y', colors=t['s1'], labelsize=10.5)
    ax1.tick_params(axis='x', colors=t['ink2'], labelsize=10.5)
    ax2.tick_params(axis='y', colors=t['s2'], labelsize=10.5)

    # no title block, so the axes take the full canvas
    fig.subplots_adjust(left=0.11, right=0.89, top=0.965, bottom=0.14)
    suffix = '' if mode == 'light' else '_dark'
    p = out_path.format(suffix=suffix)
    fig.savefig(p, facecolor=t['surface'])
    print(f'  -> {p}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--out', default=os.path.join(OUT, 'temperature_sweep{suffix}.png'))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    tau, solvable, s_ci, walls, w_ci, n = load(args.csv)
    print(f'[temperature] {args.csv}')
    print(f'{"temperature":>11} {"solvable":>18} {"avg walls":>17}')
    for i in range(len(tau)):
        print(f'{tau[i]:>11.2f} {solvable[i]:>11.2f} +/-{s_ci[i]:<4.2f} '
              f'{walls[i]:>10.2f} +/-{w_ci[i]:.2f}')

    for mode in ('light', 'dark'):
        draw(mode, tau, solvable, s_ci, walls, w_ci, n, args.out)


if __name__ == '__main__':
    main()
