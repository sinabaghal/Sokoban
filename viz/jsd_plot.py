"""Tile-pattern JSD against sample size: generated puzzles vs real held-out.

The figure behind the distribution claim. Both series are measured the same
way against the same 450,000-puzzle training reference, so the real-data
series is the floor -- what a perfect generator would score at that sample
size. The model tracking that floor down is the result; a genuine
distributional error would show as the model's curve flattening while the
floor keeps falling.

Log-log, because the floor decays as a power law in sample size and the
overlap is only legible on those axes.

Data: viz/csv/jsd_sample_size.csv, written by metrics/jsd_curve.py. Nothing
here recomputes a divergence, so editing a label cannot move a number.

    python jsd_plot.py
"""

import csv
import os

import numpy as np

import labels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(SCRIPT_DIR, 'csv', 'jsd_sample_size.csv')
OUT = os.path.join(SCRIPT_DIR, 'out')

# Applied to dpi, not figsize, so the figure scales uniformly instead of
# leaving the text at its old point size. Matches the other plot scripts.
SCALE = 0.6
BASE_DPI = 130

T = labels.JSD_SAMPLE_SIZE

# Validated categorical slots 1 and 2, as everywhere else in this project.
THEME = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e',
                  grid='#e2e1dd', s1='#2a78d6', s2='#eb6834'),
    'dark':  dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7',
                  grid='#333331', s1='#3987e5', s2='#d95926'),
}


def load():
    k, floor, model = [], [], []
    for r in csv.DictReader(open(CSV)):
        k.append(int(r['k']))
        floor.append(float(r['floor_jsd']) if r['floor_jsd'] else np.nan)
        model.append(float(r['model_jsd']))
    return np.array(k), np.array(floor), np.array(model)


def draw(mode, k, floor, model):
    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=BASE_DPI * SCALE)
    fig.patch.set_facecolor(t['surface'])
    ax.set_facecolor(t['surface'])

    fin = ~np.isnan(floor)
    # floor first and heavier: it is the reference the other series is read against
    ax.plot(k[fin], floor[fin], color=t['s1'], lw=2.4, marker='o', ms=6,
            mfc=t['surface'], mew=2, zorder=3, label=T['floor'])
    ax.plot(k, model, color=t['s2'], lw=2, marker='s', ms=5,
            mfc=t['surface'], mew=2, zorder=4, ls='--', label=T['model'])

    ax.set_xscale('log')
    ax.set_yscale('log')

    # mark where held-out data runs out and the model curve continues alone
    kmax_floor = k[fin].max()
    ax.axvline(kmax_floor, color=t['grid'], lw=1.2, ls=':', zorder=1)
    ax.set_xlabel(T['x'], fontsize=11, color=t['ink2'])
    ax.set_ylabel(T['y'], fontsize=11, color=t['ink2'])
    ax.legend(loc='lower left', frameon=False, fontsize=10.5,
              labelcolor=t['ink'], handlelength=2.2)

    ax.grid(which='major', color=t['grid'], lw=0.8, zorder=0)
    ax.grid(which='minor', color=t['grid'], lw=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['grid'])
    ax.tick_params(colors=t['ink2'], labelsize=10.5, which='both')

    fig.subplots_adjust(left=0.115, right=0.975, top=0.965, bottom=0.145)
    suffix = '' if mode == 'light' else '_dark'
    for ext in ('png', 'pdf') if mode == 'light' else ('png',):
        p = os.path.join(OUT, f'jsd_sample_size{suffix}.{ext}')
        fig.savefig(p, facecolor=t['surface'])
        print(f'  -> {p}')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    k, floor, model = load()
    print(f'[jsd] {CSV}')
    print(f'{"K":>9}  {"floor":>8}  {"model":>8}  {"gap":>9}')
    for i in range(len(k)):
        f = '--' if np.isnan(floor[i]) else f'{floor[i]:.4f}'
        g = '--' if np.isnan(floor[i]) else f'{model[i]-floor[i]:+.4f}'
        print(f'{k[i]:>9,}  {f:>8}  {model[i]:>8.4f}  {g:>9}')
    for mode in ('light', 'dark'):
        draw(mode, k, floor, model)


if __name__ == '__main__':
    main()
