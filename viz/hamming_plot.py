"""Nearest-neighbour Hamming distance to the training corpus.

The memorisation figure. Both series are measured identically against the same
450,000-puzzle training set, so the held-out series is the reference: whatever
a genuine puzzle drawn from the true distribution scores is what a generator
that memorised nothing should also score. Two curves sitting on top of each
other is the result; generated mass piled up near zero would be memorisation.

Data: viz/csv/hamming.csv, written by metrics/hamming_distances.py.

    python hamming_plot.py
"""

import csv
import os

import numpy as np

import labels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(SCRIPT_DIR, 'csv', 'hamming.csv')
OUT = os.path.join(SCRIPT_DIR, 'out')

# Applied to dpi, not figsize, so the figure scales uniformly. Matches the
# other plot scripts.
SCALE = 0.6
BASE_DPI = 130

T = labels.HAMMING

THEME = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e',
                  grid='#e2e1dd', s1='#2a78d6', s2='#eb6834'),
    'dark':  dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7',
                  grid='#333331', s1='#3987e5', s2='#d95926'),
}


def load():
    d, gen, held = [], [], []
    for r in csv.DictReader(open(CSV)):
        d.append(int(r['distance']))
        gen.append(float(r['generated_pct']))
        held.append(float(r['held_out_pct']))
    return np.array(d), np.array(gen), np.array(held)


def draw(mode, d, gen, held):
    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=BASE_DPI * SCALE)
    fig.patch.set_facecolor(t['surface'])
    ax.set_facecolor(t['surface'])

    # held-out first and heavier: it is the reference the other is read against
    ax.plot(d, held, color=t['s1'], lw=2.4, marker='o', ms=4.5,
            mfc=t['surface'], mew=1.6, zorder=3, label=T['held_out'])
    ax.plot(d, gen, color=t['s2'], lw=2, marker='s', ms=4, ls='--',
            mfc=t['surface'], mew=1.6, zorder=4, label=T['generated'])

    ax.set_xlabel(T['x'], fontsize=11, color=t['ink2'])
    ax.set_ylabel(T['y'], fontsize=11, color=t['ink2'])
    ax.legend(loc='upper right', frameon=False, fontsize=10.5,
              labelcolor=t['ink'], handlelength=2.2)

    ax.set_xlim(0, d.max())
    ax.grid(axis='y', color=t['grid'], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['grid'])
    ax.tick_params(colors=t['ink2'], labelsize=10.5)

    fig.subplots_adjust(left=0.1, right=0.975, top=0.965, bottom=0.14)
    suffix = '' if mode == 'light' else '_dark'
    for ext in ('png', 'pdf') if mode == 'light' else ('png',):
        p = os.path.join(OUT, f'hamming{suffix}.{ext}')
        fig.savefig(p, facecolor=t['surface'])
        print(f'  -> {p}')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    d, gen, held = load()
    print(f'[hamming] {CSV}')
    print(f'{"dist":>5} {"generated":>11} {"held-out":>10}')
    for i in range(len(d)):
        if gen[i] or held[i]:
            print(f'{d[i]:>5} {gen[i]:>10.3f}% {held[i]:>9.3f}%')
    for mode in ('light', 'dark'):
        draw(mode, d, gen, held)


if __name__ == '__main__':
    main()
