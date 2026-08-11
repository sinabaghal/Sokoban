"""Extract every figure's data into viz/csv/ — run once, plot many times.

The heavy sources (reveal.csv is 500k rows, prob_trace.parquet is 271 MB) are read
here and nowhere else. Each figure script then reads only its own small CSV, so
re-rendering after a title or caption change is instant and cannot accidentally
change the numbers.

    python make_figure_data.py            # rebuild every CSV
    python make_figure_data.py --only solvability

Outputs (all in viz/csv/):
    solvability.csv          exact and effective solvability per checkpoint
    culprit_confidence.csv   commit-probability histogram, culprit vs other walls
    culprit_summary.csv      n / median / share below 0.7 for those two groups
    commit_curve.csv         median commit probability by reveal step
    commit_cells.csv         per-cell record for the nine heat-map levels
    wallfix_cells.csv        per-cell record for the nine wall-fix levels
    wallfix_culprits.csv     the culprit wall of each, and its confidence
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
CSV_DIR = os.path.join(SCRIPT_DIR, 'csv')
DEFAULT_CKP = 'ckp_255000'
THRESH = 0.7


def write(name, header, rows):
    os.makedirs(CSV_DIR, exist_ok=True)
    p = os.path.join(CSV_DIR, name)
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'  -> {p}  ({len(rows)} rows)')


def solvability():
    """Exact and effective solvability per checkpoint, with 95% intervals.

    effective = s + (1-s)*f1, so its interval propagates BOTH the solvability
    estimate (n=5,000) and the wall-fix share (n=100). The n=100 term dominates.
    """
    src = os.path.join(SOURCE_DIR, 'eval', 'output', 'perf.csv')
    rows = []
    for r in csv.DictReader(open(src)):
        n = int(r['num'])
        s = float(r['solvable_pct']) / 100
        f1 = float(r['wallfix_1wall_pct']) / 100
        eff = s + (1 - s) * f1
        var_s, var_f = s * (1 - s) / n, f1 * (1 - f1) / 100
        rows.append([r['step'], n, round(100 * s, 2), round(100 * f1, 1),
                     round(100 * eff, 2),
                     round(100 * 1.96 * np.sqrt(var_s), 2),
                     round(100 * 1.96 * np.sqrt((1 - f1) ** 2 * var_s
                                                + (1 - s) ** 2 * var_f), 2)])
    write('solvability.csv',
          ['step', 'n_samples', 'solvable_pct', 'wallfix_1wall_pct',
           'effective_pct', 'solvable_ci95', 'effective_ci95'], rows)


def _walls(ckp):
    """Interior wall commit records for the levels a single wall repairs."""
    d = os.path.join(SOURCE_DIR, 'eval', 'output', ckp)
    fixes = defaultdict(set)
    for r in csv.DictReader(open(os.path.join(d, 'wallfix.csv'))):
        if r['category'] == '1wall':
            fixes[int(r['sample_id'])].add(int(r['w1_row']) * 10 + int(r['w1_col']))
    cul, oth = [], []
    with open(os.path.join(d, 'reveal.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            sid, ri, ci = int(row[0]), int(row[1]), int(row[2])
            if sid not in fixes or int(row[4]) != 0:
                continue
            if not (0 < ri < 9 and 0 < ci < 9):        # interior only
                continue
            (cul if ri * 10 + ci in fixes[sid] else oth).append(float(row[5]))
    return np.array(cul), np.array(oth), len(fixes)


def culprit(ckp):
    cul, oth, npz = _walls(ckp)
    bins = np.linspace(0, 1, 11)
    hc = 100 * np.histogram(cul, bins=bins)[0] / len(cul)
    ho = 100 * np.histogram(oth, bins=bins)[0] / len(oth)
    write('culprit_confidence.csv',
          ['bin_lo', 'bin_hi', 'culprit_pct', 'other_pct'],
          [[round(bins[i], 1), round(bins[i + 1], 1), round(hc[i], 3), round(ho[i], 3)]
           for i in range(10)])
    write('culprit_summary.csv',
          ['group', 'n', 'median_p', 'pct_below_0.7', 'n_levels'],
          [['culprit', len(cul), round(float(np.median(cul)), 4),
            round(100 * float(np.mean(cul < THRESH)), 1), npz],
           ['other_interior', len(oth), round(float(np.median(oth)), 4),
            round(100 * float(np.mean(oth < THRESH)), 1), npz]])


def commit(ckp, samples):
    """Per-cell records for the nine panels, plus the reveal-step curve."""
    d = os.path.join(SOURCE_DIR, 'eval', 'output', ckp)
    sol = {int(r['sample_id']): r
           for r in csv.DictReader(open(os.path.join(d, 'solvability.csv')))}
    want = set(samples)
    cells = []
    agg = {'interior': defaultdict(list), 'border': defaultdict(list)}
    with open(os.path.join(d, 'reveal.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            sid, ri, ci = int(row[0]), int(row[1]), int(row[2])
            step, cls, p = int(row[3]), int(row[4]), float(row[5])
            edge = ri in (0, 9) or ci in (0, 9)
            agg['border' if edge else 'interior'][step].append(p)
            if sid in want:
                cells.append([sid, ri, ci, step, cls, round(p, 5),
                              int(sol[sid]['solvable'])])
    write('commit_cells.csv',
          ['sample_id', 'row', 'col', 'revealed_at', 'committed_class',
           'commit_prob', 'solvable'], cells)

    rows = []
    for grp in ('interior', 'border'):
        for step in sorted(agg[grp]):
            v = agg[grp][step]
            rows.append([grp, step, len(v), round(float(np.median(v)), 5)])
    write('commit_curve.csv', ['group', 'reveal_step', 'n', 'median_p'], rows)


def wallfix(ckp, n_panels=9):
    """The nine wall-fix levels: every cell, plus which wall breaks each one.

    Lets wallfix_replay.py run without touching reveal.csv, so the animation can
    be rebuilt from a few hundred rows instead of half a million.
    """
    d = os.path.join(SOURCE_DIR, 'eval', 'output', ckp)
    fixes = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(d, 'wallfix.csv'))):
        if r['category'] == '1wall':
            fixes[int(r['sample_id'])].append(int(r['w1_row']) * 10 + int(r['w1_col']))
    keys = sorted(fixes)
    sel = [keys[i] for i in np.linspace(0, len(keys) - 1, n_panels).astype(int)]

    per = {s: {} for s in sel}
    with open(os.path.join(d, 'reveal.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            sid = int(row[0])
            if sid in per:
                per[sid][int(row[1]) * 10 + int(row[2])] = (
                    int(row[3]), int(row[4]), float(row[5]))

    cells, culp = [], []
    for sid in sel:
        for i in range(100):
            ra, cc, cp = per[sid][i]
            cells.append([sid, i // 10, i % 10, ra, cc, round(cp, 5)])
        w = fixes[sid][0]
        walls = [per[sid][i][2] for i in range(100)
                 if per[sid][i][1] == 0 and 0 < i // 10 < 9 and 0 < i % 10 < 9]
        culp.append([sid, w // 10, w % 10, round(per[sid][w][2], 5),
                     round(float(np.median(walls)), 5), len(fixes[sid])])

    write('wallfix_cells.csv',
          ['sample_id', 'row', 'col', 'revealed_at', 'committed_class', 'commit_prob'],
          cells)
    write('wallfix_culprits.csv',
          ['sample_id', 'culprit_row', 'culprit_col', 'culprit_prob',
           'other_walls_median', 'n_fixing_walls'], culp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckp', default=DEFAULT_CKP)
    ap.add_argument('--samples', type=int, nargs='*',
                    default=list(np.linspace(0, 200, 9).astype(int)))
    ap.add_argument('--only', choices=['solvability', 'culprit', 'commit', 'wallfix'])
    args = ap.parse_args()

    print(f'[figure data] {args.ckp} -> {CSV_DIR}')
    if args.only in (None, 'solvability'):
        solvability()
    if args.only in (None, 'culprit'):
        culprit(args.ckp)
    if args.only in (None, 'commit'):
        commit(args.ckp, args.samples)
    if args.only in (None, 'wallfix'):
        wallfix(args.ckp)


if __name__ == '__main__':
    main()
