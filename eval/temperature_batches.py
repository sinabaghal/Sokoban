"""
Batch the temperature sweep into equal-sized chunks to put error bars on it.

Each temperature has 10,000 generated puzzles, already solved. Splitting them
into B independent batches and measuring each one separately gives an
*empirical* spread: the mean across batches is the point estimate, and the
standard error across batches is the uncertainty on it.

Batch size matters. 5,000 leaves only two batches, which cannot estimate a
variance at all; 1,000 leaves ten, which can. The choice does not change the
answer -- the batches are independent draws either way -- only whether the
spread is measurable.

As a check, the script also prints the analytic binomial interval,
sqrt(p(1-p)/n), for solvability. The two should agree closely: puzzles are
i.i.d. draws from the model, so batching recovers the same uncertainty the
closed form already predicts. Agreement is evidence that nothing in the
generation pipeline induces correlation between samples.

    python temperature_batches.py
    python temperature_batches.py --batch 2000
"""

import argparse
import csv
import math
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)
IN_CSV = os.path.join(SCRIPT_DIR, 'output', 'temperature_sweep_10k_puzzles.csv')
OUT_CSV = os.path.join(SOURCE_DIR, 'viz', 'csv', 'temperature_sweep.csv')


def mean(v):
    return sum(v) / len(v)


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=IN_CSV)
    ap.add_argument('--batch', type=int, default=1000)
    ap.add_argument('--out', default=OUT_CSV)
    args = ap.parse_args()

    per_tau = defaultdict(list)
    for r in csv.DictReader(open(args.inp)):
        per_tau[float(r['temperature'])].append(
            (r['solvable'] == 'True', int(r['walls'])))

    print(f'batch size {args.batch:,}\n')
    print(f'{"tau":>5} {"batches":>8} {"solvable %":>22} {"binomial +/-":>13} '
          f'{"avg walls":>18}')
    rows = []
    for tau in sorted(per_tau):
        data = per_tau[tau]
        n = len(data)
        nb = n // args.batch
        solv_b, wall_b = [], []
        for b in range(nb):
            chunk = data[b * args.batch:(b + 1) * args.batch]
            solv_b.append(100 * sum(1 for s, _ in chunk if s) / len(chunk))
            wall_b.append(mean([w for _, w in chunk]))

        # mean across batches, and the 95% CI on that mean
        s_mu, s_ci = mean(solv_b), 1.96 * stdev(solv_b) / math.sqrt(nb)
        w_mu, w_ci = mean(wall_b), 1.96 * stdev(wall_b) / math.sqrt(nb)

        # closed form, for comparison: the same quantity if draws are i.i.d.
        p = s_mu / 100
        binom = 1.96 * math.sqrt(p * (1 - p) / n) * 100
        # walls have no closed form, but the SE of the mean over all n puzzles
        # uses every sample rather than the nb batch means, so it is the tighter
        # estimate of the same thing
        w_direct = 1.96 * stdev([w for _, w in data]) / math.sqrt(n)

        print(f'{tau:>5.1f} {nb:>8} {s_mu:>15.2f} +/-{s_ci:<5.2f} {binom:>12.2f} '
              f'{w_mu:>12.2f} +/-{w_ci:.2f}')
        rows.append((tau, n, nb, s_mu, s_ci, binom, w_mu, w_ci, w_direct))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['temperature', 'n', 'n_batches', 'solvable_pct',
                    'solvable_ci95_batched', 'solvable_ci95',
                    'avg_walls', 'avg_walls_ci95_batched', 'avg_walls_ci95'])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f'{r[3]:.4f}', f'{r[4]:.4f}',
                        f'{r[5]:.4f}', f'{r[6]:.4f}', f'{r[7]:.4f}',
                        f'{r[8]:.4f}'])
    print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()
