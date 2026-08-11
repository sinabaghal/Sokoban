"""
Compare a generated tile-pattern distribution against the training
distribution via KL-divergence and Jensen-Shannon divergence.

With only ~1000 generated puzzles vs. ~450,000 training puzzles, the
generated set's realized pattern support is a small fraction of the
training set's (e.g. 14,624 vs. 264,981 unique 3x3 patterns) -- so most
training patterns simply never appear in the generated sample. Naive
KL(P_train || Q_generated) would then involve log(P/0) = infinity.

Fix: additive smoothing over the union of both realized supports, with a
small epsilon (default 1e-4) chosen to be negligible relative to either
distribution's real total window count -- just enough to avoid log(0),
not to meaningfully alter the real mass. Jensen-Shannon divergence is
reported unsmoothed alongside, since it's symmetric and always finite
(the mixture M = 0.5(P+Q) has support wherever either input does).

Usage:
    python kl_divergence.py
    python kl_divergence.py --tile-sizes 3 4 5
    python kl_divergence.py --q distributions/generated_n3.json --tile-sizes 3
"""

import argparse
import json
import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIST_DIR = os.path.join(SCRIPT_DIR, 'distributions')


def load_counts(path: str) -> dict:
    with open(path) as f:
        return json.load(f)['counts']


def smoothed_distributions(p_counts: dict, q_counts: dict, epsilon: float):
    """Additive smoothing over the union of both supports -> two aligned prob distributions."""
    support = set(p_counts) | set(q_counts)
    p_total = sum(p_counts.values()) + epsilon * len(support)
    q_total = sum(q_counts.values()) + epsilon * len(support)

    p = {x: (p_counts.get(x, 0) + epsilon) / p_total for x in support}
    q = {x: (q_counts.get(x, 0) + epsilon) / q_total for x in support}
    return p, q


def kl_divergence(p: dict, q: dict) -> float:
    return sum(p[x] * math.log2(p[x] / q[x]) for x in p if p[x] > 0)


def js_divergence(p_counts: dict, q_counts: dict) -> float:
    """No smoothing needed -- the mixture M always has support wherever P or Q does."""
    support = set(p_counts) | set(q_counts)
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())

    p = {x: p_counts.get(x, 0) / p_total for x in support}
    q = {x: q_counts.get(x, 0) / q_total for x in support}
    m = {x: 0.5 * (p[x] + q[x]) for x in support}

    def kl(a, b):
        return sum(a[x] * math.log2(a[x] / b[x]) for x in a if a[x] > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare(p_path: str, q_path: str, tile_size: int, epsilon: float):
    p_counts = load_counts(p_path)
    q_counts = load_counts(q_path)

    p_smooth, q_smooth = smoothed_distributions(p_counts, q_counts, epsilon)
    kl_pq = kl_divergence(p_smooth, q_smooth)
    kl_qp = kl_divergence(q_smooth, p_smooth)
    jsd = js_divergence(p_counts, q_counts)

    only_in_q = set(q_counts) - set(p_counts)
    frac_novel_windows = sum(q_counts[x] for x in only_in_q) / sum(q_counts.values())

    print(f"\n{'=' * 60}")
    print(f"N = {tile_size}x{tile_size}")
    print(f"{'=' * 60}")
    print(f"  P (training):  {p_path}  [{len(p_counts):,} unique patterns]")
    print(f"  Q (generated): {q_path}  [{len(q_counts):,} unique patterns]")
    print(f"  KL(P || Q)  [smoothed, eps={epsilon}]: {kl_pq:.4f} bits")
    print(f"  KL(Q || P)  [smoothed, eps={epsilon}]: {kl_qp:.4f} bits")
    print(f"  Jensen-Shannon divergence:              {jsd:.4f} bits (max {math.log2(2):.4f})")
    print(f"  Generated patterns never seen in training: {len(only_in_q):,}/{len(q_counts):,} unique "
          f"({100 * len(only_in_q) / len(q_counts):.1f}%), "
          f"{100 * frac_novel_windows:.2f}% of generated windows")

    return {'tile_size': tile_size, 'kl_pq': kl_pq, 'kl_qp': kl_qp, 'jsd': jsd}


def main():
    parser = argparse.ArgumentParser(
        description='KL / Jensen-Shannon divergence between training and generated tile-pattern distributions')
    parser.add_argument('--tile-sizes', '-n', type=int, nargs='+', default=[3, 4, 5],
                         help='Tile sizes to compare (default: 3 4 5)')
    parser.add_argument('--p-dir', type=str, default=DEFAULT_DIST_DIR,
                         help='Directory containing tile_pattern_n{N}.json (training)')
    parser.add_argument('--q-dir', type=str, default=DEFAULT_DIST_DIR,
                         help='Directory containing {q-prefix}_n{N}.json (generated)')
    parser.add_argument('--q-prefix', type=str, default='generated',
                         help='Filename prefix for the generated distribution (default: generated -> generated_n{N}.json)')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                         help='Additive smoothing constant for KL (default: 1e-4)')
    args = parser.parse_args()

    results = []
    for n in args.tile_sizes:
        p_path = os.path.join(args.p_dir, f'tile_pattern_n{n}.json')
        q_path = os.path.join(args.q_dir, f'{args.q_prefix}_n{n}.json')

        if not os.path.exists(p_path):
            print(f"Skipping N={n}: training distribution not found at {p_path}")
            continue
        if not os.path.exists(q_path):
            print(f"Skipping N={n}: generated distribution not found at {q_path}")
            continue

        results.append(compare(p_path, q_path, n, args.epsilon))

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        for r in results:
            print(f"  N={r['tile_size']}: KL(P||Q)={r['kl_pq']:.4f}  KL(Q||P)={r['kl_qp']:.4f}  "
                  f"JSD={r['jsd']:.4f} bits")


if __name__ == '__main__':
    main()
