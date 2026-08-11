"""Animated commitment-confidence heat map: the same nine grids, filling in.

The static figure shows WHERE the model was unsure. This shows WHEN -- each cell
appears at the step it was actually committed, shaded by the probability the
model gave it at that moment, with the newest cell ringed so the order is
followable across nine panels at once.

What it makes visible, and the reason it beats the static version: the grids
start pale and darken. Early cells are committed into an almost-empty board at
roughly coin-flip confidence; late cells land among 99 decided neighbours and are
nearly forced. Reveal order itself is uniformly random -- the model picks what
goes in a cell, never which cell comes next -- so any structure you see is in the
colour, not the sequence.

Data: viz/csv/commit_cells.csv (make_figure_data.py). No inference.
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPT_DIR)

import labels
from render import save_gif, font, LANCZOS

T = labels.HEATMAP_GIF
from commit_heatmap import RAMP, GLYPH, load_cells, GRID, L

SS = 2
SURFACE = (252, 252, 251)
INK, INK2, GRIDC = (11, 11, 11), (82, 81, 78), (226, 225, 221)
EMPTY = (240, 240, 237)
NEW = (235, 104, 52)          # ring on the cell committed this step


def ramp_rgb(p):
    """Interpolate the sequential blue ramp at p in [0, 1]."""
    xs = np.linspace(0, 1, len(RAMP))
    cols = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in RAMP])
    return tuple(int(np.interp(p, xs, cols[:, k])) for k in range(3))


LUT = [ramp_rgb(i / 255) for i in range(256)]


def frame(per, sel, sol, k, cell, rows, cols, glyphs):
    s, pad, gap, cap = cell * SS, 16 * SS, 14 * SS, 25 * SS
    head = 86 * SS
    gw = GRID * s
    W = 2 * pad + cols * gw + (cols - 1) * gap
    H = head + rows * (gw + cap) + (rows - 1) * gap + pad

    im = Image.new('RGB', (W, H), SURFACE)
    d = ImageDraw.Draw(im)
    f_t = font(17 * SS, bold=True)
    f_s = font(12 * SS)
    f_m = font(11 * SS, mono=True)

    d.text((pad, pad - 4 * SS), T['title'], font=f_t, fill=INK)
    lbl = T['counter'].format(k=k)
    d.text((W - pad - d.textlength(lbl, font=f_m), pad - 1 * SS), lbl, font=f_m, fill=INK2)

    # colour key
    bx, by, bw, bh = pad, pad + 22 * SS, W - 2 * pad, 7 * SS
    for i in range(int(bw)):
        d.rectangle([bx + i, by, bx + i + 1, by + bh], fill=LUT[int(255 * i / bw)])
    d.rectangle([bx, by, bx + bw, by + bh], outline=GRIDC, width=SS)
    d.text((bx, by + bh + 3 * SS), T['key_low'], font=f_s, fill=INK2)
    t1 = T['key_high']
    d.text((bx + bw - d.textlength(t1, font=f_s), by + bh + 3 * SS), t1, font=f_s, fill=INK2)

    for j, sid in enumerate(sel):
        r, c = divmod(j, cols)
        ox = pad + c * (gw + gap)
        oy = head + r * (gw + cap + gap)
        ra, cc, cp = per[sid]

        for i in range(L):
            y, x = divmod(i, GRID)
            box = [ox + x * s, oy + y * s,
                   ox + (x + 1) * s - 1 - SS, oy + (y + 1) * s - 1 - SS]
            if ra[i] < k:
                d.rectangle(box, fill=LUT[int(255 * min(max(cp[i], 0.0), 1.0))])
                g = glyphs[j][i]
                if g:
                    d.text((box[0] + s / 2, box[1] + s / 2), g, font=f_m,
                           fill=(255, 255, 255) if cp[i] > 0.55 else (43, 43, 43),
                           anchor='mm')
            else:
                d.rectangle(box, fill=EMPTY)
        d.rectangle([ox, oy, ox + gw - 1, oy + gw - 1], outline=GRIDC, width=SS)

        # ring the cell committed on the step that produced this frame
        new = np.flatnonzero(ra == k - 1)
        for i in new:
            y, x = divmod(int(i), GRID)
            d.rectangle([ox + x * s, oy + y * s, ox + (x + 1) * s - 1, oy + (y + 1) * s - 1],
                        outline=NEW, width=max(1, int(s * 0.13)))

        done = ra < k
        if k >= 100:
            ok = sol[sid]['solvable'] == '1'
            txt = T['caption_final'].format(
                sid=sid, status=T['solvable'] if ok else T['unsolvable'])
            d.text((ox, oy + gw + 4 * SS), txt, font=f_s,
                   fill=INK if ok else (179, 69, 63))
        elif len(new):
            d.text((ox, oy + gw + 4 * SS),
                   T['caption_step'].format(sid=sid, p=cp[new[0]]),
                   font=f_s, fill=INK2)
        else:
            d.text((ox, oy + gw + 4 * SS), T['caption_idle'].format(sid=sid),
                   font=f_s, fill=INK2)

    return im.resize((W // SS, H // SS), LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckp', default='ckp_255000')
    ap.add_argument('--samples', type=int, nargs='*', default=None)
    ap.add_argument('--rows', type=int, default=3)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--cell', type=int, default=20)
    ap.add_argument('--ms', type=int, default=80)
    ap.add_argument('--hold-ms', type=int, default=3000)
    ap.add_argument('--hold-start', type=int, default=5)
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'out', 'commit_heatmap.gif'))
    args = ap.parse_args()

    per, solv = load_cells()
    sel = sorted(per)
    sol = {k: {'solvable': '1' if v else '0'} for k, v in solv.items()}
    print(f'[heatmap gif] csv/commit_cells.csv  samples {sel}')
    glyphs = [[GLYPH[int(t)] for t in per[s][1]] for s in sel]

    frames = [frame(per, sel, sol, 0, args.cell, args.rows, args.cols, glyphs)] \
        * args.hold_start
    durs = [args.ms] * args.hold_start
    for k in range(1, 101):
        frames.append(frame(per, sel, sol, k, args.cell, args.rows, args.cols, glyphs))
        durs.append(args.ms)
    durs[-1] = args.hold_ms

    size = save_gif(frames, args.out, durs)
    print(f'  -> {args.out}  ({len(frames)} frames, {frames[0].size[0]}x'
          f'{frames[0].size[1]}, {size/1e6:.2f} MB)')

    still = os.path.splitext(args.out)[0] + '_mid.png'
    frames[len(frames) // 2].save(still)
    print(f'  -> {still}')


if __name__ == '__main__':
    main()
