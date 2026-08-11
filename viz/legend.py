"""Tile key for the README, so the animations are readable without a caption."""

import os

from PIL import Image, ImageDraw

import render as R
from render import C, SS, font

import labels

# Text lives in labels.LEGEND; this only fixes the drawing order.
ORDER = [(7, 'mask'), (0, 'wall'), (1, 'floor'), (2, 'player'),
         (3, 'box'), (4, 'goal'), (5, 'box_on_goal'), (6, 'player_on_goal')]
ITEMS = [(tok, *labels.LEGEND[key]) for tok, key in ORDER]


def build(cell=30, out=None):
    s, pad, gap = cell * SS, 14 * SS, 26 * SS
    lbl_w = 118 * SS
    col_w = s + 8 * SS + lbl_w
    cols, rows = 4, 2
    W = 2 * pad + cols * col_w + (cols - 1) * gap
    H = 2 * pad + rows * (s + 12 * SS)

    im = Image.new('RGB', (W, H), C['page'])
    d = ImageDraw.Draw(im)
    f_name = font(12 * SS, bold=True)
    f_sym = font(11 * SS, mono=True)

    for i, (tok, name, sym) in enumerate(ITEMS):
        r, c = divmod(i, cols)
        x = pad + c * (col_w + gap)
        y = pad + r * (s + 12 * SS)
        d.rectangle([x - SS, y - SS, x + s + SS, y + s + SS],
                    fill=C['panel'], outline=C['edge'], width=SS)
        R._tile(d, x, y, s, tok)
        tx = x + s + 8 * SS
        d.text((tx, y + 2 * SS), name, font=f_name, fill=C['text'])
        d.text((tx, y + 16 * SS), sym, font=f_sym, fill=C['dim'])

    im = im.resize((W // SS, H // SS), R.LANCZOS)
    if out:
        im.save(out)
    return im


if __name__ == '__main__':
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'legend.png')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    build(out=p)
    print(f'-> {p}')
