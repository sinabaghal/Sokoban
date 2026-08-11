"""Tile key on a light background, for pages that are not dark.

Identical geometry and identical tile art to legend.py -- it calls the same
render._tile -- so the swatches are pixel-for-pixel what the animations draw.
Only the surrounding canvas and the label ink change, to the light values used
by the chart figures (solvability.png and friends).

    python legend_light.py
"""

import os

from PIL import Image, ImageDraw

import render as R
from render import C, SS, font
import labels

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'out')

LIGHT_PAGE = (252, 252, 251)   # #fcfcfb, as in solvability.png
LIGHT_TEXT = (11, 11, 11)      # #0b0b0b
LIGHT_DIM = (82, 81, 78)       # #52514e

ORDER = [(7, 'mask'), (0, 'wall'), (1, 'floor'), (2, 'player'),
         (3, 'box'), (4, 'goal'), (5, 'box_on_goal'), (6, 'player_on_goal')]
ITEMS = [(tok, *labels.LEGEND[key]) for tok, key in ORDER]


def build(cell=30, out=None):
    s = cell * SS
    pad, gap = 10 * SS, 16 * SS
    lbl_w = 112 * SS
    col_w = s + 8 * SS + lbl_w
    cols, rows = 4, 2
    row_h = s + 8 * SS
    W = 2 * pad + cols * col_w + (cols - 1) * gap
    H = 2 * pad + rows * row_h

    im = Image.new('RGB', (W, H), LIGHT_PAGE)
    d = ImageDraw.Draw(im)
    f_name = font(12 * SS, bold=True)
    f_sym = font(11 * SS, mono=True)

    for i, (tok, name, sym) in enumerate(ITEMS):
        r, c = divmod(i, cols)
        x = pad + c * (col_w + gap)
        y = pad + r * row_h
        # the swatch keeps the dark panel/edge it has in the animations: these
        # are the tiles as drawn there, not recoloured for this page
        d.rectangle([x - SS, y - SS, x + s + SS, y + s + SS],
                    fill=C['panel'], outline=C['edge'], width=SS)
        R._tile(d, x, y, s, tok)
        tx = x + s + 8 * SS
        d.text((tx, y + 2 * SS), name, font=f_name, fill=LIGHT_TEXT)
        d.text((tx, y + 16 * SS), sym, font=f_sym, fill=LIGHT_DIM)

    im = im.resize((W // SS, H // SS), R.LANCZOS)
    if out:
        im.save(out)
        print(f'  -> {out}  ({im.size[0]}x{im.size[1]})')
    return im


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    build(out=os.path.join(OUT, 'legend_light.png'))
