"""Render the two algorithm boxes as PNGs for the page.

A fenced code block renders fine on GitHub but not everywhere the page might go,
and it cannot be styled. These are images, so they look the same wherever the
markdown lands.

Text lives in labels.ALGO_TRAIN / labels.ALGO_SAMPLE. The line numbers are load
bearing: the prose around each figure in SOKOBAN.md refers to specific lines
(training calls out 7 and 8, sampling calls out 4, 5 and 8), so renumbering the
algorithm means editing the prose too.

    python algorithm_figs.py
"""

import os

from PIL import Image, ImageDraw

import labels
from render import font, LANCZOS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'out')

SS = 2
SURFACE = (252, 252, 251)
PANEL = (246, 246, 243)
RULE = (222, 221, 216)
INK = (11, 11, 11)
CODE = (28, 32, 40)
NUM = (150, 150, 145)
COMMENT = (120, 130, 118)
WIDTH = 760          # matches the width= attribute used in SOKOBAN.md


def build(spec, path):
    pad, lh = 22 * SS, 21 * SS
    W = WIDTH * SS
    f_title = font(15 * SS, bold=True)
    f_code = font(13 * SS, mono=True)
    f_num = font(11 * SS, mono=True)
    f_in = font(11 * SS, mono=True)

    head = pad + 26 * SS + 22 * SS
    H = head + len(spec['lines']) * lh + pad

    im = Image.new('RGB', (W, H), SURFACE)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([pad // 2, pad // 2, W - pad // 2, H - pad // 2],
                        radius=6 * SS, fill=PANEL, outline=RULE, width=SS)

    d.text((pad, pad - 2 * SS), spec['title'], font=f_title, fill=INK)
    d.text((pad, pad + 22 * SS), spec['input'], font=f_in, fill=COMMENT)
    ry = pad + 40 * SS
    d.line([pad, ry, W - pad, ry], fill=RULE, width=SS)

    # comment column, placed past the longest line of code so nothing collides
    code_w = max(d.textlength(c, font=f_code) for _, c, _ in spec['lines'])
    cx = pad + 34 * SS + code_w + 26 * SS

    for i, (n, code, comment) in enumerate(spec['lines']):
        y = head + i * lh
        d.text((pad + 26 * SS - d.textlength(n, font=f_num), y), n,
               font=f_num, fill=NUM)
        d.text((pad + 34 * SS, y - 1 * SS), code, font=f_code, fill=CODE)
        if comment:
            d.text((cx, y), comment, font=f_num, fill=COMMENT)

    im = im.resize((W // SS, H // SS), LANCZOS)
    im.save(path)
    print(f'  -> {path}  ({im.size[0]}x{im.size[1]})')


def main():
    os.makedirs(OUT, exist_ok=True)
    build(labels.ALGO_TRAIN, os.path.join(OUT, 'algorithm_training.png'))
    build(labels.ALGO_SAMPLE, os.path.join(OUT, 'algorithm_sampling.png'))


if __name__ == '__main__':
    main()
