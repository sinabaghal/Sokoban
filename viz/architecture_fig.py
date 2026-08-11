"""Render a schematic of SokobanTransformer's architecture as a PNG.

Same light "card" style as algorithm_figs.py (PIL, not matplotlib -- this
family of figures is a rendered diagram, not a data chart), so it sits
consistently next to the two algorithm boxes it's grouped with in SOKOBAN.md.
Every box/shape here mirrors code/diffusion/model.py's SokobanTransformer
directly -- layer names, dimensions and the block count all come from there.

    python architecture_fig.py
"""

import os

from PIL import Image, ImageDraw

from render import font, LANCZOS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'out')

SS = 2
SURFACE = (252, 252, 251)
PANEL = (246, 246, 243)
BLOCK_PANEL = (240, 240, 236)
RULE = (222, 221, 216)
INK = (11, 11, 11)
CODE = (28, 32, 40)
DIM = (120, 122, 118)
ACCENT = (58, 90, 168)      # box outline for the repeated transformer block
WIDTH = 760                  # matches width= used elsewhere in SOKOBAN.md


def rrect(d, box, **kw):
    d.rounded_rectangle(box, radius=5 * SS, **kw)


def arrow_down(d, cx, y0, y1, color=DIM, w=None):
    w = w or SS
    d.line([cx, y0, cx, y1 - 6 * SS], fill=color, width=w)
    d.polygon([(cx - 4 * SS, y1 - 6 * SS), (cx + 4 * SS, y1 - 6 * SS), (cx, y1)],
              fill=color)


def centered_text(d, cx, y, text, font_, fill, anchor='mm'):
    d.text((cx, y), text, font=font_, fill=fill, anchor=anchor)


def build(path):
    pad = 22 * SS
    W = WIDTH * SS
    f_title = font(15 * SS, bold=True)
    f_sub = font(11 * SS, mono=True)
    f_box = font(12 * SS, bold=True)
    f_dim = font(int(10.5 * SS), mono=True)
    f_small = font(10 * SS)

    inner_w = W - 2 * pad

    # ---- fixed row heights, laid out top to bottom as we draw ----
    head_h = 30 * SS + 20 * SS + 14 * SS
    io_h = 30 * SS
    gap = 14 * SS
    emb_h = 40 * SS
    sum_h = 22 * SS
    block_inner_h = 30 * SS
    block_pad = 12 * SS
    n_block_rows = 6  # LN, Attn, +, LN, FFN, +
    block_h = block_pad * 2 + n_block_rows * block_inner_h + (n_block_rows - 1) * 4 * SS
    xN_h = 22 * SS
    final_h = 30 * SS

    H = (pad + head_h + io_h + gap * 2 + emb_h + gap + sum_h + gap
         + xN_h + block_h + gap
         + final_h + gap + final_h + gap + io_h + pad)

    im = Image.new('RGB', (W, H), SURFACE)
    d = ImageDraw.Draw(im)
    rrect(d, [pad // 2, pad // 2, W - pad // 2, H - pad // 2],
          fill=PANEL, outline=RULE, width=SS)

    y = pad
    d.text((pad, y), 'SokobanTransformer', font=f_title, fill=INK)
    d.text((pad, y + 22 * SS), '6-layer pre-norm transformer encoder  ·  4,879,623 parameters',
           font=f_sub, fill=DIM)
    ry = y + 42 * SS
    d.line([pad, ry, W - pad, ry], fill=RULE, width=SS)
    y = ry + 20 * SS
    cx = W // 2

    # ---- input ----
    box = [pad, y, W - pad, y + io_h]
    rrect(d, box, fill=BLOCK_PANEL, outline=RULE, width=SS)
    centered_text(d, cx, (box[1] + box[3]) // 2 - 6 * SS,
                  'input tokens  x  —  [B, 100]', f_box, CODE)
    centered_text(d, cx, (box[1] + box[3]) // 2 + 9 * SS,
                  'each cell: one of 8 ids (7 tiles + [MASK])', f_dim, DIM)
    y = box[3]
    arrow_down(d, cx, y, y + gap)
    y += gap

    # ---- three embeddings, side by side ----
    col_gap = 10 * SS
    col_w = (inner_w - 2 * col_gap) // 3
    labels3 = [
        ('token embedding', 'Embedding(8 → 256)'),
        ('row + col embedding', '2 × Embedding(10 → 256)'),
        ('timestep embedding', 'sinusoidal → Linear → GELU → Linear'),
    ]
    ex = pad
    centers = []
    for name, dim in labels3:
        box = [ex, y, ex + col_w, y + emb_h]
        rrect(d, box, fill=BLOCK_PANEL, outline=RULE, width=SS)
        bcx = (box[0] + box[2]) // 2
        centers.append(bcx)
        centered_text(d, bcx, (box[1] + box[3]) // 2 - 7 * SS, name, f_box, CODE)
        centered_text(d, bcx, (box[1] + box[3]) // 2 + 9 * SS, dim, f_dim, DIM)
        ex += col_w + col_gap
    y += emb_h

    # converge the three into the centre, then straight down
    merge_y = y + gap // 2
    for bcx in centers:
        d.line([bcx, y, bcx, merge_y], fill=DIM, width=SS)
    d.line([centers[0], merge_y, centers[-1], merge_y], fill=DIM, width=SS)
    arrow_down(d, cx, merge_y, y + gap)
    y += gap

    # ---- sum ----
    box = [cx - 60 * SS, y, cx + 60 * SS, y + sum_h]
    rrect(d, box, fill=PANEL, outline=ACCENT, width=SS)
    centered_text(d, cx, (box[1] + box[3]) // 2, 'sum  →  h  —  [B, 100, 256]', f_dim, INK)
    y = box[3]
    arrow_down(d, cx, y, y + gap)
    y += gap

    # ---- x6 label ----
    d.text((pad, y), '× 6', font=font(12 * SS, bold=True), fill=ACCENT)
    y += xN_h

    # ---- transformer block ----
    block_box = [pad, y, W - pad, y + block_h]
    rrect(d, block_box, fill=SURFACE, outline=ACCENT, width=SS)
    by = block_box[1] + block_pad
    bx0, bx1 = block_box[0] + block_pad, block_box[2] - block_pad
    rows = [
        ('LayerNorm', None),
        ('Multi-Head Self-Attention', '8 heads · d_model=256'),
        ('+ residual', None),
        ('LayerNorm', None),
        ('Feed-Forward', 'Linear(256→1024) → GELU → Linear(1024→256)'),
        ('+ residual', None),
    ]
    for i, (label, sub) in enumerate(rows):
        rb = [bx0, by, bx1, by + block_inner_h]
        is_residual = label.startswith('+')
        rrect(d, rb, fill=PANEL if not is_residual else SURFACE,
              outline=RULE, width=SS)
        bcx = (rb[0] + rb[2]) // 2
        if sub:
            centered_text(d, bcx, (rb[1] + rb[3]) // 2 - 6 * SS, label, f_box, CODE)
            centered_text(d, bcx, (rb[1] + rb[3]) // 2 + 9 * SS, sub, f_dim, DIM)
        else:
            centered_text(d, bcx, (rb[1] + rb[3]) // 2, label,
                          f_box if not is_residual else f_dim,
                          CODE if not is_residual else DIM)
        if i < len(rows) - 1:
            arrow_down(d, bcx, rb[3], rb[3] + 4 * SS, w=SS)
        by += block_inner_h + 4 * SS
    y = block_box[3]
    arrow_down(d, cx, y, y + gap)
    y += gap

    # ---- final norm ----
    box = [pad, y, W - pad, y + final_h]
    rrect(d, box, fill=BLOCK_PANEL, outline=RULE, width=SS)
    centered_text(d, cx, (box[1] + box[3]) // 2, 'Final LayerNorm', f_box, CODE)
    y = box[3]
    arrow_down(d, cx, y, y + gap)
    y += gap

    # ---- output projection ----
    box = [pad, y, W - pad, y + final_h]
    rrect(d, box, fill=BLOCK_PANEL, outline=RULE, width=SS)
    centered_text(d, cx, (box[1] + box[3]) // 2, 'Output projection  —  Linear(256 → 7)',
                  f_box, CODE)
    y = box[3]
    arrow_down(d, cx, y, y + gap)
    y += gap

    # ---- output ----
    box = [pad, y, W - pad, y + io_h]
    rrect(d, box, fill=PANEL, outline=ACCENT, width=SS)
    centered_text(d, cx, (box[1] + box[3]) // 2 - 6 * SS,
                  'logits  —  [B, 100, 7]', f_box, INK)
    centered_text(d, cx, (box[1] + box[3]) // 2 + 9 * SS,
                  'one of 7 tile classes per cell (mask token excluded)', f_dim, DIM)

    im = im.resize((W // SS, H // SS), LANCZOS)
    im.save(path)
    print(f'  -> {path}  ({im.size[0]}x{im.size[1]})')


def main():
    os.makedirs(OUT, exist_ok=True)
    build(os.path.join(OUT, 'architecture.png'))


if __name__ == '__main__':
    main()
