"""Sokoban grid rendering for figures and animations.

One module so the paper figures and the README animations look identical. Tiles
are drawn as flat vector shapes at SS x resolution and downsampled, which is what
keeps the circles and rings clean at the ~24px cell sizes a GitHub README wants.

Token ids follow diffusion/dataset.py:
    0 '#' wall   1 ' ' floor   2 '@' player   3 '$' box
    4 '.' goal   5 '*' box-on-goal   6 '+' player-on-goal   7 [MASK]
"""

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import labels

MASK_ID = 7
GRID = 10
SS = 3  # supersampling factor
LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS

# Dark scheme: the reveal reads much more clearly against a dark ground, and it
# sits acceptably on both GitHub themes.
C = {
    'page':     (17, 21, 27),
    'panel':    (26, 31, 40),
    'edge':     (44, 52, 65),
    'masked':   (33, 39, 49),
    'mask_dot': (58, 68, 84),
    'mask_glyph': (122, 138, 162),  # the 'M' on a masked cell
    'wall':     (142, 62, 52),  # brick face -- deeper and less saturated than
                                # the box orange, so the two never read alike
    'wall_top': (95, 110, 134),
    'mortar':   (188, 178, 168),  # light joints, the classic masonry read
    'floor':    (222, 219, 212),
    'grid':     (204, 200, 192),
    'goal':     (214, 118, 54),
    'box':      (198, 134, 66),
    'box_edge': (140, 90, 40),
    'box_ok':   (86, 163, 110),
    'box_ok_e': (56, 116, 76),
    'player':   (72, 133, 196),
    'player_e': (40, 88, 140),
    'text':     (206, 212, 222),
    'dim':      (124, 134, 150),
    'good':     (110, 190, 140),
    'bad':      (214, 106, 106),
    'bar':      (86, 140, 200),
    'bar_bg':   (40, 48, 60),
    'mark':     (232, 96, 96),    # culprit-wall highlight
    'mark_dim': (150, 66, 66),    # the off beat of its blink
}

# Difficulty shading. The cuts are the TRAINING CORPUS's own quartiles of solver
# search effort (states expanded) over its 450,000 solvable levels, so a tinted
# panel says "this hard relative to the data", not "this hard on some invented
# scale". Effort, not solution length, is the validated proxy (Jarusek & Pelanek
# 2010): a long solution can be entirely forced, and a short one can require
# search.
DIFF_CUTS = (625, 1638, 4177)

# A SEQUENTIAL ramp -- cool and dim for easy, warm and bright for hard -- so the
# ordering is readable without consulting a key. Four unrelated hues (green /
# blue / brown / red) encode four categories but no order, which is the wrong
# encoding for a quantity.
DIFF_TINT = [(30, 42, 54), (44, 46, 52), (62, 44, 40), (80, 38, 38)]
DIFF_EDGE = [(64, 86, 106), (92, 94, 100), (124, 86, 74), (158, 70, 70)]
DIFF_NAME = ('easiest 25%', 'lower mid', 'upper mid', 'hardest 25%')


def difficulty_band(states_expanded):
    """Which training-corpus difficulty quartile a level falls in (0..3)."""
    b = 0
    for cut in DIFF_CUTS:
        if states_expanded > cut:
            b += 1
    return b


def meter_width(unit):
    """Width diff_meter will occupy, for right-aligning it."""
    return 4 * unit * 0.85


def diff_meter(img_draw, x, y, band, unit):
    """Four rising bars, filled up to `band` -- signal-strength style.

    Colour alone is a weak encoding: it needs a key, and it fails outright for a
    red/green-blind reader. Bar height is ordered on its face, so the two
    together say the same thing twice.
    """
    for i in range(4):
        h = unit * (0.4 + 0.2 * i)
        x0 = x + i * unit * 0.85
        img_draw.rectangle([x0, y + unit - h, x0 + unit * 0.55, y + unit],
                           fill=DIFF_EDGE[band] if i <= band else None,
                           outline=DIFF_EDGE[band] if i <= band else (86, 92, 104),
                           width=max(1, int(unit * 0.09)))
    return 4 * unit * 0.85


@lru_cache(maxsize=None)
def font(size, bold=False, mono=False):
    # cached: the mask glyph asks for a font once per tile, which is ~100 cells
    # x 9 grids x 100 frames on a panel animation -- reloading the TTF each time
    # dominates the render otherwise
    name = 'consola.ttf' if mono else ('arialbd.ttf' if bold else 'segoeui.ttf')
    path = os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _brick(d, x, y, s):
    """Wall cell drawn as running-bond brickwork.

    The bond is keyed to ABSOLUTE canvas coordinates, not to the cell, so
    courses and joints line up across adjacent wall cells and a block of wall
    reads as one masonry surface instead of a grid of identical tiles. Every
    joint is clipped to the cell, so a cell never paints over its neighbour.
    """
    x1, y1 = x + s - 1, y + s - 1
    d.rectangle([x, y, x1, y1], fill=C['wall'])

    bh = max(2, int(round(s / 3.0)))       # course height: 3 courses per cell
    bw = bh * 2                            # 2:1 bricks
    lw = max(1, int(round(s / 22.0)))      # joint thickness
    m = C['mortar']

    c0 = (y - lw) // bh
    for c in range(c0, c0 + int(s / bh) + 3):
        cy = c * bh
        # bed joint (horizontal)
        ry0, ry1 = max(y, cy), min(y1, cy + lw - 1)
        if ry0 <= ry1:
            d.rectangle([x, ry0, x1, ry1], fill=m)
        # perpends (vertical), offset half a brick on alternate courses
        off = (bw // 2) if (c % 2) else 0
        vy0, vy1 = max(y, cy + lw), min(y1, cy + bh - 1)
        if vy0 > vy1:
            continue
        k0 = (x - off) // bw
        for k in range(k0, k0 + int(s / bw) + 3):
            vx = k * bw + off
            rx0, rx1 = max(x, vx), min(x1, vx + lw - 1)
            if rx0 <= rx1:
                d.rectangle([rx0, vy0, rx1, vy1], fill=m)


def _tile(d, x, y, s, tok):
    """Draw one tile with its top-left at (x, y) and side s (already scaled)."""
    box = [x, y, x + s - 1, y + s - 1]

    if tok == MASK_ID:
        d.rectangle(box, fill=C['masked'])
        # an 'M' rather than a dot: at panel cell sizes a dot reads as an
        # incidental speck, while the letter is unambiguously "not decided yet"
        d.text((x + s / 2, y + s / 2 + s * 0.02), 'M',
               font=font(max(6, int(s * 0.58)), bold=True),
               fill=C['mask_glyph'], anchor='mm')
        return

    if tok == 0:  # wall
        _brick(d, x, y, s)
        return

    # everything else sits on floor
    d.rectangle(box, fill=C['floor'])
    d.rectangle(box, outline=C['grid'], width=max(1, s // 24))

    if tok == 4:  # bare goal
        m = s * 0.30
        d.ellipse([x + m, y + m, x + s - m, y + s - m], outline=C['goal'],
                  width=max(1, int(s * 0.075)))

    if tok in (3, 5):  # box
        m = s * 0.17
        fill, edge = (C['box'], C['box_edge']) if tok == 3 else (C['box_ok'], C['box_ok_e'])
        d.rectangle([x + m, y + m, x + s - m, y + s - m], fill=fill,
                    outline=edge, width=max(1, int(s * 0.06)))
        i = s * 0.34
        d.rectangle([x + i, y + i, x + s - i, y + s - i], outline=edge,
                    width=max(1, int(s * 0.05)))

    if tok in (2, 6):  # player -- on a goal it takes the goal's ring as its
        m = s * 0.20   # outline, since a ring drawn underneath is simply covered
        d.ellipse([x + m, y + m, x + s - m, y + s - m], fill=C['player'],
                  outline=C['goal'] if tok == 6 else C['player_e'],
                  width=max(1, int(s * (0.10 if tok == 6 else 0.06))))


def draw_grid(img_draw, tokens, ox, oy, cell):
    """Draw a 10x10 grid of token ids at supersampled offset (ox, oy)."""
    for i, tok in enumerate(tokens):
        r, c = divmod(i, GRID)
        _tile(img_draw, ox + c * cell, oy + r * cell, cell, int(tok))


def mark_cell(img_draw, idx, ox, oy, cell, colour):
    """Outline one cell, to point at a specific position in a grid."""
    r, c = divmod(int(idx), GRID)
    x, y = ox + c * cell, oy + r * cell
    img_draw.rectangle([x, y, x + cell - 1, y + cell - 1],
                       outline=colour, width=max(1, int(cell * 0.11)))


def mark_label(img_draw, idx, ox, oy, cell, text, colour, fg=(255, 255, 255)):
    """A small chip pinned to one cell -- for putting a number ON the thing it
    describes, rather than in a caption the reader has to associate by position.

    Sits above the cell, or below it when the cell is on the top row, and is
    clamped horizontally so it never runs outside the grid.
    """
    r, c = divmod(int(idx), GRID)
    f = font(int(cell * 0.44), bold=True)
    tw = img_draw.textlength(text, font=f)
    pad = cell * 0.20
    w, h = tw + 2 * pad, cell * 0.66
    x = ox + c * cell + cell / 2 - w / 2
    y = oy + r * cell - h - cell * 0.14
    if r == 0:
        y = oy + cell * 1.14
    x = max(ox, min(x, ox + GRID * cell - w))
    img_draw.rounded_rectangle([x, y, x + w, y + h], radius=h * 0.32, fill=colour)
    img_draw.text((x + pad, y + h / 2), text, font=f, fill=fg, anchor='lm')


def render_grid(tokens, cell=24, pad=0, bg=None):
    """Standalone RGB image of a single 10x10 grid."""
    s = cell * SS
    p = pad * SS
    size = GRID * s + 2 * p
    im = Image.new('RGB', (size, size), bg or C['panel'])
    draw_grid(ImageDraw.Draw(im), tokens, p, p, s)
    return im.resize((GRID * cell + 2 * pad,) * 2, LANCZOS)


def save_gif(frames, path, duration, colors=128):
    """Write an animated GIF with one palette shared by every frame.

    A per-frame adaptive palette makes the static regions shimmer between frames
    and defeats delta compression, so all frames share one.

    Deriving that palette by sampling frames is not enough, and has now failed
    twice: frame 0 of a denoising run is entirely masked and holds no tile
    colours, and a highlight that appears in only a handful of frames is missed
    by any sparse sample. So the palette source is a swatch of EVERY named colour
    -- each guaranteed a slot regardless of how few frames or pixels use it --
    plus a spread of real frames to cover the antialiased blends between them.
    """
    sw = 24
    keys = sorted(C)
    swatch = Image.new('RGB', (sw * len(keys), sw))
    for i, k in enumerate(keys):
        swatch.paste(Image.new('RGB', (sw, sw), C[k]), (i * sw, 0))

    n = len(frames)
    picks = [frames[i] for i in sorted({round(j * (n - 1) / 7) for j in range(8)})]
    w, h = frames[0].size
    montage = Image.new('RGB', (w * len(picks), h + sw))
    for i, f in enumerate(picks):
        montage.paste(f, (i * w, 0))
    montage.paste(swatch.resize((montage.width, sw), Image.NEAREST), (0, h))
    pal = montage.quantize(colors=colors, method=Image.MEDIANCUT)

    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    durs = list(duration) if isinstance(duration, (list, tuple)) else [duration] * len(q)
    q[0].save(path, save_all=True, append_images=q[1:], duration=durs,
              loop=0, optimize=True, disposal=1)
    return os.path.getsize(path)


def frame_tokens(revealed_at, commit_class, k):
    """Grid state after k reveal steps: committed cells shown, the rest masked."""
    return np.where(revealed_at < k, commit_class, MASK_ID)


def progress_bar(d, x, y, w, h, frac):
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=C['bar_bg'])
    if frac > 0:
        d.rounded_rectangle([x, y, x + max(h, int(w * frac)), y + h],
                            radius=h // 2, fill=C['bar'])


def render_panel(grids, captions, title, counter, frac, cell, rows, cols,
                 marks=None, bands=None):
    """One frame of the multi-puzzle animation.

    Takes already-computed grids so every phase shares one renderer;
    `captions[i]` is a list of (text, colour) run together on the strip under
    grid i, or None for no caption yet. `marks[i]`, if given, is
    (cell_index, colour) and outlines that cell -- used to point at the wall
    whose removal makes an unsolvable level solvable. `bands[i]`, if given, is a
    difficulty quartile 0..3 and tints that panel's surround; a key is then drawn
    in the header, because an unlabelled colour scale means nothing.

    An empty `title` with `bands` set drops the title row entirely and moves
    `counter` onto the difficulty-key row instead (right after the last
    swatch), saving one row of header height -- for panels that don't need
    their own title repeated in every frame.
    """
    s, pad, gap = cell * SS, 14 * SS, 16 * SS
    cap = 26 * SS                              # caption strip under each grid
    m = 6 * SS                                 # tinted card margin around a grid
    compact = bool(bands) and not title        # no title row -- see docstring
    if compact:
        head = 54 * SS                         # bar + difficulty key, no title row
    else:
        head = (82 if bands else 60) * SS      # title + bar (+ difficulty key)
    gw = GRID * s
    W = 2 * pad + cols * gw + (cols - 1) * gap
    H = head + pad + rows * (gw + cap) + (rows - 1) * gap + pad

    im = Image.new('RGB', (W, H), C['page'])
    d = ImageDraw.Draw(im)

    f_title = font(18 * SS, bold=True)
    f_small = font(13 * SS)
    f_mono = font(12 * SS, mono=True)

    if compact:
        bar_y, key_y = pad, pad + 14 * SS
    else:
        d.text((pad, pad - 2 * SS), title, font=f_title, fill=C['text'])
        d.text((W - pad - d.textlength(counter, font=f_mono), pad + 1 * SS),
               counter, font=f_mono, fill=C['dim'])
        bar_y, key_y = pad + 28 * SS, pad + 42 * SS

    # frac=None suppresses the bar entirely -- a standalone still is not a frame
    # of anything, so a progress indicator would be meaningless on it
    if frac is not None:
        progress_bar(d, pad, bar_y, W - 2 * pad, 5 * SS, frac)

    if bands:
        lead = labels.GENERATE['difficulty_key'] + ' '
        d.text((pad, key_y), lead.strip(), font=f_small, fill=C['dim'])
        kx = pad + d.textlength(lead, font=f_small)
        for b in range(4):
            w = diff_meter(d, kx, key_y + 1 * SS, b, 12 * SS)
            d.text((kx + w + 4 * SS, key_y), DIFF_NAME[b], font=f_small,
                   fill=C['dim'])
            kx += w + 4 * SS + d.textlength(DIFF_NAME[b] + ' ', font=f_small)
        if compact:
            # right after "hardest 25%", not right-aligned to the far edge --
            # a right-aligned counter can collide with the key on a narrow panel.
            # A slightly smaller font than the title-row counter keeps this fitting
            # on the same row as the difficulty key at the panel's default width.
            f_counter = font(11 * SS, mono=True)
            d.text((kx, key_y), counter, font=f_counter, fill=C['dim'])

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ox = pad + c * (gw + gap)
        oy = head + pad + r * (gw + cap + gap)
        b = bands[i] if bands else None
        # the whole card -- grid plus its caption strip -- carries the tint, so
        # difficulty is legible at a glance rather than as a hairline border
        d.rounded_rectangle([ox - m, oy - m, ox + gw + m, oy + gw + cap],
                            radius=4 * SS,
                            fill=C['panel'] if b is None else DIFF_TINT[b],
                            outline=C['edge'] if b is None else DIFF_EDGE[b],
                            width=SS if b is None else 2 * SS)
        draw_grid(d, grids[i], ox, oy, s)
        if marks and marks[i]:
            mk = marks[i]
            mark_cell(d, mk[0], ox, oy, s, mk[1])
            if len(mk) > 2 and mk[2]:
                # chip colour is held constant while the ring blinks, so the
                # number stays readable instead of flickering
                mark_label(d, mk[0], ox, oy, s, mk[2], C['mark'])

        x = ox
        for text, colour in (captions[i] or ()):
            d.text((x, oy + gw + 5 * SS), text, font=f_small, fill=colour)
            x += d.textlength(text + '   ', font=f_small)

        # difficulty meter, right-aligned in the caption strip -- redundant with
        # the tint on purpose, so the ranking survives a reader who cannot rank
        # the colours
        if b is not None:
            u = 12 * SS
            diff_meter(d, ox + gw - meter_width(u), oy + gw + 5 * SS, b, u)

    return im.resize((W // SS, H // SS), LANCZOS)


def write_split(frames, durs, cut, out, names, hold_ms, stills=()):
    """Write one animation as two GIFs, cut at `cut`.

    The segments overlap by one frame: the last frame of the first is the first
    frame of the second, so the pair reads as a continuation rather than a jump.
    Both halves come from a single run, which is what guarantees they show the
    same levels -- regenerating the second half separately would reseed the
    sampler and silently show different puzzles.
    """
    for (a, b), name in zip(((0, cut + 1), (cut, len(frames))), names):
        seg, sd = frames[a:b], list(durs[a:b])
        sd[-1] = hold_ms
        p = os.path.join(out, name)
        size = save_gif(seg, p, sd)
        print(f"  -> {p}  ({len(seg)} frames, {seg[0].size[0]}x{seg[0].size[1]}, "
              f"{size/1e6:.2f} MB)")
    for idx, sname in stills:
        p = os.path.join(out, sname)
        frames[idx].save(p)
        print(f"  -> {p}")


def write(frames, durs, path, stills=()):
    """Write the GIF, plus any (frame_index, name) stills alongside it.

    The stills come from this same run rather than from seeking back into the
    written GIF -- PIL's optimizer merges consecutive identical frames, so GIF
    frame indices do not match the list that was handed to it.
    """
    size = save_gif(frames, path, durs)
    print(f"  -> {path}  ({len(frames)} frames, {frames[0].size[0]}x"
          f"{frames[0].size[1]}, {size/1e6:.2f} MB)")
    for idx, name in stills:
        p = os.path.join(os.path.dirname(path), name)
        frames[idx].save(p)
        print(f"  -> {p}")
