"""Every word that appears on a figure or animation. Edit here, nowhere else.

Nothing in this file affects a number. Titles, captions, axis labels and legend
text all live here so that rewording a figure is a one-file change and can never
alter what is being plotted — the data comes from `csv/`, the words come from
here, and the scripts only put them together.

Strings containing `{...}` are `str.format` templates. Keep the field names; the
scripts fill them. Available fields are listed above each block.

    python solvability_plot.py        # ~1 s to see a wording change
    python commit_heatmap_gif.py      # ~5 s
    python denoise_gif.py --mode panel --pool 64        # re-samples, ~60 s
"""

# ---------------------------------------------------------------- figures ----

# fields: none
SOLVABILITY = {
    'title':      'Failures get shallower, not just rarer',
    'subtitle':   'n=5,000 samples per checkpoint; wall-fix share from n=100',
    'exact':      'solvable as generated',
    'effective':  'after removing ≤1 wall',
    'gap_note':   'failures one wall\nfrom solvable',
    'x':          'training step (thousands)',
    'y':          'puzzles solvable (%)',
}

# fields: none
HAMMING = {
    'generated':  'generated puzzles',
    'held_out':   'real held-out puzzles',
    'x':          'cells differing from nearest training puzzle',
    'y':          '% of puzzles',
}

# fields: {kmax} largest sample size, {floor_max} where the floor series ends
JSD_SAMPLE_SIZE = {
    'title':      'The generated distribution sits on the measurement floor',
    'subtitle':   'tile-pattern JSD vs the 450,000-puzzle training corpus, by sample size',
    'floor':      'real held-out puzzles',
    'model':      'generated puzzles',
    'x':          'puzzles in sample',
    'y':          'Jensen-Shannon divergence (bits)',
    'note':       'a small sample of ANY source scores\nfar above zero — this is that floor',
    'floor_end':  'held-out data\nexhausted (50k)',
}

# fields: {n} samples per temperature
TEMPERATURE_SWEEP = {
    'title':      'Lower temperature trades wall realism for solvability',
    'subtitle':   'n={n:,} samples per temperature, T=100 baseline, step 290,000',
    'solvable':   'solvable',
    'walls':      'avg. walls per puzzle',
    'x':          'sampling temperature',
    'y_solvable': 'puzzles solvable (%)',
    'y_walls':    'average wall count',
}

# fields: {n} series size, {n_levels} number of levels, {a} {b} percentages
CULPRIT = {
    'title':      'The model was least sure about the cells that broke the level',
    'subtitle':   ('commitment probability of interior walls, {n_levels} '
                   'unsolvable levels at step 255000'),
    'culprit':    'culprit walls  (n={n})',
    'other':      'every other interior wall, same levels  (n={n})',
    'note':       '{a:.0f}% of culprits sit below 0.7\nagainst {b:.0f}% of the rest',
    'x':          'probability the model assigned the wall it committed',
    'y':          '% of that group',
}

# fields: {sid} sample id, {status} solvable/unsolvable, {med} interior median
HEATMAP = {
    'title':        'How sure was the model about each cell?',
    'subtitle':     ('commitment probability per cell, step 255000 - recorded '
                     'during generation, not re-inferred'),
    'cbar':         'probability the model assigned the tile it committed',
    'panel':        '#{sid}   {status}   interior median p={med:.2f}',
    'curve_title':  ('Confidence rises as context accumulates - except where '
                     'there was never any doubt'),
    'curve_x':      'reveal step (cell committed at this iteration)',
    'curve_y':      'median commit probability',
    'border':       'border ring - 100% wall, never in doubt',
    'interior':     'interior cells',
}

# ------------------------------------------------------------- animations ----

# fields: {k} step, {steps} total, {sid} sample id, {p} probability,
#         {status} solvable/unsolvable
HEATMAP_GIF = {
    'title':          'Which cell, when, and how sure',
    'counter':        '{k:>3}/100 cells committed',
    'key_low':        'p = 0  (coin flip on 7 tiles)',
    'key_high':       'p = 1  (forced)',
    'caption_idle':   '#{sid}',
    'caption_step':   '#{sid}  committed p={p:.2f}',
    'caption_final':  '#{sid}  {status}',
    'solvable':       'solvable',
    'unsolvable':     'unsolvable',
}

# fields: {k} cells committed, {steps} total, {pushes}, {effort}, {nn} distance,
#         {m} move number, {done} solved so far, {n} panel count
GENERATE = {
    'counter':        '{k:>3}/{steps} committed',
    'solvable':       'solvable  {pushes} pushes',
    'unsolvable':     'unsolvable',
    'effort':         '',
    'nn':             '{nn} from train',
    'difficulty_key': 'difficulty:',
    'solve_counter':  '{done}/{n} solved',
    'solved':         'solved  {pushes} pushes',
    'move':           'move {m}',
}

# fields: {k}, {steps}, {pushes}, {states}, {nn}, {m}, {total}, {confidence}
HERO = {
    'title':          'One level, 100 denoising steps',
    'counter':        '{k:>3}/{steps}',
    'confidence':     'committed at p =',
    'solvable':       'solvable   {pushes} pushes   {states:,} states expanded',
    'unsolvable':     'unsolvable',
    'nn':             'nearest training level: {nn} of 100 cells differ',
    'solve_title':    'Playing the solution back',
    'move':           'move {m} of {total}   ({pushes} pushes)',
    'solved':         'solved',
}

# `denoise_gif.py --mode wallfix` — generates fresh unsolvable levels
# fields: {k}, {steps}, {n_fixes} how many walls would work, {done}, {n}
WALLFIX = {
    'gen_title':      'Masked diffusion generating Sokoban levels',
    'counter':        '{k:>3}/{steps} cells committed',
    'unsolvable':     'unsolvable',
    'blink_title':    'One wall is in the way',
    'blink_counter':  'the model missed one cell',
    'blink_caption':  'remove this wall',
    'fixed_title':    'One wall removed',
    'fixed_counter':  'now solvable',
    'fixed_caption':  '1 wall removed',
    'fixed_alt':      '({n_fixes} would work)',
    'solve_title':    'Playing the solutions back',
    'solve_counter':  '{done}/{n} solved',
    'solved':         'solved  {pushes} pushes',
    'effort':         '',
    'move':           'move {m}',
}

# `wallfix_replay.py` — replays the recorded evaluation run, so it can also show
# the confidence the model committed each culprit wall at
# fields: {k}, {steps}, {p} culprit probability, {b} that level's wall baseline,
#         {n_fixes}, {done}, {n}, {m}
WALLFIX_REPLAY = {
    'gen_title':      'Replayed from the recorded run',
    'counter':        '{k:>3}/{steps} cells committed',
    'unsolvable':     'unsolvable',
    'blink_title':    'The model was unsure about this cell',
    'blink_counter':  'culprit vs other interior walls',
    'blink_caption':  'p={p:.2f}',
    'blink_baseline': 'vs {b:.2f} here',
    'blink_chip':     'p={p:.2f}',
    'fixed_title':    'One wall removed',
    'fixed_counter':  'now solvable',
    'fixed_caption':  '1 wall removed',
    'fixed_alt':      '({n_fixes} would work)',
    'solve_title':    'Playing the repaired levels',
    'solve_counter':  '{done}/{n} solved',
    'solved':         'solved',
    'move':           'move {m}',
}

# standalone still: the unsolvable levels with the wall that breaks each one
# fields: {p} culprit probability, {b} that level's other-wall baseline
WALLFIX_STILL = {
    'title':    'The wall that breaks each level',
    # drawn right-aligned on the title line, so it has to stay short
    'subtitle': 'proven unsolvable  ·  step 255000',
    # the chip on the wall already shows p, so the strip only needs the contrast
    'caption':  'p={p:.2f}',
    'baseline': 'vs {b:.2f} here',
    'chip':     'p={p:.2f}',
}

# tile key image
LEGEND = {
    'mask':           ('[MASK]', 'not yet committed'),
    'wall':           ('wall', '#'),
    'floor':          ('floor', 'space'),
    'player':         ('player', '@'),
    'box':            ('box', '$'),
    'goal':           ('goal', '.'),
    'box_on_goal':    ('box on goal', '*'),
    'player_on_goal': ('player on goal', '+'),
}


# Rendered algorithm boxes (assets/algorithm_*.png). Each line is
# (number, code, trailing comment); '' for a blank number keeps the gutter clean.
# The numbering is referenced by the prose around them in SOKOBAN.md -- training
# calls out lines 7 and 8, sampling calls out lines 4, 5 and 8. Renumber both
# together or the text stops matching the picture.
ALGO_TRAIN = {
    'title': 'Algorithm 1 — Training step',
    'input': 'input:  a batch of levels x0 in {0..6}^100,  horizon T = 100,  cap w_max = 10',
    'lines': [
        ('1',  't   ~ Uniform{1, ..., T}',                 'one noise level per example'),
        ('2',  'a_t <- 1 - t/T',                           'fraction of cells kept'),
        ('3',  'for every cell i:',                        'forward process'),
        ('4',  '    x_t[i] <- x0[i]   w.p. a_t',            ''),
        ('5',  '             [MASK]   otherwise',           ''),
        ('6',  'logits <- model(x_t, t)',                  '[100 x 7], all cells at once'),
        ('7',  'M <- { i : x_t[i] = [MASK] }',             'only hidden cells are scored'),
        ('8',  'w <- min( 1/(1-a_t), w_max )',             '= min(T/t, w_max)'),
        ('9',  'L <- w * (1/|M|) * sum_{i in M} CE( logits[i], x0[i] )', ''),
        ('10', 'backpropagate L, clip |g| to 1.0, AdamW step', ''),
    ],
}

ALGO_SAMPLE = {
    'title': 'Algorithm 2 — Sampling (the reverse process)',
    'input': 'input:  trained model,  T = 100 steps,  temperature tau = 1.0        output: a finished level',
    'lines': [
        ('1',  'x <- [MASK]^100',                          'nothing decided yet'),
        ('2',  'for step = 0 ... T-1:',                    ''),
        ('3',  '    t <- T - round(step * T / T)',         'walks T -> 1'),
        ('4',  '    p <- softmax( model(x, t) / tau )',    'a belief for EVERY cell'),
        ('5',  '    x^[i] ~ Categorical(p[i])  for all i', 'sampled, not argmax'),
        ('6',  '    M <- { i : x[i] = [MASK] }',           'still-hidden cells'),
        ('7',  '    n <- max( ceil( |M| / (T-step) ), 1 )', '= 1 here, since T = 100 cells'),
        ('8',  '    S <- n cells drawn uniformly at random from M', ''),
        ('9',  '    x[S] <- x^[S]',                        'committed - never revised'),
        ('10', 'return x',                                 ''),
    ],
}
