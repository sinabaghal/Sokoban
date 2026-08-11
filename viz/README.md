# Visualisation

Renders the generation process for the repository README and the paper figures.

| File | What |
|---|---|
| `render.py` | Tile drawing + GIF writing. Shared so figures and animations look identical. |
| `denoise_gif.py` | Samples the model with tracing and animates the reverse diffusion. |
| `legend.py` | Tile key (`out/legend.png`). |
| `out/` | Generated assets. |

## Outputs

```
python denoise_gif.py                    # both GIFs, default checkpoint
python denoise_gif.py --mode hero --seed 7
python legend.py
```

| Asset | |
|---|---|
| `out/denoise_panel.gif` | 3×3 levels denoising in lockstep, then being solved — the headline animation |
| `out/denoise_hero.gif` | one level, large, with the per-commit confidence readout |
| `out/panel_levels.png`, `out/hero_level.png` | last denoise frame: the levels as generated |
| `out/panel_solved.png`, `out/hero_solved.png` | last frame: every box on a goal |
| `out/wallfix_panel.gif` | 3×3 **unsolvable** levels, culprit wall highlighted, removed, then solved |
| `out/wallfix_broken.png` | as generated, unsolvable, no annotation |
| `out/wallfix_marked.png` | the same levels with each culprit wall outlined — the §6 figure |
| `out/wallfix_solved.png` | after repair, played to completion |
| `out/legend.png` | tile key |

Stills are written by the same run that writes the GIF, not by seeking back into
it — PIL's optimizer merges consecutive identical frames, so GIF frame indices do
not line up with the list handed to it.

## Two phases

Each animation runs the level forward twice. **Denoise:** 100 steps, one cell
committed per step, mask to finished level. **Solve:** the push solver's own
solution replayed one player move at a time, boxes turning green as they land on
goals. The point of the second phase is that the first phase's output is not just
plausible-looking — it is a level that can actually be played to completion, and
the animation shows it being played rather than asserting a push count.

The solution comes from `playback.solution_moves`, the same `solve_push` call the
corpus measurement uses, so a level captioned solvable here is solvable by the
paper's definition. `playback.replay` refuses to apply an illegal move and stops
instead, so a bad path truncates the animation rather than corrupting it.

Timing knobs: `--move-ms` (per move), `--move-stride` (moves per frame),
`--pause-ms` (beat on the finished level before play starts), `--no-solve` to
stop at generation. `--move-stride 2` roughly halves the solve-phase frames and
brings the panel GIF from ~6.5 MB to ~5 MB.

## The wall-fix animation

```
python denoise_gif.py --mode wallfix --pool 256 --no-hamming
```

Four phases: denoise to a level the solver proves **unsolvable**; blink the one
interior wall whose removal fixes it; delete that wall; play the repaired level
to completion. This is the paper's §6 result as an animation — the model's
failures are overwhelmingly one cell from correct.

The search is `run_eval._wallfix_worker`, imported rather than reimplemented. It
enumerates interior walls having at least one non-wall neighbour (a wall boxed in
by four walls is unreachable by player and box alike, so removing it provably
cannot change solvability) and tries each. A representative run: **50 of 52
proven-unsolvable levels were 1-wall fixable (96.2%)**, reproducing the paper's
96% independently. The caption notes when more than one wall would have worked.

**This shows participation, not blame.** "Removing wall X makes it solvable"
establishes that X takes part in the deadlock — not that committing X was the
model's mistake. The real error may have been a box placed forty steps later.
Do not caption this animation as "the model's mistake, corrected".

Needs a large `--pool`: only ~20% of samples are unsolvable, so 256 yields ~50
candidates for 9 panels. The search costs ~35 s for 52 levels.

Defaults point at `../diffusion/checkpoints_T100/step_290000.pt`, the
$T{=}100$ full-corpus run the paper reports.

## Two things the animation must not be read as saying

**Reveal order is not a model decision.** Sampling uses `strategy='random'`, so
the cell unmasked at each step is chosen uniformly at random from those still
masked. The model chooses *what* goes in a cell, never *which* cell comes next.
"The model draws the walls first, then places the boxes" is not visible here and
would not be true; that question only becomes meaningful under confidence-ordered
reveal.

**`--num-timesteps` must match training.** Checkpoints do not store $T$, and
sampling a $T{=}100$ model on a different schedule silently degrades output
rather than erroring.

## Fidelity to the evaluated sampler

`sample_traced()` mirrors `MaskedDiffusion.sample` and the instrumented copy in
`eval/run_eval.py` step for step, so the animation shows the process that
produced the paper's numbers rather than a re-enactment of it. It records
`revealed_at` and `commit_class` per cell; every intermediate frame is then
recoverable as "committed cells where `revealed_at < k`, mask elsewhere", so no
per-step grid is stored.

Puzzles are solved with the same push solver as the corpus before rendering, and
by default only solvable ones are shown, captioned with their push count.
`--no-filter` draws an unfiltered sample instead — worth using if the animation
is meant to illustrate the failure rate rather than the successes.

## Novelty caption

Each caption also carries the nearest-neighbour Hamming distance from that level
to the **whole 450,000-puzzle training set**, so the animation answers "is it
just replaying training data?" on its face rather than in a footnote. This calls
`compare/identity.py` directly instead of reimplementing it: one-hot to 700 dims
(100 cells × 7 tile types) makes the inner product the count of agreeing cells,
so distance is `100 - agreement` and `0` means a verbatim reproduction. Costs
~12 s to load the corpus; `--no-hamming` skips it.

**The raw distance is not self-interpreting.** A level 5 cells from its nearest
training neighbour sounds close, but real held-out Boxoban puzzles collide with
the training set at comparable rates — that comparison is §9 of the paper, and
the number here should be read against it, not on its own.

## GIF palette

All frames share one palette, built from a **spread** of frames rather than from
frame 0. Frame 0 of a denoising run is entirely masked and contains none of the
tile colours, so a palette derived from it renders every box, goal and player as
grey — which is exactly what the first version of this script did.
