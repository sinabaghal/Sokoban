# Evaluation Pipeline Spec (`eval.md`)

A specification (and now implementation, `run_eval.py`) for an inference-only
evaluation over training checkpoints. It generates puzzles from each selected
checkpoint, records the model's per-token probability trajectory across the
denoising iterations, and measures solvability (including push-move count). No
training is involved — checkpoints are read from disk. Everything lives under
`source/eval/`; outputs default to `source/eval/output/`.

---

## 1. Inputs / Parameters

**There are no sub-sampling knobs.** `--num N` is the single dial: everything —
probability traces, commit records, solvability, wall-fix — is recorded for **all
N** samples. This keeps every output file on the same index space, so cross-file
joins always work.

| Parameter | Meaning |
|---|---|
| `--checkpoint-dir` | Folder of training checkpoints (`step_*.pt`) to evaluate. |
| `--num-timesteps T` | Diffusion noise levels the run was trained with (must match; checkpoints don't store it). |
| `--num N` | Puzzles **generated per checkpoint** — and fully recorded. |
| `--every K` | Evaluate every **K-th** checkpoint (sorted by step). `K=1` = all. |
| `--temperature` | Sampling temperature (default 1.0). |
| `--chunk` | GPU generation batch size, VRAM-bound (default 2500). Also the Parquet row-group size. |
| `--max-states` | Push-solver state cap (default 200000). Lowering it does **not** speed things up; below ~50k it produces false unsolvables. |
| `--prob-f32` | Store trace probabilities as float32 instead of float16 (~6.5× larger; float16 error ≤ 1e-4). |
| `--workers` | Solver worker processes (default = logical cores). |
| `--out output/` | Output root (default `source/eval/output/`). |

The set of evaluated checkpoints is `sorted(step_*.pt)[::K]`. Resumable: steps
already present in `output/perf.csv` are skipped.

**Cost.** `--num` and `--every` are the only levers on runtime. The wall-fix
analysis (§5) dominates: it runs on every proven-unsolvable puzzle at ~0.6s median
(tail >15s) each, so an early checkpoint at `--num 2000` (~58% unsolvable) takes
roughly 10–20 min. Later checkpoints are much cheaper. Storage is ~190 MB per
checkpoint at `--num 2000`.

---

## 2. Output Layout

All under `source/eval/`:

```
source/eval/
  run_eval.py
  eval.md
  output/
    ckp_<step>/
      prob_trace.parquet     # per-iteration 7-class probabilities, ALL N samples (see §3)
      reveal.csv             # per-token commit record, ALL N samples (see §3b)
      solvability.csv        # per-puzzle status + push count + search effort (see §4)
      wallfix.csv            # which wall(s) fix each proven-unsolvable puzzle (see §5)
      samples.txt            # all N grids, Boxoban format, header: "; <id> <status>"
    perf.csv                 # one summary row per checkpoint (aggregates + wall-fix rates)
```

---

## 3. Probability Trace (`prob_trace.parquet`) — all N samples

The grid is 10×10 = **100 tokens**. During generation (the reverse diffusion
process), at each **iteration** the model produces, for every position, a softmax
distribution over the 7 tile types. This file records that full trajectory.

**Schema** — one row per `(sample, iteration, token)`, with the 7 class
probabilities as columns:

| Column | Type | Meaning |
|---|---|---|
| `sample_id` | int32 | Generated puzzle (0 … N−1) — **all** samples. |
| `iteration` | int16 | Denoising step (0 … `steps−1`, `steps = min(T, 100)`). |
| `row_id`, `col_id` | int8 | Grid position; `token_index = row_id*10 + col_id`. |
| `p_wall` | float16 | `#` (class 0) |
| `p_floor` | float16 | space (class 1) |
| `p_player` | float16 | `@` (class 2) |
| `p_box` | float16 | `$` (class 3) |
| `p_goal` | float16 | `.` (class 4) |
| `p_box_on_goal` | float16 | `*` (class 5) |
| `p_player_on_goal` | float16 | `+` (class 6) |

Rows: `N × steps × 100`. Written as one Parquet row group per generation chunk
(`--chunk`), streamed so memory stays bounded.

**Why Parquet + float16.** Measured on a 25-sample trace (250k rows):

| Format | Size | Write |
|---|---|---|
| 7 per-tile CSVs (the original design) | 37.5 MB | 1.9s |
| 1 Parquet, float32, zstd | 9.7 MB | 0.2s |
| **1 Parquet, float16, zstd** | **1.5 MB** | 0.2s |

25× smaller and ~10× faster to write, which is what makes recording *every* sample
affordable. float16 error is ≤ 1e-4 — finer than the old CSV's `%.5f` for small
probabilities, and far below anything analytically meaningful here. Use `--prob-f32`
if you want exactness.

`revealed_at` is deliberately **not** in this file: it is a per-`(sample, token)`
property that the old format repeated once per iteration. It lives in `reveal.csv`
— join on `(sample_id, row_id, col_id)`.

Read it with:
```python
import pyarrow.parquet as pq
df = pq.read_table('output/ckp_290000/prob_trace.parquet').to_pandas()
```

---

## 3b. Commit Record (`reveal.csv`) — all N samples

`prob_trace.parquet` is the full trajectory (a *movie*: every cell's belief at
every iteration). Most analyses only need the **commit moment** (a single
*snapshot* per cell), so that is also recorded in a lean, easily-joined file:

| Column | Meaning |
|---|---|
| `sample_id` | Generated puzzle (0 … N−1). |
| `row_id`, `col_id` | Grid position. |
| `revealed_at` | Iteration at which this cell was committed (its reveal step). |
| `committed_class` | Token id actually committed (0 `#`, 1 ` `, 2 `@`, 3 `$`, 4 `.`, 5 `*`, 6 `+`). |
| `commit_prob` | Probability of the **committed** token, at commit time. |
| `max_prob` | Probability of the model's **top choice**, at commit time. |

Size is `N × 100` rows — 100× smaller than the trace (which has a row per
*iteration* per cell). It is the workhorse for cross-file joins:
`(sample_id, row_id, col_id)` joins `wallfix.csv` and `prob_trace.parquet`, and
`sample_id` joins `solvability.csv` / `samples.txt`.

**Why `commit_prob` *and* `max_prob`.** The two separate distinct failure modes:

| Pattern | Reading |
|---|---|
| `max_prob` low (≈ `commit_prob`) | **Model ignorance** — flat distribution, it genuinely didn't know. Needs a better model. |
| `max_prob` high, `commit_prob` low | **Sampler slip** — the model's top choice was something else and the multinomial draw took the tail. Fixable for free by lowering temperature. |

Empirically ~84% of commits are argmax draws and ~16% take the tail, so both
modes occur and are distinguishable.

**Two analyses this enables** (both need `wallfix.csv` for ground truth on which
cell was the mistake):
1. **Reveal-step distribution of culprit walls** — join culprit coordinates to
   `revealed_at`. Compare against the reveal-step distribution of *all* walls in
   the same puzzles (a random wall is ≈ uniform over 0–99), or there is no null
   hypothesis to reject.
2. **Commit confidence at culprit cells** — compare culprit-cell `commit_prob`
   against (a) other wall cells in the same puzzle and (b) wall cells in solvable
   puzzles. Control for `revealed_at`: confidence rises naturally with reveal step
   (later cells have more context), so compare at matched steps or regress it out.

**Attribution caveat.** A deadlock is a *joint* property of walls, boxes and
goals. "Removing wall X fixes it" means X *participates* in the deadlock — not
that committing X was the error (the real mistake may have been a box placed 40
steps later). Read these analyses as being about *participating* cells.

---

## 4. Solvability + Push-Move Count

For each of the `N` generated puzzles, run the push-based solver
(`test_solver.solve_push`). Recorded per puzzle in `solvability.csv`:

| Column | Meaning |
|---|---|
| `sample_id` | Generated puzzle index (0 … N−1). |
| `valid` | Structurally valid (1 player, boxes = goals, ≥1 box). |
| `solvable` | 1 if the push solver found a solution. |
| `pushes` | **Number of box pushes in the solution** (−1 if not solved). |
| `states_expanded` | Push-nodes the solver expanded (search effort). |
| `status` | `solvable` / `unsolvable` / `unknown_capped` / `invalid` — see below. |

**`status` distinguishes four genuinely different outcomes.** In particular it
separates *"proven no solution exists"* from *"the solver gave up"*, which a bare
`solvable=0` conflates:

| status | Meaning |
|---|---|
| `solvable` | A solution was found. |
| `unsolvable` | **Proven** — the search exhausted the entire push-state space. |
| `unknown_capped` | The solver hit `--max-states` and gave up; solvability is **unknown**. |
| `invalid` | Structurally malformed (bad piece counts / no player). |

Detection: `solve_push`'s loop is `while q and states < max_states`, so a capped run
ends with `states_expanded == max_states`, whereas a proven-unsolvable one exits with
the queue empty *below* the cap. At the default 200000 cap the capped rate is
measured at ~0%, so this label should stay empty in normal runs — it exists so a cap
hit can never silently masquerade as unsolvable.

The push count is the length of the solution measured in *box pushes* (not player
moves). Because `solve_push` is a breadth-first search over push-states, the first
solution it finds is **push-optimal**, so `pushes` is the shortest push-solution
length.

`perf.csv` summarizes each checkpoint: `step, num, solvable_pct, median_pushes,
mean_pushes, median_states, mean_states, capped_unknown_pct`, plus the three
wall-fix rates from §5.

---

## 5. Wall-Fix Analysis (Unsolvable Puzzles)

**Motivation.** Not all unsolvable puzzles are equally broken. A puzzle that
becomes solvable by removing a *single* wall is a **near-miss** — the model got
the structure almost right and slipped on one cell. One that needs many wall
removals is deeply broken. Measuring the *minimum number of interior walls to
remove to make an unsolvable puzzle solvable* quantifies **how close the model's
failures are to correct**, which a plain solvable/unsolvable label can't.

**We only care about 1- and 2-wall fixes; anything needing more is treated as
"really unsolvable."** Every proven-unsolvable puzzle is classified into exactly
one **category**, and we record **which wall(s)** achieve the fix:

| Category | Meaning |
|---|---|
| `1wall` | At least one single interior-wall removal makes it solvable. |
| `2wall` | No single wall works, but some pair of removals does. |
| `really_unsolvable` | Not fixable by removing 1 or 2 interior walls. |

**Method — per proven-unsolvable puzzle:**
1. **Candidate walls.** Consider *interior* wall cells (rows/cols 1–8) that have
   **at least one non-wall neighbour**. A wall whose four neighbours are all walls
   can never be reached by the player or a box, so removing it *provably* cannot
   change solvability — safe to skip. (A tighter reachability-based filter — "walls
   adjacent to the player-reachable region or a box" — is **not** safe: reachability
   changes as boxes move, so it can miss genuine fixes.)
2. **Removing a wall** = flipping that cell from `#` to floor (` `). Everything
   else (player, boxes, goals) is unchanged — piece counts stay valid.
3. **1-wall search.** Try removing each single candidate wall and re-run the push
   solver. **Collect *every* single wall whose removal makes the puzzle solvable.**
   If the set is non-empty → category `1wall`, and we keep all those wall
   positions (a puzzle may have several independent single-wall fixes).
4. **2-wall search** (only if no single wall worked). Try pairs of candidate walls
   until one makes it solvable → category `2wall`, and we keep that fixing pair
   (the first found — enumerating *all* pairs is expensive and not needed).
5. **Otherwise** → category `really_unsolvable` (needs >2 removals).
6. The push solver's verdict on each modified grid is trustworthy (100% on
   guaranteed-solvable training data), so "solvable after removing wall X" is a
   real proof, not a solver timeout.

**Coverage and cost.** This runs on **every** proven-unsolvable puzzle — the rates
are exact, not estimates. `unknown_capped` puzzles (§4) are excluded: one might
actually be solvable, so "how many walls must be removed to fix it" is not a
well-posed question for it.

This is the **slowest phase of the pipeline** — the pair search is combinatorial,
and each trial solve runs on an *opened-up* grid, which is more expensive than the
original. Measured ~0.6s median per puzzle with a tail beyond 15s, so an early
checkpoint at `--num 2000` (~58% unsolvable) costs roughly 10–20 min. Control total
cost with `--num` and `--every`.

**Outputs:**
- `wallfix.csv` — **one row per fixing solution** (so a `1wall` puzzle with three
  independent single-wall fixes produces three rows; a `2wall` puzzle produces one
  row for its fixing pair; a `really_unsolvable` puzzle produces one row with no
  walls):

  | Column | Meaning |
  |---|---|
  | `sample_id` | Joins `solvability.csv` / `samples.txt`. |
  | `category` | `1wall`, `2wall`, or `really_unsolvable`. |
  | `w1_row`, `w1_col` | First removed wall (−1 if none, i.e. `really_unsolvable`). |
  | `w2_row`, `w2_col` | Second removed wall (−1 for `1wall` / `really_unsolvable`). |

- `perf.csv` gains three columns (exact fractions of the proven-unsolvable set):
  `wallfix_1wall_pct`, `wallfix_2wall_pct`, `wallfix_really_unsolvable_pct`.

**Interpretation.** A high `1wall` fraction means the model's failures are
overwhelmingly one-wall near-misses — it has the global structure right and
mis-places single cells. Because `wallfix.csv` carries the actual wall
coordinates, each near-miss is directly **inspectable and repairable**: you can
load the puzzle from `samples.txt`, flip the recorded `#` to floor, and it
solves. Watching the `1wall` fraction *rise across training* shows the model's
mistakes getting **shallower** even while the raw solvable rate plateaus — a
signal it is increasingly "understanding" solvability. (Empirically, on the
T=1000 baseline, ~89% of unsolvable puzzles were single-wall near-misses.)

---

## 6. Is Push-Move Count a Good Difficulty Metric?

**Short answer: it's a reasonable, cheap first-order signal, but a *weaker*
proxy than search effort. Record it, but don't rely on it alone.**

**Why it has some merit:**
- Free (falls out of the solver) and intuitive — a 30-push puzzle generally
  involves more manipulation than a 5-push one.
- Well-defined (push-optimal, §4) and monotone-ish with "amount of work," so it
  correlates *somewhat* with difficulty.
- Better than shallow structural features (box count, wall count), which barely
  track difficulty.

**Why it is limited — and under-measures Sokoban difficulty:**
- **It measures the solution, not the search.** Sokoban is hard because of
  dead-ends and deadlocks — wrong pushes that trap a box irrecoverably. A puzzle
  can have a *short* optimal solution that is very hard to *find*; push count is
  blind to that.
- **The literature favors search effort.** Jarušek & Pelánek and the
  "search-statistics as difficulty" work find that states **expanded** (≈ A*
  closed-list length) correlate best with human difficulty and solve time — far
  better than solution length. The solver exposes this as `states_expanded`.
- **Weak discrimination.** Two puzzles with identical push counts can differ
  wildly in hardness.

**Recommendation:** treat `pushes` as a lightweight *secondary* descriptor and
use **`states_expanded` (search effort) as the primary difficulty metric.** Both
come out of a single `solve_push(..., return_stats=True)` call, so recording both
is free and lets them be cross-checked.

---

## 7. Implementation

Implemented in **`source/eval/run_eval.py`**. It composes existing pieces and
adds the new instrumented sampler:
- **Instrumented generation** — `sample_instrumented()` mirrors
  `MaskedDiffusion.sample` (random reveal) but additionally records the
  per-iteration softmax probabilities, each token's `revealed_at` step, and its
  commit/max probability. This is the only genuinely new piece; `diffusion.py` is
  left untouched.
- **Solver with push count + search effort** —
  `test_solver.solve_push(..., return_stats=True)` (`pushes`, `states_expanded`).
- **Shared code is imported, not copied.** `config`, `model`, `diffusion`,
  `dataset` (from `source/diffusion`), `test_solver`/`generate` (from `source`)
  are imported via `sys.path` so the model architecture can never drift from the
  training code that produced the weights.
- **Pipelined** — each generated chunk's puzzles are submitted to the worker pool
  immediately, so CPU solving of chunk *n* overlaps GPU generation of chunk *n+1*.
  Per-puzzle submission also gives dynamic scheduling, which matters because solve
  cost is heavily skewed (p50 ~86ms, max ~4s).

Run (from `source/eval`):
```
python run_eval.py --checkpoint-dir ../diffusion/checkpoints_T100 \
    --num-timesteps 100 --every 5 --num 1000
```
