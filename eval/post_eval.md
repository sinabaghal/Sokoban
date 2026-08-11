# Post-Eval Analysis (`post_eval.md`)

Ideas for turning the `run_eval.py` outputs into an *understanding* of how the
masked-diffusion model actually generates puzzles — where its uncertainty lives,
whether it reasons about global constraints, and why it fails.

This is an analysis roadmap, not a spec. See `eval.md` for what the data is.

---

## 1. The data and how it joins

Per checkpoint, under `output/ckp_<step>/`:

| File | Grain | Key |
|---|---|---|
| `prob_trace.parquet` | (sample, iteration, cell) × 7 class probs | `sample_id, iteration, row_id, col_id` |
| `reveal.csv` | (sample, cell) commit record | `sample_id, row_id, col_id` |
| `solvability.csv` | (sample) status/pushes/effort | `sample_id` |
| `wallfix.csv` | (sample, fixing wall) | `sample_id, w1_row, w1_col` |
| `samples.txt` | (sample) the grid + status | `; <id> <status>` |

`reveal.csv` is the workhorse — it carries `revealed_at`, `committed_class`,
`commit_prob`, `max_prob` for **every** cell of **every** sample, and joins
everything else. Reach for `prob_trace.parquet` (50M rows/checkpoint) only when you
need the *trajectory* before commitment; filter by `sample_id` first.

---

## 2. Methodology rules (learned the hard way)

Violating any of these produces confident nonsense. All four have already bitten.

1. **Always separate border from interior cells.** The outer ring (`row/col ∈ {0,9}`,
   36 of 100 cells) is **100% wall, committed at p = 1.0000, 0% below 0.7** — fixed
   structure the model has memorized. Including it in any aggregate inflates
   confidence and shrinks apparent error rates. *Every* wall statistic must be
   interior-only or stratified. Consider stratifying interior further by
   distance-to-border: a wall at row 1 is probably far more predictable than one in
   the open middle.
2. **Every comparison needs a baseline, or there is no null.** "Culprit walls have
   confidence 0.50" is meaningless until you know innocent interior walls sit at
   0.92–0.96. Prefer *within-puzzle* baselines (other cells of the same puzzle) —
   they control for puzzle-level difficulty automatically.
3. **Reveal order is uniformly random by construction** (`strategy='random'` picks a
   random masked cell each step), so `revealed_at` is independent of cell content.
   This makes it a *free control variable*, not an outcome: if two groups have
   matched reveal steps, any confidence difference between them is **not** explained
   by how much context was available. It also means "does the model draw walls
   before boxes?" is **not** a question about the model under random reveal — the
   sampler chooses the order, not the model. That question only becomes meaningful
   under confidence-based reveal.
4. **`wallfix` identifies participation, not causation.** "Removing wall X makes it
   solvable" means X *participates* in the deadlock. The actual mistake may have
   been a box committed 40 steps later. Phrase conclusions accordingly.

---

## 3. What we already know (checkpoint 255000, interior-only)

| group | n | reveal med | commit_prob med | <0.7 |
|---|---|---|---|---|
| culprit walls | 301 | 51 | **0.502** | 76.1% |
| other interior walls, same puzzles | 2,906 | 49 | 0.957 | 24.7% |
| interior walls, solvable puzzles | 124,728 | 0.924 | 0.924 | 28.1% |

- **Confidence is a strong signal; timing is not.** Reveal steps match (51 vs 49) —
  expected, since reveal order is random — which conveniently means the confidence
  gap is *not* a context artifact.
- **~39% of failures are sampler slips, not ignorance.** At the culprit cell the mean
  belief was `p_wall 0.509 / p_floor 0.353`; the model's top choice was `floor` — the
  exact fix — in 35.5% of cases, and `p_floor > p_wall` in 39.2%. It often *knew*,
  and lost the multinomial draw. Consistent with the temperature sweep, where
  T=0.7 bought +3.6pp solvability for free.
- **As a standalone detector, confidence is weak**: `commit_prob<0.7` catches 76% of
  culprits but flags 24.7% of innocent interior walls. Since innocent walls outnumber
  culprits ~10:1, most flags are false positives.

---

## 4. Analysis ideas

Ordered roughly by (value / effort).

### A. Where does uncertainty live? (`reveal.csv` only — cheap)
1. **Confidence by tile class.** Median `commit_prob` for wall / floor / box / goal.
   Which categories is the model sure about? Expect boxes and goals (the semantically
   loaded, sparse classes) to be far less certain than floor.
2. **Confidence vs distance-to-border.** Generalizes the border insight: plot median
   `commit_prob` against `min(row, col, 9-row, 9-col)`. Quantifies how much of the
   model's apparent skill is just reproducing the frame and its neighbourhood.
3. **Confidence vs reveal step.** Does more context help? A rising curve means the
   model genuinely conditions on what's already committed; a flat one means it is
   largely drawing from a context-free prior — a sharp test of whether "diffusion" is
   doing real iterative refinement here.
4. **Ignorance vs sampler slip, by class.** Rate of `max_prob − commit_prob > ε`.
   Tells you how much headroom temperature tuning has, per tile type.

### B. Does the model do constraint propagation? (`prob_trace.parquet` — the deep question)
5. **Neighbour response to a commitment.** When a wall is committed at cell *c* at
   iteration *t*, how do the beliefs at the 4 neighbours change from *t* to *t+1*?
   A real world-model should *react* — e.g. p_box should drop next to a fresh corner.
   Flat neighbour beliefs would mean the model is largely position-conditioned rather
   than context-conditioned. **This is the single most informative experiment here**:
   it directly tests whether the model reasons about interactions or just paints a
   plausible texture.
6. **Belief trajectory before commitment.** For each cell, plot `p_committed_class`
   over iterations `0 … revealed_at`. Does the model converge gradually (evidence
   accumulating) or stay flat then jump (no real refinement)?
7. **Early-warning signal.** Compare that pre-commit trajectory for culprit vs
   innocent cells. If culprits are already ambiguous 20 iterations before commitment,
   uncertainty is *structural*, not a last-moment coin-flip — and a look-ahead
   intervention becomes possible.
8. **Does global solvability information exist anywhere?** Train a linear probe on the
   model's hidden states (or, cheaply, on the full 7-class belief vector across all
   100 cells at iteration *k*) to predict whether the finished puzzle will be
   solvable. If a probe succeeds at, say, iteration 50, the model *represents*
   solvability even though it never optimises for it. This is the most direct answer
   to "does it inherently understand solvability."

### C. Calibration
9. **Reliability diagram.** Bin cells by `commit_prob`; within each bin, measure how
   often that commitment ended up in a solvable puzzle (or wasn't a culprit). If the
   model is well-calibrated, a stated 0.7 should mean ~70%. Miscalibration would mean
   confidence-based repair needs recalibrating before it can be trusted.
10. **Does calibration improve with training?** Run 1–9 across all 6 checkpoints. The
    interesting hypothesis: raw solvability plateaus (77%) while *calibration* keeps
    improving — which would say late training is spent learning what it doesn't know.

### D. Interventions (turn understanding into solvability)
11. **Confidence-guided repair.** For each unsolvable puzzle, flip the lowest-confidence
    interior wall to floor and re-solve. Measure the lift. This needs no solver at
    selection time and directly exploits findings §3. Compare to the ceiling: 96% of
    failures are 1-wall fixable, so a perfect selector would reach ~99%.
12. **Remasking / predictor-corrector.** Re-mask the k lowest-confidence cells after
    generation and re-predict with full context. Tests whether the model can *fix
    itself* given a second look — a strong statement about whether the failure is in
    the model or in the one-shot sampling procedure.
13. **Temperature × confidence.** Re-run the temperature sweep while recording
    `commit_prob`. Does lower temperature fix precisely the 39% sampler-slip cases,
    leaving the genuine-ignorance ones untouched? That would cleanly decompose the
    +3.6pp temperature gain into its mechanism.

---

## 5. Suggested order

1. **§A1–A3** — an afternoon on `reveal.csv`, no heavy compute, and it reframes
   everything else (especially A3: does context matter at all?).
2. **§B5** — the constraint-propagation test. The highest-information single result
   about whether this model has a world-model or a texture prior.
3. **§D11** — cheap, and converts the confidence finding into a real solvability gain.
4. **§B8 / §C10** — the research-grade questions, once the groundwork above says
   which are worth the compute.
