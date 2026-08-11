# Sokoban Solvability Check: Move-Based vs Push-Based

The solvability metric (fraction of generated puzzles that can actually be solved)
is only as trustworthy as the solver behind it. This project has two solvers in
`source/test_solver.py`:

- `solve_bfs` — the original **move-based** BFS (kept for reference; no longer used
  for metrics)
- `solve_push` — the **push-based** solver (now used everywhere solvability is measured)

They can return *different answers for the same puzzle*, and the difference was
large enough to distort the entire training story — so this file explains why.

---

## TL;DR

| | Move-based BFS (`solve_bfs`) | Push-based (`solve_push`) |
|---|---|---|
| Branches on | every **player step** (U/D/L/R) | every legal **box push** |
| Branching factor | 4 per step, most of them useless walking | ~ (#boxes × 4), all meaningful |
| State space | player position × box layout — **huge** | box layout only (player position canonicalized) — **small** |
| Deadlock pruning | corner-only, per-node | precomputed **static dead-square table** |
| On real (all-solvable) training puzzles @ 200k-state cap | **80.5% solved** ❌ | **100% solved** ✅ |
| Effect on the metric | ~20 pt undercount — false "unsolvable" labels | trustworthy |

The move-based solver was reporting ~20 points *lower* solvability than reality,
because it ran out of its state budget before finding solutions to hard-but-solvable
puzzles. Switching to push-based lifted the measured solvability of the model's own
generations from ~57% to ~71% at the same checkpoint — **the "wall at 57%" was
mostly the solver, not the model.**

---

## Background: what a Sokoban state is

A puzzle state is fully described by **where the boxes are** plus **where the player
is**. The player moves one cell at a time (U/D/L/R); walking into a box *pushes* it
one cell, but only if the cell behind the box is empty. The puzzle is solved when
every box sits on a goal. The two solvers differ entirely in **what they treat as a
"move" in the search.**

---

## Move-based BFS (`solve_bfs`) — the naive approach

At every node it tries all four **player moves** and enqueues each resulting state:

```
for direction in [Up, Down, Left, Right]:
    if the player can move that way (into floor, or push a box):
        enqueue the new state
```

Why this is weak for Sokoban:

1. **The branching factor is dominated by useless walking.** The vast majority of the
   4 moves at any node just relocate the player without pushing anything. The search
   spends almost all of its effort exploring *where the player is standing*, which is
   irrelevant to whether the puzzle gets solved — only pushes change the puzzle.
2. **The state space is enormous.** It's `(player positions) × (box layouts)`. Two
   states that have identical box layouts but the player standing on different empty
   squares are treated as *different* nodes, even though they're strategically
   identical (the player can freely walk between them).
3. **It hits the `max_states` cap.** Because it wastes its budget on walking, it
   frequently exhausts the 200,000-state limit before reaching a solution — and then
   reports the puzzle as **unsolvable**, even when a solution exists. This is a
   *false negative*, and it's why it failed on 19.5% of guaranteed-solvable training
   puzzles.

Deadlock handling is also minimal — only a per-node corner check.

---

## Push-based solver (`solve_push`) — the standard approach

The key insight: **the player's exact position doesn't matter — only which boxes it
can reach and push.** So the search branches on *pushes*, not steps.

At each node:

1. **Flood-fill the player's reachable region** (`_player_reachable`): all empty
   squares the player can walk to from its current position, blocked by walls and
   boxes. Walking within this region is "free" and never branched on.
2. **Enumerate legal pushes.** For each box and each direction, a push is legal iff
   the player can reach the square on the opposite side (it's in the reachable
   region) and the box's destination square is empty. The only branches are these
   pushes — at most `#boxes × 4`.
3. **Canonicalize the state** as `(frozenset(box positions), min(reachable region))`.
   Using a single representative cell for the whole reachable region collapses all
   the "player standing on a different empty square" duplicates into one node. This
   is the biggest state-space reduction.
4. **Prune dead squares** using a precomputed static table (see below).

Because it never branches on walking and merges player-equivalent states, the search
tree is *orders of magnitude* smaller — it solves the same puzzles well within the
state cap (in practice it essentially never hits it), which is why it recovers the
full 100% on solvable training data.

### Static deadlock table (`_compute_live_squares`)

Computed **once per puzzle**, before search. A "live" square is one from which a box
can still be maneuvered onto *some* goal. It's found by reverse-reachability: start
at every goal and repeatedly "pull" a box outward one cell (the reverse of a push),
marking each square reachable that way — a pull from `s` to `s+d` is valid only if
both `s+d` and the square beyond it (where the puller would stand) are non-wall.

Any square **not** marked live is a *simple deadlock square*: a box pushed there can
never reach any goal, no matter what. During search, any push whose destination is a
dead square is discarded immediately. This prunes whole unsolvable subtrees cheaply —
most importantly, it lets the solver **prove a puzzle unsolvable fast** instead of
grinding to the state cap, so "unsolvable" answers become trustworthy too, not just
"solvable" ones.

### Path reconstruction

`solve_push` still returns a real `U/D/L/R` move string (not just yes/no): after
finding the winning push sequence, `_reconstruct_moves` stitches it back into player
moves by BFS-pathfinding the player between consecutive push positions. This keeps it
a drop-in replacement for `solve_bfs` — GIF rendering and anything else that replays
the path still works. Reconstruction was validated by replaying the output through
the independent `can_move` and confirming it reaches a solved state.

---

## Why it matters here

The generator's quality was being judged against a solver that couldn't even verify
1 in 5 *guaranteed-solvable* puzzles. That:

- **Undercounted solvability** by ~13 pt on generated puzzles (57.4% → 70.9% at
  step 90k) and ~20 pt on training data (80.5% → 100%).
- **Manufactured a fake plateau.** Time spent chasing the "~57% wall" with LR fixes
  and longer training was partly optimizing against a broken measurement that
  literally could not report above ~80%.
- **Conflated two failure modes.** The old "valid but unsolvable" bucket mixed
  genuinely-unsolvable puzzles with solvable-but-hard ones the solver timed out on.
  The push solver separates them: what it now reports as unsolvable is (almost
  entirely) genuinely unsolvable, which is the real, remaining model gap worth
  attacking.

**Consequence for reading `perf.txt`:** entries at `step ≤ 95000` were logged with
the old move-based solver and are undercounts; `step ≥ 100000` (and everything under
`samples_2/`) uses `solve_push` and is trustworthy. Don't compare across that
boundary directly.
