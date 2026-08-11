# Sokoban by masked diffusion

Code for *Masked Diffusion Generates Solvable Sokoban Puzzles Without
Solvability Supervision* — a 4.9M-parameter masked discrete diffusion model
that generates Sokoban puzzles from a per-cell tile-completion objective, with
no solver, reward, or solvability label in the training loop.

- Write-up: https://sinabaghal.github.io/sokoban/
- Playable demo and trained model: https://github.com/sinabaghal/SokobanPlayground

## Layout

| folder | what it holds |
|---|---|
| `diffusion/` | model, noise schedule, dataset, training loop, sampler |
| `solver/` | push-based solver used to decide solvability, and its tests |
| `metrics/` | solvability and wall-fix evaluation, tile-pattern JSD, nearest-neighbour Hamming distance, difficulty |
| `eval/` | evaluation drivers and the temperature sweep |
| `viz/` | every figure and animation in the write-up |
| `viz/csv/` | the measured numbers behind each figure |

`viz/csv/` is the part worth reading first if you want to check a number: each
figure script reads its CSV and does no measurement of its own, so editing a
label cannot change a result.

## Data

Training data is DeepMind's [Boxoban](https://github.com/deepmind/boxoban-levels)
`medium/train` split (450,000 puzzles); the held-out split used as the
distribution and memorization reference is its `medium/valid` (50,000).
Neither is vendored here.
