# Figure data

Every figure's numbers, extracted once so the plot scripts never touch the heavy
sources. Written by `../make_figure_data.py`.

| File | Feeds | Columns |
|---|---|---|
| `solvability.csv` | `solvability_plot.py` | `step, n_samples, solvable_pct, wallfix_1wall_pct, effective_pct, solvable_ci95, effective_ci95` |
| `culprit_confidence.csv` | `culprit_confidence.py` | `bin_lo, bin_hi, culprit_pct, other_pct` |
| `culprit_summary.csv` | `culprit_confidence.py` | `group, n, median_p, pct_below_0.7, n_levels` |
| `commit_cells.csv` | `commit_heatmap.py`, `commit_heatmap_gif.py` | `sample_id, row, col, revealed_at, committed_class, commit_prob, solvable` |
| `commit_curve.csv` | `commit_heatmap.py` | `group, reveal_step, n, median_p` |

## The point of this split

`reveal.csv` is 500,000 rows and `prob_trace.parquet` is 271 MB. Reading either
takes tens of seconds, which makes iterating on a title or a caption
disproportionately slow — and, worse, means a presentation change re-runs a data
computation that could silently differ. Extracting once fixes both: the plot
scripts read a few hundred rows, re-render in about a second, and **cannot**
change a number.

```bash
python make_figure_data.py              # rebuild all five CSVs
python make_figure_data.py --only commit
python solvability_plot.py              # ~1s, reads only its CSV
```

## Editing titles and captions

Each plot script has an **`# ---- editable text ----`** block just below its
imports holding every string it draws — title, subtitle, axis labels, series
labels, inline notes. Change those and re-run the script; nothing else needs
touching, and the data is untouched by construction.

Where a string needs a number in it, it is a `str.format` template filled from
the CSV (`'... (n={n})'`), so counts stay correct when the data is regenerated
against a different checkpoint.

## Provenance

All five derive from `source/eval/output/ckp_255000/` — the checkpoint the
paper's numbers come from — except `solvability.csv`, which spans every evaluated
checkpoint in `source/eval/output/perf.csv`. Point `make_figure_data.py --ckp` at
another `ckp_*` directory to rebuild against a different one.
