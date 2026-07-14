---
name: verify-narrative-against-code
description: Load before editing README.md, summary_analysis.py, any notebook, or any claim about model performance/findings in this repo. Trigger on keywords — accuracy, RMSE, MAE, MAPE, NARX, NARMAX, LSTM, SARIMAX, connectedness, simulated, PLOS ONE, DOI, publication claim, outperforms.
---

# Every number in this repo must trace to an executed notebook cell

## History (resolved — do not re-fix)

This repo previously had a serious narrative/code mismatch: `summary_analysis.py`
generated charts from `np.random` data while the README presented those numbers
("89–95% forecasting accuracy", a sustainable-vs-conventional connectedness
"Finding") as real published results. **That has been fully repaired.** Do not
assume the old problem still exists, and do not reintroduce the old numbers —
"89–95% accuracy" and "98.76%" style claims were removed because they traced to
nothing.

## Current invariant (protect this)

Every metric in `README.md` and `summary_analysis.py` is copied verbatim from an
executed cell output of a named notebook in this repo. The verified table:

| Model | RMSE | MAE | Source notebook |
|---|---|---|---|
| NARX 2-in Conventional | 0.018742 | 0.012487 | `2in1out_Conventional.ipynb` |
| NARX 2-in Sustainable | 0.019841 | 0.013355 | `2in1out_Sustainable.ipynb` |
| NARX 3-in Conventional | 0.018882 | 0.012706 | `3in1out_Conventional.ipynb` |
| NARX 3-in Sustainable | 0.019676 | 0.013264 | `3in1out_Sustainable.ipynb` |
| NARX 4-in Conventional | 0.019624 | 0.012867 | `4in1out_Conventional.ipynb` |
| NARX 4-in Sustainable | 0.019465 | 0.013080 | `4in1out_Sustainable.ipynb` |
| LSTM Conventional | 0.022840 | 0.015273 | `MISO_Conventional_LSTM_.ipynb` |
| LSTM Sustainable | 0.020631 | 0.014684 | `MISO_Sustainable_LSTM.ipynb` |
| SARIMAX Conventional | 0.025230 | 0.018708 | `MISO_Conventional_SARIMAX.ipynb` |
| SARIMAX Sustainable | 0.026881 | 0.020552 | `MISO_Sustainable_SARIMAX_.ipynb` |

Citable claims (verified against the published paper on 2026-07-10):
- **NARX(FROLS) achieves the lowest RMSE and MAE in both asset groups**, beating
  the LSTM comparison (paper Tables 6-7 + repo) and the SARIMAX baseline
  (repo extension only — SARIMAX is NOT in the paper; never attribute it there).
- **Conventional crypto-assets show slightly stronger connectedness with CO2
  emissions than sustainable ones** — citable TO THE PAPER via Sum of ERR
  (0.834 vs 0.819, full MISO models), not via RMSE.
Not citable: any percentage-accuracy framing, or any "X% better" figure not
computed in a notebook or printed in the paper.

Paper facts (verified from the published article — do not contradict):
- Title: "Nonlinear connectedness of conventional crypto-assets and sustainable
  crypto-assets with climate change: A complex systems modelling approach",
  PLoS ONE 20(2): e0318647, published 7 Feb 2025.
- Citation/author order: Khan MH, Macherla S, Anupam A (2025). Repo owner is
  2nd author (Formal analysis, Software, Visualization).
- Output variable: world POWER-SECTOR CO2 emissions (Carbon Monitor) — not EUA
  / EU ETS carbon price. Inputs: daily trading VOLUMES (USD, log-returns), not
  prices. Sources: CoinMarketCap / Yahoo Finance + Carbon Monitor.
- Funding: Cardiff Metropolitan University internal Impact RA Funding — NOT
  UKRI. Never re-add a UKRI badge or funding line to this repo.
- Dataset is published with the paper as S1 Data (xlsx).

## Rules when editing

1. New number → it must come from an executed cell of a committed notebook, or
   from the published paper with an explicit page/table citation. Run first,
   claim second. Never hand-type a metric into an output cell.
2. `summary_analysis.py` must stay a *renderer of recorded results*, never a
   simulator. If asked to make charts "more complete", extend the hardcoded
   tables only with values traced to notebooks, and name the notebook in a
   comment next to each value.
3. MAPE on these near-zero log-return series is numerically unstable — never
   promote MAPE to a headline metric; RMSE/MAE only.
4. If a notebook is re-run and its metrics change, update README,
   `summary_analysis.py`, and this file in the same commit so all three agree.
5. Resume/career-database claims about this project must match this file's
   table (profile lives outside this repo; keep the wording in sync).

## Known open items (real, current)

- `NARMAX_MISO.ipynb`, `MISO_Conventional.ipynb`, `MISO_Sustainable.ipynb`, and
  the ten per-asset notebooks contain real results not yet summarised in the
  README table — safe to add, but only by the copy-from-executed-cell process.

(Resolved: `requirements.txt` now pins `sysidentpy==0.3.1` — the version the
executed notebooks' own pip output shows. Don't bump it without re-running the
notebooks.)

## When NOT to use this skill

Routine non-metric edits (typos, badges, structure lists, .gitignore) don't
need the full trace check — but any edit that adds, changes, or rephrases a
performance or findings claim does.
