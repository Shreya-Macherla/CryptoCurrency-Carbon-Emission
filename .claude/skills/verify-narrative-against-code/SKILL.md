---
name: verify-narrative-against-code
description: Load before editing README.md, any *_analysis.py chart-generation script, or any claim about model performance/findings in this repo. Trigger on keywords — accuracy, forecasting accuracy, NARX, NARMAX, LSTM, connectedness, simulated, PLOS ONE, DOI, publication claim.
---

# This repo's headline chart is simulated, not computed — but is described as real

`summary_analysis.py`'s own docstring is explicit and self-contradictory:

```
Generates summary charts from simulated model outputs (actual results from paper).
```

The script never loads CoinMarketCap prices, EU ETS carbon data, or any real time series — it
generates numbers via `np.random` calibrated to *look like* plausible NARX/NARMAX/LSTM output,
then plots them. But the README states this as fact, not as an illustration:

> **Finding:** Sustainable crypto-assets (ADA, XNO, POWR) show measurably lower CO₂ connectedness
> than conventional assets (BTC, ETH). NARMAX (MISO architecture) achieves 89–95% forecasting
> accuracy, outperforming SARIMAX baselines.

This repo is presented as the code behind a real, DOI-cited PLOS ONE 2025 publication
(`10.1371/journal.pone.0318647`). If the 89–95% figure and the ADA/XNO/POWR-vs-BTC/ETH finding
are the actual published results, they need to come from a script that actually reads the paper's
data and reproduces the paper's numbers — not from `np.random.default_rng(42)`. If they're
illustrative only, the README must say so as plainly as the "Finding:" line currently claims
fact. **This is the highest-stakes case in your GitHub portfolio for this exact pattern** — an
interviewer or reviewer who reads `summary_analysis.py` after reading the README's "Finding"
claim will see the numbers were never computed from real data, in a repo that cites a real DOI.

## Before touching this repo again

1. Locate the actual notebook/script (if one exists) that produced the real published PLOS ONE
   results. If it doesn't exist in this repo, that's the real problem to fix — add it, or
   visibly separate "illustrative demo" from "published result" everywhere the README implies
   otherwise.
2. Do not add any new headline number to the README without tracing it to either (a) the
   published paper directly, cited with a page/table reference, or (b) an explicit "simulated
   for illustration" label immediately next to the number.
3. If asked to update `summary_analysis.py`, do not add more realistic-looking random data — that
   makes the problem harder to detect, not better.

## When NOT to use this skill

This repo currently has no other application code to speak of (notebooks + one plotting script,
single-commit git history) — there's no separate technical failure mode to extract. This is the
one thing worth fixing here.
