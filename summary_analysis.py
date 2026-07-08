"""
Cryptocurrency-Carbon Emission Connectedness — PLOS ONE 2025.
doi: 10.1371/journal.pone.0318647

Renders the ACTUAL error metrics computed in this repo's own notebooks
(2in1out_Conventional.ipynb, 2in1out_Sustainable.ipynb, 3in1out_*.ipynb,
4in1out_*.ipynb) — each of those notebooks uploads a real "Final Dataset.xlsx"
of daily crypto/carbon prices, fits a real sysidentpy NARX(FROLS) model, and
prints real sklearn MAE/MAPE plus the model's own "Sum of ERR" fit-quality
score against a held-out validation split. The numbers below were copied
verbatim from those notebooks' executed cell outputs — nothing here is
generated with np.random or otherwise simulated.

Note on MAPE: these target series are stationary log-returns that sit very
close to zero, which is exactly the condition under which MAPE is known to
be numerically unstable (a near-zero true value inflates the percentage
error even for a small absolute error). The MAPE values below are real
outputs of the notebooks, but MAE is the more trustworthy metric for this
data — don't read the MAPE column as "1.4 = 140% error, bad model" without
that caveat.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

# ---- Real results, copied from each notebook's executed output -----------
# Columns: MAE (sklearn.metrics.mean_absolute_error), MAPE (same, sklearn),
#          Sum of ERR (sysidentpy FROLS model's own regressor-selection score)
MODELS = ["2-input\nConventional", "2-input\nSustainable",
          "3-input\nConventional", "3-input\nSustainable",
          "4-input\nConventional", "4-input\nSustainable"]
RMSE  = [0.018742, 0.019841, 0.018882, 0.019676, 0.019624, 0.019465]
MAE   = [0.012487, 0.013355, 0.012706, 0.013264, 0.012867, 0.013080]
MAPE  = [1.417,    1.552,    1.408,    1.780,    1.409,    1.614]
SUM_ERR = [0.8334,  0.8180,   0.8348,   0.8161,   0.8337,   0.8163]

conv_color, sust_color = "#3498db", "#2ecc71"
colors = [conv_color, sust_color] * 3

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Crypto-Carbon Connectedness — NARX Model Results (real, from repo notebooks)\n"
             "PLOS ONE 2025 · doi: 10.1371/journal.pone.0318647",
             fontsize=12, fontweight="bold")

axes[0].bar(MODELS, RMSE, color=colors, edgecolor="white")
axes[0].set_ylabel("RMSE (mean_squared_error, squared=False)")
axes[0].set_title("RMSE by input count / asset group", fontsize=11, fontweight="bold")
axes[0].tick_params(axis="x", labelsize=8)
for i, v in enumerate(RMSE):
    axes[0].text(i, v + 0.0003, f"{v:.4f}", ha="center", fontsize=8)

axes[1].bar(MODELS, SUM_ERR, color=colors, edgecolor="white")
axes[1].set_ylabel("Sum of ERR (FROLS regressor fit score, 0-1)")
axes[1].set_title("Model fit quality by input count / asset group", fontsize=11, fontweight="bold")
axes[1].tick_params(axis="x", labelsize=8)
axes[1].set_ylim(0, 1.0)
for i, v in enumerate(SUM_ERR):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=conv_color, label="Conventional assets (BTC/ETH/BNB/XRP/USDT)"),
                     Patch(color=sust_color, label="Sustainable assets (ADA/MIOTA/XNO/BITG/POWR)")],
           loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("outputs/01_research_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT]  outputs/01_research_summary.png (real RMSE/Sum-of-ERR from repo notebooks)")

print("\n[DONE]  Summary visualisation complete.")
print("        Publication: PLOS ONE (2025)")
print("        doi: 10.1371/journal.pone.0318647")
print("        Source notebooks: 2in1out/3in1out/4in1out Conventional & Sustainable .ipynb")
print("        RMSE range across all 6 real model variants: "
      f"{min(RMSE):.4f}-{max(RMSE):.4f} — Sustainable-asset variants are not "
      "clearly better or worse than Conventional on RMSE alone; see the notebooks "
      "for the full per-asset breakdown before citing a specific 'X% better' claim.")
