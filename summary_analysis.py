"""
Cryptocurrency-Carbon Emission Connectedness — PLOS ONE 2025.
doi: 10.1371/journal.pone.0318647

Renders the ACTUAL error metrics computed in this repo's own notebooks.
Two figures:

  outputs/01_research_summary.png   — the six NARX(FROLS) variants
      (2/3/4-input x Conventional/Sustainable), numbers copied verbatim from
      2in1out_*.ipynb / 3in1out_*.ipynb / 4in1out_*.ipynb executed cells.

  outputs/02_model_comparison.png   — NARX vs LSTM vs SARIMAX cross-family
      comparison. LSTM and SARIMAX numbers copied verbatim from the executed
      outputs of MISO_Conventional_LSTM_.ipynb, MISO_Sustainable_LSTM.ipynb,
      MISO_Conventional_SARIMAX.ipynb, MISO_Sustainable_SARIMAX_.ipynb.

Each source notebook uploads a real "Final Dataset.xlsx" of daily
crypto trading volumes and power-sector CO2 emissions (log-return series), fits a real model (sysidentpy NARX(FROLS), Keras
stacked LSTM, or statsmodels SARIMAX), and prints real sklearn RMSE/MAE/MAPE
against a held-out chronological split. Nothing here is generated with
np.random or otherwise simulated.

Note on MAPE: these target series are stationary log-returns that sit very
close to zero, which is exactly the condition under which MAPE is known to
be numerically unstable (a near-zero true value inflates the percentage
error even for a small absolute error). The MAPE values are real outputs of
the notebooks, but RMSE/MAE are the trustworthy metrics for this data.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.makedirs("outputs", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

conv_color, sust_color = "#3498db", "#2ecc71"

# ============================================================================
# Figure 1 — NARX variants (unchanged; copied from the six NARX notebooks)
# ============================================================================
MODELS = ["2-input\nConventional", "2-input\nSustainable",
          "3-input\nConventional", "3-input\nSustainable",
          "4-input\nConventional", "4-input\nSustainable"]
RMSE  = [0.018742, 0.019841, 0.018882, 0.019676, 0.019624, 0.019465]
MAE   = [0.012487, 0.013355, 0.012706, 0.013264, 0.012867, 0.013080]
MAPE  = [1.417,    1.552,    1.408,    1.780,    1.409,    1.614]
SUM_ERR = [0.8334,  0.8180,   0.8348,   0.8161,   0.8337,   0.8163]

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

fig.legend(handles=[Patch(color=conv_color, label="Conventional assets (BTC/ETH/BNB/XRP/USDT)"),
                    Patch(color=sust_color, label="Sustainable assets (ADA/MIOTA/XNO/BITG/POWR)")],
           loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("outputs/01_research_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT]  outputs/01_research_summary.png (real RMSE/Sum-of-ERR from repo notebooks)")

# ============================================================================
# Figure 2 — Cross-family comparison: NARX vs LSTM vs SARIMAX
# All numbers copied verbatim from executed cells of the named notebooks.
# ============================================================================
# NARX: best variant per asset group from the table above
#   Conventional best = 2-input (RMSE 0.018742, MAE 0.012487)
#   Sustainable  best = 4-input (RMSE 0.019465, MAE 0.013080)
# LSTM (stacked 350->300->250->200->150 units, Dropout 0.3, Dense(1); 85/15 split):
#   Conventional — MISO_Conventional_LSTM_.ipynb
#   Sustainable  — MISO_Sustainable_LSTM.ipynb
# SARIMAX (grid-searched; Conventional (0,0,2)x(1,0,2,12),
#          Sustainable (1,1,2)x(0,0,1,12); crypto series as exogenous inputs):
#   Conventional — MISO_Conventional_SARIMAX.ipynb
#   Sustainable  — MISO_Sustainable_SARIMAX_.ipynb

COMPARE = {
    #  family:        (Conventional RMSE, MAE),      (Sustainable RMSE, MAE)
    "NARX (FROLS)\nbest variant": ((0.018742, 0.012487), (0.019465, 0.013080)),
    "LSTM\n(stacked, Keras)":     ((0.022840, 0.015273), (0.020631, 0.014684)),
    "SARIMAX\n(exog inputs)":     ((0.025230, 0.018708), (0.026881, 0.020552)),
}

families = list(COMPARE.keys())
x = range(len(families))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("NARX vs LSTM vs SARIMAX — same target (power-sector CO2 emissions), period, chronological split, sklearn metrics\n"
             "All values copied from each notebook's own executed output",
             fontsize=12, fontweight="bold")

for ax, metric_idx, label in ((axes[0], 0, "RMSE"), (axes[1], 1, "MAE")):
    conv_vals = [COMPARE[f][0][metric_idx] for f in families]
    sust_vals = [COMPARE[f][1][metric_idx] for f in families]
    ax.bar([i - width / 2 for i in x], conv_vals, width, color=conv_color,
           edgecolor="white", label="Conventional")
    ax.bar([i + width / 2 for i in x], sust_vals, width, color=sust_color,
           edgecolor="white", label="Sustainable")
    ax.set_xticks(list(x)); ax.set_xticklabels(families, fontsize=9)
    ax.set_ylabel(label)
    ax.set_title(f"{label} by model family", fontsize=11, fontweight="bold")
    for i, v in enumerate(conv_vals):
        ax.text(i - width / 2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=7.5)
    for i, v in enumerate(sust_vals):
        ax.text(i + width / 2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=7.5)

axes[0].legend(fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("outputs/02_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT]  outputs/02_model_comparison.png (NARX vs LSTM vs SARIMAX, real numbers)")

# ============================================================================
print("\n[DONE]  Summary visualisation complete.")
print("        Publication: PLOS ONE (2025)")
print("        doi: 10.1371/journal.pone.0318647")
print("        NARX source notebooks: 2in1out/3in1out/4in1out Conventional & Sustainable .ipynb")
print("        LSTM source notebooks: MISO_Conventional_LSTM_.ipynb, MISO_Sustainable_LSTM.ipynb")
print("        SARIMAX source notebooks: MISO_Conventional_SARIMAX.ipynb, MISO_Sustainable_SARIMAX_.ipynb")
print(f"        NARX RMSE range (6 variants): {min(RMSE):.4f}-{max(RMSE):.4f}")
print("        Cross-family result: NARX(FROLS) achieves the lowest RMSE and MAE in BOTH")
print("        asset groups — lower than LSTM (RMSE 0.0206-0.0228) and SARIMAX")
print("        (RMSE 0.0252-0.0269) under the same protocol. Within NARX alone,")
print("        Sustainable-asset variants are not clearly better or worse than")
print("        Conventional on RMSE; see the notebooks for per-asset breakdowns.")
