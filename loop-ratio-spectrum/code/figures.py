"""Figure 3 (correlation heatmap) and Figure 5 (grounded Laplacian under pinning).

Both figures are written at a width of 6.5 inches with fonts sized for that
width, so they can be placed at full text width without shrinking the labels.

Requires correlation_table.csv and pinning_summary.csv at the repository root
(produced by compute_lrs.py).
Output: figures/fig3_correlation_heatmap.png, figures/fig5_pinning.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from paths import FIG, ROOT

if "Times New Roman" in {f.name for f in font_manager.fontManager.ttflist}:
    plt.rcParams["font.family"] = "Times New Roman"
else:
    plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"mathtext.fontset": "stix", "axes.linewidth": 0.8, "font.size": 7,
                     "axes.labelsize": 7, "axes.titlesize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5})
C = plt.get_cmap("tab10").colors

# ------------------------------------------------------------------ Figure 3
ct = pd.read_csv(os.path.join(ROOT, "correlation_table.csv"))
feats = ["rho_E", "rho_V", "mu_L", "H", "H_norm", "C_hat", "mean_p", "CV_p", "frac_p0", "kappa_over_m"]
targets = sorted(set(p.split("__")[1] for p in ct["metric_pair"]))
M = np.full((len(feats), len(targets)), np.nan)
for _, row in ct.iterrows():
    a, b = row["metric_pair"].split("__")
    if a in feats and b in targets:
        M[feats.index(a), targets.index(b)] = row["spearman_rho"]

XL = {"avg_clustering": "clustering", "avg_degree": "avg degree", "beta_c": r"$\beta_c$",
      "degree_assortativity": "assortativity", "density": "density",
      "final_size_at_beta_0_25": "final size", "lambda1": r"$\lambda_1$",
      "lambda2": r"$\lambda_2$", "max_degree": "max degree", "modularity": "modularity",
      "percolation_AUC_random": "AUC (perc)", "powerlaw_alpha": r"power-law $\alpha$",
      "q_c_random": r"$q_c$", "transitivity": "transitivity"}
YL = {"rho_E": r"$\rho_E$", "rho_V": r"$\rho_V$", "mu_L": r"$\mu_L$", "H": "H",
      "H_norm": r"$\hat{H}$", "C_hat": r"$\hat{C}$", "mean_p": r"mean $p$",
      "CV_p": r"CV $p$", "frac_p0": r"frac $p{=}0$", "kappa_over_m": r"$\kappa/m$"}

fig, ax = plt.subplots(figsize=(6.5, 6.5 / 2.3333))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(targets)))
ax.set_xticklabels([XL.get(t, t) for t in targets], rotation=90, fontsize=7)
ax.set_yticks(range(len(feats)))
ax.set_yticklabels([YL.get(f, f) for f in feats], fontsize=7)
for i in range(len(feats)):
    for j in range(len(targets)):
        if not np.isnan(M[i, j]):
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=5.2)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label(r"Spearman $\rho$", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax.set_title("LRS features vs structural/dynamic metrics (Spearman)", fontsize=7.5, pad=3)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "fig3_correlation_heatmap.png"), dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ Figure 5
with open(os.path.join(ROOT, "pinning_summary.csv"), encoding="utf-8") as f:
    pin = list(csv.DictReader(f))
for r in pin:
    r["f"] = float(r["f"])
    r["lambda_min"] = float(r["lambda_min"])

rep = [("dolphins", "Dolphins"), ("polbooks", "Polbooks"),
       ("email", "Email-Eu-core"), ("powergrid", "US power grid")]
styles = {"random": ("o", C[0], "-"), "degree_desc": ("s", C[1], "--"), "loop_desc": ("^", C[2], "-.")}
fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5 / 1.5))
for ax, (name, lbl) in zip(axes.flat, rep):
    for strat, (mk, col, ls) in styles.items():
        fs = sorted(set(r["f"] for r in pin if r["network"] == name))
        ys = []
        for fv in fs:
            m = [r["lambda_min"] for r in pin
                 if r["network"] == name and r["f"] == fv and r["strategy"] == strat]
            ys.append(m[0] if m else np.nan)
        ax.plot(fs, ys, marker=mk, color=col, ls=ls, ms=3, lw=0.9, label=strat)
    ax.set_xscale("log")
    ax.set_xlabel("pinning fraction $f$", fontsize=7)
    ax.set_ylabel(r"$\lambda_{min}(L_g)$", fontsize=7)
    ax.set_title(lbl, fontsize=7.5)
    ax.legend(fontsize=6, handlelength=1.6, borderpad=0.3)
    ax.grid(alpha=0.3, lw=0.4)
    ax.tick_params(labelsize=6)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIG, "fig5_pinning.png"), dpi=300)
plt.close(fig)
print("figures written to", FIG)
