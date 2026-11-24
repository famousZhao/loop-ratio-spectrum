# -*- coding: utf-8 -*-
"""Figure 9: descriptor map of the bearing state networks and the
percolation attack comparison."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import FIG, RES
OUT = RES
d = json.load(open(os.path.join(OUT, "mfpt_records.json"), encoding="utf-8"))
rows = d["record_rows"]; att = d["attack"]
COL = {"healthy": "#2b6cb0", "outer": "#c05621", "inner": "#2f855a"}
LAB = {"healthy": "healthy", "outer": "outer race", "inner": "inner race"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})

for c in ("healthy", "outer", "inner"):
    xs = [r["mu_L"] for r in rows if r["cls"] == c]
    ys = [r["rho_V"] for r in rows if r["cls"] == c]
    a1.scatter(xs, ys, s=44, color=COL[c], edgecolor="k", linewidth=0.4, label=LAB[c], zorder=3)
a1.set_xlabel(r"mean cycle length $\mu_L$")
a1.set_ylabel(r"node loop ratio $\rho_V$")
a1.set_title("(a) state networks of 17 bearing records", fontsize=10)
a1.legend(frameon=False, fontsize=8, loc="upper left")
a1.grid(alpha=0.25, linewidth=0.6)

strat = ["random", "degree", "loop", "betweenness"]
xt = np.arange(len(strat)); w = 0.26
for i, c in enumerate(("healthy", "outer", "inner")):
    vals = [att[c][s]["mean_auc"] for s in strat]
    errs = [att[c][s]["sd"] for s in strat]
    a2.bar(xt + (i - 1) * w, vals, w, yerr=errs, capsize=2.5, color=COL[c],
           edgecolor="k", linewidth=0.4, label=LAB[c], error_kw={"linewidth": 0.7})
a2.set_xticks(xt); a2.set_xticklabels(["random", "degree", "loop", "betweenness"], fontsize=9)
a2.set_ylabel("percolation AUC")
a2.set_ylim(0, 0.66)
a2.set_title("(b) attack on the same state networks", fontsize=10)
a2.legend(frameon=False, fontsize=8, loc="upper right", borderaxespad=0.2)
a2.grid(axis="y", alpha=0.25, linewidth=0.6)

for ax in (a1, a2):
    ax.tick_params(labelsize=8.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
fig.tight_layout()
p = os.path.join(FIG, "fig9_bearing.png")
fig.savefig(p, dpi=300)
print("saved", p, os.path.getsize(p))
