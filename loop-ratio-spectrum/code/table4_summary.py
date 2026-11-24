"""Partial correlations reported in Table 4.

Each loop feature is correlated with three outcomes with the average degree
partialled out (Pearson partial correlation, one control variable).

Requires network_stats.csv, SIR_summary.csv and percolation_summary.csv at
the repository root (produced by compute_lrs.py).
"""
import csv
import os

import numpy as np
from scipy.stats import pearsonr, t as tdist

from paths import ROOT

stats = {r["network"]: r for r in
         csv.DictReader(open(os.path.join(ROOT, "network_stats.csv"), encoding="utf-8"))}
names = list(stats)

final = {}
for r in csv.DictReader(open(os.path.join(ROOT, "SIR_summary.csv"), encoding="utf-8")):
    b = float(r["beta"])
    if r["network"] not in final or abs(b - 0.25) < abs(float(final[r["network"]][0]) - 0.25):
        final[r["network"]] = (b, float(r["final_size"]))

percolation = {}
for r in csv.DictReader(open(os.path.join(ROOT, "percolation_summary.csv"), encoding="utf-8")):
    if r["strategy"] == "random":
        percolation[r["network"]] = (float(r["AUC"]), float(r["q_c"]))

degree = np.array([float(stats[n]["avg_degree"]) for n in names])


def partial(x, y, z=degree):
    rxy = pearsonr(x, y)[0]
    rxz = pearsonr(x, z)[0]
    ryz = pearsonr(y, z)[0]
    r = (rxy - rxz * ryz) / np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    n = len(x)
    t = r * np.sqrt((n - 3) / max(1 - r ** 2, 1e-12))
    return rxy, r, 2 * (1 - tdist.cdf(abs(t), n - 3))


outcomes = {"AUC random": np.array([percolation[n][0] for n in names]),
            "q_c random": np.array([percolation[n][1] for n in names]),
            "final size": np.array([final[n][1] for n in names])}

print("%-6s %-24s %8s %10s %8s" % ("feature", "outcome", "raw r", "partial r", "p"))
series = {"kappa_over_m": np.array([float(stats[n]["kappa"]) / float(stats[n]["m"]) for n in names]),
          "rho_E": np.array([float(stats[n]["rho_E"]) for n in names]),
          "rho_V": np.array([float(stats[n]["rho_V"]) for n in names]),
          "mu_L": np.array([float(stats[n]["mu_L"]) for n in names]),
          "H": np.array([float(stats[n]["H"]) for n in names])}
for k, x in series.items():
    for label, y in outcomes.items():
        rxy, rp, p = partial(x, y)
        print("%-6s %-24s %+8.3f %+10.3f %8.3f" % (k, label, rxy, rp, p))
