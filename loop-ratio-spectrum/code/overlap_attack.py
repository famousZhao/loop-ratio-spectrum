"""Overlap between the node loop ratio and degree, and the threshold margin
used by the attack-selection rule.

Output: results/overlap_attack.json
"""
import json
import os

import numpy as np
import networkx as nx
from scipy.stats import spearmanr

from attacks import attack_curve, summ
from lrs_common import load_graph, lrs_core, NAMES, DISPLAY
from paths import RES

CACHE = {}

print("%-13s %5s %6s %6s %6s %6s %6s %6s"
      % ("network", "rho", "J5%", "J10%", "AUCdeg", "AUCloop", "AUCstrat", "ratio"))

for name in NAMES:
    G = load_graph(name)
    n = G.number_of_nodes()
    core = lrs_core(G)
    p = core["p"]
    deg = np.array([d for _, d in G.degree()], dtype=float)
    rho = spearmanr(p, deg).statistic

    def jaccard(k):
        a = set(np.argsort(-p)[:k])
        b = set(np.argsort(-deg)[:k])
        return len(a & b) / len(a | b)

    k5, k10 = max(1, int(0.05 * n)), max(1, int(0.10 * n))
    j5, j10 = jaccard(k5), jaccard(k10)

    step = 0.01 if n <= 500 else 0.02
    qs, S = attack_curve(G, np.argsort(-deg), step)
    auc_deg = summ(qs, S)[0]
    qs, S = attack_curve(G, np.argsort(-p), step)
    auc_loop = summ(qs, S)[0]
    med = np.median(deg)
    low = np.where(deg <= med)[0]
    high = np.where(deg > med)[0]
    strat = np.concatenate([low[np.argsort(-p[low])], high[np.argsort(-p[high])]])
    qs, S = attack_curve(G, strat, step)
    auc_strat = summ(qs, S)[0]

    bc = nx.betweenness_centrality(G, normalized=True, seed=20261008)
    order_bc = np.argsort(-np.array([bc[i] for i in range(n)]))
    qs, S = attack_curve(G, order_bc, step)
    auc_btw = summ(qs, S)[0]

    ratio = auc_loop / auc_btw
    rhoE = core["rho_E"]
    CACHE[name] = dict(rho_pv_deg=float(rho), j5=j5, j10=j10, auc_deg=auc_deg,
                       auc_loop=auc_loop, auc_strat=auc_strat, auc_btw=auc_btw,
                       ratio_loop_btw=ratio, rho_E=rhoE)
    print("%-13s %5.2f %6.2f %6.2f %6.3f %6.3f %6.3f %6.2f  rhoE=%.3f"
          % (DISPLAY[name], rho, j5, j10, auc_deg, auc_loop, auc_strat, ratio, rhoE))

with open(os.path.join(RES, "overlap_attack.json"), "w") as f:
    json.dump(CACHE, f, indent=1)

good = [k for k in NAMES if CACHE[k]["ratio_loop_btw"] <= 1.20]
bad = [k for k in NAMES if CACHE[k]["ratio_loop_btw"] > 1.40]
print("\nthreshold margin: highest rhoE among ratio > 1.4 = %.3f, "
      "lowest rhoE among ratio <= 1.2 = %.3f"
      % (max(CACHE[k]["rho_E"] for k in bad), min(CACHE[k]["rho_E"] for k in good)))
print("within 1.2 of betweenness:",
      [(DISPLAY[k], round(CACHE[k]["rho_E"], 3), round(CACHE[k]["ratio_loop_btw"], 2)) for k in good])
print("beyond 1.4:",
      [(DISPLAY[k], round(CACHE[k]["rho_E"], 3), round(CACHE[k]["ratio_loop_btw"], 2)) for k in bad])
