"""Dependence of the spreading results on the transmission probability.

The paper reports the SIR outcome at beta = 0.25.  This script repeats the
same protocol at six values of beta and recomputes (i) the Spearman
correlation between the mean cycle length and the final infected fraction,
and (ii) the same correlation with the average degree partialled out.

Protocol: five initially infected nodes, gamma = 1, 200 realizations per
network and per value of beta, seeded from 20261008.

Output: results/sir_beta.json
"""
import json
import os

import numpy as np
import networkx as nx
from scipy.stats import pearsonr, spearmanr, t as tdist

from lrs_common import load_graph, lrs_core, NAMES
from paths import RES

BETAS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
REPS = 200


def sir_final_fraction(G, beta, n_init, reps, seed=20261008):
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), format="csr").astype(float)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(reps):
        seeds = rng.choice(n, size=n_init, replace=False)
        S = np.ones(n, bool)
        I = np.zeros(n, bool)
        I[seeds] = True
        S[seeds] = False
        infected = n_init
        while I.any():
            contact = (A @ I.astype(float) > 0)
            new = contact & S & (rng.random(n) < beta)
            if not new.any():
                break
            I = new
            S &= ~new
            infected += int(new.sum())
        total += infected
    return total / reps / n


def partial(x, y, z):
    rxy = pearsonr(x, y)[0]
    rxz = pearsonr(x, z)[0]
    ryz = pearsonr(y, z)[0]
    r = (rxy - rxz * ryz) / np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    n = len(x)
    t = r * np.sqrt((n - 3) / max(1 - r ** 2, 1e-12))
    return r, 2 * (1 - tdist.cdf(abs(t), n - 3))


rows = {}
for name in NAMES:
    G = load_graph(name)
    core = lrs_core(G)
    rows[name] = dict(mu_L=core["mu_L"], H=core["H"],
                      kappa_over_m=core["kappa"] / G.number_of_edges(),
                      avg_degree=2 * G.number_of_edges() / G.number_of_nodes())
    n_init = 5 if G.number_of_nodes() <= 2000 else 10
    for b in BETAS:
        rows[name]["f%.2f" % b] = sir_final_fraction(G, b, n_init, REPS)
    print("%-10s" % name, " ".join("%.2f" % rows[name]["f%.2f" % b] for b in BETAS), flush=True)

print("\n%-5s %-34s %-30s %s" % ("beta", "mu_L: rho / partial(degree)", "H: rho / partial", "mean fraction"))
out = {}
for b in BETAS:
    y = np.array([rows[n]["f%.2f" % b] for n in NAMES])
    deg = np.array([rows[n]["avg_degree"] for n in NAMES])
    entry = {}
    for k in ("mu_L", "H"):
        x = np.array([rows[n][k] for n in NAMES])
        r1, p1 = spearmanr(x, y)
        r2, p2 = partial(x, y, deg)
        entry[k] = dict(spearman=float(r1), spearman_p=float(p1),
                        partial=float(r2), partial_p=float(p2))
    entry["mean_fraction"] = float(np.mean(y))
    out["%.2f" % b] = entry
    print("%.2f  rho=%+.2f p=%.3f / partial=%+.3f p=%.3f   rho=%+.2f / partial=%+.3f p=%.3f   %.2f"
          % (b, entry["mu_L"]["spearman"], entry["mu_L"]["spearman_p"],
             entry["mu_L"]["partial"], entry["mu_L"]["partial_p"],
             entry["H"]["spearman"], entry["H"]["partial"], entry["H"]["partial_p"],
             entry["mean_fraction"]))

with open(os.path.join(RES, "sir_beta.json"), "w") as f:
    json.dump(dict(per_network=rows, summary=out, reps=REPS), f, indent=1)
