# -*- coding: utf-8 -*-
"""Controlled experiment: identical n, m and degree sequence, rewiring only changes cycle lengths."""
import os, json
import numpy as np
import networkx as nx
from scipy.stats import spearmanr

from paths import RES

OUT = RES
N, K = 500, 6
BETAS = [0.0, 0.05, 0.2, 0.5, 1.0]
SEEDS = 10
QS = np.arange(0.0, 0.91, 0.02)

def descriptors(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    c = nx.number_connected_components(G)
    basis = nx.cycle_basis(G); k = max(len(basis), 1)
    lens = np.array([len(b) for b in basis], float)
    counts = np.zeros(int(lens.max()) + 1)
    for L in lens: counts[int(L)] += 1
    r = counts / k
    pos = r[r > 0]
    H = float(-np.sum(pos * np.log(pos)))
    pv = np.zeros(n)
    for cy in basis:
        for v in cy: pv[v] += 1
    pv /= k
    return dict(kappa_over_m=(m - n + c) / m, mu_L=float(np.sum(np.arange(len(r)) * r)),
                H=H, rho_V=float(np.count_nonzero(pv) / n),
                rho_E=1 - len(list(nx.bridges(G))) / m, c=c,
                lmax=float(lens.max()))

def auc_random(G, reps=20):
    n = G.number_of_nodes()
    rng = np.random.default_rng(7)
    curve = np.zeros(len(QS))
    for _ in range(reps):
        order = rng.permutation(n)
        H_ = G.copy(); S = []
        removed = 0
        for qi, q in enumerate(QS):
            target = int(round(q * n))
            while removed < target:
                H_.remove_node(int(order[removed])); removed += 1
            if H_.number_of_nodes() == 0:
                S.append(0.0)
            else:
                S.append(max(len(x) for x in nx.connected_components(H_)) / n)
        curve += np.array(S) / reps
    return float(np.trapezoid(curve, QS) / (QS[-1] - QS[0])), float(curve[-1])

def sir_final(G, beta=0.25, mu=1.0, reps=20, seed=11):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes(); deg = np.array([d for _, d in G.degree()], float)
    tot = 0.0
    for _ in range(reps):
        I = np.zeros(n, bool); I[rng.integers(n)] = True
        ever = I.copy()
        while I.any():
            new = np.zeros(n, bool)
            for v in np.flatnonzero(I):
                for u in G.neighbors(int(v)):
                    if not ever[u] and rng.random() < beta:
                        new[u] = True
            ever |= new
            I = new
        tot += int(ever.sum())
    return tot / reps

if __name__ == "__main__":
    rows = []
    for b in BETAS:
        for s in range(SEEDS):
            G = nx.watts_strogatz_graph(N, K, b, seed=1000 + s)
            d = descriptors(G)
            auc, tail = auc_random(G)
            fs = sir_final(G)
            d.update(beta=b, seed=s, auc=auc, tail=tail, final=fs,
                     C=nx.average_clustering(G), deg=2 * G.number_of_edges() / G.number_of_nodes())
            rows.append(d)
        sub = [r for r in rows if r["beta"] == b]
        print("beta=%.2f C=%.3f kappa/m=%.3f mu_L=%.2f H=%.2f rho_V=%.3f lmax=%.0f  AUC=%.3f tail=%.3f SIR=%.1f"
              % (b, np.mean([r["C"] for r in sub]), np.mean([r["kappa_over_m"] for r in sub]), np.mean([r["mu_L"] for r in sub]),
                 np.mean([r["H"] for r in sub]), np.mean([r["rho_V"] for r in sub]),
                 np.mean([r["lmax"] for r in sub]), np.mean([r["auc"] for r in sub]),
                 np.mean([r["tail"] for r in sub]), np.mean([r["final"] for r in sub])), flush=True)
    json.dump(rows, open(os.path.join(OUT, "rewire_exp.json"), "w"))

    kappa = [r["kappa_over_m"] for r in rows]
    print("\nkappa/m unique values:", sorted(set(round(x, 6) for x in kappa)))
    for k in ("mu_L", "H", "rho_V", "lmax"):
        print("  rho(%s, AUC)=%.3f p=%.2e | rho(%s, SIR)=%.3f p=%.2e"
              % (k, spearmanr([r[k] for r in rows], [r["auc"] for r in rows])[0],
                 spearmanr([r[k] for r in rows], [r["auc"] for r in rows])[1],
                 k, spearmanr([r[k] for r in rows], [r["final"] for r in rows])[0],
                 spearmanr([r[k] for r in rows], [r["final"] for r in rows])[1]))
