"""Minimum cycle basis as a third basis family.

The depth-first basis used in the paper is one choice among many.  This
script recomputes the descriptors from a minimum cycle basis on the three
networks for which the computation completes in reasonable time, and then
checks whether the cross-network association with the percolation outcome
survives when those three values replace the depth-first ones.

Output: results/min_cycle_basis.json
"""
import json
import random
import time
from collections import deque

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from attacks import attack_curve, summ
from lrs_common import load_graph, lrs_core, NAMES
from paths import RES

SMALL = ["karate", "dolphins", "polbooks"]
REPS = 10


def describe(cycles):
    lengths = np.array([len(c) for c in cycles], float)
    k = len(lengths)
    counts = np.zeros(int(lengths.max()) + 1)
    for L in lengths:
        counts[int(L)] += 1
    r = counts / k
    pos = r[r > 0]
    return (float(np.sum(np.arange(len(r)) * r)), float(-np.sum(pos * np.log(pos))), float(r.max()))


def nodes_of_cycle(cyc):
    if len(cyc) and isinstance(cyc[0], (tuple, list)) and len(cyc[0]) == 2:
        s = set()
        for u, v in cyc:
            s.add(u)
            s.add(v)
        return list(s)
    return list(cyc)


def bfs_basis(G, seed):
    """Fundamental basis of a randomly rooted breadth-first spanning tree."""
    rnd = random.Random(seed)
    root = rnd.choice(list(G.nodes()))
    par, depth, tree_edges = {root: None}, {root: 0}, set()
    q = deque([root])
    while q:
        u = q.popleft()
        nbrs = list(G[u])
        rnd.shuffle(nbrs)
        for v in nbrs:
            if v not in par:
                par[v] = u
                depth[v] = depth[u] + 1
                tree_edges.add((min(u, v), max(u, v)))
                q.append(v)
    cycles = []
    for u, v in G.edges():
        if (min(u, v), max(u, v)) in tree_edges:
            continue
        a, b = u, v
        up, down = [], []
        while depth[a] > depth[b]:
            up.append(a)
            a = par[a]
        while depth[b] > depth[a]:
            down.append(b)
            b = par[b]
        while a != b:
            up.append(a)
            down.append(b)
            a = par[a]
            b = par[b]
        cycles.append(up + [a] + down[::-1])
    return cycles


def coverage(G, cycles):
    s = set()
    for c in cycles:
        s.update(c)
    return len(s) / G.number_of_nodes()


res = {}
for name in SMALL:
    G = load_graph(name)
    dfs = nx.cycle_basis(G)
    d_dfs = describe(dfs)
    bf = np.array([describe(bfs_basis(G, s)) for s in range(6)])
    t0 = time.time()
    mcb = [nodes_of_cycle(c) for c in nx.minimum_cycle_basis(G)]
    t_mcb = time.time() - t0
    d_mcb = describe(mcb)
    res[name] = dict(n=G.number_of_nodes(), m=G.number_of_edges(),
                     dfs=dict(mu_L=d_dfs[0], H=d_dfs[1], C_hat=d_dfs[2], rho_V=coverage(G, dfs)),
                     bfs=dict(mu_L=float(bf[:, 0].mean()), H=float(bf[:, 1].mean()),
                              C_hat=float(bf[:, 2].mean())),
                     mcb=dict(mu_L=d_mcb[0], H=d_mcb[1], C_hat=d_mcb[2], rho_V=coverage(G, mcb)),
                     mcb_seconds=t_mcb)
    print("%-9s n=%3d | DFS mu %.2f H %.2f C %.3f rho_V %.3f | BFS mu %.2f | MCB mu %.2f H %.2f "
          "C %.3f rho_V %.3f (%.1f s)"
          % (name, G.number_of_nodes(), d_dfs[0], d_dfs[1], d_dfs[2], res[name]["dfs"]["rho_V"],
             np.mean(bf[:, 0]), d_mcb[0], d_mcb[1], d_mcb[2], res[name]["mcb"]["rho_V"], t_mcb),
          flush=True)

# percolation outcome for all eight networks, random removal averaged over REPS orders
outcome = {}
for name in NAMES:
    G = load_graph(name)
    n = G.number_of_nodes()
    step = 0.01 if n <= 500 else 0.02
    aucs, qcs = [], []
    for rep in range(REPS):
        order = np.random.default_rng(20261008 + rep).permutation(n)
        qs, S = attack_curve(G, order, step)
        a, q = summ(qs, S)
        aucs.append(a)
        qcs.append(q)
    outcome[name] = dict(auc=float(np.mean(aucs)), q_c=float(np.mean(qcs)))
    print("  %-10s AUC=%.3f q_c=%.3f" % (name, outcome[name]["auc"], outcome[name]["q_c"]), flush=True)

dfs_mu = {n: lrs_core(load_graph(n))["mu_L"] for n in NAMES}
dfs_H = {n: lrs_core(load_graph(n))["H"] for n in NAMES}
hyb_mu = [res[n]["mcb"]["mu_L"] if n in res else dfs_mu[n] for n in NAMES]
hyb_H = [res[n]["mcb"]["H"] if n in res else dfs_H[n] for n in NAMES]

print("\ncross-network Spearman, eight networks")
for label, dfs_v, hyb_v in (("mu_L", [dfs_mu[n] for n in NAMES], hyb_mu),
                            ("H", [dfs_H[n] for n in NAMES], hyb_H)):
    for key in ("auc", "q_c"):
        y = [outcome[n][key] for n in NAMES]
        r1, p1 = spearmanr(dfs_v, y)
        r2, p2 = spearmanr(hyb_v, y)
        print("  %-4s %-4s depth-first rho=%+.3f (p=%.3f) | MCB-substituted rho=%+.3f (p=%.3f)"
              % (label, key, r1, p1, r2, p2))

res["outcome"] = outcome
with open(RES + "/min_cycle_basis.json", "w") as f:
    json.dump(res, f, indent=1)
print("\nsaved results/min_cycle_basis.json")
