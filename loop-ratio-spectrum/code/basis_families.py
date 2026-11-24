# -*- coding: utf-8 -*-
"""Basis-family stability: fundamental bases from randomly rooted
breadth-first spanning trees, compared with the depth-first basis."""
import os, json, random
import numpy as np
import networkx as nx
from collections import deque
import lrs_common as lc

from paths import RES
OUT = os.path.join(RES, "basis_stability.json")

def bfs_basis(G, seed):
    rnd = random.Random(seed)
    root = rnd.choice(list(G.nodes()))
    par, depth, tree_edges = {root: None}, {root: 0}, set()
    q = deque([root])
    while q:
        u = q.popleft()
        nbrs = list(G[u]); rnd.shuffle(nbrs)
        for v in nbrs:
            if v not in par:
                par[v] = u; depth[v] = depth[u] + 1
                tree_edges.add((min(u, v), max(u, v)))
                q.append(v)
    cycles = []
    for u, v in G.edges():
        if (min(u, v), max(u, v)) in tree_edges:
            continue
        a, b = u, v
        up, down = [], []
        while depth[a] > depth[b]:
            up.append(a); a = par[a]
        while depth[b] > depth[a]:
            down.append(b); b = par[b]
        while a != b:
            up.append(a); down.append(b)
            a = par[a]; b = par[b]
        cyc = up + [a] + down[::-1]
        cycles.append(cyc)
    return cycles

def desc(cycles):
    L = np.array([len(c) for c in cycles], dtype=float)
    k = len(L)
    counts = np.zeros(int(L.max()) + 1)
    for x in L:
        counts[int(x)] += 1
    r = counts / k
    rp = r[r > 0]
    return float(np.sum(np.arange(len(r)) * r)), float(-np.sum(rp * np.log(rp))), float(r.max())

res = {}
for name in lc.NAMES:
    G = lc.load_graph(name)
    base = desc(nx.cycle_basis(G))
    rows = [desc(bfs_basis(G, s)) for s in range(10)]
    arr = np.array(rows)
    res[name] = dict(base=base,
                     mu=dict(mean=float(arr[:, 0].mean()), min=float(arr[:, 0].min()), max=float(arr[:, 0].max())),
                     H=dict(mean=float(arr[:, 1].mean()), min=float(arr[:, 1].min()), max=float(arr[:, 1].max())),
                     C=dict(mean=float(arr[:, 2].mean()), min=float(arr[:, 2].min()), max=float(arr[:, 2].max())))
    rel_mu = 100 * (arr[:, 0].max() - arr[:, 0].min()) / arr[:, 0].mean()
    rel_H = 100 * (arr[:, 1].max() - arr[:, 1].min()) / arr[:, 1].mean()
    rel_C = 100 * (arr[:, 2].max() - arr[:, 2].min()) / arr[:, 2].mean()
    print("%-13s DFS-base mu %.2f H %.3f | BFS-bases mu %.2f [%.2f,%.2f] rel %4.1f%% | H %.3f [%.3f,%.3f] rel %4.1f%% | C rel %4.1f%%"
          % (lc.DISPLAY[name], base[0], base[1], arr[:, 0].mean(), arr[:, 0].min(), arr[:, 0].max(), rel_mu,
             arr[:, 1].mean(), arr[:, 1].min(), arr[:, 1].max(), rel_H, rel_C))

for name in lc.NAMES:
    G = lc.load_graph(name)
    # Horton's algorithm is superlinear in the graph size and does not finish
    # within a useful time budget above roughly a hundred nodes.
    if G.number_of_nodes() > 150:
        res[name]["minbasis"] = "skipped (not tractable at this size)"
        continue
    cyc = nx.minimum_cycle_basis(G)
    mb = desc(cyc)
    res[name]["minbasis"] = mb
    print("%-13s minimum basis: mu %.2f H %.3f C %.3f (k=%d)"
          % (lc.DISPLAY[name], mb[0], mb[1], mb[2], len(cyc)))

json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
