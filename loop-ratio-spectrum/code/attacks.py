"""Node-removal attack curves used by the percolation and overlap scripts."""
import numpy as np


def lcc_frac(n, edge_pairs, removed):
    """Relative size of the largest connected component after removing nodes."""
    parent = list(range(n))
    active = np.ones(n, dtype=bool)
    for r in removed:
        active[r] = False

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for u, v in edge_pairs:
        if active[u] and active[v]:
            union(u, v)
    sizes = {}
    ac = 0
    for x in range(n):
        if active[x]:
            ac += 1
            r = find(x)
            sizes[r] = sizes.get(r, 0) + 1
    if ac == 0:
        return 0.0
    return max(sizes.values()) / n


def attack_curve(G, order, step):
    """S(q) for removing nodes in the given order, q on [0, 0.9]."""
    n = G.number_of_nodes()
    edge_pairs = np.array([[u, v] for u, v in G.edges()], dtype=int)
    qs = np.round(np.arange(0, 0.9 + 1e-9, step), 4)
    S = np.array([lcc_frac(n, edge_pairs, order[:int(round(q * n))]) for q in qs])
    return qs, S


def summ(qs, S):
    """Area under S(q) and the first q at which S drops below 0.5."""
    auc = float(np.trapezoid(S, qs))
    below = np.where(S < 0.5)[0]
    qc = float(qs[below[0]]) if len(below) else float("nan")
    return auc, qc
