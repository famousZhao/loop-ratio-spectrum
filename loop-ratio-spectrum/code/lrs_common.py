# -*- coding: utf-8 -*-
"""Common loading + LRS core math for Loop Ratio Spectrum analysis."""
import os, math
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SEED = 20261008

NAMES = ["karate","dolphins","polbooks","celegans","powergrid","email","euroroad","yeast"]
DISPLAY = {"karate":"Karate","dolphins":"Dolphins","polbooks":"Polbooks",
           "celegans":"C. elegans","powergrid":"US power grid","email":"Email-Eu-core",
           "euroroad":"Euroroad","yeast":"Yeast"}

def load_graph(name):
    G = nx.Graph()
    with open(os.path.join(DATA, f"{name}.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                G.add_edge(int(p[0]), int(p[1]))
    G = nx.relabel_nodes(G, {n:i for i,n in enumerate(sorted(G.nodes()))})
    return G

def girth_bfs(G):
    """Shortest cycle length via per-node BFS; global min."""
    adj = {n: set(G[n]) for n in G.nodes()}
    best = math.inf
    nodes = list(G.nodes())
    for root in nodes:
        if best == 3:
            return 3
        dist = {root:0}
        level = [root]
        d = 0
        while level and (2*d + 1) < best:
            nxt = []
            for u in level:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = d+1
                        nxt.append(v)
                    else:
                        # edge u-v closes a cycle if v is not the parent layer marker
                        cand = dist[u] + dist[v] + 1
                        if cand < best and cand >= 3:
                            best = cand
            level = nxt; d += 1
    return best

def lrs_core(G):
    n = G.number_of_nodes(); m = G.number_of_edges()
    c = nx.number_connected_components(G)
    kappa = m - n + c
    basis = nx.cycle_basis(G)
    k = len(basis)
    lengths = np.array([len(c) for c in basis], dtype=float)
    # spectrum r_l
    lmin, lmax = int(lengths.min()), int(lengths.max())
    counts = np.zeros(lmax+1)
    for L in lengths:
        counts[int(L)] += 1
    r = counts / k
    # node participation
    p = np.zeros(n)
    for cyc in basis:
        for v in cyc:
            p[v] += 1
    p = p / k
    nbridges = len(list(nx.bridges(G)))
    rho_E = 1.0 - nbridges/m
    rho_V = np.count_nonzero(p)/n
    support = counts[counts>0]
    H = -np.sum(support*np.log(support/k))  # note r already normalized; recompute cleanly
    r_pos = r[r>0]
    H = -np.sum(r_pos*np.log(r_pos))
    H_norm = H/math.log(len(r_pos)) if len(r_pos)>1 else 0.0
    mu_L = np.sum(np.arange(len(r))*r)
    C_hat = r.max()
    g = girth_bfs(G)
    return dict(n=n,m=m,c=c,kappa=kappa,k_basis=k,lengths=lengths,counts=counts,r=r,
                p=p,nbridges=nbridges,rho_E=rho_E,rho_V=rho_V,girth=g,
                H=H,H_norm=H_norm,mu_L=mu_L,C_hat=C_hat)

def eig_metrics(G):
    A = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()), format="csr")
    L = sparse.csr_matrix(nx.laplacian_matrix(G, nodelist=sorted(G.nodes())))
    lam1 = eigsh(A.astype(float), k=1, which="LA", return_eigenvectors=False, tol=1e-3, maxiter=5000)[0]
    # algebraic connectivity: 2 smallest Laplacian eigenvalues
    ev = eigsh(L.astype(float), k=2, sigma=0.0, which="LM", return_eigenvectors=False,
               tol=1e-3, maxiter=8000)
    ev = np.sort(ev)
    lam2 = ev[1]
    return float(lam1), float(lam2)

def structural(G, lrs):
    n = G.number_of_nodes(); m = G.number_of_edges()
    degs = np.array([d for _,d in G.degree()], dtype=float)
    density = 2*m/(n*(n-1))
    avg_deg = 2*m/n
    max_deg = degs.max()
    acc = nx.average_clustering(G)
    trans = nx.transitivity(G)
    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = float("nan")
    if n <= 500:
        asp = nx.average_shortest_path_length(G)
        diam = nx.diameter(G)
        asp_approx = False
    else:
        rng = np.random.default_rng(SEED)
        srcs = rng.choice(n, size=min(60,n), replace=False)
        dists = []
        for s in srcs:
            dd = nx.single_source_shortest_path_length(G, int(s))
            dists.extend(dd.values())
        asp = float(np.mean(dists)); diam = int(np.max(dists)); asp_approx = True
    # modularity
    try:
        comms = list(nx.community.greedy_modularity_communities(G))
        mod = nx.community.modularity(G, comms)
    except Exception:
        mod = float("nan")
    kmin = degs.min()
    alpha = 1.0 + n/np.sum(np.log(degs/kmin))
    return dict(density=density,avg_degree=avg_deg,max_degree=max_deg,
                avg_clustering=acc,transitivity=trans,assortativity=assort,
                avg_shortest_path=asp,diameter=diam,asp_approx=asp_approx,
                modularity=mod,powerlaw_alpha=alpha)

if __name__ == "__main__":
    for name in NAMES:
        G = load_graph(name)
        t0 = __import__("time").time()
        core = lrs_core(G)
        print(name, "n=",core["n"],"m=",core["m"],"kappa=",core["kappa"],
              "k_basis=",core["k_basis"],"girth=",core["girth"],
              "rho_E=",round(core["rho_E"],3),"rho_V=",round(core["rho_V"],3),
              "H=",round(core["H"],3),"muL=",round(core["mu_L"],2),
              "t=",round(__import__("time").time()-t0,1))
