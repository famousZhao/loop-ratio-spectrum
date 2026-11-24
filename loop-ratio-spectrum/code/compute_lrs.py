# -*- coding: utf-8 -*-
"""
Loop Ratio Spectrum (LRS) analysis: full computation pipeline.

Reproduces all CSVs, pickled checkpoints, and figures for the LRS paper.
Run from the repository root: python code/compute_lrs.py [stage]

Stages:
    core    -> LRS core + structural + eig + exact enumeration  (results/core.pkl)
    pin     -> consensus pinning control                          (results/pin.pkl)
    sir     -> discrete-time SIR                                 (results/sir.pkl)
    perc    -> node-percolation robustness                        (results/perc.pkl)
    null    -> ER / configuration-model null models + JSD        (results/null.pkl)
    tables  -> build all CSV deliverables from checkpoints
    figs    -> build all 8 figures
    all     -> run everything in order

Random seed fixed SEED=20261008 throughout.
"""
import os, math, time, pickle, argparse, warnings
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh, eigs
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, RES, FIG as FIGD

SEED = 20261008
NAMES = ["karate", "dolphins", "polbooks", "celegans",
         "powergrid", "email", "euroroad", "yeast"]
DISPLAY = {"karate": "Karate", "dolphins": "Dolphins", "polbooks": "Polbooks",
           "celegans": "C. elegans", "powergrid": "US power grid",
           "email": "Email-Eu-core", "euroroad": "Euroroad", "yeast": "Yeast"}

# ---------------------------------------------------------------- graph loading
def load_graph(name):
    G = nx.Graph()
    with open(os.path.join(DATA, f"{name}.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                G.add_edge(int(p[0]), int(p[1]))
    G.remove_edges_from(nx.selfloop_edges(G))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(comps[0]).copy()
    G = nx.relabel_nodes(G, {n: i for i, n in enumerate(sorted(G.nodes()))})
    return G

# ---------------------------------------------------------------- girth (fixed)
def girth_bfs(G):
    """Shortest cycle length: per-root BFS with parent tracking, global min."""
    adj = {v: list(G.neighbors(v)) for v in G.nodes()}
    best = math.inf
    for root in G.nodes():
        if best == 3:
            return 3
        dist = {root: 0}
        parent = {root: -1}
        level = [root]
        d = 0
        while level and (2 * d + 1) < best:
            nxt = []
            for u in level:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = d + 1
                        parent[v] = u
                        nxt.append(v)
                    elif v != parent[u]:
                        cand = dist[u] + dist[v] + 1
                        if cand < best and cand >= 3:
                            best = cand
            level = nxt; d += 1
    return int(best)

# ---------------------------------------------------------------- LRS core
def lrs_core(G):
    n = G.number_of_nodes(); m = G.number_of_edges()
    c = nx.number_connected_components(G)
    kappa = m - n + c
    basis = nx.cycle_basis(G)
    k = len(basis)
    lengths = np.array([len(cyc) for cyc in basis], dtype=float)
    lmax = int(lengths.max())
    counts = np.zeros(lmax + 1)
    for L in lengths:
        counts[int(L)] += 1
    r = counts / k
    p = np.zeros(n)
    for cyc in basis:
        for v in cyc:
            p[v] += 1
    p = p / k
    nbridges = len(list(nx.bridges(G)))
    rho_E = 1.0 - nbridges / m
    rho_V = float(np.count_nonzero(p) / n)
    r_pos = r[r > 0]
    H = float(-np.sum(r_pos * np.log(r_pos)))
    H_norm = float(H / math.log(len(r_pos))) if len(r_pos) > 1 else 0.0
    mu_L = float(np.sum(np.arange(len(r)) * r))
    C_hat = float(r.max())
    g = girth_bfs(G)
    mean_p = float(np.mean(p))
    std_p = float(np.std(p))
    CV_p = std_p / mean_p if mean_p > 0 else float("nan")
    frac_p0 = float(np.mean(p == 0))
    return dict(n=n, m=m, c=c, kappa=kappa, k_basis=k, lengths=lengths,
                counts=counts, r=r, p=p, nbridges=nbridges, rho_E=rho_E,
                rho_V=rho_V, girth=g, H=H, H_norm=H_norm, mu_L=mu_L,
                C_hat=C_hat, mean_p=mean_p, CV_p=CV_p, frac_p0=frac_p0)

def eig_metrics(G):
    nodelist = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format="csr").astype(float)
    L = nx.laplacian_matrix(G, nodelist=nodelist).tocsr().astype(float)
    lam1 = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                       tol=1e-4, maxiter=5000)[0])
    ev = eigsh(L, k=2, sigma=0.0, which="LM", return_eigenvectors=False,
               tol=1e-4, maxiter=10000)
    ev = np.sort(ev)
    lam2 = float(max(ev[1], 0.0))
    return lam1, lam2

def structural(G, core):
    n = G.number_of_nodes(); m = G.number_of_edges()
    degs = np.array([d for _, d in G.degree()], dtype=float)
    density = 2 * m / (n * (n - 1))
    avg_deg = 2 * m / n
    max_deg = float(degs.max())
    acc = float(nx.average_clustering(G))
    trans = float(nx.transitivity(G))
    try:
        assort = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        assort = float("nan")
    if n <= 500:
        asp = float(nx.average_shortest_path_length(G))
        diam = int(nx.diameter(G))
        asp_approx = False
    else:
        rng = np.random.default_rng(SEED)
        pairs = rng.integers(0, n, size=(2000, 2))
        dists = []
        sp_cache = {}
        for a, b in pairs:
            a = int(a); b = int(b)
            if a == b:
                continue
            if a not in sp_cache:
                sp_cache[a] = nx.single_source_shortest_path_length(G, a)
            dists.append(sp_cache[a].get(b, np.nan))
        dists = np.array(dists); dists = dists[~np.isnan(dists)]
        asp = float(np.mean(dists))
        # diameter estimate via sampled BFS maxima
        srcs = rng.choice(n, size=min(80, n), replace=False)
        diam = 0
        for s in srcs:
            dd = nx.single_source_shortest_path_length(G, int(s))
            diam = max(diam, max(dd.values()))
        asp_approx = True
    try:
        comms = list(nx.community.greedy_modularity_communities(G))
        mod = float(nx.community.modularity(G, comms))
    except Exception:
        mod = float("nan")
    kmin = degs.min()
    alpha = 1.0 + n / np.sum(np.log(degs / kmin))
    return dict(density=density, avg_degree=avg_deg, max_degree=max_deg,
                avg_clustering=acc, transitivity=trans, assortativity=assort,
                avg_shortest_path=asp, diameter=diam, asp_approx=asp_approx,
                modularity=mod, powerlaw_alpha=alpha)

# ---------------------------------------------------------------- exact cycle enumeration
def enumerate_simple_cycles(G, L):
    """All simple cycles of length 3..L, deduped (rotation/reversal = same).
    Standard DFS anchored at smallest node to avoid duplicates."""
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    found = {}  # canonical tuple -> length
    nodes = sorted(G.nodes())
    for s in nodes:
        # iterative DFS; path starts at s; only visit nodes > s
        stack = [(s, [s], {s})]
        while stack:
            u, path, pset = stack.pop()
            if len(path) >= 3:
                if s in adj[u]:
                    # closes a cycle back to s
                    cyc = path
                    n = len(cyc)
                    rots = []
                    for seq in (cyc, list(reversed(cyc))):
                        for i in range(n):
                            rots.append(tuple(seq[i:] + seq[:i]))
                    found[min(rots)] = n
            if len(path) >= L:
                continue
            for w in adj[u]:
                if w > s and w not in pset:
                    stack.append((w, path + [w], pset | {w}))
    lengths = np.array(list(found.values()), dtype=int)
    return lengths

def exact_spectrum_corr(G, L):
    lengths = enumerate_simple_cycles(G, L)
    if len(lengths) == 0:
        return None
    lmax = int(lengths.max())
    cnt_exact = np.zeros(lmax + 1)
    for v in lengths:
        cnt_exact[v] += 1
    r_exact = cnt_exact / len(lengths)
    # basis spectrum on common support
    basis = nx.cycle_basis(G)
    blens = np.array([len(c) for c in basis])
    bmax = int(blens.max())
    cnt_basis = np.zeros(max(lmax, bmax) + 1)
    for v in blens:
        cnt_basis[int(v)] += 1
    r_basis = cnt_basis / len(blens)
    lo, hi = 3, lmax
    common = np.arange(lo, hi + 1)
    a = r_exact[common]
    b = r_basis[common]
    if np.std(a) == 0 or np.std(b) == 0:
        r_p, p_p = float("nan"), float("nan")
    else:
        r_p, p_p = pearsonr(a, b)
    return dict(L=L, n_exact=int(len(lengths)), r_exact=r_exact,
                r_basis=r_basis, lo=lo, hi=hi,
                pearson_r=float(r_p), pearson_p=float(p_p))

# ================================================================ STAGE core
def stage_core():
    res = {}
    for name in NAMES:
        t0 = time.time()
        G = load_graph(name)
        core = lrs_core(G)
        lam1, lam2 = eig_metrics(G)
        core["lambda1"] = lam1; core["lambda2"] = lam2
        st = structural(G, core)
        core["struct"] = st
        print(f"[core] {name:11s} n={core['n']:5d} m={core['m']:6d} "
              f"kappa={core['kappa']:6d} k_basis={core['k_basis']:6d} "
              f"girth={core['girth']:3d} rhoE={core['rho_E']:.3f} "
              f"rhoV={core['rho_V']:.3f} H={core['H']:.3f} "
              f"t={time.time()-t0:.1f}s", flush=True)
        res[name] = core
    # exact enumeration for small networks
    exact = {}
    for name, L in [("karate", 8), ("dolphins", 8), ("polbooks", 6)]:
        t0 = time.time()
        G = load_graph(name)
        try:
            r = exact_spectrum_corr(G, L)
        except Exception as e:
            print(f"[exact] {name} failed: {e}", flush=True); r = None
        exact[name] = r
        if r:
            print(f"[exact] {name:10s} L={L} n_exact={r['n_exact']:6d} "
                  f"pearson_r={r['pearson_r']:.4f} p={r['pearson_p']:.2e} "
                  f"t={time.time()-t0:.1f}s", flush=True)
    with open(os.path.join(RES, "core.pkl"), "wb") as f:
        pickle.dump({"core": res, "exact": exact}, f)
    print("CORE DONE", flush=True)

# ================================================================ STAGE pinning
def grounded_lambda_min(G, pin):
    """L_g = Laplacian restricted to free nodes; return smallest eigenvalue."""
    n = G.number_of_nodes()
    pin_set = set(int(x) for x in pin)
    free = np.array([v for v in range(n) if v not in pin_set], dtype=int)
    if len(free) == 0:
        return 0.0
    L = nx.laplacian_matrix(G, nodelist=list(range(n))).tocsr().astype(float)
    sub = L[free, :][:, free]
    sub = (sub + sub.T) / 2.0  # symmetrize
    k = min(1, sub.shape[0] - 1)
    if k == 0:
        return float(sub[0, 0])
    try:
        ev = eigsh(sub, k=k, sigma=1e-3, which="LM",
                   return_eigenvectors=False, tol=1e-4, maxiter=5000)
        return float(np.min(np.abs(ev)))
    except Exception:
        try:
            ev = eigs(sub, k=k, which="SM", return_eigenvectors=False, tol=1e-3)
            return float(np.min(np.abs(ev.real)))
        except Exception:
            return float("nan")

def stage_pin():
    with open(os.path.join(RES, "core.pkl"), "rb") as f:
        core = pickle.load(f)["core"]
    out = []
    FV = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    for name in NAMES:
        G = load_graph(name)
        n = G.number_of_nodes()
        p = core[name]["p"]
        degs = np.array([d for _, d in G.degree()], dtype=float)
        order_deg = np.argsort(-degs)
        order_loop = np.lexsort((-degs, -p))  # primary -p, tie-break -deg
        lam2 = core[name]["lambda2"]
        for f in FV:
            k = 0 if f == 0.0 else max(1, int(round(f * n)))
            if f == 0.0:
                out.append(dict(network=name, f=f, strategy="baseline",
                                 lambda_min=lam2))
                continue
            # degree_desc
            pin = order_deg[:k]
            out.append(dict(network=name, f=f, strategy="degree_desc",
                            lambda_min=grounded_lambda_min(G, pin)))
            # loop_desc
            pin = order_loop[:k]
            out.append(dict(network=name, f=f, strategy="loop_desc",
                            lambda_min=grounded_lambda_min(G, pin)))
            # random, 20 reps (deterministic per-network seed offset)
            off = {"karate": 11, "dolphins": 23, "polbooks": 37, "celegans": 51,
                   "powergrid": 67, "email": 83, "euroroad": 97, "yeast": 113}[name]
            rng = np.random.default_rng(SEED + off)
            vals = []
            for rep in range(20):
                pin = rng.choice(n, size=k, replace=False)
                vals.append(grounded_lambda_min(G, pin))
            out.append(dict(network=name, f=f, strategy="random",
                            lambda_min=float(np.mean(vals))))
        print(f"[pin] {name} done", flush=True)
    with open(os.path.join(RES, "pin.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("PIN DONE", flush=True)

# ================================================================ STAGE SIR
def sir_curve(G, beta_grid, n_init, n_rep, rng):
    """Discrete-time SIR, gamma=1.0. Vectorized via sparse matvec.
    Returns mean final_size and mean peak over reps for each beta."""
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), format="csr").astype(float)
    finals = np.zeros(len(beta_grid))
    peaks = np.zeros(len(beta_grid))
    for bi, beta in enumerate(beta_grid):
        fs = np.zeros(n_rep); pk = np.zeros(n_rep)
        for rep in range(n_rep):
            seeds = rng.choice(n, size=n_init, replace=False)
            S = np.ones(n, dtype=bool)
            I = np.zeros(n, dtype=bool)
            I[seeds] = True
            S[seeds] = False
            total_inf = n_init
            peak = n_init
            while I.any():
                # number of infected neighbors
                cnt = A.dot(I.astype(float))
                p_inf = 1.0 - (1.0 - beta) ** cnt
                draw = rng.random(n)
                newI = (draw < p_inf) & S
                S &= ~newI
                I = newI
                nin = int(newI.sum())
                total_inf += nin
                peak = max(peak, nin)
            fs[rep] = total_inf / n
            pk[rep] = peak / n
        finals[bi] = fs.mean(); peaks[bi] = pk.mean()
    return finals, peaks

def stage_sir():
    out_rows = []; thr_rows = []
    for name in NAMES:
        t0 = time.time()
        G = load_graph(name)
        n = G.number_of_nodes()
        n_init = 5 if n <= 2000 else 10
        n_rep = 200 if n <= 1500 else 100
        rng = np.random.default_rng(SEED)
        # choose beta range adaptively: log-uniform 24 points
        # scan a coarse range to pick final_size ~0 to >0.8
        coarse = np.logspace(-3, 0, 13)
        # quick pilot with fewer reps to calibrate range
        pilot_f = []
        for b in coarse:
            fi, _ = sir_curve(G, np.array([b]), n_init,
                              max(3, n_rep // 10), rng)
            pilot_f.append(fi[0])
        pilot_f = np.array(pilot_f)
        lo = coarse[0]; hi = coarse[-1]
        # find beta where final crosses 0.8 and where it crosses 0.01
        above = np.where(pilot_f > 0.8)[0]
        below = np.where(pilot_f < 0.01)[0]
        if len(above) > 0:
            hi = coarse[above[0]]
        if len(below) > 0:
            lo = coarse[min(below[-1] + 1, len(coarse) - 1)]
        if hi <= lo:
            lo = coarse[0]; hi = coarse[-1]
        beta_grid = np.logspace(math.log10(lo), math.log10(hi), 24)
        finals, peaks = sir_curve(G, beta_grid, n_init, n_rep, rng)
        for b, fv, pv in zip(beta_grid, finals, peaks):
            out_rows.append(dict(network=name, beta=float(b),
                                  final_size=float(fv), peak=float(pv)))
        # beta_c: smallest beta with final >= 0.01
        bc = float("nan")
        idx = np.where(finals >= 0.01)[0]
        if len(idx) > 0:
            bc = float(beta_grid[idx[0]])
        # direct evaluation at beta=0.25 with full MC repetitions
        f025, p025 = sir_curve(G, np.array([0.25]), n_init, n_rep, rng)
        thr_rows.append(dict(network=name, beta_c=bc,
                             final_size_at_beta_0_25=float(f025[0]),
                             peak_at_beta_0_25=float(p025[0]),
                             beta_used_at_025=0.25))
        print(f"[sir] {name:11s} n={n:5d} betas=[{lo:.4g},{hi:.4g}] "
              f"beta_c={bc:.4g} f@.25={f025[0]:.3f} t={time.time()-t0:.1f}s",
              flush=True)
    with open(os.path.join(RES, "sir.pkl"), "wb") as f:
        pickle.dump(dict(rows=out_rows, thr=thr_rows), f)
    print("SIR DONE", flush=True)

# ================================================================ STAGE percolation
def lcc_frac(n, edge_index_pairs, removed):
    """Union-find LCC fraction after removing `removed` nodes."""
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
    for u, v in edge_index_pairs:
        if active[u] and active[v]:
            union(u, v)
    sizes = {}
    active_count = 0
    for x in range(n):
        if active[x]:
            active_count += 1
            r = find(x)
            sizes[r] = sizes.get(r, 0) + 1
    if active_count == 0:
        return 0.0
    return max(sizes.values()) / n

def stage_perc():
    with open(os.path.join(RES, "core.pkl"), "rb") as f:
        core = pickle.load(f)["core"]
    out = []
    curves = {}
    for name in NAMES:
        G = load_graph(name)
        n = G.number_of_nodes()
        edges = list(G.edges())
        edge_pairs = np.array([[u, v] for u, v in edges], dtype=int)
        degs = np.array([d for _, d in G.degree()], dtype=float)
        order_deg = np.argsort(-degs)
        p = core[name]["p"]
        order_loop = np.argsort(-p)
        step = 0.01 if n <= 500 else 0.02
        qs = np.round(np.arange(0, 0.9 + 1e-9, step), 4)
        for strategy in ["random", "degree_desc", "loop_desc"]:
            if strategy == "random":
                S_runs = []
                for rep in range(20):
                    rng = np.random.default_rng(SEED + rep)
                    order = rng.permutation(n)
                    S = []
                    for q in qs:
                        k = int(round(q * n))
                        S.append(lcc_frac(n, edge_pairs, order[:k]))
                    S_runs.append(S)
                S = np.mean(S_runs, axis=0)
            else:
                order = order_deg if strategy == "degree_desc" else order_loop
                S = []
                for q in qs:
                    k = int(round(q * n))
                    S.append(lcc_frac(n, edge_pairs, order[:k]))
            auc = float(np.trapezoid(S, qs))
            qc = float("nan")
            below = np.where(np.array(S) < 0.5)[0]
            if len(below) > 0:
                qc = float(qs[below[0]])
            out.append(dict(network=name, strategy=strategy, AUC=auc, q_c=qc))
            curves[(name, strategy)] = (qs, S)
        print(f"[perc] {name} done", flush=True)
    with open(os.path.join(RES, "perc.pkl"), "wb") as f:
        pickle.dump(dict(rows=out, curves=curves), f)
    print("PERC DONE", flush=True)

# ================================================================ STAGE null models
def jsd(p, q, eps=1e-12):
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)

def spectrum_from_basis(basis, maxlen):
    counts = np.zeros(maxlen + 1)
    for cyc in basis:
        L = len(cyc)
        if L <= maxlen:
            counts[L] += 1
    return counts / len(basis)

def basis_hist(basis):
    """Return (counts array indexed by length, total cycles)."""
    if not basis:
        return None, 0
    mx = max(len(c) for c in basis)
    counts = np.zeros(mx + 1)
    for c in basis:
        counts[len(c)] += 1
    return counts, len(basis)

def stage_null():
    with open(os.path.join(RES, "core.pkl"), "rb") as f:
        core = pickle.load(f)["core"]
    out = []
    for name in NAMES:
        t0 = time.time()
        G = load_graph(name)
        n = G.number_of_nodes(); m = G.number_of_edges()
        k_basis = core[name]["k_basis"]
        obs_r = core[name]["r"]
        obs_max = len(obs_r) - 1
        obs_H = core[name]["H"]
        n_null = 20 if n > 1000 else 30
        er_hists = []; cm_hists = []; er_H = []; cm_H = []
        swaps = min(10 * m, 300000)
        for i in range(n_null):
            rng = np.random.default_rng(SEED + i)
            # ER G(n,m)
            Ger = nx.gnm_random_graph(n, m, seed=rng)
            Ger = nx.Graph(Ger); Ger.remove_edges_from(nx.selfloop_edges(Ger))
            if nx.number_connected_components(Ger) > 1:
                cc = sorted(nx.connected_components(Ger), key=len, reverse=True)
                Ger = Ger.subgraph(cc[0]).copy()
            try:
                b = nx.cycle_basis(Ger)
                h, kb = basis_hist(b)
                if h is not None:
                    er_hists.append(h)
                    hp = h / kb
                    er_H.append(-np.sum(hp[hp > 0] * np.log(hp[hp > 0])))
            except Exception as e:
                print('  ER fail', e, flush=True)
            # configuration model via double edge swap
            Gcm = G.copy()
            try:
                nx.double_edge_swap(Gcm, nswap=swaps, max_tries=swaps * 5, seed=rng)
            except Exception:
                pass
            Gcm.remove_edges_from(nx.selfloop_edges(Gcm))
            try:
                b = nx.cycle_basis(Gcm)
                h, kb = basis_hist(b)
                if h is not None:
                    cm_hists.append(h)
                    hp = h / kb
                    cm_H.append(-np.sum(hp[hp > 0] * np.log(hp[hp > 0])))
            except Exception as e:
                print('  CM fail', e, flush=True)
        # align on global support 3..Lmax
        Lmax = obs_max
        for h in er_hists + cm_hists:
            Lmax = max(Lmax, len(h) - 1)
        lo = 3
        def pad(h):
            a = np.zeros(Lmax + 1)
            a[:len(h)] = h
            return a[lo:Lmax + 1]
        p_obs = np.zeros(Lmax + 1)
        p_obs[:len(obs_r)] = obs_r
        p_obs = p_obs[lo:Lmax + 1]
        q_er = np.mean([pad(h) for h in er_hists], axis=0) if er_hists else np.zeros_like(p_obs)
        q_cm = np.mean([pad(h) for h in cm_hists], axis=0) if cm_hists else np.zeros_like(p_obs)
        jsd_er = jsd(p_obs, q_er)
        jsd_cm = jsd(p_obs, q_cm)
        out.append(dict(network=name, JSD_ER=jsd_er, JSD_CM=jsd_cm,
                        H_obs=obs_H,
                        H_ER_mean=float(np.mean(er_H)) if er_H else float("nan"),
                        H_CM_mean=float(np.mean(cm_H)) if cm_H else float("nan"),
                        n_er=len(er_hists), n_cm=len(cm_hists)))
        print(f"[null] {name:11s} JSD_ER={jsd_er:.4f} JSD_CM={jsd_cm:.4f} "
              f"n_er={len(er_hists)} n_cm={len(cm_hists)} Lmax={Lmax} t={time.time()-t0:.1f}s",
              flush=True)
    with open(os.path.join(RES, "null.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("NULL DONE", flush=True)

# ================================================================ STAGE tables
def stage_tables():
    with open(os.path.join(RES, "core.pkl"), "rb") as f:
        c = pickle.load(f)
    core, exact = c["core"], c["exact"]
    with open(os.path.join(RES, "pin.pkl"), "rb") as f:
        pin = pickle.load(f)
    with open(os.path.join(RES, "sir.pkl"), "rb") as f:
        sir = pickle.load(f); sir_rows = sir["rows"]; sir_thr = sir["thr"]
    with open(os.path.join(RES, "perc.pkl"), "rb") as f:
        perc = pickle.load(f); perc_rows = perc["rows"]; perc_curves = perc["curves"]
    with open(os.path.join(RES, "null.pkl"), "rb") as f:
        null = pickle.load(f)

    import pandas as pd
    # ---- network_stats.csv
    stats = []
    for name in NAMES:
        cc = core[name]; st = cc["struct"]
        stats.append(dict(
            network=name, n=cc["n"], m=cc["m"], LCC_size=cc["n"],
            density=st["density"], avg_degree=st["avg_degree"],
            max_degree=st["max_degree"], avg_clustering=st["avg_clustering"],
            transitivity=st["transitivity"],
            degree_assortativity=st["assortativity"],
            avg_shortest_path=st["avg_shortest_path"],
            diameter=st["diameter"], diameter_approx=st["asp_approx"],
            girth=cc["girth"], modularity=st["modularity"],
            degree_powerlaw_exponent=st["powerlaw_alpha"],
            lambda1=cc["lambda1"], lambda2=cc["lambda2"],
            kappa=cc["kappa"], k_basis=cc["k_basis"],
            rho_E=cc["rho_E"], rho_V=cc["rho_V"], mu_L=cc["mu_L"],
            H=cc["H"], H_norm=cc["H_norm"], C_hat=cc["C_hat"],
            mean_p=cc["mean_p"], CV_p=cc["CV_p"], frac_p0=cc["frac_p0"]))
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(os.path.join(ROOT, "network_stats.csv"), index=False)

    # ---- table_networks.csv
    tn = []
    for name in NAMES:
        cc = core[name]; st = cc["struct"]
        tn.append(dict(
            Network=DISPLAY[name], n=cc["n"], m=cc["m"],
            avg_degree=round(st["avg_degree"], 3),
            density=round(st["density"], 4),
            avg_clustering=round(st["avg_clustering"], 4),
            transitivity=round(st["transitivity"], 4),
            assortativity=round(st["assortativity"], 4),
            girth=cc["girth"], modularity=round(st["modularity"], 4),
            kappa=cc["kappa"], rho_E=round(cc["rho_E"], 4),
            rho_V=round(cc["rho_V"], 4), mu_L=round(cc["mu_L"], 3),
            H=round(cc["H"], 4), H_norm=round(cc["H_norm"], 4),
            C_hat=round(cc["C_hat"], 4),
            lambda1=round(cc["lambda1"], 3), lambda2=round(cc["lambda2"], 4)))
    pd.DataFrame(tn).to_csv(os.path.join(ROOT, "table_networks.csv"), index=False)

    # ---- pinning_summary.csv
    pd.DataFrame(pin).to_csv(os.path.join(ROOT, "pinning_summary.csv"), index=False)
    # ---- SIR
    pd.DataFrame(sir_rows).to_csv(os.path.join(ROOT, "SIR_summary.csv"), index=False)
    pd.DataFrame(sir_thr).to_csv(os.path.join(ROOT, "SIR_thresholds.csv"), index=False)
    # ---- percolation
    pd.DataFrame(perc_rows).to_csv(os.path.join(ROOT, "percolation_summary.csv"), index=False)
    # ---- null
    pd.DataFrame(null).to_csv(os.path.join(ROOT, "null_model_summary.csv"), index=False)

    # ---- correlation_table.csv
    # Build per-network feature matrix
    lrs_feats = ["rho_E", "rho_V", "mu_L", "H", "H_norm", "C_hat",
                 "mean_p", "CV_p", "frac_p0"]
    dyn = {}
    for t in sir_thr:
        dyn.setdefault(t["network"], {})["beta_c"] = t["beta_c"]
        dyn[t["network"]]["final_size_at_beta_0_25"] = t["final_size_at_beta_0_25"]
    for r in perc_rows:
        if r["strategy"] == "random":
            dyn.setdefault(r["network"], {})["percolation_AUC_random"] = r["AUC"]
            dyn[r["network"]]["q_c_random"] = r["q_c"]
    rows_corr = []
    feat_vecs = {f: np.array([core[n][f] for n in NAMES]) for f in lrs_feats}
    feat_vecs["kappa_over_m"] = np.array([core[n]["kappa"] / core[n]["m"] for n in NAMES])
    struct_feats = ["density", "avg_degree", "max_degree", "avg_clustering",
                    "transitivity", "degree_assortativity", "modularity",
                    "lambda1", "lambda2", "girth", "powerlaw_alpha"]
    dyn_feats = ["beta_c", "final_size_at_beta_0_25",
                 "percolation_AUC_random", "q_c_random"]
    struct_vecs = {}
    for sf in struct_feats:
        key = sf if sf != "powerlaw_alpha" else "powerlaw_alpha"
        if sf in ("density", "avg_degree", "max_degree", "avg_clustering",
                  "transitivity", "degree_assortativity", "modularity",
                  "powerlaw_alpha"):
            struct_vecs[sf] = np.array([core[n]["struct"][
                "powerlaw_alpha" if sf == "powerlaw_alpha" else
                {"density": "density", "avg_degree": "avg_degree",
                 "max_degree": "max_degree",
                 "avg_clustering": "avg_clustering",
                 "transitivity": "transitivity",
                 "degree_assortativity": "assortativity",
                 "modularity": "modularity"}[sf]] for n in NAMES])
        elif sf == "lambda1":
            struct_vecs[sf] = np.array([core[n]["lambda1"] for n in NAMES])
        elif sf == "lambda2":
            struct_vecs[sf] = np.array([core[n]["lambda2"] for n in NAMES])
        elif sf == "girth":
            struct_vecs[sf] = np.array([core[n]["girth"] for n in NAMES])
    dyn_vecs = {df: np.array([dyn[n][df] for n in NAMES]) for df in dyn_feats}
    all_right = {**struct_vecs, **dyn_vecs}
    for lf, lv in feat_vecs.items():
        for rf, rv in all_right.items():
            msk = (~np.isnan(lv)) & (~np.isnan(rv))
            if msk.sum() < 4 or np.std(lv[msk]) == 0 or np.std(rv[msk]) == 0:
                continue
            try:
                sr, sp = spearmanr(lv[msk], rv[msk])
                pr, pp = pearsonr(lv[msk], rv[msk])
                rows_corr.append(dict(metric_pair=f"{lf}__{rf}",
                                       spearman_rho=float(sr), spearman_p=float(sp),
                                       pearson_r=float(pr), pearson_p=float(pp)))
            except Exception:
                pass
    pd.DataFrame(rows_corr).to_csv(os.path.join(ROOT, "correlation_table.csv"), index=False)
    print("TABLES DONE", flush=True)

# ================================================================ STAGE figures
def stage_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # font
    if "Times New Roman" in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.linewidth"] = 1.0
    C = plt.get_cmap("tab10").colors

    with open(os.path.join(RES, "core.pkl"), "rb") as f:
        c = pickle.load(f)
    core, exact = c["core"], c["exact"]
    with open(os.path.join(RES, "pin.pkl"), "rb") as f:
        pin = pickle.load(f)
    with open(os.path.join(RES, "sir.pkl"), "rb") as f:
        sir = pickle.load(f); sir_rows = sir["rows"]; sir_thr = sir["thr"]
    with open(os.path.join(RES, "perc.pkl"), "rb") as f:
        perc = pickle.load(f); perc_rows = perc["rows"]; perc_curves = perc["curves"]
    with open(os.path.join(RES, "null.pkl"), "rb") as f:
        null = pickle.load(f)
    import pandas as pd

    # ---- fig1 LRS spectra
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, name in zip(axes.flat, NAMES):
        r = core[name]["r"]
        ls = np.arange(len(r))
        msk = r > 0
        ax.bar(ls[msk], r[msk], width=0.8, color=C[0], edgecolor="k", linewidth=0.3)
        ax.set_title(f"{DISPLAY[name]}  " + r"($\rho_E$=" + f"{core[name]['rho_E']:.2f}, "
                     + r"$\hat H$=" + f"{core[name]['H_norm']:.2f})", fontsize=10)
        ax.set_xlabel(r"cycle length $\ell$"); ax.set_ylabel(r"$r_\ell$")
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "fig1_lrs.png"), dpi=300)
    plt.close(fig)

    # ---- fig2 loop participation
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, name in zip(axes.flat, NAMES):
        p = core[name]["p"]
        nz = p[p > 0]
        ax.hist(nz, bins=20, color=C[1], alpha=0.8, edgecolor="k", lw=0.3)
        ax.set_yscale("log")
        ax.set_title(f"{DISPLAY[name]}  " + r"$\rho_V$=" + f"{core[name]['rho_V']:.2f}",
                     fontsize=10)
        ax.set_xlabel(r"$p_v$"); ax.set_ylabel("count (log)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "fig2_loop_participation.png"), dpi=300)
    plt.close(fig)

    # ---- fig3 correlation heatmap
    ct = pd.read_csv(os.path.join(ROOT, "correlation_table.csv"))
    lrs_feats = ["rho_E", "rho_V", "mu_L", "H", "H_norm", "C_hat",
                 "mean_p", "CV_p", "frac_p0", "kappa_over_m"]
    right_labels = sorted(set(p.split("__")[1] for p in ct["metric_pair"]))
    M = np.full((len(lrs_feats), len(right_labels)), np.nan)
    for _, row in ct.iterrows():
        lf, rf = row["metric_pair"].split("__")
        if lf in lrs_feats and rf in right_labels:
            M[lrs_feats.index(lf), right_labels.index(rf)] = row["spearman_rho"]
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(right_labels))); ax.set_xticklabels(right_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(lrs_feats))); ax.set_yticklabels(lrs_feats, fontsize=9)
    for i in range(len(lrs_feats)):
        for j in range(len(right_labels)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Spearman $\\rho$")
    ax.set_title("LRS features vs structural/dynamic metrics (Spearman)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "fig3_correlation_heatmap.png"), dpi=300)
    plt.close(fig)

    # ---- fig4 SIR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, name in enumerate(NAMES):
        rs = [r for r in sir_rows if r["network"] == name]
        bs = [r["beta"] for r in rs]; fs = [r["final_size"] for r in rs]; pk = [r["peak"] for r in rs]
        a1.plot(bs, fs, "-o", color=C[i], ms=3, lw=1.2, label=DISPLAY[name])
        a2.plot(bs, pk, "-o", color=C[i], ms=3, lw=1.2, label=DISPLAY[name])
        thr = [t for t in sir_thr if t["network"] == name][0]
        if not np.isnan(thr["beta_c"]):
            a1.axvline(thr["beta_c"], color=C[i], ls="--", lw=0.7, alpha=0.6)
    for a, ttl in [(a1, "Final size"), (a2, "Peak prevalence")]:
        a.set_xscale("log"); a.set_xlabel(r"$\beta$"); a.set_ylabel(ttl)
        a.legend(fontsize=7, ncol=2); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "fig4_sir.png"), dpi=300)
    plt.close(fig)

    # ---- fig5 pinning
    rep = [("dolphins", "Dolphins"), ("polbooks", "Polbooks"),
           ("email", "Email-Eu-core"), ("powergrid", "US power grid")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    styles = {"random": ("o", C[0], "-"), "degree_desc": ("s", C[1], "--"),
              "loop_desc": ("^", C[2], "-.")}
    for ax, (name, lbl) in zip(axes.flat, rep):
        for strat, (mk, col, ls) in styles.items():
            fs = sorted(set(r["f"] for r in pin if r["network"] == name))
            ys = []
            for fv in fs:
                m = [r["lambda_min"] for r in pin
                     if r["network"] == name and r["f"] == fv and r["strategy"] == strat]
                ys.append(m[0] if m else np.nan)
            ax.plot(fs, ys, marker=mk, color=col, ls=ls, ms=5, label=strat)
        ax.set_xscale("log"); ax.set_xlabel("pinning fraction $f$")
        ax.set_ylabel(r"$\lambda_{min}(L_g)$"); ax.set_title(lbl)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "fig5_pinning.png"), dpi=300)
    plt.close(fig)

    # ---- fig6 percolation
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, (name, strat) in enumerate([("email", "random"), ("email", "degree_desc"),
                                      ("email", "loop_desc"), ("powergrid", "random"),
                                      ("powergrid", "degree_desc"), ("powergrid", "loop_desc")]):
        qs, S = perc_curves[(name, strat)]
        lbl = f"{DISPLAY[name]}-{strat}"
        a1.plot(qs, S, lw=1.3, label=lbl)
    a1.set_xlabel("removed fraction $q$"); a1.set_ylabel("$S(q)$ (LCC fraction)")
    a1.legend(fontsize=7); a1.grid(alpha=0.3); a1.set_title("Robustness curves")
    # grouped bar AUC
    strat_names = ["random", "degree_desc", "loop_desc"]
    x = np.arange(len(NAMES)); w = 0.25
    for j, strat in enumerate(strat_names):
        aucs = [next(r["AUC"] for r in perc_rows if r["network"] == n and r["strategy"] == strat)
                for n in NAMES]
        a2.bar(x + (j - 1) * w, aucs, w, label=strat, color=C[j])
    a2.set_xticks(x); a2.set_xticklabels([DISPLAY[n] for n in NAMES], rotation=45, ha="right", fontsize=8)
    a2.set_ylabel("Robustness AUC"); a2.legend(fontsize=8); a2.set_title("AUC by strategy")
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "fig6_percolation.png"), dpi=300)
    plt.close(fig)

    # ---- fig7 null model
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(NAMES)); w = 0.35
    a1.bar(x - w/2, [r["JSD_ER"] for r in null], w, label="JSD vs ER", color=C[0])
    a1.bar(x + w/2, [r["JSD_CM"] for r in null], w, label="JSD vs CM", color=C[1])
    a1.set_xticks(x); a1.set_xticklabels([DISPLAY[n] for n in NAMES], rotation=45, ha="right", fontsize=8)
    a1.set_ylabel("JSD"); a1.legend(); a1.set_title("Spectrum divergence from null models")
    ho = [r["H_obs"] for r in null]; he = [r["H_ER_mean"] for r in null]
    a2.scatter(ho, he, c=[C[i] for i in range(len(NAMES))], s=60, zorder=3)
    lo = min(min(ho), min(np.nan_to_num(he, nan=min(ho)))) - 0.1
    hi = max(max(ho), max(np.nan_to_num(he, nan=max(ho)))) + 0.1
    a2.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
    for i, name in enumerate(NAMES):
        a2.annotate(DISPLAY[name], (ho[i], he[i]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    a2.set_xlabel(r"$H_{obs}$"); a2.set_ylabel(r"$H_{ER,mean}$"); a2.legend()
    a2.set_title("Observed vs ER-null entropy"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "fig7_null_model.png"), dpi=300)
    plt.close(fig)

    # ---- fig8 scatter correlations
    ct = pd.read_csv(os.path.join(ROOT, "correlation_table.csv"))
    # strongest LRS feature vs final_size_at_beta_0_25
    sub = ct[ct["metric_pair"].str.endswith("__final_size_at_beta_0_25")].copy()
    sub["absr"] = sub["spearman_rho"].abs()
    top = sub.sort_values("absr", ascending=False).iloc[0]
    lf_left = top["metric_pair"].split("__")[0]
    fs_vals = np.array([t["final_size_at_beta_0_25"] for t in sir_thr])
    if lf_left == "kappa_over_m":
        lf_vals = np.array([core[n]["kappa"] / core[n]["m"] for n in NAMES])
    else:
        lf_vals = np.array([core[n][lf_left] for n in NAMES])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for i, name in enumerate(NAMES):
        a1.scatter(lf_vals[i], fs_vals[i], color=C[i], s=60, zorder=3)
        a1.annotate(DISPLAY[name], (lf_vals[i], fs_vals[i]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    a1.set_xlabel(lf_left); a1.set_ylabel("final size at $\\beta$=0.25")
    a1.set_title(f"{lf_left} vs epidemic final size\n"
                 f"Spearman $\\rho$={top['spearman_rho']:.3f}, p={top['spearman_p']:.3f}")
    a1.grid(alpha=0.3)
    # rho_E vs AUC_random
    rhoE = np.array([core[n]["rho_E"] for n in NAMES])
    auc_r = np.array([next(r["AUC"] for r in perc_rows if r["network"] == n and r["strategy"] == "random")
                      for n in NAMES])
    sr2, sp2 = spearmanr(rhoE, auc_r)
    for i, name in enumerate(NAMES):
        a2.scatter(rhoE[i], auc_r[i], color=C[i], s=60, zorder=3)
        a2.annotate(DISPLAY[name], (rhoE[i], auc_r[i]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    a2.set_xlabel(r"$\rho_E$"); a2.set_ylabel("percolation AUC (random)")
    a2.set_title(f"$\\rho_E$ vs robustness AUC\n"
                 f"Spearman $\\rho$={sr2:.3f}, p={sp2:.3f}")
    a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "fig8_scatter_correlations.png"), dpi=300)
    plt.close(fig)
    print("FIGS DONE", flush=True)

# ================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["core", "pin", "sir", "perc", "null",
                             "tables", "figs", "all"])
    args = ap.parse_args()
    stages = {"core": stage_core, "pin": stage_pin, "sir": stage_sir,
              "perc": stage_perc, "null": stage_null,
              "tables": stage_tables, "figs": stage_figs}
    if args.stage == "all":
        for st in ["core", "pin", "sir", "perc", "null", "tables", "figs"]:
            stages[st]()
    else:
        stages[args.stage]()
