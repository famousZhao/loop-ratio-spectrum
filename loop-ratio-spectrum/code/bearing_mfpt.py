"""Bearing-state networks built from the MFPT vibration records.

Each record is cut into non-overlapping windows; a window is subsampled,
delay embedded, and turned into an epsilon-recurrence graph whose epsilon
links five percent of the point pairs.  The descriptors of that graph are
the loop ratio features used in Section 9 of the paper.

Output: results/mfpt_records.json (per-record descriptors, class means,
load trends, leave-one-record-out predictions, attack comparison).
"""
import json
import os
import time

import numpy as np
import networkx as nx
import scipy.io as sio
from scipy.stats import kruskal, spearmanr

from attacks import attack_curve, summ
from paths import MFPT, RES

WIN, STEP, DIM, TAU = 4096, 8, 3, 16
RR = 0.05
MAX_WIN = 40
SEED = 20261008

RECORDS = [
    ("baseline_1.mat", "healthy", "1 - Three Baseline Conditions"),
    ("baseline_2.mat", "healthy", "1 - Three Baseline Conditions"),
    ("baseline_3.mat", "healthy", "1 - Three Baseline Conditions"),
]
for _i in range(1, 8):
    RECORDS.append(("OuterRaceFault_vload_%d.mat" % _i, "outer",
                    "3 - Seven More Outer Race Fault Conditions"))
for _i in range(1, 8):
    RECORDS.append(("InnerRaceFault_vload_%d.mat" % _i, "inner",
                    "4 - Seven Inner Race Fault Conditions"))

KEYS = ["kappa_over_m", "rho_E", "rho_V", "mu_L", "H", "H_norm", "C_hat", "lcc", "c"]


def load_record(fname, sub):
    d = sio.loadmat(os.path.join(MFPT, sub, fname))["bearing"]
    fs = int(np.asarray(d["sr"][0][0]).ravel()[0])
    load = int(np.asarray(d["load"][0][0]).ravel()[0])
    gs = np.asarray(d["gs"][0][0]).ravel().astype(float)
    return fs, load, gs


def build(gs, w):
    seg = gs[w * WIN:(w + 1) * WIN][::STEP]
    s = (seg - seg.mean()) / (seg.std() + 1e-12)
    N = len(s) - (DIM - 1) * TAU
    X = np.array([s[i:i + (DIM - 1) * TAU + 1:TAU] for i in range(N)])
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    eps = np.quantile(D[np.triu_indices(N, 1)], RR)
    A = (D < eps).astype(int)
    np.fill_diagonal(A, 0)
    return nx.from_numpy_array(A)


def descriptors(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    c = nx.number_connected_components(G)
    kappa = m - n + c
    basis = nx.cycle_basis(G)
    k = max(len(basis), 1)
    lens = np.array([len(b) for b in basis], float)
    counts = np.zeros(int(lens.max()) + 1)
    for L in lens:
        counts[int(L)] += 1
    r = counts / k
    pos = r[r > 0]
    H = float(-np.sum(pos * np.log(pos)))
    Hn = H / np.log(len(pos)) if len(pos) > 1 else 0.0
    pv = np.zeros(n)
    for cy in basis:
        for v in cy:
            pv[v] += 1
    pv /= k
    nbr = len(list(nx.bridges(G)))
    lcc = max(len(x) for x in nx.connected_components(G)) / n
    return dict(n=n, m=m, kappa=kappa, kappa_over_m=kappa / m, c=c,
                rho_E=1 - nbr / m, rho_V=float(np.count_nonzero(pv) / n),
                mu_L=float(np.sum(np.arange(len(r)) * r)), H=H, H_norm=Hn,
                C_hat=float(r.max()), lcc=lcc, p=pv)


def main():
    t0 = time.time()
    feats = []
    for fname, cls, sub in RECORDS:
        fs, load, gs = load_record(fname, sub)
        if fs >= 90000:
            gs = gs[::2]
        nwin = min(MAX_WIN, len(gs) // WIN)
        for w in range(nwin):
            f = descriptors(build(gs, w))
            f.pop("p")
            f.update(record=fname, cls=cls, win=w, load=load)
            feats.append(f)
        print("  %-28s windows=%d" % (fname, nwin), flush=True)

    A = {k: np.array([f[k] for f in feats]) for k in KEYS}
    cls = np.array([f["cls"] for f in feats])
    rec = np.array([f["record"] for f in feats])
    ulist = sorted(set(rec))
    loads = {f["record"]: f["load"] for f in feats}

    rows = []
    for u in ulist:
        msk = rec == u
        row = dict(record=u, cls=cls[msk][0], load=loads[u], n_win=int(msk.sum()))
        for k in KEYS:
            row[k] = float(np.mean(A[k][msk]))
            row[k + "_sd"] = float(np.std(A[k][msk], ddof=1))
        rows.append(row)

    stats = {}
    for k in KEYS:
        vals = {c: np.array([r[k] for r in rows if r["cls"] == c])
                for c in ("healthy", "outer", "inner")}
        H_, p_ = kruskal(vals["healthy"], vals["outer"], vals["inner"])
        stats[k] = dict(kruskal_H=float(H_), p=float(p_),
                        mean={c: float(np.mean(v)) for c, v in vals.items()},
                        sd={c: float(np.std(v, ddof=1)) for c, v in vals.items()})

    trend = {}
    for k in ("kappa_over_m", "mu_L", "H", "C_hat", "c"):
        for c in ("outer", "inner"):
            v = np.array([r[k] for r in rows if r["cls"] == c])
            l = np.array([r["load"] for r in rows if r["cls"] == c])
            rho_, p_ = spearmanr(l, v)
            trend["%s:%s" % (k, c)] = [float(rho_), float(p_)]
    for k in ("kappa_over_m", "mu_L", "H", "C_hat", "c"):
        v = np.array([r[k] for r in rows if r["cls"] != "healthy"])
        l = np.array([r["load"] for r in rows if r["cls"] != "healthy"])
        rho_, p_ = spearmanr(l, v)
        trend["%s:pooled" % k] = [float(rho_), float(p_)]

    X = np.column_stack([A[k] for k in ("kappa_over_m", "mu_L", "H", "H_norm", "C_hat", "lcc")])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    correct = 0
    preds = []
    for u in ulist:
        tr, te = rec != u, rec == u
        cents = {c: Z[tr & (cls == c)].mean(0) for c in ("healthy", "outer", "inner")}
        dist = {c: float(np.linalg.norm(Z[te] - v, axis=1).mean()) for c, v in cents.items()}
        pr = min(dist, key=dist.get)
        preds.append([u, cls[te][0], pr])
        correct += int(pr == cls[te][0])

    print("  attack subset", flush=True)
    att = {}
    for c in ("healthy", "outer", "inner"):
        acc = {"random": [], "degree": [], "loop": [], "betweenness": []}
        subset = [r for r in RECORDS if r[1] == c][:3]
        for fname, _, sub in subset:
            fs, load, gs = load_record(fname, sub)
            if fs >= 90000:
                gs = gs[::2]
            for w in range(8):
                G = build(gs, w)
                n = G.number_of_nodes()
                basis = nx.cycle_basis(G)
                pv = np.zeros(n)
                for cy in basis:
                    for v in cy:
                        pv[v] += 1
                pv /= max(len(basis), 1)
                degs = np.array([x for _, x in G.degree()], float)
                bc = nx.betweenness_centrality(G, normalized=True, seed=SEED)
                orders = {"degree": np.argsort(-degs), "loop": np.argsort(-pv),
                          "betweenness": np.argsort(-np.array([bc[i] for i in range(n)])),
                          "random": np.random.default_rng(SEED).permutation(n)}
                for k, o in orders.items():
                    qs, S = attack_curve(G, o, 0.02)
                    acc[k].append(summ(qs, S)[0])
        att[c] = {k: dict(mean_auc=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))
                  for k, v in acc.items()}
        print("    %-7s loop=%.3f btw=%.3f deg=%.3f rand=%.3f"
              % (c, att[c]["loop"]["mean_auc"], att[c]["betweenness"]["mean_auc"],
                 att[c]["degree"]["mean_auc"], att[c]["random"]["mean_auc"]), flush=True)

    out = dict(record_rows=rows, stats=stats, trend=trend,
               loro=dict(accuracy=correct / len(ulist), n_records=len(ulist), preds=preds),
               attack=att, n_windows=len(feats),
               protocol=dict(window=WIN, step=STEP, dim=DIM, tau=TAU, rr=RR, max_win=MAX_WIN))
    with open(os.path.join(RES, "mfpt_records.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("done in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
