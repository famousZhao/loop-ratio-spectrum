# -*- coding: utf-8 -*-
"""Preset parameter grid: embedding dimension crossed with recurrence
rate.  All nine combinations are reported; none is selected."""
import os, sys, json, time
import numpy as np
import networkx as nx
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bearing_mfpt import RECORDS as MFPT_RECORDS
from paths import MFPT, RES

OUT = RES
WIN, STEP, TAU, MAX_WIN = 4096, 8, 16, 40
FEATS = ["kappa_over_m", "mu_L", "H", "H_norm", "C_hat", "lcc"]
SETTINGS = [(d, rr) for d in (2, 3, 4) for rr in (0.03, 0.05, 0.08)]

def build(gs, w, DIM, RR):
    seg = gs[w * WIN:(w + 1) * WIN][::STEP]
    s = (seg - seg.mean()) / (seg.std() + 1e-12)
    N = len(s) - (DIM - 1) * TAU
    X = np.array([s[i:i + (DIM - 1) * TAU + 1:TAU] for i in range(N)])
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    eps = np.quantile(D[np.triu_indices(N, 1)], RR)
    A = (D < eps).astype(int); np.fill_diagonal(A, 0)
    return nx.from_numpy_array(A)

def feat(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    c = nx.number_connected_components(G)
    basis = nx.cycle_basis(G); k = max(len(basis), 1)
    lens = np.array([len(b) for b in basis], float)
    counts = np.zeros(int(lens.max()) + 1)
    for L in lens: counts[int(L)] += 1
    r = counts / k
    pos = r[r > 0]
    H = float(-np.sum(pos * np.log(pos))); Hn = H / np.log(len(pos)) if len(pos) > 1 else 0.0
    pv = np.zeros(n)
    for cy in basis:
        for v in cy: pv[v] += 1
    pv /= k
    return dict(n=n, m=m, kappa_over_m=(m - n + c) / max(m, 1),
                rho_E=1 - len(list(nx.bridges(G))) / max(m, 1),
                rho_V=float(np.count_nonzero(pv) / n), mu_L=float(np.sum(np.arange(len(r)) * r)),
                H=H, H_norm=Hn, C_hat=float(r.max()),
                lcc=max(len(x) for x in nx.connected_components(G)) / n)

def evaluate(rows):
    X = np.array([[r[f] for f in FEATS] for r in rows], float)
    cls = np.array([r["cls"] for r in rows]); rec = np.array([r["record"] for r in rows])
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0; preds = []
    for u in sorted(set(rec)):
        tr, te = rec != u, rec == u
        cents = {c: Z[tr & (cls == c)].mean(0) for c in ("healthy", "outer", "inner")}
        dd = {c: float(np.linalg.norm(Z[te] - v, axis=1).mean()) for c, v in cents.items()}
        pr = min(dd, key=dd.get); ok += int(pr == cls[te][0]); preds.append((u, pr))
    # record-level variant
    ok2 = 0
    recs = sorted(set(rec))
    means = {u: X[rec == u].mean(0) for u in recs}
    for u in recs:
        tr = rec != u
        cents = {}
        for c in ("healthy", "outer", "inner"):
            idx = [w for w in recs if cls[rec == w][0] == c and w != u]
            cents[c] = np.mean([means[w] for w in idx], axis=0)
        pr = min(cents, key=lambda c: np.linalg.norm(means[u] - cents[c]))
        ok2 += int(pr == cls[rec == u][0])
    return ok, ok2, preds

if __name__ == "__main__":
    res = {}
    recfile = os.path.join(OUT, "grid_results.json")
    if os.path.exists(recfile):
        res = json.load(open(recfile))
    for DIM, RR in SETTINGS:
        key = "dim%d_rr%d" % (DIM, int(RR * 100))
        if key in res:
            print("skip", key, flush=True); continue
        t0 = time.time(); rows = []
        for fname, cls, sub in MFPT_RECORDS:
            d = sio.loadmat(os.path.join(MFPT, sub, fname))["bearing"]
            fs = int(np.asarray(d["sr"][0][0]).ravel()[0])
            gs = np.asarray(d["gs"][0][0]).ravel().astype(float)
            if fs >= 90000: gs = gs[::2]
            nwin = min(MAX_WIN, len(gs) // WIN)
            for w in range(nwin):
                f = feat(build(gs, w, DIM, RR))
                f.update(record=fname, cls=cls)
                rows.append(f)
        ok_w, ok_r, preds = evaluate(rows)
        res[key] = dict(window=ok_w, record=ok_r, n=len(rows),
                        wrong=[p for p in preds if p[1] != dict((r["record"], r["cls"]) for r in rows)[p[0]]],
                        secs=round(time.time() - t0, 1))
        json.dump(res, open(recfile, "w"))
        print(key, "win %d/17 rec %d/17  (%.0fs)" % (ok_w, ok_r, res[key]["secs"]), flush=True)
    print("\nGRID SUMMARY")
    for k, v in res.items():
        print("%-12s window %2d/17   record %2d/17" % (k, v["window"], v["record"]))
