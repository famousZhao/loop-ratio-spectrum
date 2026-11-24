# -*- coding: utf-8 -*-
"""CWRU 12k drive-end: normal / IR007 / Ball007 / OR007@6, four loads each."""
import os, json
import numpy as np
import networkx as nx
import scipy.io as sio
from scipy.stats import kruskal

from paths import CWRU, RES
DIR = CWRU
OUT = RES
WIN, STEP, DIM, RR, MAX_WIN, FS = 4096, 8, 3, 0.05, 40, 12000
GROUPS = {"97": ("normal", 0), "98": ("normal", 1), "99": ("normal", 2), "100": ("normal", 3),
          "105": ("inner", 0), "106": ("inner", 1), "107": ("inner", 2), "108": ("inner", 3),
          "118": ("ball", 0), "119": ("ball", 1), "120": ("ball", 2), "121": ("ball", 3),
          "130": ("outer", 0), "131": ("outer", 1), "132": ("outer", 2), "133": ("outer", 3)}
FEATS = ["kappa_over_m", "mu_L", "H", "H_norm", "C_hat", "lcc"]

def build(seg, tau):
    s = (seg - seg.mean()) / (seg.std() + 1e-12)
    N = len(s) - (DIM - 1) * tau
    X = np.array([s[i:i + (DIM - 1) * tau + 1:tau] for i in range(N)])
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
    return dict(kappa_over_m=(m - n + c) / max(m, 1),
                rho_E=1 - len(list(nx.bridges(G))) / max(m, 1),
                rho_V=float(np.count_nonzero(pv) / n), mu_L=float(np.sum(np.arange(len(r)) * r)),
                H=H, H_norm=Hn, C_hat=float(r.max()), c=c,
                lcc=max(len(x) for x in nx.connected_components(G)) / n)

def load_de(code):
    d = sio.loadmat(os.path.join(DIR, code + ".mat"))
    keys = [k for k in d.keys() if k.endswith("_DE_time")]
    x = np.asarray(d[keys[0]]).ravel().astype(float)
    return x

def extract(tau):
    rows = []
    for code, (cls, load) in GROUPS.items():
        x = load_de(code)
        nwin = min(MAX_WIN, len(x) // WIN)
        for w in range(nwin):
            f = feat(build(x[w * WIN:(w + 1) * WIN][::STEP], tau))
            f.update(record=code, cls=cls, load=load)
            rows.append(f)
    return rows

def evaluate(rows):
    X = np.array([[r[k] for k in FEATS] for r in rows], float)
    cls = np.array([r["cls"] for r in rows]); rec = np.array([r["record"] for r in rows])
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    classes = ("normal", "inner", "ball", "outer")
    ok = 0; wrong = []
    for u in sorted(set(rec)):
        tr, te = rec != u, rec == u
        cents = {c: Z[tr & (cls == c)].mean(0) for c in classes}
        dd = {c: float(np.linalg.norm(Z[te] - v, axis=1).mean()) for c, v in cents.items()}
        pr = min(dd, key=dd.get); ok += int(pr == cls[te][0])
        if pr != cls[te][0]: wrong.append((u, cls[te][0], pr))
    return ok, wrong

if __name__ == "__main__":
    lengths = {c: len(load_de(c)) for c in GROUPS}
    print("lengths:", lengths)
    for tau in (16, 4):
        print("\n--- tau = %d (%.2f ms) ---" % (tau, 1000 * tau / FS), flush=True)
        rows = extract(tau)
        ok, wrong = evaluate(rows)
        print("windows=%d  LORO %d/16 (chance 1/4)" % (len(rows), ok), flush=True)
        print("  wrong:", wrong)
        # group means + Kruskal
        for k in ["kappa_over_m", "rho_E", "rho_V", "mu_L", "H", "H_norm", "C_hat", "c", "lcc"]:
            recs = sorted(set(r["record"] for r in rows))
            g = {c: np.array([np.mean([r[k] for r in rows if r["record"] == u])
                              for u in recs if next(r["cls"] for r in rows if r["record"] == u) == c])
                 for c in ("normal", "inner", "ball", "outer")}
            print("  %-12s %s   Kruskal p=%.4f" %
                  (k, " ".join("%.3f" % v.mean() for v in g.values()), kruskal(*g.values())[1]))
        json.dump(rows, open(os.path.join(OUT, "cwru_tau%d.json" % tau), "w"))
