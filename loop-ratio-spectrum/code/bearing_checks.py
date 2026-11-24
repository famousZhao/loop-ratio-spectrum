"""Two controls on the bearing application.

1. Decimation control: the three healthy records are sampled at 97,656 Hz and
   are decimated by two before windowing.  This recomputes them at the full
   rate with a physically equivalent window and compares the descriptors.
2. Load control: at load levels shared by both fault classes, compare the
   outer-race and inner-race records directly.

Requires results/mfpt_records.json (run bearing_mfpt.py first).
Output: results/bearing_checks.json
"""
import json
import os

import numpy as np
import networkx as nx
from scipy.stats import wilcoxon, spearmanr

from bearing_mfpt import DIM, RR, RECORDS, descriptors, load_record
from paths import RES

KEYS = ["kappa_over_m", "rho_V", "mu_L", "H", "H_norm", "C_hat"]

with open(os.path.join(RES, "mfpt_records.json")) as f:
    saved = json.load(f)
rows = saved["record_rows"]

print("decimation control (healthy records at the full sampling rate)")
full = []
for fname, cls, sub in RECORDS:
    if cls != "healthy":
        continue
    fs, load, gs = load_record(fname, sub)
    win, step, tau = 2 * 4096, 2 * 8, 2 * 16
    nwin = min(40, len(gs) // win)
    vals = []
    for w in range(nwin):
        seg = gs[w * win:(w + 1) * win][::step]
        s = (seg - seg.mean()) / (seg.std() + 1e-12)
        N = len(s) - (DIM - 1) * tau
        X = np.array([s[i:i + (DIM - 1) * tau + 1:tau] for i in range(N)])
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        eps = np.quantile(D[np.triu_indices(N, 1)], RR)
        A = (D < eps).astype(int)
        np.fill_diagonal(A, 0)
        f = descriptors(nx.from_numpy_array(A))
        vals.append([f[k] for k in KEYS])
    m = np.mean(np.array(vals), axis=0)
    full.append(m)
    print("  %-14s mu_L=%.2f H=%.2f C_hat=%.3f rho_V=%.3f"
          % (fname, m[2], m[3], m[5], m[1]))

dec = np.array([[r[k] for k in KEYS] for r in rows if r["cls"] == "healthy"])
print("  decimated mean  mu_L=%.2f H=%.2f C_hat=%.3f rho_V=%.3f"
      % (dec[:, 2].mean(), dec[:, 3].mean(), dec[:, 5].mean(), dec[:, 1].mean()))

print("\nload control (shared load levels)")
rec = {r["record"]: r for r in rows}
loads = sorted({r["load"] for r in rows if r["cls"] != "healthy"})
pairs = []
for L in loads:
    o = [k for k, r in rec.items() if r["cls"] == "outer" and r["load"] == L]
    i = [k for k, r in rec.items() if r["cls"] == "inner" and r["load"] == L]
    if o and i:
        pairs.append((L, rec[o[0]], rec[i[0]]))
        print("  %4d lb  mu_L %.2f vs %.2f   H %.2f vs %.2f"
              % (L, rec[o[0]]["mu_L"], rec[i[0]]["mu_L"],
                 rec[o[0]]["H"], rec[i[0]]["H"]))

for k in ("mu_L", "H"):
    a = np.array([p[1][k] for p in pairs])
    b = np.array([p[2][k] for p in pairs])
    st = wilcoxon(a, b)
    print("  %-4s paired Wilcoxon over %d loads: p=%.4f, mean gap=%.3f"
          % (k, len(pairs), st.pvalue, np.mean(a - b)))

for c in ("outer", "inner"):
    ks = [r for r in rows if r["cls"] == c]
    l = [r["load"] for r in ks]
    for k in ("mu_L", "H"):
        rho, p = spearmanr(l, [r[k] for r in ks])
        print("  within %-6s Spearman(load, %-4s) = %+.3f (p=%.3f)" % (c, k, rho, p))

out = dict(full_rate=np.array(full).tolist(),
           decimated=dec.tolist(),
           pairs=[[p[0], p[1]["record"], p[2]["record"]] for p in pairs])
with open(os.path.join(RES, "bearing_checks.json"), "w") as f:
    json.dump(out, f, indent=1)
