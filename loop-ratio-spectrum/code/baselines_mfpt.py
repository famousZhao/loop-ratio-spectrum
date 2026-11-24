# -*- coding: utf-8 -*-
"""Window-level amplitude baselines on the MFPT records (RMS, kurtosis,
skewness, crest factor, envelope peak ratio, and the five classical
time-domain features taken jointly)."""
import os, json, glob
import numpy as np
from scipy.io import loadmat
import scipy.stats as st
from scipy.signal import hilbert

from paths import MFPT, RES
ROOT = MFPT
WIN, MAX_WIN = 4096, 40
SUBS = [("1 - Three Baseline Conditions", "healthy"),
        ("3 - Seven More Outer Race Fault Conditions", "outer"),
        ("4 - Seven Inner Race Fault Conditions", "inner")]

rows = []
for sub, cls in SUBS:
    for f in sorted(glob.glob(os.path.join(ROOT, sub, "*.mat"))):
        d = loadmat(f)["bearing"]
        fsr = int(np.asarray(d["sr"][0][0]).ravel()[0])
        gs = np.asarray(d["gs"][0][0]).ravel().astype(float)
        if fsr >= 90000:
            gs = gs[::2]
        nwin = min(MAX_WIN, len(gs) // WIN)
        for w in range(nwin):
            x = gs[w * WIN:(w + 1) * WIN]
            x = x - x.mean()
            rms = float(np.sqrt((x ** 2).mean()))
            kur = float(st.kurtosis(x, fisher=True))
            crest = float(np.abs(x).max() / (rms + 1e-12))
            skw = float(st.skew(x))
            env = np.abs(hilbert(x))
            E = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(len(env)))
                       )
            E[0] = 0.0
            pk = float(E.max() / (E.mean() + 1e-12))
            rows.append({"record": os.path.basename(f), "cls": cls, "win": w,
                         "rms": rms, "kurt": kur, "crest": crest,
                         "skewness": skw, "env_peak": pk})
print("windows:", len(rows), "records:", len(set(r["record"] for r in rows)))

def evaluate(feats):
    X = np.array([[r[k] for k in feats] for r in rows], float)
    cls = np.array([r["cls"] for r in rows]); rec = np.array([r["record"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    ok, preds = 0, []
    for u in sorted(set(rec)):
        tr, te = rec != u, rec == u
        cents = {c: Z[tr & (cls == c)].mean(0) for c in ("healthy", "outer", "inner")}
        dd = {c: float(np.linalg.norm(Z[te] - v, axis=1).mean()) for c, v in cents.items()}
        pr = min(dd, key=dd.get)
        preds.append((u, cls[te][0], pr)); ok += int(pr == cls[te][0])
    return ok, preds, X

tests = [("RMS", ["rms"]), ("kurtosis", ["kurt"]), ("crest factor", ["crest"]),
         ("skewness", ["skewness"]), ("env peak ratio", ["env_peak"]),
         ("RMS + kurtosis", ["rms", "kurt"]),
         ("classic 5", ["rms", "kurt", "crest", "skewness", "env_peak"])]
for name, fl in tests:
    ok, preds, X = evaluate(fl)
    print("%-16s %d/17" % (name, ok))

for c in ("healthy", "outer", "inner"):
    rr = [r for r in rows if r["cls"] == c]
    print("%-8s rms %.4f kurt %.2f crest %.2f env %.2f" %
          (c, np.mean([r["rms"] for r in rr]), np.mean([r["kurt"] for r in rr]),
           np.mean([r["crest"] for r in rr]), np.mean([r["env_peak"] for r in rr])))

agg = {c: [np.mean([r["kurt"] for r in rows if r["record"] == n])
           for n in sorted(set(r["record"] for r in rows if r["cls"] == c))]
       for c in ("healthy", "outer", "inner")}
print("kurtosis Kruskal p = %.5f" % st.kruskal(*agg.values())[1])
json.dump(rows, open(os.path.join(RES, "baseline_features_mfpt.json"), "w"))
print("saved classic_feats.json")
