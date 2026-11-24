# -*- coding: utf-8 -*-
"""Classic baselines on the CWRU windows, same 4-class leave-one-record-out protocol."""
import os, json
import numpy as np
import scipy.io as sio
import scipy.stats as st
from scipy.signal import hilbert

from paths import CWRU, RES
DIR = CWRU
WIN, STEP, MAX_WIN = 4096, 8, 40
GROUPS = {"97": ("normal", 0), "98": ("normal", 1), "99": ("normal", 2), "100": ("normal", 3),
          "105": ("inner", 0), "106": ("inner", 1), "107": ("inner", 2), "108": ("inner", 3),
          "118": ("ball", 0), "119": ("ball", 1), "120": ("ball", 2), "121": ("ball", 3),
          "130": ("outer", 0), "131": ("outer", 1), "132": ("outer", 2), "133": ("outer", 3)}

rows = []
for code, (cls, load) in GROUPS.items():
    d = sio.loadmat(os.path.join(DIR, code + ".mat"))
    key = [k for k in d.keys() if k.endswith("_DE_time")][0]
    x = np.asarray(d[key]).ravel().astype(float)
    nwin = min(MAX_WIN, len(x) // WIN)
    for w in range(nwin):
        seg = x[w * WIN:(w + 1) * WIN][::STEP]
        seg = seg - seg.mean()
        rms = float(np.sqrt((seg ** 2).mean()))
        env = np.abs(hilbert(seg))
        E = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(len(env)))); E[0] = 0
        rows.append({"record": code, "cls": cls,
                     "rms": rms,
                     "kurt": float(st.kurtosis(seg, fisher=True)),
                     "crest": float(np.abs(seg).max() / (rms + 1e-12)),
                     "skewness": float(st.skew(seg)),
                     "env_peak": float(E.max() / (E.mean() + 1e-12))})

def evaluate(feats):
    X = np.array([[r[k] for k in feats] for r in rows], float)
    cls = np.array([r["cls"] for r in rows]); rec = np.array([r["record"] for r in rows])
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    classes = ("normal", "inner", "ball", "outer")
    ok, wrong = 0, []
    for u in sorted(set(rec)):
        tr, te = rec != u, rec == u
        cents = {c: Z[tr & (cls == c)].mean(0) for c in classes}
        dd = {c: float(np.linalg.norm(Z[te] - v, axis=1).mean()) for c, v in cents.items()}
        pr = min(dd, key=dd.get); ok += int(pr == cls[te][0])
        if pr != cls[te][0]: wrong.append((u, cls[te][0], pr))
    return ok, wrong

print("windows:", len(rows))
summary = {}
for name, fl in [("RMS", ["rms"]), ("kurtosis", ["kurt"]), ("crest", ["crest"]),
                 ("skewness", ["skewness"]), ("env peak ratio", ["env_peak"]),
                 ("classic 5", ["rms", "kurt", "crest", "skewness", "env_peak"])]:
    ok, wrong = evaluate(fl)
    summary[name] = ok
    print("%-16s %d/16  wrong=%s" % (name, ok, wrong if ok < 14 else ""))

for c in ("normal", "inner", "ball", "outer"):
    rr = [r for r in rows if r["cls"] == c]
    print(" %-7s rms %.3f kurt %6.2f crest %5.2f env %6.2f" %
          (c, np.mean([r["rms"] for r in rr]), np.mean([r["kurt"] for r in rr]),
           np.mean([r["crest"] for r in rr]), np.mean([r["env_peak"] for r in rr])))

json.dump(dict(rows=rows, leave_one_record_out=summary),
          open(os.path.join(RES, "baseline_features_cwru.json"), "w"))
