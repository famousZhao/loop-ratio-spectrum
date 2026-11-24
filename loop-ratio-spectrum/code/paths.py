"""Filesystem locations shared by the scripts.

Inputs are read from DATA and outputs are written to RES.  Both can be
overridden with the environment variables LRS_DATA and LRS_RES, which is
how the batch runs redirect everything into a scratch directory.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("LRS_DATA", os.path.join(ROOT, "data"))
RES = os.environ.get("LRS_RES", os.path.join(ROOT, "results"))
FIG = os.environ.get("LRS_FIG", os.path.join(ROOT, "figures"))
MFPT = os.path.join(DATA, "mfpt")
CWRU = os.path.join(DATA, "cwru")

for _d in (RES, FIG):
    os.makedirs(_d, exist_ok=True)
