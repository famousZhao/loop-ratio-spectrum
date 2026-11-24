# -*- coding: utf-8 -*-
"""Download and build all 8 networks -> data/<name>.txt (undirected simple LCC, relabeled 0..n-1)."""
import os, io, ssl, gzip, tarfile, zipfile, urllib.request
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
HDR = {"User-Agent":"Mozilla/5.0","Referer":"https://networkrepository.com/"}

def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()

def save_raw(tag, b):
    p = os.path.join(DATA, f"_raw_{tag}.bin"); open(p,"wb").write(b); return p

def finalize(G, name):
    G = nx.Graph(G)  # undirected simple
    G.remove_edges_from(nx.selfloop_edges(G))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    H = G.subgraph(comps[0]).copy()
    H = nx.relabel_nodes(H, {n:i for i,n in enumerate(sorted(H.nodes()))})
    H = nx.relabel_nodes(H, {n:i for i,n in enumerate(H.nodes())})  # ensure 0..n-1
    n, m = H.number_of_nodes(), H.number_of_edges()
    with open(os.path.join(DATA, f"{name}.txt"), "w") as f:
        for u,v in H.edges():
            f.write(f"{u} {v}\n")
    print(f"[BUILT] {name:14s} n={n:5d} m={m:6d} c={nx.number_connected_components(G)}", flush=True)
    return H, n, m

def parse_mtx(b):
    txt = b.decode("utf-8","replace")
    edges=[]; header=[]
    for line in txt.splitlines():
        if line.startswith("%") or not line.strip(): continue
        parts=line.split()
        if len(parts)==3 and all(p.lstrip("-").isdigit() for p in parts) and len(header)<1:
            # dimension line
            header.append(parts); continue
        if len(parts)>=2 and parts[0].lstrip("-").isdigit():
            edges.append((int(parts[0]), int(parts[1])))
    return edges

def load_gml(z, suffix=".gml"):
    import re
    name = [n for n in z.namelist() if n.endswith(suffix)][0]
    txt = z.read(name).decode("utf-8","replace")
    kept = [ln for ln in txt.splitlines() if not ln.lstrip().startswith("Creator")]
    try:
        return nx.parse_gml("\n".join(kept))
    except Exception:
        # fallback: regex extract edges (source target) and node ids
        G = nx.Graph()
        for m in re.finditer(r"source\s+(\S+)\s+target\s+(\S+)", txt):
            G.add_edge(m.group(1), m.group(2))
        if G.number_of_edges()==0:
            for m in re.finditer(r"id\s+(\S+)", txt):
                G.add_node(m.group(1))
        return G

sources_log = {}

# 1 Karate
G = nx.karate_club_graph()
finalize(G, "karate"); sources_log["karate"]="networkx builtin nx.karate_club_graph()"

# 2 Dolphins (Newman gml)
b = fetch("http://www-personal.umich.edu/~mejn/netdata/dolphins.zip"); save_raw("dolphins",b)
z=zipfile.ZipFile(io.BytesIO(b)); G=load_gml(z); finalize(G,"dolphins"); sources_log["dolphins"]="http://www-personal.umich.edu/~mejn/netdata/dolphins.zip"

# 3 Polbooks (nrvis mtx)
b = fetch("http://nrvis.com/download/data/misc/polbooks.zip"); save_raw("polbooks",b)
z=zipfile.ZipFile(io.BytesIO(b)); mtx=z.read([n for n in z.namelist() if n.endswith(".mtx")][0])
G=nx.Graph(); G.add_edges_from(parse_mtx(mtx)); finalize(G,"polbooks"); sources_log["polbooks"]="http://nrvis.com/download/data/misc/polbooks.zip (MatrixMarket)"

# 4 C. elegans neural (Newman gml)
b = fetch("http://www-personal.umich.edu/~mejn/netdata/celegansneural.zip"); save_raw("celegans",b)
z=zipfile.ZipFile(io.BytesIO(b)); G=load_gml(z); finalize(G,"celegans"); sources_log["celegans"]="http://www-personal.umich.edu/~mejn/netdata/celegansneural.zip"

# 5 US power grid (Newman gml)
b = fetch("http://www-personal.umich.edu/~mejn/netdata/power.zip"); save_raw("power",b)
z=zipfile.ZipFile(io.BytesIO(b)); G=load_gml(z); finalize(G,"powergrid"); sources_log["powergrid"]="http://www-personal.umich.edu/~mejn/netdata/power.zip"

# 6 Email-Eu-core (SNAP gz, directed -> undirected)
b = fetch("https://snap.stanford.edu/data/email-Eu-core.txt.gz"); save_raw("email",b)
txt=gzip.decompress(b).decode()
G=nx.Graph()
for line in txt.splitlines():
    p=line.split()
    if len(p)>=2: G.add_edge(int(p[0]),int(p[1]))
finalize(G,"email"); sources_log["email"]="https://snap.stanford.edu/data/email-Eu-core.txt.gz"

# 7 Euroroad (konect tar.bz2)
b = fetch("http://konect.cc/files/download.tsv.subelj_euroroad.tar.bz2"); save_raw("euroroad",b)
tf=tarfile.open(fileobj=io.BytesIO(b), mode="r:bz2")
members=tf.getmembers()
member=[m for m in members if "out." in m.name and m.isfile()]
if not member:
    member=[m for m in members if m.isfile()]
member=member[0]
data=tf.extractfile(member).read().decode()
G=nx.Graph()
for line in data.splitlines():
    if line.startswith("%") or not line.strip(): continue
    p=line.split()
    if len(p)>=2 and p[0].lstrip("-").isdigit(): G.add_edge(int(p[0]),int(p[1]))
finalize(G,"euroroad"); sources_log["euroroad"]="http://konect.cc/files/download.tsv.subelj_euroroad.tar.bz2"

# 8 Yeast (nrvis bio-yeast.mtx)
b = fetch("http://nrvis.com/download/data/bio/bio-yeast.zip"); save_raw("yeast",b)
z=zipfile.ZipFile(io.BytesIO(b)); mtx=z.read([n for n in z.namelist() if n.endswith(".mtx")][0])
G=nx.Graph(); G.add_edges_from(parse_mtx(mtx)); finalize(G,"yeast"); sources_log["yeast"]="http://nrvis.com/download/data/bio/bio-yeast.zip (MatrixMarket)"

import json
open(os.path.join(DATA,"_sources.json"),"w").write(json.dumps(sources_log,indent=2))
print("DONE")
