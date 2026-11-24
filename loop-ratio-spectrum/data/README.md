# Data

## Benchmark networks

`karate.txt`, `dolphins.txt`, `polbooks.txt`, `celegans.txt`, `powergrid.txt`,
`email.txt`, `euroroad.txt`, `yeast.txt` hold one edge per line as two
zero-based integer node indices.  Each file is the largest connected component
of the corresponding public network with nodes relabelled `0..n-1`.

Original sources, all public:

| file | source |
| --- | --- |
| karate | Zachary karate club, as distributed with NetworkX |
| dolphins | Lusseau dolphin social network, http://www-personal.umich.edu/~mejn/netdata/ |
| polbooks | Newman, books about US politics, http://www-personal.umich.edu/~mejn/netdata/ |
| celegans | C. elegans neural network, http://www-personal.umich.edu/~mejn/netdata/ |
| powergrid | US western power grid, http://www-personal.umich.edu/~mejn/netdata/ |
| email | email-Eu-core, https://snap.stanford.edu/data/email-Eu-core.html |
| euroroad | European road network, KONECT / NR |
| yeast | yeast protein interaction network, KONECT / NR |

`code/build_graphs.py` downloads the archives and rebuilds these files, so the
edge lists here can be checked against the original sources.

## Bearing recordings

The two bearing experiments read third-party archives that are not
redistributed in this repository:

* **MFPT** - Machinery Failure Prevention Technology Society bearing data,
  https://www.mfpt.org/fault-data-sets/ .  Unpack the archives so that
  `data/mfpt/<condition>/<record>.mat` exists, where `<condition>` is one of
  `1 - Three Baseline Conditions`, `3 - Seven More Outer Race Fault Conditions`,
  `4 - Seven Inner Race Fault Conditions`, `6 - Real World Examples`.  The
  pipeline uses the first four baseline recordings, the seven outer-race and
  the seven inner-race recordings.
* **CWRU** - Case Western Reserve University Bearing Data Center,
  https://engineering.case.edu/bearingdatacenter .  Place the drive-end
  recordings `97.mat` ... `100.mat`, `105.mat` ... `108.mat`, `118.mat` ...
  `121.mat` and `130.mat` ... `133.mat` in `data/cwru/`.  All of them are 12 kHz
  drive-end signals.

Record identifiers, class labels and load levels are listed in
`code/bearing_mfpt.py` and `code/cwru.py`.
