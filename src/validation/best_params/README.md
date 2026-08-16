# Persisted best parameter vectors

These four files are the exact `best_params.csv` outputs read by
`replay_best_cmaes.py`, `replay_best_gwo.py`, `replay_best_pso.py`, and
`replay_best_qpso.py` for the report's Level 2 replay validation
(Section 16.2, Table 16.2). Each is the winning 30-variable vector
(5 parameters x 6 joints) for that algorithm on Path 1
(I -> _), copied unmodified from the optimization run.

Provenance (from the report's own Section 16.2 "Provenance disclosure"):

| File | Algorithm | Source run | Source best J |
|---|---|---|---|
| `cmaes_best_params.csv` | CMA-ES | Original Path 1 Run 3 | 0.640563 |
| `gwo_best_params.csv` | GWO | Fresh 60-evaluation re-run (post-fix) | 0.661123 |
| `pso_best_params.csv` | PSO | Fresh 60-evaluation re-run (post-fix) | 0.647897 |
| `qpso_best_params.csv` | QPSO | Original Path 1 Run 3 | 0.664796 |

`best_params.csv` persistence originally existed only in the CMA-ES
script. It was added to GWO and PSO during the project, which could
not retroactively recover their already-completed Run 3 vectors from
the original campaign, so their replay vectors instead come from a
fresh 60-evaluation search run after the persistence fix (source J
0.661123 and 0.647897, close to but not identical with the original
Run 3 values 0.661045 and 0.645937, as expected for stochastic
searches). QPSO's persistence existed from when `qpso_optimizer.py`
was first written, so its file is the original Run 3 vector directly.

CMA-ES's vector is also the one shown in Figure 15.1 (the
kinematically-inactive-joints figure) and referenced by Section 11.5's
bound-saturation count; PSO's and GWO's vectors are the ones Section
11.5 counts its "17 of 30" and "1 of 30" bound-saturation figures
from, respectively.
