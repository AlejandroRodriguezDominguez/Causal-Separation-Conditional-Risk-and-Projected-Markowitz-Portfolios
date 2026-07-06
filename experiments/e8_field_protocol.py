"""E8 -- field-data protocol runner (paper, Sec. 6.9).

Usage on real data:
    python3 e8_field_protocol.py returns.csv drivers.csv report.csv
Without arguments it runs a smoke test on a synthetic panel written to CSV, to
verify the full pipeline (load -> rolling windows -> greedy selection with
sample splitting -> holdout certification -> PCA reference -> report)
end to end. The smoke test is a pipeline check, not an experiment of the paper.
"""
import sys, csv, numpy as np
sys.path.insert(0, "../src")
from causal_markowitz import NormalFormDGP
from causal_markowitz.field_protocol import run_protocol

if len(sys.argv) == 4:
    run_protocol(sys.argv[1], sys.argv[2], sys.argv[3])
else:
    rng = np.random.default_rng(81)
    dgp = NormalFormDGP(n=20, m=2, seed=81)
    sim = dgp.simulate(1600, rng=rng)
    P = sim["Z"][:, 0] + 0.2 * rng.normal(size=len(sim["Z"]))
    W = rng.normal(size=len(sim["Z"]))
    dates = [f"d{t:05d}" for t in range(len(sim["R"]))]
    with open("/tmp/ret.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date"] + [f"a{i}" for i in range(20)])
        for t, d in enumerate(dates): w.writerow([d] + list(sim["R"][t]))
    with open("/tmp/drv.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "Z1", "Z2", "P", "Wn"])
        for t, d in enumerate(dates): w.writerow([d] + [sim["Z"][t, 0], sim["Z"][t, 1], P[t], W[t]])
    recs = run_protocol("/tmp/ret.csv", "/tmp/drv.csv", "/tmp/field_report.csv",
                        window=756, step=252)
    for r in recs: print(r)
