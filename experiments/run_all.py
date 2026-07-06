"""Run the full experiment suite (fixed seeds; ~3-6 minutes on a laptop)."""
import time
import e1_diagonality, e2_oos, e3_identities, e4_frontier, e5_robustness, e6_intervention, e7_scaling
for mod in [e1_diagonality, e3_identities, e4_frontier, e5_robustness, e7_scaling, e6_intervention, e2_oos]:
    t0 = time.time(); print(f"==== {mod.__name__} ===="); mod.run()
    print(f"[{mod.__name__}: {time.time()-t0:.1f}s]\n")
print("ALL EXPERIMENTS DONE")
