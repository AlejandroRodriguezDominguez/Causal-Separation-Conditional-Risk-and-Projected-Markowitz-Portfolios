"""E7 -- Two-stage computation: exactness and O(n k^2) scaling vs dense O(n^3)."""
from common import *
from causal_markowitz import NormalFormDGP, optimizer

def run(seed=71):
    k = 10; ns = [250, 500, 1000, 2000, 4000]
    t_dense, t_two, maxdiff = [], [], []
    for n_ in ns:
        dgp = NormalFormDGP(n=n_, m=k, seed=1)
        mu = dgp.mu_cond(np.random.default_rng(seed).normal(size=k))
        A = np.ones((1, n_)); b = np.array([1.0])
        reps = 3
        t0 = time.perf_counter()
        for _ in range(reps): w_d, _ = optimizer.kkt_solve(dgp.Q, mu, A, b, 3.0)
        t_dense.append((time.perf_counter() - t0) / reps)
        t0 = time.perf_counter()
        for _ in range(reps):
            w_2 = optimizer.two_stage_solve(dgp.varsigma**2, dgp.B, dgp.Lam, mu, A, b, 3.0)
        t_two.append((time.perf_counter() - t0) / reps)
        maxdiff.append(np.abs(w_d - w_2).max())
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.loglog(ns, t_dense, "o-", color="C3", label="dense KKT")
    ax.loglog(ns, t_two, "s-", color="C0", label="two-stage (Prop.~4.8)")
    s_d = np.polyfit(np.log(ns), np.log(t_dense), 1)[0]
    s_t = np.polyfit(np.log(ns), np.log(t_two), 1)[0]
    ax.set_xlabel(r"$n$"); ax.set_ylabel("seconds per solve")
    ax.legend(frameon=False, title=f"slopes: {s_d:.2f} vs {s_t:.2f}", fontsize=8)
    savefig(fig, "e7_scaling.pdf")
    print("max |w_dense - w_two| over n:", max(maxdiff), "slopes", s_d, s_t)
    return {"maxdiff": max(maxdiff), "slopes": (s_d, s_t),
            "t_dense": t_dense, "t_two": t_two}

if __name__ == "__main__": run()
