"""E5 -- Approximate separation.  Inject residual cross-correlation of maximal
size eps (the value of the dep functional) and compare, over eps:
  (i)  the actual displacement of the solution and of the frontier potential;
  (ii) the EXACT first-order sensitivity of Prop. 5.1,
           dw = -M dQ w*,    dDelta = -(M mu)' dQ (M mu);
  (iii) the distribution-free worst-case bound of Cor. 5.2.
(ii) should overlay (i) up to O(eps^2); (iii) is a uniform guarantee over all
dependence patterns at the same eps and is conservative by construction."""
from common import *
from causal_markowitz import NormalFormDGP, optimizer

def run(seed=51):
    rng = np.random.default_rng(seed)
    dgp = NormalFormDGP(n=60, m=3, seed=13, prem_scale=0.25)
    mu = dgp.mu_cond(rng.normal(size=3))
    A = np.ones((1, dgp.n)); b = np.array([1.0]); gam = 3.0
    M, w0 = optimizer.M_and_w0(dgp.Q, A, b)
    w_star = w0 + gam * M @ mu
    Delta0 = mu @ M @ mu
    s = dgp.varsigma
    E1 = np.outer(s, s) - np.diag(s**2)               # unit-eps injected pattern
    eps_grid = np.linspace(0, 0.30, 16)
    dw_act, dD_act = [], []
    for eps in eps_grid:
        Qe = dgp.Q + eps * E1
        Me, w0e = optimizer.M_and_w0(Qe, A, b)
        we = w0e + gam * Me @ mu
        dw_act.append(np.linalg.norm(we - w_star))
        dD_act.append(abs(mu @ Me @ mu - Delta0))
    # exact first order (Prop 5.1) for this pattern
    dw_fo = eps_grid * np.linalg.norm(M @ (E1 @ w_star))
    dD_fo = eps_grid * abs((M @ mu) @ E1 @ (M @ mu))
    # distribution-free worst case (Cor 5.2)
    dw_bnd = eps_grid * (np.linalg.norm(s)**2) * np.linalg.norm(w_star) / s.min()**2
    dD_bnd = eps_grid * (s @ np.abs(M @ mu))**2
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.9))
    for a_, act, fo, bnd, lbl in [
        (ax[0], dw_act, dw_fo, dw_bnd, r"$\|w^*_\epsilon-w^*\|$"),
        (ax[1], dD_act, dD_fo, dD_bnd, r"$|\Delta_\epsilon-\Delta|$")]:
        a_.plot(eps_grid, act, "o", ms=4, color="C0", label="actual " + lbl)
        a_.plot(eps_grid, fo, "-", color="C0", lw=1.2, label="exact first order (Prop.~5.1)")
        a_.plot(eps_grid, bnd, "k--", lw=1, label="worst-case bound (Cor.~5.2)")
        a_.set_yscale("log"); a_.set_xlabel(r"residual dependence $\epsilon$")
        a_.set_ylim(bottom=1e-6)
        a_.legend(frameon=False, fontsize=7.2, loc="lower right")
    savefig(fig, "e5_robustness.pdf")
    i, j = 5, -1
    print(f"eps={eps_grid[i]:.2f}: act {dw_act[i]:.3f} fo {dw_fo[i]:.3f} | "
          f"eps={eps_grid[j]:.2f}: act {dw_act[j]:.3f} fo {dw_fo[j]:.3f}")
    print(f"Delta: act {dD_act[i]:.5f} fo {dD_fo[i]:.5f}")
    print(f"conservatism at eps=0.10: dw bound/actual {dw_bnd[i]/dw_act[i]:.0f}x, "
          f"dDelta bound/actual {dD_bnd[i]/dD_act[i]:.0f}x")

if __name__ == "__main__": run()
