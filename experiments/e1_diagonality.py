"""E1 -- Conditional factorization and the two windows.

Validates: (i) residuals conditioned on the horizon-closed driver information
are cross-sectionally independent (Fisher-z size at nominal level); (ii)
residuals conditioned on the past window only are NOT (power ~ 1), because the
common driver innovation is still live -- the empirical content of Rem. 2.3
and Prop. 2.7 vs Eq. (Q)."""
from common import *
from causal_markowitz import NormalFormDGP, estimators

def fisher_z_reject(resid, alpha, dof_adj):
    T, n = resid.shape
    Cm = np.corrcoef(resid.T)
    z = np.sqrt(max(T - dof_adj - 3, 1)) * np.arctanh(np.clip(Cm, -0.999, 0.999))
    from scipy.stats import norm
    crit = norm.ppf(1 - alpha / 2)
    iu = np.triu_indices(n, 1)
    return (np.abs(z[iu]) > crit)

def run(seed=11):
    n, m, T, Rreps = 30, 3, 500, 400
    alphas = np.linspace(0.01, 0.20, 12)
    rej_closed = np.zeros((Rreps, len(alphas))); rej_past = np.zeros((Rreps, len(alphas)))
    rng = np.random.default_rng(seed)
    for r in range(Rreps):
        dgp = NormalFormDGP(n=n, m=m, seed=rng.integers(1e9))
        sim = dgp.simulate(T, rng=np.random.default_rng(rng.integers(1e9)))
        res_c = estimators.fit_causal(sim["R"], sim["Z"])["resid"]
        res_p = estimators.fit_past_only(sim["R"], sim["Z"])["resid"]
        for j, a in enumerate(alphas):
            rej_closed[r, j] = fisher_z_reject(res_c, a, 1 + 2 * m).mean()
            rej_past[r, j] = fisher_z_reject(res_p, a, 1 + m).mean()
    # heatmaps from one long run
    dgp = NormalFormDGP(n=n, m=m, seed=7); sim = dgp.simulate(20000)
    Cc = np.corrcoef(estimators.fit_causal(sim["R"], sim["Z"])["resid"].T)
    Cp = np.corrcoef(estimators.fit_past_only(sim["R"], sim["Z"])["resid"].T)
    np.fill_diagonal(Cc, 0); np.fill_diagonal(Cp, 0)
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.9),
                           gridspec_kw={"width_ratios": [1, 1, 1.25]})
    v = max(np.abs(Cp).max(), 0.2)
    im0 = ax[0].imshow(Cp, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[0].set_title(r"residuals on $Z_t$ (past)", fontsize=8.5)
    im1 = ax[1].imshow(Cc, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[1].set_title(r"residuals on $(Z_t,\hat\eta_{t+1})$ (closed)", fontsize=8.5)
    for a_ in ax[:2]: a_.grid(False)
    fig.colorbar(im1, ax=ax[:2], shrink=0.8, pad=0.02)
    ax[2].plot(alphas, rej_past.mean(0), "o-", color="C3", label="past window (power)")
    ax[2].plot(alphas, rej_closed.mean(0), "s-", color="C0", label="closed window (size)")
    ax[2].plot(alphas, alphas, "k--", lw=1, label=r"45$^\circ$ (nominal)")
    ax[2].set_xlabel(r"nominal level $\alpha$"); ax[2].set_ylabel("rejection rate")
    ax[2].legend(frameon=False); ax[2].set_ylim(-0.02, 1.02)
    savefig(fig, "e1_diagonality.pdf")
    out = {"maxabs_offdiag_past": float(np.abs(Cp).max()),
           "maxabs_offdiag_closed": float(np.abs(Cc).max()),
           "size_at_5": float(rej_closed.mean(0)[np.argmin(np.abs(alphas-0.05))]),
           "power_at_5": float(rej_past.mean(0)[np.argmin(np.abs(alphas-0.05))])}
    print(json.dumps(out, indent=1)); return out

if __name__ == "__main__": run()
