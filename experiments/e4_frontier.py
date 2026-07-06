"""E4 -- The conditional frontier is real: predicted conditional Sharpe of the
projected-Markowitz portfolio vs realized standardized OOS return, binned.
Oracle parameters (pure theory validation) and estimated parameters
(attenuation from estimation error, reported honestly)."""
from common import *
from causal_markowitz import NormalFormDGP, estimators, optimizer

def path_stats(dgp, sim, params, T0, gamma=3.0):
    a, G, Q = params
    A = np.ones((1, dgp.n)); b = np.array([1.0])
    M, w0 = optimizer.M_and_w0(Q, A, b)
    pred, real = [], []
    for t in range(T0, sim["R"].shape[0]):
        mu = a + G @ sim["Z"][t]
        w = w0 + gamma * (M @ mu)
        s2 = w @ Q @ w
        pred.append((w @ mu) / np.sqrt(s2))
        real.append((sim["R"][t] @ w) / np.sqrt(s2))
    return np.array(pred), np.array(real)

def run(seed=41):
    dgp = NormalFormDGP(n=50, m=3, seed=3, prem_scale=0.25)
    sim = dgp.simulate(30000, rng=np.random.default_rng(seed))
    p_o, r_o = path_stats(dgp, sim, (dgp.a, dgp.G, dgp.Q), 500)
    fit = estimators.fit_causal(sim["R"][:500], sim["Z"][:501])
    p_e, r_e = path_stats(dgp, sim, (fit["a"], fit["G"], fit["Q"]), 500)
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True)
    for axi, (p, r, tt) in zip(ax, [(p_o, r_o, "oracle parameters"),
                                    (p_e, r_e, "estimated ($T=500$)")]):
        qs = np.quantile(p, np.linspace(0, 1, 13))
        cx, cy, se = [], [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m_ = (p >= lo) & (p < hi)
            cx.append(p[m_].mean()); cy.append(r[m_].mean())
            se.append(r[m_].std() / np.sqrt(m_.sum()))
        axi.errorbar(cx, cy, yerr=2*np.array(se), fmt="o", ms=4, color="C0",
                     label="binned realized mean")
        lim = [min(cx)-0.02, max(cx)+0.02]
        axi.plot(lim, lim, "k--", lw=1, label=r"45$^\circ$")
        sl = np.polyfit(p, r, 1)[0]
        axi.set_title(tt + f"  (slope {sl:.2f})")
        axi.set_xlabel(r"predicted conditional Sharpe $w^{*\top}\mu_t/\sqrt{w^{*\top}Qw^*}$")
    ax[0].set_ylabel("realized standardized return")
    ax[0].legend(frameon=False, fontsize=8)
    savefig(fig, "e4_frontier.pdf")
    print("slopes:", np.polyfit(p_o, r_o, 1)[0], np.polyfit(p_e, r_e, 1)[0])

if __name__ == "__main__": run()
