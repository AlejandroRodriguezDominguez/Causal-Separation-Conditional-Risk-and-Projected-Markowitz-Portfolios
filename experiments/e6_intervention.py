"""E6 -- Causal vs correlational separator under intervention (rebalanced).

Universe: true drivers {Z1, Z2}, a tight proxy P = Z1 + tau*noise
(corr(P,Z1) ~ 0.98), and the DGP unchanged throughout.  Portfolios rebalance
every step using their fitted response maps.  At mid-test the proxy relation
is broken -- do(P): P becomes independent noise with the same marginal law --
an intervention on a non-parent of returns.  Measured out of sample, per step:
annualized Sharpe and one-step variance calibration (realized/predicted),
pre and post intervention, plus the in-sample dep-functional value that a
certificate would already report."""
from common import *
from causal_markowitz import NormalFormDGP, estimators, optimizer

def rolling_eval(Rte, Dte, fit, gam=3.0):
    """Rebalance each step with fixed fitted params, current state z_t."""
    n = Rte.shape[1]
    A = np.ones((1, n)); b = np.array([1.0])
    M, w0 = optimizer.M_and_w0(fit["Q"], A, b)
    x, pv = [], []
    for t in range(Rte.shape[0]):
        mu = fit["a"] + fit["G"] @ Dte[t]
        w = w0 + gam * (M @ mu)
        x.append(Rte[t] @ w); pv.append(w @ fit["Q"] @ w)
    x = np.array(x); pv = np.array(pv)
    sh = np.mean(x) / np.std(x, ddof=1) * np.sqrt(252)
    # one-step calibration: realized squared demeaned innovation / predicted var
    mu_pred = np.array([fit["a"] @ np.zeros(1).sum() for _ in x])  # placeholder
    return sh, x, pv

def run(seed=61):
    reps, T, T_out = 250, 750, 250
    tau = 0.20
    rng = np.random.default_rng(seed)
    res = {k: {"sh_pre": [], "sh_post": [], "vr_pre": [], "vr_post": [],
               "ic_pre": [], "ic_post": [], "eps_in": [], "to": [], "gross": []}
           for k in ["causal", "proxy"]}
    lattice = {}
    for r in range(reps):
        dgp = NormalFormDGP(n=40, m=2, seed=rng.integers(1e9), prem_scale=0.035)
        lrng = np.random.default_rng(rng.integers(1e9))
        sim = dgp.simulate(T + 2 * T_out, rng=lrng)
        sdZ1 = sim["Z"][:, 0].std()
        P = sim["Z"][:, 0] + tau * sdZ1 * lrng.normal(size=T + 2 * T_out + 1)
        P_int = P.copy()
        P_int[T + T_out:] = sdZ1 * np.sqrt(1 + tau**2) * lrng.normal(size=T_out + 1)
        D_true = sim["Z"]
        D_prox = np.column_stack([P_int, sim["Z"][:, 1]])
        Rtr = sim["R"][:T]
        fits = {"causal": estimators.fit_causal(Rtr, D_true[:T + 1]),
                "proxy": estimators.fit_causal(Rtr, np.column_stack([P[:T+1], sim["Z"][:T+1, 1]]))}
        # lattice search over all nonempty subsets of {Z1, Z2, P, W}
        # (W: pure-noise candidate drawn from an rng stream keyed on the rep index,
        #  so all pre-existing draws and results above are bit-identical)
        wrng = np.random.default_rng(10_000_000 + r)
        Wn = wrng.normal(size=T + 2 * T_out + 1)
        cand = {"Z1": sim["Z"][:, 0], "Z2": sim["Z"][:, 1], "P": P, "W": Wn}
        from itertools import combinations
        for kk in range(1, 5):
            for combo in combinations(["Z1", "Z2", "P", "W"], kk):
                Dm = np.column_stack([cand[c] for c in combo])
                fsub = estimators.fit_causal(Rtr, Dm[:T + 1])
                Cm2 = np.corrcoef(fsub["resid"].T); np.fill_diagonal(Cm2, 0)
                lattice.setdefault("+".join(combo), []).append(np.abs(Cm2).max())
        for k, fit in fits.items():
            Cm = np.corrcoef(fit["resid"].T); np.fill_diagonal(Cm, 0)
            res[k]["eps_in"].append(np.abs(Cm).max())
            D_all = D_true if k == "causal" else D_prox
            A = np.ones((1, dgp.n)); b = np.array([1.0])
            M, w0 = optimizer.M_and_w0(fit["Q"], A, b)
            for tag, sl in [("pre", slice(T, T + T_out)), ("post", slice(T + T_out, T + 2 * T_out))]:
                xs, cal, mus, rls = [], [], [], []
                for t in range(sl.start, sl.stop):
                    mu = fit["a"] + fit["G"] @ D_all[t]
                    w = w0 + 3.0 * (M @ mu)
                    x = sim["R"][t] @ w
                    xs.append(x)
                    cal.append((x - w @ mu)**2 / (w @ fit["Q"] @ w))
                    mus.append(mu); rls.append(sim["R"][t])
                    if tag == "pre" and t > sl.start:
                        res[k]["to"].append(np.abs(w - wlast).sum())
                        res[k]["gross"].append(np.abs(w).sum())
                    wlast = w
                xs = np.array(xs)
                res[k][f"sh_{tag}"].append(np.mean(xs)/np.std(xs, ddof=1)*np.sqrt(252))
                res[k][f"vr_{tag}"].append(np.mean(cal))
                mus = np.array(mus).ravel(); rls = np.array(rls).ravel()
                res[k][f"ic_{tag}"].append(np.corrcoef(mus, rls)[0, 1])
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 2.9))
    labels = ["pre-intervention", r"post do($P$)"]
    xpos = np.arange(2); wdt = 0.36
    for i, k in enumerate(["causal", "proxy"]):
        mv = [np.mean(res[k]["sh_pre"]), np.mean(res[k]["sh_post"])]
        sv = [2*np.std(res[k]["sh_pre"])/np.sqrt(reps), 2*np.std(res[k]["sh_post"])/np.sqrt(reps)]
        ax[0].bar(xpos + (i - .5) * wdt, mv, wdt, yerr=sv, label=f"{k} separator", color=f"C{i}")
        mv = [np.mean(res[k]["ic_pre"]), np.mean(res[k]["ic_post"])]
        sv = [2*np.std(res[k]["ic_pre"])/np.sqrt(reps), 2*np.std(res[k]["ic_post"])/np.sqrt(reps)]
        ax[1].bar(xpos + (i - .5) * wdt, mv, wdt, yerr=sv, color=f"C{i}")
    ax[0].set_xticks(xpos, labels); ax[0].set_ylabel("OOS annualized Sharpe (rebalanced)")
    ax[0].legend(frameon=False, fontsize=8); ax[0].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(xpos, labels); ax[1].set_ylabel("information coefficient of $\\hat\\mu$")
    ax[1].axhline(0.0, color="k", lw=0.8, ls="--")
    savefig(fig, "e6_intervention.pdf")
    out = {k: {kk: float(np.mean(v)) for kk, v in d.items()} for k, d in res.items()}
    for k in out:
        out[k]["to_over_gross"] = out[k]["to"] / out[k]["gross"]
    print(json.dumps(out, indent=1))
    lat = sorted(((np.mean(v), c) for c, v in lattice.items()))
    print("LATTICE (mean in-sample eps by candidate set):")
    for v, c in lat: print(f"  {c:12s} {v:.3f}")
    lines = [r"\begin{tabular}{@{}lc@{}}", r"\toprule",
             r"candidate set $\Dset$ & mean $\hat\epsilon_t(\Dset)$\\ \midrule"]
    for v, c in lat[:8]:
        lines.append(r"$\{" + c.replace("+", ",").replace("Z1","Z^1").replace("Z2","Z^2") + r"\}$ & " + f"{v:.3f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    savetab("\n".join(lines), "e6_lattice.tex")
    return out

if __name__ == "__main__": run()
