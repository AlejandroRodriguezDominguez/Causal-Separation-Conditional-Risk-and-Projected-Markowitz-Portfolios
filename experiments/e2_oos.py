"""E2 -- Estimation risk: projected/structured Markowitz vs plug-in alternatives.

Monte Carlo over (n, T). GMV portfolios from seven covariance estimators
(sample, Ledoit-Wolf, PCA with k = m-2 / m / m+2, causal structured, oracle):
OOS annualized volatility and calibration. Mean-variance Sharpe: static hold
(weights fixed at t=T for the whole OOS year, isolating one-shot deployment)
AND daily rebalanced (fixed fitted parameters, current state), with mean daily
L1 turnover of the rebalanced causal strategy."""
from common import *
from causal_markowitz import NormalFormDGP, estimators, optimizer

def gmv(Qh):
    n = Qh.shape[0]
    w, _ = optimizer.kkt_solve(Qh, np.zeros(n), np.ones((1, n)), np.array([1.0]), 0.0)
    return w

def run(seed=21):
    ns = [50, 100, 200]; Ts = [126, 252, 504]; reps = 250; T_out = 252; m = 5
    gam = 0.05
    cov_methods = ["sample", "lw", "pca_lo", "pca", "pca_hi", "causal", "oracle"]
    vol = {(n_, T_): {k: [] for k in cov_methods} for n_ in ns for T_ in Ts}
    cal = {(n_, T_): {k: [] for k in cov_methods} for n_ in ns for T_ in Ts}
    stags = ["mv_sample_st", "mv_causal_st", "mv_oracle_st", "mv_causal_rb", "mv_oracle_rb", "to_causal_rb"]
    shp = {(n_, T_): {k: [] for k in stags} for n_ in ns for T_ in Ts}
    rng = np.random.default_rng(seed)
    for n_ in ns:
        for T_ in Ts:
            for r in range(reps):
                dgp = NormalFormDGP(n=n_, m=m, seed=rng.integers(1e9))
                sim = dgp.simulate(T_ + T_out, rng=np.random.default_rng(rng.integers(1e9)))
                Rtr, Rte = sim["R"][:T_], sim["R"][T_:]
                Ztr = sim["Z"][:T_ + 1]
                cfit = estimators.fit_causal(Rtr, Ztr)
                fits = {"sample": estimators.cov_sample(Rtr),
                        "lw": estimators.cov_ledoit_wolf(Rtr),
                        "pca_lo": estimators.cov_pca(Rtr, m - 2),
                        "pca": estimators.cov_pca(Rtr, m),
                        "pca_hi": estimators.cov_pca(Rtr, m + 2),
                        "causal": cfit["Q"], "oracle": dgp.Q}
                for k, Qh in fits.items():
                    w = gmv(Qh)
                    x = Rte @ w
                    vol[(n_, T_)][k].append(np.std(x, ddof=1) * np.sqrt(252))
                    cal[(n_, T_)][k].append(np.var(x, ddof=1) / (w @ Qh @ w))
                A = np.ones((1, n_)); b = np.array([1.0])
                # static hold
                zt = sim["Z"][T_]
                for tag, (Qh, muh) in {
                    "mv_sample_st": (fits["sample"], Rtr.mean(0)),
                    "mv_causal_st": (fits["causal"], cfit["a"] + cfit["G"] @ zt),
                    "mv_oracle_st": (dgp.Q, dgp.mu_cond(zt))}.items():
                    w, _ = optimizer.kkt_solve(Qh, muh, A, b, gam)
                    x = Rte @ w
                    shp[(n_, T_)][tag].append(np.mean(x)/np.std(x, ddof=1)*np.sqrt(252))
                # daily rebalanced (fixed fitted params)
                for tag, (Qh, aa, GG) in {
                    "mv_causal_rb": (cfit["Q"], cfit["a"], cfit["G"]),
                    "mv_oracle_rb": (dgp.Q, dgp.a, dgp.G)}.items():
                    M, w0 = optimizer.M_and_w0(Qh, A, b)
                    xs, tos, wprev = [], [], None
                    for t in range(T_, T_ + T_out):
                        w = w0 + gam * (M @ (aa + GG @ sim["Z"][t]))
                        xs.append(sim["R"][t] @ w)
                        if wprev is not None: tos.append(np.abs(w - wprev).sum())
                        wprev = w
                    xs = np.array(xs)
                    shp[(n_, T_)][tag].append(np.mean(xs)/np.std(xs, ddof=1)*np.sqrt(252))
                    if tag == "mv_causal_rb":
                        shp[(n_, T_)]["to_causal_rb"].append(np.mean(tos) / np.mean([np.abs(w0 + gam*(M@(aa+GG@sim["Z"][t]))).sum() for t in range(T_, T_+T_out, 21)]))
    # ---- covariance table ----
    hdr = cov_methods
    names = {"sample": "Sample", "lw": "LW", "pca_lo": "PCA-$(m{-}2)$", "pca": "PCA-$m$",
             "pca_hi": "PCA-$(m{+}2)$", "causal": "Causal", "oracle": "Oracle"}
    lines = [r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
             r"$n$ & $T$ & " + " & ".join(names[k] for k in hdr) + r"\\ \midrule"]
    for n_ in ns:
        for T_ in Ts:
            row = [f"{100*np.mean(vol[(n_,T_)][k]):.2f}" for k in hdr]
            lines.append(f"{n_} & {T_} & " + " & ".join(row) + r"\\")
    lines += [r"\midrule", r"\multicolumn{9}{@{}l}{\emph{Calibration: realized/predicted variance (1 = perfect)}}\\"]
    for n_ in ns:
        row = [f"{np.mean(cal[(n_,252)][k]):.2f}" for k in hdr]
        lines.append(f"{n_} & 252 & " + " & ".join(row) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    savetab("\n".join(lines), "e2_table.tex")
    # ---- figure (5 original series) ----
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for k, c in zip(["sample", "lw", "pca", "causal", "oracle"], ["C3", "C1", "C2", "C0", "k"]):
        ax[0].plot(ns, [100*np.mean(vol[(n_, 252)][k]) for n_ in ns], "o-", color=c, label=names[k])
        ax[1].plot(ns, [np.mean(cal[(n_, 252)][k]) for n_ in ns], "o-", color=c)
    ax[0].set_xlabel(r"$n$ (assets), $T=252$"); ax[0].set_ylabel("OOS ann. volatility, \\% (GMV)")
    ax[0].set_yscale("log"); ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel(r"$n$ (assets), $T=252$"); ax[1].set_ylabel("realized / predicted variance")
    ax[0].legend(frameon=False, fontsize=8)
    savefig(fig, "e2_oos.pdf")
    # ---- sharpe table: fraction of oracle (rebalanced) Sharpe captured ----
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"$n$ & $T$ & Sample (static) & Causal (static) & Causal (rebal.) & TO/gross (\%/day)\\ \midrule"]
    for n_ in ns:
        for T_ in Ts:
            o_st = np.mean(shp[(n_,T_)]["mv_oracle_st"])
            o_rb = np.mean(shp[(n_,T_)]["mv_oracle_rb"])
            row = [f"{np.mean(shp[(n_,T_)]['mv_sample_st'])/o_st:.2f}",
                   f"{np.mean(shp[(n_,T_)]['mv_causal_st'])/o_st:.2f}",
                   f"{np.mean(shp[(n_,T_)]['mv_causal_rb'])/o_rb:.2f}",
                   f"{100*np.mean(shp[(n_,T_)]['to_causal_rb']):.1f}"]
            lines.append(f"{n_} & {T_} & " + " & ".join(row) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    savetab("\n".join(lines), "e2_sharpe.tex")
    se_vol = max(100*np.std(vol[(n_,T_)][k])/np.sqrt(reps) for n_ in ns for T_ in Ts
                 for k in hdr if not (n_==200 and T_==126 and k=="sample"))
    se_shp = max(np.std(shp[(n_,T_)][k])/np.sqrt(reps) for n_ in ns for T_ in Ts
                 for k in stags[:-1])
    print(f"max MC SE: vol {se_vol:.3f}pp (excl. singular cell), sharpe {se_shp:.3f}")
    print("headline n=200: T126 sample vol", f"{100*np.mean(vol[(200,126)]['sample']):.1f}",
          "| cal252 sample", f"{np.mean(cal[(200,252)]['sample']):.2f}",
          "causal", f"{np.mean(cal[(200,252)]['causal']):.2f}",
          "pca_lo", f"{np.mean(cal[(200,252)]['pca_lo']):.2f}")
    print("pca_lo vol n=200 T=252:", f"{100*np.mean(vol[(200,252)]['pca_lo']):.2f}",
          "vs causal", f"{100*np.mean(vol[(200,252)]['causal']):.2f}")
    o_st = np.mean(shp[(100,252)]['mv_oracle_st']); o_rb = np.mean(shp[(100,252)]['mv_oracle_rb'])
    print("n=100 T=252 frac: st_sample", f"{np.mean(shp[(100,252)]['mv_sample_st'])/o_st:.2f}",
          "st_causal", f"{np.mean(shp[(100,252)]['mv_causal_st'])/o_st:.2f}",
          "rb_causal", f"{np.mean(shp[(100,252)]['mv_causal_rb'])/o_rb:.2f}",
          "| levels: oracle_st", f"{o_st:.2f}", "oracle_rb", f"{o_rb:.2f}",
          "| to/gross", f"{100*np.mean(shp[(100,252)]['to_causal_rb']):.1f}")
    print("E2 done")

if __name__ == "__main__": run()
