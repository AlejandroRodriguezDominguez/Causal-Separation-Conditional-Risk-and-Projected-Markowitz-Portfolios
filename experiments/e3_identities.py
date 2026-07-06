"""E3 -- Machine-precision validation of the structural identities:
(a) HJ gap identity; (b) two-stage (Woodbury) vs dense KKT; (c) properties
(P1)-(P4) of M; (d) gauge invariance under invertible reparametrization of Z;
(e) conditioning floor ||M||_2 <= 1/varsigma_min^2."""
from common import *
from causal_markowitz import NormalFormDGP, estimators, optimizer

def run(seed=31):
    rng = np.random.default_rng(seed)
    rows = []
    dgp = NormalFormDGP(n=80, m=4, seed=5)
    Q, mu = dgp.Q, dgp.mu_cond(rng.normal(size=4))
    # (a) HJ gap
    C = rng.normal(size=(2, 80))
    lhs, rhs, err = optimizer.hj_gap_check(Q, mu, C)
    rows.append(("HJ gap identity (Prop.~4.6)", f"{err/max(abs(lhs),1e-300):.1e}"))
    # (b) two-stage vs dense
    A = np.vstack([np.ones(80), C]); b = np.array([1.0, 0, 0])
    w_d, _ = optimizer.kkt_solve(Q, mu, A, b, 3.0)
    w_2 = optimizer.two_stage_solve(dgp.varsigma**2, dgp.B, dgp.Lam, mu, A, b, 3.0)
    rows.append(("two-stage vs dense KKT (Prop.~4.8)", f"{np.abs(w_d-w_2).max():.1e}"))
    # (c) M properties
    M, w0 = optimizer.M_and_w0(Q, A, b)
    p1 = np.abs(M - M.T).max()
    p2 = np.abs(M @ Q @ M - M).max()
    p3 = np.abs(A @ M).max()
    p4 = max(np.abs(A @ w0 - b).max(), np.abs(w0 @ Q @ M).max())
    rows.append((r"(P1)--(P4) of $M$ (Thm.~4.1)", f"{max(p1,p2,p3,p4):.1e}"))
    # (d) gauge invariance: Z' = T Z + c, refit, same w*
    T_, n_, m_ = 3000, 60, 3
    dgp2 = NormalFormDGP(n=n_, m=m_, seed=9)
    sim = dgp2.simulate(T_)
    Tmat = rng.normal(size=(m_, m_)) + 2*np.eye(m_); c = rng.normal(size=m_)
    Z2 = sim["Z"] @ Tmat.T + c
    A2 = np.ones((1, n_)); b2 = np.array([1.0])
    f1 = estimators.fit_causal(sim["R"], sim["Z"]); f2 = estimators.fit_causal(sim["R"], Z2)
    z1, z2 = sim["Z"][T_], Z2[T_]
    w1, _ = optimizer.kkt_solve(f1["Q"], f1["a"]+f1["G"]@z1, A2, b2, 3.0)
    w2, _ = optimizer.kkt_solve(f2["Q"], f2["a"]+f2["G"]@z2, A2, b2, 3.0)
    rows.append(("gauge invariance $Z\\mapsto TZ+c$ (Prop.~4.7)", f"{np.abs(w1-w2).max():.1e}"))
    # (e) conditioning floor
    Mn = np.linalg.norm(M, 2); floor = 1/dgp.varsigma.min()**2
    rows.append((r"$\|M\|_2\,\underline\varsigma^{2}\le1$ (Prop.~4.5)", f"{Mn*dgp.varsigma.min()**2:.4f}"))
    lines = [r"\begin{tabular}{@{}lc@{}}", r"\toprule", r"Identity & residual / value\\ \midrule"]
    for a_, v in rows: lines.append(f"{a_} & {v}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    savetab("\n".join(lines), "e3_identities.tex")
    print(rows)

if __name__ == "__main__": run()
