"""Projected Markowitz: exact KKT solve, structural objects, two-stage solver."""
import numpy as np


def kkt_solve(Q, mu, A, b, gamma=1.0):
    """Solve max gamma*w'mu - 0.5 w'Qw  s.t. A w = b  via the KKT system.
    Returns (w, nu)."""
    n = Q.shape[0]; p = A.shape[0]
    K = np.block([[Q, A.T], [A, np.zeros((p, p))]])
    rhs = np.concatenate([gamma * mu, b])
    sol = np.linalg.solve(K, rhs)
    return sol[:n], sol[n:]


def M_and_w0(Q, A, b):
    """M = Q^{-1} - Q^{-1}A'(AQ^{-1}A')^{-1}AQ^{-1};  w0 = Q^{-1}A'(AQ^{-1}A')^{-1}b."""
    Qi = np.linalg.inv(Q)
    S = A @ Qi @ A.T
    Si = np.linalg.inv(S)
    M = Qi - Qi @ A.T @ Si @ A @ Qi
    w0 = Qi @ A.T @ Si @ b
    return M, w0


def two_stage_solve(Sc, B, Lam, mu, A, b, gamma=1.0):
    """O(n k^2) solve exploiting Q = diag(Sc) + B Lam B' (Woodbury).
    Never forms Q or Q^{-1}."""
    d = 1.0 / Sc                                   # n
    Bd = B * d[:, None]                            # D^{-1} B
    core = np.linalg.inv(np.linalg.inv(Lam) + B.T @ Bd)   # k x k

    def Qinv(X):
        X = np.atleast_2d(X.T).T if X.ndim == 1 else X
        return d[:, None] * X - Bd @ (core @ (Bd.T @ X))

    QiA = Qinv(A.T)                                # n x p
    S = A @ QiA                                    # p x p
    Qim = Qinv(mu[:, None])[:, 0]
    nu = np.linalg.solve(S, gamma * (A @ Qim) - b)
    w = gamma * Qim - QiA @ nu
    return w


def hj_gap_check(Q, mu, C):
    """Machine-precision check of the HJ gap identity:
    mu'Q^{-1}mu - mu'M_C mu == eta*' (C Q^{-1} C') eta*  with only constraint Cw=0."""
    Qi = np.linalg.inv(Q)
    S = C @ Qi @ C.T
    M_C = Qi - Qi @ C.T @ np.linalg.solve(S, C @ Qi)
    eta = np.linalg.solve(S, C @ Qi @ mu)
    lhs = mu @ Qi @ mu - mu @ M_C @ mu
    rhs = eta @ S @ eta
    return lhs, rhs, abs(lhs - rhs)
