"""Covariance / conditional-mean estimators compared in the experiments."""
import numpy as np


def fit_causal(R, Z):
    """Closed-window structured estimator (the paper's estimation map).

    1. Fit driver VAR(1): Z_{t+1} = Phi Z_t + eta  -> innovations eta_hat, Lam_hat.
    2. Regress r_{t+1} on [1, Z_t, eta_hat_{t+1}]: (a,G) past block, B innovation block.
    3. Sigma_c = diagonal residual variances (closed-window residuals).
    Returns dict with (a, G, B, Lam, Sc, Q)."""
    T = R.shape[0]
    Zp, Zn = Z[:T], Z[1:T + 1]
    Xz = np.column_stack([np.ones(T), Zp])              # intercept: affine gauge absorbed
    Cz = np.linalg.lstsq(Xz, Zn, rcond=None)[0]
    Phi = Cz[1:].T
    eta = Zn - Xz @ Cz
    Lam = np.atleast_2d(np.cov(eta.T, bias=False))
    X = np.column_stack([np.ones(T), Zp, eta])
    coef = np.linalg.lstsq(X, R, rcond=None)[0]          # (1+2m) x n
    m = Z.shape[1]
    a, G, B = coef[0], coef[1:1 + m].T, coef[1 + m:].T
    resid = R - X @ coef
    Sc = resid.var(axis=0, ddof=X.shape[1])
    Q = B @ Lam @ B.T + np.diag(Sc)
    return {"a": a, "G": G, "B": B, "Lam": Lam, "Sc": Sc, "Q": Q,
            "Phi": Phi, "eta": eta, "resid": resid}


def fit_past_only(R, Z):
    """Past-window regression only (a purely predictive conditioning)."""
    T = R.shape[0]
    X = np.column_stack([np.ones(T), Z[:T]])
    coef = np.linalg.lstsq(X, R, rcond=None)[0]
    resid = R - X @ coef
    return {"resid": resid}


def cov_sample(R):
    return np.cov(R.T, bias=False)


def cov_ledoit_wolf(R):
    """Ledoit-Wolf (2004) linear shrinkage to scaled identity."""
    T, n = R.shape
    X = R - R.mean(0)
    S = X.T @ X / T
    mu = np.trace(S) / n
    F = mu * np.eye(n)
    d2 = np.linalg.norm(S - F, 'fro')**2 / n
    b2bar = sum(np.linalg.norm(np.outer(x, x) - S, 'fro')**2 for x in X) / (T**2 * n)
    b2 = min(b2bar, d2)
    rho = b2 / d2 if d2 > 0 else 0.0
    return rho * F + (1 - rho) * S


def cov_pca(R, k):
    """Static k-factor (PCA) covariance: L L' + diag residual."""
    X = R - R.mean(0)
    S = X.T @ X / (R.shape[0] - 1)
    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(vals)[::-1][:k]
    L = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0.0))
    resid = np.maximum(np.diag(S) - (L**2).sum(1), 1e-10)
    return L @ L.T + np.diag(resid)
