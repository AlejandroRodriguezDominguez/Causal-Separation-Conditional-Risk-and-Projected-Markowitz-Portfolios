"""Data-generating process: discrete-time normal form of causal separation.

    Z_{t+1} = Phi Z_t + eta_{t+1},          eta ~ N(0, Lam)      (drivers, VAR(1))
    r_{t+1} = a + G Z_t + B eta_{t+1} + varsigma * zeta_{t+1}    (returns)

Given the *horizon-closed* driver information (Z_t and eta_{t+1}), the
components of r_{t+1} are mutually independent with diagonal residual
covariance diag(varsigma^2).  Given the *past-only* information (Z_t),
Cov(r_{t+1} | Z_t) = B Lam B' + diag(varsigma^2) = Q.
This is exactly the two-windows structure of the paper (Rem. 2.3).
"""
import numpy as np


class NormalFormDGP:
    def __init__(self, n=50, m=3, phi=0.95, seed=0,
                 b_scale=0.04, sig_lo=0.008, sig_hi=0.02,
                 prem_scale=0.12, lam_diag=None):
        rng = np.random.default_rng(seed)
        self.n, self.m, self.phi = n, m, phi
        self.B = rng.normal(0.0, b_scale, size=(n, m))
        self.a = np.zeros(n)
        self.theta = prem_scale * rng.uniform(0.5, 1.5, size=m) * np.sign(rng.normal(size=m))
        self.G = self.B @ np.diag(self.theta)
        self.varsigma = rng.uniform(sig_lo, sig_hi, size=n)
        lam = np.ones(m) if lam_diag is None else np.asarray(lam_diag, float)
        self.Lam = np.diag(lam * (1 - phi**2))          # stationary Var(Z)=diag(lam)
        self.Q = self.B @ self.Lam @ self.B.T + np.diag(self.varsigma**2)
        self.rng = rng

    def mu_cond(self, z):
        """E[r_{t+1} | Z_t = z]  (past-window conditional mean)."""
        return self.a + self.G @ z

    def simulate(self, T, z0=None, rng=None, resid_corr=0.0):
        """Simulate a path of length T. Optionally inject residual cross-
        correlation of maximal absolute size resid_corr (violating exact
        separation) through an equicorrelated Gaussian factor on zeta."""
        rng = rng or self.rng
        n, m = self.n, self.m
        Z = np.zeros((T + 1, m))
        Z[0] = z0 if z0 is not None else rng.multivariate_normal(
            np.zeros(m), np.diag(np.diag(self.Lam)) / (1 - self.phi**2))
        eta = rng.multivariate_normal(np.zeros(m), self.Lam, size=T)
        if resid_corr > 0:
            C = np.full((n, n), resid_corr); np.fill_diagonal(C, 1.0)
            L = np.linalg.cholesky(C)
            zeta = rng.normal(size=(T, n)) @ L.T
        else:
            zeta = rng.normal(size=(T, n))
        R = np.zeros((T, n))
        for t in range(T):
            Z[t + 1] = self.phi * Z[t] + eta[t]
            R[t] = self.a + self.G @ Z[t] + self.B @ eta[t] + self.varsigma * zeta[t]
        return {"Z": Z, "eta": eta, "R": R, "zeta": zeta}
