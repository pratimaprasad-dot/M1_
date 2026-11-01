"""
NumPy/CPU back-end for v3_engine.py (no torch needed).
Loads v3_engine with a minimal torch stub and replaces its torch-based Law (eq. 7-11, 32) by the NumPy class below.
Everything else (filters, rate model, WMMSE, power step, Algorithm 1, baselines, evaluate) is v3_engine unchanged.
"""
import sys, types, importlib
import numpy as np
from scipy import stats, special
from scipy.stats import qmc

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401  (use the real one if installed)
    except ImportError:
        _t = types.ModuleType("torch")
        _t.device = lambda *a, **k: "cpu"
        _t.cuda = types.SimpleNamespace(is_available=lambda: False)
        _t.float64, _t.complex128 = np.float64, np.complex128
        sys.modules["torch"] = _t

E = importlib.import_module("v3_engine")


class Law:
    """Detection law of eq.(7)-(11): Gauss-Laguerre over the Gamma texture (32), scrambled Sobol over the CSI errors."""

    def __init__(self, kshape=0.7, n_tau=24, n_e=2 ** 12, seed=0, M=4, share=0.5):
        x, w = special.roots_genlaguerre(n_tau, kshape - 1.0)
        self.tau = x / kshape
        self.wt = w / special.gamma(kshape)
        sob = qmc.Sobol(d=2 * M, scramble=True, seed=seed).random(n_e)
        z = stats.norm.ppf(np.clip(sob, 1e-12, 1 - 1e-12)) / np.sqrt(2)
        self.z = z[:, :M] + 1j * z[:, M:]
        self.n_e, self.M = n_e, M
        self.sc_c, self.sn = share, 1 - share

    @staticmethod
    def Qf(x): return 0.5 * special.erfc(x / np.sqrt(2.0))

    def prep(self, U, sc, eps):
        UP = U.conj().T @ sc.psi
        mu = np.abs(np.diagonal(UP))
        I = (1 + eps) * ((np.abs(UP) ** 2).sum(1) - mu ** 2)
        UH = U.conj().T @ sc.hhat
        if eps > 0:
            sig_h = np.sqrt(eps * np.linalg.norm(sc.hhat, axis=0) ** 2 / sc.Nr)
            coup = UH[:, None, :] + sig_h[None, None, :] * self.z[None, :, :]
        else:
            coup = UH[:, None, :]
        return dict(mu=mu, I=I, a2=np.abs(coup) ** 2, eps=eps)

    def V0(self, pr, P):
        return pr["I"][:, None] + np.einsum("kem,km->ke", pr["a2"], P) + self.sn

    def _V(self, pr, P):
        return self.V0(pr, P)[:, :, None] + self.sc_c * self.tau[None, None, :]

    def threshold(self, V, alpha, iters=45, newton=True):
        """P_FA(t) = alpha per target. newton=True: safeguarded Newton on the Gaussian-equivalent deviate Q^-1(P_FA) (same root as the 45-step bisection to
        ~1e-12 relative, verify_threshold.py; about 6 steps per call; added in Version_4 research for speed only)."""
        sd0 = np.sqrt(V / 2); wt = self.wt[None, :]
        lo = np.zeros(V.shape[0]); hi = 40 * sd0.max(axis=(1, 2))
        if not newton:
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                pf = (self.Qf(mid[:, None, None] / sd0).mean(1) * wt).sum(1)
                big = pf > alpha
                lo = np.where(big, mid, lo); hi = np.where(big, hi, mid)
            return 0.5 * (lo + hi)
        la = np.log(alpha)
        isd = 1.0 / sd0
        t = getattr(self, "_t_warm", None)                       # warm start: previous root for the same targets
        if t is None or t.shape != lo.shape or getattr(self, "_a_warm", None) != alpha or not np.all(np.isfinite(t)):
            t = np.sqrt((V.mean(axis=(1, 2))) / 2) * stats.norm.isf(alpha)
        t = np.clip(t, lo, hi)
        c = 1.0 / np.sqrt(2 * np.pi)
        gs = stats.norm.isf(alpha)
        for _ in range(60):
            z = t[:, None, None] * isd
            pf = (self.Qf(z).mean(1) * wt).sum(1)
            dens = ((np.exp(-0.5 * z * z) * isd).mean(1) * wt).sum(1) * c
            big = pf > alpha
            lo = np.where(big, t, lo); hi = np.where(big, hi, t)
            # Newton on the Gaussian-equivalent deviate g(t) = Q^-1(P_FA(t)) (exactly linear for Gaussian disturbance)
            g = stats.norm.isf(np.clip(pf, 1e-300, 1 - 1e-16))
            tn = t - (g - gs) * c * np.exp(-0.5 * g * g) / np.maximum(dens, 1e-300)
            done = np.abs(tn - t) <= 1e-13 * np.maximum(np.abs(t), 1e-300)    # converged step: accept as is
            bad = (~np.isfinite(tn) | (tn < lo) | (tn > hi)) & ~done
            tn = np.where(bad, 0.5 * (lo + hi), tn)
            t = tn
            if np.all(done): break
        self._t_warm, self._a_warm = t.copy(), alpha
        return t

    def pd_given(self, pr, V, t):
        sd1 = np.sqrt((V + pr["eps"] * pr["mu"][:, None, None] ** 2) / 2)
        return (self.Qf((t[:, None, None] - pr["mu"][:, None, None]) / sd1).mean(1) * self.wt[None, :]).sum(1)

    def solve(self, pr, P, alpha=E.ALPHA, iters=45):
        V = self._V(pr, P)
        t = self.threshold(V, alpha, iters)
        return t, self.pd_given(pr, V, t)

    def pd_of(self, U, sc, p, eps):
        pr = self.prep(U, sc, eps)
        return self.solve(pr, np.tile(p, (sc.K, 1)))[1]

    def pd_cond(self, U, sc, p, eps):
        pr = self.prep(U, sc, eps); P = np.tile(p, (sc.K, 1))
        V = self._V(pr, P); t = self.threshold(V, E.ALPHA)
        sd1 = np.sqrt((V + eps * pr["mu"][:, None, None] ** 2) / 2)
        return (self.Qf((t[:, None, None] - pr["mu"][:, None, None]) / sd1) * self.wt[None, None, :]).sum(-1)

    def roc(self, U, sc, p, eps, alphas):
        """PD_k at each false-alarm level (threshold re-calibrated to the true null for every alpha): len(alphas) x K"""
        pr = self.prep(U, sc, eps); V = self._V(pr, np.tile(p, (sc.K, 1)))
        return np.array([self.pd_given(pr, V, self.threshold(V, a)) for a in alphas])

    def max_scale(self, pr, d, target, cap, steps=20):
        K = pr["mu"].shape[0]
        f = lambda lam: self.solve(pr, lam[:, None] * np.asarray(d)[None, :])[1]
        z0 = np.zeros(K); capv = np.full(K, float(cap))
        pd0, pdc = f(z0), f(capv)
        lo, hi = z0.copy(), capv.copy()
        for _ in range(steps):
            mid = 0.5 * (lo + hi); ok = f(mid) >= target
            lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
        lam = np.where(pdc >= target, capv, lo)
        return np.where(pd0 < target, z0, lam)


E.Law = Law
