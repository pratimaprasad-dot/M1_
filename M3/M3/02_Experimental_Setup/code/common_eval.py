"""
Common evaluation used for EVERY method in Layers B and C (instruction items 2 and 5):
same channels, same detector, same false-alarm calibration, same rate metric.

Truth model (manuscript eq. (1), literal form, as in Version_3_audit/code/27_e1_literal_validation.py):
  per CPI : texture tau ~ Gamma(nu, 1/nu) (mean 1); CSI errors e_m ~ CN(0, sigma_h,m^2 I), sig_m ~ CN(0, sigma_g,m^2 I);
            target coefficients 1 + zeta_i, zeta_i ~ CN(0, eps) for every target i;
  given the CPI, the matched-filter statistic T_k = Re{e^{-j angle mu_k} u_k^H y s^*} is Gaussian with
            mean  m0 = Re{rot sum_{i!=k} (1+zeta_i) u_k^H psi_i}              (H0)
                  m1 = m0 + Re{rot mu_k (1+zeta_k)}                           (H1)
            var   (sum_m p_m |u_k^H (hhat_m + e_m)|^2 + s_n + s_c tau) / 2.
Detector (identical for all methods): T_k > t_k with t_k calibrated so that the TRUE P_FA = alpha
  (conditional Monte Carlo, common random numbers per channel). This is a clairvoyant (oracle) calibration of the
  same statistic for every method; the P_FA that each method's OWN design threshold would give is reported separately.
Rate: ergodic SAR E[0.5 sum log2(1 + gamma_m)] over Gaussian CSI errors (Monte Carlo), relay matrix as designed
  (scaled down if its expected power exceeds P^max, as v3_engine.evaluate does). The manuscript's model rate
  (v3_engine.evaluate) is reported as a secondary column.
"""
import numpy as np
from scipy import special
import v3_np
E = v3_np.E

ALPHA, BETA = 1e-3, 0.9


def Qf(x): return 0.5 * special.erfc(x / np.sqrt(2.0))


class Evaluator:
    def __init__(self, nu=0.7, share=0.5, n_det=20000, n_rate=2000, seed=12345, M=4):
        self.nu, self.sc_c, self.sn = nu, share, 1 - share
        self.n_det, self.n_rate, self.seed = n_det, n_rate, seed
        self.law = v3_np.Law(nu if np.isfinite(nu) else 0.7, n_tau=48, n_e=2 ** 14, seed=7, share=share, M=M)
        if not np.isfinite(nu): self.law.tau, self.law.wt = np.array([1.0]), np.array([1.0])   # manuscript's analytic law (Gaussianised leakage)

    # ------------------------------------------------------------------ detection
    def detection(self, sc, p, U, eps, alpha=ALPHA):
        rng = np.random.default_rng(self.seed + 7919 * int(getattr(sc, "seed", 0)))
        U = U / np.linalg.norm(U, axis=0, keepdims=True)
        N, K, M = self.n_det, sc.K, sc.M
        cn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
        tau = rng.gamma(self.nu, 1.0 / self.nu, N) if np.isfinite(self.nu) else np.ones(N)
        ze = np.sqrt(eps) * cn(N, K)
        zh = cn(N, M)                                                   # unit draws for u^H e_m (common across k: fine per target)
        sh = np.sqrt(E.sigma_h2(sc, eps))
        UP = U.conj().T @ sc.psi; UH = U.conj().T @ sc.hhat
        out = dict(pd=np.zeros(K), t=np.zeros(K), pfa_design=np.zeros(K), pd_design=np.zeros(K))
        # design-law thresholds (Gaussianised leakage, the manuscript's eq. (11)) for the P_FA check
        pr = self.law.prep(U, sc, eps); V = self.law._V(pr, np.tile(p, (K, 1)))
        t_des = self.law.threshold(V, alpha)
        out["pd_gauss"] = self.law.pd_given(pr, V, t_des)
        for k in range(K):
            mu = UP[k, k]; rot = np.exp(-1j * np.angle(mu))
            oth = [i for i in range(K) if i != k]
            m0 = np.real(rot * ((1 + ze[:, oth]) * UP[k, oth][None, :]).sum(1))
            m1 = m0 + np.real(rot * mu * (1 + ze[:, k]))
            X = (np.abs(UH[k][None, :] + sh[None, :] * zh) ** 2) @ p
            s = np.sqrt((X + self.sn + self.sc_c * tau) / 2)
            pf = lambda t: Qf((t - m0) / s).mean()
            lo, hi = m0.min() - 10 * s.max(), m0.max() + 60 * s.max()
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if pf(mid) > alpha: lo = mid
                else: hi = mid
            t = 0.5 * (lo + hi)
            out["t"][k] = t; out["pd"][k] = Qf((t - m1) / s).mean()
            out["pfa_design"][k] = pf(t_des[k]); out["pd_design"][k] = Qf((t_des[k] - m1) / s).mean()
        return out

    # ------------------------------------------------------------------ rate
    def rate(self, sc, p, W, eps):
        Pt = np.real(np.trace(W @ E.Ry(sc, p, eps) @ W.conj().T))
        W = W * (np.sqrt(sc.Pmax / Pt) if Pt > sc.Pmax else 1.0)
        rng = np.random.default_rng(self.seed + 104729 * int(getattr(sc, "seed", 0)))
        Rmod = E.rate_from(p, *E.rate_terms(W, sc, eps)[:3], 1.0, sc)[0]
        if eps <= 0:
            return dict(R_mc=float(Rmod), R_model=float(Rmod), Pt=Pt / sc.Pmax)
        sh = np.sqrt(E.sigma_h2(sc, eps)); sg = np.sqrt(E.sigma_g2(sc, eps))
        n = self.n_rate
        cn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
        H = sc.hhat[None] + sh[None, None, :] * cn(n, sc.Nr, sc.M)
        G = sc.ghat[None] + sg[None, None, :] * cn(n, sc.Nt, sc.M)
        # target coefficient perturbation of the forwarded echo (1 + zeta_k), as in the truth model
        ze = 1 + np.sqrt(eps) * cn(n, sc.K)
        GW = np.einsum("ntm,tr->nmr", G, W)                              # g_m^T W
        S = np.abs(np.einsum("nmr,nrj->nmj", GW, H)) ** 2                # |g_m^T W h_j|^2
        echo = np.abs(np.einsum("nmr,rk,nk->nmk", GW, sc.psi, ze)) ** 2
        J = echo.sum(2) + (np.abs(GW) ** 2).sum(2) + S @ p - np.einsum("nmm->nm", S) * p + sc.s2
        gam = p[None] * np.einsum("nmm->nm", S) / J
        R = 0.5 * np.log2(1 + gam).sum(1)
        return dict(R_mc=float(R.mean()), R_mc_se=float(R.std() / np.sqrt(n)), R_model=float(Rmod), Pt=Pt / sc.Pmax)

    # ------------------------------------------------------------------ all
    def __call__(self, sc, p, W, U, eps, beta=BETA):
        d = self.detection(sc, p, U, eps)
        r = self.rate(sc, p, W, eps)
        pe = self.law.pd_cond(U / np.linalg.norm(U, axis=0, keepdims=True), sc, p, eps)
        out = dict(R=r["R_mc"], R_model=r["R_model"], Pt=r["Pt"],
                   pd=[float(x) for x in d["pd"]], minpd=float(d["pd"].min()), prot=bool(d["pd"].min() >= beta),
                   minpd_gauss=float(d["pd_gauss"].min()), prot_gauss=bool(d["pd_gauss"].min() >= beta),
                   pfa_design=[float(x) for x in d["pfa_design"]], minpd_design=float(d["pd_design"].min()),
                   prot_real=float(np.mean(pe.min(0) >= beta)), Psum_mW=float(1000 * np.sum(p)), p=[float(x) for x in p])
        return out
