"""
Version_3 engine: independent implementation of the maths and Algorithm 1 of Version_3_Updated.pdf
("Detection-Certified Power Allocation and Relay Beamforming for Cognitive ISAC ...").
Modules used by v3_run.py (simulation) and v3_compare.py (comparison with the paper's printed results).
Quantities the paper does not specify are ASSUMED (radar beam, reflection phases, channel law, quadrature sizes, solver settings).
"""
import time
import numpy as np
import torch
from scipy import stats, special
from scipy.stats import qmc
from scipy.optimize import minimize

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT, CT = torch.float64, torch.complex128
SIG_R2_W = 1e-11
ALPHA, BETA, MARGIN = 1e-3, 0.9, 0.003
Q95 = lambda Nr: stats.gamma.ppf(0.95, Nr)


def dbm2w(x): return 10 ** (x / 10) / 1000


def eta_beta(beta=BETA, alpha=ALPHA):
    return 0.5 * (stats.norm.isf(alpha) - stats.norm.isf(beta)) ** 2


# ------------------------------------------------------------------ scenario (ASSUMED, same as 03_*.py)
def steer(N, th):
    n = np.arange(N)
    return np.exp(-1j * np.pi * np.sin(th) * ((N - 1) / 2 - n))


class Scn:
    def __init__(self, seed, Nt=8, Nr=8, M=4, K=4, Pc_dbm=20, Pmax_dbm=20, Ps_dbm=20, beta_pow=1e-9,
                 theta=(-np.pi / 4, 0.0, np.pi / 4, np.pi / 2), d_m=100.0, iota=4.0):
        rng = np.random.default_rng(seed)
        self.Nt, self.Nr, self.M, self.K = Nt, Nr, M, K
        th = np.array(theta)[:K]
        self.Pc = dbm2w(Pc_dbm); self.Pmax = dbm2w(Pmax_dbm) / SIG_R2_W; self.Ps = dbm2w(Ps_dbm)
        w = sum(np.conj(steer(Nt, t)) for t in th)
        v = np.sqrt(self.Ps) * w / np.linalg.norm(w)
        sc = 1 / np.sqrt(SIG_R2_W)
        psi = []
        for t in th:
            b = np.sqrt(beta_pow) * np.exp(1j * rng.uniform(0, 2 * np.pi))
            psi.append(b * steer(Nr, t) * (steer(Nt, t) @ v))
        self.psi = np.array(psi).T * sc                                   # Nr x K
        cn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
        self.hhat = cn(Nr, M) * np.sqrt(d_m ** (-iota)) * sc              # Nr x M (carries 1/sqrt(sigma_r^2))
        self.ghat = cn(Nt, M) * np.sqrt(d_m ** (-iota))                   # Nt x M (dimensionless gain)
        self.sig_r2 = 1.0; self.s2 = 1.0


# ------------------------------------------------------------------ detection law (eq. 7-11), batched over targets
class Law:
    def __init__(self, kshape=0.7, n_tau=24, n_e=2 ** 12, seed=0, M=4, share=0.5):
        x, w = special.roots_genlaguerre(n_tau, kshape - 1.0)
        self.tau = torch.tensor(x / kshape, dtype=DT, device=dev)
        self.wt = torch.tensor(w / special.gamma(kshape), dtype=DT, device=dev)
        sob = qmc.Sobol(d=2 * M, scramble=True, seed=seed).random(n_e)
        z = stats.norm.ppf(np.clip(sob, 1e-12, 1 - 1e-12)) / np.sqrt(2)
        self.z = torch.tensor(z[:, :M] + 1j * z[:, M:], dtype=CT, device=dev)
        self.n_e = n_e; self.M = M
        self.sc_c = share; self.sn = 1 - share            # sigma_n^2 = sn*sigma_r^2 , sigma_c^2 = share*sigma_r^2

    @staticmethod
    def Qf(x): return 0.5 * torch.erfc(x / np.sqrt(2.0))

    def prep(self, U, sc, eps):
        U_ = torch.tensor(U, dtype=CT, device=dev); P_ = torch.tensor(sc.psi, dtype=CT, device=dev); H_ = torch.tensor(sc.hhat, dtype=CT, device=dev)
        UP = U_.conj().T @ P_
        mu = torch.diagonal(UP).abs()
        I = (1 + eps) * ((UP.abs() ** 2).sum(1) - mu ** 2)
        UH = U_.conj().T @ H_
        if eps > 0:
            sig_h = torch.tensor(np.sqrt(eps * np.linalg.norm(sc.hhat, axis=0) ** 2 / sc.Nr), dtype=DT, device=dev)
            coup = UH[:, None, :] + sig_h[None, None, :] * self.z[None, :, :]
        else:
            coup = UH[:, None, :]
        return dict(mu=mu, I=I, a2=coup.abs() ** 2, eps=eps)

    def V0(self, pr, P):                                                   # P: K x M tensor
        return pr["I"][:, None] + torch.einsum("kem,km->ke", pr["a2"], P) + self.sn

    def solve(self, pr, P, alpha=ALPHA, iters=50):
        V0 = self.V0(pr, P)
        V = V0[:, :, None] + self.sc_c * self.tau[None, None, :]
        sd0 = torch.sqrt(V / 2); wt = self.wt[None, :]
        Kn = V.shape[0]
        lo = torch.zeros(Kn, dtype=DT, device=dev); hi = 40 * sd0.amax(dim=(1, 2))
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            pf = (self.Qf(mid[:, None, None] / sd0).mean(1) * wt).sum(1)
            big = pf > alpha
            lo = torch.where(big, mid, lo); hi = torch.where(big, hi, mid)
        t = 0.5 * (lo + hi)
        sd1 = torch.sqrt((V + pr["eps"] * pr["mu"][:, None, None] ** 2) / 2)
        pd = (self.Qf((t[:, None, None] - pr["mu"][:, None, None]) / sd1).mean(1) * wt).sum(1)
        return t, pd

    def pd_of(self, U, sc, p, eps):
        pr = self.prep(U, sc, eps)
        P = torch.tensor(np.tile(p, (sc.K, 1)), dtype=DT, device=dev)
        return self.solve(pr, P)[1].cpu().numpy()

    def pd_cond(self, U, sc, p, eps):
        """conditional PD_k(e) for every error draw (threshold calibrated on the e-averaged null, eq.(11))"""
        pr = self.prep(U, sc, eps)
        P = torch.tensor(np.tile(p, (sc.K, 1)), dtype=DT, device=dev)
        t, _ = self.solve(pr, P)
        V = self.V0(pr, P)[:, :, None] + self.sc_c * self.tau[None, None, :]
        sd1 = torch.sqrt((V + eps * pr["mu"][:, None, None] ** 2) / 2)
        return (self.Qf((t[:, None, None] - pr["mu"][:, None, None]) / sd1) * self.wt[None, None, :]).sum(-1).cpu().numpy()

    def max_scale(self, pr, d, target, cap, steps=20):
        """per-target largest lam in [0,cap] with PD_k(lam*d) >= target (Corollary 1), batched bisection"""
        K = pr["mu"].shape[0]; dd = torch.tensor(d, dtype=DT, device=dev)
        f = lambda lam: self.solve(pr, lam[:, None] * dd[None, :])[1]
        z0 = torch.zeros(K, dtype=DT, device=dev); capv = torch.full((K,), float(cap), dtype=DT, device=dev)
        pd0 = f(z0); pdc = f(capv)
        lo = z0.clone(); hi = capv.clone()
        for _ in range(steps):
            mid = 0.5 * (lo + hi); ok = f(mid) >= target
            lo = torch.where(ok, mid, lo); hi = torch.where(ok, hi, mid)
        lam = torch.where(pdc >= target, capv, lo)
        lam = torch.where(pd0 < target, z0, lam)
        return lam.cpu().numpy()


# ------------------------------------------------------------------ filters
def sigma_h2(sc, eps): return eps * np.linalg.norm(sc.hhat, axis=0) ** 2 / sc.Nr
def sigma_g2(sc, eps): return eps * np.linalg.norm(sc.ghat, axis=0) ** 2 / sc.Nt


def mvdr(sc, p, eps_f, k):
    R = sc.sig_r2 * np.eye(sc.Nr, dtype=complex)
    for i in range(sc.K):
        if i != k: R += (1 + eps_f) * np.outer(sc.psi[:, i], sc.psi[:, i].conj())
    sh = sigma_h2(sc, eps_f)
    for m in range(sc.M): R += p[m] * (np.outer(sc.hhat[:, m], sc.hhat[:, m].conj()) + sh[m] * np.eye(sc.Nr))
    x = np.linalg.solve(R, sc.psi[:, k]); return x / np.linalg.norm(x)


def filters(sc, p, eps_f): return np.stack([mvdr(sc, p, eps_f, k) for k in range(sc.K)], 1)


# ------------------------------------------------------------------ rate model (eq. 15-21)
def Ry(sc, p, eps, echo=None):
    sh = sigma_h2(sc, eps)
    R = (1 + eps) * sc.psi @ sc.psi.conj().T + sc.sig_r2 * np.eye(sc.Nr)
    for m in range(sc.M): R = R + p[m] * (np.outer(sc.hhat[:, m], sc.hhat[:, m].conj()) + sh[m] * np.eye(sc.Nr))
    return R


def rate_terms(W, sc, eps):
    Nt, Nr, M, K = sc.Nt, sc.Nr, sc.M, sc.K
    sh, sg = sigma_h2(sc, eps), sigma_g2(sc, eps)
    gW = sc.ghat.T @ W                                                    # M x Nr
    GWH = gW @ sc.hhat                                                    # M x M : g_m^T W h_j
    fro2 = np.linalg.norm(W, "fro") ** 2
    nW = np.linalg.norm(gW, axis=1) ** 2
    nWh = np.linalg.norm(W @ sc.hhat, axis=0) ** 2
    C = np.abs(GWH) ** 2 + nW[:, None] * sh[None, :] + sg[:, None] * nWh[None, :] + sg[:, None] * sh[None, :] * fro2
    nWpsi = np.linalg.norm(W @ sc.psi, axis=0) ** 2
    D = (1 + eps) * (np.abs(gW @ sc.psi) ** 2).sum(1) + sc.sig_r2 * nW + sg * ((1 + eps) * nWpsi.sum() + sc.sig_r2 * fro2) + sc.s2
    S = np.abs(np.diag(GWH)) ** 2
    a = nWh + sh * fro2
    b = (1 + eps) * nWpsi.sum() + sc.sig_r2 * fro2
    return C, D, S, a, b


def rate_from(p, C, D, S, kappa2, sc):
    Om = C @ p + D - sc.s2; Omp = Om - p * S
    g = kappa2 * p * S / (kappa2 * Omp + sc.s2)
    return 0.5 * np.log2(1 + g).sum(), g


def tr_worst(W, h, dlt):
    Mm = W.conj().T @ W; lam, Q = np.linalg.eigh(Mm); c = Q.conj().T @ h
    base = float(np.sum(lam * np.abs(c) ** 2))
    if dlt <= 0: return base, np.zeros_like(h)
    lmax = lam[-1]
    f = lambda kap: np.sqrt(np.sum((lam * np.abs(c) / (kap - lam)) ** 2))
    lo = lmax * (1 + 1e-13) + 1e-300
    if f(lo) <= dlt:
        Dl = lam * c / (lo - lam); Dl = Dl.astype(complex)
        extra = np.sqrt(max(dlt ** 2 - np.sum(np.abs(Dl) ** 2), 0)); Dl[-1] += extra
    else:
        hi = lmax * 2 + 1.0
        while f(hi) > dlt: hi *= 2
        for _ in range(100):
            mid = np.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi)
            if f(mid) > dlt: lo = mid
            else: hi = mid
        kap = 0.5 * (lo + hi); Dl = lam * c / (kap - lam)
    Delta = Q @ Dl
    return float(np.linalg.norm(W @ (h + Delta)) ** 2), Delta


def power_terms_wc(W, sc, eps_ball):
    q95 = Q95(sc.Nr); dl = np.sqrt(sigma_h2(sc, eps_ball) * q95)
    a = np.zeros(sc.M); Dels = np.zeros((sc.Nr, sc.M), dtype=complex)
    for j in range(sc.M):
        a[j], Dels[:, j] = tr_worst(W, sc.hhat[:, j], dl[j])
    fro2 = np.linalg.norm(W, "fro") ** 2
    b = np.linalg.norm(W @ sc.psi) ** 2 + sc.sig_r2 * fro2
    return a, b, Dels


# ------------------------------------------------------------------ WMMSE relay step (eq. 28-31)
def wmmse_step(W, p, sc, eps, Pmax, Rpow=None):
    """one WMMSE update of W with the rate model at eps; power constraint w^H (Rpow^T kron I) w <= Pmax.
    Rpow defaults to Ry(eps) (average power eq. 20)."""
    Nt, Nr, M = sc.Nt, sc.Nr, sc.M
    R = Ry(sc, p, eps); Rp = R if Rpow is None else Rpow
    sg = sigma_g2(sc, eps)
    C, D, S, a, b = rate_terms(W, sc, eps)
    w0 = W.reshape(-1, order="F")
    Theta = np.array([np.real(w0.conj() @ B_m(sc, R, sg, m) @ w0) + sc.s2 for m in range(M)])
    f = np.array([np.kron(sc.hhat[:, m][None, :], sc.ghat[:, m][None, :]) for m in range(M)])[:, 0, :]   # M x NtNr
    gmat = f @ w0                                                         # g_m^T W h_m
    r = np.sqrt(p) * gmat.conj() / Theta
    e = np.abs(r) ** 2 * Theta - 2 * np.real(r * np.sqrt(p) * gmat) + 1
    om = 1.0 / e
    A = sum(om[m] * abs(r[m]) ** 2 * B_m(sc, R, sg, m) for m in range(M))
    q = sum(om[m] * r[m] * np.sqrt(p[m]) * f[m] for m in range(M))        # row
    c = q.conj()
    P = np.kron(Rp.T, np.eye(Nt))
    sPm = np.sqrt(Pmax)
    A_s, c_s = Pmax * A, sPm * c                                         # variable x = w/sqrt(Pmax)
    n = A.shape[0]; reg = 1e-14 * np.trace(A_s).real / n
    def sol(lam): return np.linalg.solve(A_s + lam * P + reg * np.eye(n), c_s)
    def pw(x): return np.real(x.conj() @ P @ x)
    x = sol(0.0)
    if pw(x) > 1:
        lo, hi = 1e-12 * np.trace(A_s).real / np.trace(P).real, 1.0 * np.trace(A_s).real / np.trace(P).real
        while pw(sol(hi)) > 1: hi *= 4
        for _ in range(60):
            mid = np.sqrt(lo * hi);
            if pw(sol(mid)) > 1: lo = mid
            else: hi = mid
        x = sol(hi)
    return (sPm * x).reshape(Nt, Nr, order="F")


def B_m(sc, R, sg, m):
    Nt, Nr = sc.Nt, sc.Nr
    Gm = np.kron(np.eye(Nr), sc.ghat[:, m][None, :])                      # Nr x NtNr
    return Gm.conj().T @ R.T @ Gm + sg[m] * np.kron(R.T, np.eye(Nt))


def scale_to_power(W, sc, p, eps, wc=False):
    if wc:
        a, b, _ = power_terms_wc(W, sc, eps)
        Pt = a @ p + b
    else:
        Pt = np.real(np.trace(W @ Ry(sc, p, eps) @ W.conj().T))
    return W * np.sqrt(sc.Pmax / Pt)


# ------------------------------------------------------------------ fixed relay structures (as in isac_cr/benchmarks.py)
def fixed_relay(sc, kind):
    G, H = sc.ghat, sc.hhat
    if kind == "MRC/MRT": return G.conj() @ H.conj().T
    if kind == "ZF":
        GtG = G.T @ G.conj(); HhH = H.conj().T @ H; reg = 1e-10 * np.trace(GtG).real / sc.M
        Gp = G.conj() @ np.linalg.inv(GtG + reg * np.eye(sc.M)); regh = 1e-10 * np.trace(HhH).real / sc.M
        Hp = np.linalg.inv(HhH + regh * np.eye(sc.M)) @ H.conj().T
        return Gp @ Hp
    if kind == "LMMSE":
        d = min(sc.M, sc.Nr, sc.Nt)
        U1, _, _ = np.linalg.svd(H); _, _, V2h = np.linalg.svd(G.T)
        return V2h[:d, :].conj().T @ U1[:, :d].conj().T
    raise ValueError(kind)


# ------------------------------------------------------------------ power step (eq. 26-27) by 3-start SLSQP
def power_step(p_cur, W0, sc, eps_rate, rows, rhs, a_b=None):
    C, D, S, a, b = rate_terms(W0, sc, eps_rate)
    if a_b is not None: a, b = a_b
    Pc = sc.Pmax_p if hasattr(sc, "Pmax_p") else sc.Pc

    def parts(x):
        p = Pc * x; k2 = sc.Pmax / (a @ p + b)
        Om = C @ p + D - sc.s2; Omp = Om - p * S
        Th, Thp = k2 * Om + sc.s2, k2 * Omp + sc.s2
        return p, k2, Om, Omp, Th, Thp

    def f(x):
        p, k2, Om, Omp, Th, Thp = parts(x)
        return -np.sum(np.log(Th / Thp)) / np.log(2) * 0.5

    def g(x):
        p, k2, Om, Omp, Th, Thp = parts(x)
        zeta = -k2 * a / (a @ p + b)
        G = (k2 * C + np.outer(Om, zeta)) / Th[:, None] - (k2 * (C - np.diag(S)) + np.outer(Omp, zeta)) / Thp[:, None]
        return -G.sum(0) * Pc / np.log(2) * 0.5

    x0 = p_cur / Pc
    f0 = f(x0)
    if np.any(rhs <= 0): return p_cur * 0.0 if np.all(rhs <= 0) else p_cur, f0, f0
    An = rows * Pc / rhs[:, None]
    cons = [{"type": "ineq", "fun": lambda x: 1 - An @ x, "jac": lambda x: -An}]
    best, fb = x0.copy(), f0
    starts = [x0, 0.5 * x0, np.full(sc.M, 1.0) * min(1.0, 1.0 / max(An.sum(1).max(), 1e-12))]
    for xs in starts:
        try:
            r = minimize(f, np.clip(xs, 0, 1), jac=g, bounds=[(0, 1)] * sc.M, constraints=cons, method="SLSQP", options={"maxiter": 100, "ftol": 1e-10})
        except Exception:
            continue
        x = np.clip(r.x, 0, 1)
        if np.all(An @ x <= 1 + 1e-7) and f(x) < fb - 1e-12: best, fb = x, f(x)
    return Pc * best, f0, fb


# ------------------------------------------------------------------ Algorithm 1 and baselines
class Design:
    pass


def hu_rows(sc, U, eta, delta=None):
    K, M = sc.K, sc.M
    mu2 = np.abs(np.einsum("rk,rk->k", U.conj(), sc.psi)) ** 2
    Inom = (np.abs(U.conj().T @ sc.psi) ** 2).sum(1) - mu2
    cpl = np.abs(U.conj().T @ sc.hhat)
    if delta is not None: cpl = cpl + delta[None, :]
    rows = cpl ** 2
    rhs = mu2 / eta - Inom - sc.sig_r2
    return rows, np.maximum(rhs, 0.0)


def run_design(sc, scheme, eps_true, law, eta_floor=None, relay="opt", max_iter=40, tol=1e-4, log=None):
    """scheme in {'prop','ign','perfect','hu','robust'}; relay in {'opt','ZF','MRC/MRT','LMMSE'}"""
    t0 = time.perf_counter()
    cert = scheme in ("prop", "ign", "perfect")
    eps_d = eps_true if scheme == "prop" else 0.0                         # sensing/filters design epsilon
    eps_rate = eps_true if scheme in ("prop", "robust") else 0.0          # rate / power model epsilon used inside the design
    wc = scheme == "robust"
    q95 = Q95(sc.Nr); delta = np.sqrt(sigma_h2(sc, eps_true) * q95) if wc else None
    target = BETA + MARGIN
    eta = eta_floor if eta_floor is not None else eta_beta()
    M, Pc = sc.M, sc.Pc
    d0 = np.full(M, Pc)

    # ---- step 1: p <- largest common scaling of Pc*1 meeting the sensing constraint
    if cert:
        lo, hi = 0.0, 1.0
        def okc(s):
            p = s * d0; U = filters(sc, p, eps_d); return law.pd_of(U, sc, p, eps_d).min() >= target
        if okc(1.0): lo = 1.0
        else:
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                if okc(mid): lo = mid
                else: hi = mid
        p = lo * d0
    else:
        U = filters(sc, d0, 0.0); rows, rhs = hu_rows(sc, U, eta, delta)
        lam = min(1.0, float(np.min(rhs / np.maximum(rows @ d0, 1e-30)))); p = max(lam, 0.0) * d0
    # ---- step 2: relay init
    if relay == "opt":
        W = scale_to_power(sc.ghat.conj() @ sc.hhat.conj().T, sc, p, eps_rate, wc)
        for _ in range(4):
            W = wmmse_step(W, np.maximum(p, 1e-12), sc, eps_rate, sc.Pmax, Rpow=(wc_cov(sc, W, p, eps_true) if wc else None))
            W = scale_to_power(W, sc, p, eps_rate, wc)
    else:
        W = scale_to_power(fixed_relay(sc, relay), sc, p, eps_rate, wc)
    Rhist, prev = [], None
    it = 0
    for it in range(1, max_iter + 1):
        # ---- filters and constraint rows
        U = filters(sc, p, eps_d)
        if cert:
            pr = law.prep(U, sc, eps_d)
            lam = law.max_scale(pr, np.maximum(p, 1e-30 * Pc), target, cap=2 * Pc / max(p.max(), 1e-30) if p.max() > 0 else 1.0)
            sh = sigma_h2(sc, eps_d); cbar = np.abs(U.conj().T @ sc.hhat) ** 2 + sh[None, :]
            rows = cbar; rhs = lam * (cbar @ p)
        else:
            rows, rhs = hu_rows(sc, U, eta, delta)
        # ---- power step
        if wc:
            a, b, _ = power_terms_wc(W, sc, eps_true); ab = (a, b)
        else: ab = None
        p_new, f0, f1 = power_step(p, W, sc, eps_rate, rows, rhs, ab)
        if p_new.max() <= 0: p_new = p
        p = p_new
        W = scale_to_power(W, sc, p, eps_rate, wc)
        # ---- relay step
        if relay == "opt":
            Rp = wc_cov(sc, W, p, eps_true) if wc else None
            W = wmmse_step(W, np.maximum(p, 1e-12), sc, eps_rate, sc.Pmax, Rpow=Rp)
            W = scale_to_power(W, sc, p, eps_rate, wc)
        else:
            W = scale_to_power(fixed_relay(sc, relay), sc, p, eps_rate, wc)
        C, D, S, a_, b_ = rate_terms(W, sc, eps_rate)
        Rm = model_rate(sc, W, p, eps_rate)
        Rhist.append(Rm)
        if log is not None:
            pmin = float(law.pd_of(filters(sc, p, eps_d), sc, p, eps_true).min())   # TRUE worst-target PD of the current iterate
            log(dict(it=it, R=Rm, p=p.copy(), minpd=pmin))
        if prev is not None and abs(Rm - prev) / max(abs(prev), 1e-12) < tol: break
        prev = Rm
    # ---- final repair
    U = filters(sc, p, eps_d)
    repaired = 1.0
    if cert:
        pr = law.prep(U, sc, eps_d)
        lam = law.max_scale(pr, p, target, cap=1.0)
        repaired = float(np.clip(lam.min(), 0.0, 1.0))
    else:
        rows, rhs = hu_rows(sc, U, eta, delta)
        repaired = float(np.clip(np.min(rhs / np.maximum(rows @ p, 1e-30)), 0, 1))
    p = repaired * p
    W = scale_to_power(W, sc, p, eps_rate, wc) if relay == "opt" or True else W
    Dg = Design(); Dg.p, Dg.W, Dg.U, Dg.iters, Dg.Rhist, Dg.repair, Dg.time = p, W, U, it, Rhist, repaired, time.perf_counter() - t0
    return Dg


def wc_cov(sc, W, p, eps_ball):
    a, b, Dels = power_terms_wc(W, sc, eps_ball)
    R = sc.psi @ sc.psi.conj().T + sc.sig_r2 * np.eye(sc.Nr)
    for m in range(sc.M):
        h = sc.hhat[:, m] + Dels[:, m]; R = R + p[m] * np.outer(h, h.conj())
    return R


def model_rate(sc, W, p, eps):
    C, D, S, a, b = rate_terms(W, sc, eps)
    Pt = a @ p + b; kap2 = sc.Pmax / Pt if True else 1.0
    # rate at the actual W (already scaled): kappa2 = 1 for W as given
    return rate_from(p, C, D, S, 1.0, sc)[0]


# ------------------------------------------------------------------ evaluation (Sec. VI-A): true eps, gain reduced if power exceeds Pmax, PD with calibrated threshold
def evaluate(sc, dg, eps_true, law_eval):
    Pt = np.real(np.trace(dg.W @ Ry(sc, dg.p, eps_true) @ dg.W.conj().T))
    W = dg.W * (np.sqrt(sc.Pmax / Pt) if Pt > sc.Pmax else 1.0)
    C, D, S, a, b = rate_terms(W, sc, eps_true)
    R, g = rate_from(dg.p, C, D, S, 1.0, sc)
    pd = law_eval.pd_of(dg.U, sc, dg.p, eps_true)
    pe = law_eval.pd_cond(dg.U, sc, dg.p, eps_true)
    return dict(R=R, pd=pd, minpd=float(pd.min()), prot=bool(pd.min() >= BETA), Pt=Pt / sc.Pmax,
                prot_real=float(np.mean(pe.min(0) >= BETA)), minpd_p5=float(np.percentile(pe.min(0), 5)))
