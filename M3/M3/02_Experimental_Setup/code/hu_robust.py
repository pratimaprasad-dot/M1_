"""
[Hu] Section IV: robust design under bounded CSI errors (problem (23)) by GLDT + cutting set (Algorithms 2 and 3).

  Uncertainty sets (21)-(22): h_m = hbar_m + delta_m, ||delta_m|| <= eps_m ;  g_m = gbar_m + sig_m, ||sig_m|| <= eps'_m.
  Algorithm 2 (sample problem (26)): GLDT surrogates (31), (33), (35) with closed-form auxiliaries (32), (34), (36);
      block 1 = (alpha, v, r) and block 2 = (W, u_k, r) of problem (37), alternated until |sum r change| < nu.
  Algorithm 3 (cutting set): pessimisation oracles (38)-(40); every violated constraint adds its worst-case
      error to the sample set; repeat until no constraint is violated.

Pessimisation oracles (exact, replacing [Hu]'s SDR + Dinkelbach, which [Hu] states is tight):
  (38) relay power: per m, max ||W(hbar_m + delta)||^2 over the ball  -> trust-region (secular equation).
  (39) rate of SR_m: alternate over sig_m (Dinkelbach on a ratio of two quadratics; each Dinkelbach step is a
       trust-region subproblem solved globally) and over Delta (closed form: delta_m shrinks |g^T W h_m|,
       delta_j, j != m, maximise |g^T W h_j|).
  (40) sensing SINR of target k for fixed u_k: closed form, delta_m = eps_m e^{j angle(u_k^H h_m)} u_k/||u_k||.
Units and assumptions as in hu_core.py.
"""
import time
import numpy as np
import cvxpy as cp
from hu_core import HuSys, _cplx_sq, SOLVER_OPTS, algorithm1, init_point


def _solve(prob):
    """CLARABEL, then SCS as fallback; accepts optimal and optimal_inaccurate"""
    for opts in (SOLVER_OPTS, dict(solver=cp.SCS, max_iters=20000, eps=1e-7, canon_backend=cp.SCIPY_CANON_BACKEND)):
        try:
            prob.solve(**opts)
            if prob.status in ("optimal", "optimal_inaccurate"): return True
        except cp.error.SolverError:
            pass
    return False


# ====================================================================== trust-region subproblem
def trs_min(Q, b, c, rad):
    """global min over ||x|| <= rad of x^H Q x + 2 Re(b^H x) + c  (Q Hermitian, possibly indefinite)"""
    lam, V = np.linalg.eigh(Q); bt = V.conj().T @ b
    lmin = lam[0]
    def xn(mu): return np.sqrt(np.sum(np.abs(bt) ** 2 / (lam + mu) ** 2))
    if lmin > 0:
        x = -np.linalg.solve(Q, b)
        if np.linalg.norm(x) <= rad:
            return x, float(np.real(x.conj() @ Q @ x) + 2 * np.real(b.conj() @ x) + c)
    lo = max(0.0, -lmin) + 1e-14 * max(1.0, abs(lmin))
    if xn(lo) <= rad:                                                     # hard case
        y = -bt / (lam + lo); y[0] = 0.0 if abs(bt[0]) < 1e-14 * np.linalg.norm(bt) + 1e-300 else y[0]
        extra = np.sqrt(max(rad ** 2 - np.sum(np.abs(y) ** 2), 0.0)); y[0] = y[0] + extra
        x = V @ y
    else:
        hi = lo + np.linalg.norm(b) / rad + abs(lmin) + 1.0
        while xn(hi) > rad: hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if xn(mid) > rad: lo = mid
            else: hi = mid
            if hi - lo < 1e-13 * hi: break
        x = V @ (-bt / (lam + hi))
    return x, float(np.real(x.conj() @ Q @ x) + 2 * np.real(b.conj() @ x) + c)


# ====================================================================== robust problem object
class Robust:
    def __init__(self, sys: HuSys, eps_h, eps_g, eta_db, nu=1e-5):
        """eps_h, eps_g: per-user ball radii in the units of sys.H and sys.G"""
        self.s = sys; self.eh = np.asarray(eps_h, float); self.eg = np.asarray(eps_g, float)
        self.eta = 10 ** (eta_db / 10); self.eta_db = eta_db; self.nu = nu
        self.samples = [(np.zeros_like(sys.H), np.zeros_like(sys.G))]

    # ---- per-sample channels (internal scaling)
    def chans(self, t):
        dH, dG = self.samples[t]
        return (self.s.H + dH) * np.sqrt(self.s.Pc), (self.s.G + dG) * np.sqrt(self.s.Pmax_n)

    # ---- oracles -----------------------------------------------------------------
    def worst_power(self, X, q, psi):
        """(38): per-user worst relay power; returns (normalised power, Delta)"""
        s = self.s; Hs = s.H * np.sqrt(s.Pc); D = np.zeros_like(s.H)
        tot = float(np.linalg.norm(X @ psi) ** 2 + np.linalg.norm(X) ** 2)
        Mx = X.conj().T @ X
        for m in range(s.M):
            r = self.eh[m] * np.sqrt(s.Pc)
            x, val = trs_min(-Mx, -Mx @ Hs[:, m], -np.real(Hs[:, m].conj() @ Mx @ Hs[:, m]), r)
            D[:, m] = x / np.sqrt(s.Pc); tot += q[m] * (-val)
        return tot, D

    def worst_sense(self, U, q, psi):
        """(40): per-target worst sensing SINR for fixed u_k; returns (sinr vector, list of Delta)"""
        s = self.s; out = []; Ds = []
        for k in range(s.K):
            u = U[:, k]; nu_ = np.linalg.norm(u)
            D = np.zeros_like(s.H)
            for m in range(s.M):
                ph = np.angle(u.conj() @ s.H[:, m])
                D[:, m] = self.eh[m] * np.exp(1j * ph) * u / nu_
            Ht = (s.H + D) * np.sqrt(s.Pc)
            UP = np.abs(u.conj() @ psi) ** 2
            intf = UP.sum() - UP[k] + (np.abs(u.conj() @ Ht) ** 2) @ q + nu_ ** 2
            out.append(UP[k] / intf); Ds.append(D)
        return np.array(out), Ds

    def worst_rate(self, X, q, psi, m, iters=8):
        """(39)/(50): min gamma_m over (Delta, sig_m) by alternation; returns (gamma, dH, dG)"""
        s = self.s; sPc, sPm = np.sqrt(s.Pc), np.sqrt(s.Pmax_n)
        dH = np.zeros_like(s.H); dG = np.zeros_like(s.G)
        gam_prev = None
        for _ in range(iters):
            # --- sig_m step: gamma = (g^H A g) / (g^H B g + 1), g = gbar + sig (internal scaling of g)
            Ht = (s.H + dH) * sPc
            w = X @ Ht                                                     # Nt x M : W h_j
            A = q[m] * np.outer(w[:, m].conj(), w[:, m])                   # |w^T g|^2 = g^H (w* w^T) g
            others = [j for j in range(s.M) if j != m]
            Bm = sum(np.outer(c.conj(), c) for c in (X @ psi).T) + X.conj() @ X.T
            for j in others: Bm = Bm + q[j] * np.outer(w[:, j].conj(), w[:, j])
            gb = s.G[:, m] * sPm; rad = self.eg[m] * sPm
            N = lambda x: np.real((gb + x).conj() @ A @ (gb + x))
            Dn = lambda x: np.real((gb + x).conj() @ Bm @ (gb + x)) + 1.0
            x = dG[:, m] * sPm; chi = N(x) / Dn(x)
            for _ in range(50):
                Q = A - chi * Bm; b = Q @ gb; c = np.real(gb.conj() @ Q @ gb) - chi
                x, f = trs_min(Q, b, c, rad)
                chi_new = N(x) / Dn(x)
                if abs(chi_new - chi) <= 1e-10 * max(chi, 1e-30) or f > -1e-12 * max(abs(chi), 1e-30): chi = min(chi, chi_new); break
                chi = chi_new
            dG[:, m] = x / sPm
            # --- Delta step (closed form)
            g = (s.G[:, m] + dG[:, m]) * sPm
            cvec = (g @ X)                                                 # row g^T W (Nr), g^T W h = cvec @ h
            nc = np.linalg.norm(cvec)
            for j in range(s.M):
                hj = s.H[:, j] * sPc; z = cvec @ hj; r = self.eh[j] * sPc
                dirn = cvec.conj() / nc if nc > 0 else np.zeros_like(cvec)
                if j == m:   # shrink |c h|: move against the phase of z
                    step = -min(r, abs(z) / nc if nc > 0 else 0.0) * np.exp(1j * np.angle(z)) * dirn
                else:        # grow |c h|
                    step = r * np.exp(1j * np.angle(z)) * dirn
                dH[:, j] = step / sPc
            gam = self.gamma_at(X, q, psi, dH, dG, m)
            if gam_prev is not None and abs(gam - gam_prev) <= 1e-9 * gam_prev: break
            gam_prev = gam
        return gam, dH, dG

    def gamma_at(self, X, q, psi, dH, dG, m):
        s = self.s
        g, _ = s.comm_sinr(X, q, psi, H=s.H + dH, G=s.G + dG)
        return g[m]

    def worst_sar(self, X, q, psi):
        """sum_m 0.5 log2(1 + min gamma_m): the worst-case SAR reported in [Hu, Figs. 9-11]"""
        gs = [self.worst_rate(X, q, psi, m)[0] for m in range(self.s.M)]
        return 0.5 * np.sum(np.log2(1 + np.array(gs))), np.array(gs)

    # ---- Algorithm 2 on the current sample set ------------------------------------
    def restore(self, q, vt, X, U):
        """make the warm start feasible for every sample: shrink q (sensing), then scale X (power)"""
        s = self.s; psi = s.psi(vt)
        def sens_ok(qq):
            for t in range(len(self.samples)):
                Ht, _ = self.chans(t)
                UP = np.abs(U.conj().T @ psi) ** 2
                intf = UP.sum(1) - np.diag(UP) + (np.abs(U.conj().T @ Ht) ** 2) @ qq + np.linalg.norm(U, axis=0) ** 2
                if np.any(np.diag(UP) / intf < self.eta * (1 + 1e-9)): return False
            return True
        if not sens_ok(q):
            lo, hi = 0.0, 1.0
            if not sens_ok(0 * q): return None
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if sens_ok(mid * q): lo = mid
                else: hi = mid
            q = lo * q
        pw = max(self.power_at(X, q, psi, t) for t in range(len(self.samples)))
        X = X / np.sqrt(pw)
        return q, X

    def power_at(self, X, q, psi, t):
        Ht, _ = self.chans(t)
        R = psi @ psi.conj().T + (Ht * q) @ Ht.conj().T + np.eye(self.s.Nr)
        return float(np.real(np.trace(X @ R @ X.conj().T)))

    def aux(self, X, q, vt, U):
        s = self.s; psi = s.psi(vt); T = len(self.samples)
        zeta = np.zeros((T, s.K), complex); lam = np.zeros((T, s.M)); mu = np.zeros((T, s.M), complex)
        for t in range(T):
            Ht, Gt = self.chans(t)
            UP = U.conj().T @ psi
            phi = (np.abs(UP) ** 2).sum(1) - np.abs(np.diag(UP)) ** 2 + (np.abs(U.conj().T @ Ht) ** 2) @ q + np.linalg.norm(U, axis=0) ** 2
            zeta[t] = np.diag(UP) / phi                                    # (32)
            R = Gt.T @ X; Sx = R @ Ht
            J = (np.abs(R @ psi) ** 2).sum(1) + np.linalg.norm(R, axis=1) ** 2 + (np.abs(Sx) ** 2) @ q + 1.0   # Jbar (incl. own)
            sig = q * np.abs(np.diag(Sx)) ** 2
            lam[t] = sig / (J - sig)                                       # (34)
            mu[t] = np.sqrt(q) * np.diag(Sx) / J                           # (36)
        return zeta, lam, mu

    def block1(self, X, U, a, vt, zeta, lam, mu, kappa=True):
        """problem (37) over (alpha, v, r) for fixed (W, u).
        kappa=True: W = kappa X with kappa^2 = 1/max_t P_t^t(X; alpha, v) (relay-gain normalisation, as in
        hu_core.step_alpha_v): the noise term '1' of every Jbar becomes the auxiliary P >= P_t^t(X; alpha, v) (convex),
        and the per-sample power rows (26d) hold by construction. X must satisfy max_t P_t^t = 1 on entry."""
        s = self.s; M, K, T = s.M, s.K, len(self.samples)
        av = cp.Variable(M, nonneg=True); r = cp.Variable(M)
        if s.v_free:
            v = cp.Variable(s.Nt, complex=True)
            psi_e = [s.At[k] @ v for k in range(K)]
        psi_c = s.psi(vt)
        cons = [av <= 1, r <= 50]
        if s.v_free: cons.append(cp.norm(v, 2) <= 1)
        Paux = cp.Variable() if kappa else None
        for t in range(T):
            Ht, Gt = self.chans(t)
            hp = np.linalg.norm(X @ Ht, axis=0) ** 2
            ep = sum(_cplx_sq(X @ psi_e[k]) for k in range(K)) if s.v_free else float(np.linalg.norm(X @ psi_c) ** 2)
            Pt_t = ep + cp.sum(cp.multiply(hp, cp.square(av))) + float(np.linalg.norm(X) ** 2)
            cons.append(Pt_t <= (Paux if kappa else 1))
        for t in range(T):
            Ht, Gt = self.chans(t)
            R = Gt.T @ X; Sx = R @ Ht; Sabs2 = np.abs(Sx) ** 2
            for m in range(M):
                echo = _cplx_sq(cp.hstack([R[m] @ psi_e[k] for k in range(K)])) if s.v_free else float(np.sum(np.abs(R[m] @ psi_c) ** 2))
                Jb = echo + float(np.linalg.norm(R[m]) ** 2) + cp.sum(cp.multiply(Sabs2[m], cp.square(av))) + (Paux if kappa else 1.0)
                l = lam[t, m]
                cons.append((np.log1p(l) - l - r[m]) / (1 + l) + 2 * np.real(np.conj(mu[t, m]) * Sx[m, m]) * av[m] - abs(mu[t, m]) ** 2 * Jb >= 0)
            UH2 = np.abs(U.conj().T @ Ht) ** 2
            for k in range(K):
                u = U[:, k]
                if s.v_free:
                    num = u.conj() @ s.At[k] @ v
                    leak = _cplx_sq(cp.hstack([u.conj() @ s.At[i] @ v for i in range(K) if i != k]))
                else:
                    num = complex(u.conj() @ psi_c[:, k])
                    leak = float(sum(abs(u.conj() @ psi_c[:, i]) ** 2 for i in range(K) if i != k))
                phi = leak + cp.sum(cp.multiply(UH2[k], cp.square(av))) + float(np.linalg.norm(u) ** 2)
                cons.append(2 * cp.real(np.conj(zeta[t, k]) * num) - abs(zeta[t, k]) ** 2 * phi >= self.eta)
        prob = cp.Problem(cp.Maximize(cp.sum(r)), cons)
        if not _solve(prob) or av.value is None: return None
        vt_new = None
        if s.v_free:
            vt_new = np.asarray(v.value).ravel(); vt_new = vt_new / max(1.0, np.linalg.norm(vt_new))
        return np.clip(av.value, 0, 1), vt_new

    def block2(self, X, U, a, vt, zeta, lam, mu):
        """problem (37) over (W, u_k, r) for fixed (alpha, v)"""
        s = self.s; M, K, T = s.M, s.K, len(self.samples); q = a ** 2
        Xv = cp.Variable((s.Nt, s.Nr), complex=True); Uv = cp.Variable((s.Nr, K), complex=True); r = cp.Variable(M)
        psi = s.psi(vt)
        cons = [r <= 50]
        for t in range(T):
            Ht, Gt = self.chans(t)
            for m in range(M):
                row = Gt[:, m] @ Xv
                Jb = _cplx_sq(row @ psi) + _cplx_sq(row) + _cplx_sq(cp.multiply(np.sqrt(q), row @ Ht)) + 1.0
                l = lam[t, m]
                cons.append((np.log1p(l) - l - r[m]) / (1 + l) + 2 * cp.real(np.conj(mu[t, m]) * a[m] * (row @ Ht[:, m])) - abs(mu[t, m]) ** 2 * Jb >= 0)
            for k in range(K):
                u = Uv[:, k]
                num = psi[:, k] @ cp.conj(u)                                   # u_k^H psi_k
                others = [i for i in range(K) if i != k]
                phi = _cplx_sq(psi[:, others].T @ cp.conj(u)) \
                      + _cplx_sq(cp.multiply(np.sqrt(q), Ht.T @ cp.conj(u))) + _cplx_sq(u)
                cons.append(2 * cp.real(np.conj(zeta[t, k]) * num) - abs(zeta[t, k]) ** 2 * phi >= self.eta)
            Rc = psi @ psi.conj().T + (Ht * q) @ Ht.conj().T + np.eye(s.Nr)
            L = np.linalg.cholesky(Rc)
            cons.append(_cplx_sq(Xv @ L) <= 1)
        prob = cp.Problem(cp.Maximize(cp.sum(r)), cons)
        if not _solve(prob) or Xv.value is None: return None
        return Xv.value, Uv.value, float(np.sum(r.value))

    # ---- fast exact solvers of the two convex blocks (fixed radar beam): SLSQP with analytic gradients -------------
    def block1_fast(self, X, U, a, zeta, lam, mu, kappa=True):
        """same convex problem as block1 (fixed beam): variables (alpha, r, P)"""
        from scipy.optimize import minimize
        s = self.s; M, K, T = s.M, s.K, len(self.samples); psi = s.psi()
        rows_r, rows_s, rows_p = [], [], []
        for t in range(T):
            Ht, Gt = self.chans(t)
            R = Gt.T @ X; Sx = R @ Ht; Sabs2 = np.abs(Sx) ** 2
            c0 = (np.abs(R @ psi) ** 2).sum(1) + np.linalg.norm(R, axis=1) ** 2
            l = lam[t]
            rows_r.append(((np.log1p(l) - l) / (1 + l), 1 / (1 + l), 2 * np.real(np.conj(mu[t]) * np.diag(Sx)), np.abs(mu[t]) ** 2, c0, Sabs2))
            UP = U.conj().T @ psi; UH2 = np.abs(U.conj().T @ Ht) ** 2
            leak = (np.abs(UP) ** 2).sum(1) - np.abs(np.diag(UP)) ** 2 + np.linalg.norm(U, axis=0) ** 2
            rows_s.append((2 * np.real(np.conj(zeta[t]) * np.diag(UP)), np.abs(zeta[t]) ** 2, leak, UH2))
            rows_p.append((float(np.linalg.norm(X @ psi) ** 2 + np.linalg.norm(X) ** 2), np.linalg.norm(X @ Ht, axis=0) ** 2))
        n = 2 * M + 1
        def unpack(x): return x[:M], x[M:2 * M], x[2 * M]
        def cons(x):
            av, r, P = unpack(x); a2 = av ** 2; out = []
            Pn = P if kappa else 1.0
            for (k0, k1, lin, m2, c0, S2) in rows_r:
                out.append(k0 - k1 * r + lin * av - m2 * (c0 + S2 @ a2 + Pn))
            for (num, z2, leak, UH2) in rows_s:
                out.append(num - z2 * (leak + UH2 @ a2) - self.eta)
            for (p0, hp) in rows_p:
                out.append(np.atleast_1d((P if kappa else 1.0) - p0 - hp @ a2))
            return np.concatenate(out)
        def jac(x):
            av, r, P = unpack(x); J = []
            for (k0, k1, lin, m2, c0, S2) in rows_r:
                B = np.zeros((M, n))
                B[:, :M] = np.diag(lin) - 2 * m2[:, None] * S2 * av[None, :]
                B[:, M:2 * M] = -np.diag(k1)
                B[:, 2 * M] = -m2 if kappa else 0.0
                J.append(B)
            for (num, z2, leak, UH2) in rows_s:
                B = np.zeros((K, n)); B[:, :M] = -2 * z2[:, None] * UH2 * av[None, :]; J.append(B)
            for (p0, hp) in rows_p:
                B = np.zeros((1, n)); B[0, :M] = -2 * hp * av; B[0, 2 * M] = 1.0 if kappa else 0.0; J.append(B)
            return np.vstack(J)
        f = lambda x: -np.sum(x[M:2 * M])
        g = lambda x: np.concatenate([np.zeros(M), -np.ones(M), [0.0]])
        P0 = max(self.max_power(X, a ** 2, psi), 1e-12)
        r0 = np.min([(k0 + lin * a - m2 * (c0 + S2 @ a ** 2 + (P0 if kappa else 1.0))) / k1 for (k0, k1, lin, m2, c0, S2) in rows_r], axis=0)
        x0 = np.concatenate([a, r0 - 1e-9, [P0]])
        bnds = [(0, 1)] * M + [(None, 50)] * M + [(1e-12, None)]
        res = minimize(f, x0, jac=g, constraints=[dict(type="ineq", fun=cons, jac=jac)], bounds=bnds, method="SLSQP",
                       options=dict(maxiter=300, ftol=1e-12))
        x = res.x
        if not np.all(np.isfinite(x)) or cons(x).min() < -1e-6 * max(1.0, self.eta): return None
        if -f(x) < -f(x0) - 1e-9: return None
        return np.clip(x[:M], 0, 1), None

    def block2_fast(self, X, U, a, zeta, lam, mu):
        """same convex problem as block2 for W (fixed beam), variables (Re W, Im W, r); u_k updated separately"""
        from scipy.optimize import minimize
        s = self.s; M, T = s.M, len(self.samples); psi = s.psi(); q = a ** 2
        Nt, Nr = s.Nt, s.Nr; nw = Nt * Nr
        Cs, Gs, Hs = [], [], []
        for t in range(T):
            Ht, Gt = self.chans(t)
            Cs.append(psi @ psi.conj().T + np.eye(Nr) + (Ht * q) @ Ht.conj().T)      # rho C rho^H = Jbar - 1 ; tr(W C W^H) = P_t
            Gs.append(Gt); Hs.append(Ht)
        Cs = np.array(Cs); Gs = np.array(Gs); Hs = np.array(Hs)
        k0 = (np.log1p(lam) - lam) / (1 + lam); k1 = 1 / (1 + lam); m2 = np.abs(mu) ** 2
        cm = np.conj(mu) * a[None, :]                                             # T x M
        def W_of(x): return (x[:nw] + 1j * x[nw:2 * nw]).reshape(Nt, Nr)
        def cons(x):
            W = W_of(x); r = x[2 * nw:]
            rho = np.einsum("tnm,nr->tmr", Gs, W)                                 # g_m^T W per sample
            quad = np.real(np.einsum("tmr,trs,tms->tm", rho, Cs, rho.conj()))
            sig = np.einsum("tmr,trm->tm", rho, Hs)                               # g_m^T W h_m
            rate = k0 - k1 * r[None, :] + 2 * np.real(cm * sig) - m2 * (quad + 1.0)
            pw = 1.0 - np.real(np.einsum("nr,trs,ns->t", W, Cs, W.conj()))
            return np.concatenate([rate.ravel(), pw])
        def jac(x):
            W = W_of(x)
            rho = np.einsum("tnm,nr->tmr", Gs, W)
            rC = np.einsum("tmr,trs->tms", rho, Cs)
            dW = (-m2[:, :, None, None] * np.einsum("tnm,tms->tmns", Gs.conj(), rC)
                  + np.conj(cm)[:, :, None, None] * np.einsum("tnm,tsm->tmns", Gs.conj(), Hs.conj()))
            Jr = np.zeros((T, M, 2 * nw + M))
            Jr[:, :, :nw] = 2 * np.real(dW).reshape(T, M, nw); Jr[:, :, nw:2 * nw] = 2 * np.imag(dW).reshape(T, M, nw)
            for m in range(M): Jr[:, m, 2 * nw + m] = -k1[:, m]
            dP = -np.einsum("nr,trs->tns", W, Cs)
            Jp = np.zeros((T, 2 * nw + M)); Jp[:, :nw] = 2 * np.real(dP).reshape(T, nw); Jp[:, nw:2 * nw] = 2 * np.imag(dP).reshape(T, nw)
            return np.vstack([Jr.reshape(T * M, -1), Jp])
        x0w = np.concatenate([X.real.ravel(), X.imag.ravel()])
        c = cons(np.concatenate([x0w, np.zeros(M)]))
        r0 = np.min(c[:T * M].reshape(T, M) / k1, axis=0) - 1e-9
        x0 = np.concatenate([x0w, r0])
        f = lambda x: -np.sum(x[2 * nw:])
        g = lambda x: np.concatenate([np.zeros(2 * nw), -np.ones(M)])
        res = minimize(f, x0, jac=g, constraints=[dict(type="ineq", fun=cons, jac=jac)], method="SLSQP",
                       bounds=[(None, None)] * (2 * nw) + [(None, 50)] * M, options=dict(maxiter=400, ftol=1e-12))
        x = res.x
        if not np.all(np.isfinite(x)) or cons(x).min() < -1e-6: return None
        if -f(x) < -f(x0) - 1e-9: return None
        return W_of(x), float(np.sum(x[2 * nw:]))

    def u_update(self, U, q, vt):
        """u_k: diagonally loaded MVDR (robust to the h-ball), kept only if it raises the worst-sample sensing SINR"""
        s = self.s; psi = s.psi(vt)
        def wsinr(Uc):
            out = np.full(s.K, np.inf)
            for t in range(len(self.samples)):
                Ht, _ = self.chans(t)
                UP = np.abs(Uc.conj().T @ psi) ** 2
                intf = UP.sum(1) - np.diag(UP) + (np.abs(Uc.conj().T @ Ht) ** 2) @ q + np.linalg.norm(Uc, axis=0) ** 2
                out = np.minimum(out, np.diag(UP) / intf)
            return out
        Hs = s.H * np.sqrt(s.Pc)
        Rall = psi @ psi.conj().T + (Hs * q) @ Hs.conj().T + np.eye(s.Nr) * (1 + np.sum(q * (self.eh * np.sqrt(s.Pc)) ** 2))
        Un = U.copy(); old = wsinr(U)
        for k in range(s.K):
            Rk = Rall - np.outer(psi[:, k], psi[:, k].conj())
            x = np.linalg.solve(Rk, psi[:, k]); x = x / np.linalg.norm(x)
            Ut = Un.copy(); Ut[:, k] = x
            if wsinr(Ut)[k] > old[k]: Un = Ut
        return Un

    def max_power(self, X, q, psi):
        return max(self.power_at(X, q, psi, t) for t in range(len(self.samples)))

    def algorithm2(self, a, vt, X, U, max_iter=100, kappa=True):
        fast = not self.s.v_free
        hist = []; prev = None; fails = 0; it = 0
        for it in range(1, max_iter + 1):
            if kappa: X = X / np.sqrt(self.max_power(X, a ** 2, self.s.psi(vt)))
            zeta, lam, mu = self.aux(X, a ** 2, vt, U)
            b1 = self.block1_fast(X, U, a, zeta, lam, mu, kappa) if fast else self.block1(X, U, a, vt, zeta, lam, mu, kappa=kappa)
            if b1 is None: fails += 1
            else:
                a, vt_ = b1
                if not fast: vt = vt_
                if kappa: X = X / np.sqrt(self.max_power(X, a ** 2, self.s.psi(vt)))
            if fast:
                b2 = self.block2_fast(X, U, a, zeta, lam, mu)
                if b2 is not None:
                    X, rs = b2
                    U = self.u_update(U, a ** 2, vt)
            else:
                b2 = self.block2(X, U, a, vt, zeta, lam, mu)
                if b2 is not None: X, U, rs = b2
            if b2 is None:
                fails += 1
                if b1 is None: break
                continue
            hist.append(rs)
            if prev is not None and abs(rs - prev) < self.nu: break
            prev = rs
        return a, vt, X, U, it, fails, hist

    # ---- Algorithm 3 -------------------------------------------------------------
    def algorithm3(self, start=None, max_outer=25, max_inner=100, tol=1e-4, log=None, kappa=True):
        t0 = time.perf_counter(); s = self.s
        if start is None:
            st = init_point(s, self.eta)
            if st is None: return None
            q, vt, X = st
            U = s.mvdr(q, s.psi(vt))
        else:
            q, vt, X, U = start
        inner_total, fails, outer = 0, 0, 0
        status = "max_outer"
        for outer in range(1, max_outer + 1):
            rs = self.restore(q, vt, X, U)
            if rs is None: status = "infeasible_at_zero"; break
            q, X = rs
            a, vt, X, U, it, f, hist = self.algorithm2(np.sqrt(q), vt, X, U, max_inner, kappa=kappa)
            q = a ** 2; inner_total += it; fails += f
            psi = s.psi(vt)
            # pessimisation and constraint checks (26b)-(26d)
            new = []
            pw, Dp = self.worst_power(X, q, psi)
            if pw > 1 + tol: new.append((Dp, np.zeros_like(s.G)))
            sv, Ds = self.worst_sense(U, q, psi)
            for k in range(s.K):
                if sv[k] < self.eta * (1 - tol): new.append((Ds[k], np.zeros_like(s.G)))
            # rate cuts: compare worst-case rate of each user with its epigraph value at the sample set
            gnom = np.min([s.comm_sinr(X, q, psi, H=s.H + dH, G=s.G + dG)[0] for dH, dG in self.samples], axis=0)
            for m in range(s.M):
                gw, dH, dG = self.worst_rate(X, q, psi, m)
                if np.log1p(gw) < np.log1p(gnom[m]) - tol: new.append((dH, dG))
            if log: log(outer, len(self.samples), len(new), self.worst_sar(X, q, psi)[0])
            if not new: status = "converged"; break
            self.samples += new
        psi = s.psi(vt)
        U_fin = U
        res = dict(q=q, vt=vt, X=X, U=U_fin, outer=outer, inner=inner_total, fails=fails, status=status,
                   T=len(self.samples), time=time.perf_counter() - t0)
        if status != "infeasible_at_zero":
            res["wc_sar"], res["wc_gam"] = self.worst_sar(X, q, psi)
            res["wc_sinr_db"] = float(10 * np.log10(self.worst_sense(U_fin, q, psi)[0].min()))
            res["wc_power"] = float(self.worst_power(X, q, psi)[0])
            res["nom_sar"] = float(s.sar(X, q, psi))
        return res


def radii_hu(sys, eps_hat):
    """[Hu, Sec. VI]: normalised radius eps_hat = eps_m/||hbar_m|| = eps'_m/||gbar_m||"""
    return eps_hat * np.linalg.norm(sys.H, axis=0), eps_hat * np.linalg.norm(sys.G, axis=0)
