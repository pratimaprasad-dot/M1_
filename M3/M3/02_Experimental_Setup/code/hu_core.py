"""
Independent implementation of the base paper's algorithms:
  C. Hu, Q. Li, Q. Zhang, Q. Li, "Cognitive Radio-Enabled ISAC Systems: Joint Power Allocation and Collaborative
  Beamforming", IEEE TCCN, vol. 12, pp. 8579-8596, 2026 (called [Hu] below).

  Algorithm 1 (PF-IBCD, perfect CSI, problem (8)):  u_k by MVDR (10); (alpha, v) by the convex surrogate (17);
                                                     W by the convex surrogate (20); stop when |dSAR| < nu.
  Fixed-relay benchmarks of [Hu, Sec. VI-A]: ZF [48], MRC/MRT [49], LMMSE [50] (W structure fixed, scaled to (8c);
                                                     LMMSE's diagonal Phi optimised by the same surrogate as (20)).
  Algorithm 2/3 (GLDT-CS, bounded CSI errors, problem (23)) are in hu_robust.py.

Normalised units (all powers divided by sigma_r^2 = sigma^2):
  H  (Nr x M): ST->DBS channels divided by sigma_r;  G (Nt x M): DBS->SR channels (column g_m, [Hu] uses g_m^T);
  p_m = |alpha_m|^2 in W (<= Pc);  W such that P_t / sigma^2 <= Pmax_n;  noise = 1 at the DBS and at every SR.
  Echo: either a FIXED radar beam (echo vectors psi_k = A_k v / sigma_r, Nr x K) or a FREE beam v with
        A_k / sigma_r = sqrt(xi / Ps) b(theta_k) a(theta_k)^T e^{j phi_k} and ||v||^2 <= Ps (see ASSUMPTION A2).
Internally the solver works with q = p/Pc, Ht = H sqrt(Pc), X = W/sqrt(Pmax_n), Gt = G sqrt(Pmax_n), vt = v/sqrt(Ps),
so that every variable is O(1) for the interior-point solver.

ASSUMPTIONS (not stated in [Hu]; documented in the report):
  A1  target reflection gain: |beta_k|^2 Ps / sigma_r^2 = xi (one scalar, calibrated on development channels).
  A2  radar transmit budget ||v||^2 <= Ps.  [Hu]'s problem (8) has no radar power constraint; v enters only through
      A_k v (rank K < Nt), so without a budget the sensing constraint can be met by an unbounded echo that W then
      nulls. Ps = P^max is used, which only fixes the scale of xi.
  A3  initial point: equal-gain multibeam v, MRC/MRT W, and the largest common ST power meeting (8d).
"""
import time
import numpy as np
import cvxpy as cp

# canon_backend SCIPY: pure-Python model building (same problems); the compiled C++ backend (_cvxcore) is blocked
# by Windows Application Control on this machine since 2026-10-27.
SOLVER_OPTS = dict(solver=cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)


def dbm2w(x): return 10 ** (x / 10) / 1000


def steer(N, th):
    n = np.arange(N)
    return np.exp(-1j * np.pi * np.sin(th) * ((N - 1) / 2 - n))      # [Hu] a(theta), b(theta): centred ULA


class HuSys:
    """Scenario in normalised units. Use HuSys.from_hu(...) for the base paper's environment and
    HuSys.from_scn(sc) to wrap a v3_engine.Scn (common environment, fixed radar beam)."""

    def __init__(self, H, G, Pc, Pmax_n, psi=None, Ak=None, Ps=None):
        self.H, self.G = np.asarray(H, complex), np.asarray(G, complex)
        self.Nr, self.M = self.H.shape
        self.Nt = self.G.shape[0]
        self.Pc, self.Pmax_n = float(Pc), float(Pmax_n)
        self.v_free = Ak is not None
        if self.v_free:
            self.Ak, self.Ps = np.asarray(Ak, complex), float(Ps)        # K x Nr x Nt
            self.K = self.Ak.shape[0]
            self.At = self.Ak * np.sqrt(self.Ps)
        else:
            self.psi0 = np.asarray(psi, complex); self.K = self.psi0.shape[1]
        self.Ht = self.H * np.sqrt(self.Pc)
        self.Gt = self.G * np.sqrt(self.Pmax_n)

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_hu(cls, seed, Nt=8, Nr=8, M=4, K=4, Pc_dbm=20.0, Pmax_dbm=20.0, xi_db=10.0, d=100.0, iota=4.0,
                noise_dbm=-80.0, theta=(-np.pi / 4, 0.0, np.pi / 4, np.pi / 2), Hhat=None, Ghat=None):
        """[Hu, Sec. VI]: h_m = sqrt(d^-iota) CN(0,I), g_m likewise, d = 100 m, iota = 4, sigma^2 = -80 dBm."""
        rng = np.random.default_rng(seed)
        s2 = dbm2w(noise_dbm)
        cn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
        Hh = cn(Nr, M) if Hhat is None else Hhat
        Gh = cn(Nt, M) if Ghat is None else Ghat
        H = Hh * np.sqrt(d ** -iota) / np.sqrt(s2)
        G = Gh * np.sqrt(d ** -iota)
        Pmax = dbm2w(Pmax_dbm); Ps = Pmax                                   # A2
        ph = rng.uniform(0, 2 * np.pi, K)
        amp = np.sqrt(10 ** (xi_db / 10) / Ps)                              # A1: |beta|^2/sigma_r^2 = xi/Ps
        Ak = np.array([amp * np.exp(1j * ph[k]) * np.outer(steer(Nr, theta[k]), steer(Nt, theta[k])) for k in range(K)])
        o = cls(H, G, dbm2w(Pc_dbm), Pmax / s2, Ak=Ak, Ps=Ps)
        o.theta = np.array(theta[:K]); o.Hhat_raw, o.Ghat_raw = Hh, Gh
        return o

    @classmethod
    def from_scn(cls, sc, H=None, G=None):
        """wrap v3_engine.Scn: hhat already divided by sigma_r, Pmax already divided by sigma^2, fixed beam psi."""
        return cls(sc.hhat if H is None else H, sc.ghat if G is None else G, sc.Pc, sc.Pmax, psi=sc.psi)

    def with_channels(self, H, G):
        o = object.__new__(HuSys); o.__dict__.update(self.__dict__)
        o.H, o.G = np.asarray(H, complex), np.asarray(G, complex)
        o.Ht = o.H * np.sqrt(o.Pc); o.Gt = o.G * np.sqrt(o.Pmax_n)
        return o

    # ---------------------------------------------------------------- model (normalised; vt = v/sqrt(Ps))
    def psi(self, vt=None):
        """echo vectors A_k v / sigma_r (Nr x K)"""
        if not self.v_free: return self.psi0
        return np.einsum("krt,t->rk", self.At, vt)

    def v_init(self, eta=None, iters=60):
        """equal-gain multibeam; if eta is given, per-target weights are adapted (fixed-point power control at q = 0)
        so that the initial beam meets the sensing floor whenever a multibeam can (A3)."""
        A = np.array([np.conj(steer(self.Nt, t)) for t in self.theta])
        wts = np.ones(self.K)
        v = (wts[:, None] * A).sum(0); v /= np.linalg.norm(v)
        if eta is None: return v
        q0 = np.zeros(self.M)
        for _ in range(iters):
            psi = self.psi(v); s = self.sense_sinr(self.mvdr(q0, psi), q0, psi)
            if s.min() >= 2 * eta: break
            wts = wts * np.clip(np.sqrt(np.median(s) / s), 0.5, 2.0)
            v = (wts[:, None] * A).sum(0); v /= np.linalg.norm(v)
        return v

    def sense_sinr(self, U, q, psi, H=None):
        """eq. (2) for every k: |u_k^H A_k v|^2 / phi_k ; U is Nr x K"""
        Ht = self.Ht if H is None else H * np.sqrt(self.Pc)
        UP = np.abs(U.conj().T @ psi) ** 2
        sig = np.diag(UP).copy()
        intf = UP.sum(1) - sig + (np.abs(U.conj().T @ Ht) ** 2) @ q + np.linalg.norm(U, axis=0) ** 2
        return sig / intf

    def mvdr(self, q, psi, H=None):
        """eqs. (10)-(11) (normalised; the scale of u_k is irrelevant for the SINR)"""
        Ht = self.Ht if H is None else H * np.sqrt(self.Pc)
        Rall = psi @ psi.conj().T + (Ht * q) @ Ht.conj().T + np.eye(self.Nr)
        U = np.zeros((self.Nr, self.K), complex)
        for k in range(self.K):
            Rk = Rall - np.outer(psi[:, k], psi[:, k].conj())
            x = np.linalg.solve(Rk, psi[:, k]); U[:, k] = x / np.linalg.norm(x)
        return U

    def comm_sinr(self, X, q, psi, H=None, G=None):
        """eq. (6) for every m (normalised: X = W/sqrt(Pmax_n))"""
        Ht = self.Ht if H is None else H * np.sqrt(self.Pc)
        Gt = self.Gt if G is None else G * np.sqrt(self.Pmax_n)
        R = Gt.T @ X                                                      # M x Nr, row m = g_m^T W
        S = np.abs(R @ Ht) ** 2                                           # |g_m^T W h_j|^2
        Jm = (np.abs(R @ psi) ** 2).sum(1) + np.linalg.norm(R, axis=1) ** 2 + S @ q - np.diag(S) * q + 1.0
        return q * np.diag(S) / Jm, Jm

    def sar(self, X, q, psi, H=None, G=None):
        """sum achievable rate, eq. (7), bit/s/Hz (includes the 1/2 of the two-slot protocol)"""
        g, _ = self.comm_sinr(X, q, psi, H, G)
        return 0.5 * np.sum(np.log2(1 + g))

    def relay_power(self, X, q, psi, H=None):
        """eq. (4) / Pmax (normalised; feasible iff <= 1)"""
        Ht = self.Ht if H is None else H * np.sqrt(self.Pc)
        R = psi @ psi.conj().T + (Ht * q) @ Ht.conj().T + np.eye(self.Nr)
        return float(np.real(np.trace(X @ R @ X.conj().T)))

    def scale_X(self, X, q, psi, H=None):
        return X / np.sqrt(self.relay_power(X, q, psi, H))

    # ---------------------------------------------------------------- unnormalise
    def W_of(self, X): return X * np.sqrt(self.Pmax_n)
    def p_of(self, q): return q * self.Pc


# ====================================================================== fixed relay structures [Hu, Sec. VI-A]
def relay_structure(sys, kind):
    G, H = sys.G, sys.H
    Ghu = G.T                                                              # [Hu]'s G = [g_1,...,g_M]^T (M x Nt)
    if kind == "MRC/MRT":
        return Ghu.conj().T @ H.conj().T
    if kind == "ZF":
        return Ghu.conj().T @ np.linalg.inv(Ghu @ Ghu.conj().T) @ np.linalg.inv(H.conj().T @ H) @ H.conj().T
    if kind == "LMMSE":                                                    # basis only; Phi optimised separately
        d = min(sys.M, sys.Nr, sys.Nt)
        U1, _, _ = np.linalg.svd(H.conj().T)                               # [Hu]: H = U1 L1 V1^H with H = [h_1..h_M]
        _, _, V2h = np.linalg.svd(Ghu)                                     # G = U2 L2 V2^H
        # W maps Nr -> Nt: W = V2[:, :d] Phi U1'[:, :d]^H where U1' spans the column space of H (Nr x d)
        Uh, _, _ = np.linalg.svd(H)
        return V2h.conj().T[:, :d], Uh[:, :d]
    raise ValueError(kind)


# ====================================================================== Algorithm 1 sub-steps
def _cplx_sq(e):
    """|e|^2 summed, for a complex cvxpy expression (vector or scalar)"""
    return cp.sum_squares(cp.real(e)) + cp.sum_squares(cp.imag(e))


def step_alpha_v(sys, X, U, a_hat, vt_hat, eta, kappa=False):
    """Subproblem (17): concave surrogate (15) of the SAR in (alpha, v) with the DC sensing row (16) and (8b), (8c).
    alpha is the amplitude sqrt(q) (real, >= 0, WLOG since only |alpha|^2 matters).
    kappa=True: relay-gain-normalised variant. W = kappa X with kappa^2 = 1/P_t(X; alpha, v), so (8c) holds with
    equality for every (alpha, v) and gamma_m = q_m |g^T X h_m|^2 / (J_m - 1 + P_t(X; alpha, v)): the denominator is
    still convex in (alpha, v), Lemma 1 applies unchanged, and the ST powers are no longer locked by a fixed W.
    X must satisfy P_t(X; a_hat, v_hat) = 1 on entry; the caller rescales X after the step."""
    M, K = sys.M, sys.K
    a = cp.Variable(M, nonneg=True)
    R = sys.Gt.T @ X                                                       # rows g_m^T W
    S = np.abs(R @ sys.Ht) ** 2
    q_hat = a_hat ** 2
    if sys.v_free:
        v = cp.Variable(sys.Nt, complex=True)
        psi_v = [sys.At[k] @ v for k in range(K)]                          # affine echoes
        psi_hat = sys.psi(vt_hat)
    else:
        psi_hat = sys.psi()
    gam_hat, J_hat = sys.comm_sinr(X, q_hat, psi_hat)
    if sys.v_free:
        echo_pow = sum(_cplx_sq(X @ psi_v[k]) for k in range(K))
    else:
        echo_pow = float(np.linalg.norm(X @ psi_hat) ** 2)
    hp = np.linalg.norm(X @ sys.Ht, axis=0) ** 2
    Pt_expr = echo_pow + cp.sum(cp.multiply(hp, cp.square(a))) + float(np.linalg.norm(X) ** 2)
    obj = 0
    for m in range(M):
        c = gam_hat[m] / (1 + gam_hat[m])
        if sys.v_free:
            echo = _cplx_sq(cp.hstack([R[m] @ psi_v[k] for k in range(K)]))
        else:
            echo = float(np.sum(np.abs(R[m] @ psi_hat) ** 2))
        Jm = echo + float(np.linalg.norm(R[m]) ** 2) + cp.sum(cp.multiply(np.delete(S[m], m), cp.square(cp.hstack([a[j] for j in range(M) if j != m]))))              + (Pt_expr if kappa else 1.0)
        xterm = a_hat[m] * cp.inv_pos(2 * a[m] - a_hat[m])                 # >= a_hat^2 / a^2
        obj = obj + c * (2 - xterm - Jm / J_hat[m])
    cons = [a <= 1.0, a >= 0.5 * a_hat + 1e-9]
    if sys.v_free: cons.append(cp.norm(v, 2) <= 1.0)                       # A2
    if not kappa: cons.append(Pt_expr <= 1.0)                               # (8c) relay power, normalised
    # (16) DC sensing rows (u_k fixed)
    UH2 = np.abs(U.conj().T @ sys.Ht) ** 2
    for k in range(K):
        uk = U[:, k]
        if sys.v_free:
            sig_lin_hat = uk.conj() @ sys.At[k] @ vt_hat
            sig_lin = uk.conj() @ sys.At[k] @ v
            lin = 2 * cp.real(np.conj(sig_lin_hat) * sig_lin) - abs(sig_lin_hat) ** 2
            leak = _cplx_sq(cp.hstack([uk.conj() @ sys.At[i] @ v for i in range(K) if i != k])) if K > 1 else 0
        else:
            lin = float(abs(uk.conj() @ psi_hat[:, k]) ** 2)
            leak = float(sum(abs(uk.conj() @ psi_hat[:, i]) ** 2 for i in range(K) if i != k))
        cons.append(eta * (leak + cp.sum(cp.multiply(UH2[k], cp.square(a))) + float(np.linalg.norm(uk) ** 2)) - lin <= 0)
    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(**SOLVER_OPTS)
    except cp.error.SolverError:
        return None
    if a.value is None or prob.status not in ("optimal", "optimal_inaccurate"): return None
    a_new = np.clip(np.asarray(a.value).ravel(), 0, 1)
    vt_new = None
    if sys.v_free:
        vt_new = np.asarray(v.value).ravel(); nv = np.linalg.norm(vt_new)
        if nv > 1: vt_new = vt_new / nv
    return a_new, vt_new


def step_W(sys, X_hat, q, psi, basis=None):
    """Subproblem (20): concave surrogate (19) of the SAR in W subject to (8c).
    basis=(V2d, U1d) restricts W = V2d diag(phi) U1d^H (LMMSE benchmark, phi complex)."""
    M, K = sys.M, sys.K
    if basis is None:
        Xv = cp.Variable((sys.Nt, sys.Nr), complex=True); Xe = Xv
    else:
        V2d, U1d = basis; d = V2d.shape[1]
        phi = cp.Variable(d, complex=True)
        Xe = V2d @ cp.diag(phi) @ U1d.conj().T
    gam_hat, J_hat = sys.comm_sinr(X_hat, q, psi)
    Rh = sys.Gt.T @ X_hat
    obj = 0
    for m in range(M):
        gm = sys.Gt[:, m]
        r = gm @ Xe                                                       # row g_m^T W (affine)
        s_hat = Rh[m] @ sys.Ht[:, m]
        s_lin = r @ sys.Ht[:, m]
        lin = 2 * cp.real(np.conj(s_hat) * s_lin) - abs(s_hat) ** 2       # <= |g^T W h|^2
        ell = abs(s_hat) ** 2 * cp.inv_pos(lin)
        others = [j for j in range(M) if j != m]
        Jm = _cplx_sq(r @ psi) + _cplx_sq(r) + _cplx_sq(cp.multiply(np.sqrt(q[others]), r @ sys.Ht[:, others])) + 1.0
        c = gam_hat[m] / (1 + gam_hat[m])
        obj = obj + c * (2 - ell - Jm / J_hat[m])
    Rc = psi @ psi.conj().T + (sys.Ht * q) @ sys.Ht.conj().T + np.eye(sys.Nr)
    L = np.linalg.cholesky(Rc)
    cons = [_cplx_sq(Xe @ L) <= 1.0]
    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(**SOLVER_OPTS)
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"): return None
    Xn = Xv.value if basis is None else V2d @ np.diag(phi.value) @ U1d.conj().T
    return Xn


# ====================================================================== Algorithm 1
class Result:
    pass


def init_point(sys, eta, X0=None):
    """A3: equal-gain beam (if free), MRC/MRT relay, largest common ST power s*1 meeting (8d) with MVDR filters"""
    vt = sys.v_init(eta) if sys.v_free else None
    psi = sys.psi(vt)
    def ok(s):
        q = np.full(sys.M, s); U = sys.mvdr(q, psi)
        return sys.sense_sinr(U, q, psi).min() >= eta * (1 + 1e-9)
    if not ok(0.0): return None
    lo, hi = 0.0, 1.0
    if ok(1.0): lo = 1.0
    else:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if ok(mid): lo = mid
            else: hi = mid
    q = np.full(sys.M, max(lo, 1e-6) * 0.999)
    X = relay_structure(sys, "MRC/MRT") if X0 is None else X0
    X = sys.scale_X(X / np.sqrt(sys.Pmax_n), q, psi)
    return q, vt, X


def algorithm1(sys, eta_db, relay="opt", max_iter=100, tol=1e-5, log=None, kappa=False):
    """[Hu] Algorithm 1 (relay='opt') or a fixed-relay benchmark (relay in ZF, MRC/MRT, LMMSE).
    kappa=False: literal Algorithm 1 (W fixed with (8c) inside subproblem (17)).
    kappa=True : relay-gain-normalised subproblem (17) (see step_alpha_v); everything else identical.
    Returns the design and the SAR history; tol is [Hu]'s nu on |sum ln(1+gamma) change| (eq. (8a) units)."""
    t0 = time.perf_counter()
    eta = 10 ** (eta_db / 10)
    st = init_point(sys, eta)
    res = Result(); res.ok = st is not None
    if st is None:
        res.sar, res.iters, res.time, res.hist, res.feasible = 0.0, 0, time.perf_counter() - t0, [], False
        res.min_sinr_db, res.Pt, res.fails = -np.inf, 0.0, 0; return res
    q, vt, X = st
    basis = None
    if relay in ("ZF", "MRC/MRT"):
        Xfix = relay_structure(sys, relay) / np.sqrt(sys.Pmax_n)
        X = sys.scale_X(Xfix, q, sys.psi(vt))
    elif relay == "LMMSE":
        basis = relay_structure(sys, "LMMSE")
        X = sys.scale_X(basis[0] @ basis[1].conj().T, q, sys.psi(vt))
    a = np.sqrt(q)
    hist, fails = [], 0
    obj_prev = None
    for it in range(1, max_iter + 1):
        psi = sys.psi(vt)
        U = sys.mvdr(a ** 2, psi)                                          # step 2, eq. (10)
        if kappa or relay != "opt": X = sys.scale_X(X, a ** 2, psi)       # P_t = P^max at the current point
        r = step_alpha_v(sys, X, U, a, vt, eta, kappa=kappa)              # step 3, problem (17)
        if r is None: fails += 1
        else: a, vt = r
        psi = sys.psi(vt)
        if kappa or relay != "opt": X = sys.scale_X(X, a ** 2, psi)       # W <- kappa X (fixed relays: rho in [Hu])
        if relay == "opt":
            Xn = step_W(sys, X, a ** 2, psi)                               # step 4, problem (20)
            if Xn is None: fails += 1
            else: X = Xn
            X = X / max(1.0, np.sqrt(sys.relay_power(X, a ** 2, psi)))
        elif relay == "LMMSE":
            Xn = step_W(sys, X, a ** 2, psi, basis=basis)
            if Xn is None: fails += 1
            else: X = Xn
            X = X / max(1.0, np.sqrt(sys.relay_power(X, a ** 2, psi)))
        else:
            X = sys.scale_X(Xfix, a ** 2, psi)
        g, _ = sys.comm_sinr(X, a ** 2, psi)
        obj = np.sum(np.log(1 + g))
        hist.append(0.5 * obj / np.log(2))
        if log: log(it, hist[-1])
        if obj_prev is not None and abs(obj - obj_prev) < tol: break
        obj_prev = obj
    psi = sys.psi(vt); q = a ** 2
    U = sys.mvdr(q, psi)
    res.q, res.vt, res.X, res.U, res.psi = q, vt, X, U, psi
    res.sar = sys.sar(X, q, psi)
    res.min_sinr_db = 10 * np.log10(sys.sense_sinr(U, q, psi).min())
    res.Pt = sys.relay_power(X, q, psi)
    res.iters, res.hist, res.fails, res.time = it, hist, fails, time.perf_counter() - t0
    res.feasible = bool(res.min_sinr_db >= eta_db - 0.01 and res.Pt <= 1 + 1e-6)
    return res
