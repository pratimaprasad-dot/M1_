"""
Layers B and C in the COMMON environment (the manuscript's model, v3_engine.Scn):
  same channel realisations (seeds), same budgets (P_c = P^max = P_s = 20 dBm), same arrays (Nt = Nr = 8, M = K = 4),
  same FIXED radar beam for every method, same evaluation (common_eval.Evaluator: literal signal model, one detector,
  oracle P_FA calibration, ergodic rate).

Methods (key field 'method', 'param'):
  prop        manuscript's proposed design (v3_engine.run_design 'prop'), param = beta (required P_D)
  hu_nom      [Hu] Algorithm 1 on the estimated channels (relay-gain-normalised variant K), param = eta_min (dB)
  hu_rob      [Hu] Algorithm 3 (GLDT-CS, variant K), bounded balls on h AND g, radius = 95 % per-link containment of the
              Gaussian error (delta_m = sigma_h,m sqrt(F^-1_Gamma(Nr,1)(0.95)), likewise for g), param = eta_min (dB)
  adapt_rob   the manuscript's earlier adapted worst-case baseline (v3_engine.run_design 'robust'), param = eta (dB)
  cell        Layer C cells (filter x constraint), param = cell name (see run_cell)
  nlp         alternative solver for the proposed formulation (direct SLSQP on p with P_D constraints), param = beta
Every design is saved to data/designs/<tag>.npz so that it can be re-evaluated (other clutter shapes, P_FA checks).

Usage: python b_campaign.py <experiment>     experiments: main, tune, tradeoff, cells, scale, nlp
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "1")   # before numpy: one BLAS thread per worker

import os, sys, json, time, warnings
import numpy as np
import runner

HERE = os.path.dirname(os.path.abspath(__file__))
DES = os.path.join(runner.DATA, "designs")
os.makedirs(DES, exist_ok=True)

EPS_GRID = [0.0, 5.48e-4, 1.52e-3, 3e-3, 1e-2, 3e-2]          # 5.48e-4 / 1.52e-3 = [Hu]'s normalised radii 0.03 / 0.05
TEST = list(range(1000, 1060))
DEV = list(range(2000, 2020))
NU0 = 0.7

_G = {}


def G():
    if not _G:
        warnings.filterwarnings("ignore")
        import v3_np, common_eval, hu_core, hu_robust
        _G.update(E=v3_np.E, np_=v3_np, ce=common_eval, hc=hu_core, hr=hu_robust, laws={}, evs={})
    return _G


def law(nu, M=4):
    g = G()
    if (nu, M) not in g["laws"]:
        L = g["np_"].Law(nu if np.isfinite(nu) else 0.7, n_tau=24, n_e=2 ** 12, seed=1, M=M)
        if not np.isfinite(nu): L.tau, L.wt = np.array([1.0]), np.array([1.0])          # Gaussian clutter (deterministic texture)
        g["laws"][(nu, M)] = L
    return g["laws"][(nu, M)]


def evaluator(nu, M=4):
    g = G()
    if (nu, M) not in g["evs"]: g["evs"][(nu, M)] = g["ce"].Evaluator(nu=nu, M=M)
    return g["evs"][(nu, M)]


def scn(seed, Nr=8, Nt=8, M=4, K=4):
    E = G()["E"]
    sc = E.Scn(seed, Nt=Nt, Nr=Nr, M=M, K=K); sc.seed = seed
    return sc


def radii(sc, eps, c=0.95):
    E = G()["E"]; from scipy import stats
    return (np.sqrt(E.sigma_h2(sc, eps) * stats.gamma.ppf(c, sc.Nr)), np.sqrt(E.sigma_g2(sc, eps) * stats.gamma.ppf(c, sc.Nt)))


# ---------------------------------------------------------------------- Layer C: cells with common algorithm
def run_cell(sc, eps, cell, max_iter=200):
    """common algorithm (v3 Algorithm 1 structure: filter step, constraint rows, SLSQP power step, WMMSE relay step,
    rate model at the true eps) with ONE component changed:
       F0 / F1 : receive filter nominal (eps = 0 in eq. (13)) / error-aware (true eps)
       S       : SINR floor eta(beta) = 9.80 dB on the nominal rows (Gaussian-equivalent of P_D >= 0.9)
       B       : SINR floor eta(beta) with bounded-ball rows (|u^H h| + delta)^2 (the worst-case protection of [Hu])
       P       : P_D certificate, compound-Gaussian clutter (nu = 0.7) and Gaussian CSI-error law  (proposed)
       PG      : P_D certificate with GAUSSIAN clutter and the Gaussian CSI-error law (detection-based, clutter-unaware)
    """
    g = G(); E = g["E"]
    filt, cons = cell.split("-")
    fe = 0.0 if filt == "F0" else eps
    eta = E.eta_beta(0.9)
    t0 = time.perf_counter()
    cert = cons in ("P", "PG")
    L = law(NU0 if cons == "P" else np.inf, sc.M)
    delta = radii(sc, eps)[0] if cons == "B" else None
    target = E.BETA + E.MARGIN
    M, Pc = sc.M, sc.Pc; d0 = np.full(M, Pc)
    if cert:
        def okc(s):
            p = s * d0; U = E.filters(sc, p, fe); return L.pd_of(U, sc, p, eps).min() >= target
        lo, hi = 0.0, 1.0
        if okc(1.0): lo = 1.0
        else:
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                if okc(mid): lo = mid
                else: hi = mid
        p = lo * d0
    else:
        U = E.filters(sc, d0, fe); rows, rhs = E.hu_rows(sc, U, eta, delta)
        lam = min(1.0, float(np.min(rhs / np.maximum(rows @ d0, 1e-30)))); p = max(lam, 0.0) * d0
    W = E.scale_to_power(sc.ghat.conj() @ sc.hhat.conj().T, sc, p, eps)
    for _ in range(4):
        W = E.wmmse_step(W, np.maximum(p, 1e-12), sc, eps, sc.Pmax); W = E.scale_to_power(W, sc, p, eps)
    prev, it = None, 0
    for it in range(1, max_iter + 1):
        U = E.filters(sc, p, fe)
        if cert:
            pr = L.prep(U, sc, eps)
            lamk = L.max_scale(pr, np.maximum(p, 1e-30 * Pc), target, cap=2 * Pc / max(p.max(), 1e-30) if p.max() > 0 else 1.0)
            sh = E.sigma_h2(sc, eps); cbar = np.abs(U.conj().T @ sc.hhat) ** 2 + sh[None, :]
            rows, rhs = cbar, lamk * (cbar @ p)
        else:
            rows, rhs = E.hu_rows(sc, U, eta, delta)
        p_new, _, _ = E.power_step(p, W, sc, eps, rows, rhs)
        if p_new.max() <= 0: p_new = p
        p = p_new
        W = E.scale_to_power(W, sc, p, eps)
        W = E.wmmse_step(W, np.maximum(p, 1e-12), sc, eps, sc.Pmax); W = E.scale_to_power(W, sc, p, eps)
        Rm = E.model_rate(sc, W, p, eps)
        if prev is not None and abs(Rm - prev) / max(abs(prev), 1e-12) < 1e-4: break
        prev = Rm
    U = E.filters(sc, p, fe)
    if cert:
        pr = L.prep(U, sc, eps); rep = float(np.clip(L.max_scale(pr, p, target, cap=1.0).min(), 0, 1))
    else:
        rows, rhs = E.hu_rows(sc, U, eta, delta); rep = float(np.clip(np.min(rhs / np.maximum(rows @ p, 1e-30)), 0, 1))
    p = rep * p; W = E.scale_to_power(W, sc, p, eps)
    return dict(p=p, W=W, U=U, iters=it, time=time.perf_counter() - t0, status="ok", repair=rep)


# ---------------------------------------------------------------------- alternative solver for the proposed formulation
def run_nlp(sc, eps, beta, nu=NU0, max_outer=30):
    """Direct NLP: alternate (i) SLSQP over p in [0, Pc]^M maximising the model rate (kappa(p) included) subject to the
    NONLINEAR constraints PD_k(p) >= beta + margin (finite-difference Jacobian, filters recomputed at each p), and
    (ii) WMMSE relay update. No certificate rows, no bisection: separates formulation from algorithm."""
    from scipy.optimize import minimize
    g = G(); E = g["E"]; L = law(nu, sc.M); t0 = time.perf_counter()
    target = beta + E.MARGIN
    Pc = sc.Pc
    def pd(p): return L.pd_of(E.filters(sc, p, eps), sc, p, eps)
    # feasible start: largest common scaling
    lo, hi = 0.0, 1.0
    if pd(np.full(sc.M, Pc)).min() >= target: lo = 1.0
    else:
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if pd(np.full(sc.M, mid * Pc)).min() >= target: lo = mid
            else: hi = mid
    p = np.full(sc.M, lo * Pc)
    W = E.scale_to_power(sc.ghat.conj() @ sc.hhat.conj().T, sc, p, eps)
    for _ in range(4):
        W = E.wmmse_step(W, np.maximum(p, 1e-12), sc, eps, sc.Pmax); W = E.scale_to_power(W, sc, p, eps)
    prev, it, nfev = None, 0, 0
    for it in range(1, max_outer + 1):
        C, D, S, a, b = E.rate_terms(W, sc, eps)
        def f(x):
            pp = Pc * x; k2 = sc.Pmax / (a @ pp + b)
            return -E.rate_from(pp, C, D, S, k2, sc)[0]
        cons = dict(type="ineq", fun=lambda x: pd(Pc * x) - target)
        x0 = p / Pc
        r = minimize(f, x0, bounds=[(0, 1)] * sc.M, constraints=[cons], method="SLSQP", options=dict(maxiter=60, ftol=1e-8, eps=1e-6))
        nfev += r.nfev
        x = np.clip(r.x, 0, 1)
        if pd(Pc * x).min() >= target - 1e-6 and f(x) <= f(x0) + 1e-12: p = Pc * x
        W = E.scale_to_power(W, sc, p, eps)
        W = E.wmmse_step(W, np.maximum(p, 1e-12), sc, eps, sc.Pmax); W = E.scale_to_power(W, sc, p, eps)
        Rm = E.model_rate(sc, W, p, eps)
        if prev is not None and abs(Rm - prev) / max(abs(prev), 1e-12) < 1e-4: break
        prev = Rm
    U = E.filters(sc, p, eps)
    # final verification + repair (same rule as Algorithm 1)
    if pd(p).min() < target:
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if pd(mid * p).min() >= target: lo = mid
            else: hi = mid
        p = lo * p; U = E.filters(sc, p, eps); W = E.scale_to_power(W, sc, p, eps)
    return dict(p=p, W=W, U=U, iters=it, time=time.perf_counter() - t0, status="ok", nfev=int(nfev))


# ---------------------------------------------------------------------- design dispatcher
def design(sc, eps, method, param, nu=NU0):
    g = G(); E, hc, hr = g["E"], g["hc"], g["hr"]
    if method == "prop":
        L = law(nu, sc.M)
        old = E.BETA; E.BETA = float(param)
        try:
            dg = E.run_design(sc, "prop", eps, L, max_iter=200)
        finally:
            E.BETA = old
        return dict(p=dg.p, W=dg.W, U=dg.U, iters=dg.iters, time=dg.time, status="ok", repair=dg.repair)
    if method == "adapt_rob":
        dg = E.run_design(sc, "robust", eps, law(nu, sc.M), eta_floor=10 ** (float(param) / 10), max_iter=200)
        return dict(p=dg.p, W=dg.W, U=dg.U, iters=dg.iters, time=dg.time, status="ok", repair=dg.repair)
    if method == "hu_nom":
        s = hc.HuSys.from_scn(sc)
        r = hc.algorithm1(s, float(param), kappa=True, max_iter=100, tol=1e-5)
        if not r.ok: return dict(p=np.zeros(sc.M), W=np.zeros((sc.Nt, sc.Nr), complex), U=E.filters(sc, np.zeros(sc.M), 0.0), iters=0, time=r.time, status="infeasible")
        return dict(p=r.q * sc.Pc, W=s.W_of(r.X), U=r.U, iters=r.iters, time=r.time, status="ok", fails=r.fails, hist=r.hist)
    if method == "hu_rob":
        s = hc.HuSys.from_scn(sc)
        eh, eg = radii(sc, eps)
        if eps == 0: return design(sc, eps, "hu_nom", param, nu)
        R = hr.Robust(s, eh, eg, float(param), nu=1e-4)
        t0 = time.perf_counter()
        st = hc.algorithm1(s, float(param), kappa=True, max_iter=100, tol=1e-5)          # warm start: nominal Alg. 1
        start = (st.q, None, st.X, st.U) if st.ok else None
        res = R.algorithm3(start=start, max_outer=15, max_inner=40, tol=1e-4)
        if res is None or res["status"] == "infeasible_at_zero":
            return dict(p=np.zeros(sc.M), W=np.zeros((sc.Nt, sc.Nr), complex), U=E.filters(sc, np.zeros(sc.M), 0.0), iters=0,
                        time=time.perf_counter() - t0, status="infeasible")
        return dict(p=res["q"] * sc.Pc, W=s.W_of(res["X"]), U=res["U"], iters=res["inner"], outer=res["outer"], T=res["T"],
                    time=time.perf_counter() - t0, status=res["status"], wc_sar_design=float(res["wc_sar"]))
    if method == "cell":
        return run_cell(sc, eps, param)
    if method == "nlp":
        return run_nlp(sc, eps, float(param), nu)
    raise ValueError(method)


def mech(sc, eps, p, U):
    """mechanism diagnostics: expected ST interference at each radar filter output (relative to sigma_r^2)"""
    E = G()["E"]
    Un = U / np.linalg.norm(U, axis=0, keepdims=True)
    sh = E.sigma_h2(sc, eps)
    I = (np.abs(Un.conj().T @ sc.hhat) ** 2 + sh[None, :]) @ p
    return dict(radar_intf=[float(x) for x in I], radar_intf_max=float(I.max()))


def worst_case_sar(sc, eps, p, W, U):
    """[Hu]'s own criterion for any design: worst-case SAR over the 95 % balls on h and g (pessimisation oracle)"""
    g = G(); hc, hr = g["hc"], g["hr"]
    if eps == 0: return None
    s = hc.HuSys.from_scn(sc); eh, eg = radii(sc, eps)
    R = hr.Robust(s, eh, eg, 10.0)
    X = W / np.sqrt(s.Pmax_n); q = p / sc.Pc
    pw, _ = R.worst_power(X, q, s.psi())
    Xs = X / np.sqrt(max(pw, 1.0))                                                   # relay scaled to the worst-case budget
    return float(R.worst_sar(Xs, q, s.psi())[0])


def work(key):
    exp, seed, eps, method, param, nu, geo = key
    Nr, Nt, M, K = geo
    sc = scn(seed, Nr=Nr, Nt=Nt, M=M, K=K)
    dz = design(sc, eps, method, param, nu)
    tag = "_".join(str(x) for x in (exp, seed, eps, method, param, nu, Nr, Nt, M, K)).replace("/", "")
    np.savez_compressed(os.path.join(DES, tag + ".npz"), p=dz["p"], W=dz["W"], U=dz["U"])
    ev = evaluator(nu, M)(sc, dz["p"], dz["W"], dz["U"], eps)
    out = dict(ev, **mech(sc, eps, dz["p"], dz["U"]), iters=int(dz.get("iters", 0)), time=float(dz["time"]), status=dz["status"],
               design_tag=tag)
    for k in ("outer", "T", "wc_sar_design", "repair", "fails", "nfev"):
        if k in dz: out[k] = dz[k]
    if eps > 0 and geo == (8, 8, 4, 4):
        out["wc_sar"] = worst_case_sar(sc, eps, dz["p"], dz["W"], dz["U"])
    return out


def show(r):
    return f"R={r['R']:.2f} minPD={r['minpd']:.3f}{'*' if r['prot'] else ' '} t={r['time']:.0f}s {r['status']}"


GEO0 = (8, 8, 4, 4)


def keys(exp):
    K = []
    if exp == "main":
        for e in EPS_GRID:
            for s in TEST:
                K += [("main", s, e, "prop", 0.9, NU0, GEO0), ("main", s, e, "hu_nom", 10.0, NU0, GEO0),
                      ("main", s, e, "adapt_rob", 9.8017, NU0, GEO0)]
                if e > 0: K.append(("main", s, e, "hu_rob", 10.0, NU0, GEO0))
    elif exp == "tune":
        for e in EPS_GRID[1:]:
            for s in DEV:
                K += [("tune", s, e, "hu_nom", x, NU0, GEO0) for x in (10.0, 13.0, 16.0, 19.0, 22.0, 25.0)]
                if e in (1.52e-3, 1e-2):
                    K += [("tune", s, e, "hu_rob", x, NU0, GEO0) for x in (0.0, 4.0, 7.0, 10.0)]
    elif exp == "tradeoff":
        e = 1e-2
        for s in TEST[:40]:
            K += [("tradeoff", s, e, "prop", b, NU0, GEO0) for b in (0.5, 0.7, 0.8, 0.95, 0.99)]
            K += [("tradeoff", s, e, "hu_nom", x, NU0, GEO0) for x in (13.0, 16.0, 19.0, 22.0, 25.0)]
            K += [("tradeoff", s, e, "hu_rob", x, NU0, GEO0) for x in (0.0, 4.0, 7.0)]
            K += [("tradeoff", s, e, "adapt_rob", x, NU0, GEO0) for x in (4.0, 7.0, 13.0)]
    elif exp == "cells":
        for e in (1.52e-3, 1e-2):
            for s in TEST[:40]:
                K += [("cells", s, e, "cell", c, NU0, GEO0) for c in ("F0-S", "F1-S", "F0-B", "F1-B", "F0-P", "F1-P", "F1-PG")]
    elif exp == "nu":                                                      # heavy-tail sweep, fixed mean clutter power
        for nu in (0.3, 0.5, 1.0, 2.0, 5.0, 50.0):
            for s in TEST[:40]:
                K.append(("nu", s, 1e-2, "prop", 0.9, nu, GEO0))
    elif exp == "tuned":                                                   # tuned-threshold baselines on the TEST seeds
        tun = json.load(open(os.path.join(runner.DATA, "tuned_eta.json")))
        for e in EPS_GRID[1:]:
            for m in ("hu_nom", "hu_rob"):
                x = tun.get(f"{m}|{e:g}")
                if x is None: continue
                for s in TEST:
                    K.append(("tuned", s, e, m, float(x), NU0, GEO0))
    elif exp == "nlp":
        for s in TEST[:30]:
            K.append(("nlp", s, 1e-2, "nlp", 0.9, NU0, GEO0))
    elif exp == "scale":
        geos = [(6, 8, 4, 4), (12, 8, 4, 4), (16, 8, 4, 4), (12, 8, 2, 4), (12, 8, 6, 4), (12, 8, 4, 2)]
        for gq in geos:
            for s in range(3000, 3010):
                K += [("scale", s, 1e-2, m, x, NU0, gq) for m, x in (("prop", 0.9), ("hu_nom", 10.0), ("hu_rob", 10.0))]
    return K


if __name__ == "__main__":
    exp = sys.argv[1]
    runner.run(f"b_{exp}.jsonl", keys(exp), work, show=show)
