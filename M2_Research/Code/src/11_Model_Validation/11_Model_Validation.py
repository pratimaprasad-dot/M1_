"""Experiment 11: Validation of the detection and rate models against brute-force simulation

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime: about 3-6 min.
Output:   results/validation.json and the PASS/FAIL table printed at the end.

Independent checks of every model reduction used by the design:
1. Detection: semi-analytic P_FA/P_D vs brute-force simulation of the full Nr-antenna signal
   (true channels, CSI errors, Gamma texture, noise), 400,000 trials per target.
2. Rate/relay: closed-form error-averaged terms vs Monte-Carlo averages over CSI errors.
3. Worst-case relay power (trust-region solution, used by Hu robust) vs random search on the ball.
4. The WMMSE relay step never decreases the design sum rate.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # before numpy: one BLAS thread per process
from pathlib import Path
import json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
import time
t_start = time.time()
print('Running 4 validation checks (about 3-6 min)...', flush=True)
from mimo_core import Cfg, Scenario, mvdr, sensing_terms, DetLaw, cn, relay_cov, rate_terms, wmmse_W, sar
from schemes import trs_max, Design

OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
res = {}

# ---------------------------------------------------------------- 1 detection
def brute_detection(c, seed, p, n=400_000, chunk=50_000):
    sc = Scenario(c, seed)
    rng = np.random.default_rng(10_000 + seed)
    U = mvdr(sc, p, c.eps)
    mu2, I, C, sh = sensing_terms(sc, U, c.eps)
    out = []
    for k in range(c.K):
        lw = DetLaw(sc, mu2[k], I[k], C[k], sh, c.eps, 64, 8192, 3 + k)
        pd_a, t = lw.pd(p, c.alpha)
        u = U[:, k]
        mu = u.conj() @ sc.E[:, k]
        ph = np.exp(-1j * np.angle(mu))
        h0 = h1 = 0
        for _ in range(n // chunk):
            # per CPI: CSI errors, texture, echo error; per snapshot (L=1) symbols/noise
            dH = cn(rng, chunk, c.Nr, c.M) * np.sqrt(sh)[None, None, :]
            Ht = sc.H[None] + dH
            tau = rng.gamma(c.shape, 1 / c.shape, chunk)
            x = cn(rng, chunk, c.M) * np.sqrt(p)[None, :]
            y = np.einsum('nrm,nm->nr', Ht, x)
            y += np.sqrt(sc.sn) * cn(rng, chunk, c.Nr) + np.sqrt(tau * sc.sc)[:, None] * cn(rng, chunk, c.Nr)
            for i in range(c.K):        # other targets: random-phase, error-perturbed echoes
                if i != k:
                    y += sc.E[None, :, i] * ((1 + np.sqrt(c.eps) * cn(rng, chunk)) * np.exp(2j * np.pi * rng.random(chunk)))[:, None]
            T0 = np.real(y @ u.conj() * ph)
            a = 1 + np.sqrt(c.eps) * cn(rng, chunk)
            T1 = np.real((y + sc.E[None, :, k] * a[:, None]) @ u.conj() * ph)
            h0 += int(np.sum(T0 > t)); h1 += int(np.sum(T1 > t))
        out.append(dict(target=k, PFA_model=c.alpha, PFA_sim=h0 / n, PD_model=pd_a, PD_sim=h1 / n,
                        PD_se=float(np.sqrt(pd_a * (1 - pd_a) / n)), PFA_se=float(np.sqrt(c.alpha * (1 - c.alpha) / n))))
    return out

det = []
for eps, shape, seed, p in [(0.01, .7, 1, [.035, .032, .035, .047]), (0.05, .7, 3, [.01, .03, .02, .02]),
                            (0.0, .3, 2, [.1, .1, .1, 0.]), (0.03, 2., 4, [.02, .02, .02, .02])]:
    c = Cfg(eps=eps, shape=shape)
    for r in brute_detection(c, seed, np.array(p)):
        r.update(eps=eps, shape=shape, seed=seed)
        det.append(r)
res['detection'] = det
zs = [abs(r['PD_sim'] - r['PD_model']) / max(r['PD_se'], 1e-12) for r in det]
zf = [abs(r['PFA_sim'] - r['PFA_model']) / r['PFA_se'] for r in det]
res['detection_max_abs_z'] = dict(PD=float(max(zs)), PFA=float(max(zf)))
print('detection max |z|: PD %.2f  PFA %.2f' % (max(zs), max(zf)))

# ------------------------------------------------------------ 2 rate terms
c = Cfg(eps=0.03)
sc = Scenario(c, 5)
d = Design(sc, 'Proposed', c.eps, 5).run()
W, p = d.W, d.p
S, C, D, a, b = rate_terms(sc, W, c.eps)
rng = np.random.default_rng(77)
sh, sg = sc.err(c.eps)
n = 20_000
tot = np.zeros(c.M); sig = np.zeros(c.M, complex); relay = 0.
for _ in range(n // 1000):
    dH = cn(rng, 1000, c.Nr, c.M) * np.sqrt(sh)[None, None, :]
    dG = cn(rng, 1000, c.M, c.Nt) * np.sqrt(sg)[None, :, None]
    ea = np.sqrt(c.eps) * cn(rng, 1000, c.K)
    Ht = sc.H[None] + dH; Gt = sc.G[None] + dG
    Et = sc.E[None] * (1 + ea)[:, None, :]
    GW = np.einsum('nmt,tr->nmr', Gt, W)
    hp = np.abs(np.einsum('nmr,nrj->nmj', GW, Ht)) ** 2 * p[None, None, :]
    ep = np.abs(np.einsum('nmr,nrk->nmk', GW, Et)) ** 2
    tot += (hp.sum(2) + ep.sum(2) + np.sum(abs(GW) ** 2, 2) + 1).sum(0)
    sig += np.einsum('nmr,nrm->nm', GW, Ht).sum(0)
    WH = np.einsum('tr,nrj->ntj', W, Ht); WE = np.einsum('tr,nrk->ntk', W, Et)
    relay += float((np.sum(abs(WH) ** 2 * p[None, None, :], (1, 2)) + np.sum(abs(WE) ** 2, (1, 2)) + np.sum(abs(W) ** 2)).sum())
tot /= n; sig /= n; relay /= n
closed_tot = C @ p + D
res['rate_terms'] = dict(total_rel_err=float(np.max(abs(tot - closed_tot) / closed_tot)),
                         mean_gain_rel_err=float(np.max(abs(abs(sig) ** 2 - S) / S)),
                         relay_rel_err=float(abs(relay - (a @ p + b)) / (a @ p + b)), draws=n)
print('rate terms', res['rate_terms'])

# --------------------------------------------------------- 3 worst-case relay
rng = np.random.default_rng(9)
gaps = []
for m in range(c.M):
    h = sc.H[:, m]; delta = 0.3 * np.linalg.norm(h)
    best = trs_max(W, h, delta)
    dirs = cn(rng, 200_000, c.Nr); dirs *= delta / np.linalg.norm(dirs, axis=1, keepdims=True)
    samp = np.max(np.sum(abs((h[None] + dirs) @ W.T) ** 2, 1))
    gaps.append(dict(trs=best, random_max=float(samp), trs_minus_random_rel=float((best - samp) / best)))
res['trust_region'] = gaps
print('TRS >= random search:', all(g['trs_minus_random_rel'] >= -1e-9 for g in gaps))

# ----------------------------------------------------------- 4 WMMSE ascent
inc = []
for seed in range(6):
    sc = Scenario(Cfg(), seed)
    p = np.full(4, sc.Pc * .5)
    from mimo_core import w_structured
    W = w_structured(sc, 'MRC/MRT'); W *= np.sqrt(sc.Pmax / relay_cov(sc, p, 0).trace().real / np.sum(abs(W) ** 2))
    vals = [sar(sc, W, p, 0.)]
    for _ in range(15):
        W = wmmse_W(sc, W, p, 0., inner=1); vals.append(sar(sc, W, p, 0.))
    inc.append(float(np.min(np.diff(vals))))
res['wmmse_min_increment'] = inc
print('WMMSE min increment', min(inc))
(OUT / 'validation.json').write_text(json.dumps(res, indent=2))

# ----------------------------------------------------------- summary
z = res['detection_max_abs_z']
rt = res['rate_terms']
checks = [
    ('1. Detection P_D: model vs simulation', f"max |z| = {z['PD']:.2f}", z['PD'] < 4),
    ('1. Detection P_FA: model vs simulation', f"max |z| = {z['PFA']:.2f}", z['PFA'] < 4),
    ('2. Rate terms: closed form vs Monte Carlo', f"max rel. error = {max(rt['total_rel_err'], rt['mean_gain_rel_err']):.4f}",
     max(rt['total_rel_err'], rt['mean_gain_rel_err']) < 0.02),
    ('2. Relay power: closed form vs Monte Carlo', f"rel. error = {rt['relay_rel_err']:.4f}", rt['relay_rel_err'] < 0.02),
    ('3. Worst-case relay power >= random search', f"min margin = {min(g['trs_minus_random_rel'] for g in gaps):+.2e}",
     all(g['trs_minus_random_rel'] >= -1e-9 for g in gaps)),
    ('4. WMMSE step never lowers the sum rate', f"min increment = {min(inc):+.2e}", min(inc) >= -1e-9),
]
print('\nDetection check, every target and scenario (400,000 trials each):')
print(f"{'eps':>6s}{'k':>6s}{'target':>8s}{'P_FA model':>12s}{'P_FA sim':>10s}{'P_D model':>11s}{'P_D sim':>9s}")
for r in res['detection']:
    print(f"{r['eps']:6g}{r['shape']:6g}{r['target']:8d}{r['PFA_model']:12.4f}{r['PFA_sim']:10.4f}{r['PD_model']:11.4f}{r['PD_sim']:9.4f}")
print('\n' + '=' * 92)
print(f"{'Check':48s}{'Result':28s}{'Status':>8s}")
print('-' * 92)
for name, val, ok in checks:
    print(f"{name:48s}{val:28s}{'PASS' if ok else 'FAIL':>8s}")
print('=' * 92)
print(f"|z| < 4 means model and simulation agree within simulation noise. Finished in {(time.time() - t_start) / 60:.1f} min.")
print('Results:', OUT / 'validation.json')
