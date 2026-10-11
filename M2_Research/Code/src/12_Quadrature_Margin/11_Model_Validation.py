"""Experiment 12: Quadrature accuracy and the design safety margin

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with SEEDS = 20: about 5-15 min.
Options:  python run_experiment.py --seeds=60      (paper accuracy, about 3x longer)
Output:   results/quadrature_margin.json, results/fig_quadrature_margin.png/.pdf and the table printed below.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import banner, save, open_file, WORKERS
import matplotlib.pyplot as plt

SEEDS = 20                                        # random channels (the paper uses 60)
SETTINGS = [(1024, 128), (4096, 256), (8192, 512)]  # (CSI-error points, grouping) used at design time
MARGIN = 0.003                                    # P_D margin the design adds on top of beta

WHAT = """The design evaluates P_D with a finite number of quadrature points (fast but approximate).
This experiment compares that design-time P_D with a very dense reference (32 texture nodes x 32768 points)
over many channels, CSI-error levels and clutter shapes, at the power levels where the constraint binds.
Shows: with 8192 points the largest over-estimate of P_D is about 0.001, so the 0.003 margin the design
adds is enough to guarantee P_D >= beta under the true law."""


def job(seed):
    warnings.filterwarnings('ignore')
    from mimo_core import Cfg, Scenario, mvdr, sensing_terms, DetLaw
    out = []
    for eps, shape in [(0.01, 0.7), (0.03, 0.7), (0.01, 0.3), (0.1, 0.7)]:
        c = Cfg(eps=eps, shape=shape)
        sc = Scenario(c, seed)
        rng = np.random.default_rng(seed)
        for _ in range(2):
            p = sc.Pc * rng.uniform(0.05, 1, c.M)
            U = mvdr(sc, p, eps)
            mu2, I, C, sh = sensing_terms(sc, U, eps)
            for k in range(c.K):
                ref = DetLaw(sc, mu2[k], I[k], C[k], sh, eps, 32, 32768, 999).pd(p, c.alpha)[0]
                if not 0.3 < ref < 0.995:
                    continue
                row = dict(eps=eps, shape=shape, ref=ref)
                for nx, ng in SETTINGS:
                    row[f'err_{nx}'] = DetLaw(sc, mu2[k], I[k], C[k], sh, eps, 16, nx, seed * 31 + k,
                                              n_group=ng).pd(p, c.alpha)[0] - ref
                out.append(row)
    return out


if __name__ == '__main__':
    seeds = next((int(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--seeds=')), SEEDS)
    banner(f'{HERE.name}  ({seeds} channels)')
    print(WHAT, '\n', flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(min(WORKERS, 4)) as ex:      # memory-heavy: at most 4 processes
        for i, rr in enumerate(ex.map(job, range(3000, 3000 + seeds)), 1):
            rows += rr
            print(f'  channel {i}/{seeds}   elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    summ = {}
    print(f'\n{"Design points":>14s}{"cases":>8s}{"largest over-estimate":>24s}{"99th pct |error|":>19s}'
          f'{"mean |error|":>15s}{"covered by margin " + str(MARGIN):>26s}')
    print('-' * 106)
    for nx, _ in SETTINGS:
        e = np.array([r[f'err_{nx}'] for r in rows])
        summ[nx] = dict(n=len(e), max_over=float(e.max()), p99_abs=float(np.percentile(abs(e), 99)),
                        mean_abs=float(abs(e).mean()))
        ok = 'YES' if e.max() <= MARGIN else 'NO'
        print(f'{nx:14d}{len(e):8d}{e.max():24.4f}{summ[nx]["p99_abs"]:19.4f}{summ[nx]["mean_abs"]:15.5f}{ok:>26s}')
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    (out / 'quadrature_margin.json').write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    data = [np.array([r[f'err_{nx}'] for r in rows]) for nx, _ in SETTINGS]
    ax.boxplot(data, whis=(0, 100))
    ax.set_xticks(range(1, len(SETTINGS) + 1), [str(nx) for nx, _ in SETTINGS])
    ax.axhline(MARGIN, color='#d62728', ls='--', lw=1, label=f'design margin {MARGIN}')
    ax.axhline(0, color='#444444', lw=.6)
    ax.set(xlabel='CSI-error quadrature points at design time', ylabel=r'$P_D$ error (design $-$ reference)')
    ax.legend(fontsize=6.5)
    png = save(fig, out, 'fig_quadrature_margin')
    banner(f'Finished in {(time.time() - t0) / 60:.1f} min. Results in {out}')
    open_file(png)
