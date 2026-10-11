"""Experiment 10: Convergence of the design algorithms (Hu Fig. 2 analogue)

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with RUNS = 10: about 1-2 min.
Options:  python run_experiment.py --runs=30      (paper accuracy)
Output:   results/convergence.json, results/fig_convergence.png/.pdf and the table printed below.
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
from matplotlib.lines import Line2D

RUNS = 10                       # channels per beta (the paper uses 30)
BETAS = [0.8, 0.9, 0.95]
SCH = {'Proposed': ('#2a78d6', 'o', 'Proposed'),
       'Hu Alg.1, eta(beta)': ('#1baf7a', '^', 'Hu Alg. 1 (non-robust)'),
       'Hu robust (worst-case), eta(beta)': ('#e87ba4', 'D', 'Hu robust')}
LS = {0.8: '-', 0.9: '--', 0.95: ':'}

WHAT = """Runs each design algorithm for up to 40 outer iterations and records the design-model sum rate
after every iteration, averaged over channels, for beta = 0.8, 0.9 and 0.95.
Shows: all three algorithms converge (curves flatten) within about 10-20 iterations."""


def job(a):
    warnings.filterwarnings('ignore')
    from mimo_core import Cfg, Scenario
    from schemes import Design
    beta, run = a
    c = Cfg(beta=beta, maxit=40, tol=1e-6)
    sc = Scenario(c, 1000 + run)
    return {s: Design(sc, s, c.eps, 1000 + run).run().history for s in SCH}


if __name__ == '__main__':
    runs = next((int(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--runs=')), RUNS)
    banner(f'{HERE.name}  ({runs} channels per beta)')
    print(WHAT, '\n', flush=True)
    t0 = time.time()
    tasks = [(b, r) for b in BETAS for r in range(runs)]
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(job, tasks))
    hist = {f'{s}|{b}': [] for s in SCH for b in BETAS}
    for (b, _), h in zip(tasks, res):
        for s in SCH:
            v = h[s] + [h[s][-1]] * (41 - len(h[s]))      # hold the converged value
            hist[f'{s}|{b}'].append(v[:41])
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    (out / 'convergence.json').write_text(json.dumps(hist))

    print(f'{"Scheme, beta":42s}{"iter 1":>8s}{"iter 5":>8s}{"iter 10":>9s}{"iter 20":>9s}{"iter 40":>9s}'
          f'{"  iterations to reach 99.9% of final":>38s}')
    print('-' * 123)
    for s in SCH:
        for b in BETAS:
            m = np.mean(hist[f'{s}|{b}'], 0)
            n99 = int(np.argmax(m >= m[-1] - 1e-3 * abs(m[-1]))) + 1
            print(f'{SCH[s][2] + f", beta={b}":42s}{m[0]:8.2f}{m[4]:8.2f}{m[9]:9.2f}{m[19]:9.2f}{m[40]:9.2f}{n99:24d}')

    plt.rcParams.update({'figure.constrained_layout.use': False})
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for s, (col, mk, _) in SCH.items():
        for b, ls in LS.items():
            m = np.mean(hist[f'{s}|{b}'], 0)
            ax.plot(np.arange(1, len(m) + 1), m, color=col, marker=mk, markevery=5, ms=4, mfc='white', ls=ls, lw=1.2)
    ax.set(xlabel='Number of iterations', ylabel='Design-model SAR (bps/Hz)')
    h1 = [Line2D([], [], color=c, marker=mk, mfc='white', ms=4, lw=1.2, label=lab) for c, mk, lab in SCH.values()]
    h2 = [Line2D([], [], color='#444444', ls=ls, lw=1.2, label=rf'$\beta$={b}') for b, ls in LS.items()]
    fig.legend(handles=h1, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=6.5, frameon=False)
    fig.legend(handles=h2, loc='lower center', bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=6.5, frameon=False)
    fig.subplots_adjust(bottom=0.27, left=0.15, right=0.97, top=0.97)
    png = save(fig, out, 'fig_convergence')
    banner(f'Finished in {(time.time() - t0) / 60:.1f} min. Results in {out}')
    open_file(png)
