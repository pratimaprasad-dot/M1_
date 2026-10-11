"""Experiment 09: Rate-detection trade-off (Hu Fig. 6 analogue)

Run this file (VS Code Run button, or:  python run_experiment.py).
It reuses the results of experiment 02 if they exist (instant); otherwise it runs the beta sweep
(about 3-5 min with RUNS = 20).
Options:  python run_experiment.py --runs=100     (paper accuracy)
          python run_experiment.py --paper        (no simulation: use the paper's saved 100-channel results)
Output:   results/fig_tradeoff.png/.pdf and the table printed below.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import banner, run_sweep, group, plot_tradeoff, open_file, ORDER

RUNS = 20
USE_PAPER_DATA = False

WHAT = """Each point is one required beta (0.5 ... 0.95): x = the worst-target P_D actually achieved
(averaged over channels), y = the average sum rate. A design is good when it reaches the far right
(high detection) without dropping (rate).
Shows: the proposed curve moves right as beta grows and achieves what it promises; Hu's non-robust
designs stay to the left (they promise beta but achieve less); Hu's robust design is far right but low."""

if __name__ == '__main__':
    use_paper = USE_PAPER_DATA or '--paper' in sys.argv
    runs = next((int(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--runs=')), RUNS)
    banner(f'{HERE.name}')
    print(WHAT, '\n', flush=True)
    t0 = time.time()
    prev = HERE.parent / '02_Detection_Requirement_beta' / 'results' / 'sweep_beta.csv'
    if not use_paper and prev.exists() and '--runs=' not in ' '.join(sys.argv):
        rows = list(csv.DictReader(prev.open()))
        print(f'Reusing experiment 02 results: {prev}')
    else:
        rows = run_sweep('beta', HERE / 'results', runs, use_paper)
    g = group(rows)
    xs = sorted({x for (_, x) in g})
    print('\nEach cell: achieved worst-target P_D / average sum rate (bps/Hz)')
    print(f'{"Scheme":38s}' + ''.join(f'{"beta = " + format(x, "g"):>17s}' for x in xs))
    print('-' * (38 + 17 * len(xs)))
    for s in ORDER:
        if not any((s, x) in g for x in xs):
            continue
        cells = []
        for x in xs:
            rr = list(g[(s, x)].values())
            cells.append(f'{np.mean([float(r["PD_min"]) for r in rr]):10.3f} /{np.mean([float(r["SAR"]) for r in rr]):5.2f}')
        print(f'{s:38s}' + ''.join(cells))
    png = plot_tradeoff(rows, HERE / 'results')
    banner(f'Finished in {(time.time() - t0) / 60:.1f} min. Results in {HERE / "results"}')
    open_file(png)
