"""Experiment 01: Default comparison (the paper's main result table)

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with RUNS = 20: about 1 min.
Options:  python run_experiment.py --runs=100     (paper accuracy)
          python run_experiment.py --paper        (no simulation: use the paper's saved 100-channel results)
Output:   results/sweep_default.csv, results/paired_comparison.csv, results/fig_default_comparison.png/.pdf
"""
import sys, csv, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import banner, run_sweep, paired_table, plot_bars, open_file

RUNS = 20                 # channels (the paper uses 100; same channels, seed = 1000 + run)
USE_PAPER_DATA = False    # True = use the paper's saved results

WHAT = """Default setting: Nt = Nr = 8, M = 4 secondary pairs, K = 4 targets, Pc = Pmax = 20 dBm,
CSI error eps = 0.01, clutter shape k = 0.7, beta = 0.9, P_FA = 1e-3.
All six schemes are designed on the SAME channels and judged by the true detection law.
Shows: the proposed design protects every target on every channel at almost the same rate as Hu's
non-robust design (which protects only ~10 % of channels), and ~2 bps/Hz above Hu's robust design."""

if __name__ == '__main__':
    use_paper = USE_PAPER_DATA or '--paper' in sys.argv
    runs = next((int(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--runs=')), RUNS)
    banner(f'{HERE.name}  ({"paper data" if use_paper else f"{runs} channels"})')
    print(WHAT, '\n', flush=True)
    t0 = time.time()
    rows = run_sweep('default', HERE / 'results', runs, use_paper)
    out = paired_table(rows, 0.9)
    with (HERE / 'results' / 'paired_comparison.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(out[0]))
        w.writeheader()
        w.writerows(out)
    png = plot_bars(rows, HERE / 'results')
    banner(f'Finished in {(time.time() - t0) / 60:.1f} min. Results in {HERE / "results"}')
    open_file(png)
