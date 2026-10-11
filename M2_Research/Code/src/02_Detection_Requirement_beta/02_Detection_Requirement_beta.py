"""Experiment 02: Detection Requirement beta

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with RUNS = 20: about 3-5 min.
Options:  python run_experiment.py --runs=100     (paper accuracy, about 5x longer)
          python run_experiment.py --paper        (no simulation: plot the paper's saved 100-channel results)
Output:   results/sweep_beta.csv, results/fig_beta.png/.pdf, and the tables printed below.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import sweep_experiment

RUNS = 20                 # channels per x-value (the paper uses 100; same channels, seed = 1000 + run)
USE_PAPER_DATA = False    # True = just plot the paper's saved results

WHAT = """Sweeps the required detection probability beta = 0.5 ... 0.95 (paper Fig. beta, Hu Fig. 9 analogue).
Shows: the proposed design protects every target at every beta; Hu's non-robust designs reach a
similar rate but leave most channels unprotected; Hu's robust design is protected but loses rate."""

if __name__ == '__main__':
    sweep_experiment('beta', HERE, RUNS, USE_PAPER_DATA, WHAT)
