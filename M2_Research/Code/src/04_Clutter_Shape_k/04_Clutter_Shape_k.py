"""Experiment 04: Clutter Shape k

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with RUNS = 20: about 4-6 min.
Options:  python run_experiment.py --runs=100     (paper accuracy, about 5x longer)
          python run_experiment.py --paper        (no simulation: plot the paper's saved 100-channel results)
Output:   results/sweep_shape.csv, results/fig_shape.png/.pdf, and the tables printed below.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import sweep_experiment

RUNS = 20                 # channels per x-value (the paper uses 100; same channels, seed = 1000 + run)
USE_PAPER_DATA = False    # True = just plot the paper's saved results

WHAT = """Sweeps the clutter texture shape k = 0.3 (very spiky) ... 50 (almost Gaussian) at eps = 0.01.
Shows: the proposed design stays protected for every clutter shape. At eps = 0.01 the CSI leakage
dominates, so the baselines are unprotected for all k (the clutter effect alone shows at eps = 0)."""

if __name__ == '__main__':
    sweep_experiment('shape', HERE, RUNS, USE_PAPER_DATA, WHAT)
