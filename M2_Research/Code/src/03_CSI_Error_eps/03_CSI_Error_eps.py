"""Experiment 03: CSI Error eps

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with RUNS = 20: about 3-5 min.
Options:  python run_experiment.py --runs=100     (paper accuracy, about 5x longer)
          python run_experiment.py --paper        (no simulation: plot the paper's saved 100-channel results)
Output:   results/sweep_eps.csv, results/fig_eps.png/.pdf, and the tables printed below.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import sweep_experiment

RUNS = 20                 # channels per x-value (the paper uses 100; same channels, seed = 1000 + run)
USE_PAPER_DATA = False    # True = just plot the paper's saved results

WHAT = """Sweeps the CSI-error variance ratio eps = 0 ... 0.1.
Shows: at eps = 0 all designs protect the radar; as soon as the CSI is imperfect the SINR-floor
designs lose protection, while the proposed design keeps 100 % and pays only a small rate cost."""

if __name__ == '__main__':
    sweep_experiment('eps', HERE, RUNS, USE_PAPER_DATA, WHAT)
