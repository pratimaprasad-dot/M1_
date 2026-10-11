"""Experiment 07: Receive Antennas Nr

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with RUNS = 20: about 6-10 min.
Options:  python run_experiment.py --runs=100     (paper accuracy, about 5x longer)
          python run_experiment.py --paper        (no simulation: plot the paper's saved 100-channel results)
Output:   results/sweep_Nr.csv, results/fig_Nr.png/.pdf, and the tables printed below.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import sweep_experiment

RUNS = 20                 # channels per x-value (the paper uses 100; same channels, seed = 1000 + run)
USE_PAPER_DATA = False    # True = just plot the paper's saved results

WHAT = """Sweeps the number of DBS receive antennas Nr = 6 ... 16 (Hu Fig. 7 analogue).
Shows: more receive antennas give more interference suppression; the gap to Hu's design closes at
Nr = 16, where the SINR floor already guarantees detection."""

if __name__ == '__main__':
    sweep_experiment('Nr', HERE, RUNS, USE_PAPER_DATA, WHAT)
