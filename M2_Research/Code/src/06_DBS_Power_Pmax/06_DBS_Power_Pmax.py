"""Experiment 06: DBS Power Pmax

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with RUNS = 20: about 3-5 min.
Options:  python run_experiment.py --runs=100     (paper accuracy, about 5x longer)
          python run_experiment.py --paper        (no simulation: plot the paper's saved 100-channel results)
Output:   results/sweep_Pmax.csv, results/fig_Pmax.png/.pdf, and the tables printed below.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import sweep_experiment

RUNS = 20                 # channels per x-value (the paper uses 100; same channels, seed = 1000 + run)
USE_PAPER_DATA = False    # True = just plot the paper's saved results

WHAT = """Sweeps the DBS relay power budget Pmax = 15 ... 30 dBm (Hu Fig. 5 analogue).
Shows: more relay power raises every design's rate; only the proposed and the worst-case designs
keep all targets protected, and the proposed one does so at a higher rate."""

if __name__ == '__main__':
    sweep_experiment('Pmax', HERE, RUNS, USE_PAPER_DATA, WHAT)
