"""Experiment 05: Secondary Power Pc

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with RUNS = 20: about 6-10 min.
Options:  python run_experiment.py --runs=100     (paper accuracy, about 5x longer)
          python run_experiment.py --paper        (no simulation: plot the paper's saved 100-channel results)
Output:   results/sweep_Pc.csv, results/fig_Pc.png/.pdf, and the tables printed below.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import sweep_experiment

RUNS = 20                 # channels per x-value (the paper uses 100; same channels, seed = 1000 + run)
USE_PAPER_DATA = False    # True = just plot the paper's saved results

WHAT = """Sweeps the secondary-transmitter power budget Pc = 0 ... 25 dBm (Hu Fig. 4 analogue),
including the structured ZF, MRC/MRT and LMMSE relays.
Shows: the proposed rate grows with Pc while every target stays protected; the structured relays
are protected but give much lower rate."""

if __name__ == '__main__':
    sweep_experiment('Pc', HERE, RUNS, USE_PAPER_DATA, WHAT)
