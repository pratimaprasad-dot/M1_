"""Experiment 08: Transmit Antennas Nt Relays

Run this file (VS Code Run button, or:  python run_experiment.py).  Runtime with RUNS = 20: about 6-10 min.
Options:  python run_experiment.py --runs=100     (paper accuracy, about 5x longer)
          python run_experiment.py --paper        (no simulation: plot the paper's saved 100-channel results)
Output:   results/sweep_Nt.csv, results/fig_Nt.png/.pdf, and the tables printed below.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from exp_tools import sweep_experiment

RUNS = 20                 # channels per x-value (the paper uses 100; same channels, seed = 1000 + run)
USE_PAPER_DATA = False    # True = just plot the paper's saved results

WHAT = """Sweeps the number of DBS transmit antennas Nt = 4 ... 12 (Nr = 12), comparing relay designs:
the optimized (WMMSE) relay of the proposed design against ZF, MRC/MRT and LMMSE relays (Hu Fig. 8 analogue).
Shows: the optimized relay gives the highest protected rate at every Nt."""

if __name__ == '__main__':
    sweep_experiment('Nt', HERE, RUNS, USE_PAPER_DATA, WHAT)
