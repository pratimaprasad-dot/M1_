# Code

Detection-certified power allocation for cognitive ISAC relaying under compound-Gaussian clutter and
imperfect CSI, compared with all baselines of Hu et al. (IEEE TCCN 2026) in their MIMO setting.

## Requirements
Python 3.9+ with numpy, scipy, matplotlib:

    python -m pip install numpy scipy matplotlib

Tested with Python 3.14.6, numpy 2.5.1, scipy 1.18.0, matplotlib 3.11.0 on Windows 11 (12 cores).

## How to run
Every experiment is one folder with one file. Open `src/<experiment>/run_experiment.py` in VS Code
and press Run, or in a terminal:

    cd src/01_Default_Comparison
    python run_experiment.py              # new simulation, 20 channels (fast)
    python run_experiment.py --runs=100   # paper accuracy; identical numbers to the report
    python run_experiment.py --paper      # no simulation: show the saved 100-channel results in ../Data

Each run prints its tables, writes `results/` inside its own folder (CSV + PNG + PDF), and opens the figure.

| Folder in `src/` | Report item | What it shows | Time (default) |
|---|---|---|---|
| `01_Default_Comparison` | Table 5 | All 6 schemes on the same channels: rate, protection, paired gain, Wilcoxon p | ~1 min |
| `02_Detection_Requirement_beta` | Fig. 2 | Sweep of the required detection probability beta | ~3-5 min |
| `03_CSI_Error_eps` | Fig. 3 | Sweep of the CSI-error variance ratio eps | ~3-5 min |
| `04_Clutter_Shape_k` | Table 6 | Sweep of the clutter texture shape k | ~4-6 min |
| `05_Secondary_Power_Pc` | Fig. 4 | Sweep of the secondary power Pc, incl. ZF / MRC-MRT / LMMSE relays | ~6-10 min |
| `06_DBS_Power_Pmax` | Table 6 | Sweep of the DBS relay power Pmax | ~3-5 min |
| `07_Receive_Antennas_Nr` | Table 6 | Sweep of the receive antennas Nr | ~6-10 min |
| `08_Transmit_Antennas_Nt_Relays` | Table 6 | Sweep of the transmit antennas Nt, relay designs | ~6-10 min |
| `09_Rate_Detection_Tradeoff` | Fig. 5 (left) | Rate vs achieved detection probability (reuses 02) | instant |
| `10_Convergence` | Fig. 5 (right) | Sum rate vs iteration of each algorithm | ~1-2 min |
| `11_Model_Validation` | Table 4 | Detection and rate models vs brute-force simulation (PASS/FAIL) | ~1-5 min |
| `12_Quadrature_Margin` | Sec. 6 | Accuracy of the design-time P_D and the 0.003 safety margin | ~5-15 min |

`10_Convergence` takes `--runs=30` and `12_Quadrature_Margin` takes `--seeds=60` for paper accuracy.

## Code structure (`src/common/`)
| File | Contents |
|---|---|
| `mimo_core.py` | System model (`Cfg`, `Scenario`), MVDR sensing filters, detection law `DetLaw` (Gauss-Laguerre texture nodes x Sobol CSI-error points, threshold calibrated under the true null), rate and relay-power terms, WMMSE and LMMSE relay steps, power update `p_update` |
| `schemes.py` | `Design`: the proposed algorithm (certificate rows by bisection, power step, relay step, repair) and every baseline (Hu Alg. 1 with eta(beta) and 10 dB, Hu robust worst-case, error-ignoring P_D, ZF / MRC-MRT / LMMSE relays); `evaluate()` judges every design by the true detection law |
| `exp_tools.py` | Parallel Monte-Carlo runner (same channels for every scheme, seed = 1000 + run), result tables, Hu-style figures |

## Reproducibility
Channel seed = 1000 + run, identical for all schemes at a sweep point, so `--runs=100` reproduces the
report exactly (checked: zero difference on the default point). One BLAS thread per worker process is
set automatically; without it the parallel run is up to 100x slower.
