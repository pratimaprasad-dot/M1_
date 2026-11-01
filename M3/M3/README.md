# M3 Research Project Package

**Detection-Constrained Power Allocation for Cognitive ISAC Relays under Imperfect CSI**
(work submitted to NCC 2027)

## Scope

This package contains **only** the material behind the submitted NCC 2027 paper:
- the method,
- the simulation code that produced its results,
- the data behind its figures and reported numbers,
- the figures themselves.

No other experiments or results are included. `03_Results/reported_values_check.csv` recomputes every number reported in the paper from the data files.

## M3 requirements and where they are covered

| M3 requirement | Folder | Document |
|---|---|---|
| Final proposed method, model or framework | `01_Proposed_Method/` | `METHOD.pdf` |
| Experimental setup: data, assumptions, configuration, metrics, and the code that runs the experiments | `02_Experimental_Setup/` | `SETUP.md`, `code/` |
| Results showing the behaviour and effectiveness of the method | `03_Results/` | `RESULTS.pdf` |
| Comparison with baselines | `04_Baseline_Comparison/` | `BASELINES.pdf` |
| Revised SOTA positioning supported by the results | `05_SOTA_Positioning/` | `SOTA.pdf` |

## Folder layout

```
README.md
01_Proposed_Method/       METHOD.pdf (Sec. II-IV, Algorithm 1), Fig1_system_model.png / .svg
02_Experimental_Setup/    SETUP.md (Table I, data, assumptions, metrics, how to run)
  code/                   simulation code + run_ncc_experiments.sh
03_Results/               RESULTS.pdf (results with Figs. 2-4), reported_values_check.csv
  figures/                Figs. 2-4 as used in the paper
  data/                   the five data files + the three figure scripts
04_Baseline_Comparison/   BASELINES.pdf (Table II, fairness, comparison tables)
05_SOTA_Positioning/      SOTA.pdf (related work, positioning, limits of the claim)
```

## Reproducing the results

```
pip install -r 02_Experimental_Setup/code/requirements.txt
bash 02_Experimental_Setup/code/run_ncc_experiments.sh
```

This runs every simulation stage. It then writes the five data files and Figs. 2-4 to `02_Experimental_Setup/results_ncc/`. Details and the reproducibility checks are in `02_Experimental_Setup/SETUP.md`.

Reference numbers such as [3] follow the reference list of the NCC 2027 paper.
