# 2. Experimental Setup

Source: submitted NCC 2027 paper, Table I, Sec. IV-C and Sec. V-A.

## Simulation settings (Table I)

| Parameter | Value |
|---|---|
| Antennas, pairs, targets | N_t = N_r = 8, M = 4, K = 4 |
| Target angles | -45°, 0°, 45°, 90° |
| User, relay, radar budgets | 20 dBm each |
| Distance, path-loss exponent | 100 m, 4 (normalised) |
| Disturbance and user noise | -80 dBm; equal clutter and noise power |
| Target reflection power gain | -90 dB |
| Texture shape, CSI error | ν = 0.7, ε = 0.01 |
| False alarm, detection target | α = 10⁻³, β = 0.9 |
| Channels, clutter endpoints | 40 paired channels (60 in Fig. 2); ν = 50 and ν = 0.3 |

Parameters vary where a figure says so. Fig. 2 sweeps the CSI-error ratio over ε ∈ {0, 5.5·10⁻⁴, 1.5·10⁻³, 3·10⁻³, 10⁻², 3·10⁻²}.

## Data

The data are simulated channel realisations; no external dataset is used.

- Every scheme is run on the **same** channel realisations, so all comparisons are paired. In the data files, the column `seed` identifies the channel.
- **Tuning channels.** The tuned SINR floors of the Hu baselines are chosen on 20 separate development channels. These channels are never used for evaluation.
- **Data files.** `03_Results/data/` holds one row per channel × scheme × setting.

## Assumptions (stated in the paper)

- **Link model.** The model is normalised. It does not specify a reference-distance loss, carrier frequency or bandwidth.
- **Other-target leakage.** It is modelled by a Gaussian surrogate, which is an approximation.
- **Detection requirement.** It is **averaged** over the assumed clutter and channel-error distributions. It is not enforced on every error realisation.
- **Feasibility.** The check is numerical, with a 0.003 design margin on β. It is not a formal certificate.

## Evaluation protocol and metrics

- **One evaluator for every scheme.** It uses the received-signal model and a threshold calibrated to P_FA = 10⁻³.
- **Protected channel.** Every target reaches P_D >= 0.9.
- **Protected SAR** (bit/s/Hz). The channel's SAR if the channel is protected, and zero otherwise. It is averaged over the channels. Main metric in Fig. 2 and Sec. V-E.
- **Average SAR** and **% of protected channels**. Used in Fig. 3. The 95 % confidence intervals are bootstrap intervals for the rate and Clopper-Pearson intervals for the percentage.
- **Worst-target detection probability** P_D^worst = min_k P_D,k, and **total admitted ST power** Σ p_m (mW). Used in Fig. 4.
- **Per-error-draw protection** and **worst-case SAR** over the bounded error sets. Used in Sec. V-F.

## Data column reference (`03_Results/data/*.csv`)

| Column | Meaning |
|---|---|
| `experiment`, `seed` | stage name; channel realisation (the same seed means the same channel in every scheme) |
| `eps`, `nu` | CSI-error ratio ε; clutter texture shape ν |
| `method` | `prop` proposed · `hu_nom` Hu non-robust · `hu_rob` Hu robust · `adapt_rob` adapted worst-case · `cell` component-study variant |
| `param` | β for `prop`; SINR floor in dB for the baselines; variant name for `cell` |
| `Nr, Nt, M, K` | array sizes, number of pairs, number of targets |
| `R`, `protected`, `R_protected` | SAR (bit/s/Hz); 1 if every target has P_D >= 0.9; R × protected |
| `minPD`, `PD_1..PD_4` | worst-target P_D; per-target P_D |
| `Psum_mW`, `p1_mW..p4_mW` | total and per-ST admitted power |
| `per_draw_protection` | share of error draws on which every target meets P_D >= 0.9 |
| `wc_sar` | worst-case SAR over the bounded error sets |
| `status` | solver status |

## Code: run the experiments and generate the results (`code/`)

These are the scripts that produced every NCC 2027 result.

| File | Role |
|---|---|
| `run_ncc_experiments.sh` | Runs every stage below in order, then builds the data files and redraws Figs. 2-4 |
| `b_campaign.py` | Simulation campaign: builds the channels (seeds), designs each scheme, scores it and saves one record per channel × scheme × setting to `../data/*.jsonl` |
| `v3_engine.py`, `v3_np.py` | **Proposed method** (Algorithm 1) and the adapted worst-case scheme; `v3_np.py` is the NumPy back-end (no GPU or torch needed) |
| `hu_core.py`, `hu_robust.py` | Hu et al. [3] baselines: non-robust Algorithm 1, and robust GLDT + cutting-set algorithm |
| `common_eval.py` | The common evaluator: received-signal model, threshold calibrated to P_FA = 10⁻³, P_D, protected SAR |
| `runner.py` | Resumable parallel task runner (an interrupted stage continues where it stopped) |
| `analyze.py` | `python analyze.py T` picks the tuned Hu floors on the development channels (`../data/tuned_eta.json`) |
| `published_hu.py`, `plotstyle.py` | Helpers imported by `analyze.py` |
| `export_csv.py` | Exports `../data/*.jsonl` to `../results_csv/*.csv` |
| `make_ncc_data.py` | Builds the five NCC data files (`../results_ncc/`, same format as `03_Results/data/`) |

**Stages and what they produce**

| Command | Produces | Used in |
|---|---|---|
| `python b_campaign.py main` | 4 schemes × 6 ε × 60 test channels (seeds 1000-1059) | Fig. 2, Fig. 4, Sec. V-F |
| `python b_campaign.py tune` | Hu floors on 20 development channels (seeds 2000-2019) | tuning only |
| `python analyze.py T` | `tuned_eta.json`: the smallest floor protecting ≥ 95 % of development channels | tuning only |
| `python b_campaign.py tuned` | Hu schemes at the tuned floors on the test channels | Fig. 2, Fig. 4 |
| `python b_campaign.py tradeoff` | each scheme swept over its parameter, ε = 0.01, 40 channels | Fig. 3 |
| `python b_campaign.py cells` | component study | Sec. V-E |
| `python b_campaign.py nu` | clutter-shape study | Sec. V-E |

The `cells` and `nu` stages compute a few more settings than the paper reports. Those extra settings are another filter variant, ε = 1.5·10⁻³, and the intermediate ν values. `make_ncc_data.py` keeps only the rows reported in the paper.

**How to run**

```
pip install -r code/requirements.txt
bash code/run_ncc_experiments.sh
```

- The full campaign takes several hours on a multi-core workstation. The robust Hu baseline is the slowest part, at about 4 minutes per design on one core.
- Set `NWORK` to the number of CPU cores to use; the default is 11.
- Results appear in `results_ncc/`: the five data files and Figs. 2-4.

**Reproducibility checks done for this package**

- **Single designs.** Re-running two designs on 1 Nov 2026 reproduced the stored rows exactly: the proposed method and Hu robust, each on channel 1000 at ε = 0.01. The rate and worst-target P_D matched to every printed digit for both designs, as did the worst-case SAR of the proposed design.
- **Export path.** `export_csv.py` → `make_ncc_data.py` on the stored raw records reproduces all five files in `03_Results/data/` byte for byte.
- **Tuned floors.** `analyze.py T` reproduces the tuned floors used in the paper: Hu non-robust 13 / 13 / 16 / 19 / 22 dB for ε = 5.5·10⁻⁴ … 3·10⁻², and Hu robust 7 / 4 dB for ε = 1.5·10⁻³ / 10⁻².
- **Not done.** The full multi-hour campaign was not re-run.
