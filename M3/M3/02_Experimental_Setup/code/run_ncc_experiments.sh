#!/usr/bin/env bash
# Runs every simulation stage behind the NCC 2027 results, then builds the NCC data files and redraws Figs. 2-4.
# Every stage resumes from ../data/*.jsonl, so an interrupted run continues where it stopped.
set -e
cd "$(dirname "$0")"
export NWORK=${NWORK:-11}                     # parallel worker processes
python b_campaign.py main                     # Fig. 2, Fig. 4, Sec. V-F: 4 schemes x 6 eps x 60 test channels
python b_campaign.py tune                     # Hu floors on the 20 development channels
python analyze.py T                           # pick the tuned floors -> ../data/tuned_eta.json
python b_campaign.py tuned                    # Fig. 2, Fig. 4: Hu schemes at their tuned floors, test channels
python b_campaign.py tradeoff                 # Fig. 3: each scheme swept over its design parameter, eps = 0.01
python b_campaign.py cells                    # Sec. V-E: component study
python b_campaign.py nu                       # Sec. V-E: clutter-shape study
python export_csv.py                          # ../data/*.jsonl -> ../results_csv/*.csv
python make_ncc_data.py                       # -> ../results_ncc/ (the five NCC data files)
cp ../../03_Results/data/plot_*.py ../results_ncc/
cd ../results_ncc
python plot_protected_sar_vs_eps.py           # Fig. 2
python plot_protection_rate_panels.py         # Fig. 3
python plot_power_vs_pd.py                    # Fig. 4
