"""
Builds the five NCC 2027 data files from the exported campaign CSVs (../results_csv/, written by export_csv.py):
main_comparison.csv, tuned_baselines.csv and tradeoff.csv in full; the component study at eps = 0.01 with the four
variants reported in Sec. V-E; the clutter study at the two reported shapes nu = 50 and nu = 0.3.
Only the columns used in the paper are kept. Output: ../results_ncc/ (same format as 03_Results/data/).
Usage: python make_ncc_data.py
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results_csv")
OUT = os.path.join(HERE, "..", "results_ncc")
KEEP = ["experiment", "seed", "eps", "method", "param", "nu", "Nr", "Nt", "M", "K", "R", "protected", "R_protected",
        "minPD", "PD_1", "PD_2", "PD_3", "PD_4", "Psum_mW", "p1_mW", "p2_mW", "p3_mW", "p4_mW",
        "per_draw_protection", "wc_sar", "status"]
FILES = [  # source csv -> NCC csv, row filter
    ("main_comparison.csv", "main_comparison.csv", lambda r: True),
    ("tuned_baselines.csv", "tuned_baselines.csv", lambda r: True),
    ("tradeoff.csv", "tradeoff.csv", lambda r: True),
    ("component_cells.csv", "component_study_eps0.01.csv",
     lambda r: r["eps"] == "0.01" and r["param"] in ("F1-S", "F1-PG", "F1-B", "F1-P")),
    ("clutter_shape.csv", "clutter_shape_endpoints.csv", lambda r: r["nu"] in ("50.0", "0.3")),
]

os.makedirs(OUT, exist_ok=True)
for src, dst, keep in FILES:
    rows = [r for r in csv.DictReader(open(os.path.join(SRC, src))) if keep(r)]
    with open(os.path.join(OUT, dst), "w", newline="") as f:
        w = csv.DictWriter(f, KEEP); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in KEEP})
    print(f"{dst}: {len(rows)} rows")
