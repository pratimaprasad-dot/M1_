"""
Export every experiment record (data/*.jsonl) to tidy CSV files in research/results_csv/ for plotting with any tool.
One row per channel x method x setting; column meanings are listed in results_csv/README.md.
Safe to re-run at any time (it rewrites the CSVs from whatever has finished so far).
Usage: python export_csv.py
"""
import os, csv, json, shutil
import runner

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results_csv")
os.makedirs(OUT, exist_ok=True)

DESIGN_COLS = ["experiment", "seed", "eps", "method", "param", "nu", "Nr", "Nt", "M", "K",
               "R", "protected", "R_protected", "R_model", "minPD", "PD_1", "PD_2", "PD_3", "PD_4", "PD_5", "PD_6",
               "minPD_designlaw", "protected_designlaw", "PFA_designthreshold_max", "per_draw_protection",
               "Psum_mW", "p1_mW", "p2_mW", "p3_mW", "p4_mW", "p5_mW", "p6_mW", "radar_intf_max", "relay_power_ratio",
               "wc_sar", "iters", "time_s", "status"]

EXPERIMENTS = {  # jsonl file -> csv file
    "b_main.jsonl": "main_comparison.csv", "b_tune.jsonl": "tuning_dev.csv", "b_tuned.jsonl": "tuned_baselines.csv",
    "b_cells.jsonl": "component_cells.csv", "b_tradeoff.jsonl": "tradeoff.csv", "b_nu.jsonl": "clutter_shape.csv",
    "b_nlp.jsonl": "alternative_solver.csv", "b_scale.jsonl": "runtime_scaling.csv",
}


def latest(name):
    """last non-error record per key (resumed runs may hold duplicates)"""
    d = {}
    for r in runner.load(name):
        if "error" not in r: d[runner._tup(r["key"])] = r
    return list(d.values())


def pad(xs, n): xs = list(xs); return xs + [""] * (n - len(xs))


def design_row(r):
    exp, seed, eps, method, param, nu, geo = r["key"]
    Nr, Nt, M, K = geo
    p = [1000 * x for x in r.get("p", [])]
    return dict(zip(DESIGN_COLS, [
        exp, seed, eps, method, param, nu, Nr, Nt, M, K,
        r["R"], int(r["prot"]), r["R"] * int(r["prot"]), r.get("R_model", ""), r["minpd"], *pad(r.get("pd", []), 6)[:6],
        r.get("minpd_gauss", ""), int(r.get("prot_gauss", False)), max(r.get("pfa_design", [0]) or [0]), r.get("prot_real", ""),
        r.get("Psum_mW", ""), *pad(p, 6)[:6], r.get("radar_intf_max", ""), r.get("Pt", ""),
        r.get("wc_sar", "") if r.get("wc_sar") is not None else "", r.get("iters", ""), r.get("time", ""), r.get("status", "")]))


def write(fname, cols, rows):
    with open(os.path.join(OUT, fname), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for row in rows: w.writerow(row)
    return len(rows)


def main():
    counts = {}
    allrows = []
    for src, dst in EXPERIMENTS.items():
        rows = sorted((design_row(r) for r in latest(src)), key=lambda x: (str(x["method"]), str(x["param"]), x["eps"], x["nu"], x["seed"]))
        if rows: counts[dst] = write(dst, DESIGN_COLS, rows); allrows += rows
    if allrows: counts["all_designs.csv"] = write("all_designs.csv", DESIGN_COLS, allrows)
    # Hu et al. reproduction (Layer A; finished before it was dropped)
    rep_cols = ["stage", "seed", "xi_dB", "Pc_dBm", "eta_dB", "relay", "variant", "SAR", "feasible", "min_sensing_SINR_dB", "relay_power_ratio", "iters", "time_s"]
    rows = []
    for f in ("a1_cal.jsonl", "a1_fig3.jsonl", "a1_fig9.jsonl"):
        for r in latest(f):
            k = list(r["key"]) + (["L"] if len(r["key"]) == 6 else [])
            rows.append(dict(zip(rep_cols, [k[0], k[1], k[2], k[3], k[4], k[5], k[6], r["sar"], int(r["feasible"]), r["min_sinr_db"], r["Pt"], r["iters"], r["time"]])))
    if rows: counts["hu_reproduction.csv"] = write("hu_reproduction.csv", rep_cols, rows)
    # false-alarm validation: one row per target
    pcols = ["design_set", "seed", "eps", "method", "param", "nu", "target", "N", "PFA_calibrated", "PFA_designlaw", "PD_bruteforce", "PD_evaluator"]
    rows = []
    for r in latest("e_pfa.jsonl"):
        exp, seed, eps, method, param, nu, N = r["key"]
        for k in range(len(r["pfa_cal"])):
            rows.append(dict(zip(pcols, [exp, seed, eps, method, param, nu, k + 1, N, r["pfa_cal"][k] / N, r["pfa_des"][k] / N, r["pd_bf"][k], r["pd_cmc"][k]])))
    if rows: counts["pfa_validation.csv"] = write("pfa_validation.csv", pcols, rows)
    # small-system reference: one row per seed x solver
    scols = ["seed", "solver", "R_model", "R", "minPD", "protected", "time_s", "p1_mW", "p2_mW"]
    rows = []
    for r in latest("d_smallref.jsonl"):
        for s in ("grid", "alg1", "nlp"):
            x = r[s]; p = [1000 * v for v in x["p"]]
            rows.append(dict(zip(scols, [r["key"], s, x["R_model"], x["R"], x["minpd"], int(x["prot"]), x["time"], *pad(p, 2)[:2]])))
    if rows: counts["small_system_reference.csv"] = write("small_system_reference.csv", scols, rows)
    tp = os.path.join(runner.DATA, "tuned_eta.json")
    if os.path.exists(tp): shutil.copy(tp, os.path.join(OUT, "tuned_eta.json"))
    rt = os.path.join(runner.DATA, "results_tables.md")
    if os.path.exists(rt): shutil.copy(rt, os.path.join(OUT, "summary_tables.md"))
    for k, v in counts.items(): print(f"{k:32s} {v:6d} rows")


if __name__ == "__main__":
    main()
