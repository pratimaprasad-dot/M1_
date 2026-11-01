"""
Analysis of all Version_4 research runs -> figures/*.png|pdf (base-paper look, plotstyle.py) and data/results_tables.md
(every number in the report comes from this file). Statistics: paired bootstrap over channel IDs (10,000 resamples),
exact Clopper-Pearson intervals for proportions, Wilcoxon signed-rank for paired differences.
Usage: python analyze.py [section ...]   sections: A T B X C M N F S R (default: all available)
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "1")   # before numpy: one BLAS thread per worker

import sys, json, warnings
from collections import defaultdict
import numpy as np
from scipy import stats
import plotstyle as PS
import matplotlib.pyplot as plt
import runner
import published_hu as PUB

warnings.filterwarnings("ignore")
PS.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
OUTMD = os.path.join(runner.DATA, "results_tables.md")
os.makedirs(FIG, exist_ok=True)
F = lambda name: os.path.join(FIG, name)

LAB = {"prop": "Proposed (P_D certificate)", "hu_rob": "Hu robust GLDT-CS, eta=10 dB", "hu_nom": "Hu non-robust Alg. 1, eta=10 dB",
       "hu_nom_T": "Hu non-robust, tuned eta", "hu_rob_T": "Hu robust, tuned eta", "adapt_rob": "Adapted worst-case (earlier baseline)",
       "perfect": "Perfect CSI (proposed at eps=0)"}
MD = []


def md(*lines): MD.extend(lines)


def cp(k, n, conf=0.95):
    a = 1 - conf
    return (stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0, stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0)


def boot_mean(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def boot_ratio(a, b, n=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-12) - 1
    return a.mean() / max(b.mean(), 1e-12) - 1, np.percentile(r, 2.5), np.percentile(r, 97.5)


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.all(np.abs(d) < 1e-12): return 1.0
    try: return stats.wilcoxon(d).pvalue
    except ValueError: return np.nan


def ci_s(m, lo, hi, f="{:.2f}"): return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"
def err(ms): return [[m[0] - m[1] for m in ms], [m[2] - m[0] for m in ms]]
def load(name): return [r for r in runner.load(name) if "error" not in r]
def fmt_eps(e): return "0" if e == 0 else f"{e:.2g}"


def eps_axis(ax, eps_list):
    ax.set_xticks(range(len(eps_list)))
    lab = {0.0: "$0$", 5.48e-4: "$5.5{\\times}10^{-4}$", 1.52e-3: "$1.5{\\times}10^{-3}$", 3e-3: "$3{\\times}10^{-3}$",
           1e-2: "$10^{-2}$", 3e-2: "$3{\\times}10^{-2}$", 1e-1: "$10^{-1}$"}
    ax.set_xticklabels([lab.get(e, f"{e:g}") for e in eps_list], fontsize=7)
    ax.set_xlabel(r"CSI-error ratio $\epsilon$")
    ax.set_xlim(-0.3, len(eps_list) - 0.7)


# =================================================================================================== Layer A
def section_A():
    md("## Layer A - reproduction of Hu et al. (their environment, their assumptions)", "")
    cal = load("a1_cal.jsonl")
    for var, lab in (("L", "literal Alg. 1 (W fixed inside (17))"), ("K", "relay-gain-normalised Alg. 1")):
        rows = defaultdict(list)
        for r in cal:
            k = r["key"]; v = k[6] if len(k) > 6 else "L"
            if v == var: rows[k[2]].append(r["sar"])
        if not rows: continue
        md(f"**Calibration of xi on development seeds 5000-5019, {lab}** (anchor: Fig. 3, P_c = 20 dBm, eta = 10 dB -> 8.80)", "",
           "| xi (dB) | mean SAR [95% CI] | n |", "|---|---|---|")
        for x in sorted(rows):
            md(f"| {x:g} | {ci_s(*boot_mean(rows[x]))} | {len(rows[x])} |")
        md("")
    f9 = load("a1_fig9.jsonl"); f3 = load("a1_fig3.jsonl")

    def grp(recs, var, idx):
        d, it, feas = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in recs:
            k = r["key"]; v = k[6] if len(k) > 6 else "L"
            if v != var: continue
            kk = idx(k); d[kk].append(r["sar"]); it[kk].append(r["iters"]); feas[kk].append(r["feasible"])
        return d, it, feas

    # ---- Fig. 3 counterpart
    md("**Fig. 3 reproduction (test seeds 1000-1029): SAR vs P_c, eta = 10 dB, perfect CSI**", "",
       "| Scheme | Variant | " + " | ".join(f"P_c={p} dBm" for p in PUB.FIG3_PC) + " | mean abs. dev. | feasible |", "|---|---|" + "---|" * (len(PUB.FIG3_PC) + 2))
    cols = {"opt": (PS.C["blue"], "*"), "LMMSE": (PS.C["green"], "d"), "ZF": (PS.C["red"], "v"), "MRC/MRT": (PS.C["lightblue"], ">")}
    f, ax = PS.fig(PS.COL_W, 2.9)
    for rl in ("opt", "LMMSE", "ZF", "MRC/MRT"):
        pub = PUB.FIG3[rl]; lbl = "Alg. 1" if rl == "opt" else rl
        c, mk = cols[rl]
        md(f"| {lbl} | published | " + " | ".join(f"{x:.2f}" for x in pub) + " | - | - |")
        PS.line(ax, PUB.FIG3_PC, pub, color=c, marker=mk, ls=(0, (1, 1.5)), label=f"{lbl} [Hu, Fig. 3]", lw=0.9)
        for var in ("K", "L"):
            d, it, feas = grp(f3, var, lambda k: (k[5], k[3]))
            pts = [d.get((rl, float(p)), []) for p in PUB.FIG3_PC]
            if not all(pts): continue
            ms = [boot_mean(x) for x in pts]
            mad = np.mean([abs(m[0] - p) for m, p in zip(ms, pub)])
            fe = np.mean([np.mean(feas[(rl, float(p))]) for p in PUB.FIG3_PC])
            md(f"| {lbl} | reproduced {var} | " + " | ".join(ci_s(*m) for m in ms) + f" | {mad:.2f} | {100*fe:.0f}% |")
            if var == "K":
                PS.line(ax, PUB.FIG3_PC, [m[0] for m in ms], color=c, marker=mk, ls="-", yerr=err(ms), label=f"{lbl} (reproduced)", filled=True)
            elif rl == "opt":
                PS.line(ax, PUB.FIG3_PC, [m[0] for m in ms], color=PS.C["purple"], marker="p", ls="-.", yerr=err(ms), label="Alg. 1 literal (reproduced)")
    ax.set_xlabel(r"Transmit power of the secondary transmitters $P_c$ (dBm)"); ax.set_ylabel("SAR (bps/Hz)")
    ax.set_xlim(-0.5, 20.5); ax.set_xticks([0, 5, 10, 15, 20])
    ax.legend(loc="upper left", ncol=2, fontsize=5.6, handlelength=2.2, columnspacing=0.8)
    ax.set_ylim(0, 13.5)
    PS.save(f, F("fig01a_reproduction_hu_fig3"))
    md("")

    # ---- Fig. 9 / Fig. 2 counterpart
    md("**Fig. 9 / Fig. 2 reproduction (perfect CSI): SAR vs eta_min, P_c = 20 dBm, test seeds 1000-1049**", "",
       "| eta (dB) | published Fig. 9 | published Fig. 2 | reproduced K [95% CI] | iters K (mean / hit cap 100) | reproduced literal [95% CI] | iters literal |", "|---|---|---|---|---|---|---|")
    dK, itK, fK = grp(f9, "K", lambda k: k[4]); dL, itL, fL = grp(f9, "L", lambda k: k[4])
    pub9 = dict(zip(PUB.FIG9_ETA, PUB.FIG9["perfect"]))
    for e in sorted(set(dK) | set(dL)):
        a = boot_mean(dK[e]) if dK.get(e) else None; b = boot_mean(dL[e]) if dL.get(e) else None
        md(f"| {e:g} | {pub9.get(int(e), '-')} | {PUB.FIG2.get(e, '-')} | {ci_s(*a) if a else '-'} | "
           f"{np.mean(itK[e]):.0f} / {sum(np.array(itK[e]) >= 100)} | {ci_s(*b) if b else '-'} | {np.mean(itL[e]):.0f} / {sum(np.array(itL[e]) >= 100) if itL.get(e) else '-'} |")
    f, ax = PS.fig(PS.COL_W, 2.9)
    PS.line(ax, PUB.FIG9_ETA, PUB.FIG9["perfect"], color=PS.C["blue"], marker="*", ls=(0, (1, 1.5)), label=r"perfect CSI [Hu, Fig. 9]", lw=0.9)
    PS.line(ax, list(PUB.FIG2), list(PUB.FIG2.values()), color=PS.C["grey"], marker="o", ls="none", label=r"Alg. 1 converged [Hu, Fig. 2]")
    for d, c, mk, ls, lab in ((dK, PS.C["blue"], "*", "-", "perfect CSI (reproduced)"), (dL, PS.C["purple"], "p", "-.", "perfect CSI, literal Alg. 1 (reproduced)")):
        es = sorted(d); ms = [boot_mean(d[e]) for e in es]
        PS.line(ax, es, [m[0] for m in ms], color=c, marker=mk, ls=ls, yerr=err(ms), label=lab, filled=True)
    rb = load("a2_fig9_robust.jsonl")
    if rb:
        md("", "**Fig. 9 robust reproduction (variant K; worst-case SAR over the balls; seeds 1000-1007)**", "",
           "| eps_hat | eta (dB) | published Alg3 | reproduced Alg3 [95% CI] | published non-robust | reproduced non-robust [95% CI] | Alg3 worst-case SINR >= eta | Alg3 time (s) | outer / inner iters |",
           "|---|---|---|---|---|---|---|---|---|")
        for eh, c3, cn in ((0.03, PS.C["brown"], PS.C["maroon"]), (0.05, PS.C["orange"], PS.C["tan"])):
            xs3, m3, xsn, mn = [], [], [], []
            for e in (0.0, 10.0, 15.0):
                rr = [r for r in rb if r["key"][2] == eh and r["key"][1] == e and "alg3_wc" in r]
                if not rr: continue
                pa = np.interp(e, PUB.FIG9_ETA, PUB.FIG9[f"alg3_{eh}"]); pn = np.interp(e, PUB.FIG9_ETA, PUB.FIG9[f"nonrob_{eh}"])
                a = boot_mean([r["alg3_wc"] for r in rr]); n_ = boot_mean([r["nonrob_wc"] for r in rr])
                ok = np.mean([r["alg3_wc_sinr_db"] >= e - 0.01 for r in rr])
                md(f"| {eh} | {e:g} | {pa:.2f} | {ci_s(*a)} | {pn:.2f} | {ci_s(*n_)} | {100*ok:.0f}% | {np.mean([r['alg3_time'] for r in rr]):.0f} | "
                   f"{np.mean([r['alg3_outer'] for r in rr]):.1f} / {np.mean([r['alg3_inner'] for r in rr]):.0f} |")
                xs3.append(e); m3.append(a); xsn.append(e); mn.append(n_)
            if xs3:
                PS.line(ax, PUB.FIG9_ETA, PUB.FIG9[f"alg3_{eh}"], color=c3, marker="s", ls=(0, (1, 1.5)), label=rf"Alg3, $\epsilon={eh}$ [Hu]", lw=0.9)
                PS.line(ax, xs3, [m[0] for m in m3], color=c3, marker="s", ls="-", yerr=err(m3), label=rf"Alg3, $\epsilon={eh}$ (reproduced)", filled=True)
                PS.line(ax, PUB.FIG9_ETA, PUB.FIG9[f"nonrob_{eh}"], color=cn, marker="<", ls=(0, (1, 1.5)), label=rf"Non-robust, $\epsilon={eh}$ [Hu]", lw=0.9)
                PS.line(ax, xsn, [m[0] for m in mn], color=cn, marker="<", ls="--", yerr=err(mn), label=rf"Non-robust, $\epsilon={eh}$ (reproduced)", filled=True)
    ax.set_xlabel(r"Minimum sensing SINR, $\eta_{\min}$ (dB)"); ax.set_ylabel("SAR (bps/Hz)")
    ax.set_xticks([0, 3, 6, 9, 12, 15]); ax.set_xlim(-0.5, 15.5); ax.set_ylim(0, 13.8)
    ax.legend(loc="lower left", fontsize=5.3, ncol=2, handlelength=2.0, columnspacing=0.6)
    PS.save(f, F("fig01b_reproduction_hu_fig9"))
    md("")


# =================================================================================================== helpers for B
def table(name, exp=None):
    T = {}
    for r in load(name):
        k = r["key"]
        if exp and k[0] != exp: continue
        T[(k[1], k[2], k[3], float(k[4]) if not isinstance(k[4], str) else k[4], k[5])] = r
    return T


def series(T, eps, method, param, nu=0.7, seeds=None):
    out = {}
    for (s, e, m, p, n), r in T.items():
        if abs(e - eps) < 1e-12 and m == method and (param is None or p == param) and abs(n - nu) < 1e-9 and (seeds is None or s in seeds):
            out[s] = r
    return out


def psar(r): return r["R"] * (1.0 if r["prot"] else 0.0)


# =================================================================================================== tuning
def section_T():
    tune = table("b_tune.jsonl")
    if not tune: return {}
    md("## Tuned SINR thresholds (development seeds 2000-2019, disjoint from the test seeds)", "",
       "Rule fixed before looking at test data: smallest eta on the grid whose development protection rate (worst-target P_D >= 0.9 under the common evaluator) is >= 95%; "
       "if none reaches 95%, the grid value with the highest protection (reported as 'not reached').", "",
       "| method | eps | eta grid: dev protection % (mean SAR) | tuned eta |", "|---|---|---|---|")
    out = {}
    for m, grid in (("hu_nom", (10.0, 13.0, 16.0, 19.0, 22.0, 25.0)), ("hu_rob", (0.0, 4.0, 7.0, 10.0))):
        for e in sorted({k[1] for k in tune}):
            cells, pick, best = [], None, (-1, None)
            for x in grid:
                rr = series(tune, e, m, x)
                if len(rr) < 15: continue
                pr = np.mean([r["prot"] for r in rr.values()]); R = np.mean([r["R"] for r in rr.values()])
                cells.append(f"{x:g}: {100*pr:.0f}% ({R:.2f})")
                if pick is None and pr >= 0.95: pick = x
                if pr > best[0]: best = (pr, x)
            if not cells: continue
            note = ""
            if pick is None: pick = best[1]; note = " (95% not reached)"
            out[f"{m}|{e:g}"] = pick
            md(f"| {m} | {fmt_eps(e)} | " + "; ".join(cells) + f" | {pick:g}{note} |")
    json.dump(out, open(os.path.join(runner.DATA, "tuned_eta.json"), "w"), indent=1)
    md("")
    return out


# =================================================================================================== Layer B main
def collect_main():
    T = table("b_main.jsonl", "main")
    TT = table("b_tuned.jsonl", "tuned")
    tp = os.path.join(runner.DATA, "tuned_eta.json")
    tun = json.load(open(tp)) if os.path.exists(tp) else {}
    eps_list = sorted({k[1] for k in T})
    D = {}
    for e in eps_list:
        D[(e, "prop")] = series(T, e, "prop", 0.9)
        D[(e, "hu_nom")] = series(T, e, "hu_nom", 10.0)
        D[(e, "adapt_rob")] = series(T, e, "adapt_rob", 9.8017)
        D[(e, "hu_rob")] = series(T, e, "hu_rob", 10.0) if e > 0 else series(T, e, "hu_nom", 10.0)
        for m in ("hu_nom", "hu_rob"):
            x = tun.get(f"{m}|{e:g}")
            if x is not None:
                cand = series(TT, e, m, float(x)) or series(T, e, m, float(x))
                if cand: D[(e, m + "_T")] = cand
        D[(e, "perfect")] = series(T, 0.0, "prop", 0.9)
    # keep only epsilon levels with a complete main design set
    eps_list = [e for e in eps_list if len(D[(e, "prop")]) >= 20 and (e == 0 or len(D[(e, "hu_rob")]) >= 20)]
    return D, eps_list


def common_seeds(D, e, methods):
    s = None
    for m in methods:
        ks = set(D.get((e, m), {}))
        s = ks if s is None else s & ks
    return sorted(s or [])


def section_B():
    D, eps_list = collect_main()
    if not eps_list: return None
    methods = [m for m in ["prop", "hu_rob", "hu_rob_T", "hu_nom", "hu_nom_T", "adapt_rob"]]
    md("## Layer B - full-method comparison in the common environment", "",
       "Common environment: manuscript model (Nt = Nr = 8, M = K = 4, P_c = P^max = P_s = 20 dBm, fixed equal-gain radar beam for every method, "
       "compound-Gaussian clutter nu = 0.7 with clutter share 0.5, Gaussian CSI errors of variance eps on h, g and the target coefficients). "
       "Evaluation: common_eval (literal signal model, one matched-filter detector, threshold calibrated to the true P_FA = 1e-3, "
       "ergodic SAR by Monte Carlo). Protected = worst-target P_D >= 0.9. Paired over the same test channels.", "")
    md("**Main table: protected SAR (bit/s/Hz) [95% bootstrap CI]; % protected [Clopper-Pearson]; raw SAR; n channels**", "",
       "| method | " + " | ".join(f"eps={fmt_eps(e)}" for e in eps_list) + " |", "|---|" + "---|" * len(eps_list))
    S = {}
    for m in methods + ["perfect"]:
        cells = []
        for e in eps_list:
            rr = D.get((e, m), {})
            if not rr: cells.append("-"); continue
            z = [psar(r) for r in rr.values()]; kp = sum(r["prot"] for r in rr.values()); n = len(rr)
            mz = boot_mean(z); lo, hi = cp(kp, n); raw = boot_mean([r["R"] for r in rr.values()])
            S[(e, m)] = dict(psar=mz, prot=(kp / n, lo, hi), raw=raw, n=n)
            cells.append(f"{mz[0]:.2f} [{mz[1]:.2f}, {mz[2]:.2f}]; {100*kp/n:.0f}% [{100*lo:.0f}, {100*hi:.0f}]; raw {raw[0]:.2f}; n={n}")
        md(f"| {LAB[m]} | " + " | ".join(cells) + " |")
    md("")
    md("**Paired difference in protected SAR, proposed minus baseline (bit/s/Hz) [95% CI]; relative gain [95% CI]; channels won; Wilcoxon p**", "",
       "| baseline | " + " | ".join(f"eps={fmt_eps(e)}" for e in eps_list) + " |", "|---|" + "---|" * len(eps_list))
    for b in ["hu_rob", "hu_rob_T", "hu_nom", "hu_nom_T", "adapt_rob"]:
        cells = []
        for e in eps_list:
            ss = common_seeds(D, e, ["prop", b])
            if len(ss) < 5: cells.append("-"); continue
            a = np.array([psar(D[(e, "prop")][s]) for s in ss]); c = np.array([psar(D[(e, b)][s]) for s in ss])
            dm = boot_mean(a - c); rg = boot_ratio(a, c); won = int(np.sum(a > c + 1e-9))
            rgs = f"{100*rg[0]:+.1f}% [{100*rg[1]:+.1f}, {100*rg[2]:+.1f}]" if c.mean() > 0.05 else "n/a (baseline ~0)"
            cells.append(f"{dm[0]:+.2f} [{dm[1]:+.2f}, {dm[2]:+.2f}]; {rgs}; {won}/{len(ss)}; p={wil(a, c):.1g}")
        md(f"| {LAB[b]} | " + " | ".join(cells) + " |")
    md("")
    md("**Raw-rate difference (proposed minus baseline, bit/s/Hz) [95% CI] - shown so the protected metric cannot hide where the gain comes from**", "",
       "| baseline | " + " | ".join(f"eps={fmt_eps(e)}" for e in eps_list) + " |", "|---|" + "---|" * len(eps_list))
    for b in ["hu_rob", "hu_rob_T", "hu_nom", "hu_nom_T", "adapt_rob"]:
        cells = []
        for e in eps_list:
            ss = common_seeds(D, e, ["prop", b])
            if len(ss) < 5: cells.append("-"); continue
            cells.append(ci_s(*boot_mean([D[(e, 'prop')][s]["R"] - D[(e, b)][s]["R"] for s in ss]), f="{:+.2f}"))
        md(f"| {LAB[b]} | " + " | ".join(cells) + " |")
    md("")
    md("**Hu's own criterion: worst-case SAR over the 95% uncertainty balls (bit/s/Hz, mean [95% CI]), evaluated for every design**", "",
       "| method | " + " | ".join(f"eps={fmt_eps(e)}" for e in eps_list if e > 0) + " |", "|---|" + "---|" * len([e for e in eps_list if e > 0]))
    for m in methods:
        cells = []
        for e in eps_list:
            if e == 0: continue
            rr = [r.get("wc_sar") for r in D.get((e, m), {}).values() if r.get("wc_sar") is not None]
            cells.append(ci_s(*boot_mean(rr)) if rr else "-")
        md(f"| {LAB[m]} | " + " | ".join(cells) + " |")
    md("")
    X = {e: i for i, e in enumerate(eps_list)}
    # ---- three single-column figures
    for fname, key, ylab, scale in (("fig02_protected_sar_vs_eps", "psar", "Protected SAR (bps/Hz)", 1),
                                    ("fig03_protected_percent_vs_eps", "prot", "Protected configurations (%)", 100),
                                    ("fig04_raw_sar_vs_eps", "raw", "Average SAR, unprotected (bps/Hz)", 1)):
        f, ax = PS.fig(PS.COL_W, 2.8)
        for m in methods + (["perfect"] if key != "prot" else []):
            es = [e for e in eps_list if (e, m) in S and not (e == 0 and m in ("hu_rob", "hu_rob_T"))]   # radius 0: identical to non-robust
            if not es: continue
            v = [S[(e, m)][key] for e in es]
            v = [(scale * t[0], scale * t[1], scale * t[2]) for t in v]
            PS.line(ax, [X[e] + {"prop": 0, "hu_rob": 0.025, "hu_rob_T": 0.05, "hu_nom": -0.025, "hu_nom_T": -0.05, "adapt_rob": 0.075, "perfect": 0}[m] for e in es],
                    [t[0] for t in v], key=m, yerr=err(v))
        eps_axis(ax, eps_list); ax.set_ylabel(ylab)
        if key == "prot":
            ax.axhline(95, color="black", lw=0.6, ls=(0, (6, 3))); ax.set_ylim(-3, 103)
            ax.text(len(eps_list) - 0.75, 96.5, "95%", fontsize=6.5, ha="right")
        if key != "prot": ax.set_ylim(bottom=0)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", ncol=2, fontsize=5.4, borderaxespad=0)
        PS.save(f, F(fname))
    # ---- detection distribution
    for e in (1.52e-3, 1e-2):
        if e not in eps_list: continue
        f, ax = PS.fig(PS.COL_W, 2.7)
        for m in methods:
            rr = D.get((e, m), {})
            if not rr: continue
            c, mk, ls, lab = PS.METHOD[m]
            x = np.sort([r["minpd"] for r in rr.values()]); y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=c, ls=ls, lw=1.1, label=lab)
            ax.plot(x[::max(1, len(x) // 8)], y[::max(1, len(x) // 8)], ls="none", marker=mk, mfc="white" if mk not in "x*" else c, mec=c, ms=4 if mk != "*" else 6)
        ax.axvline(0.9, color="black", lw=0.6, ls=(0, (6, 3))); ax.text(0.893, 0.03, r"$\beta=0.9$", rotation=90, fontsize=6.5, ha="right")
        ax.set_xlabel(r"Worst-target detection probability $\min_k P_{D,k}$"); ax.set_ylabel("Empirical CDF")
        ax.set_xlim(max(0, min(r["minpd"] for m in methods for r in D.get((e, m), {}).values()) - 0.02), 1.005); ax.set_ylim(0, 1.02)
        ax.legend(loc="upper left", fontsize=5.6)
        PS.save(f, F(f"fig05_pd_cdf_eps{e:g}"))
    md("**Detection shortfalls (worst-target P_D < 0.9): frequency and severity**", "",
       "| method | eps | % below 0.9 | mean shortfall when below | 5th pct worst-target P_D | mean worst-target P_D |", "|---|---|---|---|---|---|")
    for e in (5.48e-4, 1.52e-3, 1e-2, 3e-2):
        if e not in eps_list: continue
        for m in methods:
            rr = D.get((e, m), {})
            if not rr: continue
            x = np.array([r["minpd"] for r in rr.values()]); bel = x < 0.9
            md(f"| {LAB[m]} | {fmt_eps(e)} | {100*bel.mean():.0f}% | {(0.9 - x[bel]).mean() if bel.any() else 0:.3f} | {np.percentile(x, 5):.3f} | {x.mean():.3f} |")
    md("")
    return D, S


# =================================================================================================== trade-off
def section_X(D):
    TO = table("b_tradeoff.jsonl", "tradeoff")
    if not TO: return
    e = 1e-2
    pts = defaultdict(dict)
    for (s, ee, m, p, n), r in TO.items():
        if abs(ee - e) < 1e-12: pts[m].setdefault(p, {})[s] = r
    pts["prop"][0.9] = D.get((e, "prop"), {}); pts["hu_nom"][10.0] = D.get((e, "hu_nom"), {})
    pts["hu_rob"][10.0] = D.get((e, "hu_rob"), {}); pts["adapt_rob"][9.8017] = D.get((e, "adapt_rob"), {})
    seeds = None
    for m in pts:
        for p in pts[m]:
            if pts[m][p]: seeds = set(pts[m][p]) if seeds is None else seeds & set(pts[m][p])
    seeds = sorted(seeds or [])
    if len(seeds) < 10: return
    md("## Protection-rate trade-off and matched-protection comparison (eps = 0.01, common test seeds)", "",
       f"Declared design parameter per method: proposed beta; Hu non-robust eta_min; Hu robust eta_min; adapted worst-case eta. n = {len(seeds)} paired channels.", "",
       "| method | design parameter | % protected [CP 95%] | mean SAR (raw) [95% CI] | protected SAR [95% CI] | mean worst-target P_D | mean total ST power (mW) |", "|---|---|---|---|---|---|---|")
    curve = {}
    f, ax = PS.fig(PS.COL_W, 2.9)
    for m in ("prop", "hu_rob", "hu_nom", "adapt_rob"):
        rows = []
        for p in sorted(pts[m]):
            rr = [pts[m][p][s] for s in seeds if s in pts[m][p]]
            if len(rr) < len(seeds): continue
            kp = sum(r["prot"] for r in rr); lo, hi = cp(kp, len(rr))
            R = boot_mean([r["R"] for r in rr]); Z = boot_mean([psar(r) for r in rr])
            rows.append((p, kp / len(rr), lo, hi, R, Z, np.mean([r["minpd"] for r in rr]), np.mean([r["Psum_mW"] for r in rr])))
            md(f"| {LAB[m].split(',')[0]} | {p:g} | {100*kp/len(rr):.0f}% [{100*lo:.0f}, {100*hi:.0f}] | {ci_s(*R)} | {ci_s(*Z)} | {rows[-1][6]:.3f} | {rows[-1][7]:.0f} |")
        curve[m] = rows
        rs = sorted(rows, key=lambda t: (t[1], -t[4][0]))
        tl = {"prop": r"Proposed, $\beta$ swept", "hu_rob": r"Hu et al. robust, $\eta_{\min}$ swept", "hu_nom": r"Hu et al. non-robust, $\eta_{\min}$ swept", "adapt_rob": r"Adapted worst-case, $\eta$ swept"}[m]
        PS.line(ax, [100 * t[1] for t in rs], [t[4][0] for t in rs], key=m, label=tl, yerr=err([t[4] for t in rs]),
                xerr=[[100 * (t[1] - t[2]) for t in rs], [100 * (t[3] - t[1]) for t in rs]])
        # labels: neighbouring operating points share one label; at 100 % only the best (matched-protection) point is named
        fmt = (lambda v: f"{v:g}") if m == "prop" else (lambda v: f"{v:.3g}")
        unit = "" if m == "prop" else " dB"
        pre = r"$\beta$=" if m == "prop" else ""
        dy = {"prop": 6, "hu_rob": 6, "hu_nom": -11, "adapt_rob": 6}[m]
        low = [t for t in rs if t[1] < 0.99]
        groups = []
        for t in low:
            if groups and 100 * (t[1] - groups[-1][-1][1]) < 12: groups[-1].append(t)
            else: groups.append([t])
        for g in groups:
            ps = sorted(t[0] for t in g)
            nm = pre + (fmt(ps[0]) if len(ps) == 1 else f"{fmt(ps[0])}–{fmt(ps[-1])}") + unit
            x = np.mean([100 * t[1] for t in g]); y = np.mean([t[4][0] for t in g])
            ax.annotate(nm, (x, y), textcoords="offset points", xytext=(0, dy), fontsize=5.5, ha="center", color=PS.METHOD[m][0])
        full = [t for t in rs if t[1] >= 0.99 and t[4][0] > 0]
        if full:
            t = max(full, key=lambda t: t[4][0])
            ax.annotate(pre + fmt(t[0]) + unit, (104.5, t[4][0]), textcoords="offset points", xytext=(4, 0),
                        fontsize=5.5, ha="left", va="center", color=PS.METHOD[m][0], annotation_clip=False)
    ax.set_xlabel(r"Achieved protection: channels with $\min_k P_{D,k}\geq 0.9$ (%)"); ax.set_ylabel("Average SAR (bps/Hz)")
    ax.axvline(95, color="black", lw=0.6, ls=(0, (6, 3))); ax.legend(loc="lower left", fontsize=5.6)
    PS.save(f, F("fig06_protection_rate_tradeoff"))
    md("")
    md("**Rate at matched protection** (highest-SAR operating point whose achieved protection is >= the level; 'infeasible' if none reaches it)", "",
       "| protection level | " + " | ".join(LAB[m] for m in curve) + " |", "|---|" + "---|" * len(curve))
    for lev in (0.80, 0.90, 0.95, 0.99):
        cells = []
        for m, rows in curve.items():
            ok = [t for t in rows if t[1] >= lev - 1e-9]
            if not ok: cells.append("infeasible"); continue
            t = max(ok, key=lambda t: t[4][0]); cells.append(f"{ci_s(*t[4])} (param {t[0]:g}, prot {100*t[1]:.0f}%)")
        md(f"| >= {100*lev:.0f}% | " + " | ".join(cells) + " |")
    md("", "**Protection at matched rate** (highest-protection operating point whose mean SAR is >= the level)", "",
       "| SAR level | " + " | ".join(LAB[m] for m in curve) + " |", "|---|" + "---|" * len(curve))
    for lev in (4.0, 5.0, 6.0, 7.0, 7.5, 8.0):
        cells = []
        for m, rows in curve.items():
            ok = [t for t in rows if t[4][0] >= lev]
            if not ok: cells.append("infeasible"); continue
            t = max(ok, key=lambda t: t[1]); cells.append(f"{100*t[1]:.0f}% [{100*t[2]:.0f}, {100*t[3]:.0f}] (param {t[0]:g})")
        md(f"| >= {lev:g} | " + " | ".join(cells) + " |")
    md("")


# =================================================================================================== Layer C
CELLS = ("F0-S", "F1-S", "F0-B", "F1-B", "F0-P", "F1-P", "F1-PG")


def section_C():
    T = table("b_cells.jsonl", "cells")
    if not T: return
    md("## Layer C - controlled component comparison (common algorithm, ONE component changed)", "",
       "Cells: filter F0 = nominal MVDR, F1 = error-aware MVDR; constraint S = SINR floor eta(beta) = 9.80 dB on nominal rows, "
       "B = same floor with bounded-ball rows (Hu's worst-case protection), P = P_D certificate (compound-Gaussian clutter + Gaussian error law, proposed), "
       "PG = P_D certificate with Gaussian clutter. Everything else identical (power step, WMMSE relay, rate model, stopping, final repair).", "")
    res = {}
    for e in (1.52e-3, 1e-2):
        data = {c: series(T, e, "cell", c) for c in CELLS}
        if not all(data.values()): continue
        ss = sorted(set.intersection(*[set(v) for v in data.values()]))
        if len(ss) < 5: continue
        res[e] = (data, ss)
        md(f"**eps = {e:g}, n = {len(ss)} paired channels**", "",
           "| cell | % protected [CP] | protected SAR [CI] | raw SAR | mean worst P_D | total ST power (mW) | mean max radar ST interference / sigma_r^2 | diff vs F1-P [CI] |", "|---|---|---|---|---|---|---|---|")
        ref = np.array([psar(data["F1-P"][s]) for s in ss])
        for c in CELLS:
            rr = [data[c][s] for s in ss]; z = np.array([psar(r) for r in rr]); kp = sum(r["prot"] for r in rr); lo, hi = cp(kp, len(rr))
            md(f"| {c} | {100*kp/len(rr):.0f}% [{100*lo:.0f}, {100*hi:.0f}] | {ci_s(*boot_mean(z))} | {np.mean([r['R'] for r in rr]):.2f} | {np.mean([r['minpd'] for r in rr]):.3f} | "
               f"{np.mean([r['Psum_mW'] for r in rr]):.0f} | {np.mean([r['radar_intf_max'] for r in rr]):.2f} | " + ("-" if c == "F1-P" else ci_s(*boot_mean(z - ref), f="{:+.2f}")) + " |")
        g = {c: np.array([psar(data[c][s]) for s in ss]) for c in CELLS}
        eff = {"filter effect within SINR floor (F1-S - F0-S)": g["F1-S"] - g["F0-S"],
               "filter effect within bounded-ball floor (F1-B - F0-B)": g["F1-B"] - g["F0-B"],
               "filter effect within P_D certificate (F1-P - F0-P)": g["F1-P"] - g["F0-P"],
               "constraint effect, nominal filter (F0-P - F0-S)": g["F0-P"] - g["F0-S"],
               "constraint effect, aware filter (F1-P - F1-S)": g["F1-P"] - g["F1-S"],
               "interaction ((F1-P - F0-P) - (F1-S - F0-S))": (g["F1-P"] - g["F0-P"]) - (g["F1-S"] - g["F0-S"]),
               "P_D certificate vs bounded-ball floor, aware filter (F1-P - F1-B)": g["F1-P"] - g["F1-B"],
               "clutter model inside the certificate (F1-P - F1-PG)": g["F1-P"] - g["F1-PG"]}
        md("", "Effects on protected SAR (bit/s/Hz, mean [95% paired bootstrap CI]):")
        for k, v in eff.items(): md(f"- {k}: {ci_s(*boot_mean(v), f='{:+.2f}')}")
        md("")
    if not res: return
    # MATLAB-style grouped bars: protected SAR (eps = 1.5e-3 and 1e-2) with % protected on top
    f, ax = PS.fig(PS.COL_W * 1.05, 2.7)
    x = np.arange(len(CELLS)); w = 0.38
    for j, (e, colr) in enumerate(((1.52e-3, PS.C["blue"]), (1e-2, PS.C["orange"]))):
        if e not in res: continue
        data, ss = res[e]
        z = [boot_mean([psar(data[c][s]) for s in ss]) for c in CELLS]
        pr = [np.mean([data[c][s]["prot"] for s in ss]) for c in CELLS]
        xx = x + (j - 0.5) * w
        ax.bar(xx, [t[0] for t in z], w, color=colr, edgecolor="black", lw=0.5, label=rf"$\epsilon={e:g}$",
               yerr=err(z), error_kw=dict(elinewidth=0.6, capsize=1.6, capthick=0.6), zorder=3)
        for xi, t, p in zip(xx, z, pr):
            ax.text(xi, t[2] + 0.15, f"{100*p:.0f}", ha="center", fontsize=5.2, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(CELLS, fontsize=7)
    ax.set_ylabel("Protected SAR (bps/Hz)"); ax.set_xlabel("Receive filter / protection constraint")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12); ax.legend(loc="upper left", fontsize=6.5)
    ax.text(0.99, 0.97, "numbers on bars: % protected", transform=ax.transAxes, ha="right", va="top", fontsize=5.6)
    PS.save(f, F("fig07_component_cells"))


# =================================================================================================== mechanism
def section_M(D):
    e = 1e-2
    md("## Mechanism (eps = 0.01): power allocation, radar interference, detection margins", "",
       "| method | total ST power (mW) | users switched off (p < 1 mW), mean count | mean max radar ST interference / sigma_r^2 | mean P_D margin (worst P_D - 0.9) | per-error-draw protection | relay power used / P^max |",
       "|---|---|---|---|---|---|---|")
    ms = [m for m in ("prop", "hu_rob", "hu_rob_T", "hu_nom", "hu_nom_T", "adapt_rob") if D.get((e, m))]
    for m in ms:
        rr = list(D[(e, m)].values())
        off = np.mean([np.sum(np.array(r["p"]) < 1e-3) for r in rr])
        md(f"| {LAB[m]} | {np.mean([r['Psum_mW'] for r in rr]):.0f} | {off:.2f} | {np.mean([r['radar_intf_max'] for r in rr]):.2f} | "
           f"{np.mean([r['minpd'] - 0.9 for r in rr]):+.3f} | {np.mean([r['prot_real'] for r in rr]):.2f} | {np.mean([r['Pt'] for r in rr]):.2f} |")
    md("")
    # scatter: admitted ST power vs worst-target P_D (one point per channel), Hu-style markers
    f, ax = PS.fig(PS.COL_W, 2.8)
    for m in ms:
        c, mk, ls, lab = PS.METHOD[m]
        rr = list(D[(e, m)].values())
        ax.plot([r["Psum_mW"] for r in rr], [r["minpd"] for r in rr], ls="none", marker=mk, mfc="white" if mk not in "x*" else c, mec=c,
                ms=4 if mk != "*" else 6, mew=0.7, label=lab, alpha=0.9)
    ax.axhline(0.9, color="black", lw=0.6, ls=(0, (6, 3)))
    ax.set_xlabel(r"Total admitted ST power $\sum_m p_m$ (mW)"); ax.set_ylabel(r"Worst-target $P_D$")
    ax.legend(loc="lower left", fontsize=5.4, markerscale=1.0)
    PS.save(f, F("fig08_mechanism_power_vs_pd"))


# =================================================================================================== heavy tail
def section_N(D):
    T = table("b_nu.jsonl", "nu")
    if not T: return
    import b_campaign as b
    md("## Heavy-tail sensitivity (eps = 0.01, mean clutter power fixed: share 0.5 of sigma_r^2)", "",
       "Proposed is re-designed for each nu (its law uses nu); the SINR-based designs do not depend on nu, so the SAME designs are re-evaluated "
       "under each nu by the common evaluator (tuned baselines keep the threshold tuned at nu = 0.7).", "")
    re_path = os.path.join(runner.DATA, "b_nu_reeval.json")
    cache = json.load(open(re_path)) if os.path.exists(re_path) else {}
    nus = sorted({k[4] for k in T} | {0.7}); e = 1e-2
    res = defaultdict(dict)
    for nu in nus:
        res["prop"][nu] = series(T, e, "prop", 0.9, nu=nu) if nu != 0.7 else D.get((e, "prop"), {})
    seeds = sorted(set.intersection(*[set(v) for v in res["prop"].values() if v]))
    for m in ("hu_rob", "hu_rob_T", "hu_nom", "hu_nom_T", "adapt_rob"):
        base = D.get((e, m), {})
        if not base: continue
        for nu in nus:
            if nu == 0.7: res[m][nu] = base; continue
            out = {}
            for s in seeds:
                if s not in base: continue
                key = f"{m}|{s}|{nu}"
                if key not in cache:
                    r = base[s]; d = np.load(os.path.join(b.DES, r["design_tag"] + ".npz"))
                    ev = b.evaluator(nu)(b.scn(s), d["p"], d["W"], d["U"], e)
                    cache[key] = dict(R=ev["R"], prot=ev["prot"], minpd=ev["minpd"])
                out[s] = cache[key]
            res[m][nu] = out
    json.dump(cache, open(re_path, "w"))
    md("| method | " + " | ".join(f"nu={nu:g}" for nu in nus) + " |", "|---|" + "---|" * len(nus))
    curves = {}
    for m in res:
        cells, xs, zs, ps = [], [], [], []
        for nu in nus:
            rr = [res[m][nu][s] for s in seeds if s in res[m].get(nu, {})]
            if not rr: cells.append("-"); continue
            z = boot_mean([psar(r) for r in rr]); kp = sum(r["prot"] for r in rr); lo, hi = cp(kp, len(rr))
            cells.append(f"{ci_s(*z)}; {100*kp/len(rr):.0f}%; raw {np.mean([r['R'] for r in rr]):.2f}")
            xs.append(nu); zs.append(z); ps.append((100 * kp / len(rr), 100 * lo, 100 * hi))
        md(f"| {LAB[m]} | " + " | ".join(cells) + " |")
        curves[m] = (xs, zs, ps)
    md(f"(n = {len(seeds)} paired channels; cell = protected SAR [CI]; % protected; raw SAR)", "")
    for fname, idx, ylab in (("fig09a_heavy_tail_rate", 1, "Protected SAR (bps/Hz)"), ("fig09b_heavy_tail_protection", 2, "Protected configurations (%)")):
        f, ax = PS.fig(PS.COL_W, 2.7)
        for m, (xs, zs, ps) in curves.items():
            v = zs if idx == 1 else ps
            PS.line(ax, xs, [t[0] for t in v], key=m, yerr=err(v))
        ax.set_xscale("log"); ax.set_xlabel(r"Texture shape $\nu$ (smaller = spikier)"); ax.set_ylabel(ylab)
        ax.set_xticks([0.3, 0.5, 1, 2, 5, 50]); ax.set_xticklabels(["0.3", "0.5", "1", "2", "5", "50"])
        if idx == 2: ax.axhline(95, color="black", lw=0.6, ls=(0, (6, 3))); ax.set_ylim(-3, 103)
        ax.legend(loc="lower right" if idx == 1 else "center right", fontsize=5.6)
        PS.save(f, F(fname))


# =================================================================================================== P_FA validation
def section_F():
    R = load("e_pfa.jsonl")
    if not R: return
    md("## False-alarm validation (independent brute-force Monte Carlo, 2e6 null draws per target)", "",
       "| design set | eps | nu | targets | empirical P_FA at calibrated threshold: mean [min, max] | share of targets whose 95% CI covers 1e-3 | empirical P_FA at the manuscript's design-law threshold: mean [min, max] | max |P_D(brute force) - P_D(evaluator)| |",
       "|---|---|---|---|---|---|---|---|")
    grp = defaultdict(list)
    for r in R:
        k = r["key"]; grp[(k[3] if k[0] == "main" else "prop (nu sweep)", k[2], k[5])].append(r)
    agg = defaultdict(dict)
    for (m, e, nu), rr in sorted(grp.items(), key=lambda t: (t[0][2], t[0][1], t[0][0])):
        N = rr[0]["N"]
        pc = np.concatenate([np.array(r["pfa_cal"]) / N for r in rr]); pdz = np.concatenate([np.array(r["pfa_des"]) / N for r in rr])
        cov = np.mean([cp(int(round(x * N)), N)[0] <= 1e-3 <= cp(int(round(x * N)), N)[1] for x in pc])
        dpd = max(np.max(np.abs(np.array(r["pd_bf"]) - np.array(r["pd_cmc"]))) for r in rr)
        md(f"| {m} | {fmt_eps(e)} | {nu:g} | {len(pc)} | {pc.mean():.2e} [{pc.min():.2e}, {pc.max():.2e}] | {100*cov:.0f}% | {pdz.mean():.2e} [{pdz.min():.2e}, {pdz.max():.2e}] | {dpd:.4f} |")
        if nu == 0.7 and m in ("prop", "hu_rob", "hu_nom", "adapt_rob"): agg[m][e] = (pc, pdz)
    md("")
    # per-target spread matters more than the mean: median with 5-95 % range over all targets of all methods
    f, ax = PS.fig(PS.COL_W, 2.7)
    eps_list = sorted({e for m in agg for e in agg[m]})
    X = {e: i for i, e in enumerate(eps_list)}
    for j, (lab, c, mk, off) in enumerate(((r"common evaluator: threshold calibrated on true null", PS.C["blue"], "o", -0.08),
                                           (r"manuscript design-law threshold (eq. (11))", PS.C["red"], "s", 0.08))):
        xs, med, lo, hi = [], [], [], []
        for e in eps_list:
            v = 1e3 * np.concatenate([agg[m][e][j] for m in agg if e in agg[m]])
            xs.append(X[e] + off); med.append(np.median(v)); lo.append(np.percentile(v, 5)); hi.append(np.percentile(v, 95))
        PS.line(ax, xs, med, color=c, marker=mk, ls="none", label=lab + " (median, 5–95% of targets)",
                yerr=[[m - l for m, l in zip(med, lo)], [h - m for m, h in zip(med, hi)]])
    ax.axhline(1.0, color="black", lw=0.6, ls=(0, (6, 3)))
    ax.set_yscale("log"); ax.set_ylim(0.15, 6)
    ax.set_yticks([0.2, 0.5, 1, 2, 5]); ax.set_yticklabels(["0.2", "0.5", "1", "2", "5"])
    from matplotlib.ticker import NullFormatter
    ax.yaxis.set_minor_formatter(NullFormatter())
    eps_axis(ax, eps_list); ax.set_ylabel(r"Measured $P_{FA}$ per target ($\times 10^{-3}$)")
    ax.legend(loc="upper left", fontsize=5.4)
    PS.save(f, F("fig10_pfa_validation"))


# =================================================================================================== cost / scaling
def section_S(D):
    md("## Computational cost", "")
    if D:
        e = 1e-2
        md("**Default size (8,8,4,4), eps = 0.01, main campaign (9 parallel single-thread workers on a 12-core laptop CPU; per-design wall time, includes contention)**", "",
           "| method | mean time (s) | median | iterations (mean) | status not ok |", "|---|---|---|---|---|")
        for m in ("prop", "hu_nom", "hu_rob", "adapt_rob"):
            rr = list(D.get((e, m), {}).values())
            if not rr: continue
            t = np.array([r["time"] for r in rr])
            md(f"| {LAB[m]} | {t.mean():.1f} | {np.median(t):.1f} | {np.mean([r['iters'] for r in rr]):.1f} | {sum(r['status'] not in ('ok', 'converged') for r in rr)} |")
        md("")
    last = {}
    for r in runner.load("b_scale.jsonl"):                 # latest record per task: a retry replaces an earlier failure
        k = runner._tup(r["key"])
        if k not in last or "error" in last[k]: last[k] = r
    raw = list(last.values())
    if not raw: return
    md("**Scaling (10 channels per size, eps = 0.01; 4 workers; Hu methods built with cvxpy's pure-Python backend, see report)**", "",
       "\"converged\" = the method's own stopping rule was met (Hu robust: no new cutting-set sample within 15 outer iterations); "
       "a design that stopped at the iteration cap is still a valid, evaluated design.", "",
       "| (Nr, Nt, M, K) | method | mean time (s) [95% CI] | converged | failed runs | protected % | mean SAR |", "|---|---|---|---|---|---|---|")
    geos = sorted({tuple(r["key"][6]) for r in raw}, key=lambda g: (g[2], g[3], g[0]))
    f, ax = PS.fig(PS.COL_W, 2.7)
    for m in ("prop", "hu_nom", "hu_rob"):
        xs, ys = [], []
        for i, gq in enumerate(geos):
            rr = [r for r in raw if r["key"][3] == m and tuple(r["key"][6]) == gq]
            ok = [r for r in rr if "error" not in r]
            if not rr: continue
            t = boot_mean([r["time"] for r in ok]) if ok else (np.nan,) * 3
            succ = sum(r.get("status") in ("ok", "converged") for r in ok)
            md(f"| {gq} | {LAB[m]} | {ci_s(*t, f='{:.1f}')} | {succ}/{len(rr)} | {len(rr) - len(ok)} | {100*np.mean([r['prot'] for r in ok]) if ok else 0:.0f}% | {np.mean([r['R'] for r in ok]) if ok else 0:.2f} |")
            xs.append(i); ys.append(t)
        PS.line(ax, xs, [t[0] for t in ys], key=m, ls="none", yerr=err(ys))   # separate configurations: no connecting lines
    ax.set_xticks(range(len(geos))); ax.set_xticklabels([f"({g[0]},{g[2]},{g[3]})" for g in geos], fontsize=6.5)
    ax.set_yscale("log"); ax.set_xlabel(r"$(N_r, M, K)$, $N_t=8$"); ax.set_ylabel("Run time per design (s)")
    ax.legend(loc="upper left", fontsize=5.6)
    PS.save(f, F("fig11_cost_scaling"))
    md("")


# =================================================================================================== small-system reference + NLP
def section_R():
    R = load("d_smallref.jsonl")
    if R:
        md("## Small-system reference (Nt = Nr = 4, M = K = 2, eps = 0.01): exhaustive 41x41 power grid vs Algorithm 1 vs direct NLP", "",
           "| quantity | grid search | Algorithm 1 (proposed) | direct NLP |", "|---|---|---|---|")
        for fl, lab in (("R_model", "design-model SAR (bit/s/Hz)"), ("R", "evaluated ergodic SAR"), ("minpd", "worst-target P_D"), ("time", "time (s)")):
            md(f"| {lab} | " + " | ".join(ci_s(*boot_mean([r[k][fl] for r in R])) for k in ("grid", "alg1", "nlp")) + " |")
        gap = np.array([r["grid"]["R_model"] - r["alg1"]["R_model"] for r in R]); gapn = np.array([r["grid"]["R_model"] - r["nlp"]["R_model"] for r in R])
        md(f"| gap to grid optimum (model SAR) | - | {ci_s(*boot_mean(gap), f='{:+.3f}')}; within 0.05: {np.mean(gap < 0.05)*100:.0f}% | {ci_s(*boot_mean(gapn), f='{:+.3f}')}; within 0.05: {np.mean(gapn < 0.05)*100:.0f}% |",
           f"| protected (common evaluator) | {np.mean([r['grid']['prot'] for r in R])*100:.0f}% | {np.mean([r['alg1']['prot'] for r in R])*100:.0f}% | {np.mean([r['nlp']['prot'] for r in R])*100:.0f}% |",
           "", f"n = {len(R)} channels. Grid resolution 2.5 mW per ST; at every grid point W comes from 40 WMMSE iterations (local), so the grid is a reference over p only.", "")
    N = table("b_nlp.jsonl", "nlp")
    if N:
        D, _ = collect_main()
        ss = sorted(s for (s, e, m, p, n) in N if s in D.get((1e-2, "prop"), {}))
        if ss:
            a = [psar(N[(s, 1e-2, "nlp", 0.9, 0.7)]) for s in ss]; pr = [psar(D[(1e-2, "prop")][s]) for s in ss]
            md("## Alternative solver for the proposed formulation (eps = 0.01)", "",
               "| solver | protected SAR [CI] | % protected | raw SAR | mean time (s) |", "|---|---|---|---|---|")
            md(f"| Algorithm 1 (certificate rows + SQP) | {ci_s(*boot_mean(pr))} | {100*np.mean([D[(1e-2,'prop')][s]['prot'] for s in ss]):.0f}% | {np.mean([D[(1e-2,'prop')][s]['R'] for s in ss]):.2f} | {np.mean([D[(1e-2,'prop')][s]['time'] for s in ss]):.1f} |")
            md(f"| direct NLP (SLSQP with nonlinear P_D constraints) | {ci_s(*boot_mean(a))} | {100*np.mean([N[(s,1e-2,'nlp',0.9,0.7)]['prot'] for s in ss]):.0f}% | {np.mean([N[(s,1e-2,'nlp',0.9,0.7)]['R'] for s in ss]):.2f} | {np.mean([N[(s,1e-2,'nlp',0.9,0.7)]['time'] for s in ss]):.1f} |")
            md("", f"Paired difference (Algorithm 1 - NLP), n = {len(ss)}: {ci_s(*boot_mean(np.array(pr) - np.array(a)), f='{:+.2f}')}", "")


if __name__ == "__main__":
    want = set(sys.argv[1:]) or set("ATBXCMNFSR")
    md("# Version_4 research results (auto-generated by analyze.py)", "")
    if "A" in want: section_A()
    if "T" in want: section_T()
    D = None
    if want & set("BXMNS"):
        out = section_B()
        if out: D, _ = out
    if "X" in want and D: section_X(D)
    if "C" in want: section_C()
    if "M" in want and D: section_M(D)
    if "N" in want and D: section_N(D)
    if "F" in want: section_F()
    if "S" in want: section_S(D)
    if "R" in want: section_R()
    open(OUTMD, "w", encoding="utf8").write("\n".join(MD) + "\n")
    print("\n".join(MD))
