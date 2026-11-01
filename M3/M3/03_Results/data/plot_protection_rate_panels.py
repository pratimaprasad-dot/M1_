"""
Fig. 3: protection-rate trade-off at CSI error eps = 0.01 (40 paired test channels), one panel per method, drawn in the
same style and at the same size as Fig. 2 (plot_protected_sar_vs_eps.py) so both figures sit level in the paper.
Each point is one design setting: x = % of channels on which every target has P_D >= 0.9 (exact 95 % Clopper-Pearson
interval), y = average sum rate (95 % bootstrap interval). Dashed vertical line: 95 % protection requirement.
Reads tradeoff.csv and main_comparison.csv from this folder; writes protection_rate_panels.pdf (vector) and .png.
Needs: numpy, scipy, matplotlib.   Usage: python plot_protection_rate_panels.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
EPS, REF = 0.01, 95.0
FIG_W, FIG_H = 4.47, 3.51          # same canvas as Fig. 2, so the two figures have equal height at column width

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 7.6, "axes.linewidth": 0.6, "axes.edgecolor": "#262626",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.color": "#262626", "ytick.color": "#262626",
    "legend.frameon": True, "legend.edgecolor": "#262626", "legend.fancybox": False, "legend.framealpha": 1.0,
    "legend.borderpad": 0.35, "legend.labelspacing": 0.25, "legend.handlelength": 2.2, "legend.handletextpad": 0.5,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# same colours and markers as Fig. 2; the legend carries the citation of the compared designs
METHODS = [
    ("prop",      "#0072BD", "s", True,  "Proposed",                    r"$\beta$={:g}"),
    ("hu_rob",    "#D95319", "^", False, "Hu et al. [3], robust",       "{:.0f} dB"),
    ("hu_nom",    "#7E2F8E", "o", False, "Hu et al. [3], non-robust",   "{:.0f} dB"),
    ("adapt_rob", "#77AC30", "p", False, "Adapted worst-case",          "{:.2g} dB"),
]


def load():
    pts = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(os.path.join(HERE, "tradeoff.csv"))):
        pts[r["method"]][float(r["param"])][int(r["seed"])] = (float(r["R"]), int(r["protected"]))
    for r in csv.DictReader(open(os.path.join(HERE, "main_comparison.csv"))):
        if abs(float(r["eps"]) - EPS) < 1e-12:
            pts[r["method"]][float(r["param"])][int(r["seed"])] = (float(r["R"]), int(r["protected"]))
    seeds = sorted(set.intersection(*[set(v) for m in pts for v in pts[m].values()]))
    return pts, seeds


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def cp(k, n):
    return (stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def main():
    pts, seeds = load()
    f, axs = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), sharex=True, sharey=True,
                          gridspec_kw=dict(hspace=0.12, wspace=0.08))
    for ax, (m, colr, mk, filled, name, fmt) in zip(axs.flat, METHODS):
        rows = []
        for p in sorted(pts[m]):
            rr = [pts[m][p][s] for s in seeds]; k = sum(q for _, q in rr); lo, hi = cp(k, len(rr))
            rows.append((p, 100 * k / len(rr), 100 * lo, 100 * hi, *boot([v for v, _ in rr])))
        rows.sort(key=lambda t: (t[1], -t[4]))
        x = [t[1] for t in rows]; y = [t[4] for t in rows]
        ax.errorbar(x, y, xerr=[[t[1] - t[2] for t in rows], [t[3] - t[1] for t in rows]],
                    yerr=[[t[4] - t[5] for t in rows], [t[6] - t[4] for t in rows]], color=colr, lw=1.1,
                    ls="--" if m == "hu_nom" else "-", marker=mk, ms=3.6 if filled else 4.5, mfc=colr if filled else "white", mec=colr,
                    mew=0.9, elinewidth=0.5, capsize=1.4, capthick=0.5, zorder=3)
        ax.axvline(REF, color="#262626", lw=0.7, ls=(0, (4, 2)), zorder=2)
        # label the best setting that meets the requirement
        ok = [t for t in rows if t[1] >= REF and t[4] > 0]
        if ok:
            b = max(ok, key=lambda t: t[4])
            ax.annotate(fmt.format(b[0]), (b[1], b[4]), textcoords="offset points", xytext=(-9, 6),
                        fontsize=7.5, color=colr, ha="right", va="bottom", bbox=dict(fc="white", ec="none", pad=0.5), zorder=6)
        ax.text(0.04, 0.06, name, transform=ax.transAxes, fontsize=7.8, color=colr, ha="left", va="bottom",
                fontweight="bold")
        ax.set_xlim(-3, 105); ax.set_ylim(-0.4, 10)
        ax.xaxis.set_major_locator(MultipleLocator(25)); ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(2)); ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, which="major", color="#D0D0D0", lw=0.5); ax.set_axisbelow(True)
    f.supxlabel(r"Channels with $P_{D,k}\geq0.9$ for all $k$ (%)", fontsize=10, y=0.0)
    f.supylabel("Average SAR (bit/s/Hz)", fontsize=10, x=0.02)
    for ext in ("pdf", "png"):
        f.savefig(os.path.join(HERE, f"protection_rate_panels.{ext}"))
    print("wrote protection_rate_panels.pdf / .png")


if __name__ == "__main__":
    main()
