"""
Fig. 4: worst-target detection probability P_D^worst = min_k P_{D,k} versus total admitted secondary (ST) power at CSI
error eps = 0.01, one point per channel and scheme (60 test channels), in the same style as Fig. 2.
Full-range plot plus a zoomed inset of the dense band P_D^worst = 0.88-1.00. Reading aids: the region below the
required 0.9 is shaded, three short notes explain the three behaviours, and the legend gives each scheme's protected
sum rate. Legend below the axes.
Reads main_comparison.csv and tuned_baselines.csv from this folder (columns Psum_mW, minPD, R_protected) and writes
power_vs_pd.pdf (vector) and .png (600 dpi).
Needs: numpy, matplotlib.   Usage: python plot_power_vs_pd.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
EPS, BETA = 0.01, 0.9
FAIL = "#FBE3E1"            # shading of the region that violates the requirement
RED = "#B22222"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 8.5, "axes.linewidth": 0.5, "axes.edgecolor": "#262626",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.color": "#262626", "ytick.color": "#262626",
    "legend.frameon": True, "legend.edgecolor": "#262626", "legend.fancybox": False, "legend.framealpha": 1.0,
    "legend.borderpad": 0.4, "legend.labelspacing": 0.3, "legend.handletextpad": 0.3, "legend.columnspacing": 1.0,
    "patch.linewidth": 0.5, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# same colours as Fig. 2; small, distinct markers for all six schemes
BLUE, ORANGE, PURPLE, GREEN = "#0072BD", "#D95319", "#7E2F8E", "#77AC30"
STYLE = [  # key, colour, marker, filled?, size, label
    ("prop",      BLUE,   "s", True,  3.4, r"Proposed ($\beta=0.9$)"),
    ("adapt_rob", GREEN,  "p", False, 3.8, r"Adapted worst-case ($\eta=9.80$ dB)"),
    ("hu_rob",    ORANGE, "^", False, 3.8, r"Hu robust [3], $\eta_{\min}=10$ dB"),
    ("hu_rob_T",  ORANGE, "v", False, 3.8, r"Hu robust [3], tuned $\eta_{\min}$"),
    ("hu_nom",    PURPLE, "o", False, 3.6, r"Hu non-robust [3], $\eta_{\min}=10$ dB"),
    ("hu_nom_T",  PURPLE, "d", False, 3.8, r"Hu non-robust [3], tuned $\eta_{\min}$"),
]
ZORDER = {"prop": 6, "adapt_rob": 5}


def load():
    d = defaultdict(list)                   # method -> [(power mW, worst-target P_D, protected rate)]
    for fname, suffix in (("main_comparison.csv", ""), ("tuned_baselines.csv", "_T")):
        for r in csv.DictReader(open(os.path.join(HERE, fname))):
            if abs(float(r["eps"]) - EPS) < 1e-12:
                d[r["method"] + suffix].append((float(r["Psum_mW"]), float(r["minPD"]), float(r["R_protected"])))
    return d


def scatter(ax, d, scale=1.0, labels=False):
    for m, c, mk, filled, ms, lab in STYLE:
        rate = np.mean([v[2] for v in d[m]])
        ax.plot([v[0] for v in d[m]], [v[1] for v in d[m]], ls="none", marker=mk, ms=ms * scale,
                mfc=c if filled else "white", mec=c, mew=0.7, zorder=ZORDER.get(m, 3),
                label=f"{lab}: {rate:.2f} bit/s/Hz" if labels else None)
    ax.axhspan(-1, BETA, color=FAIL, lw=0, zorder=0)
    ax.axhline(BETA, color="#262626", lw=0.8, ls=(0, (5, 3)), zorder=2)


def main():
    d = load()
    f, ax = plt.subplots(figsize=(5.2, 3.9))
    scatter(ax, d, labels=True)
    ax.text(415, BETA - 0.015, r"Required $P_D^{\mathrm{worst}}=0.9$", fontsize=8.5, ha="right", va="top")
    ax.set_xlim(0, 420); ax.set_ylim(0, 1.13); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.xaxis.set_major_locator(MultipleLocator(100)); ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, which="major", color="#DCDCDC", lw=0.45); ax.set_axisbelow(True)
    ax.set_xlabel(r"Total admitted ST power, $\sum_m p_m$ (mW)")
    ax.set_ylabel(r"Worst-target detection probability, $P_D^{\mathrm{worst}}$")

    # three plain-language notes
    ax.annotate("", xy=(8, 1.035), xytext=(232, 1.035),
                arrowprops=dict(arrowstyle="|-|", color="#555555", lw=0.7, mutation_scale=3))
    ax.text(120, 1.05, r"Over-protected: $P_D\approx1$, little power $\rightarrow$ low rate", fontsize=8,
            color="#555555", ha="center", va="bottom")
    ax.annotate("Proposed: just above 0.9,\nmore power $\\rightarrow$ highest rate", xy=(195, 0.912),
                xytext=(30, 0.80), fontsize=8.5, color=BLUE, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    nom = np.array([v[1] for v in d["hu_nom"]])
    ax.annotate(f"Hu non-robust (10 dB): {100 * np.mean(nom < BETA):.0f}% of\nchannels below 0.9 (radar fails)",
                xy=(296, 0.66), xytext=(30, 0.67), fontsize=8.5, color=PURPLE, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.8))

    # zoomed inset of the dense band near the requirement
    ins = ax.inset_axes([0.07, 0.06, 0.56, 0.45])
    scatter(ins, d, scale=1.1)
    ins.set_xlim(0, 420); ins.set_ylim(0.88, 1.005)
    ins.xaxis.set_major_locator(MultipleLocator(100)); ins.yaxis.set_major_locator(MultipleLocator(0.04))
    ins.tick_params(labelsize=7.5, direction="in", top=True, right=True, length=2.5)
    ins.grid(True, which="major", color="#DCDCDC", lw=0.4); ins.set_axisbelow(True)
    ins.set_facecolor("white")
    ins.text(0.03, 0.06, r"zoom: $P_D^{\mathrm{worst}}\in[0.88,1]$", transform=ins.transAxes, fontsize=7.5,
             ha="left", va="bottom", bbox=dict(fc="white", ec="none", pad=0.5))

    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, markerscale=1.3,
              title="Protected sum rate at $\\epsilon=0.01$", title_fontsize=8.5).get_frame().set_linewidth(0.5)
    for ext in ("pdf", "png"):
        f.savefig(os.path.join(HERE, f"power_vs_pd.{ext}"))
    print("wrote power_vs_pd.pdf / .png")


if __name__ == "__main__":
    main()
