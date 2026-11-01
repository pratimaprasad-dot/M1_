"""
Protected secondary sum rate versus CSI error ratio epsilon (60 paired test channels), drawn the way a MATLAB
simulation figure looks in an IEEE paper: MATLAB default colour order, 1.2-pt lines, hollow markers for the baselines
and filled markers for the proposed design, light major grid plus dotted minor grid, thin box and legend frame.
The x-axis shows the six simulated error levels at equal spacing, starting at epsilon = 0 (perfect CSI).
Hu robust coincides with Hu non-robust at epsilon = 0 (no error ball) and the tuned floors exist only for epsilon > 0,
so those curves start at the first non-zero level.
Reads main_comparison.csv and tuned_baselines.csv (one row per channel x method x eps) from this folder and writes
protected_sar_vs_eps.png (600 dpi) and .pdf. Each point is the mean protected rate over the 60 channels; set
SHOW_CI = True to add 95 % bootstrap confidence intervals.
Needs: numpy, matplotlib.   Usage: python plot_protected_sar_vs_eps.py
"""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
SHOW_CI = False

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 8.5, "axes.linewidth": 0.5, "axes.edgecolor": "#262626",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.color": "#262626", "ytick.color": "#262626",
    "legend.frameon": True, "legend.edgecolor": "#262626", "legend.fancybox": False, "legend.framealpha": 1.0,
    "legend.borderpad": 0.45, "legend.labelspacing": 0.3, "legend.handlelength": 2.6, "legend.handletextpad": 0.6,
    "patch.linewidth": 0.5, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# ---- data: per-channel protected rate (rate if every target has P_D >= 0.9, else 0)
data = defaultdict(list)                                  # (method, eps) -> R_protected of every channel
for fname, suffix in (("main_comparison.csv", ""), ("tuned_baselines.csv", "_T")):
    for r in csv.DictReader(open(os.path.join(HERE, fname))):
        data[(r["method"] + suffix, float(r["eps"]))].append(float(r["R_protected"]))
EPS = sorted({e for _, e in data})                        # 0, 5.48e-4, 1.52e-3, 3e-3, 1e-2, 3e-2
X = {e: i for i, e in enumerate(EPS)}                     # equal spacing, as in the simulation sweep


def boot(x, n=10000, seed=0):
    x = np.asarray(x); b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


# MATLAB default colour order (R2014b+); one colour per method family, dashed = tuned variant
BLUE, ORANGE, PURPLE, GREEN = "#0072BD", "#D95319", "#7E2F8E", "#77AC30"
STYLE = [  # key, colour, line style, marker, marker size, filled?, label
    ("prop",      BLUE,   "-",  "s", 5.5, True,  "Proposed"),
    ("hu_rob",    ORANGE, "-",  "^", 6.0, False, r"Hu robust [3], $\eta_{\min}=10$ dB"),
    ("hu_rob_T",  ORANGE, "--", "v", 6.0, False, r"Hu robust [3], tuned $\eta_{\min}$"),
    ("hu_nom",    PURPLE, "-",  "o", 5.5, False, r"Hu non-robust [3], $\eta_{\min}=10$ dB"),
    ("hu_nom_T",  PURPLE, "--", "d", 5.5, False, r"Hu non-robust [3], tuned $\eta_{\min}$"),
    ("adapt_rob", GREEN,  "-",  "p", 6.5, False, "Adapted worst-case"),
]

f, ax = plt.subplots(figsize=(5.2, 3.9))
Y = {}
for m, c, ls, mk, ms, filled, lab in STYLE:
    es = [e for e in EPS if (m, e) in data and not (e == 0 and m == "hu_rob")]   # robust = non-robust at eps = 0
    s = [boot(data[(m, e)]) for e in es]
    x = [X[e] for e in es]; y = [t[0] for t in s]; Y[m] = dict(zip(es, y))
    ax.plot(x, y, color=c, ls=ls, lw=1.2, marker=mk, ms=ms, mfc=c if filled else "white", mec=c, mew=1.0,
            label=lab, zorder=4 if m == "prop" else 3, dashes=(5, 2.5) if ls == "--" else (None, None))
    if SHOW_CI:
        ax.errorbar(x, y, yerr=[[t[0] - t[1] for t in s], [t[2] - t[0] for t in s]], fmt="none", ecolor=c,
                    elinewidth=0.6, capsize=2, zorder=2)
ax.axhline(Y["prop"][0.0], color="#4D4D4D", ls=(0, (6, 2, 1.5, 2)), lw=0.9, label=r"Perfect CSI ($\epsilon=0$)", zorder=2)

# axes: six error levels, equally spaced, standard notation
lab = {0.0: "$0$", 5.48e-4: r"$5.5{\times}10^{-4}$", 1.52e-3: r"$1.5{\times}10^{-3}$", 3e-3: r"$3{\times}10^{-3}$",
       1e-2: r"$10^{-2}$", 3e-2: r"$3{\times}10^{-2}$"}
ax.set_xticks(range(len(EPS))); ax.set_xticklabels([lab[e] for e in EPS])
ax.set_xlim(-0.3, len(EPS) - 0.7); ax.set_ylim(0, 12)
ax.xaxis.set_minor_locator(MultipleLocator(0.5)); ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.grid(True, which="major", color="#D0D0D0", lw=0.5, ls="-")          # MATLAB "grid on"
ax.grid(True, which="minor", color="#E2E2E2", lw=0.4, ls=(0, (1, 2)))   # MATLAB "grid minor" (dotted)
ax.set_axisbelow(True)
ax.set_xlabel(r"CSI error ratio, $\epsilon$"); ax.set_ylabel("Protected sum rate (bit/s/Hz)")

# ellipse label, as in standard simulation figures: gain of the proposed design over Hu robust at eps = 0.01
e = 1e-2; xe = X[e]
y_top, y_bot = Y["prop"][e], Y["hu_rob"][e]
ax.add_patch(Ellipse((xe, (y_top + y_bot) / 2), width=0.2, height=y_top - y_bot + 0.8, fill=False, ec="#262626",
                     lw=0.6, zorder=5))
ax.annotate(f"{100 * (y_top / y_bot - 1):.0f}% gain over\nHu robust", xy=(xe + 0.09, y_top + 0.25), xytext=(xe + 0.5, 9.3),
            fontsize=9.5, ha="center", va="bottom", arrowprops=dict(arrowstyle="-", lw=0.6, color="#262626"))

ax.legend(loc="lower left").get_frame().set_linewidth(0.5)
for ext in ("png", "pdf"):
    f.savefig(os.path.join(HERE, f"protected_sar_vs_eps.{ext}"))
print("wrote protected_sar_vs_eps.png / .pdf")
