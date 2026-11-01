"""
Figure style matched to the base paper's published figures (Hu et al., IEEE TCCN 2026; MATLAB R2022a look):
serif (Times) text with STIX math, full black box, inward ticks, dashed grey grid at the major ticks, MATLAB 'lines'
colours, hollow markers, solid lines for robust/proposed designs and dashed lines for non-robust baselines,
framed legend inside the axes, IEEE single-column size (3.5 in wide), optional zoom inset (as in Hu Figs. 3-4).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# MATLAB default 'lines' colour order + the extra tones visible in Hu's figures
C = dict(blue="#0072BD", orange="#D95319", yellow="#EDB120", purple="#7E2F8E", green="#77AC30", cyan="#4DBEEE",
         maroon="#A2142F", brown="#8C564B", red="#D62728", teal="#009E73", lightblue="#7B9CFF", tan="#C49A6C", grey="#555555")

# method -> (colour, marker, linestyle, legend label)  -- same identity in every figure
METHOD = {
    "prop":      (C["blue"],   "*", "-",  r"Proposed ($P_D$-certified)"),
    "hu_rob":    (C["maroon"], "s", "-",  r"Hu robust, $\eta_{\min}=10$ dB"),
    "hu_rob_T":  (C["orange"], "x", "-",  r"Hu robust, tuned $\eta_{\min}$"),
    "hu_nom":    (C["red"],    "<", "--", r"Hu non-robust, $\eta_{\min}=10$ dB"),
    "hu_nom_T":  (C["tan"],    "v", "--", r"Hu non-robust, tuned $\eta_{\min}$"),
    "adapt_rob": (C["green"],  "d", "-",  r"Adapted worst-case"),
    "perfect":   (C["grey"],   "o", ":",  r"Perfect CSI"),
}

COL_W, COL_H = 3.5, 2.75          # IEEE single column (inches)


def apply():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 6.8,
        "axes.edgecolor": "black", "axes.linewidth": 0.6, "axes.labelcolor": "black",
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.color": "black", "ytick.color": "black",
        "axes.grid": True, "grid.linestyle": (0, (4, 3)), "grid.linewidth": 0.45, "grid.color": "#b0b0b0", "grid.alpha": 1.0,
        "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False, "legend.framealpha": 1.0,
        "legend.borderpad": 0.35, "legend.handlelength": 2.4, "legend.labelspacing": 0.25, "patch.linewidth": 0.5,
        "lines.linewidth": 1.1, "lines.markersize": 5, "lines.markeredgewidth": 0.9,
        "errorbar.capsize": 2, "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def fig(w=COL_W, h=COL_H, ncols=1):
    f, ax = plt.subplots(1, ncols, figsize=(w * ncols, h))
    return f, ax


def line(ax, x, y, key=None, color=None, marker=None, ls=None, label=None, yerr=None, xerr=None, filled=False, ms=None, **kw):
    """MATLAB-like curve: hollow markers, optional 95% CI as thin caps"""
    if key is not None:
        c, m, l, lab = METHOD[key]
        color = color or c; marker = marker or m; ls = ls or l; label = label if label is not None else lab
    mfc = color if filled or marker in ("x", "+", "*") else "white"
    if marker == "*": ms = ms or 7
    if yerr is not None or xerr is not None:
        return ax.errorbar(x, y, yerr=yerr, xerr=xerr, color=color, marker=marker, ls=ls, label=label, mfc=mfc, mec=color,
                           ms=ms, elinewidth=0.6, capsize=1.8, capthick=0.6, **kw)
    return ax.plot(x, y, color=color, marker=marker, ls=ls, label=label, mfc=mfc, mec=color, ms=ms, **kw)


def zoom(ax, bounds, xlim, ylim, loc_corners=(2, 4)):
    """Hu-style zoom inset: bounds = [x0, y0, w, h] in axes fraction; draws the inset frame and connector lines"""
    axins = ax.inset_axes(bounds)
    for l in ax.get_lines():
        axins.plot(*l.get_data(), color=l.get_color(), ls=l.get_linestyle(), marker=l.get_marker(), mfc=l.get_mfc(),
                   mec=l.get_mec(), ms=l.get_markersize(), lw=l.get_linewidth())
    axins.set_xlim(*xlim); axins.set_ylim(*ylim)
    axins.tick_params(labelsize=6); axins.grid(True)
    mark_inset(ax, axins, loc1=loc_corners[0], loc2=loc_corners[1], fc="none", ec="black", lw=0.5, ls=":")
    return axins


def save(f, path):
    f.savefig(path + ".png"); f.savefig(path + ".pdf")
    plt.close(f)
