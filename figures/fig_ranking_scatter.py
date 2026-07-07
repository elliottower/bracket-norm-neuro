"""V4: Green separation band from v2, clean style from v3. No title, no dual names."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 400,
    "figure.dpi": 150,
})

regions = ["ORB",  "PL",   "VISpm", "MOs",  "RSP",   "VISl",  "VISam", "ACA",  "VISp"]
bn =      [0.188,  0.172,  0.163,   0.145,  0.137,   0.125,   0.092,   0.073,  0.067]
silencing=[0.309,  0.333,  0.225,   0.153,  0.142,   0.172,   0.082,   0.145,  0.141]

bn = np.array(bn)
silencing = np.array(silencing)

top3 = [0, 1, 2]
bot3 = [6, 7, 8]
mid = [3, 4, 5]

fig, ax = plt.subplots(figsize=(4.5, 3.5))

gap_top = min(bn[top3])
gap_bot = max(bn[bot3])
ax.axvspan(gap_bot, gap_top, alpha=0.08, color='green', zorder=0)

ax.scatter(bn[mid], silencing[mid], color='#888888', s=40, zorder=3,
           edgecolors='#555', linewidth=0.5)
ax.scatter(bn[top3], silencing[top3], color='#d62728', s=55, zorder=4,
           edgecolors='#8b0000', linewidth=0.6, label='top-3')
ax.scatter(bn[bot3], silencing[bot3], color='#1f77b4', s=55, zorder=4,
           edgecolors='#0b3d91', linewidth=0.6, label='bottom-3')

from numpy.polynomial.polynomial import polyfit
c = polyfit(bn, silencing, 1)
x_fit = np.linspace(bn.min() * 0.85, bn.max() * 1.1, 100)
ax.plot(x_fit, c[0] + c[1] * x_fit, '--', color='#999', linewidth=0.9, zorder=1)

offsets = {
    "ORB":   ( 0.004,  0.005),
    "PL":    ( 0.004,  0.000),
    "VISpm": ( 0.004,  0.006),
    "MOs":   ( 0.004,  0.004),
    "RSP":   ( 0.004, -0.014),
    "VISl":  ( 0.005,  0.002),
    "VISam": ( 0.004, -0.012),
    "ACA":   (-0.022, -0.014),
    "VISp":  ( 0.006,  0.008),
}

for i, region in enumerate(regions):
    dx, dy = offsets[region]
    ax.annotate(region, (bn[i], silencing[i]), xytext=(bn[i]+dx, silencing[i]+dy),
                fontsize=8, ha='left', va='bottom')

ax.set_xlabel("BN/$\\sqrt{n}$")
ax.set_ylabel("silencing effect")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=8, loc='upper left', frameon=False, handletextpad=0.3, labelspacing=0.3)

from pathlib import Path
_out = Path(__file__).resolve().parent
fig.savefig(str(_out / "fig_ranking_scatter.png"), bbox_inches="tight", pad_inches=0.08)
fig.savefig(str(_out / "fig_ranking_scatter.pdf"), bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
rho = np.corrcoef(bn, silencing)[0, 1]
print(f"Saved ranking scatter v4 (rho={rho:.3f})")
