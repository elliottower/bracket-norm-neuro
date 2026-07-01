"""V5: Just the data. Gray lines, jittered dots, nothing else.

The figure's argument is visual: 32/32 lines go up. Stats in caption.
"""
import json
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
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "savefig.dpi": 400,
    "figure.dpi": 150,
})

from pathlib import Path
RESULTS = Path(__file__).resolve().parent.parent / "results"
with open(RESULTS / "svoboda_bracket_norm_v5_20260626.json") as f:
    data = json.load(f)

pairs = data["paired_sessions"]
session_results = data.get("session_results", [])
n_neurons_map = {}
for sr in session_results:
    if isinstance(sr, dict) and "file" in sr:
        nn = sr.get("n_neurons") or sr.get("n_units")
        if nn:
            n_neurons_map[sr["file"]] = nn

ctrl = np.array([p["control"] for p in pairs]) / np.sqrt(
    [n_neurons_map.get(p["file"], 100) for p in pairs])
stim = np.array([p["photostim"] for p in pairs]) / np.sqrt(
    [n_neurons_map.get(p["file"], 100) for p in pairs])

rng = np.random.default_rng(7)
jx0 = rng.uniform(-0.06, 0.06, len(ctrl))
jx1 = rng.uniform(-0.06, 0.06, len(stim))

fig, ax = plt.subplots(figsize=(3.2, 2.8))

for i in range(len(ctrl)):
    ax.plot([0 + jx0[i], 1 + jx1[i]], [ctrl[i], stim[i]],
            color='#bbbbbb', linewidth=0.7, zorder=1)

ax.scatter(0 + jx0, ctrl, color='#1f77b4', s=20, zorder=3,
           edgecolors='white', linewidth=0.3)
ax.scatter(1 + jx1, stim, color='#d62728', s=20, zorder=3,
           edgecolors='white', linewidth=0.3)

ax.set_xticks([0, 1])
ax.set_xticklabels(["control", "photostim"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("BN/$\\sqrt{n}$")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

_out = Path(__file__).resolve().parent
fig.savefig(str(_out / "fig_photostim_slopegraph.png"), bbox_inches="tight", pad_inches=0.08)
fig.savefig(str(_out / "fig_photostim_slopegraph.pdf"), bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print(f"Saved v5: {len(pairs)} pairs, ctrl={ctrl.mean():.3f}, stim={stim.mean():.3f}")
