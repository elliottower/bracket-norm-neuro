"""Generate real-data vector field figure using kernel-smoothed displacements.

At each grid point (evidence, choice-axis projection), computes the local
choice displacement vector using Gaussian-weighted nearby trials.
ORB: arrows rotate and grow with evidence.
VISp: arrows stay uniform.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from data.steinmetz import load_all, get_region_activity, get_choice_labels

rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "savefig.dpi": 400,
    "figure.dpi": 150,
})

DECISION_WINDOW = slice(25, 35)


def get_best_session(sessions, region):
    """Find session with valid bracket norm and most neurons."""
    from crossval.bracket_norm_core import compute_bracket_norm as _bn
    best_sess = None
    best_n = 0
    for sess in sessions:
        act = get_region_activity(sess, region, time_slice=DECISION_WINDOW)
        if act is None:
            continue
        n_neurons = act.shape[1]
        if n_neurons < 5:
            continue
        act_mean = act.mean(axis=2)
        choice = get_choice_labels(sess)
        n = min(len(act_mean), len(choice))
        evidence = np.abs(sess["contrast_left"][:n] - sess["contrast_right"][:n])
        result = _bn(act_mean[:n], choice[:n], evidence, min_per_quartile=3)
        if result and n_neurons > best_n:
            best_n = n_neurons
            best_sess = sess
    return best_sess, best_n


def kernel_smoothed_field(act_mean, choice, evidence, n_ev=11, n_cc=7,
                          ev_bw=0.15, cc_bw=0.20):
    """Compute kernel-smoothed choice displacement vectors on a grid.

    At each grid point, weights all trials by a Gaussian kernel in
    (evidence, choice-axis projection) space, then computes
    weighted mean(right) - weighted mean(left).

    Returns grid coordinates and displacement vectors in the canonical
    (choice axis, rotation axis) 2D plane.
    """
    # Global choice axis
    left_mean = act_mean[choice == 0].mean(0)
    right_mean = act_mean[choice == 1].mean(0)
    choice_axis = right_mean - left_mean
    ca_norm = np.linalg.norm(choice_axis)
    if ca_norm < 1e-10:
        return None
    ca_hat = choice_axis / ca_norm

    # Bracket direction (orthogonal to choice axis)
    ev_quartiles = np.percentile(evidence, [25, 75])
    low_mask = evidence <= ev_quartiles[0]
    high_mask = evidence >= ev_quartiles[1]
    left_low = (choice == 0) & low_mask
    right_low = (choice == 1) & low_mask
    left_high = (choice == 0) & high_mask
    right_high = (choice == 1) & high_mask

    if left_low.sum() < 3 or right_low.sum() < 3 or left_high.sum() < 3 or right_high.sum() < 3:
        return None

    xi_low = act_mean[right_low].mean(0) - act_mean[left_low].mean(0)
    xi_high = act_mean[right_high].mean(0) - act_mean[left_high].mean(0)
    bracket = xi_high - xi_low
    ortho = bracket - np.dot(bracket, ca_hat) * ca_hat
    ortho_norm = np.linalg.norm(ortho)
    if ortho_norm < 1e-10:
        orth_hat = np.zeros_like(ca_hat)
        idx = 0 if abs(ca_hat[0]) < 0.9 else 1
        orth_hat[idx] = 1.0
        orth_hat -= np.dot(orth_hat, ca_hat) * ca_hat
        orth_hat /= np.linalg.norm(orth_hat)
    else:
        orth_hat = ortho / ortho_norm

    # Project trials onto choice axis for the y-coordinate
    proj = act_mean @ ca_hat

    # Normalize evidence and projections to [0, 1]
    ev_min, ev_max = evidence.min(), evidence.max()
    ev_normed = (evidence - ev_min) / (ev_max - ev_min + 1e-10)
    proj_lo, proj_hi = np.percentile(proj, [5, 95])
    proj_normed = (proj - proj_lo) / (proj_hi - proj_lo + 1e-10)

    # Grid
    ev_grid_1d = np.linspace(0, 1, n_ev)
    cc_grid_1d = np.linspace(0, 1, n_cc)
    EV, CC = np.meshgrid(ev_grid_1d, cc_grid_1d)
    U = np.zeros_like(EV)
    V = np.zeros_like(EV)
    gain = np.zeros_like(EV)
    valid = np.zeros_like(EV, dtype=bool)

    for i in range(n_cc):
        for j in range(n_ev):
            # Gaussian weights
            w_ev = np.exp(-0.5 * ((ev_normed - ev_grid_1d[j]) / ev_bw) ** 2)
            w_cc = np.exp(-0.5 * ((proj_normed - cc_grid_1d[i]) / cc_bw) ** 2)
            w = w_ev * w_cc

            w_left = w * (choice == 0)
            w_right = w * (choice == 1)

            eff_left = w_left.sum()
            eff_right = w_right.sum()

            if eff_left < 2 or eff_right < 2:
                continue

            mean_left = (act_mean.T @ w_left) / eff_left
            mean_right = (act_mean.T @ w_right) / eff_right
            disp = mean_right - mean_left

            U[i, j] = np.dot(disp, ca_hat)
            V[i, j] = np.dot(disp, orth_hat)
            n_neurons = act_mean.shape[1]
            gain[i, j] = np.linalg.norm(disp) / np.sqrt(n_neurons)
            valid[i, j] = True

    # Compute summary stats
    bn = np.linalg.norm(xi_high - xi_low)
    bn_normed = bn / np.sqrt(act_mean.shape[1])

    return EV, CC, U, V, gain, valid, act_mean.shape[1], bn, bn_normed


print("Loading Steinmetz data...")
sessions = load_all()
print(f"Loaded {len(sessions)} sessions")

fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9),
                         gridspec_kw={"wspace": 0.38, "right": 0.87})

# Compute both panels first to get shared color scale
panels = []
for region in ["ORB", "VISp"]:
    sess, n_neurons = get_best_session(sessions, region)
    if sess is None:
        panels.append(None)
        continue
    act = get_region_activity(sess, region, time_slice=DECISION_WINDOW)
    act_mean = act.mean(axis=2)
    choice = get_choice_labels(sess)
    n = min(len(act_mean), len(choice))
    act_mean, choice = act_mean[:n], choice[:n]
    evidence = np.abs(sess["contrast_left"][:n] - sess["contrast_right"][:n])
    print(f"{region}: {n} trials, {n_neurons} neurons")
    result = kernel_smoothed_field(act_mean, choice, evidence)
    panels.append(result)

# Shared color scale
all_gains = []
for p in panels:
    if p is not None:
        EV, CC, U, V, gain, valid, _, _, _ = p
        if valid.any():
            all_gains.append(gain[valid])
if all_gains:
    g_all = np.concatenate(all_gains)
    vmin, vmax = np.percentile(g_all, [5, 95])
else:
    vmin, vmax = 0, 1

for col, (region, title) in enumerate([("ORB", "ORB/OFC — high bracket norm"),
                                        ("VISp", "VISp/V1 — low bracket norm")]):
    result = panels[col]
    if result is None:
        continue
    EV, CC, U, V, gain, valid, n_neurons, bn, bn_normed = result

    ax = axes[col]

    # Normalize arrow lengths for display
    mag = np.sqrt(U**2 + V**2)
    max_mag = mag[valid].max() if valid.any() else 1.0

    q = ax.quiver(EV[valid], CC[valid], U[valid], V[valid],
                  gain[valid], cmap="magma", scale=max_mag * 12,
                  width=0.007, headwidth=3.5, headlength=4, pivot="mid",
                  clim=(vmin, vmax))

    ax.set_title(title, fontsize=11.5, pad=8)
    ax.set_xlabel("sensory evidence")
    ax.set_ylabel("choice-axis position")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    for s in ax.spines.values():
        s.set_color("#777")

    ax.text(0.03, 0.03,
            f"n = {n_neurons}    BN/√n = {bn_normed:.4f}",
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#ccc', alpha=0.85))

    if col == 0:
        q_ref = q

cb = fig.colorbar(q_ref, ax=axes, shrink=0.78, pad=0.03, aspect=22)
cb.set_label("per-neuron gain", fontsize=10.5)
cb.ax.tick_params(labelsize=9)

_out = Path(__file__).resolve().parent
fig.savefig(str(_out / "fig_quiver.png"), bbox_inches="tight", pad_inches=0.08)
fig.savefig(str(_out / "fig_quiver.pdf"), bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print("Saved fig_quiver.png + .pdf")
