"""Core bracket_norm computation — shared across datasets.

bracket_norm = ||xi(high evidence) - xi(low evidence)||

where xi(q) = mean(x_right) - mean(x_left) within evidence quartile q,
restricted to the decision window.
"""
import numpy as np
from scipy.stats import spearmanr


def partial_spearman(x, y, z):
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    rho_xy, _ = spearmanr(x, y)
    rho_xz, _ = spearmanr(x, z)
    rho_yz, _ = spearmanr(y, z)
    numerator = rho_xy - rho_xz * rho_yz
    denominator = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    if denominator < 1e-10:
        return 0.0
    return numerator / denominator


def compute_bracket_norm(activity, choice_binary, evidence, min_per_quartile=5):
    """Compute bracket_norm for a single region-session.

    Args:
        activity: (n_trials, n_neurons) — time-averaged in the decision window
        choice_binary: (n_trials,) — 0 or 1
        evidence: (n_trials,) — continuous evidence strength
        min_per_quartile: minimum trials per choice per quartile

    Returns:
        dict with bracket_norm, rotation_angle, commutativity, or None if insufficient data.
    """
    left_idx = np.where(choice_binary == 0)[0]
    right_idx = np.where(choice_binary == 1)[0]
    if len(left_idx) < 10 or len(right_idx) < 10:
        return None

    ev_quartiles = np.percentile(evidence, [25, 50, 75])
    q_labels = np.digitize(evidence, ev_quartiles)

    quartile_displacements = {}
    for q in range(4):
        q_mask = q_labels == q
        q_left = left_idx[q_mask[left_idx]]
        q_right = right_idx[q_mask[right_idx]]
        if len(q_left) < min_per_quartile or len(q_right) < min_per_quartile:
            continue
        mean_left = activity[q_left].mean(axis=0)
        mean_right = activity[q_right].mean(axis=0)
        quartile_displacements[q] = mean_right - mean_left

    if 0 not in quartile_displacements or 3 not in quartile_displacements:
        return None

    low_q = quartile_displacements[0]
    high_q = quartile_displacements[3]

    bracket = high_q - low_q
    bracket_norm = float(np.linalg.norm(bracket))

    low_norm = np.linalg.norm(low_q)
    high_norm = np.linalg.norm(high_q)
    if low_norm > 1e-10 and high_norm > 1e-10:
        cos_angle = np.dot(low_q / low_norm, high_q / high_norm)
        cos_angle = np.clip(cos_angle, -1, 1)
        rotation_angle = float(np.arccos(cos_angle))
    else:
        rotation_angle = 0.0

    mean_disp_norm = (low_norm + high_norm) / 2
    commutativity = bracket_norm / (mean_disp_norm + 1e-10)

    return {
        "bracket_norm": bracket_norm,
        "rotation_angle": rotation_angle,
        "commutativity": commutativity,
        "n_left_low": int(len(left_idx[q_labels[left_idx] == 0])),
        "n_right_high": int(len(right_idx[q_labels[right_idx] == 3])),
    }


def aggregate_region_metrics(region_metrics):
    """Aggregate per-session metrics into per-region summaries.

    Args:
        region_metrics: dict of region -> list of metric dicts

    Returns:
        dict of region -> summary dict
    """
    summary = {}
    for region, metrics_list in sorted(region_metrics.items()):
        metrics_list = [m for m in metrics_list if m is not None]
        if not metrics_list:
            continue
        s = {"n_sessions": len(metrics_list)}
        for key in ["bracket_norm", "rotation_angle", "commutativity"]:
            vals = [m[key] for m in metrics_list if key in m and np.isfinite(m[key])]
            if vals:
                s[f"{key}_mean"] = float(np.mean(vals))
                s[f"{key}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        summary[region] = s
    return summary


def correlate_with_silencing(region_summary, silencing_effects, neuron_counts,
                             metric_key="bracket_norm_mean"):
    """Compute raw and partial Spearman correlations with silencing effects.

    Returns dict with rho, p, partial, n, and per-metric details.
    """
    mx, my, mn, regions_used = [], [], [], []
    for region, effect in silencing_effects.items():
        if region in region_summary and metric_key in region_summary[region]:
            if region in neuron_counts:
                mx.append(region_summary[region][metric_key])
                my.append(effect)
                mn.append(neuron_counts[region])
                regions_used.append(region)

    if len(mx) < 5:
        return {"n": len(mx), "error": "too few overlapping regions"}

    rho, p = spearmanr(mx, my)
    rho_nc, _ = spearmanr(mx, mn)
    partial = partial_spearman(mx, my, mn)

    return {
        "rho": float(rho),
        "p": float(p),
        "rho_nc": float(rho_nc),
        "partial": float(partial),
        "n": len(mx),
        "regions_used": regions_used,
    }
