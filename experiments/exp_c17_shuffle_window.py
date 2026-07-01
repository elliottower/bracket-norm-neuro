"""Exp-C17: Shuffle the decision window — causal timing control.

Recompute SEIS equivariance using a NON-decision window (pre-stimulus baseline,
early stimulus, late post-decision) and compare to the decision-window result.

If SEIS equivariance during baseline predicts silencing at comparable rho,
the temporal specificity is spurious. If it drops to rho ~ 0, the effect
is specific to the computation window.

This is the control that reviewers will definitely request.

CPU-only, ~30 min.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA

sys.path.insert(0, str(Path(__file__).parent))
from shared_bundle import SILENCING_EFFECTS, save_results

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "c17_shuffle"
MIN_NEURONS = 15
MIN_PAIRS = 15


def match_evidence_pairs(choice_labels, cl, cr, n_pairs=300):
    evidence = np.abs(cl - cr)
    left_idx = np.where(choice_labels == 0)[0]
    right_idx = np.where(choice_labels == 1)[0]
    if len(left_idx) < 5 or len(right_idx) < 5:
        return [], []
    pairs_l, pairs_r = [], []
    ev_left = evidence[left_idx]
    ev_right = evidence[right_idx]
    for i, li in enumerate(left_idx):
        diffs = np.abs(ev_left[i] - ev_right)
        best = np.argmin(diffs)
        if diffs[best] < 0.1:
            pairs_l.append(li)
            pairs_r.append(right_idx[best])
        if len(pairs_l) >= n_pairs:
            break
    return pairs_l, pairs_r


def cca_equivariance(X, Y, n_components=None):
    n, d = X.shape
    if n_components is None:
        n_components = min(d, n // 2, 8)
    n_components = max(1, min(n_components, d, n - 1))
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    try:
        cca = CCA(n_components=n_components, max_iter=500)
        Xp, Yp = cca.fit_transform(X_c, Y_c)
        corrs = []
        for i in range(n_components):
            r = np.corrcoef(Xp[:, i], Yp[:, i])[0, 1]
            if np.isfinite(r):
                corrs.append(abs(r))
        return float(np.mean(corrs)) if corrs else 0.0
    except Exception:
        return 0.0


WINDOWS = {
    "pre_stim": slice(0, 10),
    "early_stim": slice(5, 15),
    "mid_stim": slice(10, 20),
    "peri_decision": slice(15, 25),
    "decision": slice(20, 30),
    "late_decision": slice(25, 35),
    "post_decision": slice(30, 40),
    "late": slice(35, 45),
}


def run():
    from data.steinmetz import load_all, list_regions, get_region_activity, get_choice_labels

    print(f"[{datetime.now().isoformat()}] Exp-C17: Shuffle window control")
    print(f"  Computing SEIS equivariance across ALL time windows")
    print(f"  If baseline windows predict silencing as well as decision window,")
    print(f"  the temporal specificity is spurious.\n")

    sessions = load_all()
    print(f"  {len(sessions)} sessions loaded")

    region_window_scores = {}

    for si, sess in enumerate(sessions):
        choice = get_choice_labels(sess)
        choice_binary = (choice == 1).astype(int)
        if len(set(choice_binary)) < 2:
            continue
        cl = sess["contrast_left"]
        cr = sess["contrast_right"]
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            full_activity = get_region_activity(sess, region, slice(0, 50))
            if full_activity is None:
                continue
            if full_activity.ndim == 3:
                n_bins = full_activity.shape[2]
            else:
                continue

            n_trials = min(len(full_activity), len(choice_binary))
            full_activity = full_activity[:n_trials]
            cb = choice_binary[:n_trials]
            pairs_l, pairs_r = match_evidence_pairs(cb, cl[:n_trials], cr[:n_trials])
            if len(pairs_l) < MIN_PAIRS:
                continue

            if region not in region_window_scores:
                region_window_scores[region] = {w: [] for w in WINDOWS}

            for wname, wslice in WINDOWS.items():
                if full_activity.ndim == 3:
                    act = full_activity[:, :, wslice].mean(axis=2)
                else:
                    act = full_activity

                if act.shape[1] < MIN_NEURONS:
                    continue

                X_left = act[pairs_l]
                X_right = act[pairs_r]
                eq = cca_equivariance(X_left, X_right)
                region_window_scores[region][wname].append(eq)

    region_summary = {}
    for region, windows in region_window_scores.items():
        region_summary[region] = {}
        for wname, scores in windows.items():
            if scores:
                region_summary[region][wname] = float(np.mean(scores))

    print(f"\n{'='*80}")
    print(f"SEIS equivariance by time window:")
    print(f"\n{'Region':>8s}  ", end="")
    for wname in WINDOWS:
        print(f"  {wname[:8]:>8s}", end="")
    print()
    print("-" * (10 + 10 * len(WINDOWS)))

    for region in sorted(region_summary.keys()):
        if region in SILENCING_EFFECTS:
            print(f"{region:>8s}  ", end="")
            for wname in WINDOWS:
                v = region_summary[region].get(wname)
                if v is not None:
                    print(f"  {v:8.3f}", end="")
                else:
                    print(f"  {'---':>8s}", end="")
            print(f"  sil={SILENCING_EFFECTS[region]:.2f}")

    print(f"\n{'='*80}")
    print(f"Silencing correlations by window:")
    print(f"  {'Window':>15s}  {'ρ':>7s}  {'p':>7s}  {'n':>3s}  {'sig':>3s}")
    print("  " + "-" * 45)

    window_corrs = {}
    for wname in WINDOWS:
        mx, my = [], []
        for region, effect in SILENCING_EFFECTS.items():
            if region in region_summary and wname in region_summary[region]:
                mx.append(region_summary[region][wname])
                my.append(effect)
        if len(mx) >= 5:
            rho, p = spearmanr(mx, my)
            window_corrs[wname] = {"rho": float(rho), "p": float(p), "n": len(mx)}
            star = "**" if p < 0.01 else ("*" if p < 0.05 else " ")
            print(f"  {wname:>15s}  {rho:+7.3f}  {p:7.4f}  {len(mx):3d}  {star}")

    decision_rho = window_corrs.get("decision", {}).get("rho", 0)
    pre_rho = window_corrs.get("pre_stim", {}).get("rho", 0)
    late_rho = window_corrs.get("late", {}).get("rho", 0)

    print(f"\n{'='*80}")
    print(f"CONCLUSION:")
    print(f"  Decision window rho:   {decision_rho:+.3f}")
    print(f"  Pre-stimulus rho:      {pre_rho:+.3f}")
    print(f"  Late window rho:       {late_rho:+.3f}")
    if abs(decision_rho) > abs(pre_rho) + 0.2:
        print(f"  TEMPORAL SPECIFICITY CONFIRMED: decision >> baseline")
        print(f"  Drop from decision to pre_stim: {abs(decision_rho) - abs(pre_rho):+.3f}")
    elif abs(decision_rho) > abs(pre_rho):
        print(f"  WEAK temporal specificity: decision > baseline but close")
    else:
        print(f"  WARNING: NO temporal specificity. Baseline matches or exceeds decision.")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_regions": len(region_summary),
        "region_summary": region_summary,
        "window_correlations": window_corrs,
        "conclusion": {
            "decision_rho": decision_rho,
            "pre_stim_rho": pre_rho,
            "late_rho": late_rho,
            "temporal_specificity": abs(decision_rho) > abs(pre_rho) + 0.2,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results("exp_c17_shuffle_window", results, RESULTS_DIR)
    return results


if __name__ == "__main__":
    run()
