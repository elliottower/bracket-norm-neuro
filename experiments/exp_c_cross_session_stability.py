"""Cross-session SEIS stability.

For each region with multiple sessions, compute SEIS equivariance per session
and measure cross-session coefficient of variation. Hypothesis: causally
important regions have MORE stable SEIS (lower CV).

Addresses reviewer question: "is rho=+0.706 a lucky split of 12 regions
or a stable property?"

CPU-only, uses existing session data. ~1 hr.
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

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "cross_session_stability"
MIN_NEURONS = 15
MIN_PAIRS = 15
TIME_WINDOW = slice(20, 30)


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


def run():
    from data.steinmetz import load_all, list_regions, get_region_activity, get_choice_labels

    print(f"[{datetime.now().isoformat()}] Cross-session SEIS stability")
    sessions = load_all()
    print(f"  {len(sessions)} sessions loaded")

    region_session_scores = {}

    for si, sess in enumerate(sessions):
        if si % 5 == 0:
            print(f"  Processing session {si+1}/{len(sessions)}...")
        choice = get_choice_labels(sess)
        choice_binary = (choice == 1).astype(int)
        if len(set(choice_binary)) < 2:
            continue
        cl = sess["contrast_left"]
        cr = sess["contrast_right"]
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            full_activity = get_region_activity(sess, region)
            if full_activity is None or full_activity.ndim != 3:
                continue
            if full_activity.shape[1] < MIN_NEURONS:
                continue

            n = min(full_activity.shape[0], len(choice_binary))
            full_activity = full_activity[:n]
            cb = choice_binary[:n]
            pairs_l, pairs_r = match_evidence_pairs(cb, cl[:n], cr[:n])
            if len(pairs_l) < MIN_PAIRS:
                continue

            try:
                act = full_activity[:, :, TIME_WINDOW].mean(axis=2)
            except (IndexError, ValueError):
                continue

            eq = cca_equivariance(act[pairs_l], act[pairs_r])
            if region not in region_session_scores:
                region_session_scores[region] = []
            region_session_scores[region].append({
                "session": si,
                "equivariance": eq,
                "n_neurons": full_activity.shape[1],
                "n_pairs": len(pairs_l),
            })

    region_stability = {}
    for region, scores in sorted(region_session_scores.items()):
        if len(scores) < 2:
            continue
        eq_vals = [s["equivariance"] for s in scores]
        mean_eq = float(np.mean(eq_vals))
        std_eq = float(np.std(eq_vals))
        cv = std_eq / (mean_eq + 1e-10)
        region_stability[region] = {
            "n_sessions": len(scores),
            "mean_eq": mean_eq,
            "std_eq": std_eq,
            "cv": cv,
            "min_eq": float(np.min(eq_vals)),
            "max_eq": float(np.max(eq_vals)),
            "range_eq": float(np.max(eq_vals) - np.min(eq_vals)),
        }

    print(f"\n{'='*80}")
    print(f"Cross-session SEIS stability: {len(region_stability)} regions with 2+ sessions")
    print(f"\n  {'Region':>8s}  {'n':>3s}  {'Mean':>6s}  {'Std':>6s}  {'CV':>6s}  {'Range':>7s}  {'Sil':>5s}")
    print(f"  {'-'*55}")
    for region in sorted(region_stability.keys()):
        s = region_stability[region]
        sil = SILENCING_EFFECTS.get(region, None)
        sil_str = f"{sil:.2f}" if sil else "  -  "
        marker = " *" if sil else ""
        print(f"  {region:>8s}  {s['n_sessions']:3d}  {s['mean_eq']:6.3f}  {s['std_eq']:6.3f}  "
              f"{s['cv']:6.3f}  {s['range_eq']:7.3f}  {sil_str}{marker}")

    silencing_corrs = {}
    for metric in ["mean_eq", "std_eq", "cv", "range_eq"]:
        mx, my = [], []
        for region, effect in SILENCING_EFFECTS.items():
            if region in region_stability:
                mx.append(region_stability[region][metric])
                my.append(effect)
        if len(mx) >= 5:
            rho, p = spearmanr(mx, my)
            silencing_corrs[metric] = {"rho": float(rho), "p": float(p), "n": len(mx)}

    print(f"\n{'='*80}")
    print(f"Silencing correlations:")
    print(f"  {'Metric':>10s}  {'rho':>7s}  {'p':>7s}  {'n':>3s}")
    print(f"  {'-'*35}")
    for metric, corr in sorted(silencing_corrs.items(), key=lambda x: -abs(x[1]["rho"])):
        star = "**" if corr["p"] < 0.01 else ("*" if corr["p"] < 0.05 else " ")
        print(f"  {metric:>10s}  {corr['rho']:+7.3f}  {corr['p']:7.4f}  {corr['n']:3d} {star}")

    if "cv" in silencing_corrs:
        print(f"\n  CV interpretation:")
        if silencing_corrs["cv"]["rho"] < -0.3:
            print(f"    LOW CV = MORE stable = MORE causally important")
            print(f"    SEIS equivariance is a reliable regional property in important regions")
        elif silencing_corrs["cv"]["rho"] > 0.3:
            print(f"    HIGH CV = MORE variable = MORE causally important")
            print(f"    SEIS equivariance fluctuates more in important regions")
        else:
            print(f"    CV is NOT associated with causal importance")
            print(f"    SEIS stability does not distinguish important from unimportant regions")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_regions": len(region_stability),
        "region_stability": region_stability,
        "silencing_correlations": silencing_corrs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results("cross_session_stability", results, RESULTS_DIR)
    return results


if __name__ == "__main__":
    run()
