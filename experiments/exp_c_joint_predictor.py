"""Joint predictor regression (MechKnowledge audit).

Compute bracket_norm, choice probability, and population decoding accuracy
on the same Steinmetz data with the same region definitions.  Then run
joint multiple regression predicting silencing effect.

Tests whether bracket_norm adds independent information beyond classical
measures.  Addresses parcellation coherence (same units), individual
validity (each predictor tested), and collective coverage (joint R²).

CPU-only, ~15 min.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from crossval.bracket_norm_core import (
    compute_bracket_norm,
    partial_spearman,
)

sys.path.insert(0, str(Path(__file__).parent))
from shared_bundle import SILENCING_EFFECTS, save_results

TIME_WINDOW = slice(15, 35)
MIN_NEURONS = 10


def choice_probability_region(activity, choice_binary):
    """Compute mean choice probability (auROC) across neurons in a region.

    choice_probability per neuron = P(activity | choice=1) > P(activity | choice=0).
    This is the classical Britten et al. (1996) measure.
    """
    left_idx = np.where(choice_binary == 0)[0]
    right_idx = np.where(choice_binary == 1)[0]
    if len(left_idx) < 10 or len(right_idx) < 10:
        return None

    n_neurons = activity.shape[1]
    cps = []
    for ni in range(n_neurons):
        x_left = activity[left_idx, ni]
        x_right = activity[right_idx, ni]

        n_pairs = 0
        n_correct = 0
        for r in x_right:
            for l in x_left:
                n_pairs += 1
                if r > l:
                    n_correct += 1
                elif r == l:
                    n_correct += 0.5

        if n_pairs > 0:
            cps.append(n_correct / n_pairs)

    if not cps:
        return None
    return float(np.mean(np.abs(np.array(cps) - 0.5)) + 0.5)


def choice_probability_fast(activity, choice_binary):
    """Fast approximation of mean |CP - 0.5| + 0.5 using rank-based auROC."""
    left_idx = np.where(choice_binary == 0)[0]
    right_idx = np.where(choice_binary == 1)[0]
    if len(left_idx) < 10 or len(right_idx) < 10:
        return None

    n_neurons = activity.shape[1]
    n_left = len(left_idx)
    n_right = len(right_idx)
    cps = np.zeros(n_neurons)

    for ni in range(n_neurons):
        all_vals = np.concatenate([activity[left_idx, ni], activity[right_idx, ni]])
        ranks = np.argsort(np.argsort(all_vals)).astype(float) + 1
        right_rank_sum = ranks[n_left:].sum()
        auroc = (right_rank_sum - n_right * (n_right + 1) / 2) / (n_left * n_right)
        cps[ni] = abs(auroc - 0.5) + 0.5

    return float(np.mean(cps))


def decoding_accuracy(activity, choice_binary):
    """Leave-one-out LDA decoding accuracy."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    left_idx = np.where(choice_binary == 0)[0]
    right_idx = np.where(choice_binary == 1)[0]
    if len(left_idx) < 10 or len(right_idx) < 10:
        return None

    n = len(choice_binary)
    correct = 0
    for i in range(n):
        train_idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        X_train = activity[train_idx]
        y_train = choice_binary[train_idx]
        X_test = activity[i:i+1]
        y_test = choice_binary[i]

        if len(set(y_train)) < 2:
            continue
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train, y_train)
            pred = lda.predict(X_test)[0]
            if pred == y_test:
                correct += 1
        except Exception:
            continue

    return float(correct / n) if n > 0 else None


def run():
    from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

    print(f"[{datetime.now(timezone.utc).isoformat()}] Joint predictor regression")
    sessions = load_all()
    print(f"  {len(sessions)} sessions")

    region_bn = {}
    region_cp = {}
    region_acc = {}
    region_nc = {}

    for si, sess in enumerate(tqdm(sessions, desc="Sessions")):
        choice = get_choice_labels(sess)
        choice_binary = (choice == 1).astype(int)
        if len(set(choice_binary)) < 2:
            continue
        cl = sess["contrast_left"]
        cr = sess["contrast_right"]
        n = min(sess["spks"].shape[2], len(choice_binary), len(cl))
        choice_binary = choice_binary[:n]
        evidence = np.abs(cl[:n] - cr[:n])

        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            full_activity = get_region_activity(sess, region)
            if full_activity is None or full_activity.ndim != 3:
                continue
            if full_activity.shape[1] < MIN_NEURONS:
                continue

            n_trials = min(full_activity.shape[0], n)
            activity = full_activity[:n_trials, :, TIME_WINDOW].mean(axis=2)
            cb = choice_binary[:n_trials]
            ev = evidence[:n_trials]

            bn_res = compute_bracket_norm(activity, cb, ev)
            if bn_res is None:
                continue

            cp = choice_probability_fast(activity, cb)
            if cp is None:
                continue

            acc = decoding_accuracy(activity, cb)
            if acc is None:
                continue

            if region not in region_bn:
                region_bn[region] = []
                region_cp[region] = []
                region_acc[region] = []
                region_nc[region] = []

            region_bn[region].append(bn_res["bracket_norm"])
            region_cp[region].append(cp)
            region_acc[region].append(acc)
            region_nc[region].append(full_activity.shape[1])

    print(f"\n{'='*70}")
    print(f"JOINT PREDICTOR RESULTS")
    print(f"{'='*70}")

    summary = {}
    for region in sorted(region_bn.keys()):
        summary[region] = {
            "bn_mean": float(np.mean(region_bn[region])),
            "cp_mean": float(np.mean(region_cp[region])),
            "acc_mean": float(np.mean(region_acc[region])),
            "nc_mean": float(np.mean(region_nc[region])),
            "n_sessions": len(region_bn[region]),
        }
        s = summary[region]
        print(f"  {region:8s}: BN={s['bn_mean']:.4f} CP={s['cp_mean']:.3f} "
              f"Acc={s['acc_mean']:.3f} nc={s['nc_mean']:.0f}")

    silencing_regions = [r for r in SILENCING_EFFECTS if r in summary]
    print(f"\nSilencing regions with all 3 metrics: {len(silencing_regions)}")

    if len(silencing_regions) >= 5:
        s_bn = np.array([summary[r]["bn_mean"] for r in silencing_regions])
        s_cp = np.array([summary[r]["cp_mean"] for r in silencing_regions])
        s_acc = np.array([summary[r]["acc_mean"] for r in silencing_regions])
        s_nc = np.array([summary[r]["nc_mean"] for r in silencing_regions])
        s_eff = np.array([SILENCING_EFFECTS[r] for r in silencing_regions])

        print(f"\nINDIVIDUAL CORRELATIONS WITH SILENCING:")
        for name, vals in [("BN", s_bn), ("CP", s_cp), ("Acc", s_acc), ("NC", s_nc)]:
            rho, p = spearmanr(vals, s_eff)
            partial = partial_spearman(vals, s_eff, s_nc) if name != "NC" else float("nan")
            print(f"  {name:4s}: rho={rho:+.3f} p={p:.4f} partial(nc)={partial:+.3f}")

        print(f"\nPAIRWISE CORRELATIONS BETWEEN PREDICTORS:")
        for n1, v1, n2, v2 in [("BN", s_bn, "CP", s_cp), ("BN", s_bn, "Acc", s_acc),
                                ("CP", s_cp, "Acc", s_acc), ("BN", s_bn, "NC", s_nc),
                                ("CP", s_cp, "NC", s_nc)]:
            rho, p = spearmanr(v1, v2)
            print(f"  {n1} vs {n2}: rho={rho:+.3f} p={p:.4f}")

        print(f"\nJOINT REGRESSION: silencing ~ BN + CP + Acc + NC")
        X = np.column_stack([s_bn, s_cp, s_acc, s_nc])
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        y = s_eff

        reg_full = LinearRegression().fit(X_norm, y)
        r2_full = reg_full.score(X_norm, y)
        print(f"  Full model R²: {r2_full:.3f}")
        print(f"  Coefficients: BN={reg_full.coef_[0]:+.3f} CP={reg_full.coef_[1]:+.3f} "
              f"Acc={reg_full.coef_[2]:+.3f} NC={reg_full.coef_[3]:+.3f}")

        print(f"\nDROP-ONE ANALYSIS:")
        predictors = ["BN", "CP", "Acc", "NC"]
        for drop_i, drop_name in enumerate(predictors):
            keep = [j for j in range(4) if j != drop_i]
            X_reduced = X_norm[:, keep]
            reg_reduced = LinearRegression().fit(X_reduced, y)
            r2_reduced = reg_reduced.score(X_reduced, y)
            delta_r2 = r2_full - r2_reduced
            print(f"  Drop {drop_name:4s}: R²={r2_reduced:.3f} (Δ={delta_r2:+.3f})")

        print(f"\nBN-ONLY vs NC-ONLY vs BN+NC:")
        for label, cols in [("BN only", [0]), ("NC only", [3]), ("BN+NC", [0, 3]),
                            ("CP only", [1]), ("Acc only", [2])]:
            X_sub = X_norm[:, cols]
            reg_sub = LinearRegression().fit(X_sub, y)
            r2_sub = reg_sub.score(X_sub, y)
            print(f"  {label:12s}: R²={r2_sub:.3f}")

        regression_results = {
            "full_r2": float(r2_full),
            "coefficients": {
                "BN": float(reg_full.coef_[0]),
                "CP": float(reg_full.coef_[1]),
                "Acc": float(reg_full.coef_[2]),
                "NC": float(reg_full.coef_[3]),
            },
            "n_regions": len(silencing_regions),
            "regions": silencing_regions,
        }
    else:
        regression_results = {"n": len(silencing_regions), "error": "too few"}

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "joint_predictor_regression",
        "n_regions_total": len(summary),
        "n_silencing_regions": len(silencing_regions),
        "region_summary": summary,
        "regression": regression_results,
    }

    save_results("joint_predictor", results)
    print(f"\nDone.")


if __name__ == "__main__":
    run()
