"""Baseline separation test for bracket_norm (MechVal criterion M2).

Computes bracket_norm on trial-shuffled data to establish that the real
bracket_norm is distinguishable from a null distribution.  If real BN is
not significantly higher than shuffled BN, the metric has no baseline
separation and cannot be used as evidence for any claim.

CPU-only, ~10 min.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from crossval.bracket_norm_core import (
    compute_bracket_norm,
    partial_spearman,
)

sys.path.insert(0, str(Path(__file__).parent))
from shared_bundle import SILENCING_EFFECTS, save_results

N_SHUFFLES = 1000
TIME_WINDOW = slice(15, 35)
MIN_NEURONS = 10


def run():
    from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

    print(f"[{datetime.now(timezone.utc).isoformat()}] M2 baseline separation for bracket_norm")
    sessions = load_all()
    print(f"  {len(sessions)} sessions, {N_SHUFFLES} shuffles per region-session")

    region_real_bn = {}
    region_shuffle_bns = {}
    region_neuron_counts = {}
    region_pvalues = {}

    for si, sess in enumerate(tqdm(sessions, desc="Sessions")):
        choice = get_choice_labels(sess)
        choice_binary = (choice == 1).astype(int)
        if len(set(choice_binary)) < 2:
            continue
        cl = sess["contrast_left"]
        cr = sess["contrast_right"]
        n_spk_trials = sess["spks"].shape[2]
        n = min(n_spk_trials, len(choice_binary), len(cl))
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

            real = compute_bracket_norm(activity, cb, ev)
            if real is None:
                continue

            real_bn = real["bracket_norm"]

            shuffle_bns = []
            for _ in range(N_SHUFFLES):
                perm = np.random.permutation(n_trials)
                shuf = compute_bracket_norm(activity, cb[perm], ev)
                if shuf is not None:
                    shuffle_bns.append(shuf["bracket_norm"])

            if len(shuffle_bns) < 100:
                continue

            shuffle_bns = np.array(shuffle_bns)
            percentile = float(np.mean(shuffle_bns >= real_bn))

            if region not in region_real_bn:
                region_real_bn[region] = []
                region_shuffle_bns[region] = []
                region_neuron_counts[region] = []
                region_pvalues[region] = []

            region_real_bn[region].append(real_bn)
            region_shuffle_bns[region].append(float(np.mean(shuffle_bns)))
            region_neuron_counts[region].append(full_activity.shape[1])
            region_pvalues[region].append(percentile)

    print(f"\n{'='*70}")
    print(f"BASELINE SEPARATION RESULTS")
    print(f"{'='*70}")

    summary = {}
    all_ratios = []
    for region in sorted(region_real_bn.keys()):
        real_mean = float(np.mean(region_real_bn[region]))
        shuf_mean = float(np.mean(region_shuffle_bns[region]))
        ratio = real_mean / (shuf_mean + 1e-10)
        mean_p = float(np.mean(region_pvalues[region]))
        n_sess = len(region_real_bn[region])
        n_neurons = float(np.mean(region_neuron_counts[region]))

        all_ratios.append(ratio)
        summary[region] = {
            "real_bn_mean": real_mean,
            "shuffle_bn_mean": shuf_mean,
            "ratio": ratio,
            "mean_pvalue": mean_p,
            "n_sessions": n_sess,
            "n_neurons_mean": n_neurons,
        }
        print(f"  {region:8s}: real={real_mean:.4f} shuf={shuf_mean:.4f} "
              f"ratio={ratio:.2f} p={mean_p:.4f} n_sess={n_sess}")

    silencing_regions = [r for r in SILENCING_EFFECTS if r in summary]
    print(f"\nSilencing regions ({len(silencing_regions)}):")
    for r in silencing_regions:
        s = summary[r]
        print(f"  {r:8s}: ratio={s['ratio']:.2f} p={s['mean_pvalue']:.4f}")

    n_significant = sum(1 for r, s in summary.items() if s["mean_pvalue"] < 0.05)
    n_total = len(summary)
    print(f"\n  {n_significant}/{n_total} regions have real BN > 95% of shuffles")
    print(f"  Mean ratio (real/shuffle): {np.mean(all_ratios):.2f}")

    corr_real_bn = {}
    corr_ratio = {}
    corr_nc = {}
    for region in silencing_regions:
        corr_real_bn[region] = summary[region]["real_bn_mean"]
        corr_ratio[region] = summary[region]["ratio"]
        corr_nc[region] = summary[region]["n_neurons_mean"]

    if len(silencing_regions) >= 5:
        mx = [corr_real_bn[r] for r in silencing_regions]
        my = [SILENCING_EFFECTS[r] for r in silencing_regions]
        mn = [corr_nc[r] for r in silencing_regions]
        mr = [corr_ratio[r] for r in silencing_regions]

        rho_real, p_real = spearmanr(mx, my)
        partial_real = partial_spearman(np.array(mx), np.array(my), np.array(mn))
        rho_ratio, p_ratio = spearmanr(mr, my)
        partial_ratio = partial_spearman(np.array(mr), np.array(my), np.array(mn))

        print(f"\n  Real BN vs silencing: rho={rho_real:+.3f} partial={partial_real:+.3f}")
        print(f"  BN ratio (real/shuffle) vs silencing: rho={rho_ratio:+.3f} partial={partial_ratio:+.3f}")
    else:
        rho_real = p_real = partial_real = None
        rho_ratio = p_ratio = partial_ratio = None

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "M2_baseline_separation",
        "n_shuffles": N_SHUFFLES,
        "n_regions": n_total,
        "n_significant_005": n_significant,
        "mean_ratio": float(np.mean(all_ratios)),
        "region_summary": summary,
        "silencing_correlation": {
            "rho_real": rho_real,
            "p_real": p_real,
            "partial_real": partial_real,
            "rho_ratio": rho_ratio,
            "p_ratio": p_ratio,
            "partial_ratio": partial_ratio,
            "n_regions": len(silencing_regions),
            "regions": silencing_regions,
        },
    }

    save_results("bn_baseline_separation", results)
    print(f"\nDone.")


if __name__ == "__main__":
    run()
