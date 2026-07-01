"""Mimic mechanism discriminator (MechRef audit).

High bracket_norm can arise from two generating processes:
  1. Active recurrent computation (Steinmetz cross-regional)
  2. Input-dominated relay dynamics (Svoboda photostim)

This experiment measures autocorrelation timescale and eigenspectrum slope
per Steinmetz region to discriminate.  If high-BN regions have LONG
autocorrelation timescales and GRADUAL eigenspectra, they are recurrent
computation nodes.  If SHORT timescales and STEEP spectra, they are
input-dominated relays and the cross-regional BN correlation means
something different.

CPU-only, ~20 min.
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

TIME_WINDOW = slice(15, 35)
MIN_NEURONS = 10


def autocorrelation_timescale(spks_region, max_lag=10):
    """Compute mean autocorrelation timescale across neurons.

    spks_region: (n_neurons, n_bins, n_trials) spike counts in 10ms bins.
    Returns timescale in bins (multiply by 10 for ms).
    """
    n_neurons, n_bins, n_trials = spks_region.shape
    trial_mean = spks_region.mean(axis=2)
    fluct = spks_region - trial_mean[:, :, None]

    taus = []
    for ni in range(n_neurons):
        var = np.var(fluct[ni])
        if var < 1e-10:
            continue
        ac = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag >= n_bins:
                break
            c = np.mean(fluct[ni, :n_bins - lag, :] * fluct[ni, lag:, :])
            ac[lag] = c / var
        positive = np.where(ac < 0)[0]
        if len(positive) > 0:
            tau = positive[0]
        else:
            tau = max_lag
        taus.append(tau)

    if not taus:
        return 0.0
    return float(np.mean(taus))


def eigenspectrum_slope(activity_2d):
    """Compute log-log eigenspectrum slope.

    activity_2d: (n_trials, n_neurons).
    More negative = steeper = more low-rank / input-dominated.
    """
    centered = activity_2d - activity_2d.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) < 3:
        return 0.0

    ranks = np.arange(1, len(eigvals) + 1)
    log_r = np.log(ranks)
    log_e = np.log(eigvals)

    coeffs = np.polyfit(log_r, log_e, 1)
    return float(coeffs[0])


def participation_ratio(activity_2d):
    """Participation ratio = (sum eigenvalues)^2 / sum(eigenvalues^2)."""
    centered = activity_2d - activity_2d.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 0.0
    return float(np.sum(eigvals) ** 2 / np.sum(eigvals ** 2))


def run():
    from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

    print(f"[{datetime.now(timezone.utc).isoformat()}] Mimic mechanism discriminator")
    sessions = load_all()
    print(f"  {len(sessions)} sessions")

    region_bn = {}
    region_tau = {}
    region_slope = {}
    region_pr = {}
    region_nc = {}

    for si, sess in enumerate(tqdm(sessions, desc="Sessions")):
        spks = sess["spks"]
        choice = get_choice_labels(sess)
        choice_binary = (choice == 1).astype(int)
        if len(set(choice_binary)) < 2:
            continue
        cl = sess["contrast_left"]
        cr = sess["contrast_right"]
        n = min(spks.shape[2], len(choice_binary), len(cl))
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

            res = compute_bracket_norm(activity, cb, ev)
            if res is None:
                continue

            neuron_mask = sess["brain_area"] == region
            spks_region = spks[neuron_mask][:, :, :n_trials]
            tau = autocorrelation_timescale(spks_region)
            slope = eigenspectrum_slope(activity)
            pr = participation_ratio(activity)

            if region not in region_bn:
                region_bn[region] = []
                region_tau[region] = []
                region_slope[region] = []
                region_pr[region] = []
                region_nc[region] = []

            region_bn[region].append(res["bracket_norm"])
            region_tau[region].append(tau)
            region_slope[region].append(slope)
            region_pr[region].append(pr)
            region_nc[region].append(full_activity.shape[1])

    print(f"\n{'='*70}")
    print(f"MIMIC MECHANISM DISCRIMINATOR RESULTS")
    print(f"{'='*70}")

    summary = {}
    for region in sorted(region_bn.keys()):
        summary[region] = {
            "bn_mean": float(np.mean(region_bn[region])),
            "tau_mean": float(np.mean(region_tau[region])),
            "slope_mean": float(np.mean(region_slope[region])),
            "pr_mean": float(np.mean(region_pr[region])),
            "nc_mean": float(np.mean(region_nc[region])),
            "n_sessions": len(region_bn[region]),
        }
        s = summary[region]
        print(f"  {region:8s}: BN={s['bn_mean']:.4f} tau={s['tau_mean']:.1f}bins "
              f"slope={s['slope_mean']:.2f} PR={s['pr_mean']:.1f} nc={s['nc_mean']:.0f}")

    silencing_regions = [r for r in SILENCING_EFFECTS if r in summary]

    print(f"\nCROSS-METRIC CORRELATIONS (all regions, n={len(summary)}):")

    all_bn = [summary[r]["bn_mean"] for r in summary]
    all_tau = [summary[r]["tau_mean"] for r in summary]
    all_slope = [summary[r]["slope_mean"] for r in summary]
    all_pr = [summary[r]["pr_mean"] for r in summary]
    all_nc = [summary[r]["nc_mean"] for r in summary]

    rho_bn_tau, p_bn_tau = spearmanr(all_bn, all_tau)
    rho_bn_slope, p_bn_slope = spearmanr(all_bn, all_slope)
    rho_bn_pr, p_bn_pr = spearmanr(all_bn, all_pr)

    print(f"  BN vs tau: rho={rho_bn_tau:+.3f} p={p_bn_tau:.4f}")
    print(f"  BN vs slope: rho={rho_bn_slope:+.3f} p={p_bn_slope:.4f}")
    print(f"  BN vs PR: rho={rho_bn_pr:+.3f} p={p_bn_pr:.4f}")

    interpretation = "UNKNOWN"
    if rho_bn_tau > 0.2 and rho_bn_slope > -0.2:
        interpretation = "COMPUTATION_NODES"
        print(f"\n  INTERPRETATION: High-BN regions have LONG timescales and "
              f"GRADUAL spectra → recurrent computation nodes (not relays).")
    elif rho_bn_tau < -0.2 and rho_bn_slope < -0.2:
        interpretation = "INPUT_DOMINATED"
        print(f"\n  INTERPRETATION: High-BN regions have SHORT timescales and "
              f"STEEP spectra → input-dominated relays (mimic mechanism present).")
    else:
        interpretation = "MIXED"
        print(f"\n  INTERPRETATION: Mixed pattern — no clear discrimination "
              f"between computation nodes and relays.")

    if len(silencing_regions) >= 5:
        print(f"\nSILENCING REGION CORRELATIONS (n={len(silencing_regions)}):")
        s_bn = [summary[r]["bn_mean"] for r in silencing_regions]
        s_tau = [summary[r]["tau_mean"] for r in silencing_regions]
        s_slope = [summary[r]["slope_mean"] for r in silencing_regions]
        s_pr = [summary[r]["pr_mean"] for r in silencing_regions]
        s_nc = [summary[r]["nc_mean"] for r in silencing_regions]
        s_eff = [SILENCING_EFFECTS[r] for r in silencing_regions]

        rho_tau_sil, p_tau_sil = spearmanr(s_tau, s_eff)
        partial_tau_sil = partial_spearman(np.array(s_tau), np.array(s_eff), np.array(s_nc))
        rho_slope_sil, p_slope_sil = spearmanr(s_slope, s_eff)
        partial_slope_sil = partial_spearman(np.array(s_slope), np.array(s_eff), np.array(s_nc))
        rho_pr_sil, p_pr_sil = spearmanr(s_pr, s_eff)
        partial_pr_sil = partial_spearman(np.array(s_pr), np.array(s_eff), np.array(s_nc))

        print(f"  tau vs silencing: rho={rho_tau_sil:+.3f} partial={partial_tau_sil:+.3f}")
        print(f"  slope vs silencing: rho={rho_slope_sil:+.3f} partial={partial_slope_sil:+.3f}")
        print(f"  PR vs silencing: rho={rho_pr_sil:+.3f} partial={partial_pr_sil:+.3f}")

        silencing_corr = {
            "tau_rho": float(rho_tau_sil), "tau_partial": float(partial_tau_sil),
            "slope_rho": float(rho_slope_sil), "slope_partial": float(partial_slope_sil),
            "pr_rho": float(rho_pr_sil), "pr_partial": float(partial_pr_sil),
            "n": len(silencing_regions), "regions": silencing_regions,
        }
    else:
        silencing_corr = {"n": len(silencing_regions), "error": "too few"}

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "mimic_mechanism_discriminator",
        "n_regions": len(summary),
        "interpretation": interpretation,
        "cross_metric_correlations": {
            "bn_vs_tau": {"rho": float(rho_bn_tau), "p": float(p_bn_tau)},
            "bn_vs_slope": {"rho": float(rho_bn_slope), "p": float(p_bn_slope)},
            "bn_vs_pr": {"rho": float(rho_bn_pr), "p": float(p_bn_pr)},
        },
        "silencing_correlations": silencing_corr,
        "region_summary": summary,
    }

    save_results("mimic_mechanism", results)
    print(f"\nDone.")


if __name__ == "__main__":
    run()
