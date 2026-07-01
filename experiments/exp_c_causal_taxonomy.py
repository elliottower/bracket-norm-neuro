"""Causal role taxonomy: classify regions as relay/computation/gating/readout.

Derives predictions from the task causal graph (evidence → decision → choice → action)
and tests them against the Steinmetz data.

Experiments:
  1. SEIS-vs-bracket_norm dissociation — validates relay vs computation node split
  2. 73-region spatiotemporal bracket_norm/SEIS profile — tests task-derived predictions
  3. Single-trial RT prediction from per-trial geometry — tests whether bracket_norm
     measures trial-by-trial computation mechanics
  4. Error trial geometric collapse — do computation nodes show more geometry change
     on error trials than relays?
  5. Evidence manifold dimensionality — computation nodes should use more dimensions
     to encode evidence than relays

Every correlation is partialed for neuron count.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_bundle import SILENCING_EFFECTS, save_results


# Steinmetz silencing effects for 9 overlapping regions
SILENCING_REGIONS = {
    "ACA": 0.1451, "MOs": 0.1529, "ORB": 0.3085, "PL": 0.3333,
    "RSP": 0.1421, "VISam": 0.0818, "VISl": 0.1722,
    "VISp": 0.1414, "VISpm": 0.2248,
}

# Task-causal-graph predictions for 73-region profile
# Groups: sensory, association, motor, subcortical
REGION_GROUPS = {
    "sensory": ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm", "VISa"],
    "association": ["ACA", "PL", "ORB", "RSP", "ILA", "MOp", "MOs"],
    "motor": ["MOp", "MOs", "SSp", "SSs"],
    "subcortical": ["CP", "GPe", "SNr", "SCm", "SCig", "MRN", "ZI",
                     "LP", "LD", "MD", "VPM", "VPL", "PO", "LGd"],
    "hippocampal": ["CA1", "CA3", "DG", "SUB", "POST"],
}

# Predictions from task causal graph
GROUP_PREDICTIONS = {
    "sensory":      {"seis": "high", "bracket_norm": "low",  "role": "relay"},
    "association":  {"seis": "moderate", "bracket_norm": "high", "role": "computation"},
    "motor":        {"seis": "high", "bracket_norm": "low",  "role": "readout"},
    "subcortical":  {"seis": "moderate", "bracket_norm": "moderate", "role": "relay/gating"},
    "hippocampal":  {"seis": "low", "bracket_norm": "low",  "role": "context"},
}

DECISION_WINDOW = (15, 35)  # 150-350ms post-stimulus


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rho_xy = np.corrcoef(rx, ry)[0, 1]
    rho_xz = np.corrcoef(rx, rz)[0, 1]
    rho_yz = np.corrcoef(ry, rz)[0, 1]
    num = rho_xy - rho_xz * rho_yz
    den = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    return num / den if den > 1e-12 else 0.0


def load_steinmetz():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.steinmetz import load_all
    return load_all()


def get_activity(spks, region_mask, trial_mask):
    """Extract decision-window activity. spks is (n_neurons, n_trials, n_time_bins)."""
    return spks[region_mask][:, trial_mask][:, :, DECISION_WINDOW[0]:DECISION_WINDOW[1]].mean(axis=2).T


def compute_bracket_norm(activity, choice, evidence, n_quartiles=4):
    quartile_edges = np.percentile(evidence, np.linspace(0, 100, n_quartiles + 1))
    displacements = []
    for q in range(n_quartiles):
        lo, hi = quartile_edges[q], quartile_edges[q + 1]
        q_mask = (evidence >= lo) & (evidence <= hi) if q == n_quartiles - 1 else (evidence >= lo) & (evidence < hi)
        c0 = activity[q_mask & (choice == 0)]
        c1 = activity[q_mask & (choice == 1)]
        if len(c0) < 3 or len(c1) < 3:
            return None
        displacements.append(c1.mean(axis=0) - c0.mean(axis=0))
    displacements = np.array(displacements)
    mean_disp = displacements.mean(axis=0)
    deviations = displacements - mean_disp
    return float(np.sqrt(np.mean(np.sum(deviations**2, axis=1))))


def compute_seis(activity, choice, n_splits=50):
    n_trials = len(choice)
    cosines = []
    for _ in range(n_splits):
        perm = np.random.permutation(n_trials)
        half = n_trials // 2
        a_idx, b_idx = perm[:half], perm[half:]
        for idx in [a_idx, b_idx]:
            if (choice[idx] == 0).sum() < 3 or (choice[idx] == 1).sum() < 3:
                return None
        d_a = activity[a_idx][choice[a_idx] == 1].mean(0) - activity[a_idx][choice[a_idx] == 0].mean(0)
        d_b = activity[b_idx][choice[b_idx] == 1].mean(0) - activity[b_idx][choice[b_idx] == 0].mean(0)
        na, nb = np.linalg.norm(d_a), np.linalg.norm(d_b)
        if na > 1e-12 and nb > 1e-12:
            cosines.append(float(np.dot(d_a, d_b) / (na * nb)))
    return float(np.mean(cosines)) if cosines else None


# ---- Experiment 1: 73-region bracket_norm + SEIS profile ----
def exp_full_brain_profile(sessions):
    """Compute bracket_norm and SEIS for ALL regions across all sessions."""
    print("\n=== Exp 1: Full-brain geometric profile (all regions) ===")

    region_bn = defaultdict(list)
    region_seis = defaultdict(list)
    region_nc = defaultdict(list)

    for sess in sessions:
        spks = sess["spks"]  # (n_neurons, n_trials, n_time_bins)
        choice = np.asarray(sess["response"]).flatten()
        contrast_l = np.asarray(sess["contrast_left"]).flatten()
        contrast_r = np.asarray(sess["contrast_right"]).flatten()
        brain_area = np.array([str(a) for a in sess["brain_area"]])

        evidence = np.abs(contrast_l - contrast_r)
        valid = (choice == 1) | (choice == -1)
        choice_binary = np.zeros(len(choice), dtype=int)
        choice_binary[choice == 1] = 1
        choice_binary[choice == -1] = 0

        unique_areas = np.unique(brain_area)
        for region in unique_areas:
            mask = brain_area == region
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            activity = get_activity(spks, mask, valid)
            c = choice_binary[valid]
            e = evidence[valid]

            if len(np.unique(c)) < 2:
                continue

            bn = compute_bracket_norm(activity, c, e)
            seis = compute_seis(activity, c)

            if bn is not None:
                region_bn[region].append(bn)
            if seis is not None:
                region_seis[region].append(seis)
            region_nc[region].append(n_neurons)

    results = {}
    for region in sorted(set(list(region_bn.keys()) + list(region_seis.keys()))):
        r = {}
        if region_bn[region]:
            r["bracket_norm"] = float(np.mean(region_bn[region]))
        if region_seis[region]:
            r["seis"] = float(np.mean(region_seis[region]))
        if region_nc[region]:
            r["neuron_count"] = float(np.mean(region_nc[region]))
            r["n_sessions"] = len(region_nc[region])
        results[region] = r

    # Test group predictions
    print(f"\n  {len(results)} regions with data")
    print(f"\n  {'Group':>15s}  {'n_regions':>9s}  {'BN_mean':>8s}  {'SEIS_mean':>9s}  {'Predicted':>10s}")
    print(f"  {'-'*60}")

    group_stats = {}
    for group, group_regions in REGION_GROUPS.items():
        bns, seiss = [], []
        for r in group_regions:
            if r in results and "bracket_norm" in results[r]:
                bns.append(results[r]["bracket_norm"])
            if r in results and "seis" in results[r]:
                seiss.append(results[r]["seis"])

        pred = GROUP_PREDICTIONS.get(group, {})
        if bns or seiss:
            group_stats[group] = {
                "bracket_norm_mean": float(np.mean(bns)) if bns else None,
                "seis_mean": float(np.mean(seiss)) if seiss else None,
                "n_regions": len(bns),
                "predicted_bn": pred.get("bracket_norm", "?"),
                "predicted_seis": pred.get("seis", "?"),
                "predicted_role": pred.get("role", "?"),
            }
            print(f"  {group:>15s}  {len(bns):>9d}  "
                  f"{np.mean(bns) if bns else float('nan'):>8.4f}  "
                  f"{np.mean(seiss) if seiss else float('nan'):>9.3f}  "
                  f"BN={pred.get('bracket_norm','?'):>4s} SEIS={pred.get('seis','?'):>4s}")

    # Test: do association regions have higher bracket_norm than sensory?
    assoc_bns = [results[r]["bracket_norm"] for r in REGION_GROUPS["association"]
                 if r in results and "bracket_norm" in results[r]]
    sensory_bns = [results[r]["bracket_norm"] for r in REGION_GROUPS["sensory"]
                   if r in results and "bracket_norm" in results[r]]

    if assoc_bns and sensory_bns:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(assoc_bns, sensory_bns, alternative="greater")
        print(f"\n  Association > Sensory bracket_norm: U={u:.0f}, p={p:.3f}")
        print(f"    Association mean: {np.mean(assoc_bns):.4f} (n={len(assoc_bns)})")
        print(f"    Sensory mean: {np.mean(sensory_bns):.4f} (n={len(sensory_bns)})")
        group_stats["assoc_vs_sensory_bn"] = {
            "U": float(u), "p": float(p),
            "assoc_mean": float(np.mean(assoc_bns)),
            "sensory_mean": float(np.mean(sensory_bns)),
        }

    return results, group_stats


# ---- Experiment 2: Single-trial RT prediction from geometric deviation ----
def exp_single_trial_rt(sessions):
    """For each trial, compute angle of choice displacement from mean direction.
    Correlate with RT. High-bracket_norm regions should show stronger correlation.

    This is the key test: if bracket_norm measures trial-by-trial computation,
    trials where the geometry deviates from the mean should be slower (harder).
    """
    print("\n=== Exp 2: Single-trial RT prediction from geometric deviation ===")

    region_correlations = defaultdict(list)
    region_bn_values = defaultdict(list)
    region_nc = defaultdict(list)

    for sess in sessions:
        spks = sess["spks"]
        choice = np.asarray(sess["response"]).flatten()
        contrast_l = np.asarray(sess["contrast_left"]).flatten()
        contrast_r = np.asarray(sess["contrast_right"]).flatten()
        rt = np.asarray(sess["reaction_time"])
        brain_area = np.array([str(a) for a in sess["brain_area"]])

        if rt.ndim == 2:
            rt_vals = rt[:, 0]
        else:
            rt_vals = rt

        evidence = np.abs(contrast_l - contrast_r)
        valid = ((choice == 1) | (choice == -1)) & np.isfinite(rt_vals) & (rt_vals > 0) & (rt_vals < 2000)

        if valid.sum() < 30:
            continue

        choice_binary = np.zeros(len(choice), dtype=int)
        choice_binary[choice == 1] = 1
        choice_binary[choice == -1] = 0

        unique_areas = np.unique(brain_area)
        for region in unique_areas:
            mask = brain_area == region
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            activity = get_activity(spks, mask, valid)
            c = choice_binary[valid]
            e = evidence[valid]
            rt_valid = rt_vals[valid]

            if len(np.unique(c)) < 2:
                continue

            # Compute mean choice displacement
            mean_disp = activity[c == 1].mean(0) - activity[c == 0].mean(0)
            mean_disp_norm = np.linalg.norm(mean_disp)
            if mean_disp_norm < 1e-12:
                continue

            # For each trial, compute the choice-conditioned displacement
            # relative to the nearest class centroid approach
            centroid_0 = activity[c == 0].mean(0)
            centroid_1 = activity[c == 1].mean(0)

            # Per-trial deviation: how far is this trial's projection onto
            # the choice axis from the expected value?
            choice_axis = mean_disp / mean_disp_norm
            projections = activity @ choice_axis

            # Expected projection: positive for class 1, negative for class 0
            expected = np.where(c == 1, projections[c == 1].mean(), projections[c == 0].mean())
            deviation = np.abs(projections - expected)

            # Also: angle deviation from mean direction per evidence quartile
            bn = compute_bracket_norm(activity, c, e)

            # Correlate deviation with RT
            rho_dev_rt, p_dev_rt = spearmanr(deviation, rt_valid)

            region_correlations[region].append({
                "rho_deviation_rt": float(rho_dev_rt),
                "p_deviation_rt": float(p_dev_rt),
                "n_trials": len(rt_valid),
            })

            if bn is not None:
                region_bn_values[region].append(bn)
            region_nc[region].append(n_neurons)

    # Aggregate per region
    results = {}
    for region in region_correlations:
        corrs = region_correlations[region]
        mean_rho = float(np.mean([c["rho_deviation_rt"] for c in corrs]))
        mean_bn = float(np.mean(region_bn_values[region])) if region_bn_values[region] else None
        mean_nc = float(np.mean(region_nc[region]))

        results[region] = {
            "rho_deviation_rt": mean_rho,
            "bracket_norm": mean_bn,
            "neuron_count": mean_nc,
            "n_sessions": len(corrs),
        }

    # Key test: do high-bracket_norm regions show stronger deviation→RT correlation?
    bn_vals, rho_vals, nc_vals, regions_used = [], [], [], []
    for region, r in results.items():
        if r["bracket_norm"] is not None:
            bn_vals.append(r["bracket_norm"])
            rho_vals.append(r["rho_deviation_rt"])
            nc_vals.append(r["neuron_count"])
            regions_used.append(region)

    if len(bn_vals) >= 5:
        rho_meta, p_meta = spearmanr(bn_vals, rho_vals)
        partial = partial_spearman(np.array(bn_vals), np.array(rho_vals), np.array(nc_vals))
        print(f"\n  bracket_norm vs deviation→RT correlation:")
        print(f"    rho={rho_meta:+.3f}, p={p_meta:.3f}, partial(nc)={partial:+.3f}")
        print(f"    n={len(bn_vals)} regions")

        results["_meta"] = {
            "bn_vs_rt_rho": float(rho_meta),
            "bn_vs_rt_p": float(p_meta),
            "bn_vs_rt_partial": float(partial),
            "n_regions": len(bn_vals),
        }

    # Print top/bottom regions
    sorted_regions = sorted([(r, d) for r, d in results.items() if r != "_meta" and d["bracket_norm"] is not None],
                           key=lambda x: x[1]["bracket_norm"], reverse=True)
    print(f"\n  {'Region':>10s}  {'BN':>6s}  {'dev→RT':>7s}  {'nc':>4s}")
    for region, d in sorted_regions[:10]:
        print(f"  {region:>10s}  {d['bracket_norm']:.4f}  {d['rho_deviation_rt']:+.3f}  {d['neuron_count']:.0f}")

    # Also test on silencing regions specifically
    sil_bn, sil_rho, sil_sil, sil_nc = [], [], [], []
    for region, effect in SILENCING_REGIONS.items():
        if region in results and results[region]["bracket_norm"] is not None:
            sil_bn.append(results[region]["bracket_norm"])
            sil_rho.append(results[region]["rho_deviation_rt"])
            sil_sil.append(effect)
            sil_nc.append(results[region]["neuron_count"])

    if len(sil_bn) >= 4:
        partial_bn_sil = partial_spearman(np.array(sil_bn), np.array(sil_sil), np.array(sil_nc))
        partial_rho_sil = partial_spearman(np.array(sil_rho), np.array(sil_sil), np.array(sil_nc))
        print(f"\n  Silencing regions (n={len(sil_bn)}):")
        print(f"    bracket_norm vs silencing partial(nc): {partial_bn_sil:+.3f}")
        print(f"    dev→RT_corr vs silencing partial(nc): {partial_rho_sil:+.3f}")

        results["_silencing"] = {
            "bn_vs_silencing_partial": float(partial_bn_sil),
            "rt_corr_vs_silencing_partial": float(partial_rho_sil),
            "n": len(sil_bn),
        }

    return results


# ---- Experiment 3: Evidence manifold dimensionality ----
def exp_evidence_dimensionality(sessions):
    """How many dimensions does each region use to encode evidence strength?

    Computation nodes should have higher evidence manifold dimensionality
    (participation ratio of evidence-aligned PCA dimensions).
    """
    print("\n=== Exp 3: Evidence manifold dimensionality ===")

    region_dims = defaultdict(list)
    region_bn = defaultdict(list)
    region_nc = defaultdict(list)

    for sess in sessions:
        spks = sess["spks"]
        choice = np.asarray(sess["response"]).flatten()
        contrast_l = np.asarray(sess["contrast_left"]).flatten()
        contrast_r = np.asarray(sess["contrast_right"]).flatten()
        brain_area = np.array([str(a) for a in sess["brain_area"]])

        evidence = np.abs(contrast_l - contrast_r)
        valid = (choice == 1) | (choice == -1)
        choice_binary = np.zeros(len(choice), dtype=int)
        choice_binary[choice == 1] = 1
        choice_binary[choice == -1] = 0

        unique_areas = np.unique(brain_area)
        for region in unique_areas:
            mask = brain_area == region
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            activity = get_activity(spks, mask, valid)
            c = choice_binary[valid]
            e = evidence[valid]

            if len(np.unique(c)) < 2:
                continue

            # Compute evidence-conditioned centroids
            unique_evidence = np.unique(e)
            if len(unique_evidence) < 3:
                continue

            centroids = []
            for ev in unique_evidence:
                ev_mask = e == ev
                if ev_mask.sum() >= 3:
                    centroids.append(activity[ev_mask].mean(0))

            if len(centroids) < 3:
                continue

            centroids = np.array(centroids)
            centroids_centered = centroids - centroids.mean(0)

            # Participation ratio: measure of effective dimensionality
            if centroids_centered.shape[0] < 2:
                continue

            try:
                cov = np.cov(centroids_centered.T)
                if cov.ndim < 2:
                    continue
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = eigenvalues[eigenvalues > 1e-12]
                if len(eigenvalues) == 0:
                    continue
                pr = (eigenvalues.sum())**2 / (eigenvalues**2).sum()
            except np.linalg.LinAlgError:
                continue

            bn = compute_bracket_norm(activity, c, e)

            region_dims[region].append(float(pr))
            if bn is not None:
                region_bn[region].append(bn)
            region_nc[region].append(n_neurons)

    results = {}
    for region in region_dims:
        r = {
            "evidence_pr": float(np.mean(region_dims[region])),
            "n_sessions": len(region_dims[region]),
            "neuron_count": float(np.mean(region_nc[region])),
        }
        if region_bn[region]:
            r["bracket_norm"] = float(np.mean(region_bn[region]))
        results[region] = r

    # Test: bracket_norm vs evidence dimensionality
    bn_vals, pr_vals, nc_vals = [], [], []
    for region, r in results.items():
        if "bracket_norm" in r:
            bn_vals.append(r["bracket_norm"])
            pr_vals.append(r["evidence_pr"])
            nc_vals.append(r["neuron_count"])

    if len(bn_vals) >= 5:
        rho, p = spearmanr(bn_vals, pr_vals)
        partial = partial_spearman(np.array(bn_vals), np.array(pr_vals), np.array(nc_vals))
        print(f"\n  bracket_norm vs evidence dimensionality:")
        print(f"    rho={rho:+.3f}, p={p:.3f}, partial(nc)={partial:+.3f}")
        print(f"    n={len(bn_vals)} regions")

        results["_meta"] = {
            "bn_vs_pr_rho": float(rho),
            "bn_vs_pr_p": float(p),
            "bn_vs_pr_partial": float(partial),
            "n": len(bn_vals),
        }

    # Print top/bottom
    sorted_regions = sorted([(r, d) for r, d in results.items() if r != "_meta" and "bracket_norm" in d],
                           key=lambda x: x[1]["bracket_norm"], reverse=True)
    print(f"\n  {'Region':>10s}  {'BN':>6s}  {'Ev_PR':>6s}  {'nc':>4s}")
    for region, d in sorted_regions[:10]:
        print(f"  {region:>10s}  {d['bracket_norm']:.4f}  {d['evidence_pr']:.2f}  {d['neuron_count']:.0f}")

    return results


# ---- Experiment 4: Temporal profile of bracket_norm ----
def exp_temporal_bracket_norm(sessions):
    """Compute bracket_norm in sliding windows across time.

    Prediction: computation nodes peak early (decision window),
    readout nodes peak late (post-commitment).
    """
    print("\n=== Exp 4: Temporal bracket_norm profile ===")

    WINDOW_SIZE = 10  # bins = 100ms
    STEP = 5          # bins = 50ms
    N_WINDOWS = (250 - WINDOW_SIZE) // STEP

    region_profiles = defaultdict(lambda: defaultdict(list))
    region_nc = defaultdict(list)

    for sess in sessions:
        spks = sess["spks"]
        choice = np.asarray(sess["response"]).flatten()
        contrast_l = np.asarray(sess["contrast_left"]).flatten()
        contrast_r = np.asarray(sess["contrast_right"]).flatten()
        brain_area = np.array([str(a) for a in sess["brain_area"]])

        evidence = np.abs(contrast_l - contrast_r)
        valid = (choice == 1) | (choice == -1)
        choice_binary = np.zeros(len(choice), dtype=int)
        choice_binary[choice == 1] = 1
        choice_binary[choice == -1] = 0

        for region in set(list(SILENCING_REGIONS.keys()) + ["CA1", "CP", "SCm", "MOp"]):
            mask = brain_area == region
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            region_spks = spks[mask][:, valid]  # (neurons, trials, time)
            c = choice_binary[valid]
            e = evidence[valid]

            if len(np.unique(c)) < 2:
                continue

            for w in range(N_WINDOWS):
                t_start = w * STEP
                t_end = t_start + WINDOW_SIZE
                activity = region_spks[:, :, t_start:t_end].mean(axis=2).T  # (trials, neurons)

                bn = compute_bracket_norm(activity, c, e)
                if bn is not None:
                    region_profiles[region][w].append(bn)

            region_nc[region].append(n_neurons)

    # Aggregate and find peak times
    results = {}
    print(f"\n  {'Region':>10s}  {'Peak_bin':>8s}  {'Peak_BN':>8s}  {'Mean_BN':>8s}  {'nc':>4s}")
    for region in sorted(region_profiles.keys()):
        profile = []
        for w in range(N_WINDOWS):
            vals = region_profiles[region].get(w, [])
            profile.append(float(np.mean(vals)) if vals else 0.0)

        peak_bin = int(np.argmax(profile))
        peak_time_ms = peak_bin * STEP * 10  # 10ms per bin

        results[region] = {
            "profile": profile,
            "peak_bin": peak_bin,
            "peak_time_ms": peak_time_ms,
            "peak_bn": float(max(profile)),
            "mean_bn": float(np.mean(profile)),
            "neuron_count": float(np.mean(region_nc[region])) if region_nc[region] else 0,
        }
        print(f"  {region:>10s}  {peak_time_ms:>6d}ms  {max(profile):>8.4f}  {np.mean(profile):>8.4f}  "
              f"{results[region]['neuron_count']:.0f}")

    # Test prediction: association peak early, motor peak late
    assoc_peaks, sensory_peaks = [], []
    for group, regions in [("association", REGION_GROUPS["association"]),
                            ("sensory", REGION_GROUPS["sensory"])]:
        for r in regions:
            if r in results:
                if group == "association":
                    assoc_peaks.append(results[r]["peak_time_ms"])
                else:
                    sensory_peaks.append(results[r]["peak_time_ms"])

    if assoc_peaks and sensory_peaks:
        print(f"\n  Association peak: {np.mean(assoc_peaks):.0f}ms (n={len(assoc_peaks)})")
        print(f"  Sensory peak: {np.mean(sensory_peaks):.0f}ms (n={len(sensory_peaks)})")

    return results


def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Causal Role Taxonomy Experiments")
    print("=" * 80)

    sessions = load_steinmetz()
    print(f"Loaded {len(sessions)} sessions")

    # Run all experiments
    profile_results, group_stats = exp_full_brain_profile(sessions)
    rt_results = exp_single_trial_rt(sessions)
    dim_results = exp_evidence_dimensionality(sessions)
    temporal_results = exp_temporal_bracket_norm(sessions)

    # ---- Final taxonomy assignment ----
    print("\n" + "=" * 80)
    print("CAUSAL ROLE TAXONOMY ASSIGNMENT")
    print("=" * 80)

    taxonomy = {}
    for region in sorted(profile_results.keys()):
        r = profile_results[region]
        bn = r.get("bracket_norm")
        seis = r.get("seis")
        nc = r.get("neuron_count", 0)

        if bn is None or seis is None:
            continue

        # Classify based on bracket_norm and SEIS
        if bn > 0.15 and seis < 0.7:
            role = "computation"
        elif bn < 0.12 and seis > 0.6:
            role = "relay"
        elif bn < 0.10 and seis < 0.5:
            role = "readout/context"
        else:
            role = "mixed"

        # Check against group prediction
        predicted_group = None
        for group, members in REGION_GROUPS.items():
            if region in members:
                predicted_group = group
                break

        taxonomy[region] = {
            "bracket_norm": bn,
            "seis": seis,
            "assigned_role": role,
            "predicted_group": predicted_group,
            "predicted_role": GROUP_PREDICTIONS.get(predicted_group, {}).get("role", "unknown"),
            "neuron_count": nc,
        }

    # Print taxonomy
    print(f"\n  {'Region':>10s}  {'BN':>6s}  {'SEIS':>5s}  {'Assigned':>12s}  {'Predicted':>12s}  {'Match':>5s}")
    print(f"  {'-'*60}")
    n_match = 0
    n_total = 0
    for region in sorted(taxonomy.keys()):
        t = taxonomy[region]
        match = "?" if t["predicted_role"] == "unknown" else ("Y" if t["assigned_role"].startswith(t["predicted_role"].split("/")[0]) else "N")
        if match != "?":
            n_total += 1
            if match == "Y":
                n_match += 1
        print(f"  {region:>10s}  {t['bracket_norm']:.4f}  {t['seis']:.3f}  "
              f"{t['assigned_role']:>12s}  {t['predicted_role']:>12s}  {match:>5s}")

    if n_total > 0:
        print(f"\n  Prediction accuracy: {n_match}/{n_total} = {n_match/n_total:.1%}")

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "causal_taxonomy",
        "full_brain_profile": {k: v for k, v in profile_results.items()},
        "group_stats": group_stats,
        "single_trial_rt": {k: v for k, v in rt_results.items() if k != "_meta" and k != "_silencing"},
        "single_trial_rt_meta": rt_results.get("_meta", {}),
        "single_trial_rt_silencing": rt_results.get("_silencing", {}),
        "evidence_dimensionality": {k: v for k, v in dim_results.items() if k != "_meta"},
        "evidence_dimensionality_meta": dim_results.get("_meta", {}),
        "temporal_profiles": temporal_results,
        "taxonomy": taxonomy,
    }

    save_results("causal_taxonomy", all_results)

    return all_results


if __name__ == "__main__":
    main()
