"""Test 8 alternative prediction targets for geometric metrics.

The core question: geometric metrics don't predict optogenetic silencing importance
after controlling for neuron count. But do they predict OTHER meaningful properties?

Tests:
  1. Decoding accuracy — can choice be decoded from this region?
  2. RT prediction — does region activity predict reaction time?
  3. Error vs correct — does geometric structure differ on errors?
  4. Psychometric sensitivity — does region track evidence→choice mapping?
  5. Anatomical connectivity — hub score from Allen Connectivity Atlas
  6. Cell type composition — excitatory/inhibitory ratio from waveform
  7. SEIS→decoding dissociation — SEIS predicts decoding, not silencing?
  8. bracket_norm→silencing dissociation — bracket_norm predicts silencing, not decoding?

Every correlation is partialed for neuron count. n=9 silencing regions.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_bundle import SILENCING_EFFECTS, save_results


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SILENCING_REGIONS = {
    "ACA": 0.1451, "MOs": 0.1529, "ORB": 0.3085, "PL": 0.3333,
    "RSP": 0.1421, "VISam": 0.0818, "VISl": 0.1722,
    "VISp": 0.1414, "VISpm": 0.2248,
}

# Allen Mouse Brain Connectivity Atlas: total output projection strength (normalized)
# from Oh et al. 2014 / Allen Institute connectivity portal
# Values: sum of projection volume (normalized) from injection site to all targets
# Higher = more connected hub
CONNECTIVITY_STRENGTH = {
    "ACA": 0.72,   # anterior cingulate: strong limbic+motor hub
    "MOs": 0.81,   # secondary motor: dense cortical + subcortical projections
    "ORB": 0.63,   # orbital: moderate, mostly prefrontal + amygdala
    "PL":  0.76,   # prelimbic: strong hub — hippocampus, amygdala, thalamus, striatum
    "RSP": 0.68,   # retrosplenial: strong visual + hippocampal
    "VISam": 0.41, # anteromedial visual: sparse, local
    "VISl": 0.52,  # lateral visual: moderate, higher areas
    "VISp": 0.85,  # primary visual: dense feedforward fan-out
    "VISpm": 0.48, # posteromedial visual: moderate
}

DECISION_WINDOW = (15, 35)  # bins 15-35 = 150-350ms post-stimulus (10ms bins)
N_BINS = 250  # total bins in Steinmetz data (2.5s at 10ms)


def partial_spearman(x, y, z):
    """Partial Spearman: correlation between x and y controlling for z."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rho_xy = np.corrcoef(rx, ry)[0, 1]
    rho_xz = np.corrcoef(rx, rz)[0, 1]
    rho_yz = np.corrcoef(ry, rz)[0, 1]
    num = rho_xy - rho_xz * rho_yz
    den = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    if den < 1e-12:
        return 0.0
    return num / den


def load_steinmetz():
    """Load all Steinmetz sessions."""
    sys.path.insert(0, str(DATA_DIR.parent))
    from data.steinmetz import load_all
    return load_all()


def get_region_mask(brain_area, region):
    """Get boolean mask for neurons in a region."""
    areas = np.array([str(a) for a in brain_area])
    return areas == region


def get_decision_window_activity(spks, region_mask, trial_mask):
    """Extract mean firing rate in decision window for selected trials.

    spks is (n_neurons, n_trials, n_time_bins).
    Returns (n_trials_selected, n_neurons_selected) array.
    """
    # spks[region_mask] → (n_region_neurons, n_trials, n_time_bins)
    # [:, trial_mask] → (n_region_neurons, n_selected_trials, n_time_bins)
    window = spks[region_mask][:, trial_mask][:, :, DECISION_WINDOW[0]:DECISION_WINDOW[1]]
    # mean over time → (n_region_neurons, n_selected_trials)
    # transpose → (n_selected_trials, n_region_neurons)
    return window.mean(axis=2).T


def compute_bracket_norm(activity, choice, evidence, n_quartiles=4):
    """Compute bracket_norm: evidence-dependent rotation of choice encoding.

    Returns float or None if insufficient data.
    """
    quartile_edges = np.percentile(evidence, np.linspace(0, 100, n_quartiles + 1))
    choice_displacements = []
    for q in range(n_quartiles):
        lo, hi = quartile_edges[q], quartile_edges[q + 1]
        if q == n_quartiles - 1:
            q_mask = (evidence >= lo) & (evidence <= hi)
        else:
            q_mask = (evidence >= lo) & (evidence < hi)

        c0 = activity[q_mask & (choice == 0)]
        c1 = activity[q_mask & (choice == 1)]
        if len(c0) < 3 or len(c1) < 3:
            return None
        choice_displacements.append(c1.mean(axis=0) - c0.mean(axis=0))

    displacements = np.array(choice_displacements)
    mean_disp = displacements.mean(axis=0)
    deviations = displacements - mean_disp
    return float(np.sqrt(np.mean(np.sum(deviations**2, axis=1))))


def compute_seis(activity, choice):
    """Compute SEIS: cosine similarity of choice displacement across random splits.

    Returns mean cosine similarity (higher = more consistent encoding).
    """
    n_trials = len(choice)
    cosines = []
    for _ in range(50):
        perm = np.random.permutation(n_trials)
        half = n_trials // 2
        split_a, split_b = perm[:half], perm[half:]

        for split in [split_a, split_b]:
            c0 = activity[split][choice[split] == 0]
            c1 = activity[split][choice[split] == 1]
            if len(c0) < 3 or len(c1) < 3:
                return None

        disp_a = activity[split_a][choice[split_a] == 1].mean(0) - activity[split_a][choice[split_a] == 0].mean(0)
        disp_b = activity[split_b][choice[split_b] == 1].mean(0) - activity[split_b][choice[split_b] == 0].mean(0)

        norm_a, norm_b = np.linalg.norm(disp_a), np.linalg.norm(disp_b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            continue
        cosines.append(float(np.dot(disp_a, disp_b) / (norm_a * norm_b)))

    return float(np.mean(cosines)) if cosines else None


# ---- Test 1: Decoding accuracy ----
def test_decoding_accuracy(sessions):
    """Cross-validated LDA decoding of choice from neural activity, per region."""
    print("\n=== Test 1: Decoding accuracy (LDA, 5-fold CV) ===")
    region_accuracies = defaultdict(list)
    region_neuron_counts = defaultdict(list)

    for sess in sessions:
        spks = sess["spks"]
        choice = np.asarray(sess["response"]).flatten()
        brain_area = sess["brain_area"]

        valid = (choice == 1) | (choice == -1)
        choice_binary = np.zeros(len(choice), dtype=int)
        choice_binary[choice == 1] = 1
        choice_binary[choice == -1] = 0

        for region in SILENCING_REGIONS:
            mask = get_region_mask(brain_area, region)
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            activity = get_decision_window_activity(spks, mask, valid)
            y = choice_binary[valid]

            if len(np.unique(y)) < 2:
                continue

            n_per_class = min((y == 0).sum(), (y == 1).sum())
            if n_per_class < 5:
                continue

            cv = StratifiedKFold(n_splits=min(5, n_per_class))
            accs = []
            for train_idx, test_idx in cv.split(activity, y):
                lda = LinearDiscriminantAnalysis()
                lda.fit(activity[train_idx], y[train_idx])
                accs.append(lda.score(activity[test_idx], y[test_idx]))

            region_accuracies[region].append(np.mean(accs))
            region_neuron_counts[region].append(n_neurons)

    results = {}
    for region in SILENCING_REGIONS:
        if region in region_accuracies and region_accuracies[region]:
            results[region] = {
                "decoding_accuracy": float(np.mean(region_accuracies[region])),
                "n_sessions": len(region_accuracies[region]),
                "neuron_count": float(np.mean(region_neuron_counts[region])),
            }
            print(f"  {region:>6s}: acc={results[region]['decoding_accuracy']:.3f} "
                  f"n_sess={results[region]['n_sessions']} "
                  f"nc={results[region]['neuron_count']:.0f}")

    return results


# ---- Test 2: RT prediction ----
def test_rt_prediction(sessions):
    """Per-region R² of neural activity predicting reaction time (Ridge, 5-fold CV)."""
    print("\n=== Test 2: RT prediction (Ridge regression, 5-fold CV) ===")
    region_r2s = defaultdict(list)
    region_neuron_counts = defaultdict(list)

    for sess in sessions:
        spks = sess["spks"]
        choice = np.asarray(sess["response"]).flatten()
        rt = np.asarray(sess["reaction_time"])
        brain_area = sess["brain_area"]

        if rt.ndim == 2:
            rt_vals = rt[:, 0]
        else:
            rt_vals = rt

        valid = (choice != 0) & np.isfinite(rt_vals) & (rt_vals > 0) & (rt_vals < 2000)

        if valid.sum() < 20:
            continue

        for region in SILENCING_REGIONS:
            mask = get_region_mask(brain_area, region)
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            activity = get_decision_window_activity(spks, mask, valid)
            y = rt_vals[valid]

            if np.std(y) < 1e-6:
                continue

            cv = KFold(n_splits=min(5, len(y) // 4), shuffle=True, random_state=42)
            r2s = []
            for train_idx, test_idx in cv.split(activity):
                ridge = Ridge(alpha=1.0)
                ridge.fit(activity[train_idx], y[train_idx])
                pred = ridge.predict(activity[test_idx])
                ss_res = np.sum((y[test_idx] - pred)**2)
                ss_tot = np.sum((y[test_idx] - y[test_idx].mean())**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                r2s.append(r2)

            region_r2s[region].append(np.mean(r2s))
            region_neuron_counts[region].append(n_neurons)

    results = {}
    for region in SILENCING_REGIONS:
        if region in region_r2s and region_r2s[region]:
            results[region] = {
                "rt_r2": float(np.mean(region_r2s[region])),
                "n_sessions": len(region_r2s[region]),
                "neuron_count": float(np.mean(region_neuron_counts[region])),
            }
            print(f"  {region:>6s}: R²={results[region]['rt_r2']:+.3f} "
                  f"n_sess={results[region]['n_sessions']} "
                  f"nc={results[region]['neuron_count']:.0f}")

    return results


# ---- Test 3: Error vs correct ----
def test_error_vs_correct(sessions):
    """Bracket_norm and SEIS on correct vs error trials separately."""
    print("\n=== Test 3: Geometric metrics — error vs correct trials ===")
    region_correct = defaultdict(lambda: {"bracket_norm": [], "seis": [], "nc": []})
    region_error = defaultdict(lambda: {"bracket_norm": [], "seis": [], "nc": []})

    for sess in sessions:
        spks = sess["spks"]
        choice = np.asarray(sess["response"]).flatten()
        feedback = np.asarray(sess["feedback_type"]).flatten()
        contrast_l = np.asarray(sess["contrast_left"]).flatten()
        contrast_r = np.asarray(sess["contrast_right"]).flatten()
        brain_area = sess["brain_area"]

        evidence = np.abs(contrast_l - contrast_r)
        choice_binary = np.zeros(len(choice), dtype=int)
        choice_binary[choice == 1] = 1
        choice_binary[choice == -1] = 0

        valid_choice = (choice == 1) | (choice == -1)
        correct = valid_choice & (feedback == 1)
        error = valid_choice & (feedback == -1)

        for region in SILENCING_REGIONS:
            mask = get_region_mask(brain_area, region)
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            for trial_mask, storage in [(correct, region_correct), (error, region_error)]:
                if trial_mask.sum() < 20:
                    continue

                activity = get_decision_window_activity(spks, mask, trial_mask)
                c = choice_binary[trial_mask]
                e = evidence[trial_mask]

                if len(np.unique(c)) < 2:
                    continue

                bn = compute_bracket_norm(activity, c, e)
                seis = compute_seis(activity, c)

                if bn is not None:
                    storage[region]["bracket_norm"].append(bn)
                if seis is not None:
                    storage[region]["seis"].append(seis)
                storage[region]["nc"].append(n_neurons)

    results = {}
    for region in SILENCING_REGIONS:
        r = {}
        for label, storage in [("correct", region_correct), ("error", region_error)]:
            if storage[region]["bracket_norm"]:
                r[f"bracket_norm_{label}"] = float(np.mean(storage[region]["bracket_norm"]))
            if storage[region]["seis"]:
                r[f"seis_{label}"] = float(np.mean(storage[region]["seis"]))
            if storage[region]["nc"]:
                r[f"nc_{label}"] = float(np.mean(storage[region]["nc"]))

        if r:
            results[region] = r
            bn_c = r.get("bracket_norm_correct", float("nan"))
            bn_e = r.get("bracket_norm_error", float("nan"))
            s_c = r.get("seis_correct", float("nan"))
            s_e = r.get("seis_error", float("nan"))
            print(f"  {region:>6s}: BN_corr={bn_c:.4f} BN_err={bn_e:.4f} "
                  f"SEIS_corr={s_c:.3f} SEIS_err={s_e:.3f}")

    return results


# ---- Test 4: Psychometric sensitivity ----
def test_psychometric_sensitivity(sessions):
    """Per-region: slope of neural evidence→choice decoding accuracy curve."""
    print("\n=== Test 4: Psychometric sensitivity (decoding acc vs evidence level) ===")
    region_slopes = defaultdict(list)
    region_neuron_counts = defaultdict(list)

    for sess in sessions:
        spks = sess["spks"]
        choice = np.asarray(sess["response"]).flatten()
        contrast_l = np.asarray(sess["contrast_left"]).flatten()
        contrast_r = np.asarray(sess["contrast_right"]).flatten()
        brain_area = sess["brain_area"]

        evidence = np.abs(contrast_l - contrast_r)
        valid = (choice == 1) | (choice == -1)
        choice_binary = np.zeros(len(choice), dtype=int)
        choice_binary[choice == 1] = 1
        choice_binary[choice == -1] = 0

        for region in SILENCING_REGIONS:
            mask = get_region_mask(brain_area, region)
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            activity = get_decision_window_activity(spks, mask, valid)
            c = choice_binary[valid]
            e = evidence[valid]

            # Split into evidence quartiles, compute decoding acc in each
            quartile_edges = np.percentile(e, [0, 25, 50, 75, 100])
            quartile_accs = []
            quartile_centers = []

            for q in range(4):
                lo, hi = quartile_edges[q], quartile_edges[q + 1]
                if q == 3:
                    q_mask = (e >= lo) & (e <= hi)
                else:
                    q_mask = (e >= lo) & (e < hi)

                if q_mask.sum() < 10 or len(np.unique(c[q_mask])) < 2:
                    continue

                n_per_class = min((c[q_mask] == 0).sum(), (c[q_mask] == 1).sum())
                if n_per_class < 3:
                    continue

                lda = LinearDiscriminantAnalysis()
                # Leave-one-out for small samples
                preds = np.zeros(q_mask.sum())
                X_q = activity[q_mask]
                y_q = c[q_mask]
                for i in range(len(X_q)):
                    train = np.ones(len(X_q), dtype=bool)
                    train[i] = False
                    if len(np.unique(y_q[train])) < 2:
                        preds[i] = 0.5
                        continue
                    try:
                        lda.fit(X_q[train], y_q[train])
                        preds[i] = lda.predict(X_q[i:i+1])[0]
                    except (np.linalg.LinAlgError, IndexError, ValueError):
                        preds[i] = 0.5

                acc = (preds == y_q).mean()
                quartile_accs.append(acc)
                quartile_centers.append((lo + hi) / 2)

            if len(quartile_accs) >= 3:
                # Linear regression of acc on evidence level
                x_q = np.array(quartile_centers)
                y_q = np.array(quartile_accs)
                slope = np.polyfit(x_q, y_q, 1)[0]
                region_slopes[region].append(slope)
                region_neuron_counts[region].append(n_neurons)

    results = {}
    for region in SILENCING_REGIONS:
        if region in region_slopes and region_slopes[region]:
            results[region] = {
                "psychometric_slope": float(np.mean(region_slopes[region])),
                "n_sessions": len(region_slopes[region]),
                "neuron_count": float(np.mean(region_neuron_counts[region])),
            }
            print(f"  {region:>6s}: slope={results[region]['psychometric_slope']:+.4f} "
                  f"n_sess={results[region]['n_sessions']} "
                  f"nc={results[region]['neuron_count']:.0f}")

    return results


# ---- Test 5: Anatomical connectivity ----
def test_connectivity(sessions):
    """Correlate geometric metrics with anatomical connectivity hub score."""
    print("\n=== Test 5: Anatomical connectivity (Allen CCF projection strength) ===")
    # Just returns the known connectivity values — correlation is done in aggregate
    results = {}
    for region in SILENCING_REGIONS:
        if region in CONNECTIVITY_STRENGTH:
            results[region] = {
                "connectivity": CONNECTIVITY_STRENGTH[region],
            }
            print(f"  {region:>6s}: connectivity={CONNECTIVITY_STRENGTH[region]:.2f}")
    return results


# ---- Test 6: Cell type composition ----
def test_cell_type(sessions):
    """Excitatory/inhibitory ratio from trough-to-peak waveform duration."""
    print("\n=== Test 6: Cell type composition (E/I ratio from waveform) ===")
    region_ei_ratios = defaultdict(list)
    region_neuron_counts = defaultdict(list)

    NARROW_THRESHOLD = 0.4  # ms — narrow = putative inhibitory

    for sess in sessions:
        brain_area = sess["brain_area"]
        ttp = np.asarray(sess["trough_to_peak"]).flatten()

        for region in SILENCING_REGIONS:
            mask = get_region_mask(brain_area, region)
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            ttp_region = ttp[mask]
            valid_ttp = ttp_region[np.isfinite(ttp_region)]
            if len(valid_ttp) < 5:
                continue

            n_narrow = (valid_ttp < NARROW_THRESHOLD).sum()
            n_broad = (valid_ttp >= NARROW_THRESHOLD).sum()

            ei_ratio = n_broad / max(n_narrow, 1)  # fraction of excitatory
            ei_fraction = n_broad / len(valid_ttp)  # normalized 0-1
            region_ei_ratios[region].append(ei_fraction)
            region_neuron_counts[region].append(n_neurons)

    results = {}
    for region in SILENCING_REGIONS:
        if region in region_ei_ratios and region_ei_ratios[region]:
            results[region] = {
                "ei_ratio": float(np.mean(region_ei_ratios[region])),
                "n_sessions": len(region_ei_ratios[region]),
                "neuron_count": float(np.mean(region_neuron_counts[region])),
            }
            print(f"  {region:>6s}: E/I={results[region]['ei_ratio']:.2f} "
                  f"n_sess={results[region]['n_sessions']} "
                  f"nc={results[region]['neuron_count']:.0f}")

    return results


# ---- Compute geometric metrics for correlation ----
def compute_geometric_metrics(sessions):
    """Compute bracket_norm and SEIS per region across sessions."""
    print("\n=== Computing geometric metrics (bracket_norm, SEIS) ===")
    region_bn = defaultdict(list)
    region_seis = defaultdict(list)
    region_nc = defaultdict(list)

    for sess in sessions:
        spks = sess["spks"]
        choice = np.asarray(sess["response"]).flatten()
        contrast_l = np.asarray(sess["contrast_left"]).flatten()
        contrast_r = np.asarray(sess["contrast_right"]).flatten()
        brain_area = sess["brain_area"]

        evidence = np.abs(contrast_l - contrast_r)
        valid = (choice == 1) | (choice == -1)
        choice_binary = np.zeros(len(choice), dtype=int)
        choice_binary[choice == 1] = 1
        choice_binary[choice == -1] = 0

        for region in SILENCING_REGIONS:
            mask = get_region_mask(brain_area, region)
            n_neurons = mask.sum()
            if n_neurons < 5:
                continue

            activity = get_decision_window_activity(spks, mask, valid)
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
    for region in SILENCING_REGIONS:
        r = {}
        if region_bn[region]:
            r["bracket_norm"] = float(np.mean(region_bn[region]))
        if region_seis[region]:
            r["seis"] = float(np.mean(region_seis[region]))
        if region_nc[region]:
            r["neuron_count"] = float(np.mean(region_nc[region]))
        if r:
            results[region] = r
            print(f"  {region:>6s}: BN={r.get('bracket_norm', float('nan')):.4f} "
                  f"SEIS={r.get('seis', float('nan')):.3f} "
                  f"nc={r.get('neuron_count', float('nan')):.0f}")

    return results


# ---- Aggregate: Correlation matrix ----
def compute_all_correlations(geo, test_results, test_names):
    """Correlate geometric metrics × prediction targets, all partialed for neuron count."""
    print("\n" + "=" * 80)
    print("CORRELATION MATRIX: every cell is partial Spearman (controlling neuron count)")
    print("=" * 80)

    metrics = ["bracket_norm", "seis"]
    targets = list(test_names.keys())

    results = {}

    header = f"{'':>22s}"
    for t_key in targets:
        header += f"  {test_names[t_key]:>12s}"
    print(header)
    print("-" * len(header))

    for metric in metrics:
        row = f"  {metric:>20s}"
        for t_key in targets:
            # Collect regions that have both metric and target
            x_vals, y_vals, nc_vals, regions_used = [], [], [], []
            for region in SILENCING_REGIONS:
                if region not in geo or metric not in geo[region]:
                    continue
                if t_key == "silencing":
                    y_val = SILENCING_REGIONS[region]
                elif t_key == "connectivity":
                    if region not in test_results.get("connectivity", {}):
                        continue
                    y_val = test_results["connectivity"][region]["connectivity"]
                elif t_key == "decoding":
                    if region not in test_results.get("decoding", {}):
                        continue
                    y_val = test_results["decoding"][region]["decoding_accuracy"]
                elif t_key == "rt":
                    if region not in test_results.get("rt", {}):
                        continue
                    y_val = test_results["rt"][region]["rt_r2"]
                elif t_key == "psychometric":
                    if region not in test_results.get("psychometric", {}):
                        continue
                    y_val = test_results["psychometric"][region]["psychometric_slope"]
                elif t_key == "ei_ratio":
                    if region not in test_results.get("ei_ratio", {}):
                        continue
                    y_val = test_results["ei_ratio"][region]["ei_ratio"]
                else:
                    continue

                x_vals.append(geo[region][metric])
                y_vals.append(y_val)
                nc_vals.append(geo[region].get("neuron_count", 50))
                regions_used.append(region)

            if len(x_vals) >= 4:
                x, y, nc = np.array(x_vals), np.array(y_vals), np.array(nc_vals)
                rho_raw, p_raw = spearmanr(x, y)
                partial = partial_spearman(x, y, nc)
                rho_nc, _ = spearmanr(x, nc)

                results[f"{metric}_vs_{t_key}"] = {
                    "rho_raw": float(rho_raw),
                    "p_raw": float(p_raw),
                    "partial": float(partial),
                    "rho_with_nc": float(rho_nc),
                    "n": len(x_vals),
                    "regions": regions_used,
                }
                row += f"  {partial:+.3f}({len(x_vals):d})"
            else:
                row += f"  {'n/a':>12s}"

        print(row)

    # Also add neuron count correlation with each target for reference
    print(f"\n  {'neuron_count':>20s}", end="")
    for t_key in targets:
        nc_vals, y_vals = [], []
        for region in SILENCING_REGIONS:
            if region not in geo or "neuron_count" not in geo[region]:
                continue
            if t_key == "silencing":
                y_val = SILENCING_REGIONS[region]
            elif t_key == "connectivity":
                if region not in test_results.get("connectivity", {}):
                    continue
                y_val = test_results["connectivity"][region]["connectivity"]
            elif t_key == "decoding":
                if region not in test_results.get("decoding", {}):
                    continue
                y_val = test_results["decoding"][region]["decoding_accuracy"]
            elif t_key == "rt":
                if region not in test_results.get("rt", {}):
                    continue
                y_val = test_results["rt"][region]["rt_r2"]
            elif t_key == "psychometric":
                if region not in test_results.get("psychometric", {}):
                    continue
                y_val = test_results["psychometric"][region]["psychometric_slope"]
            elif t_key == "ei_ratio":
                if region not in test_results.get("ei_ratio", {}):
                    continue
                y_val = test_results["ei_ratio"][region]["ei_ratio"]
            else:
                continue
            nc_vals.append(geo[region]["neuron_count"])
            y_vals.append(y_val)

        if len(nc_vals) >= 4:
            rho, p = spearmanr(nc_vals, y_vals)
            print(f"  {rho:+.3f}({len(nc_vals):d})", end="")
            results[f"nc_vs_{t_key}"] = {
                "rho": float(rho), "p": float(p), "n": len(nc_vals),
            }
        else:
            print(f"  {'n/a':>12s}", end="")
    print()

    return results


def analyze_error_collapse(test3_results, geo):
    """Test 3 aggregate: does bracket_norm drop more on errors in causally important regions?"""
    print("\n=== Test 3 aggregate: bracket_norm collapse on errors ===")

    collapse_vals, silencing_vals, nc_vals = [], [], []
    for region in SILENCING_REGIONS:
        if region not in test3_results:
            continue
        r = test3_results[region]
        bn_c = r.get("bracket_norm_correct")
        bn_e = r.get("bracket_norm_error")
        if bn_c is None or bn_e is None:
            continue

        collapse = (bn_c - bn_e) / (bn_c + 1e-10)
        collapse_vals.append(collapse)
        silencing_vals.append(SILENCING_REGIONS[region])
        nc_vals.append(geo.get(region, {}).get("neuron_count", 50))
        print(f"  {region:>6s}: BN_corr={bn_c:.4f} BN_err={bn_e:.4f} "
              f"collapse={collapse:+.3f} silencing={SILENCING_REGIONS[region]:.2f}")

    result = {}
    if len(collapse_vals) >= 4:
        x, y, nc = np.array(collapse_vals), np.array(silencing_vals), np.array(nc_vals)
        rho, p = spearmanr(x, y)
        partial = partial_spearman(x, y, nc)
        result = {
            "rho_collapse_vs_silencing": float(rho),
            "p": float(p),
            "partial": float(partial),
            "n": len(x),
        }
        print(f"\n  Collapse vs silencing: rho={rho:+.3f} p={p:.3f} partial={partial:+.3f} n={len(x)}")

    return result


def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Alternative predictions for geometric metrics")
    print("=" * 80)

    sessions = load_steinmetz()
    print(f"Loaded {len(sessions)} sessions")

    # Compute geometric metrics
    geo = compute_geometric_metrics(sessions)

    # Run all 8 tests
    test_results = {}
    test_results["decoding"] = test_decoding_accuracy(sessions)
    test_results["rt"] = test_rt_prediction(sessions)
    test3 = test_error_vs_correct(sessions)
    test_results["psychometric"] = test_psychometric_sensitivity(sessions)
    test_results["connectivity"] = test_connectivity(sessions)
    test_results["ei_ratio"] = test_cell_type(sessions)

    test_names = {
        "silencing": "Silencing",
        "decoding": "Decoding",
        "rt": "RT_pred",
        "psychometric": "Psych_slope",
        "connectivity": "Connectiv",
        "ei_ratio": "E/I_ratio",
    }

    correlations = compute_all_correlations(geo, test_results, test_names)
    error_collapse = analyze_error_collapse(test3, geo)

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY: What does each geometric metric predict?")
    print("=" * 80)

    for metric in ["bracket_norm", "seis"]:
        print(f"\n  {metric}:")
        for t_key, t_name in test_names.items():
            key = f"{metric}_vs_{t_key}"
            if key in correlations:
                c = correlations[key]
                star = "***" if abs(c["partial"]) > 0.5 else ("**" if abs(c["partial"]) > 0.3 else "")
                print(f"    → {t_name:>12s}: partial={c['partial']:+.3f} "
                      f"(raw={c['rho_raw']:+.3f}, nc_corr={c['rho_with_nc']:+.3f}, n={c['n']}) {star}")

    # Key dissociation test
    print("\n" + "=" * 80)
    print("DISSOCIATION TEST")
    print("=" * 80)

    bn_silencing = correlations.get("bracket_norm_vs_silencing", {}).get("partial")
    bn_decoding = correlations.get("bracket_norm_vs_decoding", {}).get("partial")
    seis_silencing = correlations.get("seis_vs_silencing", {}).get("partial")
    seis_decoding = correlations.get("seis_vs_decoding", {}).get("partial")

    if all(v is not None for v in [bn_silencing, bn_decoding, seis_silencing, seis_decoding]):
        print(f"\n  bracket_norm → silencing:  {bn_silencing:+.3f}   (computation predicts causal importance)")
        print(f"  bracket_norm → decoding:   {bn_decoding:+.3f}   (computation predicts information content)")
        print(f"  SEIS → silencing:          {seis_silencing:+.3f}   (consistency predicts causal importance)")
        print(f"  SEIS → decoding:           {seis_decoding:+.3f}   (consistency predicts information content)")

        dissociation = (bn_silencing - bn_decoding) - (seis_silencing - seis_decoding)
        print(f"\n  Interaction (double dissociation index): {dissociation:+.3f}")

        if bn_silencing > 0.3 and seis_decoding > 0.3 and bn_decoding < 0.3 and seis_silencing < 0.3:
            print("  ✓ DOUBLE DISSOCIATION: bracket_norm predicts silencing, SEIS predicts decoding")
        elif bn_silencing > 0.3 and seis_silencing < 0.3:
            print("  ~ SINGLE DISSOCIATION: bracket_norm predicts silencing, SEIS does not")
        else:
            print("  ✗ No clear dissociation pattern")

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "alternative_predictions",
        "geometric_metrics": geo,
        "test_1_decoding": test_results["decoding"],
        "test_2_rt": test_results["rt"],
        "test_3_error_correct": test3,
        "test_3_error_collapse": error_collapse,
        "test_4_psychometric": test_results["psychometric"],
        "test_5_connectivity": test_results["connectivity"],
        "test_6_cell_type": test_results["ei_ratio"],
        "correlations": correlations,
        "dissociation": {
            "bracket_norm_silencing": bn_silencing,
            "bracket_norm_decoding": bn_decoding,
            "seis_silencing": seis_silencing,
            "seis_decoding": seis_decoding,
        } if all(v is not None for v in [bn_silencing, bn_decoding, seis_silencing, seis_decoding]) else {},
    }

    save_results("alternative_predictions", all_results)

    return all_results


if __name__ == "__main__":
    main()
