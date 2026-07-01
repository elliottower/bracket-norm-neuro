"""Exp-C9/C10/C11: Robustness checks — the three write-blockers.

C9: Partial correlation of SEIS equivariance with silencing, controlling for
    neuron count. If decision-window SEIS doesn't survive partialing out
    neuron count (rho=+0.76), Paper C needs restructuring.

C10: Effective rank sign check.
    (a) Raw effective rank vs neuron count correlation
    (b) Partial rho of effective rank with silencing controlling for neuron count
    If sign flips positive after controlling: "efficiently compressed choice
    coding -> causally important" (beautiful). If stays negative: something else.

C11: LDA accuracy + SEIS joint model.
    2-predictor model: silencing ~ LDA_accuracy + SEIS_equivariance.
    Are they collinear or orthogonal? Joint R^2.

CPU-only, ~5 min total.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from shared_bundle import SILENCING_EFFECTS, save_results

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "robustness"


def partial_spearman(x, y, z):
    rho_xy, _ = spearmanr(x, y)
    rho_xz, _ = spearmanr(x, z)
    rho_yz, _ = spearmanr(y, z)
    numerator = rho_xy - rho_xz * rho_yz
    denominator = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    if denominator < 1e-10:
        return 0.0
    return numerator / denominator


def run():
    artifacts_dir = Path(__file__).parent.parent / "results" / "artifacts"

    def load_latest(directory, prefix):
        files = sorted(f for f in directory.glob("*.json") if prefix in f.name)
        if not files:
            return None
        with open(files[-1]) as f:
            return json.load(f)

    exp125 = load_latest(artifacts_dir, "exp125")
    exp127 = load_latest(artifacts_dir, "exp127")
    exp128 = load_latest(artifacts_dir, "exp128")
    exp129 = load_latest(artifacts_dir, "exp129")

    from data.steinmetz import load_all, list_regions, get_region_activity, get_choice_labels
    sessions = load_all()
    region_neuron_counts = {}
    for si, sess in enumerate(sessions):
        regions = list_regions(sess, min_neurons=15)
        for region in regions:
            activity = get_region_activity(sess, region, slice(15, 35))
            if activity is None:
                continue
            if activity.ndim == 3:
                activity = activity.mean(axis=2)
            if activity.shape[1] >= 15:
                if region not in region_neuron_counts:
                    region_neuron_counts[region] = []
                region_neuron_counts[region].append(activity.shape[1])
    for region in region_neuron_counts:
        region_neuron_counts[region] = float(np.mean(region_neuron_counts[region]))

    print("=" * 80)
    print("C9: PARTIAL CORRELATION — SEIS equivariance controlling for neuron count")
    print("=" * 80)

    metrics_to_partial = {
        "SEIS equivariance (full)": (exp125, "equivariance_mean"),
        "Decision-window SEIS": (exp128, "decision_eq"),
        "Peak SEIS": (exp128, "peak_value"),
        "Curvature index": (None, None),
        "Cloud variance": (exp129, "cloud_variance_mean"),
    }

    for name, (data, key) in metrics_to_partial.items():
        if data is None:
            continue
        region_summary = data.get("region_summary", {})
        metric_vals, silencing_vals, neuron_vals = [], [], []
        for region, effect in SILENCING_EFFECTS.items():
            if region in region_summary and region in region_neuron_counts:
                if key in region_summary[region]:
                    v = region_summary[region][key]
                elif f"{key}" in region_summary[region]:
                    v = region_summary[region][key]
                else:
                    continue
                if v is not None and np.isfinite(v):
                    metric_vals.append(v)
                    silencing_vals.append(effect)
                    neuron_vals.append(region_neuron_counts[region])

        if len(metric_vals) >= 5:
            rho_raw, p_raw = spearmanr(metric_vals, silencing_vals)
            rho_partial = partial_spearman(
                np.array(metric_vals),
                np.array(silencing_vals),
                np.array(neuron_vals)
            )
            rho_with_neurons, _ = spearmanr(metric_vals, neuron_vals)
            print(f"\n  {name}:")
            print(f"    Raw rho with silencing:     {rho_raw:+.3f} (p={p_raw:.4f})")
            print(f"    Correlation with n_neurons:  {rho_with_neurons:+.3f}")
            print(f"    Partial rho (ctrl neurons):  {rho_partial:+.3f}")
            if abs(rho_partial) > 0.3:
                print(f"    SURVIVES partialing out neuron count")
            else:
                print(f"    WARNING: does NOT survive partialing")

    neuron_silencing = []
    for region, effect in SILENCING_EFFECTS.items():
        if region in region_neuron_counts:
            neuron_silencing.append((region_neuron_counts[region], effect, region))
    if neuron_silencing:
        ns, ss, rs = zip(*neuron_silencing)
        rho_ns, p_ns = spearmanr(ns, ss)
        print(f"\n  Neuron count vs silencing: rho={rho_ns:+.3f}, p={p_ns:.4f}")

    print(f"\n{'='*80}")
    print("C10: EFFECTIVE RANK SIGN CHECK")
    print("=" * 80)

    if exp129:
        rs = exp129.get("region_summary", {})
        er_vals, nc_vals, sil_vals = [], [], []
        er_norm_vals = []
        for region, effect in SILENCING_EFFECTS.items():
            if region in rs and region in region_neuron_counts:
                er = rs[region].get("effective_rank_mean")
                er_n = rs[region].get("effective_rank_normalized_mean")
                if er is not None and np.isfinite(er):
                    er_vals.append(er)
                    nc_vals.append(region_neuron_counts[region])
                    sil_vals.append(effect)
                    er_norm_vals.append(er_n if er_n is not None else 0)

        if len(er_vals) >= 5:
            rho_er_nc, _ = spearmanr(er_vals, nc_vals)
            rho_er_sil, p_er_sil = spearmanr(er_vals, sil_vals)
            rho_ern_sil, p_ern_sil = spearmanr(er_norm_vals, sil_vals)
            rho_ern_nc, _ = spearmanr(er_norm_vals, nc_vals)
            partial_er = partial_spearman(
                np.array(er_vals), np.array(sil_vals), np.array(nc_vals)
            )
            partial_ern = partial_spearman(
                np.array(er_norm_vals), np.array(sil_vals), np.array(nc_vals)
            )

            print(f"\n  (a) Effective rank vs neuron count:   rho={rho_er_nc:+.3f}")
            print(f"  (b) Effective rank vs silencing:      rho={rho_er_sil:+.3f} (p={p_er_sil:.4f})")
            print(f"  (c) Partial ER vs silencing (ctrl nc): rho={partial_er:+.3f}")
            print(f"\n  (d) ER_normalized vs neuron count:    rho={rho_ern_nc:+.3f}")
            print(f"  (e) ER_normalized vs silencing:       rho={rho_ern_sil:+.3f} (p={p_ern_sil:.4f})")
            print(f"  (f) Partial ER_norm vs sil (ctrl nc): rho={partial_ern:+.3f}")

            if partial_er > 0:
                print(f"\n  SIGN FLIP: raw ER positive after controlling for neuron count")
                print(f"  Interpretation: efficiently compressed choice coding -> causally important")
            else:
                print(f"\n  No sign flip: ER stays {'+' if partial_er > 0 else '-'} after controlling")

    print(f"\n{'='*80}")
    print("C11: JOINT MODEL — LDA accuracy + SEIS equivariance")
    print("=" * 80)

    if exp127 and exp128:
        rs127 = exp127.get("region_summary", {})
        rs128 = exp128.get("region_summary", {})
        lda_vals, seis_vals, sil_vals = [], [], []
        regions_used = []
        for region, effect in SILENCING_EFFECTS.items():
            lda_acc = rs127.get(region, {}).get("lda_accuracy_mean")
            seis_eq = rs128.get(region, {}).get("decision_eq")
            if lda_acc is not None and seis_eq is not None:
                lda_vals.append(lda_acc)
                seis_vals.append(seis_eq)
                sil_vals.append(effect)
                regions_used.append(region)

        if len(lda_vals) >= 5:
            rho_lda, p_lda = spearmanr(lda_vals, sil_vals)
            rho_seis, p_seis = spearmanr(seis_vals, sil_vals)
            rho_lda_seis, _ = spearmanr(lda_vals, seis_vals)

            print(f"\n  LDA accuracy vs silencing:    rho={rho_lda:+.3f} (p={p_lda:.4f})")
            print(f"  Decision SEIS vs silencing:   rho={rho_seis:+.3f} (p={p_seis:.4f})")
            print(f"  LDA accuracy vs SEIS:         rho={rho_lda_seis:+.3f}")

            if abs(rho_lda_seis) > 0.7:
                print(f"  COLLINEAR: LDA accuracy and SEIS are measuring the same thing")
            elif abs(rho_lda_seis) < 0.3:
                print(f"  ORTHOGONAL: LDA accuracy and SEIS measure different properties")
            else:
                print(f"  MODERATE correlation: partially overlapping information")

            from numpy.linalg import lstsq
            X = np.column_stack([
                (np.array(lda_vals) - np.mean(lda_vals)) / (np.std(lda_vals) + 1e-10),
                (np.array(seis_vals) - np.mean(seis_vals)) / (np.std(seis_vals) + 1e-10),
            ])
            y = (np.array(sil_vals) - np.mean(sil_vals)) / (np.std(sil_vals) + 1e-10)
            beta, residuals, _, _ = lstsq(X, y, rcond=None)
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            r2_lda = rho_lda ** 2
            r2_seis = rho_seis ** 2

            print(f"\n  Individual R^2:")
            print(f"    LDA accuracy:   R^2 = {r2_lda:.3f}")
            print(f"    Decision SEIS:  R^2 = {r2_seis:.3f}")
            print(f"  Joint R^2:        R^2 = {r_squared:.3f}")
            print(f"  R^2 gain from joint:   +{r_squared - max(r2_lda, r2_seis):.3f}")
            print(f"\n  Standardized betas: LDA={beta[0]:+.3f}, SEIS={beta[1]:+.3f}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "C9/C10/C11 robustness checks — see stdout for details",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results("robustness_c9_c10_c11", results, RESULTS_DIR)


if __name__ == "__main__":
    run()
