"""Jackknife robustness check on decision-window bracket_norm.

Leave-one-out on all 9 silencing regions to check if any single
region drives the rho=+0.900 / partial=+0.753 result.
Also bootstrap 95% CI on the partial correlation.

CPU-only, <5 min.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from shared_bundle import save_results

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "bracket_jackknife"

BRACKET_DATA = {
    "ACA":   {"bracket_norm": 0.0769, "neuron_count": 76,  "silencing": 0.15},
    "MOs":   {"bracket_norm": 0.1658, "neuron_count": 102, "silencing": 0.15},
    "ORB":   {"bracket_norm": 0.2306, "neuron_count": 162, "silencing": 0.31},
    "PL":    {"bracket_norm": 0.2544, "neuron_count": 104, "silencing": 0.33},
    "RSP":   {"bracket_norm": 0.1599, "neuron_count": 85,  "silencing": 0.14},
    "VISam": {"bracket_norm": 0.1070, "neuron_count": 91,  "silencing": 0.08},
    "VISl":  {"bracket_norm": 0.1735, "neuron_count": 145, "silencing": 0.17},
    "VISp":  {"bracket_norm": 0.1096, "neuron_count": 93,  "silencing": 0.14},
    "VISpm": {"bracket_norm": 0.2274, "neuron_count": 147, "silencing": 0.22},
}


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
    print(f"[{datetime.now().isoformat()}] Bracket_norm jackknife robustness check")

    regions = sorted(BRACKET_DATA.keys())
    bn = np.array([BRACKET_DATA[r]["bracket_norm"] for r in regions])
    nc = np.array([BRACKET_DATA[r]["neuron_count"] for r in regions])
    sil = np.array([BRACKET_DATA[r]["silencing"] for r in regions])

    full_rho, full_p = spearmanr(bn, sil)
    full_partial = partial_spearman(bn, sil, nc)
    print(f"\n  Full sample (n=9):")
    print(f"    Raw rho = {full_rho:+.3f}, p = {full_p:.4f}")
    print(f"    Partial rho = {full_partial:+.3f}")

    print(f"\n{'='*70}")
    print(f"  Leave-one-out jackknife:")
    print(f"  {'Removed':>8s}  {'n':>2s}  {'Raw rho':>8s}  {'p':>7s}  {'Partial':>8s}  {'Change':>8s}")
    print(f"  {'-'*55}")

    jackknife_results = {}
    for i, drop_region in enumerate(regions):
        mask = np.ones(len(regions), dtype=bool)
        mask[i] = False
        bn_j = bn[mask]
        nc_j = nc[mask]
        sil_j = sil[mask]
        rho_j, p_j = spearmanr(bn_j, sil_j)
        partial_j = partial_spearman(bn_j, sil_j, nc_j)
        delta = partial_j - full_partial
        flag = " <<<" if abs(delta) > 0.2 else ""
        print(f"  {drop_region:>8s}  {len(bn_j):2d}  {rho_j:+8.3f}  {p_j:7.4f}  {partial_j:+8.3f}  {delta:+8.3f}{flag}")
        jackknife_results[drop_region] = {
            "rho": float(rho_j), "p": float(p_j),
            "partial": float(partial_j), "delta": float(delta),
        }

    partials = [v["partial"] for v in jackknife_results.values()]
    print(f"\n  Jackknife partial range: [{min(partials):+.3f}, {max(partials):+.3f}]")
    print(f"  Jackknife partial mean:  {np.mean(partials):+.3f}")
    print(f"  Jackknife partial std:   {np.std(partials):.3f}")

    worst = min(jackknife_results.items(), key=lambda x: x[1]["partial"])
    best = max(jackknife_results.items(), key=lambda x: x[1]["partial"])
    print(f"  Worst case: drop {worst[0]} → partial={worst[1]['partial']:+.3f}")
    print(f"  Best case:  drop {best[0]} → partial={best[1]['partial']:+.3f}")

    print(f"\n{'='*70}")
    print(f"  Bootstrap 95% CI on partial correlation (10000 resamples):")

    rng = np.random.default_rng(None)
    n_boot = 10000
    boot_partials = []
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(bn), size=len(bn))
        bn_b = bn[idx]
        nc_b = nc[idx]
        sil_b = sil[idx]
        if len(set(idx)) < 4:
            continue
        try:
            rho_b, _ = spearmanr(bn_b, sil_b)
            partial_b = partial_spearman(bn_b, sil_b, nc_b)
            if np.isfinite(partial_b) and np.isfinite(rho_b):
                boot_partials.append(partial_b)
                boot_rhos.append(rho_b)
        except Exception:
            continue

    boot_partials = np.array(boot_partials)
    boot_rhos = np.array(boot_rhos)
    ci_partial = np.percentile(boot_partials, [2.5, 97.5])
    ci_rho = np.percentile(boot_rhos, [2.5, 97.5])
    frac_positive = np.mean(boot_partials > 0)

    print(f"    Raw rho 95% CI:     [{ci_rho[0]:+.3f}, {ci_rho[1]:+.3f}]")
    print(f"    Partial rho 95% CI: [{ci_partial[0]:+.3f}, {ci_partial[1]:+.3f}]")
    print(f"    Fraction partial > 0: {frac_positive:.3f}")
    print(f"    Bootstrap mean partial: {np.mean(boot_partials):+.3f}")
    print(f"    Bootstrap median partial: {np.median(boot_partials):+.3f}")

    if ci_partial[0] > 0:
        print(f"\n  VERDICT: Partial correlation is robustly positive (lower CI > 0)")
    elif frac_positive > 0.95:
        print(f"\n  VERDICT: Partial correlation is likely positive (>95% bootstrap)")
    else:
        print(f"\n  VERDICT: Partial correlation is NOT robust ({frac_positive:.1%} positive)")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "full_sample": {"rho": float(full_rho), "p": float(full_p), "partial": float(full_partial), "n": 9},
        "jackknife": jackknife_results,
        "jackknife_summary": {
            "partial_range": [float(min(partials)), float(max(partials))],
            "partial_mean": float(np.mean(partials)),
            "partial_std": float(np.std(partials)),
            "worst_drop": worst[0],
            "worst_partial": float(worst[1]["partial"]),
        },
        "bootstrap": {
            "n_resamples": len(boot_partials),
            "partial_ci_95": [float(ci_partial[0]), float(ci_partial[1])],
            "rho_ci_95": [float(ci_rho[0]), float(ci_rho[1])],
            "frac_positive": float(frac_positive),
            "mean_partial": float(np.mean(boot_partials)),
            "median_partial": float(np.median(boot_partials)),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results("bracket_jackknife", results, RESULTS_DIR)
    return results


if __name__ == "__main__":
    run()
