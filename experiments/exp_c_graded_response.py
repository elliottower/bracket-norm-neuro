"""Graded photostimulation response (E5) — Li et al. 2019.

Tests H30: does neural suppression scale monotonically with laser power?
Uses per-neuron per-trial spike rates at 4 power levels in ALM.

We cannot compute bracket_norm (no choice/evidence labels), but we CAN test
the graded response prediction of the gain-reduction mechanism:
  - More power → more suppression of pyramidal neurons
  - More power → less suppression (or facilitation) of FS interneurons
  - Dose-response is monotonic and graded, not threshold-like

Pre-registered in mechval_audit_statement_v1.md (H30).

CPU-only, ~1 min.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, linregress

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "li_et_al_2019"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ALM_EXP_IDS = [10, 11]
S1_EXP_IDS = [1, 7, 8]


def save_results(name, data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def analyze_dose_response(unit_info, response_data, exp_ids, region_name):
    """Compute suppression ratio vs power for a set of experiment IDs."""
    mask = np.isin(unit_info[:, 7].astype(int), exp_ids)
    n_units = mask.sum()
    if n_units == 0:
        return None

    print(f"\n  {region_name}: {n_units} units across exp_ids {exp_ids}")

    all_powers = []
    all_ratios_pyr = {}
    all_ratios_fs = {}
    all_ratios_all = {}

    for ui_idx in np.where(mask)[0]:
        cell_type = int(unit_info[ui_idx, 0])  # 1=pyr, 2=FS, 0=unclassified
        rd = response_data[ui_idx]
        baseline_rates = rd[:, 0]
        stim_rates = rd[:, 1]
        powers = rd[:, 2]

        valid = ~np.isnan(powers) & (powers > 0) & (baseline_rates > 0.5)
        if valid.sum() < 5:
            continue

        unique_powers = np.unique(powers[valid])
        for p in unique_powers:
            p_mask = valid & (np.abs(powers - p) < 0.01)
            if p_mask.sum() < 2:
                continue
            mean_baseline = np.mean(baseline_rates[p_mask])
            mean_stim = np.mean(stim_rates[p_mask])
            ratio = mean_stim / mean_baseline

            p_key = round(float(p), 2)
            if p_key not in all_ratios_all:
                all_ratios_all[p_key] = []
                all_ratios_pyr[p_key] = []
                all_ratios_fs[p_key] = []

            all_ratios_all[p_key].append(ratio)
            if cell_type == 1:
                all_ratios_pyr[p_key].append(ratio)
            elif cell_type == 2:
                all_ratios_fs[p_key].append(ratio)

    if len(all_ratios_all) < 3:
        print(f"    Only {len(all_ratios_all)} power levels, need >= 3")
        return None

    sorted_powers = sorted(all_ratios_all.keys())
    print(f"    Power levels: {sorted_powers}")

    dose_response = {}
    for p in sorted_powers:
        n_all = len(all_ratios_all[p])
        n_pyr = len(all_ratios_pyr.get(p, []))
        n_fs = len(all_ratios_fs.get(p, []))
        mean_all = float(np.mean(all_ratios_all[p]))
        mean_pyr = float(np.mean(all_ratios_pyr[p])) if n_pyr > 0 else None
        mean_fs = float(np.mean(all_ratios_fs[p])) if n_fs > 0 else None
        dose_response[p] = {
            "n_all": n_all, "mean_ratio_all": mean_all,
            "n_pyr": n_pyr, "mean_ratio_pyr": mean_pyr,
            "n_fs": n_fs, "mean_ratio_fs": mean_fs,
        }
        pyr_str = f"pyr={mean_pyr:.3f}({n_pyr})" if mean_pyr else "pyr=N/A"
        fs_str = f"FS={mean_fs:.3f}({n_fs})" if mean_fs else "FS=N/A"
        print(f"    {p:6.2f} mW: all={mean_all:.3f}({n_all})  {pyr_str}  {fs_str}")

    powers_arr = np.array(sorted_powers)
    ratios_arr = np.array([dose_response[p]["mean_ratio_all"] for p in sorted_powers])

    rho, p_val = spearmanr(powers_arr, ratios_arr)
    slope, intercept, r_value, p_linreg, std_err = linregress(
        np.log10(powers_arr), ratios_arr
    )

    is_monotonic_decreasing = all(
        ratios_arr[i] >= ratios_arr[i + 1] for i in range(len(ratios_arr) - 1)
    )

    pyr_powers = sorted([p for p in sorted_powers if dose_response[p]["mean_ratio_pyr"] is not None])
    pyr_rho = None
    if len(pyr_powers) >= 3:
        pyr_arr = np.array([dose_response[p]["mean_ratio_pyr"] for p in pyr_powers])
        pyr_rho, _ = spearmanr(np.array(pyr_powers), pyr_arr)

    fs_powers = sorted([p for p in sorted_powers if dose_response[p]["mean_ratio_fs"] is not None])
    fs_rho = None
    if len(fs_powers) >= 3:
        fs_arr = np.array([dose_response[p]["mean_ratio_fs"] for p in fs_powers])
        fs_rho, _ = spearmanr(np.array(fs_powers), fs_arr)

    print(f"\n    Dose-response correlation (power vs suppression ratio):")
    print(f"      All cells:  rho={rho:+.3f}  log-linear slope={slope:.4f}  p={p_linreg:.4f}")
    print(f"      Monotonic decreasing: {is_monotonic_decreasing}")
    if pyr_rho is not None:
        print(f"      Pyramidal:  rho={pyr_rho:+.3f}")
    if fs_rho is not None:
        print(f"      FS:         rho={fs_rho:+.3f}")

    return {
        "region": region_name,
        "n_units": int(n_units),
        "n_power_levels": len(sorted_powers),
        "power_levels": sorted_powers,
        "dose_response": dose_response,
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "log_linear_slope": float(slope),
        "log_linear_r2": float(r_value ** 2),
        "log_linear_p": float(p_linreg),
        "monotonic_decreasing": is_monotonic_decreasing,
        "pyr_rho": float(pyr_rho) if pyr_rho is not None else None,
        "fs_rho": float(fs_rho) if fs_rho is not None else None,
    }


def run():
    import scipy.io as sio

    mat_path = DATA_DIR / "Fig1_data_photoinhibition_vs_power.mat"
    if not mat_path.exists():
        print(f"ERROR: {mat_path} not found. Download from GitHub first.")
        return

    print(f"[{datetime.now(timezone.utc).isoformat()}] E5 graded response — Li et al. 2019")
    d = sio.loadmat(str(mat_path), squeeze_me=True)
    unit_info = d["unit_info_all"]
    response_data = d["response_data_all"]
    print(f"  {len(response_data)} total units")

    alm_result = analyze_dose_response(unit_info, response_data, ALM_EXP_IDS, "ALM")
    s1_result = analyze_dose_response(unit_info, response_data, S1_EXP_IDS, "S1")

    print(f"\n{'='*70}")
    print("H30 ASSESSMENT: Graded Response (E5)")
    print(f"{'='*70}")

    h30 = {"pass": False}
    if alm_result:
        rho = alm_result["spearman_rho"]
        mono = alm_result["monotonic_decreasing"]
        p = alm_result["log_linear_p"]
        h30["alm_rho"] = rho
        h30["alm_monotonic"] = mono
        h30["alm_p"] = p
        h30["pass"] = (rho < -0.5) or (p < 0.05 and alm_result["log_linear_slope"] < 0)

        print(f"  ALM: rho={rho:+.3f}, monotonic={mono}, log-lin p={p:.4f}")
        print(f"  Pass threshold: rho < -0.5 OR significant negative slope (p<0.05)")
        print(f"  H30 PASS: {h30['pass']}")

        if alm_result["pyr_rho"] is not None and alm_result["fs_rho"] is not None:
            pyr_more = alm_result["pyr_rho"] < alm_result["fs_rho"]
            print(f"\n  Cell-type dissociation:")
            print(f"    Pyramidal rho={alm_result['pyr_rho']:+.3f} (should be more negative)")
            print(f"    FS rho={alm_result['fs_rho']:+.3f}")
            print(f"    Pyr more suppressed than FS: {pyr_more}")
            h30["cell_type_dissociation"] = pyr_more

    if s1_result:
        print(f"\n  S1 (comparison): rho={s1_result['spearman_rho']:+.3f}, "
              f"monotonic={s1_result['monotonic_decreasing']}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "E5_graded_response_li_et_al_2019",
        "preregistration": "mechval_audit_statement_v1.md H30",
        "h30_graded_response": h30,
        "alm": alm_result,
        "s1": s1_result,
        "note": "Tests dose-response of photoinhibition, not bracket_norm "
                "(no choice/evidence labels in dataset). E5 criterion asks "
                "whether the perturbation effect is graded, which this tests.",
    }

    save_results("graded_response_e5_h30", results)
    print("\nDone.")


if __name__ == "__main__":
    run()
