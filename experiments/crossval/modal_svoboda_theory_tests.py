"""Modal: Three theory tests for the photostim bracket_norm increase.

Theory 1 (z-score artifact): Compare raw vs z-scored bracket_norm
Theory 2 (subsampling/concentration): Neuron dropout simulation on control trials
Theory 3 (regime shift): Dose-response — bracket_norm vs photostim power

Also: accuracy-matched comparison, cell type breakdown.

Usage:
    modal run --detach experiments_crossval/modal_svoboda_theory_tests.py
"""
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import modal

_this_dir = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "hdf5-tools", "libhdf5-dev", "pkg-config")
    .pip_install(
        "setuptools<71", "wheel", "Cython",
        "numpy>=1.24,<2",
        "h5py>=3.8",
        "pynwb>=2.6",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "tqdm>=4.66",
        "matplotlib>=3.8",
    )
    .add_local_file(
        str(_this_dir / "bracket_norm_core.py"),
        "/root/bracket_norm_core.py",
    )
)

app = modal.App("svoboda-theory-tests")
volume = modal.Volume.from_name("neuro-causal-geometry-results", create_if_missing=True)

DATA_DIR = "/results/svoboda_data"
RESULTS_DIR = "/results/svoboda_theory_tests"

DECISION_WINDOW_SEC = (0.0, 0.3)
MIN_NEURONS = 10
MIN_TRIALS = 20
N_DROPOUT_REPS = 100


@app.function(image=image, cpu=4, memory=32768, timeout=86400, volumes={"/results": volume})
def run_all_tests():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logger = logging.getLogger("theory")

    import numpy as np
    import pynwb
    from scipy.stats import mannwhitneyu, wilcoxon, spearmanr
    from tqdm import tqdm

    sys.path.insert(0, "/root")
    from bracket_norm_core import compute_bracket_norm

    t0 = time.time()
    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] Svoboda theory tests")

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    volume.reload()

    nwb_files = []
    for ds_dir in sorted(Path(DATA_DIR).iterdir()):
        if ds_dir.is_dir():
            nwb_files.extend(sorted(ds_dir.glob("*.nwb")))

    logger.info(f"  {len(nwb_files)} NWB files")

    # Per-session collectors
    sessions = []

    for nwb_path in tqdm(nwb_files, desc="Processing"):
        try:
            io = pynwb.NWBHDF5IO(str(nwb_path), "r")
            nwbfile = io.read()
        except Exception:
            continue

        try:
            units = nwbfile.units
            if units is None or len(units) < MIN_NEURONS:
                io.close()
                continue

            trials = nwbfile.trials
            if trials is None:
                io.close()
                continue

            trial_cols = list(trials.colnames) if trials.colnames else []
            n_units = len(units)

            if "response" not in trial_cols:
                io.close()
                continue

            if "photo_stim_id" not in trial_cols:
                io.close()
                continue

            # Choice
            raw_choices = trials["response"].data[:]
            if hasattr(raw_choices[0], "decode"):
                raw_choices = np.array([c.decode() for c in raw_choices])
            else:
                raw_choices = np.array([str(c) for c in raw_choices])

            choice_binary = np.full(len(raw_choices), -1, dtype=int)
            for i, r in enumerate(raw_choices):
                if "L" in r and "Lick" not in r:
                    choice_binary[i] = 0
                elif "R" in r:
                    choice_binary[i] = 1

            valid = choice_binary >= 0
            if valid.sum() < MIN_TRIALS:
                io.close()
                continue

            # Accuracy
            is_correct = np.array(["Hit" in str(r) for r in raw_choices])

            # Photo stim
            photo_id_raw = trials["photo_stim_id"].data[:]
            photo_id = np.array([float(p) for p in photo_id_raw])

            # Photo stim power (if available)
            has_power = "photo_stim_power" in trial_cols
            if has_power:
                photo_power_raw = trials["photo_stim_power"].data[:]
                photo_power = np.array([float(p) for p in photo_power_raw])
            else:
                photo_power = None

            start_times = np.asarray(trials["start_time"].data[:])

            # Cell type
            unit_cols = list(units.colnames) if units.colnames else []
            cell_types = None
            if "cell_type" in unit_cols:
                cell_types = []
                for i in range(n_units):
                    ct = units["cell_type"][i]
                    if hasattr(ct, "decode"):
                        ct = ct.decode()
                    cell_types.append(str(ct))

            # Build full activity matrix
            spike_times_list = [np.asarray(units["spike_times"][i]) for i in range(n_units)]
            valid_idx = np.where(valid)[0]
            n_valid = len(valid_idx)

            activity_raw = np.zeros((n_valid, n_units), dtype=np.float32)
            for ti_local, ti_global in enumerate(valid_idx):
                t_start = start_times[ti_global] + DECISION_WINDOW_SEC[0]
                t_end = start_times[ti_global] + DECISION_WINDOW_SEC[1]
                for ui in range(n_units):
                    st = spike_times_list[ui]
                    activity_raw[ti_local, ui] = np.sum((st >= t_start) & (st < t_end)) / (t_end - t_start)

            # Z-scored version
            means = activity_raw.mean(axis=0, keepdims=True)
            stds = activity_raw.std(axis=0, keepdims=True)
            stds[stds < 1e-8] = 1.0
            activity_z = (activity_raw - means) / stds

            choice_v = choice_binary[valid]
            correct_v = is_correct[valid]
            photo_id_v = photo_id[valid]
            photo_power_v = photo_power[valid] if photo_power is not None else None
            evidence = np.linspace(1, 0, n_valid)

            is_ctrl = photo_id_v == 0
            is_stim = photo_id_v != 0

            n_ctrl = is_ctrl.sum()
            n_stim = is_stim.sum()

            if n_ctrl < MIN_TRIALS or n_stim < MIN_TRIALS:
                io.close()
                continue

            if len(np.unique(choice_v[is_ctrl])) < 2 or len(np.unique(choice_v[is_stim])) < 2:
                io.close()
                continue

            sess = {"file": nwb_path.name, "n_units": n_units, "n_ctrl": int(n_ctrl), "n_stim": int(n_stim)}

            # ============================================================
            # TEST 1: Raw vs z-scored bracket_norm
            # ============================================================
            for label, act in [("raw", activity_raw), ("zscore", activity_z)]:
                ctrl_res = compute_bracket_norm(act[is_ctrl], choice_v[is_ctrl], evidence[is_ctrl])
                stim_res = compute_bracket_norm(act[is_stim], choice_v[is_stim], evidence[is_stim])
                if ctrl_res and stim_res:
                    sess[f"bn_ctrl_{label}"] = ctrl_res["bracket_norm"]
                    sess[f"bn_stim_{label}"] = stim_res["bracket_norm"]
                    sess[f"bn_diff_{label}"] = ctrl_res["bracket_norm"] - stim_res["bracket_norm"]

            # ============================================================
            # TEST 2: Neuron dropout simulation
            # ============================================================
            # Estimate dropout fraction from firing rate suppression
            ctrl_mean_rates = activity_raw[is_ctrl].mean(axis=0)
            stim_mean_rates = activity_raw[is_stim].mean(axis=0)

            # Fraction of neurons "effectively silenced" (rate drops >80%)
            suppressed = stim_mean_rates < (0.2 * ctrl_mean_rates)
            suppression_frac = suppressed.sum() / n_units
            sess["suppression_frac"] = float(suppression_frac)

            # Also try fixed dropout fractions
            dropout_results = {}
            for frac in [0.1, 0.2, 0.3, 0.5, suppression_frac]:
                frac_label = f"drop_{frac:.2f}"
                n_keep = max(MIN_NEURONS, int(n_units * (1 - frac)))
                if n_keep >= n_units:
                    continue

                bn_samples = []
                for rep in range(N_DROPOUT_REPS):
                    keep_idx = np.random.choice(n_units, n_keep, replace=False)
                    # Use CONTROL trials only, with raw activity
                    sub_act = activity_raw[is_ctrl][:, keep_idx]
                    # Z-score the subsampled population
                    m = sub_act.mean(axis=0, keepdims=True)
                    s = sub_act.std(axis=0, keepdims=True)
                    s[s < 1e-8] = 1.0
                    sub_act_z = (sub_act - m) / s
                    res = compute_bracket_norm(sub_act_z, choice_v[is_ctrl], evidence[is_ctrl])
                    if res:
                        bn_samples.append(res["bracket_norm"])

                if bn_samples:
                    dropout_results[frac_label] = {
                        "mean": float(np.mean(bn_samples)),
                        "std": float(np.std(bn_samples)),
                        "median": float(np.median(bn_samples)),
                        "n_keep": n_keep,
                        "frac": float(frac),
                    }

            sess["dropout_simulation"] = dropout_results

            # Also: dropout on raw (no z-score)
            dropout_raw = {}
            for frac in [0.1, 0.3, 0.5, suppression_frac]:
                frac_label = f"drop_{frac:.2f}"
                n_keep = max(MIN_NEURONS, int(n_units * (1 - frac)))
                if n_keep >= n_units:
                    continue
                bn_samples = []
                for rep in range(N_DROPOUT_REPS):
                    keep_idx = np.random.choice(n_units, n_keep, replace=False)
                    sub_act = activity_raw[is_ctrl][:, keep_idx]
                    res = compute_bracket_norm(sub_act, choice_v[is_ctrl], evidence[is_ctrl])
                    if res:
                        bn_samples.append(res["bracket_norm"])
                if bn_samples:
                    dropout_raw[frac_label] = {
                        "mean": float(np.mean(bn_samples)),
                        "std": float(np.std(bn_samples)),
                        "frac": float(frac),
                    }
            sess["dropout_raw"] = dropout_raw

            # ============================================================
            # TEST 3: Dose-response (bracket_norm vs power)
            # ============================================================
            if photo_power_v is not None:
                unique_powers = np.unique(photo_power_v[is_stim])
                dose_response = []
                for pw in sorted(unique_powers):
                    pw_mask_in_stim = photo_power_v[is_stim] == pw
                    n_pw = pw_mask_in_stim.sum()
                    if n_pw < MIN_TRIALS:
                        continue
                    pw_choice = choice_v[is_stim][pw_mask_in_stim]
                    if len(np.unique(pw_choice)) < 2:
                        continue
                    pw_act_z = activity_z[is_stim][pw_mask_in_stim]
                    pw_evidence = evidence[is_stim][pw_mask_in_stim]
                    res = compute_bracket_norm(pw_act_z, pw_choice, pw_evidence)
                    if res:
                        dose_response.append({
                            "power": float(pw),
                            "bracket_norm": res["bracket_norm"],
                            "n_trials": int(n_pw),
                        })
                sess["dose_response"] = dose_response

            # ============================================================
            # TEST 4: Accuracy-matched comparison
            # ============================================================
            ctrl_correct = is_ctrl & correct_v
            stim_correct = is_stim & correct_v
            if ctrl_correct.sum() >= MIN_TRIALS and stim_correct.sum() >= MIN_TRIALS:
                if (len(np.unique(choice_v[ctrl_correct])) >= 2 and
                    len(np.unique(choice_v[stim_correct])) >= 2):
                    ctrl_corr_res = compute_bracket_norm(
                        activity_z[ctrl_correct], choice_v[ctrl_correct], evidence[ctrl_correct])
                    stim_corr_res = compute_bracket_norm(
                        activity_z[stim_correct], choice_v[stim_correct], evidence[stim_correct])
                    if ctrl_corr_res and stim_corr_res:
                        sess["bn_ctrl_correct_only"] = ctrl_corr_res["bracket_norm"]
                        sess["bn_stim_correct_only"] = stim_corr_res["bracket_norm"]
                        sess["bn_diff_correct_only"] = ctrl_corr_res["bracket_norm"] - stim_corr_res["bracket_norm"]

            # ============================================================
            # TEST 5: Cell type breakdown
            # ============================================================
            if cell_types:
                type_counts = defaultdict(int)
                for ct in cell_types:
                    type_counts[ct] += 1
                sess["cell_types"] = dict(type_counts)

                for ct_name in set(cell_types):
                    ct_mask = np.array([c == ct_name for c in cell_types])
                    if ct_mask.sum() < MIN_NEURONS:
                        continue
                    ct_act = activity_z[:, ct_mask]
                    ctrl_res = compute_bracket_norm(ct_act[is_ctrl], choice_v[is_ctrl], evidence[is_ctrl])
                    stim_res = compute_bracket_norm(ct_act[is_stim], choice_v[is_stim], evidence[is_stim])
                    if ctrl_res and stim_res:
                        sess[f"bn_ctrl_celltype_{ct_name}"] = ctrl_res["bracket_norm"]
                        sess[f"bn_stim_celltype_{ct_name}"] = stim_res["bracket_norm"]

            sessions.append(sess)
            logger.info(f"  {nwb_path.name[:45]} n={n_units} "
                       f"raw:ctrl={sess.get('bn_ctrl_raw','?'):.3f}/stim={sess.get('bn_stim_raw','?'):.3f} "
                       f"z:ctrl={sess.get('bn_ctrl_zscore','?'):.3f}/stim={sess.get('bn_stim_zscore','?'):.3f} "
                       f"supp={sess.get('suppression_frac',0):.0%}")
            io.close()

        except Exception as e:
            logger.warning(f"  Error on {nwb_path.name}: {e}")
            import traceback
            traceback.print_exc()
            try:
                io.close()
            except Exception:
                pass

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: {len(sessions)} sessions\n")

    results = {"timestamp": datetime.now(timezone.utc).isoformat(), "n_sessions": len(sessions)}

    # Test 1: Raw vs z-scored
    for label in ["raw", "zscore"]:
        ctrl_vals = [s[f"bn_ctrl_{label}"] for s in sessions if f"bn_ctrl_{label}" in s]
        stim_vals = [s[f"bn_stim_{label}"] for s in sessions if f"bn_stim_{label}" in s]
        diffs = [s[f"bn_diff_{label}"] for s in sessions if f"bn_diff_{label}" in s]

        n_ctrl_higher = sum(1 for d in diffs if d > 0)
        n_stim_higher = sum(1 for d in diffs if d < 0)

        logger.info(f"  TEST 1 ({label}):")
        logger.info(f"    Control BN: mean={np.mean(ctrl_vals):.4f} ± {np.std(ctrl_vals):.4f}")
        logger.info(f"    Photostim BN: mean={np.mean(stim_vals):.4f} ± {np.std(stim_vals):.4f}")
        logger.info(f"    Direction: ctrl>stim={n_ctrl_higher}, stim>ctrl={n_stim_higher}")

        if len(diffs) >= 5:
            W, p = wilcoxon([s[f"bn_ctrl_{label}"] for s in sessions if f"bn_diff_{label}" in s],
                           [s[f"bn_stim_{label}"] for s in sessions if f"bn_diff_{label}" in s],
                           alternative="two-sided")
            logger.info(f"    Wilcoxon two-sided: W={W:.1f}, p={p:.6f}")

            results[f"test1_{label}"] = {
                "ctrl_mean": float(np.mean(ctrl_vals)),
                "stim_mean": float(np.mean(stim_vals)),
                "mean_diff": float(np.mean(diffs)),
                "n_ctrl_higher": n_ctrl_higher,
                "n_stim_higher": n_stim_higher,
                "wilcoxon_W": float(W),
                "wilcoxon_p": float(p),
                "n": len(diffs),
            }

    # Test 2: Dropout simulation
    logger.info(f"\n  TEST 2 (neuron dropout on control trials, z-scored):")
    # Get the full-population control z-scored BN for comparison
    full_ctrl_z = [s["bn_ctrl_zscore"] for s in sessions if "bn_ctrl_zscore" in s]
    logger.info(f"    Full population control BN (z): mean={np.mean(full_ctrl_z):.4f}")

    dropout_agg = defaultdict(list)
    for s in sessions:
        for frac_label, dr in s.get("dropout_simulation", {}).items():
            dropout_agg[frac_label].append(dr["mean"])

    for frac_label in sorted(dropout_agg.keys()):
        vals = dropout_agg[frac_label]
        logger.info(f"    {frac_label}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

    results["test2_dropout_zscore"] = {
        frac_label: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        for frac_label, vals in dropout_agg.items()
    }
    results["test2_full_ctrl_z"] = float(np.mean(full_ctrl_z))

    # Dropout on raw
    logger.info(f"\n  TEST 2b (neuron dropout on control trials, RAW):")
    full_ctrl_raw = [s["bn_ctrl_raw"] for s in sessions if "bn_ctrl_raw" in s]
    logger.info(f"    Full population control BN (raw): mean={np.mean(full_ctrl_raw):.4f}")

    dropout_raw_agg = defaultdict(list)
    for s in sessions:
        for frac_label, dr in s.get("dropout_raw", {}).items():
            dropout_raw_agg[frac_label].append(dr["mean"])

    for frac_label in sorted(dropout_raw_agg.keys()):
        vals = dropout_raw_agg[frac_label]
        logger.info(f"    {frac_label}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

    results["test2_dropout_raw"] = {
        frac_label: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        for frac_label, vals in dropout_raw_agg.items()
    }

    # Suppression fraction
    supp_fracs = [s["suppression_frac"] for s in sessions]
    logger.info(f"\n  Suppression fraction: mean={np.mean(supp_fracs):.2%} ± {np.std(supp_fracs):.2%}")
    results["suppression_frac"] = {"mean": float(np.mean(supp_fracs)), "std": float(np.std(supp_fracs))}

    # Test 3: Dose-response
    logger.info(f"\n  TEST 3 (dose-response):")
    all_doses = []
    for s in sessions:
        for dr in s.get("dose_response", []):
            all_doses.append(dr)

    if all_doses:
        power_to_bns = defaultdict(list)
        for dr in all_doses:
            power_to_bns[dr["power"]].append(dr["bracket_norm"])

        for pw in sorted(power_to_bns.keys()):
            bns = power_to_bns[pw]
            logger.info(f"    power={pw:.2f}: BN mean={np.mean(bns):.4f} ± {np.std(bns):.4f} (n={len(bns)})")

        all_powers = [dr["power"] for dr in all_doses]
        all_bns_dose = [dr["bracket_norm"] for dr in all_doses]
        if len(set(all_powers)) >= 3:
            rho, p = spearmanr(all_powers, all_bns_dose)
            logger.info(f"    Spearman(power, BN): rho={rho:+.3f}, p={p:.4f}")
            results["test3_dose_response"] = {
                "spearman_rho": float(rho), "spearman_p": float(p),
                "power_levels": {str(pw): {"mean": float(np.mean(bns)), "n": len(bns)}
                                for pw, bns in power_to_bns.items()},
            }
    else:
        logger.info(f"    No dose-response data (photo_stim_power not in these sessions)")
        results["test3_dose_response"] = "no_data"

    # Test 4: Accuracy-matched
    logger.info(f"\n  TEST 4 (correct trials only, z-scored):")
    ctrl_corr = [s["bn_ctrl_correct_only"] for s in sessions if "bn_ctrl_correct_only" in s]
    stim_corr = [s["bn_stim_correct_only"] for s in sessions if "bn_stim_correct_only" in s]
    diffs_corr = [s["bn_diff_correct_only"] for s in sessions if "bn_diff_correct_only" in s]
    if diffs_corr:
        n_c = sum(1 for d in diffs_corr if d > 0)
        n_s = sum(1 for d in diffs_corr if d < 0)
        logger.info(f"    Control correct BN: mean={np.mean(ctrl_corr):.4f}")
        logger.info(f"    Stim correct BN: mean={np.mean(stim_corr):.4f}")
        logger.info(f"    Direction: ctrl>stim={n_c}, stim>ctrl={n_s}")
        if len(diffs_corr) >= 5:
            W, p = wilcoxon(ctrl_corr, stim_corr, alternative="two-sided")
            logger.info(f"    Wilcoxon: W={W:.1f}, p={p:.6f}")
            results["test4_accuracy_matched"] = {
                "ctrl_mean": float(np.mean(ctrl_corr)),
                "stim_mean": float(np.mean(stim_corr)),
                "n_ctrl_higher": n_c, "n_stim_higher": n_s,
                "wilcoxon_W": float(W), "wilcoxon_p": float(p), "n": len(diffs_corr),
            }

    # Test 5: Cell types
    logger.info(f"\n  TEST 5 (cell types):")
    all_types = set()
    for s in sessions:
        if "cell_types" in s:
            all_types.update(s["cell_types"].keys())
    logger.info(f"    Cell types found: {all_types}")

    for ct in sorted(all_types):
        ctrl_ct = [s[f"bn_ctrl_celltype_{ct}"] for s in sessions if f"bn_ctrl_celltype_{ct}" in s]
        stim_ct = [s[f"bn_stim_celltype_{ct}"] for s in sessions if f"bn_stim_celltype_{ct}" in s]
        if ctrl_ct and stim_ct:
            logger.info(f"    {ct}: ctrl={np.mean(ctrl_ct):.4f} stim={np.mean(stim_ct):.4f} "
                       f"n={min(len(ctrl_ct), len(stim_ct))}")

    results["sessions"] = sessions
    results["total_time_sec"] = time.time() - t0

    with open(out_dir / "theory_tests_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    volume.commit()

    logger.info(f"\nDone in {time.time() - t0:.0f}s")


@app.local_entrypoint()
def main():
    run_all_tests.remote()
    print("Svoboda theory tests submitted. Results at /results/svoboda_theory_tests/")
