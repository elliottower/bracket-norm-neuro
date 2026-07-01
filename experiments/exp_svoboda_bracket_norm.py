"""Standalone Svoboda ALM bracket_norm analysis — control vs photostimulation.

Reproduces Table 5 from the paper: bracket_norm is higher under photostimulation
than control conditions (paired Wilcoxon signed-rank test across sessions).

Downloads NWB files from DANDI archives 000007 and 000009, processes each file
to compute bracket_norm for control vs photostim trials (split by photo_stim_id),
and saves results to results/.

Requires: pip install pynwb dandi h5py numpy scipy tqdm

Usage:
    python experiments/exp_svoboda_bracket_norm.py
"""
import json
import logging
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
from dandi.download import download as dandi_download
from scipy.stats import mannwhitneyu, wilcoxon
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from crossval.bracket_norm_core import compute_bracket_norm

DANDI_IDS = ["000007", "000009"]
DATA_DIR = PROJECT_ROOT / "data" / "cache" / "svoboda"
RESULTS_DIR = PROJECT_ROOT / "results"

DECISION_WINDOW_SEC = (0.0, 0.3)
MIN_NEURONS = 10
MIN_TRIALS = 20


def save_results(name, data, results_dir=None):
    rd = Path(results_dir) if results_dir else RESULTS_DIR
    rd.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = rd / f"{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def download_dandi_data():
    """Download NWB files from DANDI archives 000007 and 000009."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dandiset_id in DANDI_IDS:
        ds_dir = DATA_DIR / dandiset_id
        if ds_dir.exists() and list(ds_dir.rglob("*.nwb")):
            n_existing = len(list(ds_dir.rglob("*.nwb")))
            print(f"  DANDI {dandiset_id}: {n_existing} NWB files already cached, skipping download")
            continue

        ds_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://dandiarchive.org/dandiset/{dandiset_id}/draft"
        print(f"  Downloading DANDI {dandiset_id} to {ds_dir} ...")
        dandi_download(url, output_dir=str(ds_dir), get_metadata=False, get_assets=True)
        n_downloaded = len(list(ds_dir.rglob("*.nwb")))
        print(f"  DANDI {dandiset_id}: downloaded {n_downloaded} NWB files")


def collect_nwb_files():
    """Collect all NWB files from the data cache."""
    nwb_files = []
    for ds_dir in sorted(DATA_DIR.iterdir()):
        if ds_dir.is_dir():
            nwb_files.extend(sorted(ds_dir.rglob("*.nwb")))
    return nwb_files


def process_nwb_file(nwb_path, logger):
    """Process a single NWB file and return session info dict, or None on failure."""
    try:
        io = pynwb.NWBHDF5IO(str(nwb_path), "r")
        nwbfile = io.read()
    except Exception as e:
        logger.debug(f"  Could not open {nwb_path.name}: {e}")
        return None, "open_failed"

    try:
        units = nwbfile.units
        if units is None or len(units) < MIN_NEURONS:
            io.close()
            return None, "too_few_neurons"

        trials = nwbfile.trials
        if trials is None:
            io.close()
            return None, "no_trials"

        trial_cols = list(trials.colnames) if trials.colnames else []
        n_units = len(units)

        # Region from electrodes
        region = "unknown"
        if nwbfile.electrodes is not None:
            elec_cols = list(nwbfile.electrodes.colnames) if nwbfile.electrodes.colnames else []
            if "location" in elec_cols:
                locs = set()
                for i in range(min(len(nwbfile.electrodes), 20)):
                    loc_str = str(nwbfile.electrodes["location"][i])
                    if "brain_region:" in loc_str:
                        br = loc_str.split("brain_region:")[1].split(";")[0].strip()
                        locs.add(br)
                    else:
                        locs.add(loc_str)
                if locs:
                    region = ", ".join(sorted(locs))

        # Choice from response column
        if "response" not in trial_cols:
            io.close()
            return None, "no_response"

        raw_choices = trials["response"].data[:]
        if hasattr(raw_choices[0], "decode"):
            raw_choices = np.array([c.decode() for c in raw_choices])
        else:
            raw_choices = np.array([str(c) for c in raw_choices])

        # Binary: lick left (HitL, ErrL) = 0, lick right (HitR, ErrR) = 1
        choice_binary = np.full(len(raw_choices), -1, dtype=int)
        for i, r in enumerate(raw_choices):
            if "L" in r and "Lick" not in r:
                choice_binary[i] = 0
            elif "R" in r:
                choice_binary[i] = 1

        valid_trials = choice_binary >= 0
        if valid_trials.sum() < MIN_TRIALS:
            io.close()
            return None, "too_few_valid_trials"

        start_times = np.asarray(trials["start_time"].data[:])

        # Spike times per unit
        spike_times_list = [np.asarray(units["spike_times"][i]) for i in range(n_units)]

        # Build activity matrix: spike count in decision window / window duration
        valid_idx = np.where(valid_trials)[0]
        n_valid = len(valid_idx)
        activity = np.zeros((n_valid, n_units), dtype=np.float32)
        for ti_local, ti_global in enumerate(valid_idx):
            t_start = start_times[ti_global] + DECISION_WINDOW_SEC[0]
            t_end = start_times[ti_global] + DECISION_WINDOW_SEC[1]
            for ui in range(n_units):
                st = spike_times_list[ui]
                activity[ti_local, ui] = np.sum((st >= t_start) & (st < t_end)) / (t_end - t_start)

        # Z-score normalize per neuron
        means = activity.mean(axis=0, keepdims=True)
        stds = activity.std(axis=0, keepdims=True)
        stds[stds < 1e-8] = 1.0
        activity_z = (activity - means) / stds

        choice_valid = choice_binary[valid_trials]

        # Evidence proxy
        if "stim_present" in trial_cols:
            stim = np.asarray(trials["stim_present"].data[:])[valid_trials]
            evidence = stim.astype(float)
        else:
            evidence = np.linspace(1, 0, n_valid)

        # All-trials bracket_norm
        all_result = compute_bracket_norm(activity_z, choice_valid, evidence)

        sess_info = {
            "file": nwb_path.name,
            "region": region,
            "n_units": n_units,
            "n_valid_trials": n_valid,
        }

        if all_result:
            sess_info["bracket_norm_all"] = all_result["bracket_norm"]

        # Control vs photostim split
        has_photo_id = "photo_stim_id" in trial_cols
        has_photo_type = "photo_stim_type" in trial_cols

        if has_photo_id:
            photo_id_raw = trials["photo_stim_id"].data[:]
            photo_id = np.array([float(p) for p in photo_id_raw])
            photo_id_valid = photo_id[valid_trials]
            is_control = photo_id_valid == 0
            is_stim = photo_id_valid != 0
        elif has_photo_type:
            photo_type = trials["photo_stim_type"].data[:]
            if hasattr(photo_type[0], "decode"):
                photo_type = np.array([p.decode() for p in photo_type])
            else:
                photo_type = np.array([str(p) for p in photo_type])
            photo_valid = photo_type[valid_trials]
            no_stim_labels = {"N/A", "n/a", "na", "nan", "none", "no_stim", "nostim", "0", "0.0"}
            is_control = np.array([p.strip().lower() in no_stim_labels for p in photo_valid])
            is_stim = ~is_control
        else:
            is_control = None
            is_stim = None

        ctrl_bn = None
        stim_bn = None

        if is_control is not None:
            n_ctrl = int(is_control.sum())
            n_stim = int(is_stim.sum())
            sess_info["n_control"] = n_ctrl
            sess_info["n_stim"] = n_stim

            if n_ctrl >= MIN_TRIALS and len(np.unique(choice_valid[is_control])) >= 2:
                ctrl_result = compute_bracket_norm(
                    activity_z[is_control], choice_valid[is_control], evidence[is_control]
                )
                if ctrl_result:
                    ctrl_bn = ctrl_result["bracket_norm"]
                    sess_info["bracket_norm_control"] = ctrl_bn

            if n_stim >= MIN_TRIALS and len(np.unique(choice_valid[is_stim])) >= 2:
                stim_result = compute_bracket_norm(
                    activity_z[is_stim], choice_valid[is_stim], evidence[is_stim]
                )
                if stim_result:
                    stim_bn = stim_result["bracket_norm"]
                    sess_info["bracket_norm_photostim"] = stim_bn

        io.close()
        return sess_info, (ctrl_bn, stim_bn)

    except Exception as e:
        logger.warning(f"  Error on {nwb_path.name}: {e}")
        traceback.print_exc()
        try:
            io.close()
        except Exception:
            pass
        return None, "error"


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logger = logging.getLogger("svoboda-bn")

    t0 = time.time()
    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] Svoboda ALM bracket_norm (standalone)")

    # --- Step 1: Download data ---
    logger.info("Step 1: Downloading DANDI data ...")
    download_dandi_data()

    # --- Step 2: Collect NWB files ---
    nwb_files = collect_nwb_files()
    logger.info(f"  {len(nwb_files)} NWB files found")

    if not nwb_files:
        logger.error("No NWB files found. Check data/cache/svoboda/ directory.")
        return

    # --- Step 3: Process each file ---
    all_bns = []
    control_bns = []
    photostim_bns = []
    session_results = []
    paired_sessions = []
    n_skipped = 0
    regions_seen = defaultdict(int)

    for nwb_path in tqdm(nwb_files, desc="Processing NWB files"):
        sess_info, result = process_nwb_file(nwb_path, logger)

        if sess_info is None:
            n_skipped += 1
            continue

        regions_seen[sess_info["region"]] += 1

        if "bracket_norm_all" in sess_info:
            all_bns.append(sess_info["bracket_norm_all"])

        ctrl_bn, stim_bn = result
        if ctrl_bn is not None:
            control_bns.append(ctrl_bn)
        if stim_bn is not None:
            photostim_bns.append(stim_bn)

        if ctrl_bn is not None and stim_bn is not None:
            paired_sessions.append({
                "file": nwb_path.name,
                "control": ctrl_bn,
                "photostim": stim_bn,
                "diff": ctrl_bn - stim_bn,
            })

        session_results.append(sess_info)

        bn_str = f"BN={sess_info.get('bracket_norm_all', 'N/A')}"
        ctrl_str = f"ctrl={sess_info.get('bracket_norm_control', 'N/A')}"
        stim_str = f"stim={sess_info.get('bracket_norm_photostim', 'N/A')}"
        logger.info(f"  {nwb_path.name[:50]}: {sess_info['region']} n={sess_info['n_units']} "
                    f"{bn_str} {ctrl_str} {stim_str}")

    # --- Step 4: Summary statistics ---
    logger.info(f"\n{'='*70}")
    logger.info(f"Processed: {len(session_results)}, Skipped: {n_skipped}")
    logger.info(f"Regions: {dict(regions_seen)}")

    summary = {}
    for label, bns in [("all", all_bns), ("control", control_bns), ("photostim", photostim_bns)]:
        if bns:
            summary[label] = {
                "n": len(bns),
                "mean": float(np.mean(bns)),
                "std": float(np.std(bns)),
                "median": float(np.median(bns)),
            }
        else:
            summary[label] = {"n": 0}

    logger.info(f"\n  All-trials:  {summary['all']}")
    logger.info(f"  Control:     {summary['control']}")
    logger.info(f"  Photostim:   {summary['photostim']}")

    # --- Step 5: Statistical tests ---
    tests = {}

    # Unpaired: Mann-Whitney U
    if len(control_bns) >= 5 and len(photostim_bns) >= 5:
        U, p = mannwhitneyu(control_bns, photostim_bns, alternative="greater")
        tests["mann_whitney"] = {
            "test": "Mann-Whitney U (control > stim)",
            "U": float(U), "p": float(p),
            "n_control": len(control_bns), "n_stim": len(photostim_bns),
        }
        logger.info(f"  Mann-Whitney: U={U:.1f}, p={p:.4f}")

    # Paired: Wilcoxon signed-rank
    if len(paired_sessions) >= 5:
        ctrl_vals = [s["control"] for s in paired_sessions]
        stim_vals = [s["photostim"] for s in paired_sessions]
        diffs = [s["diff"] for s in paired_sessions]

        W, p = wilcoxon(ctrl_vals, stim_vals, alternative="greater")

        # Cohen's d for paired samples
        diffs_arr = np.array(diffs)
        cohens_d = float(np.mean(diffs_arr) / (np.std(diffs_arr, ddof=1) + 1e-10))

        # Percentage increase (photostim relative to control)
        mean_ctrl = float(np.mean(ctrl_vals))
        mean_stim = float(np.mean(stim_vals))
        pct_increase = float((mean_stim - mean_ctrl) / (mean_ctrl + 1e-10) * 100)

        tests["wilcoxon_paired"] = {
            "test": "Wilcoxon signed-rank (control > stim)",
            "n_pairs": len(paired_sessions),
            "W": float(W), "p": float(p),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "n_positive": sum(1 for d in diffs if d > 0),
            "n_negative": sum(1 for d in diffs if d < 0),
            "cohens_d": cohens_d,
            "pct_increase_photostim": pct_increase,
        }
        logger.info(f"  Wilcoxon paired: W={W:.1f}, p={p:.4f}, n={len(paired_sessions)}, "
                    f"mean_diff={np.mean(diffs):.4f}, "
                    f"pos/neg={tests['wilcoxon_paired']['n_positive']}/{tests['wilcoxon_paired']['n_negative']}")

    # Show paired sessions
    logger.info(f"\n  Paired sessions ({len(paired_sessions)}):")
    for ps in paired_sessions[:20]:
        direction = "ctrl>stim" if ps["diff"] > 0 else "STIM>ctrl"
        logger.info(f"    {ps['file'][:40]}: ctrl={ps['control']:.4f} stim={ps['photostim']:.4f} "
                    f"diff={ps['diff']:+.4f} {direction}")

    # --- Step 6: Print summary matching paper ---
    print(f"\n{'='*70}")
    print("SUMMARY (Table 5 — Svoboda ALM photoinhibition)")
    print(f"{'='*70}")
    print(f"  Sessions processed: {len(session_results)}")
    print(f"  Sessions skipped:   {n_skipped}")
    print(f"  Paired sessions:    {len(paired_sessions)}")
    if "control" in summary and summary["control"]["n"] > 0:
        print(f"\n  Control bracket_norm:    {summary['control']['mean']:.3f} +/- {summary['control']['std']:.3f} "
              f"(n={summary['control']['n']})")
    if "photostim" in summary and summary["photostim"]["n"] > 0:
        print(f"  Photostim bracket_norm:  {summary['photostim']['mean']:.3f} +/- {summary['photostim']['std']:.3f} "
              f"(n={summary['photostim']['n']})")
    if "wilcoxon_paired" in tests:
        t = tests["wilcoxon_paired"]
        print(f"\n  Wilcoxon signed-rank (paired, control > stim):")
        print(f"    W = {t['W']:.1f}, p = {t['p']:.2e}")
        print(f"    {t['n_positive']}/{t['n_pairs']} sessions have control > photostim")
        print(f"    Mean paired diff: {t['mean_diff']:.4f}")
        print(f"    Cohen's d: {t['cohens_d']:.3f}")
        print(f"    Photostim increase: {t['pct_increase_photostim']:.1f}%")
    if "mann_whitney" in tests:
        t = tests["mann_whitney"]
        print(f"\n  Mann-Whitney U (unpaired, control > stim):")
        print(f"    U = {t['U']:.1f}, p = {t['p']:.2e}")
    print(f"{'='*70}")

    # --- Step 7: Save results ---
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "Svoboda ALM (DANDI 000007/000009)",
        "task": "delayed response lick left/right",
        "normalization": "z-score per neuron",
        "control_definition": "photo_stim_id == 0 (or no photostim column)",
        "n_sessions_processed": len(session_results),
        "n_skipped": n_skipped,
        "regions_seen": dict(regions_seen),
        "paired_sessions": paired_sessions,
        "session_results": session_results,
        "summary": summary,
        "statistical_tests": tests,
        "total_time_sec": time.time() - t0,
    }

    save_results("svoboda_bracket_norm", results)
    logger.info(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
