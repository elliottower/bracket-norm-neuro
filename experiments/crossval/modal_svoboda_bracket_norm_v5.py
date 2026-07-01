"""Modal: Svoboda ALM bracket_norm v5 — control vs photostim via photo_stim_id.

Control = photo_stim_id == 0, Photostim = photo_stim_id != 0.
Also normalizes activity (z-score per neuron) so bracket_norm is
scale-invariant across datasets.

Usage:
    modal run --detach experiments_crossval/modal_svoboda_bracket_norm_v5.py
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

app = modal.App("svoboda-bracket-norm-v5")
volume = modal.Volume.from_name("neuro-causal-geometry-results", create_if_missing=True)

DATA_DIR = "/results/svoboda_data"
RESULTS_DIR = "/results/svoboda_bracket_norm_v5"

DECISION_WINDOW_SEC = (0.0, 0.3)
MIN_NEURONS = 10
MIN_TRIALS = 20


@app.function(image=image, cpu=4, memory=32768, timeout=86400, volumes={"/results": volume})
def analyze():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logger = logging.getLogger("svoboda-v5")

    import numpy as np
    import pynwb
    from scipy.stats import mannwhitneyu, wilcoxon
    from tqdm import tqdm

    sys.path.insert(0, "/root")
    from bracket_norm_core import compute_bracket_norm

    t0 = time.time()
    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] Svoboda bracket_norm v5")

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    volume.reload()

    nwb_files = []
    for ds_dir in sorted(Path(DATA_DIR).iterdir()):
        if ds_dir.is_dir():
            nwb_files.extend(sorted(ds_dir.glob("*.nwb")))

    logger.info(f"  {len(nwb_files)} NWB files")

    all_bns = []
    control_bns = []
    photostim_bns = []
    session_results = []
    paired_sessions = []
    n_skipped = 0
    regions_seen = defaultdict(int)

    for nwb_path in tqdm(nwb_files, desc="Processing"):
        try:
            io = pynwb.NWBHDF5IO(str(nwb_path), "r")
            nwbfile = io.read()
        except Exception:
            n_skipped += 1
            continue

        try:
            units = nwbfile.units
            if units is None or len(units) < MIN_NEURONS:
                io.close()
                n_skipped += 1
                continue

            trials = nwbfile.trials
            if trials is None:
                io.close()
                n_skipped += 1
                continue

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
            regions_seen[region] += 1

            # Choice
            if "response" not in trial_cols:
                io.close()
                n_skipped += 1
                continue

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
                n_skipped += 1
                continue

            start_times = np.asarray(trials["start_time"].data[:])

            # Spike times
            spike_times_list = [np.asarray(units["spike_times"][i]) for i in range(n_units)]

            # Build activity matrix (ALL valid trials)
            valid_idx = np.where(valid_trials)[0]
            n_valid = len(valid_idx)
            activity = np.zeros((n_valid, n_units), dtype=np.float32)
            for ti_local, ti_global in enumerate(valid_idx):
                t_start = start_times[ti_global] + DECISION_WINDOW_SEC[0]
                t_end = start_times[ti_global] + DECISION_WINDOW_SEC[1]
                for ui in range(n_units):
                    st = spike_times_list[ui]
                    activity[ti_local, ui] = np.sum((st >= t_start) & (st < t_end)) / (t_end - t_start)

            # Z-score normalize per neuron (scale-invariant bracket_norm)
            means = activity.mean(axis=0, keepdims=True)
            stds = activity.std(axis=0, keepdims=True)
            stds[stds < 1e-8] = 1.0
            activity_z = (activity - means) / stds

            choice_valid = choice_binary[valid_trials]

            # Evidence proxy from stimulus (if stim_present column exists)
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
                all_bns.append(all_result["bracket_norm"])

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

            if is_control is not None:
                n_ctrl = is_control.sum()
                n_stim = is_stim.sum()
                sess_info["n_control"] = int(n_ctrl)
                sess_info["n_stim"] = int(n_stim)

                ctrl_bn = None
                stim_bn = None

                if n_ctrl >= MIN_TRIALS and len(np.unique(choice_valid[is_control])) >= 2:
                    ctrl_result = compute_bracket_norm(
                        activity_z[is_control], choice_valid[is_control], evidence[is_control]
                    )
                    if ctrl_result:
                        ctrl_bn = ctrl_result["bracket_norm"]
                        sess_info["bracket_norm_control"] = ctrl_bn
                        control_bns.append(ctrl_bn)

                if n_stim >= MIN_TRIALS and len(np.unique(choice_valid[is_stim])) >= 2:
                    stim_result = compute_bracket_norm(
                        activity_z[is_stim], choice_valid[is_stim], evidence[is_stim]
                    )
                    if stim_result:
                        stim_bn = stim_result["bracket_norm"]
                        sess_info["bracket_norm_photostim"] = stim_bn
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
            logger.info(f"  {nwb_path.name[:50]}: {region} n={n_units} {bn_str} {ctrl_str} {stim_str}")

            io.close()

        except Exception as e:
            logger.warning(f"  Error on {nwb_path.name}: {e}")
            import traceback
            traceback.print_exc()
            try:
                io.close()
            except Exception:
                pass

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

    # Unpaired test
    tests = {}
    if len(control_bns) >= 5 and len(photostim_bns) >= 5:
        U, p = mannwhitneyu(control_bns, photostim_bns, alternative="greater")
        tests["mann_whitney"] = {
            "test": "Mann-Whitney U (control > stim)",
            "U": float(U), "p": float(p),
            "n_control": len(control_bns), "n_stim": len(photostim_bns),
        }
        logger.info(f"  Mann-Whitney: U={U:.1f}, p={p:.4f}")

    # Paired test
    if len(paired_sessions) >= 5:
        ctrl_vals = [s["control"] for s in paired_sessions]
        stim_vals = [s["photostim"] for s in paired_sessions]
        W, p = wilcoxon(ctrl_vals, stim_vals, alternative="greater")
        diffs = [s["diff"] for s in paired_sessions]
        tests["wilcoxon_paired"] = {
            "test": "Wilcoxon signed-rank (control > stim)",
            "n_pairs": len(paired_sessions),
            "W": float(W), "p": float(p),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "n_positive": sum(1 for d in diffs if d > 0),
            "n_negative": sum(1 for d in diffs if d < 0),
        }
        logger.info(f"  Wilcoxon paired: W={W:.1f}, p={p:.4f}, n={len(paired_sessions)}, "
                    f"mean_diff={np.mean(diffs):.4f}, pos/neg={tests['wilcoxon_paired']['n_positive']}/{tests['wilcoxon_paired']['n_negative']}")

    # Show paired sessions
    logger.info(f"\n  Paired sessions ({len(paired_sessions)}):")
    for ps in paired_sessions[:20]:
        logger.info(f"    {ps['file'][:40]}: ctrl={ps['control']:.4f} stim={ps['photostim']:.4f} "
                    f"diff={ps['diff']:+.4f} {'ctrl>stim' if ps['diff'] > 0 else 'STIM>ctrl'}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "Svoboda ALM (DANDI 000007/000009)",
        "task": "delayed response lick left/right",
        "normalization": "z-score per neuron",
        "control_definition": "photo_stim_id == 0 (or no photostim column)",
        "n_sessions_processed": len(session_results),
        "n_skipped": n_skipped,
        "regions_seen": dict(regions_seen),
        "summary": summary,
        "statistical_tests": tests,
        "paired_sessions": paired_sessions,
        "session_results": session_results,
        "total_time_sec": time.time() - t0,
    }

    with open(out_dir / "svoboda_bracket_norm_v5_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    volume.commit()

    logger.info(f"\nDone in {time.time() - t0:.0f}s")


@app.local_entrypoint()
def main():
    analyze.remote()
    print("Svoboda bracket_norm v5 submitted. Results at /results/svoboda_bracket_norm_v5/")
