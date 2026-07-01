"""Modal: Human ECoG bracket_norm — DANDI 000019 (Chang lab CV syllables).

Two-step: (1) download NWB files to Modal volume, (2) analyze.
This avoids the download-in-analysis-container pattern that failed for IBL/Svoboda.

Speech production adaptation of bracket_norm:
- "Choice" = consonant identity (binary: e.g., labial /b,p/ vs coronal /d,t/)
- "Evidence" = syllable duration (short = fast/automatized = high evidence)
- "Regions" = electrode locations grouped by cortical area
- No ESM data available — descriptive only

Usage:
    # Step 1: download (run once)
    modal run --detach experiments/crossval/modal_ecog_bracket_norm.py::download_data

    # Step 2: analyze (after download completes)
    modal run --detach experiments/crossval/modal_ecog_bracket_norm.py::analyze
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
        "dandi>=0.62",
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

app = modal.App("ecog-bracket-norm")
volume = modal.Volume.from_name("neuro-causal-geometry-results", create_if_missing=True)

DANDISET_ID = "000019"
DATA_DIR = "/results/ecog_data/000019"
RESULTS_DIR = "/results/ecog_bracket_norm"
JSONL_FILE = "ecog_bracket_norm_incremental.jsonl"

# Cortical areas for grouping electrodes (Chang lab conventions)
AREA_GROUPS = {
    "vSMC": ["vSMC", "ventral sensorimotor", "precentral", "postcentral",
             "rolandic", "central sulcus", "motor", "sensorimotor"],
    "STG": ["STG", "superior temporal", "Heschl", "planum temporale",
            "auditory", "temporal"],
    "IFG": ["IFG", "inferior frontal", "Broca", "pars opercularis",
            "pars triangularis", "frontal operculum"],
    "SMA": ["SMA", "supplementary motor", "medial frontal", "premotor",
            "pre-SMA"],
    "PMC": ["PMC", "premotor", "dorsal premotor", "lateral premotor"],
}

# Map raw electrode location strings to canonical areas
def classify_electrode(location_str):
    """Map electrode location to canonical cortical area."""
    if not location_str or location_str == "unknown":
        return None
    loc_lower = location_str.lower()
    for area, keywords in AREA_GROUPS.items():
        for kw in keywords:
            if kw.lower() in loc_lower:
                return area
    return location_str.split(",")[0].strip() if "," in location_str else location_str


# ---- Step 1: Download ----

@app.function(image=image, cpu=2, memory=16384, timeout=86400, volumes={"/results": volume})
def download_data(max_files=15):
    """Download DANDI:000019 NWB files to Modal volume."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logger = logging.getLogger("ecog-download")

    from dandi.dandiapi import DandiAPIClient
    from tqdm import tqdm

    t0 = time.time()
    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] Downloading DANDI:{DANDISET_ID}")

    out_dir = Path(DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID, "draft")
    assets = list(dandiset.get_assets())
    nwb_assets = [a for a in assets if a.path.endswith(".nwb")][:max_files]

    logger.info(f"  {len(nwb_assets)} NWB files to download (of {len(assets)} total)")

    downloaded = []
    for asset in tqdm(nwb_assets, desc="Downloading"):
        local_path = out_dir / asset.path.replace("/", "_")
        if local_path.exists() and local_path.stat().st_size > 1000:
            logger.info(f"  Already exists: {local_path.name}")
            downloaded.append(str(local_path))
            continue
        try:
            asset.download(local_path)
            downloaded.append(str(local_path))
            logger.info(f"  Downloaded: {local_path.name} ({local_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            logger.warning(f"  Failed: {asset.path}: {e}")

    volume.commit()
    logger.info(f"  {len(downloaded)} files downloaded in {time.time() - t0:.0f}s")
    return {"n_downloaded": len(downloaded), "files": downloaded}


# ---- Step 2: Analyze ----

@app.function(image=image, cpu=4, memory=32768, timeout=86400, volumes={"/results": volume})
def analyze():
    """Compute bracket_norm on pre-downloaded ECoG data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logger = logging.getLogger("ecog-bracket")

    import numpy as np
    import pynwb
    from scipy.signal import hilbert
    from tqdm import tqdm

    sys.path.insert(0, "/root")
    from bracket_norm_core import compute_bracket_norm, aggregate_region_metrics

    t0 = time.time()
    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ECoG bracket_norm analysis")

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(DATA_DIR)

    volume.reload()
    nwb_files = sorted(data_dir.glob("*.nwb"))
    logger.info(f"  {len(nwb_files)} NWB files found")

    if not nwb_files:
        result = {"error": "no NWB files found — run download_data first"}
        with open(out_dir / "ecog_bracket_norm_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        volume.commit()
        return result

    region_metrics = defaultdict(list)
    region_electrode_counts = defaultdict(list)
    n_sessions = 0

    for nwb_path in tqdm(nwb_files, desc="Processing"):
        logger.info(f"\n  Processing {nwb_path.name}...")
        try:
            io = pynwb.NWBHDF5IO(str(nwb_path), "r")
            nwbfile = io.read()
        except Exception as e:
            logger.warning(f"  Failed to open: {e}")
            continue

        try:
            # Get trials/epochs
            trials = nwbfile.trials
            if trials is None:
                # Try epochs
                if hasattr(nwbfile, 'epochs') and nwbfile.epochs is not None:
                    trials = nwbfile.epochs
                else:
                    logger.info(f"  SKIP: no trials or epochs")
                    io.close()
                    continue

            trial_cols = list(trials.colnames) if trials.colnames else []
            logger.info(f"  Trial columns: {trial_cols}")

            start_times = np.asarray(trials["start_time"].data[:])
            stop_times = np.asarray(trials["stop_time"].data[:]) if "stop_time" in trial_cols else start_times + 0.5

            # Find syllable/phoneme labels
            label_col = None
            for col in ["cv", "syllable", "consonant", "phoneme", "stimulus",
                        "trial_type", "condition", "label"]:
                if col in trial_cols:
                    label_col = col
                    break

            if label_col is None:
                # Try to extract from any string column
                for col in trial_cols:
                    if col in ("start_time", "stop_time", "id"):
                        continue
                    try:
                        vals = trials[col].data[:]
                        if hasattr(vals[0], 'decode') or isinstance(vals[0], str):
                            label_col = col
                            logger.info(f"  Using '{col}' as label column")
                            break
                    except Exception:
                        continue

            if label_col is None:
                logger.info(f"  SKIP: no label column found in {trial_cols}")
                io.close()
                continue

            raw_labels = trials[label_col].data[:]
            if hasattr(raw_labels[0], 'decode'):
                raw_labels = np.array([l.decode() for l in raw_labels])
            else:
                raw_labels = np.array([str(l) for l in raw_labels])

            unique_labels = np.unique(raw_labels)
            logger.info(f"  Labels ({label_col}): {len(unique_labels)} unique: {list(unique_labels[:10])}")

            # Binary split: group labels into two categories
            # Strategy: first letter (consonant) if CV syllables, else median split
            def get_consonant(label):
                label = label.strip().lower()
                if len(label) >= 2 and label[0].isalpha():
                    return label[0]
                return label

            consonants = np.array([get_consonant(l) for l in raw_labels])
            unique_cons = np.unique(consonants)

            if len(unique_cons) >= 2:
                # Binary: first half of consonants vs second half
                sorted_cons = sorted(unique_cons)
                mid = len(sorted_cons) // 2
                group_a = set(sorted_cons[:mid])
                choice_binary = np.array([0 if c in group_a else 1 for c in consonants])
                logger.info(f"  Binary split: {sorted_cons[:mid]} vs {sorted_cons[mid:]}")
            else:
                logger.info(f"  SKIP: only {len(unique_cons)} consonant categories")
                io.close()
                continue

            # Evidence = syllable duration (inverse = speed)
            durations = stop_times - start_times
            valid = np.isfinite(durations) & (durations > 0.01) & (durations < 5.0)
            evidence = 1.0 / (durations + 0.01)  # fast = high evidence

            valid &= np.isfinite(evidence)
            n_class0 = (choice_binary[valid] == 0).sum()
            n_class1 = (choice_binary[valid] == 1).sum()
            if min(n_class0, n_class1) < 10:
                logger.info(f"  SKIP: class imbalance ({n_class0} vs {n_class1})")
                io.close()
                continue

            trial_idx = np.where(valid)[0]

            # Get electrode data
            # ECoG data might be in acquisition, processing, or electrodes
            ecog_data = None
            ecog_key = None

            # Check acquisition
            for key in nwbfile.acquisition:
                obj = nwbfile.acquisition[key]
                if hasattr(obj, 'data') and hasattr(obj, 'electrodes'):
                    ecog_data = obj
                    ecog_key = key
                    break

            # Check processing modules
            if ecog_data is None:
                for mod_name in nwbfile.processing:
                    mod = nwbfile.processing[mod_name]
                    for key in mod.data_interfaces:
                        obj = mod.data_interfaces[key]
                        if hasattr(obj, 'data') and hasattr(obj, 'electrodes'):
                            ecog_data = obj
                            ecog_key = f"{mod_name}/{key}"
                            break
                    if ecog_data is not None:
                        break

            if ecog_data is None:
                logger.info(f"  SKIP: no ECoG data found")
                logger.info(f"    acquisition keys: {list(nwbfile.acquisition.keys())}")
                logger.info(f"    processing keys: {list(nwbfile.processing.keys())}")
                io.close()
                continue

            logger.info(f"  ECoG data: {ecog_key}, shape={ecog_data.data.shape}")

            # Get electrode locations
            electrodes = nwbfile.electrodes
            if electrodes is None:
                logger.info(f"  SKIP: no electrodes table")
                io.close()
                continue

            elec_cols = list(electrodes.colnames) if electrodes.colnames else []
            location_col = None
            for col in ["location", "group_name", "label", "brain_area"]:
                if col in elec_cols:
                    location_col = col
                    break

            if location_col is None:
                logger.info(f"  SKIP: no location column in electrodes: {elec_cols}")
                io.close()
                continue

            n_electrodes = len(electrodes)
            electrode_areas = []
            for ei in range(n_electrodes):
                raw_loc = str(electrodes[location_col][ei])
                area = classify_electrode(raw_loc)
                electrode_areas.append(area)

            logger.info(f"  {n_electrodes} electrodes, areas: {dict(zip(*np.unique(electrode_areas, return_counts=True)))}")

            # Get sampling rate
            rate = ecog_data.rate if hasattr(ecog_data, 'rate') else 1000.0
            data_array = ecog_data.data

            # For each area with enough electrodes, compute bracket_norm
            area_electrodes = defaultdict(list)
            for ei, area in enumerate(electrode_areas):
                if area is not None:
                    area_electrodes[area].append(ei)

            for area, elec_idx in area_electrodes.items():
                if len(elec_idx) < 3:
                    continue

                # Extract trial-aligned high-gamma power
                # Window: 0 to 500ms post-onset
                pre_samples = int(0.05 * rate)
                post_samples = int(0.5 * rate)
                window_samples = pre_samples + post_samples

                n_valid_trials = len(trial_idx)
                activity = np.zeros((n_valid_trials, len(elec_idx)), dtype=np.float32)

                for ti_local, ti_global in enumerate(trial_idx):
                    onset_sample = int(start_times[ti_global] * rate)
                    s_start = max(0, onset_sample - pre_samples)
                    s_end = min(data_array.shape[0], onset_sample + post_samples)

                    if s_end - s_start < window_samples // 2:
                        continue

                    try:
                        chunk = np.array(data_array[s_start:s_end, :], dtype=np.float64)
                        for ei_local, ei_global in enumerate(elec_idx):
                            if ei_global < chunk.shape[1]:
                                signal = chunk[:, ei_global]
                                # High-gamma power (rough: mean absolute value as proxy)
                                activity[ti_local, ei_local] = float(np.mean(np.abs(signal)))
                    except Exception:
                        continue

                # Remove zero-activity electrodes
                active = activity.std(axis=0) > 0
                if active.sum() < 3:
                    continue
                activity = activity[:, active]

                result = compute_bracket_norm(
                    activity,
                    choice_binary[trial_idx],
                    evidence[trial_idx],
                    min_per_quartile=3,
                )

                if result is not None:
                    result["session"] = nwb_path.name
                    result["n_electrodes"] = int(active.sum())
                    result["n_trials"] = n_valid_trials
                    result["area"] = area
                    result["timestamp"] = datetime.now(timezone.utc).isoformat()
                    region_metrics[area].append(result)
                    region_electrode_counts[area].append(int(active.sum()))
                    logger.info(f"    {area}: bracket_norm={result['bracket_norm']:.4f}, "
                               f"n_elec={active.sum()}, rot={result['rotation_angle']:.3f}")
                    # JSONL incremental save
                    jsonl_path = out_dir / JSONL_FILE
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(result, default=str) + "\n")
                    volume.commit()

            n_sessions += 1
            io.close()

        except Exception as e:
            logger.warning(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            try:
                io.close()
            except Exception:
                pass

    for area in region_electrode_counts:
        region_electrode_counts[area] = float(np.mean(region_electrode_counts[area]))

    summary = aggregate_region_metrics(dict(region_metrics))

    logger.info(f"\n{'='*70}")
    logger.info(f"ECoG bracket_norm: {len(summary)} areas, {n_sessions} sessions")
    logger.info(f"  {'Area':>12s}  {'n':>3s}  {'Bracket':>8s}  {'Rot':>6s}  {'Comm':>6s}  {'nElec':>5s}")
    logger.info(f"  {'-'*55}")
    for area in sorted(summary.keys()):
        s = summary[area]
        nc = region_electrode_counts.get(area, 0)
        logger.info(f"  {area:>12s}  {s['n_sessions']:3d}  "
                    f"{s.get('bracket_norm_mean', 0):8.4f}  "
                    f"{s.get('rotation_angle_mean', 0):6.3f}  "
                    f"{s.get('commutativity_mean', 0):6.3f}  "
                    f"{nc:5.0f}")

    # No silencing ground truth — report descriptive stats only
    # Check if vSMC (known speech-critical) has higher bracket_norm than others
    if "vSMC" in summary and len(summary) >= 3:
        vsmc_bn = summary["vSMC"].get("bracket_norm_mean", 0)
        other_bns = [s.get("bracket_norm_mean", 0) for area, s in summary.items() if area != "vSMC"]
        if other_bns:
            logger.info(f"\n  vSMC bracket_norm: {vsmc_bn:.4f}")
            logger.info(f"  Other areas mean: {np.mean(other_bns):.4f}")
            logger.info(f"  vSMC is {'higher' if vsmc_bn > np.mean(other_bns) else 'lower'} "
                       f"than average ({vsmc_bn / (np.mean(other_bns) + 1e-10):.2f}x)")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "DANDI:000019 Chang lab CV syllables (Human ECoG)",
        "task": "speech production (consonant-vowel syllables)",
        "choice_definition": "consonant identity (binary split)",
        "evidence_definition": "1/syllable_duration (fast = high evidence)",
        "causal_ground_truth": "none (no ESM data in dataset)",
        "n_areas": len(summary),
        "n_sessions": n_sessions,
        "region_summary": {k: v for k, v in summary.items()},
        "electrode_counts": dict(region_electrode_counts),
        "total_time_sec": time.time() - t0,
    }

    with open(out_dir / "ecog_bracket_norm_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    volume.commit()

    logger.info(f"\nDone in {time.time() - t0:.0f}s")
    return results


# ---- Entrypoints ----

@app.local_entrypoint()
def main(step: str = "both", max_files: int = 15):
    """Run download, analyze, or both.

    Usage:
        modal run --detach ... -- --step download
        modal run --detach ... -- --step analyze
        modal run --detach ... -- --step both
    """
    if step in ("download", "both"):
        dl_result = download_data.remote(max_files=max_files)
        print(f"Download: {json.dumps(dl_result, indent=2, default=str)}")

    if step in ("analyze", "both"):
        result = analyze.remote()
        print(f"Analysis: {json.dumps(result, indent=2, default=str)[:20000]}")
