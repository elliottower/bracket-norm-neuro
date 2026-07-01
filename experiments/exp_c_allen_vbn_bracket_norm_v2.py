"""Allen Visual Behavior Neuropixels bracket_norm v2 — streaming from S3.

Computes bracket_norm per region across Allen VBN sessions using NWB
streaming (remfile + h5py) — no full NWB download needed.

Uses the change-detection task: binary choice (lick vs no-lick),
evidence = trial_length (continuous — captures expectation buildup;
longer pre-change sequences = stronger change expectation).

V2 of the trial_length variant: adds rotation_angle, commutativity,
and bn_normalized (= bracket_norm / sqrt(n_neurons)) to every record.

CPU-only, ~40 min for 20 sessions (dominated by S3 spike time downloads).
"""
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import h5py
    import math
    import numpy as np
    import remfile
    from scipy.stats import spearmanr
    from tqdm import tqdm

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from crossval.bracket_norm_core import compute_bracket_norm

    sys.path.insert(0, str(Path(__file__).parent))
    from shared_bundle import save_results
except ImportError:
    pass

S3_BUCKET = "visual-behavior-neuropixels-data"
S3_PREFIX = "visual-behavior-neuropixels/behavior_ecephys_sessions"
MIN_NEURONS = 10
MIN_VALID_TRIALS = 30
PRE_TIME = 0.25
POST_TIME = 0.75
N_SESSIONS = 20


def get_session_ids():
    """Load session IDs from cached CSV or S3."""
    csv_path = Path(__file__).parent.parent / "data" / "cache" / "allen" / "ecephys_sessions.csv"
    if csv_path.exists():
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            return [int(row["ecephys_session_id"]) for row in reader]

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    import csv
    import io

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    obj = s3.get_object(
        Bucket=S3_BUCKET,
        Key="visual-behavior-neuropixels/project_metadata/ecephys_sessions.csv",
    )
    content = obj["Body"].read().decode("utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(content)
    reader = csv.DictReader(io.StringIO(content))
    return [int(row["ecephys_session_id"]) for row in reader]


def process_session(session_id):
    """Stream one session from S3, compute bracket_norm per region."""
    url = (
        f"https://{S3_BUCKET}.s3.us-west-2.amazonaws.com/"
        f"{S3_PREFIX}/{session_id}/ecephys_session_{session_id}.nwb"
    )

    try:
        rf = remfile.File(url)
        f = h5py.File(rf, "r")
    except Exception as e:
        print(f"    Failed to open session {session_id}: {e}")
        return None

    try:
        electrodes = f["general"]["extracellular_ephys"]["electrodes"]
        el_ids = electrodes["id"][:]
        el_locs = [
            v.decode() if isinstance(v, bytes) else v
            for v in electrodes["location"][:]
        ]
        id_to_loc = dict(zip(el_ids, el_locs))

        units = f["units"]
        peak_chs = units["peak_channel_id"][:]
        unit_areas = np.array([id_to_loc.get(ch, "unknown") for ch in peak_chs])

        trials = f["intervals"]["trials"]
        is_change = trials["is_change"][:]
        hit = trials["hit"][:]
        miss = trials["miss"][:]
        false_alarm = trials["false_alarm"][:]
        correct_reject = trials["correct_reject"][:]
        trial_starts = trials["start_time"][:]
        trial_length = trials["trial_length"][:]

        change_key = "change_time_no_display_delay"
        if change_key not in trials:
            change_key = "change_time"
        if change_key in trials:
            change_times = trials[change_key][:]
        else:
            change_times = trial_starts

        valid = hit | miss | false_alarm | correct_reject
        valid_idx = np.where(valid)[0]

        if len(valid_idx) < MIN_VALID_TRIALS:
            f.close()
            return None

        choice = (hit | false_alarm)[valid_idx].astype(int)
        evidence = trial_length[valid_idx]

        if len(set(choice)) < 2 or len(np.unique(evidence)) < 4:
            f.close()
            return None

        spike_times_data = units["spike_times"][:]
        spike_index = units["spike_times_index"][:]

        n_units_total = len(spike_index)
        unit_spikes = []
        prev = 0
        for i in range(n_units_total):
            end = spike_index[i]
            unit_spikes.append(spike_times_data[prev:end])
            prev = end

        align_times = np.where(
            is_change[valid_idx], change_times[valid_idx], trial_starts[valid_idx]
        )

        regions_to_test = [
            r
            for r, c in Counter(unit_areas).items()
            if c >= MIN_NEURONS and r != "unknown"
        ]

        session_results = {}
        for region in regions_to_test:
            region_mask = np.where(unit_areas == region)[0]
            n_neurons = len(region_mask)

            activity = np.zeros((len(valid_idx), n_neurons), dtype=np.float32)
            for j, ui in enumerate(region_mask):
                st = unit_spikes[ui]
                for t in range(len(valid_idx)):
                    at = align_times[t]
                    mask = (st >= at - PRE_TIME) & (st < at + POST_TIME)
                    activity[t, j] = mask.sum()

            bn = compute_bracket_norm(activity, choice, evidence)
            if bn is not None:
                session_results[region] = {
                    "bracket_norm": bn["bracket_norm"],
                    "rotation_angle": bn.get("rotation_angle"),
                    "commutativity": bn.get("commutativity"),
                    "bn_normalized": bn["bracket_norm"] / math.sqrt(n_neurons),
                    "n_neurons": n_neurons,
                }

        f.close()
        return {
            "session_id": session_id,
            "n_valid_trials": int(len(valid_idx)),
            "n_hit": int(hit.sum()),
            "n_miss": int(miss.sum()),
            "n_fa": int(false_alarm.sum()),
            "n_cr": int(correct_reject.sum()),
            "n_regions": len(session_results),
            "regions": session_results,
        }

    except Exception as e:
        print(f"    Error processing session {session_id}: {e}")
        f.close()
        return None


def run():
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] Allen VBN bracket_norm v2 (streaming)"
    )

    session_ids = get_session_ids()
    print(f"  {len(session_ids)} sessions available, processing {N_SESSIONS}")

    all_results = []
    region_bn = {}
    region_ra = {}
    region_bnn = {}
    region_nc = {}

    for i, sid in enumerate(tqdm(session_ids[:N_SESSIONS], desc="Sessions")):
        print(f"  [{i+1}/{N_SESSIONS}] Session {sid}...")
        result = process_session(sid)
        if result is None:
            print(f"    Skipped")
            continue

        all_results.append(result)
        for region, data in result["regions"].items():
            if region not in region_bn:
                region_bn[region] = []
                region_ra[region] = []
                region_bnn[region] = []
                region_nc[region] = []
            region_bn[region].append(data["bracket_norm"])
            if data["rotation_angle"] is not None:
                region_ra[region].append(data["rotation_angle"])
            region_bnn[region].append(data["bn_normalized"])
            region_nc[region].append(data["n_neurons"])

        print(
            f"    {result['n_valid_trials']} trials, {result['n_regions']} regions"
        )

    print(f"\n{'='*70}")
    print(f"ALLEN VBN BRACKET_NORM V2 RESULTS")
    print(f"{'='*70}")

    summary = {}
    for region in sorted(region_bn.keys()):
        summary[region] = {
            "bn_mean": float(np.mean(region_bn[region])),
            "bn_std": float(np.std(region_bn[region])),
            "ra_mean": float(np.mean(region_ra[region])) if region_ra[region] else None,
            "ra_std": float(np.std(region_ra[region])) if region_ra[region] else None,
            "bnn_mean": float(np.mean(region_bnn[region])),
            "bnn_std": float(np.std(region_bnn[region])),
            "nc_mean": float(np.mean(region_nc[region])),
            "n_sessions": len(region_bn[region]),
        }
        s = summary[region]
        ra_str = f"RA={s['ra_mean']:.2f}±{s['ra_std']:.2f}" if s["ra_mean"] is not None else "RA=n/a"
        print(
            f"  {region:8s}: BN={s['bn_mean']:.2f}±{s['bn_std']:.2f} "
            f"BNn={s['bnn_mean']:.4f}±{s['bnn_std']:.4f} "
            f"{ra_str} "
            f"nc={s['nc_mean']:.0f} n_sess={s['n_sessions']}"
        )

    if len(summary) >= 5:
        all_bn = [summary[r]["bn_mean"] for r in summary]
        all_nc = [summary[r]["nc_mean"] for r in summary]
        rho, p = spearmanr(all_bn, all_nc)
        print(f"\n  BN vs neuron count: rho={rho:+.3f} p={p:.4f}")

        all_bnn = [summary[r]["bnn_mean"] for r in summary]
        rho_bnn, p_bnn = spearmanr(all_bnn, all_nc)
        print(f"  BNn vs neuron count: rho={rho_bnn:+.3f} p={p_bnn:.4f}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "allen_vbn_bracket_norm_v2",
        "n_sessions_processed": len(all_results),
        "n_sessions_requested": N_SESSIONS,
        "n_regions": len(summary),
        "region_summary": summary,
        "session_results": all_results,
    }

    save_results("allen_vbn_bracket_norm_v2", results)
    print(f"\nDone.")


if __name__ == "__main__":
    run()
