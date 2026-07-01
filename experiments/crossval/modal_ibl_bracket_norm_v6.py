"""Modal: IBL bracket_norm v6 — adds rotation_angle + BN/sqrt(n).

Same as v5 but stores rotation_angle, commutativity, and bn_normalized
per record. Fresh JSONL (v6 suffix) so it recomputes everything.

Usage:
    modal run --detach experiments/crossval/modal_ibl_bracket_norm_v6.py
"""
import json
import logging
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import modal

_this_dir = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "setuptools<71", "wheel", "Cython",
        "numpy>=1.24,<2",
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

app = modal.App("ibl-bracket-norm-v6")
volume = modal.Volume.from_name("neuro-causal-geometry-results", create_if_missing=True)

DATA_DIR = "/results/ibl_data"
RESULTS_DIR = "/results/ibl_bracket_norm_v6"
JSONL_FILE = "ibl_bracket_norm_v6_incremental.jsonl"

STEINMETZ_SILENCING = {
    "VISp": 0.1414, "VISl": 0.1722, "VISpm": 0.2248, "VISam": 0.0818,
    "ACA": 0.1451, "MOs": 0.1529, "ORB": 0.3085, "PL": 0.3333,
    "RSP": 0.1421,
}

REGION_MAP = {
    "VISp": "VISp", "VISp1": "VISp", "VISp2/3": "VISp", "VISp4": "VISp",
    "VISp5": "VISp", "VISp6a": "VISp", "VISp6b": "VISp",
    "VISl": "VISl", "VISl1": "VISl", "VISl2/3": "VISl", "VISl4": "VISl",
    "VISl5": "VISl", "VISl6a": "VISl", "VISl6b": "VISl",
    "VISpm": "VISpm", "VISpm1": "VISpm", "VISpm2/3": "VISpm", "VISpm4": "VISpm",
    "VISpm5": "VISpm", "VISpm6a": "VISpm", "VISpm6b": "VISpm",
    "VISam": "VISam", "VISam1": "VISam", "VISam2/3": "VISam", "VISam4": "VISam",
    "VISam5": "VISam", "VISam6a": "VISam", "VISam6b": "VISam",
    "VISrl": "VISrl", "VISrl1": "VISrl", "VISrl2/3": "VISrl", "VISrl4": "VISrl",
    "VISrl5": "VISrl", "VISrl6a": "VISrl", "VISrl6b": "VISrl",
    "VISa": "VISa", "VISa1": "VISa", "VISa2/3": "VISa", "VISa4": "VISa",
    "VISa5": "VISa", "VISa6a": "VISa", "VISa6b": "VISa",
    "ACA": "ACA", "ACAd": "ACA", "ACAv": "ACA",
    "ACAd1": "ACA", "ACAd2/3": "ACA", "ACAd5": "ACA", "ACAd6a": "ACA",
    "ACAv1": "ACA", "ACAv2/3": "ACA", "ACAv5": "ACA", "ACAv6a": "ACA",
    "MOs": "MOs", "MOs1": "MOs", "MOs2/3": "MOs", "MOs5": "MOs", "MOs6a": "MOs",
    "ORB": "ORB", "ORBl": "ORB", "ORBm": "ORB", "ORBvl": "ORB",
    "ORBl1": "ORB", "ORBl2/3": "ORB", "ORBl5": "ORB", "ORBl6a": "ORB",
    "PL": "PL", "PL1": "PL", "PL2/3": "PL", "PL5": "PL", "PL6a": "PL",
    "RSP": "RSP", "RSPv": "RSP", "RSPd": "RSP", "RSPagl": "RSP",
    "RSPv1": "RSP", "RSPv2/3": "RSP", "RSPv5": "RSP", "RSPv6a": "RSP",
    "RSPd1": "RSP", "RSPd2/3": "RSP", "RSPd5": "RSP", "RSPd6a": "RSP",
    "CA1": "CA1", "CA2": "CA2", "CA3": "CA3", "DG": "DG",
    "DG-mo": "DG", "DG-po": "DG", "DG-sg": "DG",
    "LP": "LP", "LGd": "LGd", "LGd-sh": "LGd", "LGd-co": "LGd",
    "LD": "LD", "PO": "PO", "VPM": "VPM", "VPL": "VPL",
    "RT": "RT", "MD": "MD",
    "SC": "SC", "SCm": "SC", "SCig": "SC", "SCsg": "SC", "SCop": "SC",
    "SCdg": "SC", "SCdw": "SC", "SCiw": "SC",
    "MRN": "MRN", "SNr": "SNr",
    "SSp": "SSp", "SSp-bfd": "SSp-bfd",
    "SSp-bfd1": "SSp-bfd", "SSp-bfd2/3": "SSp-bfd", "SSp-bfd4": "SSp-bfd",
    "SSp-bfd5": "SSp-bfd", "SSp-bfd6a": "SSp-bfd", "SSp-bfd6b": "SSp-bfd",
    "MOp": "MOp", "MOp1": "MOp", "MOp2/3": "MOp", "MOp5": "MOp", "MOp6a": "MOp",
    "CP": "CP", "ACB": "ACB", "LSr": "LSr", "LSc": "LSc",
    "CENT2": "CENT", "CENT3": "CENT",
    "ZI": "ZI",
}

DECISION_WINDOW_SEC = (0.05, 0.35)
MIN_NEURONS = 10
MIN_TRIALS = 30


def load_completed(jsonl_path):
    """Load already-computed session/probe keys from JSONL."""
    done = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(f"{rec['session_eid']}/{rec['probe']}")
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


@app.function(image=image, cpu=4, memory=32768, timeout=86400, volumes={"/results": volume})
def analyze():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logger = logging.getLogger("ibl-v5")

    import numpy as np
    from scipy.stats import spearmanr
    from tqdm import tqdm

    sys.path.insert(0, "/root")
    from bracket_norm_core import compute_bracket_norm

    t0 = time.time()
    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] IBL bracket_norm v6 (with rotation_angle + BN/sqrt(n))")

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    volume.reload()

    jsonl_path = out_dir / JSONL_FILE
    done_keys = load_completed(jsonl_path)
    if done_keys:
        logger.info(f"  Resuming: {len(done_keys)} session/probe pairs already computed")

    data_dir = Path(DATA_DIR)
    session_dirs = [d for d in sorted(data_dir.iterdir())
                    if d.is_dir() and (d / "manifest.json").exists()]
    logger.info(f"  {len(session_dirs)} sessions on volume")

    region_observations = defaultdict(list)
    n_processed = 0
    n_skipped = 0
    n_with_regions = 0
    n_region_observations = 0
    n_errors = 0

    for sess_dir in tqdm(session_dirs, desc="Processing"):
        eid = sess_dir.name
        manifest = json.load(open(sess_dir / "manifest.json"))

        for probe_name in manifest.get("probes", {}):
            key = f"{eid}/{probe_name}"
            if key in done_keys:
                n_skipped += 1
                continue

            acronym_path = sess_dir / f"{probe_name}_cluster_acronyms.npy"
            spike_times_path = sess_dir / f"{probe_name}_spike_times.npy"
            spike_clusters_path = sess_dir / f"{probe_name}_spike_clusters.npy"

            if not acronym_path.exists():
                continue
            if not spike_times_path.exists() or not spike_clusters_path.exists():
                continue

            try:
                acronyms = np.load(str(acronym_path), allow_pickle=True)
                spike_times = np.load(str(spike_times_path)).flatten()
                spike_clusters = np.load(str(spike_clusters_path)).flatten()

                choice_path = sess_dir / "trials_choice.npy"
                contrast_l_path = sess_dir / "trials_contrastLeft.npy"
                contrast_r_path = sess_dir / "trials_contrastRight.npy"
                stim_on_path = sess_dir / "trials_stimOn_times.npy"

                if not all(p.exists() for p in [choice_path, contrast_l_path, contrast_r_path, stim_on_path]):
                    continue

                choice = np.load(str(choice_path)).flatten()
                contrast_l = np.load(str(contrast_l_path)).flatten()
                contrast_r = np.load(str(contrast_r_path)).flatten()
                stim_on = np.load(str(stim_on_path)).flatten()

                valid = (choice != 0) & ~np.isnan(stim_on)
                if valid.sum() < MIN_TRIALS:
                    continue

                choice_binary = np.zeros(len(choice), dtype=int)
                choice_binary[choice == -1] = 1
                valid_idx = np.where(valid)[0]

                cl = np.nan_to_num(contrast_l, nan=0.0)
                cr = np.nan_to_num(contrast_r, nan=0.0)
                evidence = np.abs(cr - cl)

                unique_clusters = np.unique(spike_clusters)
                cluster_to_region = {}
                for ci, cid in enumerate(unique_clusters):
                    if ci < len(acronyms):
                        raw = str(acronyms[ci])
                        mapped = REGION_MAP.get(raw, raw)
                        if mapped not in ("void", "root", "", "nan"):
                            cluster_to_region[int(cid)] = mapped

                if not cluster_to_region:
                    continue

                n_with_regions += 1

                # Pre-sort spike times per cluster ONCE (the key optimization)
                cluster_spike_times = {}
                for cid in np.unique(spike_clusters):
                    cid_int = int(cid)
                    if cid_int in cluster_to_region:
                        cluster_spike_times[cid_int] = np.sort(
                            spike_times[spike_clusters == cid])

                region_clusters = defaultdict(list)
                for cid, region in cluster_to_region.items():
                    region_clusters[region].append(cid)

                n_valid = len(valid_idx)
                dt = DECISION_WINDOW_SEC[1] - DECISION_WINDOW_SEC[0]
                t_starts = stim_on[valid_idx] + DECISION_WINDOW_SEC[0]
                t_ends = stim_on[valid_idx] + DECISION_WINDOW_SEC[1]

                probe_results = []
                for region, cluster_ids in region_clusters.items():
                    if len(cluster_ids) < MIN_NEURONS:
                        continue

                    activity = np.zeros((n_valid, len(cluster_ids)), dtype=np.float32)
                    for ci, cid in enumerate(cluster_ids):
                        st = cluster_spike_times[cid]
                        # searchsorted: count spikes in [t_start, t_end) for ALL trials at once
                        left_idx = np.searchsorted(st, t_starts, side="left")
                        right_idx = np.searchsorted(st, t_ends, side="left")
                        activity[:, ci] = (right_idx - left_idx) / dt

                    means = activity.mean(axis=0, keepdims=True)
                    stds = activity.std(axis=0, keepdims=True)
                    stds[stds < 1e-8] = 1.0
                    activity_z = (activity - means) / stds

                    res = compute_bracket_norm(
                        activity_z, choice_binary[valid], evidence[valid])
                    if res:
                        n_neur = len(cluster_ids)
                        rec = {
                            "session_eid": eid,
                            "probe": probe_name,
                            "region": region,
                            "bracket_norm": res["bracket_norm"],
                            "rotation_angle": res["rotation_angle"],
                            "commutativity": res["commutativity"],
                            "bn_normalized": res["bracket_norm"] / np.sqrt(n_neur),
                            "n_neurons": n_neur,
                            "n_trials": n_valid,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        probe_results.append(rec)
                        region_observations[region].append(rec)
                        n_region_observations += 1
                        logger.info(f"  {eid[:8]}/{probe_name}: {region} "
                                   f"BN={res['bracket_norm']:.4f} rot={res['rotation_angle']:.3f} "
                                   f"n={n_neur}")

                if probe_results:
                    with open(jsonl_path, "a") as f:
                        for rec in probe_results:
                            f.write(json.dumps(rec, default=str) + "\n")
                    volume.commit()

            except Exception as e:
                n_errors += 1
                logger.error(f"  ERROR {eid[:8]}/{probe_name}: {type(e).__name__}: {e}")
                traceback.print_exc()

        n_processed += 1

        if n_processed % 10 == 0:
            logger.info(f"  Progress: {n_processed}/{len(session_dirs)} sessions, "
                       f"{n_skipped} skipped, {n_region_observations} observations, "
                       f"{n_errors} errors")

    # ============================================================
    # AGGREGATE from full JSONL (includes prior runs)
    # ============================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"AGGREGATING from JSONL (all runs combined)")

    all_region_obs = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                all_region_obs[rec["region"]].append(rec)
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info(f"  {sum(len(v) for v in all_region_obs.values())} total observations "
               f"across {len(all_region_obs)} regions\n")

    region_summary = {}
    for region, obs_list in sorted(all_region_obs.items(), key=lambda x: -len(x[1])):
        bns = [o["bracket_norm"] for o in obs_list]
        rots = [o["rotation_angle"] for o in obs_list]
        comms = [o["commutativity"] for o in obs_list]
        bn_norms = [o["bn_normalized"] for o in obs_list]
        ns = [o["n_neurons"] for o in obs_list]
        region_summary[region] = {
            "n_observations": len(obs_list),
            "bracket_norm_mean": float(np.mean(bns)),
            "bracket_norm_std": float(np.std(bns)),
            "rotation_angle_mean": float(np.mean(rots)),
            "rotation_angle_std": float(np.std(rots)),
            "commutativity_mean": float(np.mean(comms)),
            "bn_normalized_mean": float(np.mean(bn_norms)),
            "bn_normalized_std": float(np.std(bn_norms)),
            "n_neurons_mean": float(np.mean(ns)),
        }
        logger.info(f"  {region:10s}: BN={np.mean(bns):.4f} rot={np.mean(rots):.3f} "
                    f"BN/√n={np.mean(bn_norms):.4f} (n={len(obs_list)}, neurons={np.mean(ns):.0f})")

    logger.info(f"\nSTEINMETZ CORRELATION — ALL THREE METRICS")
    mx_bn, mx_rot, mx_bnn, my, mn, regions_used = [], [], [], [], [], []
    for region, effect in STEINMETZ_SILENCING.items():
        if region in region_summary:
            s = region_summary[region]
            mx_bn.append(s["bracket_norm_mean"])
            mx_rot.append(s["rotation_angle_mean"])
            mx_bnn.append(s["bn_normalized_mean"])
            my.append(effect)
            mn.append(s["n_neurons_mean"])
            regions_used.append(region)
            logger.info(f"  {region:6s}: BN={s['bracket_norm_mean']:.4f} "
                       f"rot={s['rotation_angle_mean']:.3f} "
                       f"BN/√n={s['bn_normalized_mean']:.4f} "
                       f"silencing={effect:.4f}")

    corr = {}
    if len(mx_bn) >= 4:
        from bracket_norm_core import partial_spearman

        for label, vals in [("BN_raw", mx_bn), ("rotation_angle", mx_rot), ("BN_normalized", mx_bnn)]:
            rho, p = spearmanr(vals, my)
            partial = partial_spearman(np.array(vals), np.array(my), np.array(mn))
            rho_n, p_n = spearmanr(vals, mn)
            logger.info(f"\n  {label}:")
            logger.info(f"    vs silencing:     rho={rho:+.3f} (p={p:.4f})")
            logger.info(f"    partial (ctrl n): rho={partial:+.3f}")
            logger.info(f"    vs neuron count:  rho={rho_n:+.3f} (p={p_n:.4f})")
            corr[label] = {
                "spearman_rho": float(rho), "spearman_p": float(p),
                "partial_rho": float(partial),
                "rho_vs_n": float(rho_n), "p_vs_n": float(p_n),
            }

        corr["n_regions"] = len(mx_bn)
        corr["regions"] = regions_used
        corr["silencing_values"] = my
        corr["bn_values"] = mx_bn
        corr["rotation_angle_values"] = mx_rot
        corr["bn_normalized_values"] = mx_bnn
        corr["n_neurons_values"] = mn
    else:
        logger.info(f"  Only {len(mx_bn)} overlapping regions — insufficient for correlation")

    extra_regions = [r for r in region_summary if r not in STEINMETZ_SILENCING]
    if extra_regions:
        logger.info(f"\n  Extra regions (not in Steinmetz): {extra_regions[:20]}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_sessions_total": len(session_dirs),
        "n_sessions_processed_this_run": n_processed,
        "n_skipped_from_checkpoint": n_skipped,
        "n_errors": n_errors,
        "n_region_observations": sum(len(v) for v in all_region_obs.values()),
        "n_regions": len(region_summary),
        "region_summary": region_summary,
        "steinmetz_correlation": corr,
        "total_time_sec": time.time() - t0,
    }

    with open(out_dir / "ibl_bracket_norm_v6_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    volume.commit()

    logger.info(f"\nDone in {time.time() - t0:.0f}s ({n_errors} errors)")


@app.local_entrypoint()
def main():
    analyze.spawn()
    print("IBL bracket_norm v6 spawned (with rotation_angle + BN/sqrt(n)).")
