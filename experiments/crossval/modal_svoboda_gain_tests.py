"""Modal: Recurrent gain reduction theory — confirmatory tests.

Tests:
1. Population autocorrelation timescale (ctrl vs stim)
2. Evidence-level accuracy split (low vs high evidence, ctrl vs stim)
3. Temporal bracket_norm profile (sliding window across trial)
4. Choice signal latency (when does peak selectivity occur?)
5. Noise correlation structure (eigenspectrum shape change)

Usage:
    modal run --detach experiments_crossval/modal_svoboda_gain_tests.py
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

app = modal.App("svoboda-gain-tests")
volume = modal.Volume.from_name("neuro-causal-geometry-results", create_if_missing=True)

DATA_DIR = "/results/svoboda_data"
RESULTS_DIR = "/results/svoboda_gain_tests"

MIN_NEURONS = 10
MIN_TRIALS = 20

# Full trial window for temporal analyses (relative to trial start)
TIME_BINS_SEC = [(t / 1000, (t + 100) / 1000) for t in range(-200, 1500, 50)]
DECISION_WINDOW_SEC = (0.0, 0.3)


def population_autocorrelation(spike_times_list, trial_starts, trial_mask, n_units,
                                bin_size=0.01, max_lag_bins=30):
    """Compute population-averaged autocorrelation timescale."""
    import numpy as np

    valid_starts = trial_starts[trial_mask]
    # Use a fixed 1.5s window per trial
    window = 1.5
    n_bins = int(window / bin_size)

    # Bin spikes per trial
    all_binned = []
    for t_start in valid_starts[:50]:  # cap at 50 trials for speed
        trial_binned = np.zeros((n_bins, n_units), dtype=np.float32)
        for ui in range(n_units):
            st = spike_times_list[ui]
            for b in range(n_bins):
                b_start = t_start + b * bin_size
                b_end = b_start + bin_size
                trial_binned[b, ui] = np.sum((st >= b_start) & (st < b_end))
        all_binned.append(trial_binned)

    if not all_binned:
        return None

    all_binned = np.array(all_binned)  # (n_trials, n_bins, n_units)

    # Population rate: average across neurons per bin
    pop_rate = all_binned.mean(axis=2)  # (n_trials, n_bins)

    # Autocorrelation per trial, then average
    autocorrs = []
    for trial_rate in pop_rate:
        trial_rate = trial_rate - trial_rate.mean()
        norm = np.sum(trial_rate ** 2)
        if norm < 1e-10:
            continue
        ac = np.correlate(trial_rate, trial_rate, mode='full')
        ac = ac[len(ac) // 2:]  # positive lags only
        ac = ac / norm
        autocorrs.append(ac[:max_lag_bins])

    if not autocorrs:
        return None

    mean_ac = np.mean(autocorrs, axis=0)

    # Timescale: lag at which autocorrelation drops to 1/e
    threshold = 1.0 / np.e
    timescale_bins = max_lag_bins
    for i in range(1, len(mean_ac)):
        if mean_ac[i] < threshold:
            # Linear interpolation
            if mean_ac[i - 1] > threshold:
                frac = (mean_ac[i - 1] - threshold) / (mean_ac[i - 1] - mean_ac[i])
                timescale_bins = (i - 1) + frac
            else:
                timescale_bins = i
            break

    timescale_ms = float(timescale_bins * bin_size * 1000)
    return {
        "timescale_ms": timescale_ms,
        "autocorr_profile": [float(x) for x in mean_ac[:15]],
    }


@app.function(image=image, cpu=4, memory=32768, timeout=86400, volumes={"/results": volume})
def run_tests():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logger = logging.getLogger("gain")

    import numpy as np
    import pynwb
    from scipy.stats import wilcoxon, spearmanr
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm

    sys.path.insert(0, "/root")
    from bracket_norm_core import compute_bracket_norm

    t0 = time.time()
    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] Recurrent gain reduction tests")

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    volume.reload()

    nwb_files = []
    for ds_dir in sorted(Path(DATA_DIR).iterdir()):
        if ds_dir.is_dir():
            nwb_files.extend(sorted(ds_dir.glob("*.nwb")))

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

            if "response" not in trial_cols or "photo_stim_id" not in trial_cols:
                io.close()
                continue

            raw_choices = trials["response"].data[:]
            if hasattr(raw_choices[0], "decode"):
                raw_choices = np.array([c.decode() for c in raw_choices])
            else:
                raw_choices = np.array([str(c) for c in raw_choices])

            choice_binary = np.full(len(raw_choices), -1, dtype=int)
            is_correct = np.zeros(len(raw_choices), dtype=bool)
            for i, r in enumerate(raw_choices):
                if "L" in r and "Lick" not in r:
                    choice_binary[i] = 0
                elif "R" in r:
                    choice_binary[i] = 1
                if "Hit" in r:
                    is_correct[i] = True

            valid = choice_binary >= 0
            if valid.sum() < MIN_TRIALS:
                io.close()
                continue

            photo_id = np.array([float(p) for p in trials["photo_stim_id"].data[:]])
            start_times = np.asarray(trials["start_time"].data[:])

            spike_times_list = [np.asarray(units["spike_times"][i]) for i in range(n_units)]

            valid_idx = np.where(valid)[0]
            n_valid = len(valid_idx)

            choice_v = choice_binary[valid]
            correct_v = is_correct[valid]
            photo_v = photo_id[valid]
            starts_v = start_times[valid]

            is_ctrl = photo_v == 0
            is_stim = photo_v != 0

            n_ctrl = is_ctrl.sum()
            n_stim = is_stim.sum()

            if n_ctrl < MIN_TRIALS or n_stim < MIN_TRIALS:
                io.close()
                continue
            if len(np.unique(choice_v[is_ctrl])) < 2 or len(np.unique(choice_v[is_stim])) < 2:
                io.close()
                continue

            sess = {"file": nwb_path.name, "n_units": n_units,
                    "n_ctrl": int(n_ctrl), "n_stim": int(n_stim)}

            # ============================================================
            # TEST 1: AUTOCORRELATION TIMESCALE
            # ============================================================
            ac_ctrl = population_autocorrelation(
                spike_times_list, starts_v,
                is_ctrl, n_units)
            ac_stim = population_autocorrelation(
                spike_times_list, starts_v,
                is_stim, n_units)

            if ac_ctrl and ac_stim:
                sess["autocorr_timescale_ctrl_ms"] = ac_ctrl["timescale_ms"]
                sess["autocorr_timescale_stim_ms"] = ac_stim["timescale_ms"]
                sess["autocorr_profile_ctrl"] = ac_ctrl["autocorr_profile"]
                sess["autocorr_profile_stim"] = ac_stim["autocorr_profile"]

            # ============================================================
            # TEST 2: EVIDENCE-SPLIT ACCURACY
            # ============================================================
            # Use stim_present if available, else trial index
            if "stim_present" in trial_cols:
                stim_val = np.asarray(trials["stim_present"].data[:])[valid]
                evidence = stim_val.astype(float)
            else:
                evidence = np.linspace(1, 0, n_valid)

            # Split into high/low evidence (median split)
            med_ev = np.median(evidence)
            high_ev = evidence >= med_ev
            low_ev = evidence < med_ev

            for ev_label, ev_mask in [("high_evidence", high_ev), ("low_evidence", low_ev)]:
                ctrl_ev = is_ctrl & ev_mask
                stim_ev = is_stim & ev_mask
                if ctrl_ev.sum() >= 5 and stim_ev.sum() >= 5:
                    sess[f"accuracy_ctrl_{ev_label}"] = float(correct_v[ctrl_ev].mean())
                    sess[f"accuracy_stim_{ev_label}"] = float(correct_v[stim_ev].mean())
                    sess[f"accuracy_drop_{ev_label}"] = float(
                        correct_v[ctrl_ev].mean() - correct_v[stim_ev].mean())

            # ============================================================
            # TEST 3: TEMPORAL BRACKET_NORM PROFILE
            # ============================================================
            temporal_bn_ctrl = []
            temporal_bn_stim = []

            for t_start, t_end in TIME_BINS_SEC:
                # Build activity for this time window
                activity_win = np.zeros((n_valid, n_units), dtype=np.float32)
                duration = t_end - t_start
                for ti, t_trial in enumerate(starts_v):
                    for ui in range(n_units):
                        st = spike_times_list[ui]
                        activity_win[ti, ui] = np.sum(
                            (st >= t_trial + t_start) & (st < t_trial + t_end)
                        ) / duration

                # Z-score
                m = activity_win.mean(axis=0, keepdims=True)
                s = activity_win.std(axis=0, keepdims=True)
                s[s < 1e-8] = 1.0
                activity_z = (activity_win - m) / s

                ctrl_res = compute_bracket_norm(activity_z[is_ctrl], choice_v[is_ctrl], evidence[is_ctrl])
                stim_res = compute_bracket_norm(activity_z[is_stim], choice_v[is_stim], evidence[is_stim])

                temporal_bn_ctrl.append(ctrl_res["bracket_norm"] if ctrl_res else None)
                temporal_bn_stim.append(stim_res["bracket_norm"] if stim_res else None)

            sess["temporal_bn_ctrl"] = temporal_bn_ctrl
            sess["temporal_bn_stim"] = temporal_bn_stim
            sess["temporal_time_centers"] = [(t[0] + t[1]) / 2 * 1000 for t in TIME_BINS_SEC]

            # ============================================================
            # TEST 4: CHOICE SIGNAL LATENCY
            # ============================================================
            # auROC in each time bin → find peak latency
            auroc_profile_ctrl = []
            auroc_profile_stim = []

            for t_start, t_end in TIME_BINS_SEC:
                activity_win = np.zeros((n_valid, n_units), dtype=np.float32)
                duration = t_end - t_start
                for ti, t_trial in enumerate(starts_v):
                    for ui in range(n_units):
                        st = spike_times_list[ui]
                        activity_win[ti, ui] = np.sum(
                            (st >= t_trial + t_start) & (st < t_trial + t_end)
                        ) / duration

                # Mean auROC across neurons
                for mask, profile in [(is_ctrl, auroc_profile_ctrl), (is_stim, auroc_profile_stim)]:
                    if mask.sum() >= 10:
                        aurocs = []
                        for ui in range(n_units):
                            try:
                                auc = roc_auc_score(choice_v[mask], activity_win[mask, ui])
                                aurocs.append(abs(auc - 0.5) + 0.5)
                            except ValueError:
                                aurocs.append(0.5)
                        profile.append(float(np.mean(aurocs)))
                    else:
                        profile.append(None)

            sess["auroc_profile_ctrl"] = auroc_profile_ctrl
            sess["auroc_profile_stim"] = auroc_profile_stim

            # Find peak latency
            valid_ctrl = [(i, v) for i, v in enumerate(auroc_profile_ctrl) if v is not None]
            valid_stim = [(i, v) for i, v in enumerate(auroc_profile_stim) if v is not None]
            if valid_ctrl and valid_stim:
                peak_ctrl_idx = max(valid_ctrl, key=lambda x: x[1])[0]
                peak_stim_idx = max(valid_stim, key=lambda x: x[1])[0]
                time_centers = [(t[0] + t[1]) / 2 * 1000 for t in TIME_BINS_SEC]
                sess["peak_auroc_latency_ctrl_ms"] = time_centers[peak_ctrl_idx]
                sess["peak_auroc_latency_stim_ms"] = time_centers[peak_stim_idx]
                sess["peak_auroc_value_ctrl"] = max(valid_ctrl, key=lambda x: x[1])[1]
                sess["peak_auroc_value_stim"] = max(valid_stim, key=lambda x: x[1])[1]

            # ============================================================
            # TEST 5: NOISE CORRELATION EIGENSPECTRUM
            # ============================================================
            # Activity in decision window
            activity_dw = np.zeros((n_valid, n_units), dtype=np.float32)
            for ti, t_trial in enumerate(starts_v):
                for ui in range(n_units):
                    st = spike_times_list[ui]
                    activity_dw[ti, ui] = np.sum(
                        (st >= t_trial + DECISION_WINDOW_SEC[0]) &
                        (st < t_trial + DECISION_WINDOW_SEC[1])
                    ) / (DECISION_WINDOW_SEC[1] - DECISION_WINDOW_SEC[0])

            # Noise correlations: residuals after subtracting choice mean
            for label, mask in [("ctrl", is_ctrl), ("stim", is_stim)]:
                act = activity_dw[mask]
                ch = choice_v[mask]
                # Subtract choice-conditioned mean
                residuals = act.copy()
                for c in [0, 1]:
                    c_mask = ch == c
                    if c_mask.sum() > 1:
                        residuals[c_mask] -= act[c_mask].mean(axis=0)

                if residuals.shape[0] > residuals.shape[1]:
                    cov = np.cov(residuals.T)
                    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                    eigvals = eigvals[eigvals > 0]
                    if len(eigvals) > 0:
                        # Normalized eigenspectrum (fraction of total)
                        eigvals_norm = eigvals / eigvals.sum()
                        # Power law exponent (log-log slope)
                        ranks = np.arange(1, len(eigvals_norm) + 1)
                        log_ranks = np.log(ranks[:min(20, len(ranks))])
                        log_eigvals = np.log(eigvals_norm[:min(20, len(eigvals_norm))])
                        if len(log_ranks) >= 3:
                            slope = float(np.polyfit(log_ranks, log_eigvals, 1)[0])
                            sess[f"eigenspectrum_slope_{label}"] = slope
                        # Top-3 eigenvalue fraction
                        sess[f"top3_eigval_frac_{label}"] = float(eigvals_norm[:3].sum())

            sessions.append(sess)
            logger.info(f"  {nwb_path.name[:40]} n={n_units} "
                       f"tau_ctrl={sess.get('autocorr_timescale_ctrl_ms','?')}ms "
                       f"tau_stim={sess.get('autocorr_timescale_stim_ms','?')}ms")
            io.close()

        except Exception as e:
            logger.warning(f"  Error: {nwb_path.name}: {e}")
            import traceback
            traceback.print_exc()
            try:
                io.close()
            except Exception:
                pass

    # ============================================================
    # AGGREGATE
    # ============================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: {len(sessions)} sessions\n")

    agg = {}

    # Test 1: Autocorrelation timescale
    logger.info("TEST 1: AUTOCORRELATION TIMESCALE")
    tau_ctrl = [s["autocorr_timescale_ctrl_ms"] for s in sessions if "autocorr_timescale_ctrl_ms" in s]
    tau_stim = [s["autocorr_timescale_stim_ms"] for s in sessions if "autocorr_timescale_stim_ms" in s]
    if tau_ctrl and tau_stim:
        n_decrease = sum(1 for c, s in zip(tau_ctrl, tau_stim) if s < c)
        logger.info(f"  tau ctrl: {np.mean(tau_ctrl):.1f}ms ± {np.std(tau_ctrl):.1f}")
        logger.info(f"  tau stim: {np.mean(tau_stim):.1f}ms ± {np.std(tau_stim):.1f}")
        logger.info(f"  Stim<ctrl (faster decay): {n_decrease}/{len(tau_ctrl)}")
        if len(tau_ctrl) >= 5:
            W, p = wilcoxon(tau_ctrl, tau_stim, alternative="two-sided")
            logger.info(f"  Wilcoxon: W={W:.1f}, p={p:.6f}")
            agg["test1"] = {
                "tau_ctrl_mean": float(np.mean(tau_ctrl)),
                "tau_stim_mean": float(np.mean(tau_stim)),
                "n_decrease": n_decrease, "n": len(tau_ctrl),
                "wilcoxon_p": float(p),
            }

    # Test 2: Evidence-split accuracy
    logger.info(f"\nTEST 2: EVIDENCE-SPLIT ACCURACY")
    for ev_label in ["high_evidence", "low_evidence"]:
        drops = [s[f"accuracy_drop_{ev_label}"] for s in sessions if f"accuracy_drop_{ev_label}" in s]
        ctrl_acc = [s[f"accuracy_ctrl_{ev_label}"] for s in sessions if f"accuracy_ctrl_{ev_label}" in s]
        stim_acc = [s[f"accuracy_stim_{ev_label}"] for s in sessions if f"accuracy_stim_{ev_label}" in s]
        if drops:
            logger.info(f"  {ev_label}: ctrl={np.mean(ctrl_acc):.3f} stim={np.mean(stim_acc):.3f} "
                       f"drop={np.mean(drops):.3f} ± {np.std(drops):.3f}")

    high_drops = [s["accuracy_drop_high_evidence"] for s in sessions if "accuracy_drop_high_evidence" in s]
    low_drops = [s["accuracy_drop_low_evidence"] for s in sessions if "accuracy_drop_low_evidence" in s]
    if high_drops and low_drops:
        logger.info(f"  Prediction: low_ev drop > high_ev drop")
        logger.info(f"  High ev drop: {np.mean(high_drops):.3f}")
        logger.info(f"  Low ev drop:  {np.mean(low_drops):.3f}")
        if len(high_drops) >= 5:
            W, p = wilcoxon(low_drops, high_drops, alternative="greater")
            logger.info(f"  Wilcoxon (low>high): W={W:.1f}, p={p:.6f}")
            agg["test2"] = {
                "high_ev_drop": float(np.mean(high_drops)),
                "low_ev_drop": float(np.mean(low_drops)),
                "wilcoxon_p": float(p),
            }

    # Test 3: Temporal BN profile
    logger.info(f"\nTEST 3: TEMPORAL BRACKET_NORM PROFILE")
    time_centers = sessions[0]["temporal_time_centers"] if sessions else []
    mean_ctrl_profile = []
    mean_stim_profile = []
    for bi in range(len(time_centers)):
        ctrl_vals = [s["temporal_bn_ctrl"][bi] for s in sessions if s["temporal_bn_ctrl"][bi] is not None]
        stim_vals = [s["temporal_bn_stim"][bi] for s in sessions if s["temporal_bn_stim"][bi] is not None]
        mean_ctrl_profile.append(float(np.mean(ctrl_vals)) if ctrl_vals else None)
        mean_stim_profile.append(float(np.mean(stim_vals)) if stim_vals else None)

    # Find peak times
    valid_ctrl_prof = [(i, v) for i, v in enumerate(mean_ctrl_profile) if v is not None]
    valid_stim_prof = [(i, v) for i, v in enumerate(mean_stim_profile) if v is not None]
    if valid_ctrl_prof and valid_stim_prof:
        peak_ctrl = max(valid_ctrl_prof, key=lambda x: x[1])
        peak_stim = max(valid_stim_prof, key=lambda x: x[1])
        logger.info(f"  Ctrl peak BN: {peak_ctrl[1]:.4f} at {time_centers[peak_ctrl[0]]:.0f}ms")
        logger.info(f"  Stim peak BN: {peak_stim[1]:.4f} at {time_centers[peak_stim[0]]:.0f}ms")
        agg["test3"] = {
            "ctrl_peak_bn": peak_ctrl[1], "ctrl_peak_time_ms": time_centers[peak_ctrl[0]],
            "stim_peak_bn": peak_stim[1], "stim_peak_time_ms": time_centers[peak_stim[0]],
            "profile_ctrl": mean_ctrl_profile,
            "profile_stim": mean_stim_profile,
            "time_centers_ms": time_centers,
        }

    # Print temporal profiles
    for bi in range(0, len(time_centers), 4):
        t = time_centers[bi]
        c = mean_ctrl_profile[bi]
        s = mean_stim_profile[bi]
        if c is not None and s is not None:
            marker = " ***" if s > c * 1.5 else ""
            logger.info(f"  t={t:6.0f}ms: ctrl={c:.4f} stim={s:.4f} ratio={s/c:.2f}{marker}")

    # Test 4: Choice signal latency
    logger.info(f"\nTEST 4: CHOICE SIGNAL LATENCY")
    peak_ctrl_lats = [s["peak_auroc_latency_ctrl_ms"] for s in sessions if "peak_auroc_latency_ctrl_ms" in s]
    peak_stim_lats = [s["peak_auroc_latency_stim_ms"] for s in sessions if "peak_auroc_latency_stim_ms" in s]
    if peak_ctrl_lats and peak_stim_lats:
        n_earlier = sum(1 for c, s in zip(peak_ctrl_lats, peak_stim_lats) if s < c)
        logger.info(f"  Peak auROC latency ctrl: {np.mean(peak_ctrl_lats):.0f}ms ± {np.std(peak_ctrl_lats):.0f}")
        logger.info(f"  Peak auROC latency stim: {np.mean(peak_stim_lats):.0f}ms ± {np.std(peak_stim_lats):.0f}")
        logger.info(f"  Stim earlier: {n_earlier}/{len(peak_ctrl_lats)}")
        agg["test4"] = {
            "latency_ctrl_ms": float(np.mean(peak_ctrl_lats)),
            "latency_stim_ms": float(np.mean(peak_stim_lats)),
            "n_earlier": n_earlier, "n": len(peak_ctrl_lats),
        }

    # Test 5: Eigenspectrum
    logger.info(f"\nTEST 5: NOISE CORRELATION EIGENSPECTRUM")
    slope_ctrl = [s["eigenspectrum_slope_ctrl"] for s in sessions if "eigenspectrum_slope_ctrl" in s]
    slope_stim = [s["eigenspectrum_slope_stim"] for s in sessions if "eigenspectrum_slope_stim" in s]
    top3_ctrl = [s["top3_eigval_frac_ctrl"] for s in sessions if "top3_eigval_frac_ctrl" in s]
    top3_stim = [s["top3_eigval_frac_stim"] for s in sessions if "top3_eigval_frac_stim" in s]
    if slope_ctrl and slope_stim:
        logger.info(f"  Eigenspectrum slope ctrl: {np.mean(slope_ctrl):.3f} ± {np.std(slope_ctrl):.3f}")
        logger.info(f"  Eigenspectrum slope stim: {np.mean(slope_stim):.3f} ± {np.std(slope_stim):.3f}")
        logger.info(f"  (More negative = steeper = more low-rank)")
    if top3_ctrl and top3_stim:
        logger.info(f"  Top-3 eigval frac ctrl: {np.mean(top3_ctrl):.3f}")
        logger.info(f"  Top-3 eigval frac stim: {np.mean(top3_stim):.3f}")
        n_more_lowrank = sum(1 for c, s in zip(top3_ctrl, top3_stim) if s > c)
        logger.info(f"  Stim more low-rank: {n_more_lowrank}/{len(top3_ctrl)}")
        agg["test5"] = {
            "slope_ctrl": float(np.mean(slope_ctrl)),
            "slope_stim": float(np.mean(slope_stim)),
            "top3_ctrl": float(np.mean(top3_ctrl)),
            "top3_stim": float(np.mean(top3_stim)),
            "n_more_lowrank": n_more_lowrank,
        }

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_sessions": len(sessions),
        "aggregate": agg,
        "sessions": sessions,
        "total_time_sec": time.time() - t0,
    }

    with open(out_dir / "gain_tests_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    volume.commit()

    logger.info(f"\nDone in {time.time() - t0:.0f}s")


@app.local_entrypoint()
def main():
    run_tests.remote()
    print("Gain reduction tests submitted. Results at /results/svoboda_gain_tests/")
