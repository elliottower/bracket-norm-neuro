"""Modal: Attractor release theory tests on Svoboda photostim data.

Tests:
1. TARGETED dropout — drop the specific neurons suppressed by photostim
   from control trials. Does bracket_norm increase like real photostim?
2. Dimensionality — participation ratio of population covariance,
   control vs stim trials.
3. Trial-to-trial variability — Fano factor and shared variance.
4. Across vs within quartile decomposition of bracket_norm.
5. Per-neuron choice selectivity (auROC), control vs stim.
6. bracket_norm increase vs accuracy decrease correlation across sessions.

Usage:
    modal run --detach experiments_crossval/modal_svoboda_attractor_tests.py
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

app = modal.App("svoboda-attractor-tests")
volume = modal.Volume.from_name("neuro-causal-geometry-results", create_if_missing=True)

DATA_DIR = "/results/svoboda_data"
RESULTS_DIR = "/results/svoboda_attractor_tests"

DECISION_WINDOW_SEC = (0.0, 0.3)
MIN_NEURONS = 10
MIN_TRIALS = 20
N_DROPOUT_REPS = 200


def participation_ratio(activity):
    """Participation ratio of the covariance matrix eigenvalues."""
    import numpy as np
    cov = np.cov(activity.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return 0.0
    return float((eigvals.sum() ** 2) / (eigvals ** 2).sum())


def neuron_auroc(activity, choice):
    """auROC for each neuron's choice selectivity."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    aurocs = []
    for i in range(activity.shape[1]):
        try:
            auc = roc_auc_score(choice, activity[:, i])
            aurocs.append(abs(auc - 0.5) + 0.5)
        except ValueError:
            aurocs.append(0.5)
    return np.array(aurocs)


def decompose_bracket_norm(activity, choice, evidence, n_quartiles=4):
    """Decompose bracket_norm into across-quartile and within-quartile components."""
    import numpy as np
    quartiles = np.percentile(evidence, np.linspace(0, 100, n_quartiles + 1))
    quartile_labels = np.digitize(evidence, quartiles[1:-1])

    displacements = []
    within_vars = []
    for q in range(n_quartiles):
        q_mask = quartile_labels == q
        if q_mask.sum() < 4:
            return None

        q_act = activity[q_mask]
        q_choice = choice[q_mask]

        c0 = q_act[q_choice == 0]
        c1 = q_act[q_choice == 1]
        if len(c0) < 2 or len(c1) < 2:
            return None

        displacement = c1.mean(axis=0) - c0.mean(axis=0)
        displacements.append(displacement)

        within_var = np.mean([np.var(c0, axis=0).mean(), np.var(c1, axis=0).mean()])
        within_vars.append(within_var)

    displacements = np.array(displacements)

    across_var = np.var(displacements, axis=0).mean()
    mean_within_var = np.mean(within_vars)

    bracket_norm = np.linalg.norm(displacements[-1] - displacements[0])

    return {
        "bracket_norm": float(bracket_norm),
        "across_quartile_var": float(across_var),
        "within_quartile_var": float(mean_within_var),
        "ratio_across_within": float(across_var / mean_within_var) if mean_within_var > 0 else float("inf"),
    }


@app.function(image=image, cpu=4, memory=32768, timeout=86400, volumes={"/results": volume})
def run_tests():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logger = logging.getLogger("attractor")

    import numpy as np
    import pynwb
    from scipy.stats import spearmanr, wilcoxon
    from tqdm import tqdm

    sys.path.insert(0, "/root")
    from bracket_norm_core import compute_bracket_norm

    t0 = time.time()
    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] Attractor release theory tests")

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

            activity = np.zeros((n_valid, n_units), dtype=np.float32)
            for ti_local, ti_global in enumerate(valid_idx):
                t_start = start_times[ti_global] + DECISION_WINDOW_SEC[0]
                t_end = start_times[ti_global] + DECISION_WINDOW_SEC[1]
                for ui in range(n_units):
                    st = spike_times_list[ui]
                    activity[ti_local, ui] = np.sum((st >= t_start) & (st < t_end)) / (t_end - t_start)

            # Z-score
            means = activity.mean(axis=0, keepdims=True)
            stds = activity.std(axis=0, keepdims=True)
            stds[stds < 1e-8] = 1.0
            activity_z = (activity - means) / stds

            choice_v = choice_binary[valid]
            correct_v = is_correct[valid]
            photo_v = photo_id[valid]
            evidence = np.linspace(1, 0, n_valid)

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

            # Baseline bracket_norm
            ctrl_bn_res = compute_bracket_norm(activity_z[is_ctrl], choice_v[is_ctrl], evidence[is_ctrl])
            stim_bn_res = compute_bracket_norm(activity_z[is_stim], choice_v[is_stim], evidence[is_stim])
            if not ctrl_bn_res or not stim_bn_res:
                io.close()
                continue

            sess["bn_ctrl"] = ctrl_bn_res["bracket_norm"]
            sess["bn_stim"] = stim_bn_res["bracket_norm"]
            sess["bn_diff"] = stim_bn_res["bracket_norm"] - ctrl_bn_res["bracket_norm"]

            # ============================================================
            # TEST 1: TARGETED vs RANDOM dropout
            # ============================================================
            # Identify suppressed neurons: rate drops >50% during stim
            ctrl_rates = activity[is_ctrl].mean(axis=0)
            stim_rates = activity[is_stim].mean(axis=0)

            rate_ratios = np.zeros(n_units)
            for i in range(n_units):
                if ctrl_rates[i] > 0.5:  # at least 0.5 Hz baseline
                    rate_ratios[i] = stim_rates[i] / ctrl_rates[i]
                else:
                    rate_ratios[i] = 1.0  # don't count silent neurons

            # Suppressed = rate drops to <50% of control
            suppressed_mask = rate_ratios < 0.5
            n_suppressed = suppressed_mask.sum()
            sess["n_suppressed"] = int(n_suppressed)
            sess["suppression_frac"] = float(n_suppressed / n_units)

            if n_suppressed > 0 and (n_units - n_suppressed) >= MIN_NEURONS:
                # TARGETED dropout: remove suppressed neurons from control trials
                keep_mask = ~suppressed_mask
                targeted_act = activity_z[is_ctrl][:, keep_mask]
                # Re-z-score the subpopulation
                m = targeted_act.mean(axis=0, keepdims=True)
                s = targeted_act.std(axis=0, keepdims=True)
                s[s < 1e-8] = 1.0
                targeted_act_z = (targeted_act - m) / s

                targeted_res = compute_bracket_norm(targeted_act_z, choice_v[is_ctrl], evidence[is_ctrl])
                if targeted_res:
                    sess["bn_targeted_dropout"] = targeted_res["bracket_norm"]

                # RANDOM dropout: remove same NUMBER of random neurons (200 reps)
                random_bns = []
                n_keep = n_units - n_suppressed
                for _ in range(N_DROPOUT_REPS):
                    idx = np.random.choice(n_units, n_keep, replace=False)
                    sub = activity_z[is_ctrl][:, idx]
                    m = sub.mean(axis=0, keepdims=True)
                    s = sub.std(axis=0, keepdims=True)
                    s[s < 1e-8] = 1.0
                    sub_z = (sub - m) / s
                    res = compute_bracket_norm(sub_z, choice_v[is_ctrl], evidence[is_ctrl])
                    if res:
                        random_bns.append(res["bracket_norm"])

                if random_bns:
                    sess["bn_random_dropout_mean"] = float(np.mean(random_bns))
                    sess["bn_random_dropout_std"] = float(np.std(random_bns))
                    # Where does targeted dropout fall in the random distribution?
                    if "bn_targeted_dropout" in sess:
                        percentile = float(np.mean(np.array(random_bns) <= sess["bn_targeted_dropout"]) * 100)
                        sess["targeted_vs_random_percentile"] = percentile

            # ============================================================
            # TEST 2: DIMENSIONALITY (participation ratio)
            # ============================================================
            pr_ctrl = participation_ratio(activity_z[is_ctrl])
            pr_stim = participation_ratio(activity_z[is_stim])
            sess["pr_ctrl"] = pr_ctrl
            sess["pr_stim"] = pr_stim
            sess["pr_diff"] = pr_stim - pr_ctrl

            # ============================================================
            # TEST 3: TRIAL-TO-TRIAL VARIABILITY
            # ============================================================
            # Fano factor per neuron, averaged
            ctrl_var = activity[is_ctrl].var(axis=0)
            ctrl_mean_act = activity[is_ctrl].mean(axis=0)
            stim_var = activity[is_stim].var(axis=0)
            stim_mean_act = activity[is_stim].mean(axis=0)

            active_mask = ctrl_mean_act > 0.5
            if active_mask.sum() > 5:
                ff_ctrl = float(np.mean(ctrl_var[active_mask] / ctrl_mean_act[active_mask]))
                ff_stim = float(np.mean(stim_var[active_mask] / stim_mean_act[active_mask]))
                sess["fano_factor_ctrl"] = ff_ctrl
                sess["fano_factor_stim"] = ff_stim

            # Shared variance (top eigenvalue fraction)
            eigvals_ctrl = np.linalg.eigvalsh(np.cov(activity_z[is_ctrl].T))
            eigvals_stim = np.linalg.eigvalsh(np.cov(activity_z[is_stim].T))
            sess["shared_var_ctrl"] = float(eigvals_ctrl[-1] / eigvals_ctrl.sum()) if eigvals_ctrl.sum() > 0 else 0
            sess["shared_var_stim"] = float(eigvals_stim[-1] / eigvals_stim.sum()) if eigvals_stim.sum() > 0 else 0

            # ============================================================
            # TEST 4: ACROSS vs WITHIN quartile decomposition
            # ============================================================
            decomp_ctrl = decompose_bracket_norm(activity_z[is_ctrl], choice_v[is_ctrl], evidence[is_ctrl])
            decomp_stim = decompose_bracket_norm(activity_z[is_stim], choice_v[is_stim], evidence[is_stim])
            if decomp_ctrl:
                sess["decomp_ctrl"] = decomp_ctrl
            if decomp_stim:
                sess["decomp_stim"] = decomp_stim

            # ============================================================
            # TEST 5: PER-NEURON CHOICE SELECTIVITY
            # ============================================================
            auroc_ctrl = neuron_auroc(activity_z[is_ctrl], choice_v[is_ctrl])
            auroc_stim = neuron_auroc(activity_z[is_stim], choice_v[is_stim])
            sess["mean_auroc_ctrl"] = float(np.mean(auroc_ctrl))
            sess["mean_auroc_stim"] = float(np.mean(auroc_stim))
            sess["median_auroc_ctrl"] = float(np.median(auroc_ctrl))
            sess["median_auroc_stim"] = float(np.median(auroc_stim))

            # ============================================================
            # TEST 6: ACCURACY
            # ============================================================
            acc_ctrl = float(correct_v[is_ctrl].mean())
            acc_stim = float(correct_v[is_stim].mean())
            sess["accuracy_ctrl"] = acc_ctrl
            sess["accuracy_stim"] = acc_stim
            sess["accuracy_diff"] = acc_ctrl - acc_stim

            sessions.append(sess)
            logger.info(f"  {nwb_path.name[:40]} n={n_units} "
                       f"BN:ctrl={sess['bn_ctrl']:.3f}/stim={sess['bn_stim']:.3f} "
                       f"PR:ctrl={pr_ctrl:.1f}/stim={pr_stim:.1f} "
                       f"supp={sess['suppression_frac']:.0%}")
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

    # Test 1: Targeted vs random dropout
    logger.info("TEST 1: TARGETED vs RANDOM DROPOUT")
    targeted = [s for s in sessions if "bn_targeted_dropout" in s and "bn_random_dropout_mean" in s]
    if targeted:
        for s in targeted[:10]:
            logger.info(f"  {s['file'][:35]}: ctrl={s['bn_ctrl']:.3f} "
                       f"targeted={s['bn_targeted_dropout']:.3f} "
                       f"random={s['bn_random_dropout_mean']:.3f}±{s['bn_random_dropout_std']:.3f} "
                       f"stim={s['bn_stim']:.3f} "
                       f"targeted_pctile={s.get('targeted_vs_random_percentile','?'):.0f}%")

        targeted_bns = [s["bn_targeted_dropout"] for s in targeted]
        random_bns = [s["bn_random_dropout_mean"] for s in targeted]
        stim_bns = [s["bn_stim"] for s in targeted]
        ctrl_bns = [s["bn_ctrl"] for s in targeted]

        # Does targeted dropout reproduce the stim effect?
        targeted_reproduces = float(np.mean([(t - c) / (st - c) if st != c else 0
                                             for t, c, st in zip(targeted_bns, ctrl_bns, stim_bns)]))
        logger.info(f"\n  Targeted dropout reproduces {targeted_reproduces:.0%} of the stim effect")
        logger.info(f"  Mean targeted BN: {np.mean(targeted_bns):.4f}")
        logger.info(f"  Mean random BN:   {np.mean(random_bns):.4f}")
        logger.info(f"  Mean stim BN:     {np.mean(stim_bns):.4f}")
        logger.info(f"  Mean ctrl BN:     {np.mean(ctrl_bns):.4f}")

        # Percentile: how extreme is targeted vs random?
        pctiles = [s["targeted_vs_random_percentile"] for s in targeted if "targeted_vs_random_percentile" in s]
        logger.info(f"  Mean percentile of targeted in random dist: {np.mean(pctiles):.1f}%")

        agg["test1"] = {
            "n": len(targeted),
            "ctrl_mean": float(np.mean(ctrl_bns)),
            "targeted_mean": float(np.mean(targeted_bns)),
            "random_mean": float(np.mean(random_bns)),
            "stim_mean": float(np.mean(stim_bns)),
            "reproduces_frac": targeted_reproduces,
            "targeted_percentile_mean": float(np.mean(pctiles)),
        }

    # Test 2: Dimensionality
    logger.info(f"\nTEST 2: DIMENSIONALITY")
    pr_ctrl_all = [s["pr_ctrl"] for s in sessions]
    pr_stim_all = [s["pr_stim"] for s in sessions]
    pr_diffs = [s["pr_diff"] for s in sessions]
    n_pr_increase = sum(1 for d in pr_diffs if d > 0)
    logger.info(f"  PR ctrl: {np.mean(pr_ctrl_all):.2f} ± {np.std(pr_ctrl_all):.2f}")
    logger.info(f"  PR stim: {np.mean(pr_stim_all):.2f} ± {np.std(pr_stim_all):.2f}")
    logger.info(f"  Stim>ctrl: {n_pr_increase}/{len(pr_diffs)}")
    if len(pr_diffs) >= 5:
        W, p = wilcoxon(pr_ctrl_all, pr_stim_all, alternative="two-sided")
        logger.info(f"  Wilcoxon: W={W:.1f}, p={p:.6f}")
        agg["test2"] = {
            "pr_ctrl": float(np.mean(pr_ctrl_all)), "pr_stim": float(np.mean(pr_stim_all)),
            "n_increase": n_pr_increase, "n": len(pr_diffs),
            "wilcoxon_p": float(p),
        }

    # Test 3: Variability
    logger.info(f"\nTEST 3: TRIAL-TO-TRIAL VARIABILITY")
    ff_ctrl = [s["fano_factor_ctrl"] for s in sessions if "fano_factor_ctrl" in s]
    ff_stim = [s["fano_factor_stim"] for s in sessions if "fano_factor_stim" in s]
    if ff_ctrl:
        logger.info(f"  Fano ctrl: {np.mean(ff_ctrl):.3f} ± {np.std(ff_ctrl):.3f}")
        logger.info(f"  Fano stim: {np.mean(ff_stim):.3f} ± {np.std(ff_stim):.3f}")

    sv_ctrl = [s["shared_var_ctrl"] for s in sessions]
    sv_stim = [s["shared_var_stim"] for s in sessions]
    logger.info(f"  Shared var ctrl: {np.mean(sv_ctrl):.4f}")
    logger.info(f"  Shared var stim: {np.mean(sv_stim):.4f}")

    agg["test3"] = {
        "fano_ctrl": float(np.mean(ff_ctrl)) if ff_ctrl else None,
        "fano_stim": float(np.mean(ff_stim)) if ff_stim else None,
        "shared_var_ctrl": float(np.mean(sv_ctrl)),
        "shared_var_stim": float(np.mean(sv_stim)),
    }

    # Test 4: Decomposition
    logger.info(f"\nTEST 4: ACROSS vs WITHIN QUARTILE")
    for label, key in [("ctrl", "decomp_ctrl"), ("stim", "decomp_stim")]:
        across = [s[key]["across_quartile_var"] for s in sessions if key in s]
        within = [s[key]["within_quartile_var"] for s in sessions if key in s]
        ratios = [s[key]["ratio_across_within"] for s in sessions if key in s]
        if across:
            logger.info(f"  {label}: across={np.mean(across):.4f} within={np.mean(within):.4f} "
                       f"ratio={np.mean(ratios):.4f}")
    agg["test4"] = {
        "ctrl_across": float(np.mean([s["decomp_ctrl"]["across_quartile_var"] for s in sessions if "decomp_ctrl" in s])),
        "ctrl_within": float(np.mean([s["decomp_ctrl"]["within_quartile_var"] for s in sessions if "decomp_ctrl" in s])),
        "stim_across": float(np.mean([s["decomp_stim"]["across_quartile_var"] for s in sessions if "decomp_stim" in s])),
        "stim_within": float(np.mean([s["decomp_stim"]["within_quartile_var"] for s in sessions if "decomp_stim" in s])),
    }

    # Test 5: auROC
    logger.info(f"\nTEST 5: PER-NEURON CHOICE SELECTIVITY")
    auroc_ctrl = [s["mean_auroc_ctrl"] for s in sessions]
    auroc_stim = [s["mean_auroc_stim"] for s in sessions]
    logger.info(f"  auROC ctrl: {np.mean(auroc_ctrl):.4f}")
    logger.info(f"  auROC stim: {np.mean(auroc_stim):.4f}")
    n_aur_increase = sum(1 for c, s in zip(auroc_ctrl, auroc_stim) if s > c)
    logger.info(f"  Stim>ctrl: {n_aur_increase}/{len(sessions)}")
    if len(sessions) >= 5:
        W, p = wilcoxon(auroc_ctrl, auroc_stim, alternative="two-sided")
        logger.info(f"  Wilcoxon: W={W:.1f}, p={p:.6f}")
    agg["test5"] = {
        "auroc_ctrl": float(np.mean(auroc_ctrl)), "auroc_stim": float(np.mean(auroc_stim)),
        "n_increase": n_aur_increase, "n": len(sessions),
    }

    # Test 6: BN increase vs accuracy decrease
    logger.info(f"\nTEST 6: BN INCREASE vs ACCURACY DECREASE")
    bn_increases = [s["bn_diff"] for s in sessions]
    acc_decreases = [s["accuracy_diff"] for s in sessions]
    rho, p = spearmanr(bn_increases, acc_decreases)
    logger.info(f"  Spearman(BN_increase, accuracy_decrease): rho={rho:+.3f}, p={p:.4f}")
    logger.info(f"  Mean accuracy ctrl: {np.mean([s['accuracy_ctrl'] for s in sessions]):.3f}")
    logger.info(f"  Mean accuracy stim: {np.mean([s['accuracy_stim'] for s in sessions]):.3f}")
    agg["test6"] = {"spearman_rho": float(rho), "spearman_p": float(p)}

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_sessions": len(sessions),
        "aggregate": agg,
        "sessions": sessions,
        "total_time_sec": time.time() - t0,
    }

    with open(out_dir / "attractor_tests_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    volume.commit()

    logger.info(f"\nDone in {time.time() - t0:.0f}s")


@app.local_entrypoint()
def main():
    run_tests.remote()
    print("Attractor tests submitted. Results at /results/svoboda_attractor_tests/")
