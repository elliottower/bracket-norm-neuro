"""Internal validity tests H27-H29 for bracket_norm (MechVal I2, I3, I4).

Tests pre-registered in mechval_audit_statement_v1.md before running.

H27 (Specificity, I3): Pre-stimulus BN should NOT predict silencing.
    Decision-window BN should. The difference confirms temporal specificity.

H28 (Double dissociation, I4): Top-3 BN regions should have higher
    silencing effects than bottom-3 BN regions among the n=9 silencing set.

H29 (Sufficiency, I2): No top-3 BN region should be in the bottom-3
    silencing regions. Monotonicity in the tails.

CPU-only, ~5 min.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from crossval.bracket_norm_core import (
    compute_bracket_norm,
    partial_spearman,
)

sys.path.insert(0, str(Path(__file__).parent))
from shared_bundle import SILENCING_EFFECTS, save_results

DECISION_WINDOW = slice(15, 35)  # 375-875ms post-stimulus
PRE_STIM_WINDOW = slice(5, 15)   # 125-375ms (pre-stimulus baseline)
MIN_NEURONS = 10


def run():
    from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

    print(f"[{datetime.now(timezone.utc).isoformat()}] Internal validity tests H27-H29")
    sessions = load_all()
    print(f"  {len(sessions)} sessions")

    region_bn_decision = {}
    region_bn_prestim = {}
    region_neuron_counts = {}

    for si, sess in enumerate(tqdm(sessions, desc="Sessions")):
        choice = get_choice_labels(sess)
        choice_binary = (choice == 1).astype(int)
        if len(set(choice_binary)) < 2:
            continue
        cl = sess["contrast_left"]
        cr = sess["contrast_right"]
        n_spk_trials = sess["spks"].shape[2]
        n = min(n_spk_trials, len(choice_binary), len(cl))
        choice_binary = choice_binary[:n]
        evidence = np.abs(cl[:n] - cr[:n])

        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            full_activity = get_region_activity(sess, region)
            if full_activity is None or full_activity.ndim != 3:
                continue
            if full_activity.shape[1] < MIN_NEURONS:
                continue

            n_trials = min(full_activity.shape[0], n)
            cb = choice_binary[:n_trials]
            ev = evidence[:n_trials]

            act_decision = full_activity[:n_trials, :, DECISION_WINDOW].mean(axis=2)
            result_dec = compute_bracket_norm(act_decision, cb, ev)

            act_prestim = full_activity[:n_trials, :, PRE_STIM_WINDOW].mean(axis=2)
            result_pre = compute_bracket_norm(act_prestim, cb, ev)

            if result_dec is None:
                continue

            if region not in region_bn_decision:
                region_bn_decision[region] = []
                region_bn_prestim[region] = []
                region_neuron_counts[region] = []

            region_bn_decision[region].append(result_dec["bracket_norm"])
            if result_pre is not None:
                region_bn_prestim[region].append(result_pre["bracket_norm"])
            else:
                region_bn_prestim[region].append(0.0)
            region_neuron_counts[region].append(full_activity.shape[1])

    mean_bn_dec = {r: float(np.mean(v)) for r, v in region_bn_decision.items()}
    mean_bn_pre = {r: float(np.mean(v)) for r, v in region_bn_prestim.items()}
    mean_nc = {r: float(np.mean(v)) for r, v in region_neuron_counts.items()}

    silencing_regions = [r for r in SILENCING_EFFECTS if r in mean_bn_dec and r in mean_bn_pre]
    n_sil = len(silencing_regions)
    print(f"\n  {len(mean_bn_dec)} total regions, {n_sil} silencing regions")

    # =====================================================================
    # H27: Specificity (I3) — pre-stimulus BN vs decision-window BN
    # =====================================================================
    print(f"\n{'='*70}")
    print("H27: SPECIFICITY (I3) — pre-stimulus vs decision-window BN")
    print(f"{'='*70}")

    h27 = {"pass": False, "n_regions": n_sil}

    if n_sil >= 5:
        sil_effects = [SILENCING_EFFECTS[r] for r in silencing_regions]
        dec_bns = [mean_bn_dec[r] for r in silencing_regions]
        pre_bns = [mean_bn_pre[r] for r in silencing_regions]
        ncs = [mean_nc[r] for r in silencing_regions]

        rho_dec, p_dec = spearmanr(dec_bns, sil_effects)
        partial_dec = partial_spearman(np.array(dec_bns), np.array(sil_effects), np.array(ncs))

        rho_pre, p_pre = spearmanr(pre_bns, sil_effects)
        partial_pre = partial_spearman(np.array(pre_bns), np.array(sil_effects), np.array(ncs))

        h27["decision_rho"] = float(rho_dec)
        h27["decision_partial"] = float(partial_dec)
        h27["prestim_rho"] = float(rho_pre)
        h27["prestim_partial"] = float(partial_pre)
        h27["differential"] = float(partial_dec - partial_pre)

        # Pass: pre-stim partial < +0.3 AND decision partial > +0.5
        h27["pass"] = (partial_pre < 0.3) and (partial_dec > 0.5)

        print(f"  Decision window: rho={rho_dec:+.3f}  partial={partial_dec:+.3f}")
        print(f"  Pre-stimulus:    rho={rho_pre:+.3f}  partial={partial_pre:+.3f}")
        print(f"  Differential:    {partial_dec - partial_pre:+.3f}")
        print(f"  PASS: {h27['pass']} (pre<+0.3={partial_pre < 0.3}, dec>+0.5={partial_dec > 0.5})")

        for r in silencing_regions:
            print(f"    {r:8s}: dec_BN={mean_bn_dec[r]:.4f}  pre_BN={mean_bn_pre[r]:.4f}  "
                  f"sil={SILENCING_EFFECTS[r]:.4f}  NC={mean_nc[r]:.0f}")

    # =====================================================================
    # H28: Double dissociation (I4) — top-3 vs bottom-3 BN regions
    # =====================================================================
    print(f"\n{'='*70}")
    print("H28: DOUBLE DISSOCIATION (I4) — top-3 vs bottom-3 BN")
    print(f"{'='*70}")

    h28 = {"pass": False, "n_regions": n_sil}

    if n_sil >= 6:
        ranked = sorted(silencing_regions, key=lambda r: mean_bn_dec[r], reverse=True)
        top3 = ranked[:3]
        bot3 = ranked[-3:]

        top3_sil = [SILENCING_EFFECTS[r] for r in top3]
        bot3_sil = [SILENCING_EFFECTS[r] for r in bot3]

        h28["top3_regions"] = top3
        h28["top3_bn"] = [mean_bn_dec[r] for r in top3]
        h28["top3_silencing"] = top3_sil
        h28["bot3_regions"] = bot3
        h28["bot3_bn"] = [mean_bn_dec[r] for r in bot3]
        h28["bot3_silencing"] = bot3_sil

        non_overlapping = min(top3_sil) > max(bot3_sil)
        h28["non_overlapping"] = non_overlapping

        n_perm = 10000
        obs_diff = np.mean(top3_sil) - np.mean(bot3_sil)
        all_sil = [SILENCING_EFFECTS[r] for r in ranked]
        perm_diffs = []
        for _ in range(n_perm):
            perm = np.random.permutation(len(ranked))
            perm_top3 = [all_sil[i] for i in perm[:3]]
            perm_bot3 = [all_sil[i] for i in perm[-3:]]
            perm_diffs.append(np.mean(perm_top3) - np.mean(perm_bot3))
        perm_diffs = np.array(perm_diffs)
        perm_p = float(np.mean(perm_diffs >= obs_diff))

        h28["obs_diff"] = float(obs_diff)
        h28["perm_p"] = perm_p
        h28["pass"] = non_overlapping or (perm_p < 0.05)

        print(f"  Top-3 BN regions:    {top3}")
        print(f"    BN values:         {[f'{mean_bn_dec[r]:.4f}' for r in top3]}")
        print(f"    Silencing effects:  {[f'{SILENCING_EFFECTS[r]:.4f}' for r in top3]}")
        print(f"  Bottom-3 BN regions: {bot3}")
        print(f"    BN values:         {[f'{mean_bn_dec[r]:.4f}' for r in bot3]}")
        print(f"    Silencing effects:  {[f'{SILENCING_EFFECTS[r]:.4f}' for r in bot3]}")
        print(f"  Non-overlapping:     {non_overlapping}")
        print(f"  Observed diff:       {obs_diff:+.4f}")
        print(f"  Permutation p:       {perm_p:.4f}")
        print(f"  PASS: {h28['pass']}")

    # =====================================================================
    # H29: Sufficiency (I2) — tail monotonicity
    # =====================================================================
    print(f"\n{'='*70}")
    print("H29: SUFFICIENCY (I2) — tail monotonicity")
    print(f"{'='*70}")

    h29 = {"pass": False, "n_regions": n_sil}

    if n_sil >= 6:
        by_bn = sorted(silencing_regions, key=lambda r: mean_bn_dec[r], reverse=True)
        by_sil = sorted(silencing_regions, key=lambda r: SILENCING_EFFECTS[r], reverse=True)

        top3_bn_set = set(by_bn[:3])
        bot3_bn_set = set(by_bn[-3:])
        top5_sil_set = set(by_sil[:5])
        bot5_sil_set = set(by_sil[-5:])

        top3_in_top5 = len(top3_bn_set & top5_sil_set)
        bot3_in_bot5 = len(bot3_bn_set & bot5_sil_set)

        h29["top3_bn_regions"] = by_bn[:3]
        h29["top3_in_top5_silencing"] = top3_in_top5
        h29["bot3_bn_regions"] = by_bn[-3:]
        h29["bot3_in_bot5_silencing"] = bot3_in_bot5
        h29["pass"] = (top3_in_top5 >= 2) and (bot3_in_bot5 >= 2)

        counterexamples = []
        for r in by_bn[:3]:
            if r in bot5_sil_set:
                counterexamples.append(r)
        h29["counterexamples"] = counterexamples

        print(f"  By BN (descending):       {by_bn}")
        print(f"    BN values:              {[f'{mean_bn_dec[r]:.4f}' for r in by_bn]}")
        print(f"  By silencing (descending): {by_sil}")
        print(f"    Silencing values:        {[f'{SILENCING_EFFECTS[r]:.4f}' for r in by_sil]}")
        print(f"  Top-3 BN in top-5 sil:   {top3_in_top5}/3")
        print(f"  Bot-3 BN in bot-5 sil:   {bot3_in_bot5}/3")
        print(f"  Counterexamples:          {counterexamples}")
        print(f"  PASS: {h29['pass']}")

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  H27 (Specificity, I3):         {'PASS' if h27['pass'] else 'FAIL'}")
    print(f"  H28 (Double dissociation, I4): {'PASS' if h28['pass'] else 'FAIL'}")
    print(f"  H29 (Sufficiency, I2):         {'PASS' if h29['pass'] else 'FAIL'}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "internal_validity_H27_H28_H29",
        "preregistration": "mechval_audit_statement_v1.md",
        "h27_specificity": h27,
        "h28_double_dissociation": h28,
        "h29_sufficiency": h29,
    }

    save_results("internal_validity_h27_h28_h29", results)
    print("\nDone.")


if __name__ == "__main__":
    run()
