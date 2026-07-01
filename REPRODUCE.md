# Reproduction Guide

Complete mapping from every table and figure in Tower (2026) to the scripts,
pre-computed results, and commands needed to reproduce them.

## Setup

```bash
pip install -r requirements.txt
```

Core requirements: numpy, scipy, matplotlib, tqdm, requests, scikit-learn.

Cross-dataset validation scripts run on Modal and install their own dependencies
inside the container image (pynwb, h5py, dandi, ONE-api, allensdk, etc.).
Pre-computed results are included in `results/` so these can be verified without
re-running.

---

## Figures

### Figure 1 -- Vector field visualization (fig:quiver)

- **Script:** `figures/fig_quiver.py`
- **Output:** `figures/fig_quiver.pdf`
- **Dependencies:** Core requirements only (uses Steinmetz data, auto-downloaded)
- **Runtime:** ~2 min (kernel smoothing over grid)
- **Command:**
  ```bash
  python figures/fig_quiver.py
  ```

### Figure 2 -- BN/sqrt(n) vs silencing scatter (fig:ranking)

- **Script:** `figures/fig_ranking_scatter.py`
- **Output:** `figures/fig_ranking_scatter.pdf`
- **Dependencies:** Core requirements only (hardcoded data from survey)
- **Runtime:** <10 sec
- **Command:**
  ```bash
  python figures/fig_ranking_scatter.py
  ```

### Figure 3 -- ALM control vs photostim slopegraph (fig:photostim)

- **Script:** `figures/fig_photostim_slopegraph.py`
- **Output:** `figures/fig_photostim_slopegraph.pdf`
- **Dependencies:** Core requirements only (reads pre-computed Svoboda results)
- **Pre-computed data:** `results/svoboda_bracket_norm_v5_20260626.json`
- **Runtime:** <10 sec
- **Command:**
  ```bash
  python figures/fig_photostim_slopegraph.py
  ```

---

## Main text tables

### Table 1 -- Confound correlations (tab:confound)

19 geometric metrics vs optogenetic silencing effect, testing whether each metric
predicts causal importance after controlling for neuron count.

- **Script:** No single script. Values are derived from the Steinmetz survey
  analysis and composed manually in the paper LaTeX. The per-metric computations
  are spread across `experiments/exp_c_alternative_predictions.py` and the survey
  characterization scripts.
- **Pre-computed data:** `results/alternative_predictions_20260626_035223.json`
  contains the metric-by-metric partial correlations.
- **Runtime:** N/A (manual composition)

### Table 2 -- Sign-reversal metrics (tab:signflip)

Metrics that reverse sign under subsampling or time-window changes.

- **Script:** Underlying data computed by `experiments/exp_c_alternative_predictions.py`.
  Final table values composed manually in the paper.
- **Pre-computed data:** `results/alternative_predictions_20260626_035223.json`
- **Runtime:** ~10 min (CPU)
- **Command:**
  ```bash
  python experiments/exp_c_alternative_predictions.py
  ```

### Table 3 -- Subsampling invariance (tab:invariance)

Synthetic subsampling test showing bracket norm is invariant to neuron count.

- **Script:** Part of the initial characterization in
  `experiments/exp_c_internal_validity.py`. No separate standalone script.
- **Pre-computed data:** `results/internal_validity_h27_h28_h29_20260626_174805.json`
- **Runtime:** ~5 min (CPU)
- **Command:**
  ```bash
  python experiments/exp_c_internal_validity.py
  ```

### Table 4 -- Region rankings BN vs silencing (tab:tails)

Leave-one-out jackknife and bootstrap CI on the rank correlation between
bracket norm and silencing effect.

- **Script:** `experiments/exp_c_bracket_jackknife.py`
- **Pre-computed data:** `results/artifacts/bracket_jackknife_bracket_jackknife_20260626_024924.json`
- **Runtime:** <5 min (CPU)
- **Command:**
  ```bash
  python experiments/exp_c_bracket_jackknife.py
  ```

### Table 5 -- 12 mechanism tests (tab:mechanism)

Svoboda ALM photostimulation: control vs photostim bracket norm, gain modulation,
attractor dynamics, and theory tests across 32 paired sessions.

- **Standalone script (no Modal needed):**
  - `experiments/exp_svoboda_bracket_norm.py` -- downloads DANDI:000007/000009
    NWB files and computes control vs photostim bracket norm locally.
  - **Dependencies:** `pip install pynwb dandi h5py`
  - **Runtime:** ~30 min (data download + processing)
  - **Command:**
    ```bash
    python experiments/exp_svoboda_bracket_norm.py
    ```
- **Modal alternatives** (produce the same results plus gain/attractor/theory tests):
  - `experiments/crossval/modal_svoboda_bracket_norm_v5.py` -- core BN comparison
  - `experiments/crossval/modal_svoboda_gain_tests.py` -- gain modulation tests
  - `experiments/crossval/modal_svoboda_theory_tests.py` -- theory tests (z-score artifact, subsampling, dose-response)
  - `experiments/crossval/modal_svoboda_attractor_tests.py` -- attractor release tests
- **Pre-computed data:**
  - `results/svoboda_bracket_norm_v5_20260626.json`
  - `results/svoboda_gain_tests_20260626.json`
  - `results/svoboda_theory_tests_20260626.json`
  - `results/svoboda_attractor_tests_20260626.json`

### Table 6 -- Human ECoG (tab:ecog)

Bracket norm applied to human ECoG speech production data (DANDI:000019).

- **Script:** `experiments/crossval/modal_ecog_bracket_norm.py`
- **Pre-computed data:** `results/ecog_bracket_norm_full_results.json`
- **Dependencies:** Modal, pynwb, h5py, dandi (installed in Modal container)
- **Runtime:** ~20 min on Modal (two-step: download then analyze)
- **Commands:**
  ```bash
  # Step 1: download NWB files (run once)
  modal run --detach experiments/crossval/modal_ecog_bracket_norm.py::download_data
  # Step 2: analyze (after download completes)
  modal run --detach experiments/crossval/modal_ecog_bracket_norm.py::analyze
  ```

### Table 7 -- IBL diagnostics (tab:ibl)

IBL Brain-Wide Map replication with rotation angle, commutativity, and BN/sqrt(n).

- **Script:** `experiments/crossval/modal_ibl_bracket_norm_v6.py`
- **Pre-computed data:** `results/ibl_v6_results.json`
- **Dependencies:** Modal, ONE-api, scipy, scikit-learn (installed in Modal container)
- **Runtime:** ~45 min on Modal
- **Command:**
  ```bash
  modal run --detach experiments/crossval/modal_ibl_bracket_norm_v6.py
  ```

### Table 8 -- Allen VBN (tab:allen)

Allen Visual Behavior Neuropixels bracket norm across 49 regions.

- **Script:** `experiments/exp_c_allen_vbn_bracket_norm_v2.py`
- **Pre-computed data:** `results/allen_vbn_bracket_norm_20260626_165627.json`
- **Dependencies:** Core requirements + allensdk (pip install allensdk)
- **Runtime:** ~15 min (CPU, data auto-downloaded)
- **Command:**
  ```bash
  python experiments/exp_c_allen_vbn_bracket_norm_v2.py
  ```

---

## Appendix tables

### Table 9 -- Window robustness (tab:windows)

Shuffle and window-size robustness checks on bracket norm.

- **Script:** `experiments/exp_c17_shuffle_window.py`
- **Pre-computed data:** `results/artifacts/c17_shuffle_exp_c17_shuffle_window_20260626_005024.json`
- **Runtime:** ~5 min (CPU)
- **Command:**
  ```bash
  python experiments/exp_c17_shuffle_window.py
  ```

### Table 10 -- Joint OLS predictor (tab:joint)

Joint OLS regression: bracket norm + neuron count predicting silencing effect.

- **Script:** `experiments/exp_c_joint_predictor.py`
- **Pre-computed data:** `results/joint_predictor_20260626_141137.json`
- **Runtime:** <1 min (CPU)
- **Command:**
  ```bash
  python experiments/exp_c_joint_predictor.py
  ```

### Table 11 -- Allen VBN per-region detail (tab:vbn_regions)

Per-region bracket norm values for all 49 Allen VBN regions.

- **Script:** `experiments/exp_c_allen_vbn_bracket_norm_v2.py` (same script as Table 8)
- **Pre-computed data:** `results/allen_vbn_bracket_norm_20260626_165627.json`
- **Runtime:** ~15 min (CPU)
- **Command:**
  ```bash
  python experiments/exp_c_allen_vbn_bracket_norm_v2.py
  ```

### Table 12 -- IBL per-region detail (tab:ibl_detail)

Per-region bracket norm, rotation angle, and commutativity for all IBL regions.

- **Script:** `experiments/crossval/modal_ibl_bracket_norm_v6.py` (same script as Table 7)
- **Pre-computed data:** `results/ibl_v6_results.json`
- **Runtime:** ~45 min on Modal
- **Command:**
  ```bash
  modal run --detach experiments/crossval/modal_ibl_bracket_norm_v6.py
  ```

---

## Additional appendix experiments

These scripts produce supplementary results referenced in the appendix but not
assigned numbered tables:

| Script | Description | Runtime |
|--------|-------------|---------|
| `experiments/exp_c9_c10_c11_robustness.py` | Robustness checks (curvature, LDA) | ~10 min (CPU) |
| `experiments/exp_c_bn_baseline_separation.py` | Baseline separation analysis | ~5 min (CPU) |
| `experiments/exp_c_fake_metrics_null.py` | Null model permutation test | ~5 min (CPU) |
| `experiments/exp_c_mimic_mechanism.py` | Mechanism discriminator | ~5 min (CPU) |
| `experiments/exp_c_causal_taxonomy.py` | Causal taxonomy classification | ~10 min (CPU) |
| `experiments/exp_c_cross_session_stability.py` | Cross-session stability | ~5 min (CPU) |
| `experiments/exp_c_graded_response.py` | Graded photostim dose-response | ~1 min (CPU) |

Pre-computed results for all of these are in `results/` and `results/artifacts/`.

---

## Notes

- All CPU scripts auto-download data on first run. Steinmetz data comes from
  DANDI:000028 via the data loaders in `data/`.
- Modal scripts install their own dependencies (pynwb, h5py, ONE-api, etc.)
  inside the container image. You do not need these installed locally.
- Pre-computed results in `results/` match the values reported in the paper.
  Re-running scripts will overwrite these with freshly computed results (which
  should match within numerical precision).
- All scripts are designed to run from the repo root directory.
