# Bracket Norm Identifies Causally Important Brain Regions From Population Geometry

Code and results for Tower (2026).

## Structure

```
paper/              LaTeX source, bibliography, compiled PDF
figures/            Figure generation scripts and compiled PDFs
experiments/        Steinmetz-based analyses (CPU, run locally)
experiments/crossval/  Cross-dataset validation (IBL, ECoG, Svoboda; requires GPU/Modal)
results/            Pre-computed JSON results for all tables and appendices
data/               Data loaders (auto-download from public archives)
geometry/           Subspace and distance utilities
tests/              Unit tests
```

## Datasets

All datasets are publicly available and downloaded automatically by the data loaders:

- **Steinmetz Neuropixels** (DANDI:000028) -- 39 sessions, 42 brain regions
- **IBL Brain-Wide Map** -- 8 regions with silencing data
- **Allen Visual Behavior Neuropixels** -- 20 sessions, 49 regions
- **Svoboda ALM photoinhibition** (DANDI:000007/000009) -- 32 paired sessions
- **Human ECoG speech production** (DANDI:000019)

## Requirements

```bash
pip install -r requirements.txt
```

For cross-dataset validation scripts (Modal/GPU), additional dependencies are installed
inside the Modal container image. See individual scripts for details.

## Reproducing figures

All figure scripts run from the repo root:

```bash
python figures/fig_quiver.py                # Figure 1: vector field visualization
python figures/fig_ranking_scatter.py       # Figure 2: BN/sqrt(n) vs silencing effect
python figures/fig_photostim_slopegraph.py  # Figure 3: ALM control vs photostim
```

## Reproducing tables

### Tables composed from survey data (Tables 1-3)

Tables 1-3 are derived from the Steinmetz survey analysis. The per-region metrics
are computed across multiple experiment scripts; final table values are composed
manually in the paper LaTeX.

- **Table 1** (confound correlations): 19 geometric metrics vs silencing. Per-metric
  computations are spread across the experiment scripts; the table is assembled in
  the paper.
- **Table 2** (sign-reversal): Sign-flip metrics from the survey. Underlying data
  computed by `experiments/exp_c_alternative_predictions.py`.
- **Table 3** (subsampling invariance): Part of the initial characterization; no
  standalone script.

### Steinmetz analyses (CPU, no GPU needed)

These scripts read cached Steinmetz data (auto-downloaded on first run) and write
JSON results to `results/`.

| Script | Paper table(s) |
|--------|---------------|
| `experiments/exp_c_bracket_jackknife.py` | Table 4 (region rankings) |
| `experiments/exp_c17_shuffle_window.py` | Table 9 (window robustness) |
| `experiments/exp_c_joint_predictor.py` | Table 10 (joint OLS) |
| `experiments/exp_c_allen_vbn_bracket_norm_v2.py` | Tables 8, 11 (Allen VBN + per-region detail) |
| `experiments/exp_c_alternative_predictions.py` | Table 2 data (sign-flip metrics) |
| `experiments/exp_c_internal_validity.py` | Table 3 data (invariance) |
| `experiments/exp_c9_c10_c11_robustness.py` | Appendix (robustness checks) |
| `experiments/exp_c_bn_baseline_separation.py` | Appendix (baseline separation) |
| `experiments/exp_c_fake_metrics_null.py` | Appendix (null model) |
| `experiments/exp_c_mimic_mechanism.py` | Appendix (mechanism discriminator) |
| `experiments/exp_c_causal_taxonomy.py` | Appendix (causal taxonomy) |
| `experiments/exp_c_cross_session_stability.py` | Appendix (cross-session stability) |
| `experiments/exp_c_graded_response.py` | Appendix (graded photostim response) |

### Svoboda ALM photoinhibition (Table 5)

Table 5 can be reproduced locally (no Modal) using the standalone script:

```bash
pip install pynwb dandi h5py
python experiments/exp_svoboda_bracket_norm.py
```

This downloads NWB files from DANDI:000007/000009 and computes bracket norm for
control vs photostimulation conditions. Alternatively, the Modal versions in
`experiments/crossval/modal_svoboda_*.py` produce the same results.

### Cross-dataset validation (GPU / Modal)

IBL and ECoG results require GPU access via Modal. Pre-computed results are
included in `results/` so the paper can be verified without re-running.

| Script | Paper table(s) |
|--------|---------------|
| `experiments/crossval/modal_ecog_bracket_norm.py` | Table 6 (human ECoG) |
| `experiments/crossval/modal_ibl_bracket_norm_v6.py` | Tables 7, 12 (IBL + per-region detail) |

## Tests

```bash
pip install pytest
pytest tests/
```

## Detailed reproduction

See [REPRODUCE.md](REPRODUCE.md) for a complete mapping from every paper table and
figure to scripts, pre-computed results, dependencies, and expected runtimes.

## License

MIT
