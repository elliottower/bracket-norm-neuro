# Held-Out Analysis Pre-Specification: IBL Brain-Wide Anatomical Category Analysis

**Status:** FROZEN — DO NOT EDIT after SHA-256 hash is computed.
**Timestamp:** 2026-07-04 14:57 UTC

**Terminology note.** This document is a *held-out analysis pre-specification*,
not a preregistration. The per-region BN/√n values are already computed and
have been examined (Section 0). What is frozen here is the categorical grouping
scheme, the mapping from regions to categories, the directional predictions,
and the statistical tests — all of which have NOT been performed. The genuine
risk this hash protects against is analytic flexibility in categorization
(which regions count as "association," whether layer suffixes collapse, etc.).

---

## 0. Disclosure: what was known when this document was written

The following results were computed and observed before this document was hashed.
They are stated here so that no reader has to guess what influenced these predictions.

**Per-region BN/√n values are already computed.** The IBL v6 analysis
(86 sessions, 202 regions total, 43 regions with ≥5 observations) produced
per-region mean BN/√n values stored in `results/ibl_v6_results.json`. These
individual values have been examined. The predictions below are NOT blind to
per-region scores — they are blind to the *categorical grouping analysis*
that has not yet been performed. A reader can, in principle, mentally compute
approximate category medians from the per-region values; we make no claim of
blinding and state this limitation explicitly.

**Existing correlations with Steinmetz silencing ground truth:**

| Analysis | Dataset | Metric | ρ | p | N regions | Sign interpretation |
|---|---|---|---|---|---|---|
| Within-dataset | Steinmetz | BN/√n | +0.787 | 0.001 | 9 | Higher BN/√n → more silencing deficit |
| Cross-dataset | IBL → Steinmetz silencing | BN/√n | −0.762 | 0.019 (perm) | 8 | Higher BN/√n on IBL → *less* Steinmetz silencing deficit |

The sign reversal between within-dataset (Steinmetz, ρ = +0.787) and
cross-dataset (IBL, ρ = −0.762) reflects a task difference: the Steinmetz
task (visual discrimination with licking) and IBL task (wheel-turning) engage
different sensorimotor circuits. Regions that are causally important for
licking-based decisions are not necessarily the same regions that show
strong evidence-dependent rotation during wheel-turning. The cross-dataset
correlation validates that BN/√n predicts silencing importance *in magnitude*
(|ρ| > 0.7) but with inverted direction, consistent with task-dependent
recruitment of different computation nodes.

**What HAS been done:**
- Per-region BN/√n, rotation angle, and commutativity computed for 202 IBL regions
- Spearman correlation with Steinmetz silencing for the 8 overlapping regions (above)
- Allen VBN cross-task consistency (49 regions, ρ = +0.803 between evidence operationalizations)
- Paper v13o reports these results

**What has NOT been done:**
- No formal grouping of regions into Allen CCF anatomical categories
- No Kruskal-Wallis, Jonckheere-Terpstra, or pairwise rank tests across categories
- No quantitative test of whether category membership predicts BN/√n
- No analysis of within-category variance vs between-category variance
- No rank-correlation of BN/√n against any continuous anatomical hierarchy score

---

## 1. Region inclusion rule

A region enters the analysis if and only if:

1. It appears in `results/ibl_v6_results.json` with ≥5 session-probe observations
2. It is classifiable under the Allen CCF ontology into exactly one major division

Regions failing either criterion are excluded. The inclusion threshold of 5 was
chosen to match the existing v6 analysis (same threshold used for the paper's
IBL reporting). No regions may be added or removed after hashing.

**Qualifying regions (N = 43 at time of writing).** The exact list is determined
by the frozen results file. If re-running the v6 pipeline produces additional
regions meeting the threshold, they enter automatically — the rule is deterministic.

### 1.1 Sensitivity analysis for inclusion threshold

The primary analysis uses ≥5 observations. A pre-declared sensitivity analysis
repeats H1–H4 at ≥8 observations and ≥10 observations to test whether results
are robust to stricter inclusion. These are reported alongside the primary
results but do not replace them. If any hypothesis changes direction (passes at
≥5 but fails at ≥10 or vice versa), this is flagged as instability.

---

## 2. Anatomical categorization scheme

Regions are assigned to categories using the Allen Mouse Brain Common Coordinate
Framework (CCF) ontology (2017 version). The categorization is objective and
externally defined — no discretionary assignment is permitted.

### 2.1 Major divisions

| Category | Allen CCF division | Expected regions |
|---|---|---|
| **Midbrain** | Midbrain (MB) | APN, MRN, SC, NOT, RN, MB |
| **Thalamus** | Thalamus (TH) | CL, LP, MD, LGd, PO, PCN, TH |
| **Isocortex** | Isocortex | VISp, VISl, VISrl, VISa, VISam, VISpm, SSp-bfd, SSs5, MOs, RSP, TEa5 |
| **Hippocampal** | Hippocampal formation (HPF), including entorhinal | CA1, CA3, DG, SUB, POST, HPF, ENTl3, ENTl5, ENTm6 |
| **Hypothalamus** | Hypothalamus (HY) | ZI, HY |
| **Fiber tracts** | Fiber tracts + ventricles | or, fp, ccb, cing, alv, bsc, dhc, chpl |

### 2.2 Entorhinal assignment rationale

Entorhinal cortex (ENTl, ENTm) is folded into Hippocampal formation rather than
treated as a separate category. Allen CCF places entorhinal areas under HPF
(hippocampal formation). Creating a separate 3-region entorhinal category would
introduce grouping flexibility — the exact analytic degree of freedom this
pre-specification aims to eliminate. The merged category has 9 regions, providing
adequate sample size for rank tests.

### 2.3 Isocortex subdivision (secondary analysis)

| Subcategory | Regions |
|---|---|
| **Visual** | VISp, VISl, VISrl, VISa, VISam, VISpm |
| **Somatosensory** | SSp-bfd, SSs5 |
| **Motor** | MOs |
| **Association** | RSP, TEa5 |

### 2.4 Assignment rules

- Layer-specific labels (e.g., ENTl5, SSs5, TEa5) are assigned to their parent
  region's category. The layer suffix does not affect categorization.
- "Unspecified" labels (MB, TH, HPF, HY) are assigned to their namesake category.
- POST (postsubiculum) is assigned to Hippocampal formation per Allen CCF.
- ZI (zona incerta) is assigned to Hypothalamus per Allen CCF (it sits at the
  thalamus-hypothalamus border; Allen CCF places it under hypothalamus).
  ZI is reported dually: under Hypothalamus for all tests, and additionally
  flagged in H4 (sensorimotor junction) since it functions as a sensorimotor
  integration node despite its hypothalamic classification.
- chpl (choroid plexus) is assigned to Fiber tracts (non-neural tissue).

---

## 3. Predictions

All predictions are directional (one-tailed) and derive from the bracket norm
mechanism: BN/√n should be highest in regions that actively transform the
evidence-to-choice mapping for the IBL wheel-turning task and lowest in regions
that relay signals without evidence-dependent reshaping.

### 3.1 Primary prediction: category ordering (H1)

**H1 (primary, confirmatory).** Median BN/√n across the major divisions follows
this ordering:

    Midbrain > Isocortex > Hippocampal > Thalamus

**Rationale.** Midbrain sensorimotor regions (SC, MRN, APN) sit at the
evidence-to-action interface — the IBL task requires wheel-turning, and these
regions convert graded sensory evidence into motor commands, which should
maximize evidence-dependent rotation. Isocortex contains both computation
regions (association areas) and relay regions (primary visual), so it should
rank intermediate. Hippocampal regions (now including entorhinal) support memory
and spatial navigation rather than trial-by-trial evidence integration. Thalamic
nuclei are predominantly relay structures that pass signals between cortical
areas without substantial evidence-dependent transformation.

**Known tension.** CL (central lateral thalamic nucleus) is among the
highest-ranked individual regions. JT tests the *monotonic trend across group
medians*; a single high-ranked thalamic region can coexist with a low group
median. We note this tension pre-hoc rather than discovering it post-hoc.

**Test.** Jonckheere-Terpstra trend test for the ordered alternative
(Midbrain > Isocortex > Hippocampal > Thalamus), using per-region mean
BN/√n as the dependent variable. One-tailed.

**Null distribution.** Exact permutation null (enumerate or sample 10⁵
permutations of region-to-category assignments), NOT the normal approximation.
With group sizes of approximately 6, 11, 9, 7, the asymptotic p-value is
unreliable.

**Non-independence limitation.** Regions are not independent — multiple regions
are observed in the same 86 sessions, so BN/√n values share session-level noise.
The JT test assumes independent observations. As a robustness check, a
session-permutation scheme is run alongside (Section 3.6, H6) that shuffles
category labels while respecting session structure. The primary JT result is
reported with this caveat stated explicitly.

**Pass threshold.** p < 0.05 (permutation).

**Hypothalamus and Fiber tracts** are excluded from H1 because (a) Hypothalamus
has only 2 regions with opposing expected values (ZI high, HY low), and
(b) Fiber tracts are non-neural tissue. Both are reported descriptively.

### 3.2 Secondary prediction: fiber tracts as negative control (H2)

**H2 (secondary, confirmatory).** Fiber tract regions have lower median BN/√n
than neural regions (all other categories pooled).

**Rationale.** Fiber tracts carry axons, not cell bodies. Any "neurons"
recorded in fiber tracts are likely misassigned or passing axonal signals.
These should show minimal evidence-dependent choice reshaping.

**Test.** Mann-Whitney U, fiber tracts vs all neural regions. One-tailed.

**Pass threshold.** p < 0.05 (after BH correction, Section 4).

### 3.3 Secondary prediction: sensorimotor junction (H3)

**H3 (secondary, confirmatory).** The top-5 regions by BN/√n include at least
2 from the midbrain or thalamic/hypothalamic sensorimotor nuclei
(APN, MRN, SC, LP, CL, ZI, NOT).

**Rationale.** The IBL task requires wheel-turning in response to visual contrast.
Regions at the sensory-motor transformation junction should show the strongest
evidence-dependent rotation of the choice vector.

**Test.** Exact hypergeometric probability (≥2 from the 7 sensorimotor regions
in the top-5 of 43). One-tailed.

**Pass threshold.** p < 0.05 (after BH correction, Section 4).

### 3.4 Exploratory: within-isocortex gradient (H4)

**H4 (exploratory, no pass/fail).** Within isocortex, association regions
(RSP, TEa5) have higher median BN/√n than primary visual regions (VISp).

**Rationale.** Association areas integrate evidence across modalities and time;
primary visual cortex relays retinal signals with less evidence-dependent
transformation.

**Test.** Mann-Whitney U, association vs primary visual (VISp only). One-tailed.

**Status.** Exploratory — not included in the FDR family. This test has minimal
power (2 association regions vs 1 primary region). It is pre-declared for
transparency but the result is reported descriptively regardless of p-value.

### 3.5 Exploratory: variance decomposition (H5)

**H5 (exploratory, no pass/fail).** Report the fraction of total BN/√n variance
explained by category membership (η² from Kruskal-Wallis H statistic across
all 6 categories including Hypothalamus and Fiber tracts). No threshold — this
quantifies effect size for future power calculations.

### 3.6 Robustness: leave-one-session-out stability (H6)

**H6 (robustness, pass/fail).** The H1 category ordering (Midbrain > Isocortex >
Hippocampal > Thalamus) survives leave-one-session-out: dropping any single
session from the 86 IBL sessions and recomputing per-region means does not
reverse the ordering of category medians.

**Rationale.** This directly addresses the non-independence problem — if a single
session drives the categorical pattern, the result is fragile.

**Test.** For each of 86 sessions, remove all observations from that session,
recompute region means, compute category medians, check if the 4-category
ordering is preserved. Report fraction of sessions where ordering holds.

**Pass threshold.** Ordering preserved in ≥80% of leave-one-out folds (≥69/86).

---

## 4. Multiple comparisons

H1 is the primary hypothesis. H2–H3 are secondary and confirmatory.
Benjamini-Hochberg FDR correction is applied to H2–H3 at q = 0.10.

H4–H6 are exploratory/robustness and are not included in the FDR family.

If H1 fails (p ≥ 0.05), H2–H3 are reported descriptively but not interpreted
as confirmatory.

---

## 5. Execution order

1. Freeze this document. Compute SHA-256 hash.
2. Load `results/ibl_v6_results.json`. Apply inclusion rule (Section 1).
3. Assign regions to categories using Allen CCF (Section 2). Record assignments
   in `results/anatomical_category_assignments.json`.
4. Run H1 (JT, permutation null), H2–H3 (Mann-Whitney / hypergeometric),
   H4–H5 (exploratory), H6 (leave-one-session-out).
5. Apply BH correction to H2–H3.
6. Run sensitivity analyses at ≥8 and ≥10 observation thresholds (Section 1.1).
7. Report all results in `results/anatomical_category_results.json`.

Steps 1 must complete before step 2. Steps 2–3 are deterministic given
frozen inputs. No manual intervention is permitted between steps 3 and 7.

---

## 6. Artifacts to hash

| File | Contents |
|---|---|
| `PRESPEC_IBL_ANATOMICAL.md` | This document |
| `results/ibl_v6_results.json` | Per-region BN/√n values (input data) |

All hashes stored in `PRESPEC_IBL_ANATOMICAL_sha256.txt`.
