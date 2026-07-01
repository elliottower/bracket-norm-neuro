import numpy as np
import pytest
from scipy.linalg import svd
from scipy.stats import spearmanr

from geometry.distances import (
    cka,
    chordal_distance,
    debiased_cka,
    grassmannian_distance,
    principal_angles,
    subspace_overlap,
)
from geometry.subspace import fit_lda_subspace, fit_pca_subspace
from crossval.bracket_norm_core import (
    aggregate_region_metrics,
    compute_bracket_norm,
    correlate_with_silencing,
    partial_spearman,
)
from shared_bundle import (
    build_coboundary,
    build_scalar_simplicial,
    edge_distance_from_R,
    hodge_decompose_edge,
    procrustes_rotation,
)


# ---------------------------------------------------------------------------
# geometry/distances.py — principal_angles
# ---------------------------------------------------------------------------


class TestPrincipalAngles:
    def test_identical_subspaces_give_zero_angles(self):
        rng = np.random.default_rng()
        A = rng.standard_normal((20, 3))
        U, _, _ = svd(A, full_matrices=False)
        Q = U[:, :3]
        angles = principal_angles(Q, Q)
        assert all(a == pytest.approx(0.0, abs=1e-6) for a in angles)

    def test_orthogonal_subspaces_give_pi_over_2(self):
        U = np.eye(6, 3)
        V = np.eye(6, 3, k=3)
        angles = principal_angles(U, V)
        assert all(a == pytest.approx(np.pi / 2, abs=1e-10) for a in angles)

    def test_known_2d_rotation_gives_rotation_angle(self):
        theta = 0.3
        # Rotating a 2D subspace in R^4 by theta should give one principal
        # angle of theta and one of 0. But since U and V span the SAME 2D
        # subspace (rotation within the subspace), both angles are 0.
        # To get a nonzero principal angle, we need subspaces that differ.
        # Use a single vector rotation instead.
        u = np.array([[1.0], [0.0], [0.0], [0.0]])
        v = np.array([[np.cos(theta)], [np.sin(theta)], [0.0], [0.0]])
        angles = principal_angles(u, v)
        assert len(angles) == 1
        assert angles[0] == pytest.approx(theta, abs=1e-10)

    def test_single_vector_rotation(self):
        theta = 0.7
        u = np.array([[1.0], [0.0], [0.0]])
        v = np.array([[np.cos(theta)], [np.sin(theta)], [0.0]])
        angles = principal_angles(u, v)
        assert len(angles) == 1
        assert angles[0] == pytest.approx(theta, abs=1e-10)

    def test_angles_nonnegative(self):
        rng = np.random.default_rng()
        A = rng.standard_normal((10, 4))
        B = rng.standard_normal((10, 4))
        Ua, _, _ = svd(A, full_matrices=False)
        Ub, _, _ = svd(B, full_matrices=False)
        angles = principal_angles(Ua[:, :4], Ub[:, :4])
        assert all(a >= -1e-10 for a in angles)
        assert all(a <= np.pi / 2 + 1e-10 for a in angles)

    def test_mismatched_dimensions_uses_min(self):
        U = np.eye(8, 2)
        V = np.eye(8, 5, k=0)
        angles = principal_angles(U, V)
        assert len(angles) == 2


# ---------------------------------------------------------------------------
# geometry/distances.py — grassmannian_distance
# ---------------------------------------------------------------------------


class TestGrassmannianDistance:
    def test_same_subspace_gives_zero(self):
        rng = np.random.default_rng()
        A = rng.standard_normal((15, 3))
        U, _, _ = svd(A, full_matrices=False)
        Q = U[:, :3]
        assert grassmannian_distance(Q, Q) == pytest.approx(0.0, abs=1e-6)

    def test_different_subspaces_positive(self):
        U = np.eye(6, 2)
        V = np.eye(6, 2, k=2)
        d = grassmannian_distance(U, V)
        assert d > 0.0

    def test_orthogonal_subspaces_distance(self):
        U = np.eye(6, 2)
        V = np.eye(6, 2, k=2)
        d = grassmannian_distance(U, V)
        assert d == pytest.approx(np.pi / 2 * np.sqrt(2), abs=1e-10)

    def test_triangle_inequality(self):
        rng = np.random.default_rng()
        n, k = 20, 3
        bases = []
        for _ in range(3):
            A = rng.standard_normal((n, k))
            U, _, _ = svd(A, full_matrices=False)
            bases.append(U[:, :k])

        d01 = grassmannian_distance(bases[0], bases[1])
        d12 = grassmannian_distance(bases[1], bases[2])
        d02 = grassmannian_distance(bases[0], bases[2])
        assert d02 <= d01 + d12 + 1e-10

    def test_symmetric(self):
        rng = np.random.default_rng()
        A = rng.standard_normal((10, 3))
        B = rng.standard_normal((10, 3))
        Ua, _, _ = svd(A, full_matrices=False)
        Ub, _, _ = svd(B, full_matrices=False)
        U, V = Ua[:, :3], Ub[:, :3]
        assert grassmannian_distance(U, V) == pytest.approx(
            grassmannian_distance(V, U), abs=1e-10
        )


# ---------------------------------------------------------------------------
# geometry/distances.py — subspace_overlap
# ---------------------------------------------------------------------------


class TestSubspaceOverlap:
    def test_identical_gives_one(self):
        Q = np.eye(6, 3)
        assert subspace_overlap(Q, Q) == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_gives_zero(self):
        U = np.eye(6, 3)
        V = np.eye(6, 3, k=3)
        assert subspace_overlap(U, V) == pytest.approx(0.0, abs=1e-10)

    def test_between_zero_and_one_for_general_subspaces(self):
        rng = np.random.default_rng()
        A = rng.standard_normal((10, 3))
        B = rng.standard_normal((10, 3))
        Ua, _, _ = svd(A, full_matrices=False)
        Ub, _, _ = svd(B, full_matrices=False)
        o = subspace_overlap(Ua[:, :3], Ub[:, :3])
        assert 0.0 <= o <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# geometry/distances.py — cka
# ---------------------------------------------------------------------------


class TestCKA:
    def test_identical_matrices_give_one(self):
        rng = np.random.default_rng()
        X = rng.standard_normal((50, 10))
        assert cka(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self):
        rng = np.random.default_rng()
        X = rng.standard_normal((50, 10))
        Y = rng.standard_normal((50, 8))
        assert cka(X, Y) == pytest.approx(cka(Y, X), abs=1e-10)

    def test_independent_random_matrices_low_cka(self):
        rng = np.random.default_rng()
        X = rng.standard_normal((500, 50))
        Y = rng.standard_normal((500, 50))
        val = cka(X, Y)
        assert val < 0.25

    def test_scaled_copy_gives_one(self):
        rng = np.random.default_rng()
        X = rng.standard_normal((50, 10))
        assert cka(X, X * 3.7) == pytest.approx(1.0, abs=1e-6)

    def test_in_unit_interval(self):
        rng = np.random.default_rng()
        X = rng.standard_normal((40, 10))
        Y = rng.standard_normal((40, 8))
        val = cka(X, Y)
        assert -0.01 <= val <= 1.01


# ---------------------------------------------------------------------------
# geometry/distances.py — debiased_cka
# ---------------------------------------------------------------------------


class TestDebiasedCKA:
    def test_identical_matrices_give_one(self):
        rng = np.random.default_rng()
        X = rng.standard_normal((80, 10))
        assert debiased_cka(X, X) == pytest.approx(1.0, abs=0.05)

    def test_symmetric(self):
        rng = np.random.default_rng()
        X = rng.standard_normal((60, 10))
        Y = rng.standard_normal((60, 8))
        assert debiased_cka(X, Y) == pytest.approx(debiased_cka(Y, X), abs=1e-10)

    def test_too_few_samples_returns_zero(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Y = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        assert debiased_cka(X, Y) == 0.0

    def test_independent_data_near_zero(self):
        rng = np.random.default_rng()
        X = rng.standard_normal((200, 5))
        Y = rng.standard_normal((200, 5))
        val = debiased_cka(X, Y)
        assert abs(val) < 0.15


# ---------------------------------------------------------------------------
# geometry/distances.py — chordal_distance
# ---------------------------------------------------------------------------


class TestChordalDistance:
    def test_same_subspace_gives_zero(self):
        Q = np.eye(8, 3)
        assert chordal_distance(Q, Q) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_subspaces(self):
        U = np.eye(6, 2)
        V = np.eye(6, 2, k=2)
        d = chordal_distance(U, V)
        assert d == pytest.approx(np.sqrt(2), abs=1e-10)

    def test_positive_for_different_subspaces(self):
        rng = np.random.default_rng()
        A = rng.standard_normal((10, 3))
        B = rng.standard_normal((10, 3))
        Ua, _, _ = svd(A, full_matrices=False)
        Ub, _, _ = svd(B, full_matrices=False)
        d = chordal_distance(Ua[:, :3], Ub[:, :3])
        assert d > 0.0

    def test_bounded_by_sqrt_k(self):
        rng = np.random.default_rng()
        k = 4
        n = 12
        A = rng.standard_normal((n, k))
        B = rng.standard_normal((n, k))
        Ua, _, _ = svd(A, full_matrices=False)
        Ub, _, _ = svd(B, full_matrices=False)
        d = chordal_distance(Ua[:, :k], Ub[:, :k])
        assert d <= np.sqrt(k) + 1e-10


# ---------------------------------------------------------------------------
# geometry/subspace.py — fit_pca_subspace
# ---------------------------------------------------------------------------


class TestFitPCASubspace:
    def test_returns_orthonormal_basis(self):
        rng = np.random.default_rng()
        n, p, k = 200, 20, 3
        X = rng.standard_normal((n, p))
        labels = (rng.random(n) > 0.5).astype(int)
        basis = fit_pca_subspace(X, labels, k=k)
        assert basis.shape == (p, k)
        gram = basis.T @ basis
        assert gram == pytest.approx(np.eye(k), abs=1e-10)

    def test_captures_discriminative_direction(self):
        rng = np.random.default_rng()
        n = 500
        p = 10
        labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        X = rng.standard_normal((n, p)) * 0.1
        X[labels == 1, 0] += 3.0
        basis = fit_pca_subspace(X, labels, k=2)
        proj = X @ basis
        mean_diff = proj[labels == 1].mean(axis=0) - proj[labels == 0].mean(axis=0)
        assert np.linalg.norm(mean_diff) > 2.0

    def test_rejects_non_binary_labels(self):
        rng = np.random.default_rng()
        X = rng.standard_normal((100, 10))
        labels = np.zeros(100, dtype=int)
        labels[:33] = 0
        labels[33:66] = 1
        labels[66:] = 2
        with pytest.raises(ValueError, match="binary"):
            fit_pca_subspace(X, labels, k=2)


# ---------------------------------------------------------------------------
# geometry/subspace.py — fit_lda_subspace
# ---------------------------------------------------------------------------


class TestFitLDASubspace:
    def test_returns_orthonormal_basis(self):
        rng = np.random.default_rng()
        n, p, k = 200, 20, 3
        labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        X = rng.standard_normal((n, p))
        X[labels == 1, 0] += 2.0
        basis = fit_lda_subspace(X, labels, k=k)
        assert basis.shape == (p, k)
        gram = basis.T @ basis
        assert gram == pytest.approx(np.eye(k), abs=1e-10)

    def test_lda_separates_better_than_random(self):
        rng = np.random.default_rng()
        n = 600
        p = 15
        labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        X = rng.standard_normal((n, p))
        X[labels == 1, 0] += 2.0
        X[labels == 1, 1] += 1.0

        lda_basis = fit_lda_subspace(X, labels, k=1)
        lda_proj = X @ lda_basis
        lda_sep = abs(
            lda_proj[labels == 1].mean() - lda_proj[labels == 0].mean()
        )

        random_dir = rng.standard_normal((p, 1))
        random_dir /= np.linalg.norm(random_dir)
        random_proj = X @ random_dir
        random_sep = abs(
            random_proj[labels == 1].mean() - random_proj[labels == 0].mean()
        )

        assert lda_sep > random_sep

    def test_high_dimensional_with_pca_prewhitening(self):
        rng = np.random.default_rng()
        n, p, k = 50, 200, 3
        labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        X = rng.standard_normal((n, p))
        X[labels == 1, :5] += 2.0
        basis = fit_lda_subspace(X, labels, k=k)
        assert basis.shape == (p, k)
        gram = basis.T @ basis
        assert gram == pytest.approx(np.eye(k), abs=1e-10)


# ---------------------------------------------------------------------------
# bracket_norm_core.py — compute_bracket_norm
# ---------------------------------------------------------------------------


class TestComputeBracketNorm:
    def test_returns_none_for_too_few_trials(self):
        rng = np.random.default_rng()
        activity = rng.standard_normal((15, 10))
        choice = np.concatenate([np.zeros(5), np.ones(10)]).astype(int)
        evidence = rng.standard_normal(15)
        result = compute_bracket_norm(activity, choice, evidence)
        assert result is None

    def test_returns_positive_bn_for_well_separated_data(self):
        rng = np.random.default_rng()
        n = 400
        p = 10
        # Interleave choice labels so both classes span all evidence levels
        choice = (np.arange(n) % 2).astype(int)
        evidence = rng.uniform(0, 1, n)
        activity = rng.standard_normal((n, p)) * 0.1
        # Make choice displacement depend on evidence level
        activity[choice == 1, 0] += 1.0 + evidence[choice == 1] * 3.0
        activity[choice == 0, 0] -= 1.0

        result = compute_bracket_norm(activity, choice, evidence)
        assert result is not None
        assert result["bracket_norm"] > 0.0

    def test_rotation_angle_in_valid_range(self):
        rng = np.random.default_rng()
        n = 400
        p = 10
        choice = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        evidence = np.linspace(-2, 2, n)
        activity = rng.standard_normal((n, p))
        activity[choice == 1, 0] += evidence[choice == 1]

        result = compute_bracket_norm(activity, choice, evidence)
        if result is not None:
            assert 0.0 <= result["rotation_angle"] <= np.pi + 1e-10

    def test_commutativity_nonnegative(self):
        rng = np.random.default_rng()
        n = 400
        p = 10
        choice = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        evidence = np.linspace(-2, 2, n)
        activity = rng.standard_normal((n, p))
        activity[choice == 1, 0] += evidence[choice == 1] * 2

        result = compute_bracket_norm(activity, choice, evidence)
        if result is not None:
            assert result["commutativity"] >= 0.0

    def test_returns_none_when_quartiles_too_sparse(self):
        rng = np.random.default_rng()
        n = 40
        choice = np.concatenate([np.zeros(20), np.ones(20)]).astype(int)
        evidence = rng.standard_normal(n)
        activity = rng.standard_normal((n, 5))
        result = compute_bracket_norm(activity, choice, evidence, min_per_quartile=50)
        assert result is None


# ---------------------------------------------------------------------------
# bracket_norm_core.py — partial_spearman
# ---------------------------------------------------------------------------


class TestPartialSpearman:
    def test_mediated_relationship_gives_near_zero(self):
        rng = np.random.default_rng()
        n = 5000
        z = rng.standard_normal(n)
        x = z + rng.standard_normal(n) * 0.01
        y = z + rng.standard_normal(n) * 0.01
        ps = partial_spearman(x, y, z)
        assert abs(ps) < 0.1

    def test_direct_relationship_survives_partialing(self):
        rng = np.random.default_rng()
        n = 2000
        z = rng.standard_normal(n)
        x = z + rng.standard_normal(n) * 0.5
        y = 2 * x + rng.standard_normal(n) * 0.3
        ps = partial_spearman(x, y, z)
        assert ps > 0.5

    def test_degenerate_constant_input_returns_finite_or_nan(self):
        x = np.ones(20)
        y = np.arange(20, dtype=float)
        z = np.arange(20, dtype=float)
        ps = partial_spearman(x, y, z)
        # Constant x makes spearmanr undefined; result should be 0 or NaN
        assert ps == 0.0 or np.isnan(ps)


# ---------------------------------------------------------------------------
# bracket_norm_core.py — aggregate_region_metrics
# ---------------------------------------------------------------------------


class TestAggregateRegionMetrics:
    def test_correct_aggregation(self):
        metrics = {
            "V1": [
                {"bracket_norm": 1.0, "rotation_angle": 0.5, "commutativity": 0.2},
                {"bracket_norm": 3.0, "rotation_angle": 1.5, "commutativity": 0.8},
            ],
            "M1": [
                {"bracket_norm": 2.0, "rotation_angle": 1.0, "commutativity": 0.5},
            ],
        }
        summary = aggregate_region_metrics(metrics)
        assert "V1" in summary
        assert "M1" in summary
        assert summary["V1"]["n_sessions"] == 2
        assert summary["V1"]["bracket_norm_mean"] == pytest.approx(2.0)
        assert summary["V1"]["bracket_norm_std"] == pytest.approx(1.0)
        assert summary["M1"]["n_sessions"] == 1
        assert summary["M1"]["bracket_norm_mean"] == pytest.approx(2.0)
        assert summary["M1"]["bracket_norm_std"] == pytest.approx(0.0)

    def test_filters_none_entries(self):
        metrics = {
            "V1": [None, {"bracket_norm": 5.0, "rotation_angle": 0.3, "commutativity": 0.1}],
        }
        summary = aggregate_region_metrics(metrics)
        assert summary["V1"]["n_sessions"] == 1
        assert summary["V1"]["bracket_norm_mean"] == pytest.approx(5.0)

    def test_skips_regions_with_all_none(self):
        metrics = {"V1": [None, None]}
        summary = aggregate_region_metrics(metrics)
        assert "V1" not in summary


# ---------------------------------------------------------------------------
# bracket_norm_core.py — correlate_with_silencing
# ---------------------------------------------------------------------------


class TestCorrelateWithSilencing:
    def test_perfect_correlation(self):
        regions = ["A", "B", "C", "D", "E", "F"]
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        region_summary = {r: {"bn_mean": v} for r, v in zip(regions, vals)}
        silencing = {r: v for r, v in zip(regions, vals)}
        neuron_counts = {r: 100 for r in regions}

        result = correlate_with_silencing(
            region_summary, silencing, neuron_counts, metric_key="bn_mean"
        )
        assert result["rho"] == pytest.approx(1.0, abs=1e-10)
        assert result["n"] == 6

    def test_too_few_regions_returns_error(self):
        region_summary = {"A": {"bn": 1.0}}
        silencing = {"A": 0.5}
        neuron_counts = {"A": 50}
        result = correlate_with_silencing(
            region_summary, silencing, neuron_counts, metric_key="bn"
        )
        assert "error" in result

    def test_partial_included_in_result(self):
        regions = ["A", "B", "C", "D", "E", "F", "G"]
        rng = np.random.default_rng()
        vals = rng.standard_normal(7)
        region_summary = {r: {"x": float(v)} for r, v in zip(regions, vals)}
        silencing = {r: float(v) for r, v in zip(regions, rng.standard_normal(7))}
        neuron_counts = {r: int(c) for r, c in zip(regions, rng.integers(10, 100, 7))}

        result = correlate_with_silencing(
            region_summary, silencing, neuron_counts, metric_key="x"
        )
        assert "partial" in result
        assert isinstance(result["partial"], float)


# ---------------------------------------------------------------------------
# shared_bundle.py — procrustes_rotation
# ---------------------------------------------------------------------------


class TestProcrustesRotation:
    def test_returns_orthogonal_matrix(self):
        rng = np.random.default_rng()
        P = rng.standard_normal((100, 3))
        Q = rng.standard_normal((100, 3))
        R = procrustes_rotation(P, Q)
        assert R.shape == (3, 3)
        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-10)
        assert R @ R.T == pytest.approx(np.eye(3), abs=1e-10)

    def test_aligns_known_rotation(self):
        rng = np.random.default_rng()
        theta = 0.5
        c, s = np.cos(theta), np.sin(theta)
        R_true = np.array([[c, -s], [s, c]])

        P = rng.standard_normal((200, 2))
        Q = P @ R_true.T

        # procrustes_rotation(P, Q) computes U @ Vt from SVD of P.T @ Q
        # For Q = P @ R.T: P.T @ Q = P.T @ P @ R.T, polar factor is R.T
        R_est = procrustes_rotation(P, Q)
        assert R_est == pytest.approx(R_true.T, abs=1e-6)

    def test_identity_when_same_points(self):
        rng = np.random.default_rng()
        P = rng.standard_normal((100, 4))
        R = procrustes_rotation(P, P)
        assert R == pytest.approx(np.eye(4), abs=1e-10)


# ---------------------------------------------------------------------------
# shared_bundle.py — edge_distance_from_R
# ---------------------------------------------------------------------------


class TestEdgeDistanceFromR:
    def test_identity_gives_zero(self):
        R = np.eye(3)
        assert edge_distance_from_R(R) == pytest.approx(0.0, abs=1e-10)

    def test_known_2d_rotation(self):
        theta = 0.8
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        assert edge_distance_from_R(R) == pytest.approx(theta, abs=1e-10)

    def test_negative_det_gives_pi(self):
        R = np.diag([1.0, -1.0])
        assert edge_distance_from_R(R) == pytest.approx(np.pi, abs=1e-10)

    def test_pi_rotation_gives_pi(self):
        R = np.array([[-1.0, 0.0], [0.0, -1.0]])
        assert edge_distance_from_R(R) == pytest.approx(np.pi, abs=1e-10)

    def test_small_rotation_near_zero(self):
        theta = 0.01
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        assert edge_distance_from_R(R) == pytest.approx(theta, abs=1e-6)


# ---------------------------------------------------------------------------
# shared_bundle.py — build_scalar_simplicial (exactness: delta_1 @ delta_0 = 0)
# ---------------------------------------------------------------------------


class TestBuildScalarSimplicial:
    def _make_complex(self):
        areas = {"A", "B", "C", "D"}
        edges = [("A", "B"), ("B", "C"), ("A", "C"), ("A", "D"), ("B", "D")]
        triangles = [("A", "B", "C"), ("A", "B", "D")]
        return areas, edges, triangles

    def test_exactness_delta1_delta0_is_zero(self):
        areas, edges, triangles = self._make_complex()
        d0, d1, _, _ = build_scalar_simplicial(areas, edges, triangles)
        product = d1 @ d0
        assert product == pytest.approx(np.zeros_like(product), abs=1e-10)

    def test_delta0_shape(self):
        areas, edges, triangles = self._make_complex()
        d0, d1, verts, edge_list = build_scalar_simplicial(areas, edges, triangles)
        assert d0.shape == (len(edge_list), len(verts))
        assert d1.shape == (len(triangles), len(edge_list))

    def test_coboundary_of_constant_is_zero(self):
        areas, edges, triangles = self._make_complex()
        d0, _, verts, _ = build_scalar_simplicial(areas, edges, triangles)
        f_const = np.ones(len(verts))
        assert d0 @ f_const == pytest.approx(np.zeros(d0.shape[0]), abs=1e-10)

    def test_larger_complex_exactness(self):
        areas = {f"N{i}" for i in range(6)}
        edges = [
            ("N0", "N1"), ("N1", "N2"), ("N0", "N2"),
            ("N2", "N3"), ("N3", "N4"), ("N2", "N4"),
            ("N4", "N5"), ("N0", "N5"),
        ]
        triangles = [("N0", "N1", "N2"), ("N2", "N3", "N4")]
        d0, d1, _, _ = build_scalar_simplicial(areas, edges, triangles)
        product = d1 @ d0
        assert product == pytest.approx(np.zeros_like(product), abs=1e-10)


# ---------------------------------------------------------------------------
# shared_bundle.py — hodge_decompose_edge
# ---------------------------------------------------------------------------


class TestHodgeDecomposeEdge:
    def _make_complex(self):
        areas = {"A", "B", "C", "D"}
        edges = [("A", "B"), ("B", "C"), ("A", "C"), ("A", "D"), ("B", "D")]
        triangles = [("A", "B", "C"), ("A", "B", "D")]
        return build_scalar_simplicial(areas, edges, triangles)

    def test_fractions_sum_to_one(self):
        d0, d1, _, edge_list = self._make_complex()
        rng = np.random.default_rng()
        omega = rng.standard_normal(len(edge_list))
        result = hodge_decompose_edge(d0, d1, omega)
        total = result["grad_frac"] + result["curl_frac"] + result["harm_frac"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_gradient_of_node_function_has_zero_curl(self):
        d0, d1, verts, edge_list = self._make_complex()
        rng = np.random.default_rng()
        f = rng.standard_normal(len(verts))
        omega = d0 @ f
        result = hodge_decompose_edge(d0, d1, omega)
        assert result["grad_frac"] == pytest.approx(1.0, abs=1e-6)
        assert result["curl_frac"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_form_returns_zero_norms(self):
        d0, d1, _, edge_list = self._make_complex()
        omega = np.zeros(len(edge_list))
        result = hodge_decompose_edge(d0, d1, omega)
        assert result["total_norm"] == pytest.approx(0.0, abs=1e-10)
        assert result["grad_frac"] == 0
        assert result["curl_frac"] == 0
        assert result["harm_frac"] == 0

    def test_curl_form_has_zero_gradient(self):
        d0, d1, _, edge_list = self._make_complex()
        if d1.shape[0] == 0:
            pytest.skip("No triangles in complex")
        rng = np.random.default_rng()
        eta = rng.standard_normal(d1.shape[0])
        omega = d1.T @ eta
        result = hodge_decompose_edge(d0, d1, omega)
        assert result["curl_frac"] == pytest.approx(1.0, abs=1e-6)
        assert result["grad_frac"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# shared_bundle.py — build_coboundary
# ---------------------------------------------------------------------------


class TestBuildCoboundary:
    def test_shape(self):
        k = 2
        areas = {"A", "B", "C"}
        restriction_maps = {
            ("A", "B"): np.eye(k),
            ("B", "C"): np.eye(k),
        }
        delta, edges, verts = build_coboundary(restriction_maps, areas, k)
        assert delta.shape == (len(edges) * k, len(verts) * k)

    def test_flat_section_in_kernel(self):
        k = 2
        areas = {"A", "B", "C"}
        restriction_maps = {
            ("A", "B"): np.eye(k),
            ("B", "C"): np.eye(k),
        }
        delta, edges, verts = build_coboundary(restriction_maps, areas, k)
        v = np.array([1.0, 2.0])
        section = np.tile(v, len(verts))
        result = delta @ section
        assert result == pytest.approx(np.zeros(len(edges) * k), abs=1e-10)
