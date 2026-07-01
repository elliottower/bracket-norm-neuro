"""Grassmannian distances and representational similarity measures.

Core functions:
- principal_angles: canonical angles between two subspaces
- grassmannian_distance: geodesic on Gr(k, n)
- gauge_normalized_distance: after removing neuron-permutation / scaling gauge
- cka: centered kernel alignment (the baseline we're comparing against)
"""
import numpy as np
from scipy.linalg import svd


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute principal angles between subspaces spanned by columns of U and V.

    Args:
        U: (n, k1) orthonormal basis for subspace 1
        V: (n, k2) orthonormal basis for subspace 2

    Returns:
        (min(k1,k2),) array of principal angles in radians, sorted descending
    """
    _, s, _ = svd(U.T @ V, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def grassmannian_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Geodesic distance on the Grassmannian Gr(k, n).

    d(S1, S2) = sqrt(sum(theta_i^2)) where theta_i are principal angles.
    """
    angles = principal_angles(U, V)
    return float(np.sqrt(np.sum(angles**2)))


def subspace_overlap(U: np.ndarray, V: np.ndarray) -> float:
    """Mean cosine of principal angles — 1.0 = aligned, 0.0 = orthogonal."""
    angles = principal_angles(U, V)
    return float(np.mean(np.cos(angles)))


def gauge_normalized_distance(
    U: np.ndarray,
    V: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
) -> float:
    """Grassmannian distance after gauge normalization.

    Gauge symmetries in neural recordings:
    1. Neuron permutation across sessions (handled by fitting subspaces
       from trial-averaged responses to matched stimuli)
    2. Rotation within the causal subspace (absorbed by Grassmannian metric)
    3. Amplitude scaling (normalized by effective rank)

    Args:
        U, V: (n, k) orthonormal bases for subspaces
        X1, X2: (n_trials, n_neurons) activity matrices for scaling normalization

    Returns:
        Gauge-normalized geodesic distance
    """
    _, s1, _ = svd(X1, full_matrices=False)
    _, s2, _ = svd(X2, full_matrices=False)
    eff_rank_1 = _effective_rank(s1)
    eff_rank_2 = _effective_rank(s2)
    scale = np.sqrt(eff_rank_1 * eff_rank_2)

    raw_dist = grassmannian_distance(U, V)
    return raw_dist / max(scale, 1e-8)


def _effective_rank(singular_values: np.ndarray) -> float:
    """Effective rank via participation ratio of singular values."""
    s = singular_values[singular_values > 1e-10]
    p = s**2 / np.sum(s**2)
    return float(np.exp(-np.sum(p * np.log(p + 1e-10))))


def cka(X: np.ndarray, Y: np.ndarray, kernel: str = "linear") -> float:
    """Centered Kernel Alignment between two population activity matrices.

    Args:
        X: (n_stimuli, n_neurons_1) — trial-averaged responses
        Y: (n_stimuli, n_neurons_2)
        kernel: 'linear' or 'rbf'

    Returns:
        CKA similarity in [0, 1]
    """
    if kernel == "linear":
        K = X @ X.T
        L = Y @ Y.T
    elif kernel == "rbf":
        from scipy.spatial.distance import cdist

        sigma_x = np.median(cdist(X, X, "euclidean"))
        sigma_y = np.median(cdist(Y, Y, "euclidean"))
        K = np.exp(-cdist(X, X, "sqeuclidean") / (2 * sigma_x**2))
        L = np.exp(-cdist(Y, Y, "sqeuclidean") / (2 * sigma_y**2))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    K = _center_kernel(K)
    L = _center_kernel(L)

    hsic = np.sum(K * L)
    norm = np.sqrt(np.sum(K * K) * np.sum(L * L))
    return float(hsic / max(norm, 1e-10))


def _center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def debiased_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased (unbiased) linear CKA using the HSIC_1 estimator from Song et al. 2012.

    Corrects the finite-sample bias identified by Murphy, Zylberberg & Fyshe 2024
    (arXiv:2405.01012) where standard CKA produces inflated values when sample-feature
    ratios differ across regions.

    Args:
        X: (n, p1) activity matrix
        Y: (n, p2) activity matrix

    Returns:
        Debiased CKA in approximately [0, 1] (can go slightly negative)
    """
    n = X.shape[0]
    if n < 4:
        return 0.0

    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = _unbiased_hsic(K, L, n)
    hsic_kk = _unbiased_hsic(K, K, n)
    hsic_ll = _unbiased_hsic(L, L, n)

    denom = np.sqrt(max(hsic_kk, 0.0) * max(hsic_ll, 0.0))
    if denom < 1e-10:
        return 0.0
    return float(hsic_kl / denom)


def _unbiased_hsic(K: np.ndarray, L: np.ndarray, n: int) -> float:
    """Unbiased HSIC estimator (Song et al. 2012, Eq. 3).

    HSIC_1 = 1/(n(n-3)) * [tr(K'L') + 1'K'1 * 1'L'1 / ((n-1)(n-2)) - 2/(n-2) * 1'K'L'1]
    where K' = K with diagonal zeroed, same for L'.
    """
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)

    trace_kl = np.sum(K * L)
    sum_k = K.sum()
    sum_l = L.sum()
    sum_kl = (K @ L).trace()

    # Avoid division by zero for tiny n
    term1 = trace_kl
    term2 = sum_k * sum_l / ((n - 1) * (n - 2))
    term3 = 2.0 * sum_kl / (n - 2)

    return float((term1 + term2 - term3) / (n * (n - 3)))


def chordal_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Chordal (projection Frobenius) distance between subspaces.

    d_c(S1, S2) = sqrt(sum(sin^2(theta_i))) where theta_i are principal angles.
    Equivalent to (1/sqrt(2)) * ||P_U - P_V||_F where P is the projection matrix.
    """
    angles = principal_angles(U, V)
    return float(np.sqrt(np.sum(np.sin(angles) ** 2)))


def all_subspace_distances(U: np.ndarray, V: np.ndarray) -> dict:
    """Compute all three subspace distance metrics at once.

    Returns dict with grassmannian (geodesic), chordal, mean_principal_angle_deg,
    max_principal_angle_deg, and subspace_overlap.
    """
    angles = principal_angles(U, V)
    return {
        "grassmannian": float(np.sqrt(np.sum(angles ** 2))),
        "chordal": float(np.sqrt(np.sum(np.sin(angles) ** 2))),
        "mean_principal_angle_deg": float(np.degrees(np.mean(angles))),
        "max_principal_angle_deg": float(np.degrees(np.max(angles))),
        "subspace_overlap": float(np.mean(np.cos(angles))),
    }
