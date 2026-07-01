"""Causal subspace fitting via LDA, PCA, and DAS.

LDA (Linear Discriminant Analysis): maximizes between-class / within-class variance.
For high-dimensional neural data (n_neurons >> n_trials), uses PCA pre-whitening.
This is the primary method for observational neural data.

PCA baseline: top-k PCs of activity. Captures variance, not discrimination —
locomotion/arousal signals dominate choice in most brain regions.

DAS (Distributed Alignment Search): optimize a rotation matrix R such that
intervening on the first k dimensions of R^T @ x causes behavior to match
the source trial. Requires causal intervention capability (transformers only).
"""
import logging

import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

logger = logging.getLogger(__name__)


def fit_pca_subspace(
    activity: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """PCA subspace from trial-averaged activity difference.

    Args:
        activity: (n_trials, n_neurons) population activity (time-averaged or at one bin)
        labels: (n_trials,) binary condition labels (e.g., left/right choice)
        k: subspace dimensionality

    Returns:
        (n_neurons, k) orthonormal basis for the top-k PC subspace of the
        between-condition difference.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError(f"Expected binary labels, got {len(unique_labels)} unique values")

    mean_0 = activity[labels == unique_labels[0]].mean(axis=0)
    mean_1 = activity[labels == unique_labels[1]].mean(axis=0)
    diff = mean_1 - mean_0

    centered = activity - activity.mean(axis=0, keepdims=True)
    U, s, Vt = svd(centered, full_matrices=False)

    proj_diff = Vt[:k] @ diff
    order = np.argsort(-np.abs(proj_diff))
    return Vt[order[:k]].T


def fit_lda_subspace(
    activity: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """LDA subspace: maximally discriminative directions for the labeled condition.

    For binary labels, LDA produces 1 discriminative dimension. Remaining k-1
    dimensions are filled with PCA of within-class residuals (the next most
    variable directions orthogonal to the discriminant).

    When n_neurons >= n_trials, applies PCA pre-whitening first (standard
    high-dimensional LDA pipeline).

    Args:
        activity: (n_trials, n_neurons)
        labels: (n_trials,) binary or multi-class
        k: subspace dimensionality

    Returns:
        (n_neurons, k) orthonormal basis
    """
    n_trials, n_neurons = activity.shape
    n_classes = len(np.unique(labels))
    k = min(k, n_neurons - 1)

    if n_neurons >= n_trials:
        n_pca = min(n_trials - 1, n_neurons - 1, 50)
        pca = PCA(n_components=n_pca)
        activity_reduced = pca.fit_transform(activity)
        pca_basis = pca.components_  # (n_pca, n_neurons)
    else:
        activity_reduced = activity
        pca_basis = np.eye(n_neurons)

    lda_k = min(k, n_classes - 1, activity_reduced.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=lda_k)
    lda.fit(activity_reduced, labels)

    lda_dirs = lda.scalings_[:, :lda_k]  # (n_reduced, lda_k)
    lda_in_neurons = pca_basis.T @ lda_dirs  # (n_neurons, lda_k)

    if k > lda_k:
        proj = lda_in_neurons @ lda_in_neurons.T
        residuals = activity - activity @ proj
        n_extra = min(k - lda_k, n_neurons - lda_k, n_trials - 1)
        pca_resid = PCA(n_components=n_extra)
        pca_resid.fit(residuals)
        extra = pca_resid.components_.T  # (n_neurons, n_extra)
        combined = np.hstack([lda_in_neurons, extra])
    else:
        combined = lda_in_neurons

    U, _ = np.linalg.qr(combined)
    return U[:, :k]


def fit_das_subspace(
    activity: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    n_epochs: int = 500,
    lr: float = 0.01,
    device: str = "cpu",
) -> np.ndarray:
    """Fit a causal subspace via Distributed Alignment Search.

    Optimizes rotation R on the Stiefel manifold St(k, n) such that
    swapping the top-k rotated dimensions between source and base trials
    minimizes cross-entropy loss on the behavioral variable.

    Args:
        activity: (n_trials, n_neurons) population activity
        labels: (n_trials,) binary condition labels
        k: subspace dimensionality
        n_epochs: training epochs
        lr: learning rate
        device: 'cpu' or 'cuda'

    Returns:
        (n_neurons, k) orthonormal basis for the causal subspace
    """
    import torch
    import torch.nn as nn

    n_neurons = activity.shape[1]
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    R = nn.Parameter(torch.randn(n_neurons, k, device=device) * 0.01)
    classifier = nn.Linear(n_neurons, 2).to(device)

    optimizer = torch.optim.Adam(list(classifier.parameters()) + [R], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = len(X)
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        base_idx = perm[: n // 2]
        source_idx = perm[n // 2 : n]

        Q, _ = torch.linalg.qr(R)

        base_acts = X[base_idx]
        source_acts = X[source_idx]

        proj_base = base_acts @ Q  # (batch, k)
        proj_source = source_acts @ Q

        intervened = base_acts + (proj_source - proj_base) @ Q.T

        logits = classifier(intervened)
        loss = loss_fn(logits, y[source_idx[: len(base_idx)]])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            acc = (logits.argmax(dim=1) == y[source_idx[: len(base_idx)]]).float().mean()
            logger.info(f"Epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, IIA={acc.item():.3f}")

    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
    return Q.cpu().numpy()
