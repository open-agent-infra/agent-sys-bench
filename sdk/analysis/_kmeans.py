"""Tiny k-means (numpy only, no sklearn) for breakdown example selection."""

import numpy as np


def kmeans(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 0):
    """Return (centers, labels). Deterministic."""
    n = len(X)
    if n == 0 or k <= 0:
        return np.zeros((0,)), []
    k = min(k, n)
    rng = np.random.default_rng(seed)
    # k-means++ init
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d = np.min([np.linalg.norm(X - c, axis=1) for c in centers], axis=0)
        if d.sum() == 0:
            centers.append(X[rng.integers(n)])
        else:
            probs = d / d.sum()
            centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # assign
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
    return centers, labels
