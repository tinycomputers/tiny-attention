"""Float64 attention: the ground truth every error number is measured against."""

import numpy as np


def attention(q, k, v, causal=False):
    """softmax(q @ k.T / sqrt(d)) @ v. Returns (out, probs)."""
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    n, d = q.shape
    s = q @ k.T / np.sqrt(d)
    if causal:
        s = np.where(np.tril(np.ones((n, n), dtype=bool)), s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return p @ v, p
