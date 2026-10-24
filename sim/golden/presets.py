"""Preset Q/K/V for the visualizer. Float matrices, N x 4. Phase 8 exports these to web/src/lib/presets.ts.

All presets are deterministic (fixed seeds). `labels` are token strings for the heatmap axes.
"""

from __future__ import annotations

import numpy as np

D = 4

TOY_TOKENS = ["<s>", "The", "cat", "chased", "the", "mouse", "because", "it", "was", "very", "hungry", "."]
_NOUNS = {"cat": 1.0, "mouse": 1.0}
_SUBJECT = {"cat": 1.0}
_PRONOUNS = {"it"}


def toy_sentence(theta: float = 0.9, prev_strength: float = 2.0, noun_pull: float = 3.0):
    """Handcrafted d=4 embeddings. dim0 = noun-ness (keys) / wants-a-noun (query), dims 1-2 = a rotary
    position code that makes every token attend to its predecessor, dim3 = subject-ness. The pronoun
    'it' has no positional pull and wants a noun that is a subject: it attends to 'cat', not 'mouse'."""
    n = len(TOY_TOKENS)
    q = np.zeros((n, D))
    k = np.zeros((n, D))
    v = np.zeros((n, D))
    rng = np.random.default_rng(7)
    for i, tok in enumerate(TOY_TOKENS):
        k[i] = [_NOUNS.get(tok, 0.0), np.cos(theta * i), np.sin(theta * i), _SUBJECT.get(tok, 0.0)]
        if tok in _PRONOUNS:
            q[i] = [noun_pull, 0.0, 0.0, noun_pull]
        else:
            q[i] = [0.0, prev_strength * np.cos(theta * (i - 1)), prev_strength * np.sin(theta * (i - 1)), 0.0]
        v[i] = rng.normal(0, 1, D)
    return dict(name="toy_sentence", labels=TOY_TOKENS, q=q, k=k, v=v, causal=True)


def _values(n, seed):
    return np.random.default_rng(seed).normal(0, 1, (n, D))


def previous_token(n=16):
    th = 0.8
    pos = np.array([[np.cos(th * i), np.sin(th * i)] for i in range(n)])
    prev = np.array([[np.cos(th * (i - 1)), np.sin(th * (i - 1))] for i in range(n)])
    z = np.zeros((n, 2))
    return dict(name="previous_token", labels=[str(i) for i in range(n)],
                q=np.hstack([3 * prev, z]), k=np.hstack([pos, z]), v=_values(n, 1), causal=True)


def attention_sink(n=16):
    q = np.zeros((n, D))
    k = np.zeros((n, D))
    q[:, 0] = 4.0
    k[0, 0] = 2.0  # every query agrees with token 0
    return dict(name="attention_sink", labels=["sink"] + [str(i) for i in range(1, n)], q=q, k=k, v=_values(n, 2), causal=True)


def uniform(n=16):
    z = np.zeros((n, D))
    return dict(name="uniform", labels=[str(i) for i in range(n)], q=z, k=z, v=_values(n, 3), causal=False)


def self_attention(n=16):
    x = np.random.default_rng(4).normal(0, 1, (n, D))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return dict(name="self", labels=[str(i) for i in range(n)], q=4 * x, k=4 * x, v=_values(n, 5), causal=False)


def random_qkv(n=16, seed=0, amp=2.0):
    r = np.random.default_rng(seed)
    return dict(name="random", labels=[str(i) for i in range(n)],
                q=r.normal(0, amp, (n, D)), k=r.normal(0, amp, (n, D)), v=r.normal(0, 1, (n, D)), causal=False)


ALL = [toy_sentence, previous_token, attention_sink, uniform, self_attention, random_qkv]
