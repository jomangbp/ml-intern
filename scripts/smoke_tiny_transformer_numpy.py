#!/usr/bin/env python3
"""CPU-only tiny transformer smoke test (numpy)."""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def tiny_transformer_forward(batch=2, seq=8, d_model=32, n_heads=4, vocab=128):
    assert d_model % n_heads == 0
    d_head = d_model // n_heads

    rng = np.random.default_rng(0)
    token_ids = rng.integers(0, vocab, size=(batch, seq))

    emb = rng.normal(0, 0.02, size=(vocab, d_model))
    pos = rng.normal(0, 0.02, size=(seq, d_model))

    x = emb[token_ids] + pos[None, :, :]

    # Single self-attention block
    Wq = rng.normal(0, 0.02, size=(d_model, d_model))
    Wk = rng.normal(0, 0.02, size=(d_model, d_model))
    Wv = rng.normal(0, 0.02, size=(d_model, d_model))
    Wo = rng.normal(0, 0.02, size=(d_model, d_model))

    q = x @ Wq
    k = x @ Wk
    v = x @ Wv

    # [B, T, D] -> [B, H, T, Dh]
    q = q.reshape(batch, seq, n_heads, d_head).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq, n_heads, d_head).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, n_heads, d_head).transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
    attn = softmax(scores, axis=-1)
    ctx = attn @ v
    ctx = ctx.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)

    x = layer_norm(x + (ctx @ Wo))

    # MLP block
    W1 = rng.normal(0, 0.02, size=(d_model, 4 * d_model))
    W2 = rng.normal(0, 0.02, size=(4 * d_model, d_model))
    ff = np.maximum(0, x @ W1) @ W2
    x = layer_norm(x + ff)

    lm_head = rng.normal(0, 0.02, size=(d_model, vocab))
    logits = x @ lm_head

    assert logits.shape == (batch, seq, vocab)
    assert np.isfinite(logits).all()
    return logits


if __name__ == "__main__":
    out = tiny_transformer_forward()
    print("tiny-transformer numpy smoke PASSED", out.shape, float(out.mean()), float(out.std()))
