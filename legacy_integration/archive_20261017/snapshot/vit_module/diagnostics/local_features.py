"""Spatial descriptors in each encoder's own coordinates."""

import numpy as np


def pool_tokens(tokens, grid=3, has_cls=False):
    tokens = np.asarray(tokens)
    patches = tokens[:, 1:] if has_cls else tokens
    side = int(np.sqrt(patches.shape[1]))
    if side * side != patches.shape[1] or not 1 < grid <= side:
        raise ValueError("Square patch lattice and grid between 2 and lattice size required")
    boundaries = np.linspace(0, side, grid + 1)
    starts = np.arange(side)
    weights = np.stack([
        np.maximum(0, np.minimum(starts + 1, boundaries[index + 1]) -
                   np.maximum(starts, boundaries[index]))
        for index in range(grid)
    ])
    weights /= weights.sum(axis=1, keepdims=True)
    lattice = patches.reshape(len(patches), side, side, patches.shape[-1])
    return np.einsum("ah,bw,nhwd->nabd", weights, weights, lattice, optimize=True).reshape(
        len(patches), grid * grid, patches.shape[-1])


def relations(regions):
    regions = np.asarray(regions, dtype=np.float64)
    normalized = regions / np.linalg.norm(regions, axis=-1, keepdims=True).clip(1e-12)
    gram = normalized @ normalized.transpose(0, 2, 1)
    pairs = np.triu_indices(regions.shape[1], k=1)
    pooled = regions.mean(axis=1)
    pooled /= np.linalg.norm(pooled, axis=-1, keepdims=True).clip(1e-12)
    agreement = np.einsum("nrd,nd->nr", normalized, pooled)
    return np.concatenate([gram[:, pairs[0], pairs[1]], agreement], axis=1)
