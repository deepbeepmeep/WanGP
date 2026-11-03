# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Immutable CPU cache segments: share a stable prefix, copy only the changed tail."""

import torch

from mmgp import transfer


def snapshot_cache(tensor, ranges, *, axis, previous=None, reuse=0):
    segments = []
    remaining = reuse
    if previous is not None:
        for start, saved in previous["segments"]:
            count = min(remaining, saved.shape[axis])
            if count == 0:
                break
            segments.append((start, saved.narrow(axis, 0, count)))
            remaining -= count
    assert remaining == 0, "Saved cache does not cover its reusable prefix."
    remaining = reuse
    for start, end in ranges:
        skipped = min(remaining, end - start)
        start += skipped
        remaining -= skipped
        if start < end:
            segments.append((start, transfer.move(tensor.detach().as_subclass(torch.Tensor).narrow(axis, start, end - start), "cpu")))
    return {"shape": tuple(tensor.shape), "axis": axis, "segments": segments}


def restore_cache(tensor, snapshot):
    axis = snapshot["axis"]
    for start, saved in snapshot["segments"]:
        transfer.copy_(tensor.narrow(axis, start, saved.shape[axis]), saved)
