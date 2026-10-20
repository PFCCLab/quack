# Copyright (C) 2025, Tri Dao.
"""Segmented B (`B_idx`): one varlen-M GEMM whose K axis is the concatenation
of `num_segs` B segments must equal the same GEMM over an explicitly
concatenated weight, bit for bit, and match fp64 math."""

import paddle

paddle.enable_compat()

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from quack.gemm_config import GemmConfig  # noqa: E402
from quack.gemm_interface import gemm  # noqa: E402


def _dev(a, dtype):
    return paddle.to_tensor(np.ascontiguousarray(a)).astype(dtype)


def _config():
    return GemmConfig(
        tile_m=128,
        tile_n=256,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
        device_capacity=paddle.device.cuda.get_device_capability()[0],
    )


def _case(rng, E, I, H, c, sizes):
    cum = np.zeros(len(sizes) + 1, np.int32)
    cum[1:] = np.cumsum(sizes)
    bidx = np.stack([rng.choice(E, c, replace=False) for _ in sizes]).astype(np.int32)
    A = (paddle.randn([max(1, int(cum[-1])), c * I], dtype=paddle.float32) * 0.5).astype(paddle.bfloat16)
    W = (paddle.randn([E, H, I], dtype=paddle.float32) * 0.05).astype(paddle.bfloat16)  # (L, N, K)
    return cum, bidx, A, W


@pytest.mark.skipif(paddle.device.cuda.get_device_capability()[0] < 10, reason="SM100+")
@pytest.mark.parametrize("c", [2, 3, 4])
# 16 / 32 / 48 / 96: segments shorter than (or not a multiple of) the default
# 64-wide k-tile run with a 1- or 2-instruction k-tile
@pytest.mark.parametrize("I", [16, 32, 48, 64, 96, 128])
def test_segmented_b_matches_concatenated_weight(c, I):
    rng = np.random.RandomState(0)
    E, H = 32, 512
    sizes = [0, 1, 127, 128, 129, 300, 0, 64]  # empty, partial and multi-tile groups
    cum, bidx, A, W = _case(rng, E, I, H, c, sizes)
    n = int(cum[-1])
    out = paddle.zeros([n, H], dtype=paddle.bfloat16)
    gemm(A[:n], W.permute(0, 2, 1), out=out, cu_seqlens_m=_dev(cum, paddle.int32),
         B_idx=_dev(bidx, paddle.int32), tuned=False, config=_config())
    # reference: materialise [W[e_0] | ... | W[e_{c-1}]] per group, same kernel config
    Wcat = paddle.concat([W[paddle.to_tensor(bidx[:, j].astype(np.int64))] for j in range(c)], axis=2)
    ref = paddle.zeros([n, H], dtype=paddle.bfloat16)
    gemm(A[:n], Wcat.permute(0, 2, 1), out=ref, cu_seqlens_m=_dev(cum, paddle.int32), tuned=False,
         config=_config())
    paddle.device.synchronize()
    a, b = out.astype(paddle.float32).numpy(), ref.astype(paddle.float32).numpy()
    assert np.isfinite(a).all()
    assert np.array_equal(a, b), f"max |diff| {np.abs(a - b).max()}"
    Af, Wf = A[:n].astype(paddle.float32).numpy(), W.astype(paddle.float32).numpy()
    g = np.repeat(np.arange(len(sizes)), sizes)
    exp = np.zeros((n, H))
    for r in range(n):
        for j in range(c):
            exp[r] += Af[r, j * I : (j + 1) * I].astype(np.float64) @ Wf[bidx[g[r], j]].T
    assert np.linalg.norm(a - exp) / np.linalg.norm(exp) < 1e-2


@pytest.mark.skipif(paddle.device.cuda.get_device_capability()[0] < 10, reason="SM100+")
def test_segmented_b_rejects_bad_shapes():
    rng = np.random.RandomState(0)
    cum, bidx, A, W = _case(rng, 16, 128, 256, 2, [10, 20])
    out = paddle.zeros([30, 256], dtype=paddle.bfloat16)
    with pytest.raises(AssertionError):  # A's K is not num_segs * B's K
        gemm(A[:30, :192], W.permute(0, 2, 1), out=out, cu_seqlens_m=_dev(cum, paddle.int32),
             B_idx=_dev(bidx, paddle.int32), tuned=False, config=_config())
    cum2, bidx2, A2, W2 = _case(rng, 16, 24, 256, 2, [10, 20])
    with pytest.raises(AssertionError):  # B's K (24) is not whole MMA instructions (16)
        gemm(A2[:30], W2.permute(0, 2, 1), out=out, cu_seqlens_m=_dev(cum2, paddle.int32),
             B_idx=_dev(bidx2, paddle.int32), tuned=False, config=_config())
    with pytest.raises(AssertionError):  # B_idx rows != groups
        gemm(A[:30], W.permute(0, 2, 1), out=out, cu_seqlens_m=_dev(cum, paddle.int32),
             B_idx=_dev(bidx[:1], paddle.int32), tuned=False, config=_config())
    with pytest.raises(AssertionError):  # explicit config needs tuned=False
        gemm(A[:30], W.permute(0, 2, 1), out=out, cu_seqlens_m=_dev(cum, paddle.int32),
             B_idx=_dev(bidx, paddle.int32), tuned=True)
