"""shots.py — Shot 数据结构 + 预设网格 (B.1)."""
from __future__ import annotations

from llm_infer_sim.calibration.shots import (
    Shot,
    DENSE_SHOTS,
    ATTENTION_SHOTS,
    PER_SEQUENCE_SHOTS,
    all_shots_for_kind,
)


# ---- Shot 数据结构 ----

def test_shot_hydrate_roundtrip_dense():
    s = Shot(kind="dense", num_new_tokens=512)
    s2 = Shot.hydrate(s.to_dict())
    assert s == s2


def test_shot_hydrate_roundtrip_attention():
    s = Shot(
        kind="attention",
        num_new_tokens=64,
        num_decode_seqs=4,
        kv_lens_prefill=[0],
        kv_lens_decode=[256, 256, 256, 256],
        prefill_chunk=60,
    )
    s2 = Shot.hydrate(s.to_dict())
    assert s == s2


def test_csv_key_dense_is_tokens_only():
    s = Shot(kind="dense", num_new_tokens=128)
    assert s.csv_key() == (128,)


def test_csv_key_attention_is_4tuple():
    s = Shot(
        kind="attention", num_new_tokens=8, num_decode_seqs=8,
        kv_lens_decode=[1024] * 8,
    )
    # (prefill_chunk=0, kv_prefill=0, n_decode=8, kv_decode=1024)
    assert s.csv_key() == (0, 0, 8, 1024)


def test_csv_key_per_seq():
    s = Shot(kind="per_sequence", num_new_tokens=4, num_decode_seqs=4,
             kv_lens_decode=[256, 256, 256, 256])
    assert s.csv_key() == (4,)


def test_csv_key_unknown_kind_raises():
    s = Shot(kind="bogus", num_new_tokens=1)
    import pytest
    with pytest.raises(ValueError, match="Unknown shot kind"):
        s.csv_key()


# ---- 预设网格 ----

def test_dense_grid_covers_decode_to_prefill():
    """DENSE_SHOTS 至少覆盖 1 (decode) / 128 (small prefill) / 2048 (large prefill)."""
    tokens = {s.num_new_tokens for s in DENSE_SHOTS}
    assert 1 in tokens
    assert 128 in tokens
    assert 2048 in tokens
    assert len(DENSE_SHOTS) >= 6


def test_dense_shots_all_kind_dense():
    assert all(s.kind == "dense" for s in DENSE_SHOTS)


def test_per_sequence_shots_scale_with_seq_count():
    """seq 数应单调递增."""
    seq_counts = [s.num_decode_seqs for s in PER_SEQUENCE_SHOTS]
    assert seq_counts == sorted(seq_counts)
    assert seq_counts[0] >= 1


def test_attention_grid_has_decode_prefill_mixed():
    """三种 attention 形态都覆盖到."""
    decode_only = [s for s in ATTENTION_SHOTS
                   if s.prefill_chunk == 0 and s.num_decode_seqs > 0]
    prefill_only = [s for s in ATTENTION_SHOTS
                    if s.prefill_chunk > 0 and s.num_decode_seqs == 0]
    mixed = [s for s in ATTENTION_SHOTS
             if s.prefill_chunk > 0 and s.num_decode_seqs > 0]
    assert len(decode_only) > 0
    assert len(prefill_only) > 0
    assert len(mixed) > 0


def test_all_shots_for_kind_dispatcher():
    assert all_shots_for_kind("dense") is DENSE_SHOTS
    assert all_shots_for_kind("attention") is ATTENTION_SHOTS
    assert all_shots_for_kind("per_sequence") is PER_SEQUENCE_SHOTS

    import pytest
    with pytest.raises(ValueError, match="Unknown shot kind"):
        all_shots_for_kind("moe")    # 暂不在 B.1 范围
