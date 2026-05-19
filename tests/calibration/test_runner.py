"""runner.py — 顶层 orchestration (B.3).

mock engine_factory + fire_fn, 不真起 vLLM. 验证:
  1. CATEGORIES 三类完整 (dense / attention / per_sequence)
  2. catalog.slice_for_category 分类正确 (Qwen3 真 catalog 跑通)
  3. run_calibration 端到端: 三类 CSV + meta.yaml 都落地
  4. resume: 第二次跑跳过已 visited shot
  5. kinds 过滤 (只跑指定 category)
  6. _build_output_dir 路径模式
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from llm_infer_sim.calibration.catalog import Catalog
from llm_infer_sim.calibration.runner import (
    CATEGORIES,
    _build_output_dir,
    run_calibration,
)


# ---- CATEGORIES 元信息 ----

def test_categories_cover_three_kinds():
    kinds = {c.kind for c in CATEGORIES}
    assert kinds == {"dense", "attention", "per_sequence"}


def test_each_category_has_unique_filename():
    files = [c.filename for c in CATEGORIES]
    assert len(set(files)) == len(files)


# ---- catalog slice_for_category 跟 Qwen3 catalog 对齐 ----

def test_qwen3_slice_for_category_dense():
    cat = Catalog.load("qwen3")
    dense = cat.slice_for_category("dense")
    # dense 应含: embedding / layernorm / qkv_proj / qk_norm / rotary_emb / o_proj /
    #             gate_up_proj / act_fn / down_proj / final_layernorm
    assert set(dense.keys()).issuperset({
        "embedding", "layernorm", "qkv_proj", "qk_norm", "rotary_emb",
        "o_proj", "gate_up_proj", "act_fn", "down_proj", "final_layernorm",
    })
    # 不含 attention / lm_head
    assert "attention" not in dense
    assert "lm_head" not in dense


def test_qwen3_slice_for_category_attention():
    cat = Catalog.load("qwen3")
    attn = cat.slice_for_category("attention")
    assert set(attn.keys()) == {"attention"}


def test_qwen3_slice_for_category_per_sequence():
    cat = Catalog.load("qwen3")
    ps = cat.slice_for_category("per_sequence")
    # 至少含 lm_head (Qwen3 catalog 暂不含 sampler, 后续补)
    assert "lm_head" in ps
    # 不含 dense / attention 部分
    assert "qkv_proj" not in ps
    assert "attention" not in ps


# ---- _build_output_dir 路径模式 ----

def test_build_output_dir_hf_id():
    p = _build_output_dir("configs/efficiency/raw", "RTX_4090", "Qwen/Qwen3-4B",
                          "bfloat16", 1)
    assert str(p) == "configs/efficiency/raw/RTX_4090/Qwen/Qwen3-4B/bfloat16/tp1"


def test_build_output_dir_local_path():
    """local 绝对路径模型: 取最后目录名."""
    with tempfile.TemporaryDirectory() as td:
        model_dir = Path(td) / "Qwen3-4B"
        model_dir.mkdir()
        p = _build_output_dir(
            "configs/efficiency/raw", "RTX_4090", str(model_dir), "bf16", 2,
        )
        assert p.name == "tp2"
        assert p.parent.name == "bf16"
        assert p.parent.parent.name == "Qwen3-4B"


# ---- run_calibration with mock engine ----

def _fake_engine_factory(**kwargs):
    """mock vllm.LLM, 只返一个 sentinel."""
    return mock.MagicMock(name="MockLLM", _ctor_kwargs=kwargs)


def _fake_fire_fn(engine, shot_dict, slice_, kind, iterations):
    """模拟 fire_shot: 给每个 catalog slice 里 canonical 出 1 个 sample."""
    samples = [
        {"layer": canonical, "op_kind": fields["op_kind"],
         "microseconds": 10.0 + len(canonical)}  # 给点变化方便 debug
        for canonical, fields in slice_.items()
    ]
    return [samples]   # 单 rank tp=1


def test_run_calibration_writes_three_csvs_and_meta():
    with tempfile.TemporaryDirectory() as td:
        out_dir = run_calibration(
            model="Qwen/Qwen3-4B",
            model_type="qwen3",
            hardware="RTX_4090",
            dtype="bfloat16",
            output_root=td,
            tp=1,
            iterations=1,
            resume=False,
            engine_factory=_fake_engine_factory,
            fire_fn=_fake_fire_fn,
        )
        assert (out_dir / "dense.csv").exists()
        assert (out_dir / "attention.csv").exists()
        assert (out_dir / "per_sequence.csv").exists()
        assert (out_dir / "meta.yaml").exists()


def test_run_calibration_kinds_filter():
    """kinds=('dense',) 只跑 dense, 不出 attention/per_sequence."""
    with tempfile.TemporaryDirectory() as td:
        out_dir = run_calibration(
            model="Qwen/Qwen3-4B", model_type="qwen3", hardware="RTX_4090",
            output_root=td, kinds=("dense",), resume=False,
            engine_factory=_fake_engine_factory, fire_fn=_fake_fire_fn,
        )
        assert (out_dir / "dense.csv").exists()
        assert not (out_dir / "attention.csv").exists()
        assert not (out_dir / "per_sequence.csv").exists()


def test_run_calibration_invalid_kinds_raises():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="没匹中"):
            run_calibration(
                model="x", model_type="qwen3", hardware="RTX_4090",
                output_root=td, kinds=("bogus",),
                engine_factory=_fake_engine_factory, fire_fn=_fake_fire_fn,
            )


def test_run_calibration_resume_skips_visited():
    """第一次跑写 N 行, 第二次 resume=True 应跳过 visited shot, 不复 fire."""
    fire_counter = {"n": 0}

    def counting_fire(engine, shot_dict, slice_, kind, iterations):
        fire_counter["n"] += 1
        return _fake_fire_fn(engine, shot_dict, slice_, kind, iterations)

    with tempfile.TemporaryDirectory() as td:
        # 第一次跑
        run_calibration(
            model="Qwen/Qwen3-4B", model_type="qwen3", hardware="RTX_4090",
            output_root=td, kinds=("dense",), resume=False,
            engine_factory=_fake_engine_factory, fire_fn=counting_fire,
        )
        first_count = fire_counter["n"]
        assert first_count > 0

        # 第二次 resume=True, 应跳过全部
        run_calibration(
            model="Qwen/Qwen3-4B", model_type="qwen3", hardware="RTX_4090",
            output_root=td, kinds=("dense",), resume=True,
            engine_factory=_fake_engine_factory, fire_fn=counting_fire,
        )
        # fire 没被再调
        assert fire_counter["n"] == first_count


def test_run_calibration_dense_csv_rows_per_shot():
    """每 dense shot 应出 |slice| 行 (catalog slice 里每 canonical 1 行)."""
    from llm_infer_sim.calibration.csv_io import read_dense
    from llm_infer_sim.calibration.shots import DENSE_SHOTS

    with tempfile.TemporaryDirectory() as td:
        out_dir = run_calibration(
            model="Qwen/Qwen3-4B", model_type="qwen3", hardware="RTX_4090",
            output_root=td, kinds=("dense",), resume=False,
            engine_factory=_fake_engine_factory, fire_fn=_fake_fire_fn,
        )
        rows = read_dense(out_dir / "dense.csv")
        catalog = Catalog.load("qwen3")
        n_dense_canonicals = len(catalog.slice_for_category("dense"))
        # |rows| = #shots × #dense_canonicals
        assert len(rows) == len(DENSE_SHOTS) * n_dense_canonicals


def test_run_calibration_fire_failure_skipped():
    """fire_fn 抛错时, 该 shot 应被 skip, runner 继续."""
    def flaky_fire(engine, shot_dict, slice_, kind, iterations):
        if shot_dict["num_new_tokens"] == 1:
            raise RuntimeError("fake fire failure")
        return _fake_fire_fn(engine, shot_dict, slice_, kind, iterations)

    with tempfile.TemporaryDirectory() as td:
        out_dir = run_calibration(
            model="Qwen/Qwen3-4B", model_type="qwen3", hardware="RTX_4090",
            output_root=td, kinds=("dense",), resume=False,
            engine_factory=_fake_engine_factory, fire_fn=flaky_fire,
        )
        # dense.csv 应不含 tokens=1 的行
        from llm_infer_sim.calibration.csv_io import read_dense
        rows = read_dense(out_dir / "dense.csv")
        assert all(r.tokens != 1 for r in rows)


def test_run_calibration_bundle_yaml_skipped_for_mock_engine(monkeypatch):
    """mock engine 没 llm_engine, bundle.yaml 应被跳过 (warning) 但不挂."""
    with tempfile.TemporaryDirectory() as td:
        out_dir = run_calibration(
            model="x", model_type="qwen3", hardware="RTX_4090",
            output_root=td, kinds=("dense",), resume=False,
            engine_factory=_fake_engine_factory, fire_fn=_fake_fire_fn,
        )
        # mock engine → bundle.yaml 不应存在
        assert not (out_dir / "bundle.yaml").exists()
        # 但 dense.csv + meta.yaml 应正常落
        assert (out_dir / "dense.csv").exists()
        assert (out_dir / "meta.yaml").exists()


def test_run_calibration_meta_yaml_content():
    """meta.yaml 应含 model / hardware / dtype / iterations / captured_at."""
    import yaml
    with tempfile.TemporaryDirectory() as td:
        out_dir = run_calibration(
            model="Qwen/Qwen3-4B", model_type="qwen3", hardware="RTX_4090",
            dtype="bfloat16", tp=1, iterations=2,
            output_root=td, kinds=("dense",), resume=False,
            engine_factory=_fake_engine_factory, fire_fn=_fake_fire_fn,
        )
        meta = yaml.safe_load((out_dir / "meta.yaml").read_text())
        assert meta["model"] == "Qwen/Qwen3-4B"
        assert meta["hardware"] == "RTX_4090"
        assert meta["dtype"] == "bfloat16"
        assert meta["iterations"] == 2
        assert "captured_at" in meta
        assert "vllm_version" in meta
