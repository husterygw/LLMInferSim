"""pytest fixtures for collector tests."""
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """临时数据目录, 模拟 collector/data/."""
    d = tmp_path / "operator_db" / "RTX_4090" / "vllm-test"
    d.mkdir(parents=True)
    return d
