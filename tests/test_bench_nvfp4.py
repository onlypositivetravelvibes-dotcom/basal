"""basal-bench --modes nvfp4 hands the model folder's NVFP4_AMAX.json to the backend, as basal-serve does."""
import json

import pytest

from basal import bench


class Recorder:
    def __init__(self, md, dtype, quant, **kw):
        self.quant, self.kw = quant, kw


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setattr(bench, "GraphBackend", Recorder)


def test_nvfp4_reads_the_amax_file_of_the_model_folder(graph, tmp_path):
    (tmp_path / "NVFP4_AMAX.json").write_text(json.dumps({"amax": {"2.q": 3.5}}), encoding="utf-8")
    be = bench.build("nvfp4", tmp_path)
    assert be.quant == "nvfp4" and be.kw["nvfp4_amax"] == {"2.q": 3.5} and be.kw["nvfp4_scope"] == "all"


def test_nvfp4_without_the_file_passes_no_scales(graph, tmp_path):
    assert "nvfp4_amax" not in bench.build("nvfp4", tmp_path).kw


def test_other_modes_pass_no_scales_even_when_the_file_exists(graph, tmp_path):
    (tmp_path / "NVFP4_AMAX.json").write_text(json.dumps({"amax": {"2.q": 3.5}}), encoding="utf-8")
    assert "nvfp4_amax" not in bench.build("fast", tmp_path).kw
