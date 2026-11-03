"""basal-bench --out: the directory is created before the run, and memory_gb of a backend without tracking (NaN) is
written as null, so the file is valid JSON for strict parsers."""
import json
import sys

import pytest

from basal import bench


class _Tok:
    pass


class _Be:
    tok = _Tok()


@pytest.fixture
def stub_run(monkeypatch):
    groups = [({}, None, None, None)] * 40
    monkeypatch.setattr(bench, "resolve", lambda *a, **k: object())
    monkeypatch.setattr(bench, "model_format", lambda md: 1)
    monkeypatch.setattr(bench, "load_questions", lambda *a, **k: [])
    monkeypatch.setattr(bench, "reset_memory", lambda: None)
    monkeypatch.setattr(bench, "build", lambda *a, **k: _Be())
    monkeypatch.setattr(bench, "device_name", lambda be=None: "stub")
    monkeypatch.setattr(bench, "groups_for", lambda *a, **k: groups)
    monkeypatch.setattr(bench, "measure", lambda be, g, n: ([[0.0, 1.0]] * len(g), dict(
        lat1_ms=1.0, lat2_ms=2.0, lat2_p95_ms=3.0, dec_s=4.0)))
    monkeypatch.setattr(bench, "memory_gb", lambda be: float("nan"))

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["basal-bench", "--modes", "gguf", *argv])
        bench.main()
    return run


def test_out_in_missing_directory_is_created_before_the_run(stub_run, tmp_path, monkeypatch):
    out = tmp_path / "results" / "deeper" / "bench.json"
    monkeypatch.setattr(bench, "build", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):  # the run dies at the first mode; the directory already exists
        stub_run("--out", str(out))
    assert out.parent.is_dir()


def test_out_is_strict_json_without_nan(stub_run, tmp_path):
    out = tmp_path / "results" / "strict.json"
    stub_run("--out", str(out))
    text = out.read_text(encoding="utf-8")

    def refuse(c):
        raise ValueError(c)
    rows = json.loads(text, parse_constant=refuse)  # NaN / Infinity literals raise
    assert rows[0]["mem_gb"] is None and rows[0]["mode"] == "gguf"


def test_stdout_rows_are_strict_json(stub_run, tmp_path, capsys):
    stub_run("--out", str(tmp_path / "b.json"))
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("{"))
    assert "NaN" not in line


def test_interrupted_run_keeps_the_finished_modes(stub_run, tmp_path, monkeypatch):
    out = tmp_path / "bench.json"
    built = []

    def build(mode, *a, **k):
        built.append(mode)
        if len(built) > 1:
            raise RuntimeError("stop")
        return _Be()
    monkeypatch.setattr(bench, "build", build)
    with pytest.raises(RuntimeError):  # the second mode dies; the first one is already in the file
        stub_run("fast", "--out", str(out))
    assert [r["mode"] for r in json.loads(out.read_text(encoding="utf-8"))] == ["gguf"]
