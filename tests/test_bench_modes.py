"""basal-bench refuses an unknown --modes entry before it loads or measures anything."""
import sys
from unittest import mock

import pytest

try:
    import torch  # noqa: F401
    from basal import bench
except ImportError:  # no torch / transformers: stub the heavy imports, the checked logic is pure Python
    for _name in ("torch", "torch.nn", "torch.nn.functional", "transformers"):
        sys.modules.setdefault(_name, mock.MagicMock())
    from basal import bench


@pytest.mark.parametrize("mode", ["eager-fp32", "fast", "tpu", "tpu-fp32", "gguf", "gguf@m.gguf", "vllm@deterministic",
                                  "fast-exit", "fast-exit@0.99", "strix"])
def test_known_modes(mode):
    assert bench.unknown_mode(mode) is None


@pytest.mark.parametrize("mode", ["fats", "gguf2@x", "eager-fp32@x", "fast@x", ""])
def test_unknown_modes(mode):
    assert "unknown mode" in bench.unknown_mode(mode)


def test_typo_refused_before_any_load(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["basal-bench", "--modes", "eager-fp32", "fats"])
    loaded = []
    monkeypatch.setattr(bench, "resolve", lambda *a, **k: loaded.append("resolve"))
    monkeypatch.setattr(bench, "build", lambda *a, **k: loaded.append("build"))
    with pytest.raises(SystemExit) as e:
        bench.main()
    assert "unknown mode 'fats'" in str(e.value) and "fast" in str(e.value)
    assert loaded == []
