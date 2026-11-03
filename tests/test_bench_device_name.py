"""device_name() on a CPU host: a sysctl that exits non-zero with an empty stdout (Linux, procps) must fall back to
platform.processor(), not return "". torch is stubbed: only the function's own logic is run."""
import ast
import platform
import subprocess
import sys
import types
from pathlib import Path

import pytest


def load_device_name():
    src = (Path(__file__).parent.parent / "basal" / "bench.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "device_name")
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    ns = {"torch": torch, "sys": sys}
    exec(compile(ast.Module([fn], []), "bench.py", "exec"), ns)
    return ns["device_name"]


def fake_run(stdout, rc):
    return lambda *a, **k: subprocess.CompletedProcess(a, rc, stdout=stdout, stderr="")


@pytest.mark.parametrize("plat", ["linux", "win32", "darwin"])
def test_failed_sysctl_falls_back(monkeypatch, plat):
    monkeypatch.setattr(sys, "platform", plat)
    monkeypatch.setattr(subprocess, "run", fake_run("", 1))
    monkeypatch.setattr(platform, "processor", lambda: "x86_64")
    assert load_device_name()() == "x86_64"


def test_everything_empty_gives_cpu(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", fake_run("", 1))
    monkeypatch.setattr(platform, "processor", lambda: "")
    assert load_device_name()() == "cpu"


def test_macos_brand_string(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", fake_run("Apple M5\n", 0))
    assert load_device_name()() == "Apple M5"
