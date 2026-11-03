"""README: the flags listed for `basal-bench --order-check` are the ones ordercheck.main accepts."""
import re
import sys
import types
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


class Reached(Exception):
    pass


def accepts(monkeypatch, flag, value):
    """True if `basal-bench --order-check <flag> <value>` gets past argument parsing."""
    stub = types.ModuleType("basal.bench")
    stub.DEFAULT_QUESTIONS, stub.build, stub.device_name, stub.load_questions = [], None, None, None
    server = types.ModuleType("basal.server")
    server.default_mode = lambda model: "eager"
    engine = types.ModuleType("basal.engine")
    engine.resolve = lambda model: None
    prompt = types.ModuleType("basal.prompt")
    prompt.letter_ids = prompt.model_format = prompt.render = None
    monkeypatch.delitem(sys.modules, "basal.ordercheck", raising=False)
    monkeypatch.setitem(sys.modules, "basal.engine", engine)
    monkeypatch.setitem(sys.modules, "basal.prompt", prompt)
    monkeypatch.setitem(sys.modules, "basal.bench", stub)
    monkeypatch.setitem(sys.modules, "basal.server", server)
    import importlib
    ordercheck = importlib.import_module("basal.ordercheck")

    def stop(*args, **kwargs):
        raise Reached
    monkeypatch.setattr(ordercheck, "resolve", stop)
    try:
        ordercheck.main([flag, value])
    except Reached:
        return True
    except SystemExit as e:
        return e.code == 0
    return True


def test_readme_order_check_flags(monkeypatch):
    text = README.read_text(encoding="utf-8")
    m = re.search(r"`basal-bench --order-check`((?:,\s*`--[a-z-]+(?: [A-Z]+)?`)*)", text)
    assert m, "README no longer mentions `basal-bench --order-check`"
    listed = re.findall(r"`(--[a-z-]+)", m.group(1))
    for flag in listed:
        assert accepts(monkeypatch, flag, "2"), f"README lists {flag} for basal-bench --order-check, which rejects it"
