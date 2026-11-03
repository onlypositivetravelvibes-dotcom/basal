"""basal-bench and basal-bench --order-check feed a model trained with isolated options (basal.json
"option_isolation": true) the prompts as piece-by-piece token ids (OptionIds), as basal-serve does; its backend asserts
on a plain string."""
import json
import sys

import pytest

from basal import bench, ordercheck
from basal.prompt import OptionIds

QS = [{"state": "Customer wants a refund.", "question": "Dept?", "options": ["Refunds", "IT", "HR"], "gold": 0},
      {"state": "Customer wants a refund.", "question": "Team?", "options": ["Alpha", "Beta"], "gold": 1}]


class Tok:  # character-level stand-in for the tokenizer
    def apply_chat_template(self, msgs, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        ids = [[ord(c) for c in s] for s in text] if isinstance(text, list) else [ord(c) for c in text]
        return type("E", (), {"input_ids": ids})()


class Recorder:
    tok = Tok()

    def __init__(self):
        self.prompts = []

    def run_shared(self, groups, policy=None):
        for g in groups:
            self.prompts += list(g[0])
        return [[[1 / len(ids)] * len(ids) for ids in g[1]] for g in groups]


def model(tmp_path, **fields):
    (tmp_path / "basal.json").write_text(json.dumps(dict(prompt=2, **fields)), encoding="utf-8")
    return tmp_path


def bench_prompts(monkeypatch, md):
    be = Recorder()
    monkeypatch.setattr(bench, "resolve", lambda *a, **k: md)
    monkeypatch.setattr(bench, "load_questions", lambda *a, **k: QS * 4)
    monkeypatch.setattr(bench, "reset_memory", lambda: None)
    monkeypatch.setattr(bench, "build", lambda *a, **k: be)
    monkeypatch.setattr(bench, "device_name", lambda be=None: "stub")
    monkeypatch.setattr(bench, "memory_gb", lambda be: 0.0)
    monkeypatch.setattr(bench, "measure", lambda be_, g, n: (
        [[0.5, 0.5]] * len(g), (be.run_shared([(x[2], x[3]) for x in g]), dict(
            lat1_ms=1.0, lat2_ms=2.0, lat2_p95_ms=3.0, dec_s=4.0))[1]))
    monkeypatch.setattr(sys, "argv", ["basal-bench", "--modes", "fast"])
    bench.main()
    return be.prompts


def order_check_prompts(monkeypatch, md):
    be = Recorder()
    monkeypatch.setattr(ordercheck, "resolve", lambda *a, **k: md)
    monkeypatch.setattr(bench, "build", lambda *a, **k: be)
    monkeypatch.setattr(bench, "device_name", lambda be=None: "stub")
    monkeypatch.setattr(bench, "load_questions", lambda *a, **k: QS)
    ordercheck.main(["--mode", "fast", "--average", "2", "--lat-n", "0"])
    return be.prompts


@pytest.mark.parametrize("get", [bench_prompts, order_check_prompts])
def test_isolated_options_reach_the_backend_as_option_ids(monkeypatch, tmp_path, get):
    prompts = get(monkeypatch, model(tmp_path, option_isolation=True))
    assert prompts and all(isinstance(p, OptionIds) and len(p.blocks) >= 4 for p in prompts)  # prefix, options, suffix
    assert all(sum(p.blocks) == len(p) for p in prompts)


@pytest.mark.parametrize("get", [bench_prompts, order_check_prompts])
def test_an_ordinary_model_still_gets_strings(monkeypatch, tmp_path, get):
    prompts = get(monkeypatch, model(tmp_path))
    assert prompts and all(type(p) is str for p in prompts)
