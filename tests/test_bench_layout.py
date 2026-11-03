"""basal-bench and basal-bench --order-check build their prompts in the layout the model is served in (the "layout" of
its basal.json), not always in the canonical one."""
import json
import sys

import pytest

from basal import bench, ordercheck
from basal.prompt import render

QS = [{"state": "Customer wants a refund.", "question": "Dept?", "options": ["Refunds", "IT", "HR"], "gold": 0},
      {"state": "Customer wants a refund.", "question": "Team?", "options": ["Alpha", "Beta"], "gold": 1}]


class Tok:  # character-level stand-in for the tokenizer
    def apply_chat_template(self, msgs, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        ids = [[ord(c) for c in s] for s in text] if isinstance(text, list) else [ord(c) for c in text]
        return type("E", (), {"input_ids": ids})()


def expected(layout):
    out = []
    for q in QS:
        k = len(q["options"])
        out += [render(Tok(), q["state"], q["question"], [q["options"][c] for c in p], fmt=2, layout=layout)
                for p in (list(range(k)), list(range(k))[::-1])]
    return out


@pytest.fixture
def model_dir(tmp_path):
    def make(**fields):
        (tmp_path / "basal.json").write_text(json.dumps(dict(prompt=2, **fields)), encoding="utf-8")
        return tmp_path
    return make


class Recorder:
    tok = Tok()

    def __init__(self):
        self.prompts = []

    def run_shared(self, groups, policy=None):
        for g in groups:
            self.prompts += list(g[0])
        return [[[1 / len(ids)] * len(ids) for ids in g[1]] for g in groups]


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
    return be.prompts[:len(expected("canonical"))]


def order_check_prompts(monkeypatch, md):
    be = Recorder()
    monkeypatch.setattr(ordercheck, "resolve", lambda *a, **k: md)
    monkeypatch.setattr(bench, "build", lambda *a, **k: be)
    monkeypatch.setattr(bench, "device_name", lambda be=None: "stub")
    monkeypatch.setattr(bench, "load_questions", lambda *a, **k: QS)
    ordercheck.main(["--mode", "fast", "--average", "2", "--lat-n", "0"])
    return be.prompts


@pytest.mark.parametrize("layout", ["qfirst", "state_twice"])
def test_bench_uses_the_model_layout(monkeypatch, model_dir, layout):
    assert expected(layout) != expected("canonical")
    assert bench_prompts(monkeypatch, model_dir(layout=layout)) == expected(layout)


def test_bench_without_layout_stays_canonical(monkeypatch, model_dir):
    assert bench_prompts(monkeypatch, model_dir()) == expected("canonical")


def test_order_check_uses_the_model_layout(monkeypatch, model_dir):
    prompts = order_check_prompts(monkeypatch, model_dir(layout="qfirst"))
    assert prompts and all("<user>Question:" in p for p in prompts)  # qfirst opens the user turn with the question
    assert not any("<user>Question:" in p for p in expected("canonical"))
