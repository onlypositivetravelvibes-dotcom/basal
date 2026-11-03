"""basal-bench on a small sample: measure() must not crash when fewer questions are timed than the 5 warm-up ones."""
import pytest

import basal.bench as bench_module


class Be:
    def run(self, prompts, ids, policy=None):
        return [[1.0]] * len(prompts)

    def run_shared(self, groups, policy=None):
        return [[[0.5, 0.5]] * len(g[0]) for g in groups]


def groups(n):
    return [({"options": ["a", "b"]}, [[0, 1], [1, 0]], ["p0", "p1"], [[1, 2], [1, 2]]) for _ in range(n)]


@pytest.mark.parametrize("n", [1, 3, 5, 6, 8])
@pytest.mark.parametrize("lat_n", [-2, 0, 1, 100])
def test_measure_small_sample(monkeypatch, n, lat_n):
    monkeypatch.setattr(bench_module, "sync", lambda be: None)
    dec, r = bench_module.measure(Be(), groups(n), lat_n)  # main() passes min(--lat-n, len(groups) - 5), possibly <= 0
    assert len(dec) == n
    assert 0 <= r["lat2_ms"] <= r["lat2_p95_ms"]
    assert r["lat1_ms"] >= 0
