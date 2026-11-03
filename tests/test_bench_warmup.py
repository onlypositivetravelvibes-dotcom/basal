"""basal-bench measure(): the throughput pass is timed on shapes that were already used, so a backend that compiles
per shape (TPU) does not charge its compilation to dec/s."""
from basal import bench


class Compiling:
    """Stand-in backend that compiles on the first use of a (batch size, longest prompt) shape and counts the compiles
    that happen while a timed region runs."""

    def __init__(self):
        self.seen, self.timing, self.cold_compiles = set(), False, 0

    def _shape(self, groups):
        return len(groups), max(len(p) for g in groups for p in g[0])

    def run_shared(self, groups, policy=None):
        shape = self._shape(groups)
        if shape not in self.seen:
            self.seen.add(shape)
            self.cold_compiles += self.timing
        return [[[0.5, 0.5] for _ in g[0]] for g in groups]

    def run(self, prompts, ids):
        return [[0.5, 0.5] for _ in prompts]


def groups(n=128):
    out = []
    for i in range(n):
        p = "x" * (10 + i // 16)  # every chunk of 16 questions has its own prompt length, so its own shape
        out.append(({"options": ["a", "b"]}, [[0, 1], [1, 0]], [p, p], [[1, 2], [1, 2]]))
    return out


def test_throughput_pass_meets_no_new_shape(monkeypatch):
    be = Compiling()
    real = bench.timed

    def timed(fn, reps, be_=None):
        be.timing = True
        try:
            return real(fn, reps, be_)
        finally:
            be.timing = False

    monkeypatch.setattr(bench, "timed", timed)
    bench.measure(be, groups(), 6)
    assert be.cold_compiles == 0
