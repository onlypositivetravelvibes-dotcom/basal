"""basal-bench --questions: a JSONL file saved with a UTF-8 byte order mark (Windows editors, PowerShell redirection) loads."""
import json

from basal import bench


def test_load_questions_accepts_bom(tmp_path):
    item = {"state": "s", "question": "q", "options": ["a", "b"], "gold": 0}
    f = tmp_path / "q.jsonl"
    f.write_bytes(b"\xef\xbb\xbf" + (json.dumps(item) + "\n").encode("utf-8"))
    assert bench.load_questions(f, 10) == [item]


def test_load_questions_plain_utf8(tmp_path):
    item = {"state": "s", "question": "zażółć", "options": ["a", "b"]}
    f = tmp_path / "q.jsonl"
    f.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    assert bench.load_questions([f], 10) == [item]
