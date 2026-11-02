"""scripts/render_batch.py resume rule: a manifest record counts only when
newline-terminated, so a torn last record (even complete JSON) re-renders."""

import json

import render_batch as rb


def test_torn_manifest_record_rerenders(tmp_path):
    (tmp_path / "img").mkdir()
    for i in range(3):
        (tmp_path / "img" / f"a{i}.avif").write_bytes(b"x")
    good = "".join(json.dumps({"id": f"a{i}"}) + "\n" for i in range(2))
    (tmp_path / rb.MANIFEST).write_text(good + json.dumps({"id": "a2"}))
    assert rb.done_ids(tmp_path) == {"a0", "a1"}
    rb.repair_manifest(tmp_path)
    assert (tmp_path / rb.MANIFEST).read_text() == good
    assert rb.done_ids(tmp_path) == {"a0", "a1"}
