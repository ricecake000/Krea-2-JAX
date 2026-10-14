"""CPU tests of scripts/fetch_originals.py with a fake `rclone` (a Python
script that serves local fake orig-XXXX.tar / .parquet files) — no network:

  * id inputs: json list, actions export, txt; review page tabs A/B/C/E
  * span grouping (gap merging) and offsets from the index sizes
  * range mode: only the wanted members are written (byte-identical, no
    .part left), ranged reads only, an index row with status != ok is
    skipped in the offsets, ids not in any index are reported (exit 1),
    a tar whose index order disagrees with the tar falls back to stream
  * stream mode: same files; reading stops once a tar's wanted ids are
    written (rclone killed before the end of a large tar); a stream cut
    at a member boundary is an error, not a short tar
  * resume: a second run reads nothing; a truncated file is refetched
  * a failing read is retried; a second concurrent run is refused

    python tests/test_fetch_originals.py   # or: pytest tests/test_fetch_originals.py
"""

import fcntl
import io
import json
import os
import random
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import fetch_originals as fo  # noqa: E402

FAKE_RCLONE = r'''#!__PY__
import fnmatch, os, shutil, sys, time
root = os.environ["FAKE_RCLONE_ROOT"]
def local(p):
    assert p.startswith("fake:"), p
    return os.path.join(root, p[len("fake:"):].lstrip("/"))
args = sys.argv[1:]
with open(os.environ["FAKE_RCLONE_LOG"], "a") as f:
    f.write(" ".join(args) + "\n")
cmd = args.pop(0)
opts, pos = {}, []
while args:
    a = args.pop(0)
    if a.startswith("--"):
        opts[a] = args.pop(0)
    else:
        pos.append(a)
if cmd == "lsf":
    d = local(pos[0])
    for n in sorted(os.listdir(d)):
        print(n + opts.get("--separator", ";") + str(os.path.getsize(os.path.join(d, n))))
elif cmd == "copy":
    src, dst = local(pos[0]), pos[1]
    os.makedirs(dst, exist_ok=True)
    for n in sorted(os.listdir(src)):
        if fnmatch.fnmatch(n, opts["--include"]):
            shutil.copy(os.path.join(src, n), os.path.join(dst, n))
elif cmd == "cat":
    path = local(pos[0])
    name = os.path.basename(path)
    flaky = os.environ.get("FAKE_RCLONE_FLAKY", "")
    marker = os.path.join(root, ".flaked-" + name)
    out = sys.stdout.buffer
    with open(path, "rb") as f:
        f.seek(int(opts.get("--offset", 0)))
        left = int(opts.get("--count", -1))
        if name == flaky and not os.path.exists(marker):
            open(marker, "w").close()
            out.write(f.read(700)); out.flush()
            sys.exit(1)
        if name == os.environ.get("FAKE_RCLONE_CUT", ""):
            out.write(f.read(int(os.environ["FAKE_RCLONE_CUT_AT"]))); out.flush()
            sys.exit(3)
        while left != 0:
            b = f.read(65536 if left < 0 else min(65536, left))
            if not b:
                break
            out.write(b)
            left -= len(b) if left > 0 else 0
        out.flush()
    with open(os.environ["FAKE_RCLONE_LOG"], "a") as f:
        f.write("complete " + name + "\n")
else:
    sys.exit("fake rclone: unknown command " + cmd)
'''


def payload(i, size):
    r = random.Random(i)
    return bytes(r.getrandbits(8) for _ in range(size))


def write_tar(path, members):
    """members: [(id, size)] in tar order (plain ustar, like the archive)."""
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tf:
        for i, size in members:
            ti = tarfile.TarInfo(f"{i}.webp")
            ti.size = size
            ti.mtime = 1700000000
            tf.addfile(ti, io.BytesIO(payload(i, size)))


def write_index(path, rows):
    """rows: [(id, size, status)] in index order."""
    pq.write_table(pa.table({
        "id": [r[0] for r in rows], "rank": list(range(len(rows))),
        "score": [1.0] * len(rows), "size": [r[1] for r in rows],
        "ext": ["webp"] * len(rows), "w": [100] * len(rows),
        "h": [100] * len(rows), "artist": [""] * len(rows),
        "copyright": [""] * len(rows), "status": [r[2] for r in rows],
        "error": [None] * len(rows)}), path)


def sizes(base, n, seed):
    r = random.Random(seed)
    return [(base + k, r.randint(300, 3000)) for k in range(n)]


def make_world(tmp):
    """Fake archive:
    orig-0000: 100..119 (all ok, index order == tar order)
    orig-0001: 200..219 + index row 299 with status 'error' (not in tar)
    orig-0002: 300..309, index order swaps 302/303 -> range mismatch
    orig-0003: 400..401 small + 402 large (2 MiB) + 403: stream early stop
    """
    arch = tmp / "arch" / "orig"
    arch.mkdir(parents=True)
    m0 = sizes(100, 20, 0)
    write_tar(arch / "orig-0000.tar", m0)
    write_index(arch / "orig-0000.parquet", [(i, s, "ok") for i, s in m0])
    m1 = sizes(200, 20, 1)
    write_tar(arch / "orig-0001.tar", m1)
    rows1 = [(i, s, "ok") for i, s in m1]
    rows1.insert(5, (299, 0, "error"))
    write_index(arch / "orig-0001.parquet", rows1)
    m2 = sizes(300, 10, 2)
    write_tar(arch / "orig-0002.tar", m2)
    rows2 = [(i, s, "ok") for i, s in m2]
    rows2[2], rows2[3] = rows2[3], rows2[2]
    write_index(arch / "orig-0002.parquet", rows2)
    m3 = [(400, 1000), (401, 1200), (402, 2 << 20), (403, 900)]
    write_tar(arch / "orig-0003.tar", m3)
    write_index(arch / "orig-0003.parquet", [(i, s, "ok") for i, s in m3])
    fake = tmp / "bin" / "rclone"
    fake.parent.mkdir()
    fake.write_text(FAKE_RCLONE.replace("__PY__", sys.executable))
    fake.chmod(0o755)
    os.environ["FAKE_RCLONE_ROOT"] = str(tmp / "arch")
    os.environ["FAKE_RCLONE_LOG"] = str(tmp / "rclone.log")
    for k in ("FAKE_RCLONE_FLAKY", "FAKE_RCLONE_CUT", "FAKE_RCLONE_CUT_AT"):
        os.environ.pop(k, None)
    every = {i: s for m in (m0, m1, m2, m3) for i, s in m}
    return fake, every


def run(tmp, fake, ids, *extra):
    ids_file = tmp / "ids.json"
    ids_file.write_text(json.dumps(ids))
    return fo.main(["--ids", str(ids_file), "--remote", "fake:orig",
                    "--cache", str(tmp / "cache"), "--rclone", str(fake),
                    "--log-every", "0", *extra])


def rclone_log(tmp):
    p = tmp / "rclone.log"
    return p.read_text().splitlines() if p.exists() else []


def cats(tmp):
    return [line for line in rclone_log(tmp) if line.startswith("cat ")]


def check_files(tmp, every, ids):
    img = tmp / "cache" / "img"
    assert sorted(p.name for p in img.iterdir()) == sorted(
        f"{i}.webp" for i in ids), sorted(p.name for p in img.iterdir())
    for i in ids:
        assert (img / f"{i}.webp").read_bytes() == payload(i, every[i]), i


WANT = [101, 102, 110, 119, 200, 204, 206, 218, 301, 303, 309, 400]
GOT = WANT
MISSING = [299, 999]   # status 'error' / in no index


# ------------------------------------------------------------------ units
def test_read_ids():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.json").write_text("[3, 1, 2]")
        (d / "b.json").write_text(json.dumps(
            {"review_id": "x", "items": [{"id": 7, "action": "crop"},
                                         {"id": 8, "action": "remove"}]}))
        (d / "c.txt").write_text("# ids\n5\n\n6  # six\n")
        assert fo.read_ids(d / "a.json") == [3, 1, 2]
        assert fo.read_ids(d / "b.json") == [7, 8]
        assert fo.read_ids(d / "c.txt") == [5, 6]


def test_review_ids():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "review").mkdir()
        data = {"tabs": {"A": [1, 2], "B": [2, 3], "C": [4], "D": [5],
                         "E": [6]}}
        (d / "review" / "index.html").write_text(
            '<html><script id="data" type="application/json">'
            + json.dumps(data) + "</script></html>")
        assert sorted(set(fo.review_ids(d))) == [1, 2, 3, 4, 6]


def test_spans_and_offsets():
    rows = [(1, "1.webp", 100, 0), (2, "2.webp", 600, 1024),
            (3, "3.webp", 512, 2560), (4, "4.webp", 10, 3584)]
    assert fo.spans_of(rows, {1, 3}, gap=0) == [
        (0, 1024, rows[:1]), (2560, 3584, rows[2:3])]
    assert fo.spans_of(rows, {1, 3}, gap=1536) == [(0, 3584, rows[:3])]
    assert fo.spans_of(rows, {4}, gap=0) == [(3584, 4608, rows[3:])]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_tar(d / "orig-0000.tar", [(1, 100), (2, 600), (3, 512)])
        write_index(d / "orig-0000.parquet",
                    [(1, 100, "ok"), (9, 0, "error"), (2, 600, "ok"),
                     (3, 512, "ok")])
        idx = fo.load_index(d, ["orig-0000"])["orig-0000"]
        with tarfile.open(d / "orig-0000.tar") as tf:
            real = [(int(m.name.split(".")[0]), m.name, m.size, m.offset)
                    for m in tf]
        assert idx == real, (idx, real)


# ------------------------------------------------------------------ runs
def test_range_mode():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fake, every = make_world(tmp)
        rc = run(tmp, fake, WANT + MISSING, "--gap", "0")
        assert rc == 1
        check_files(tmp, every, GOT)
        rep = json.loads((tmp / "cache" / "fetch_report.json").read_text())
        assert rep["missing_from_index"] == MISSING, rep
        assert rep["failed"] == {}, rep
        assert rep["stream_fallback_tars"] == ["orig-0002"], rep
        assert rep["fetched"] == len(GOT)
        cs = cats(tmp)
        ranged = [c for c in cs if "--offset" in c]
        whole = [c for c in cs if "--offset" not in c]
        assert all("orig-0002" in c for c in whole) and whole, cs
        # orig-0000 / 0001: one ranged read per wanted member (gap 0, none
        # adjacent except 101/102)
        assert sum("orig-0000" in c for c in ranged) == 3, ranged
        assert sum("orig-0001" in c for c in ranged) == 4, ranged
        assert not list((tmp / "cache" / "img").glob("*.part"))


def test_range_merges_and_resume():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fake, every = make_world(tmp)
        assert run(tmp, fake, WANT) == 0
        check_files(tmp, every, GOT)
        cs = cats(tmp)
        assert sum("orig-0000" in c for c in cs) == 1, cs   # default gap
        n = len(rclone_log(tmp))
        assert run(tmp, fake, WANT) == 0           # resume: nothing to read
        assert not cats(tmp)[len(cs):], rclone_log(tmp)[n:]
        p = tmp / "cache" / "img" / "204.webp"
        p.write_bytes(p.read_bytes()[:10])         # truncated -> refetched
        assert run(tmp, fake, WANT) == 0
        new = cats(tmp)[len(cs):]
        assert len(new) == 1 and "orig-0001" in new[0], new
        check_files(tmp, every, GOT)


def test_stream_mode_and_early_stop():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fake, every = make_world(tmp)
        rc = run(tmp, fake, WANT + MISSING, "--mode", "stream", "--jobs", "2")
        assert rc == 1
        check_files(tmp, every, GOT)
        cs = cats(tmp)
        assert len(cs) == 4 and not any("--offset" in c for c in cs), cs
        done = [line for line in rclone_log(tmp) if line.startswith("complete")]
        # 400 is the first member of orig-0003; the 2 MiB member after it is
        # never read to the end (rclone killed once 400 was written)
        assert "complete orig-0003.tar" not in done, done


def test_stream_cut_is_error_and_retry():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fake, every = make_world(tmp)
        # cut orig-0000 right after its first member (a member boundary):
        # tarfile alone would call that a clean end; must be an error
        os.environ["FAKE_RCLONE_CUT"] = "orig-0000.tar"
        os.environ["FAKE_RCLONE_CUT_AT"] = str(512 + fo.padded(every[100]))
        rc = run(tmp, fake, [100, 101], "--mode", "stream")
        assert rc == 1
        rep = json.loads((tmp / "cache" / "fetch_report.json").read_text())
        assert rep["failed"] == {"101": "orig-0000: stream failed"}, rep
        os.environ.pop("FAKE_RCLONE_CUT")
        os.environ["FAKE_RCLONE_FLAKY"] = "orig-0001.tar"
        assert run(tmp, fake, [100, 101, 210]) == 0     # range, one retry
        check_files(tmp, every, [100, 101, 210])
        assert sum("orig-0001" in c for c in cats(tmp)) == 2


def test_lock_and_dry_run():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fake, every = make_world(tmp)
        assert run(tmp, fake, WANT, "--dry-run") == 0
        assert not list((tmp / "cache" / "img").iterdir())
        assert len(list((tmp / "cache" / "index").iterdir())) == 4
        with open(tmp / "cache" / "fetch.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(tmp, fake, WANT)
            except SystemExit as e:
                assert "another fetch_originals run" in str(e)
            else:
                raise AssertionError("second run not refused")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
