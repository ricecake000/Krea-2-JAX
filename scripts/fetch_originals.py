"""Fetch the ORIGINAL images of selected dataset ids from the Google Drive
originals archive (rclone remote), writing only the wanted members to disk.

    python scripts/fetch_originals.py --ids-from-review /home/mj/krea2-ops/eval/datafilter
    python scripts/fetch_originals.py --ids ids.json          # json list or txt (one id per line)
    python scripts/fetch_originals.py --ids ids.txt --mode stream --jobs 2

Archive layout (`--remote`, default gdrive:danbooru_pipeline/selection_v2/
originals): `orig-XXXX.tar` (members `<id>.<ext>`, plain ustar) and
`orig-XXXX.parquet` (index: id, rank, score, size, ext, w, h, artist,
copyright, status, error) whose ok rows are in the tar's member order.

1. Every `orig-*.parquet` index is copied to `<cache>/index/` (rclone copy:
   unchanged files are skipped). A tar without its index fails the run.
2. Wanted ids are mapped to (tar, member, size) via the indexes (ok rows
   only; an id in several tars is taken from the first). Ids not found ->
   listed as missing, exit status 1 at the end (the rest is still fetched).
3. Ids already present as `<cache>/img/<id>.<ext>` with the index size are
   skipped, so re-running the same command resumes.
4. `--mode range` (default): a member's byte offset in the tar follows from
   the sizes of the ok index rows before it (512-byte header + data padded to
   512). Wanted members of a tar are grouped into spans (members closer than
   `--gap` bytes share one read) and each span is read with
   `rclone cat --offset O --count N`, `--range-jobs` spans at a time. Every
   tar header in a span is checked against the index (name and size); any
   mismatch makes that tar fall back to stream mode.
   `--mode stream`: `rclone cat` of the whole tar into tarfile `r|`,
   `--jobs` tars at a time; reading stops (rclone killed) as soon as every
   wanted member of the tar is written.
5. Each file is written to `<id>.<ext>.part`, size-checked against the index
   and renamed into place. A per-run report `<cache>/fetch_report.json`
   lists missing / failed ids.

Never stores a whole tar; memory is a few MiB per concurrent read.
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REMOTE = "gdrive:danbooru_pipeline/selection_v2/originals"
CACHE = Path("/home/mj/.cache/krea2-orig")
DEFAULT_RCLONE = "/home/mj/.local/bin/rclone"
REVIEW_TABS = ("A", "B", "C", "E")   # the tabs the user may act on (D = control)
BLOCK = 512
CHUNK = 1 << 20
# Measured on the laptop (10/14): a ranged rclone cat costs ~6 s before the
# first byte, calls overlap freely, the link tops out at ~14.6 MB/s in total.
# With 16 calls in flight a call's latency is worth ~5 MiB of transfer.
GAP = 4 << 20
RETRIES = 3
TAR_RE = re.compile(r"^(orig-\d+)\.tar$")
IDX_RE = re.compile(r"^(orig-\d+)\.parquet$")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def padded(size):
    return (size + BLOCK - 1) // BLOCK * BLOCK


# ------------------------------------------------------------------ ids
def read_ids(path):
    """A json list (or {"items": [{"id": ..}]} actions export) or a text file
    with one id per line (blank lines and #-comments ignored)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if data is not None:
        if isinstance(data, dict):
            data = [it["id"] for it in data.get("items", [])]
        return [int(i) for i in data]
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(int(line))
    return out


def review_ids(review_dir, tabs=REVIEW_TABS):
    """Every id in the given tabs of DIR/review/index.html."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from modal_score_borders import page_data
    page = Path(review_dir) / "review" / "index.html"
    data = page_data(page)
    if data is None:
        raise SystemExit(f"{page}: not found")
    ids = []
    for t in tabs:
        if t not in data["tabs"]:
            raise SystemExit(f"{page}: no tab {t}")
        ids += [int(i) for i in data["tabs"][t]]
    return ids


# ------------------------------------------------------------------ rclone
def rclone_listing(rclone, remote):
    """{name: size} of the archive directory."""
    out = subprocess.run([rclone, "lsf", "--format", "ps", "--separator", "\t",
                          remote], check=True, capture_output=True, text=True)
    files = {}
    for line in out.stdout.splitlines():
        if line.strip():
            name, size = line.rsplit("\t", 1)
            files[name] = int(size)
    return files


def sync_indexes(rclone, remote, index_dir):
    """Copy every orig-*.parquet to index_dir; return {tar stem: tar size}."""
    files = rclone_listing(rclone, remote)
    tars = {TAR_RE.match(n).group(1): s for n, s in files.items()
            if TAR_RE.match(n)}
    idx = {IDX_RE.match(n).group(1) for n in files if IDX_RE.match(n)}
    if not tars:
        raise SystemExit(f"{remote}: no orig-*.tar")
    if set(tars) - idx:
        raise SystemExit(f"tars without an index: {sorted(set(tars) - idx)}")
    index_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([rclone, "copy", "--include", "orig-*.parquet", remote,
                    str(index_dir)], check=True)
    for stem in tars:
        p = index_dir / f"{stem}.parquet"
        if not p.is_file() or p.stat().st_size != files[f"{stem}.parquet"]:
            raise SystemExit(f"{p}: index copy incomplete")
    return tars


# ------------------------------------------------------------------ plan
def load_index(index_dir, stems):
    """{stem: [(id, member name, size, offset), ...]} for the ok rows in
    order, offsets assuming one plain ustar header per member."""
    import pyarrow.parquet as pq
    out = {}
    for stem in sorted(stems):
        t = pq.read_table(index_dir / f"{stem}.parquet",
                          columns=["id", "size", "ext", "status"]).to_pydict()
        rows, off, seen = [], 0, set()
        for i, size, ext, status in zip(t["id"], t["size"], t["ext"],
                                        t["status"]):
            if status != "ok":
                continue
            if int(i) in seen:   # two members, one output file: ambiguous
                raise SystemExit(f"{stem}: duplicate id {int(i)} in the index")
            seen.add(int(i))
            rows.append((int(i), f"{int(i)}.{ext}", int(size), off))
            off += BLOCK + padded(int(size))
        out[stem] = rows
    return out


def plan_fetch(index, wanted):
    """-> (jobs {stem: {id: (name, size, offset)}}, missing ids, duplicates)."""
    where, dups = {}, 0
    for stem in sorted(index):
        for i, name, size, off in index[stem]:
            if i in where:
                dups += 1
                continue
            where[i] = (stem, name, size, off)
    jobs, missing = {}, []
    for i in sorted(set(wanted)):
        if i not in where:
            missing.append(i)
            continue
        stem, name, size, off = where[i]
        jobs.setdefault(stem, {})[i] = (name, size, off)
    return jobs, missing, dups


def spans_of(rows, want, gap=GAP):
    """Group the wanted members of one tar (index rows in tar order) into
    contiguous spans: [(start, end, [rows in the span])], merging two wanted
    members when the bytes between them are <= gap. want = {id: (name, size,
    offset)} from plan_fetch; only the row at the planned offset counts."""
    spans = []
    for k, (i, name, size, off) in enumerate(rows):
        if i not in want or want[i][2] != off:
            continue
        end = off + BLOCK + padded(size)
        if spans and off - spans[-1][1] <= gap:
            spans[-1][1] = end
            spans[-1][3] = k + 1
        else:
            spans.append([off, end, k, k + 1])
    return [(s, e, rows[a:b]) for s, e, a, b in spans]


# ------------------------------------------------------------------ writing
class Progress:
    def __init__(self, n_files, n_bytes):
        self.lock = threading.Lock()
        self.n_files, self.n_bytes = n_files, n_bytes
        self.files = self.read = 0
        self.t0 = time.time()

    def add(self, files=0, read=0):
        with self.lock:
            self.files += files
            self.read += read

    def line(self):
        with self.lock:
            dt = max(time.time() - self.t0, 1e-6)
            rate = self.read / dt
            left = max(self.n_bytes - self.read, 0)
            eta = left / rate if rate > 0 else float("nan")
            return (f"files {self.files}/{self.n_files}  read "
                    f"{self.read / 2**20:.0f}/{self.n_bytes / 2**20:.0f} MiB  "
                    f"{rate / 2**20:.1f} MiB/s  ETA {eta / 60:.1f} min")


def write_member(src, size, dest, progress):
    """Copy exactly `size` bytes from a file-like into dest via .part."""
    part = dest.with_name(dest.name + ".part")
    left = size
    with open(part, "wb") as f:
        while left:
            b = src.read(min(CHUNK, left))
            if not b:
                break
            f.write(b)
            left -= len(b)
            progress.add(read=len(b))
    if part.stat().st_size != size:
        part.unlink()
        raise IOError(f"{dest.name}: short read ({size - left}/{size})")
    os.replace(part, dest)
    progress.add(files=1)


def read_exact(f, n, progress):
    buf = bytearray()
    while len(buf) < n:
        b = f.read(n - len(buf))
        if not b:
            break
        buf += b
    progress.add(read=len(buf))
    return bytes(buf)


def skip(f, n, progress):
    while n:
        b = f.read(min(CHUNK, n))
        if not b:
            raise IOError("unexpected end of stream")
        n -= len(b)
        progress.add(read=len(b))


class Mismatch(Exception):
    """A tar header disagrees with the index (range offsets unusable)."""


def kill(proc):
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    proc.stdout.close()


def fetch_span(rclone, remote, stem, span, want, img_dir, progress):
    """Read one span with a ranged rclone cat; write its wanted members
    (want = {id: planned offset}; a row elsewhere is only skipped)."""
    start, end, rows = span
    cmd = [rclone, "cat", "--offset", str(start), "--count", str(end - start),
           f"{remote}/{stem}.tar"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        for i, name, size, off in rows:
            hdr = read_exact(proc.stdout, BLOCK, progress)
            if len(hdr) < BLOCK:
                raise IOError(f"{stem}: short read at {off}")
            try:
                ti = tarfile.TarInfo.frombuf(hdr, "utf-8", "surrogateescape")
            except tarfile.TarError as e:
                raise Mismatch(f"{stem}@{off}: bad header ({e})")
            if ti.name != name or ti.size != size or not ti.isreg():
                raise Mismatch(f"{stem}@{off}: header {ti.name}/{ti.size}, "
                               f"index {name}/{size}")
            if want.get(i) == off:
                write_member(proc.stdout, size, img_dir / name, progress)
                skip(proc.stdout, padded(size) - size, progress)
            else:
                skip(proc.stdout, padded(size), progress)
    finally:
        kill(proc)   # every member was checked; rclone's own exit is moot


def fetch_stream(rclone, remote, stem, want, img_dir, progress):
    """Stream the whole tar; stop once every wanted member is written.
    Returns the set of wanted ids never seen."""
    todo = dict(want)
    by_name = {name: (i, size) for i, (name, size, _) in todo.items()}
    proc = subprocess.Popen([rclone, "cat", f"{remote}/{stem}.tar"],
                            stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            pos = 0
            for m in tf:
                progress.add(read=m.offset_data - pos)
                pos = m.offset_data + padded(m.size)
                hit = by_name.get(m.name)
                if hit is None or hit[0] not in todo:
                    progress.add(read=padded(m.size))
                    continue
                i, size = hit
                if m.size != size:
                    raise IOError(f"{stem}/{m.name}: tar size {m.size} != "
                                  f"index {size}")
                write_member(tf.extractfile(m), size, img_dir / m.name,
                             progress)
                progress.add(read=padded(size) - size)
                del todo[i]
                if not todo:
                    break
        if todo:
            # tarfile stopped at the end-of-archive blocks (rclone may still
            # be writing record padding into a full pipe: drain it, or the
            # wait below never returns) or at EOF; a stream cut at a member
            # boundary also looks like EOF, so rclone's exit status decides
            while proc.stdout.read(CHUNK):
                pass
            if proc.wait() != 0:
                raise IOError(f"{stem}: rclone exit {proc.returncode}")
    finally:
        kill(proc)
    return set(todo)


def present(img_dir, name, size):
    p = img_dir / name
    return p.is_file() and p.stat().st_size == size


def run(args):
    if args.ids_from_review:
        wanted = review_ids(args.ids_from_review)
    else:
        wanted = []
    if args.ids:
        wanted += read_ids(args.ids)
    if not wanted:
        raise SystemExit("no ids (pass --ids and/or --ids-from-review)")
    cache = Path(args.cache)
    index_dir, img_dir = cache / "index", cache / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(cache / "fetch.lock", "w") as lock_f:
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another fetch_originals run holds "
                             f"{lock_f.name}")
        return fetch(args, wanted, cache, index_dir, img_dir)


def fetch(args, wanted, cache, index_dir, img_dir):
    for p in img_dir.glob("*.part"):
        p.unlink()

    tars = sync_indexes(args.rclone, args.remote, index_dir)
    index = load_index(index_dir, tars)
    jobs, missing, dups = plan_fetch(index, wanted)
    n_want = len(set(wanted))
    have = 0
    for stem in list(jobs):
        for i, (name, size, _) in list(jobs[stem].items()):
            if present(img_dir, name, size):
                del jobs[stem][i]
                have += 1
        if not jobs[stem]:
            del jobs[stem]
    rows_of = {stem: index[stem] for stem in jobs}
    if args.mode == "range":
        spans = [(stem, sp) for stem in sorted(jobs)
                 for sp in spans_of(rows_of[stem], jobs[stem], args.gap)]
        n_bytes = sum(e - s for _, (s, e, _) in spans)
    else:
        spans = []
        n_bytes = sum(max(o + BLOCK + padded(sz)
                          for _, sz, o in jobs[stem].values())
                      for stem in jobs)
    n_files = sum(len(v) for v in jobs.values())
    log(f"wanted {n_want} ids; in index {n_want - len(missing)}, missing "
        f"{len(missing)}; already present {have}; to fetch {n_files} from "
        f"{len(jobs)} tars ({len(tars)} in the archive, {dups} duplicate "
        f"ids across tars); mode {args.mode}"
        + (f", {len(spans)} spans" if spans else "")
        + f", planned read {n_bytes / 2**30:.2f} GiB")
    if args.dry_run:
        return report(cache, n_want, have, 0, missing, {}, args, dry=True)

    progress = Progress(n_files, n_bytes)
    failed, fallback = {}, set()
    lock = threading.Lock()
    done_tars = set()

    def range_task(stem, span):
        want = {i: off for i, _, _, off in span[2]
                if i in jobs[stem] and jobs[stem][i][2] == off}
        for attempt in range(1, RETRIES + 1):
            if stem in fallback:
                return
            want = {i: off for i, off in want.items()
                    if not present(img_dir, *jobs[stem][i][:2])}
            if not want:
                return
            try:
                fetch_span(args.rclone, args.remote, stem, span, want,
                           img_dir, progress)
                return
            except Mismatch as e:
                log(f"{stem}: {e} -> stream fallback")
                with lock:
                    fallback.add(stem)
                return
            except Exception as e:  # noqa: BLE001 - retried, then reported
                log(f"{stem} span @{span[0]}: attempt {attempt}: {e}")
        with lock:
            for i in want:
                failed[i] = f"{stem}: range read failed"

    def stream_task(stem):
        want = {i: v for i, v in jobs[stem].items()
                if not present(img_dir, v[0], v[1])}
        if not want:
            return
        for attempt in range(1, RETRIES + 1):
            try:
                unseen = fetch_stream(args.rclone, args.remote, stem, want,
                                      img_dir, progress)
                with lock:
                    for i in unseen:
                        failed[i] = f"{stem}: not in tar"
                return
            except Exception as e:  # noqa: BLE001
                log(f"{stem} stream: attempt {attempt}: {e}")
                want = {i: v for i, v in want.items()
                        if not present(img_dir, v[0], v[1])}
        with lock:
            for i in want:
                failed[i] = f"{stem}: stream failed"

    last = [0.0]

    def tick(what):
        if time.time() - last[0] >= args.log_every or what.startswith("tar"):
            last[0] = time.time()
            log(f"{what}  {progress.line()}")

    if args.mode == "range":
        left = {stem: 0 for stem in jobs}
        for stem, _ in spans:
            left[stem] += 1
        with ThreadPoolExecutor(args.range_jobs) as ex:
            futs = {ex.submit(range_task, st, sp): st for st, sp in spans}
            for f in as_completed(futs):
                f.result()
                st = futs[f]
                left[st] -= 1
                if left[st] == 0:
                    done_tars.add(st)
                    tick(f"tar {st} done ({len(done_tars)}/{len(jobs)})")
                else:
                    tick("span")
    if args.mode == "stream" or fallback:
        stems = sorted(jobs) if args.mode == "stream" else sorted(fallback)
        with ThreadPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(stream_task, st): st for st in stems}
            for f in as_completed(futs):
                f.result()
                tick(f"tar {futs[f]} streamed")
    fetched = sum(present(img_dir, n, s) for v in jobs.values()
                  for n, s, _ in v.values())
    for v in jobs.values():
        for i, (n, s, _) in v.items():
            if present(img_dir, n, s):
                failed.pop(i, None)    # e.g. a failed span, then the fallback
            elif i not in failed:
                failed[i] = "not written"
    log(f"done: {progress.line()}")
    return report(cache, n_want, have, fetched, missing, failed, args,
                  fallback=sorted(fallback))


def report(cache, n_want, have, fetched, missing, failed, args, dry=False,
           fallback=()):
    rep = {"wanted": n_want, "already_present": have, "fetched": fetched,
           "missing_from_index": missing,
           "failed": {str(k): v for k, v in sorted(failed.items())},
           "stream_fallback_tars": list(fallback), "mode": args.mode,
           "dry_run": dry, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not dry:
        (Path(cache) / "fetch_report.json").write_text(json.dumps(rep,
                                                                  indent=1))
    if missing:
        log(f"MISSING from every index ({len(missing)}): {missing[:50]}"
            + (" ..." if len(missing) > 50 else ""))
    if failed:
        log(f"FAILED ({len(failed)}): {dict(list(sorted(failed.items()))[:20])}")
    ok = not missing and not failed
    if dry:
        log("dry run: nothing fetched")
    else:
        log(f"{'OK' if ok else 'INCOMPLETE'}: present {have + fetched}/"
            f"{n_want}  (report {Path(cache) / 'fetch_report.json'})")
    return 0 if ok else 1


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ids", help="json list / actions export / txt (one id per line)")
    p.add_argument("--ids-from-review", metavar="DIR",
                   help="all ids of tabs A, B, C, E of DIR/review/index.html")
    p.add_argument("--remote", default=REMOTE)
    p.add_argument("--cache", default=str(CACHE),
                   help="index/ and img/ go here (default %(default)s)")
    p.add_argument("--rclone", default=shutil.which("rclone") or DEFAULT_RCLONE)
    p.add_argument("--mode", choices=("range", "stream"), default="range")
    p.add_argument("--jobs", type=int, default=2,
                   help="tars streamed in parallel (stream mode / fallback)")
    p.add_argument("--range-jobs", type=int, default=16,
                   help="concurrent ranged reads (range mode)")
    p.add_argument("--gap", type=int, default=GAP,
                   help="range mode: merge members closer than this many bytes")
    p.add_argument("--log-every", type=float, default=30.0, help="seconds")
    p.add_argument("--dry-run", action="store_true",
                   help="sync the indexes and print the plan only")
    return p


def main(argv=None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
