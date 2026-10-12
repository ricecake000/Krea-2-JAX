"""Modal app: our JAX bench_step ladder (and optionally the matmul/attention
microbenchmarks) on one NVIDIA B300, side by side with the measured MI350X
numbers (stage 2 + go_tune, 10/12).

    modal run scripts/modal_bench_b300.py                 # cudnn ladder b2,4,8 + profile b4
    modal run scripts/modal_bench_b300.py --attn ref      # XLA reference attention ladder
    modal run scripts/modal_bench_b300.py --batches 4,8 --no-profile
    modal run scripts/modal_bench_b300.py --micro         # + env/matmul/attention micro benches
    modal run scripts/modal_bench_b300.py --quick         # micro benches only (no model bench)

Options (local entrypoint):
  --attn {ref,cudnn}     bench_step attention (default cudnn)
  --batches "2,4,8"      ladder batch sizes, run in order; stops at the first failure
  --micro/--no-micro     steps 1-3 (default off: they were already measured)
  --quick                micro benches only (implies --micro, no ladder)
  --profile/--no-profile after the timed ladder, ONE extra bench_step run with
                         --profile-dir (default on); timings of the ladder stay clean
  --profile-batch N      batch of the profiled run (default 4; falls back to the
                         first ladder batch that succeeded)

Steps (one B300 container, sequential):
  1. env: jax version, devices, device_kind, memory stats       (--micro only)
  2. bf16 square matmul a @ a (same method as the MI350X pod check:
     jit, 1 warmup call, 20 back-to-back reps, block on the last) (--micro only)
  3. jax.nn.dot_product_attention at our shape B=2 H=24 L=4608 D=128 bf16,
     implementation="xla" and "cudnn", fwd and fwd+bwd            (--micro only)
  4. scripts/bench_step.py --random-weights --mode fwdbwd 1024^2 txt512 remat
     dots --attn $ATTN --qchunk 512 --act-sharding on --steps 5 --warmup 1,
     each batch of --batches (subprocesses; stops at the first failure/OOM)
  5. profiled run: same command at --profile-batch with --profile-dir /tmp/prof
     (2 profiled steps + 3 timed); the perfetto trace (and xplane.pb) come back
     to the laptop
  6. markdown report B300 vs MI350X (ref and aiter reference rows)

Local outputs: /home/mj/krea2-ops/mi350x/results/modal/<YYYYmmdd-HHMMSS>/
  report.md, ladder.jsonl, perfetto_trace.json.gz (+ *.xplane.pb if small)

The repo is attached with add_local_dir (no copy): the container sees the
LOCAL checkout as it is when `modal run` starts (scripts/bench_step.py must
already have --attn cudnn / --profile-dir; checked locally before launch).

Steps 1-3 run in a child python process (this file with --micro-child) so
that their JAX client releases the GPU before bench_step.py starts.

Local CPU smoke test of steps 1-3 (no Modal, tiny sizes):
    JAX_PLATFORMS=cpu python scripts/modal_bench_b300.py --micro-child --allow-cpu \\
        --sizes 256,512 --attn-shape 1,2,256,64 --reps 2 --out /tmp/micro.json
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Literal

if __name__ == "__main__" and "--micro-child" in sys.argv:
    modal = None   # the micro child (also run inside the container) never needs Modal
else:
    try:
        import modal
    except ImportError:  # local CPU smoke tests without the Modal client
        modal = None

JAX_VERSION = "0.11.0"
B300_PEAK_TFLOPS = 2250.0   # ASSUMED dense bf16 peak (no sparsity)
MI350X_PEAK_TFLOPS = 2300.0
REMOTE_REPO = "/repo"
LOCAL_REPO = Path(__file__).resolve().parent.parent
LOCAL_RESULTS = Path("/home/mj/krea2-ops/mi350x/results/modal")
FUNCTION_TIMEOUT_S = 40 * 60
DEFAULT_BATCHES = "2,4,8"
DEFAULT_PROFILE_BATCH = 4
ATTN_CHOICES = ("ref", "cudnn")
FUSED_ATTN = {"cudnn", "aiter", "splash"}
REMOTE_PROFILE_ROOT = "/tmp/prof"
MAX_RETURN_BYTES = 200 * 2**20   # per returned file
ATTN_SHAPE = (2, 24, 4608, 128)   # B, H, L, D
MATMUL_SIZES = (4096, 8192, 16384)

# Measured on 1x MI350X (jax 0.11.0 ROCm). bench_step there: --base-ckpt (real
# weights) --mode fwdbwd 1024^2 txt512 dots --qchunk 512 --act-sharding on
# --steps 5 --warmup 1; shapes and analytic FLOPs are identical to
# --random-weights.
MI350X = {
    "matmul": {4096: 955.0, 8192: 1128.0},
    "bench": [
        {"attn": "ref", "batch": 2, "ms": 2247.0, "sps": 0.89, "tflops": 347.0,
         "src": "stage 2 (final30.log)"},
        {"attn": "aiter", "batch": 4, "ms": 1813.0, "sps": 2.21,
         "tflops": 860.0, "src": "stage 2"},
        {"attn": "aiter", "batch": 4, "ms": 1819.1, "sps": 2.20,
         "tflops": 857.3, "src": "go_tune base (today)"},
        {"attn": "aiter", "batch": 8, "ms": 3580.0, "sps": 2.23,
         "tflops": 871.0, "src": "stage 2"},
    ],
}

# Same pins as the MI350X pod venv (jax 0.11.0 there was ROCm; here CUDA 13).
PIP_PACKAGES = [
    f"jax[cuda13]=={JAX_VERSION}",
    f"jaxlib=={JAX_VERSION}",
    "optax==0.2.8",
    "orbax-checkpoint==0.12.7",
    "safetensors==0.8.0",
    "ml_dtypes==0.6.0",
    "numpy==2.5.4",
    "pillow==12.3.0",
]

# dockerignore-style patterns, relative to the repo root.
REPO_IGNORE = [
    ".git", ".venv", "venv*", "**/__pycache__", "**/*.safetensors", "out",
    "results", "samples", "tests", ".pytest_cache", "**/*.pyc",
]

# Baked into the image (unchanged so the cached image layers are reused);
# the bench subprocesses override JAX_COMPILATION_CACHE_DIR per attention.
GPU_ENV = {
    # CUDA first; the CPU backend stays available (krea2_jax.checkpoint stages
    # via jax.devices("cpu") when loading real weights; harmless here).
    "JAX_PLATFORMS": "cuda,cpu",
    "PYTHONUNBUFFERED": "1",
    "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_cache",
    "TORCHDYNAMO_DISABLE": "1",
}
# What the MI350X bench runs used (minus the ROCm-only command-buffer flag).
BENCH_ENV = {"JAX_PLATFORMS": "cuda,cpu",
             "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92"}


def parse_batches(s):
    try:
        out = [int(x) for x in str(s).replace(" ", "").split(",") if x]
    except ValueError:
        raise SystemExit(f"--batches must be comma-separated ints, got {s!r}")
    if not out or any(b <= 0 for b in out):
        raise SystemExit(f"--batches must be positive ints, got {s!r}")
    return out


# ---------------------------------------------------------------- micro child
def micro_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro-child", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default=",".join(map(str, MATMUL_SIZES)))
    ap.add_argument("--attn-shape", default=",".join(map(str, ATTN_SHAPE)),
                    help="B,H,L,D")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="local smoke test: do not require a GPU / jax pin")
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp

    res = {"env": {}, "matmul": {}, "attention": {}}
    env = res["env"]
    env["jax"] = jax.__version__
    devs = jax.devices()
    d = devs[0]
    env["devices"] = repr(devs)
    env["platform"] = d.platform
    env["device_kind"] = d.device_kind
    print(f"jax {jax.__version__} | devices {devs}", flush=True)
    print(f"device_kind {d.device_kind!r} platform {d.platform}", flush=True)
    try:
        ms = d.memory_stats() or {}
        env["memory_stats"] = {k: v for k, v in ms.items()
                               if "limit" in k or "in_use" in k}
        if "bytes_limit" in ms:
            env["bytes_limit_gib"] = ms["bytes_limit"] / 2**30
    except Exception as e:  # noqa: BLE001
        env["memory_stats"] = f"n/a: {e}"
    print(f"memory_stats {env['memory_stats']}", flush=True)
    if not a.allow_cpu:
        if d.platform not in ("gpu", "cuda"):
            raise SystemExit(f"FATAL: device 0 is not a GPU: {d!r} "
                             f"(platform {d.platform})")
        if jax.__version__ != JAX_VERSION:
            raise SystemExit(f"FATAL: jax {jax.__version__} != {JAX_VERSION}")

    def timeit(f, *args):
        out = f(*args)                    # compile + warmup
        jax.block_until_ready(out)
        t = time.perf_counter()
        for _ in range(a.reps):
            out = f(*args)
        jax.block_until_ready(out)
        return (time.perf_counter() - t) / a.reps

    # 2. matmul: identical to the pod check (pod_bootstrap.sh / run_stage2.sh)
    for n in map(int, a.sizes.split(",")):
        try:
            x = jax.random.normal(jax.random.PRNGKey(0), (n, n), jnp.bfloat16)
            f = jax.jit(lambda m: m @ m)
            dt = timeit(f, x)
            tf = 2 * n**3 / dt / 1e12
            res["matmul"][str(n)] = {"ms": dt * 1e3, "tflops": tf}
            print(f"MATMUL bf16 {n}^2: {dt * 1e3:.3f} ms, {tf:.1f} TFLOP/s "
                  f"({100 * tf / B300_PEAK_TFLOPS:.1f}% of assumed "
                  f"{B300_PEAK_TFLOPS:.0f})", flush=True)
            del x
        except Exception as e:  # noqa: BLE001
            res["matmul"][str(n)] = {"error": f"{type(e).__name__}: {e}"[:400]}
            print(f"MATMUL bf16 {n}^2 FAILED: {e}", flush=True)

    # 3. attention at our shape (BTNH layout)
    B, H, L, D = map(int, a.attn_shape.split(","))
    res["attention"]["shape"] = [B, H, L, D]
    fwd_flops = 4 * B * H * L * L * D
    flops = {"fwd": fwd_flops, "fwdbwd": 3.5 * fwd_flops}
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(1), 3)
    q = jax.random.normal(kq, (B, L, H, D), jnp.bfloat16)
    k = jax.random.normal(kk, (B, L, H, D), jnp.bfloat16)
    v = jax.random.normal(kv, (B, L, H, D), jnp.bfloat16)
    for impl in ("xla", "cudnn"):
        def attn(q, k, v, impl=impl):
            return jax.nn.dot_product_attention(q, k, v, implementation=impl)

        def loss(q, k, v, impl=impl):
            return jnp.sum(attn(q, k, v).astype(jnp.float32))

        fns = {"fwd": jax.jit(attn),
               "fwdbwd": jax.jit(jax.grad(loss, argnums=(0, 1, 2)))}
        for mode, f in fns.items():
            key = f"{impl}_{mode}"
            try:
                dt = timeit(f, q, k, v)
                tf = flops[mode] / dt / 1e12
                res["attention"][key] = {"ms": dt * 1e3, "tflops": tf}
                print(f"ATTN {key} B{B} H{H} L{L} D{D}: {dt * 1e3:.3f} ms, "
                      f"{tf:.1f} TFLOP/s", flush=True)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}".strip().splitlines()
                res["attention"][key] = {"error": " | ".join(msg[:4])[:400]}
                print(f"ATTN {key} FAILED: {msg[:4]}", flush=True)
    try:
        ms = d.memory_stats() or {}
        if "peak_bytes_in_use" in ms:
            env["micro_peak_gib"] = ms["peak_bytes_in_use"] / 2**30
    except Exception:  # noqa: BLE001
        pass
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}", flush=True)
    return 0


# ------------------------------------------------------------- parent helpers
def stream(cmd, env, cwd, timeout_s, tail_n=30):
    """Run cmd, echo its merged stdout/stderr live, kill it after timeout_s.
    Returns (rc, all_lines, last tail_n lines, timed_out)."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    timed_out = threading.Event()

    def kill():
        timed_out.set()
        p.kill()

    timer = threading.Timer(max(timeout_s, 1), kill)
    timer.start()
    lines, tail = [], collections.deque(maxlen=tail_n)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            lines.append(line.rstrip("\n"))
            tail.append(line.rstrip("\n"))
        rc = p.wait()
    finally:
        timer.cancel()
    return rc, lines, list(tail), timed_out.is_set()


def parse_bench(lines):
    """Pull the numbers bench_step.py prints (see its main())."""
    text = "\n".join(lines)
    pats = {
        "ms": r"step time: median ([\d.]+) ms",
        "n_timed": r"step time: median [\d.]+ ms over (\d+) steps",
        "sps": r"throughput: ([\d.]+) samples/s",
        "tflops": r"analytic ([\d.]+) TFLOP/s per device",
        "compile_s": r"first step \(compile \+ run\): ([\d.]+)s",
        "mem_gib": r"device 0 memory: \S+ ([\d.]+) GiB",
        "loss": r"\| loss ([-\d.naif]+)",
    }
    out = {}
    for k, pat in pats.items():
        m = re.search(pat, text)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                out[k] = m.group(1)
    if re.search(r"RESOURCE_EXHAUSTED|Out of memory", text):
        out["oom"] = True
    return out


def bench_cmd(python, repo, attn, batch, extra=()):
    return [python, f"{repo}/scripts/bench_step.py", "--random-weights",
            "--mode", "fwdbwd", "--width", "1024", "--height", "1024",
            "--txt-len", "512", "--remat-policy", "dots", "--attn", attn,
            "--qchunk", "512", "--act-sharding", "on", "--steps", "5",
            "--warmup", "1", *extra, "--batch-size", str(batch)]


def bench_env(base_env, attn):
    return {**base_env, **BENCH_ENV,
            "JAX_COMPILATION_CACHE_DIR": f"/tmp/jax_cache_{attn}"}


def run_one(python, repo, env, attn, b, deadline, kind, extra=()):
    """One bench_step subprocess -> result dict (never raises)."""
    left = deadline - time.time()
    if left < 120:
        r = {"kind": kind, "attn": attn, "batch": b,
             "error": f"skipped: only {left:.0f}s left before the function "
                      f"timeout"}
        print(r["error"], flush=True)
        return r
    t0 = time.time()
    rc, lines, tail, to = stream(bench_cmd(python, repo, attn, b, extra), env,
                                 repo, left - 60)
    r = {"kind": kind, "attn": attn, "batch": b, **parse_bench(lines),
         "rc": rc, "wall_s": round(time.time() - t0, 1)}
    if rc != 0:
        r["error"] = ("TIMEOUT" if to else
                      "OOM" if r.get("oom") else f"rc={rc}")
        r["tail"] = tail
        print(f"\n!!! bench_step {kind} batch {b} failed ({r['error']}); last "
              f"{len(tail)} lines:\n" + "\n".join(tail), flush=True)
    return r


def run_ladder(python, repo, env, attn, batches, deadline):
    """bench_step at each batch; stop at the first failure (OOM or other)."""
    results = []
    for b in batches:
        r = run_one(python, repo, env, attn, b, deadline, "ladder")
        results.append(r)
        if "error" in r:
            break
    return results


def pick_profile_batch(ladder, want):
    ok = [r["batch"] for r in ladder if "error" not in r]
    if want in ok:
        return want
    return ok[0] if ok else None


def find_profile_files(root):
    """(trace, xplane) paths under a jax.profiler log dir, newest first."""
    root = Path(root)
    if not root.exists():
        return None, None
    files = [p for p in root.rglob("*") if p.is_file()]

    def newest(ps):
        return max(ps, key=lambda p: p.stat().st_mtime) if ps else None

    trace = (newest([p for p in files if p.name == "perfetto_trace.json.gz"])
             or newest([p for p in files
                        if p.name.endswith(".trace.json.gz")]))
    xplane = newest([p for p in files if p.name.endswith(".xplane.pb")])
    return trace, xplane


def _fmt(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x, peak):
    return "n/a" if x is None else f"{100 * x / peak:.1f}%"


def _ratio(b3, mi, higher_is_better=True):
    if b3 is None or mi is None or b3 == 0 or mi == 0:
        return "n/a"
    return f"{(b3 / mi) if higher_is_better else (mi / b3):.2f}x"


def _attn_kind(attn):
    return "fused" if attn in FUSED_ATTN else "XLA ref"


def micro_rows(micro):
    rows = ["| measurement | B300 | MI350X | unit | B300 speedup |",
            "|---|---|---|---|---|"]
    sizes = sorted({int(n) for n in micro.get("matmul", {})} | set(MATMUL_SIZES))
    for n in sizes:
        m = micro.get("matmul", {}).get(str(n), {})
        b3 = m.get("tflops")
        mi = MI350X["matmul"].get(n)
        rows.append(f"| matmul bf16 {n}^2 | {_fmt(b3)} | {_fmt(mi)} | TFLOP/s | "
                    f"{_ratio(b3, mi)} |")
        rows.append(f"| matmul bf16 {n}^2 % of peak | "
                    f"{_pct(b3, B300_PEAK_TFLOPS)} (assumed peak) | "
                    f"{_pct(mi, MI350X_PEAK_TFLOPS)} | % | - |")
        if "error" in m:
            rows.append(f"| matmul {n}^2 error | {m['error'][:120]} | | | |")
    att = micro.get("attention", {})
    shp = att.get("shape", list(ATTN_SHAPE))
    for key in ("xla_fwd", "xla_fwdbwd", "cudnn_fwd", "cudnn_fwdbwd"):
        a = att.get(key, {})
        name = f"attention {key} B{shp[0]} H{shp[1]} L{shp[2]} D{shp[3]}"
        if "error" in a:
            rows.append(f"| {name} | ERROR: {a['error'][:160]} | not measured "
                        f"| | - |")
            continue
        rows.append(f"| {name} | {_fmt(a.get('ms'), 2)} | not measured | ms | - |")
        rows.append(f"| {name} | {_fmt(a.get('tflops'))} | not measured | "
                    f"TFLOP/s | - |")
    return rows


def build_report(micro, ladder, prof, env_extra, opts):
    """micro: dict or None; ladder: list of result dicts; prof: dict or None;
    opts: attn/batches/quick/profile_batch/notes."""
    attn = opts["attn"]
    out = [f"# B300 vs MI350X (JAX, bench_step attn={attn})", ""]
    env = (micro or {}).get("env", {})
    if env:
        out.append(f"- B300: jax {env.get('jax')} | {env.get('device_kind')} | "
                   f"{env.get('devices')}")
        if env.get("bytes_limit_gib"):
            out.append(f"- B300 XLA bytes_limit: {env['bytes_limit_gib']:.1f} GiB")
    for k, v in env_extra.items():
        out.append(f"- {k}: {v}")
    out.append(f"- Peaks: B300 dense bf16 {B300_PEAK_TFLOPS:.0f} TFLOP/s is an "
               f"ASSUMPTION (MFU columns labelled 'assumed'); MI350X "
               f"{MI350X_PEAK_TFLOPS:.0f} TFLOP/s")
    out.append("- 'B300 speedup' = B300/MI350X for rates, MI350X/B300 for ms "
               "(>1 means B300 is faster)")
    for n in opts.get("notes", []):
        out.append(f"- {n}")

    if micro is not None:
        out += ["", "## Micro benchmarks", ""] + micro_rows(micro)

    out += ["", "## B300 bench_step ladder", ""]
    if opts.get("quick"):
        out.append("skipped (--quick)")
    else:
        out += ["| run | attn | batch | step ms | samples/s | TFLOP/s/dev | "
                f"MFU (assumed {B300_PEAK_TFLOPS:.0f}) | compile s | peak GiB | "
                "status |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in ladder + ([prof] if prof else []):
            label = ("ladder" if r["kind"] == "ladder" else
                     f"profiled ({_fmt(r.get('n_timed'), 0)} timed steps; not "
                     "used below)")
            out.append(
                f"| {label} | {r['attn']} | {r['batch']} | {_fmt(r.get('ms'), 0)} "
                f"| {_fmt(r.get('sps'), 2)} | {_fmt(r.get('tflops'))} | "
                f"{_pct(r.get('tflops'), B300_PEAK_TFLOPS)} | "
                f"{_fmt(r.get('compile_s'))} | {_fmt(r.get('mem_gib'))} | "
                f"{r.get('error', 'ok')} |")
        not_run = [b for b in opts["batches"]
                   if b not in {r["batch"] for r in ladder}]
        if not_run:
            out.append(f"\nnot run (after the first failure): batches {not_run}")

    out += ["", "## MI350X reference (measured, 1 GPU)", "",
            "| attn | batch | step ms | samples/s | TFLOP/s/dev | "
            f"MFU (peak {MI350X_PEAK_TFLOPS:.0f}) | source |",
            "|---|---|---|---|---|---|---|"]
    for m in MI350X["bench"]:
        out.append(f"| {m['attn']} | {m['batch']} | {m['ms']:.0f} | "
                   f"{m['sps']:.2f} | {m['tflops']:.0f} | "
                   f"{_pct(m['tflops'], MI350X_PEAK_TFLOPS)} | {m['src']} |")

    ok = [r for r in ladder if "error" not in r]
    if not opts.get("quick"):
        out += ["", "## B300 vs MI350X (same batch)", "",
                "| comparison | B300 ms | MI350X ms | B300 samples/s | "
                "MI350X samples/s | B300 TFLOP/s | MI350X TFLOP/s | B300 MFU "
                "(assumed) | MI350X MFU | B300 speedup |",
                "|---|---|---|---|---|---|---|---|---|---|"]

        def cmp_row(label, r, m):
            out.append(
                f"| {label} | {_fmt(r.get('ms'), 0)} | {m['ms']:.0f} | "
                f"{_fmt(r.get('sps'), 2)} | {m['sps']:.2f} | "
                f"{_fmt(r.get('tflops'))} | {m['tflops']:.0f} | "
                f"{_pct(r.get('tflops'), B300_PEAK_TFLOPS)} | "
                f"{_pct(m['tflops'], MI350X_PEAK_TFLOPS)} | "
                f"{_ratio(r.get('sps'), m['sps'])} |")

        n_cmp = 0
        for r in ok:
            for m in MI350X["bench"]:
                if m["batch"] != r["batch"]:
                    continue
                cmp_row(f"B300 {r['attn']} b{r['batch']} vs MI350X "
                        f"**{m['attn']}** b{m['batch']} ({_attn_kind(r['attn'])}"
                        f" vs {_attn_kind(m['attn'])}; {m['src']})", r, m)
                n_cmp += 1
        if ok:
            best = max(ok, key=lambda r: r.get("sps") or 0)
            fused = [m for m in MI350X["bench"] if m["attn"] in FUSED_ATTN]
            mbest = max(fused, key=lambda m: m["sps"])
            cmp_row(f"best B300 {best['attn']} (b{best['batch']}) vs best "
                    f"MI350X **{mbest['attn']}** (b{mbest['batch']}) "
                    f"({_attn_kind(best['attn'])} vs {_attn_kind(mbest['attn'])})",
                    best, mbest)
            n_cmp += 1
        if not n_cmp:
            out.append("| no successful B300 ladder run | | | | | | | | | |")

    if prof is not None or opts.get("profile_note"):
        out += ["", "## Profile", ""]
        if opts.get("profile_note"):
            out.append(f"- {opts['profile_note']}")
        if prof is not None:
            out.append(f"- profiled run: attn={prof['attn']} b{prof['batch']}, "
                       f"bench_step --profile-dir {REMOTE_PROFILE_ROOT}/... "
                       "--steps 5 --warmup 1 (2 profiled steps, 3 timed); "
                       f"status {prof.get('error', 'ok')}")
        for f in opts.get("profile_files", []):
            out.append(f"- {f}")

    out += [
        "",
        "Notes:",
        f"- bench_step: --random-weights --mode fwdbwd 1024x1024 txt512 "
        f"--remat-policy dots --attn {attn} --qchunk 512 --act-sharding on "
        "--steps 5 --warmup 1, 1 device, XLA_PYTHON_CLIENT_MEM_FRACTION=0.92. "
        "MI350X used --base-ckpt (same shapes and analytic FLOPs).",
        "- attn kinds: 'fused' = cudnn (B300) / aiter (MI350X) flash kernels; "
        "'XLA ref' = --attn ref (qchunked XLA attention). cudnn vs aiter is "
        "the fused-vs-fused comparison; ref vs ref exists only at b2.",
        "- analytic TFLOP/s = 6 * params * tokens * 1.1 (dots remat) / step; "
        "attention FLOPs are not counted (same formula on both sides).",
    ]
    if micro is not None:
        out += [
            "- matmul: jit(a @ a), 1 warmup call, 20 back-to-back reps, block "
            "on the last (same as the MI350X pod check). MI350X did not run "
            "16384^2.",
            "- micro attention FLOPs: fwd = 4*B*H*L^2*D; fwd+bwd = 3.5x fwd; "
            "grads w.r.t. q, k and v. No MI350X counterpart was measured.",
        ]
    return "\n".join(out)


def check_local_bench_step(attn, profile):
    """Fail before spending credits if the local checkout (what /repo will
    contain) lacks the bench_step options this run needs."""
    src = (LOCAL_REPO / "scripts" / "bench_step.py").read_text()
    missing = []
    m = re.search(r'"--attn"\s*,\s*choices\s*=\s*\[([^\]]*)\]', src)
    if m:
        if f'"{attn}"' not in m.group(1) and f"'{attn}'" not in m.group(1):
            missing.append(f"--attn {attn} (choices: {m.group(1).strip()})")
    elif f'"{attn}"' not in src:
        missing.append(f"--attn {attn}")
    if profile and '"--profile-dir"' not in src:
        missing.append("--profile-dir")
    if missing:
        raise SystemExit(
            "local scripts/bench_step.py (mounted as /repo in the container) "
            f"lacks: {', '.join(missing)}. Check out the branch that has them "
            "before `modal run`.")


def write_local_results(res, out_root=LOCAL_RESULTS):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir, i = Path(out_root) / stamp, 1
    while out_dir.exists():   # never overwrite an earlier run's results
        out_dir, i = Path(out_root) / f"{stamp}-{i}", i + 1
    out_dir.mkdir(parents=True)
    paths = []
    p = out_dir / "report.md"
    p.write_text(res["report"] + "\n")
    paths.append(p)
    p = out_dir / "ladder.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in res.get("runs", [])))
    paths.append(p)
    for name, data in (res.get("files") or {}).items():
        p = out_dir / Path(name).name
        p.write_bytes(data)
        paths.append(p)
    return out_dir, paths


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*PIP_PACKAGES)
        .env(GPU_ENV)
        .add_local_dir(LOCAL_REPO, remote_path=REMOTE_REPO, ignore=REPO_IGNORE)
    )
    app = modal.App("krea2-bench-b300", image=image)

    @app.function(gpu="B300", timeout=FUNCTION_TIMEOUT_S, cpu=8.0,
                  memory=32 * 1024)
    def bench(attn: str = "cudnn", batches: str = DEFAULT_BATCHES,
              micro: bool = False, quick: bool = False, profile: bool = True,
              profile_batch: int = DEFAULT_PROFILE_BATCH) -> dict:
        deadline = time.time() + FUNCTION_TIMEOUT_S - 90
        env = {**os.environ, **GPU_ENV}
        bl = parse_batches(batches)
        env_extra = {}
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"], capture_output=True, text=True,
                timeout=60)
            env_extra["nvidia-smi"] = smi.stdout.strip() or smi.stderr.strip()
            hdr = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                 timeout=60).stdout
            m = re.search(r"CUDA Version:\s*([\d.]+)", hdr)
            if m:
                env_extra["driver CUDA version"] = m.group(1)
        except Exception as e:  # noqa: BLE001
            env_extra["nvidia-smi"] = f"n/a: {e}"
        try:
            from importlib import metadata
            env_extra["pip wheels"] = ", ".join(
                f"{p} {metadata.version(p)}" for p in
                ("jax", "jaxlib", "jax-cuda13-plugin", "nvidia-cuda-runtime",
                 "nvidia-cuda-nvcc", "nvidia-cudnn-cu13", "nvidia-cublas")
                if _has_dist(metadata, p))
        except Exception as e:  # noqa: BLE001
            env_extra["pip wheels"] = f"n/a: {e}"
        for k, v in env_extra.items():
            print(f"{k}: {v}", flush=True)

        micro_res = None
        if micro or quick:
            # steps 1-3 in a child process (frees the GPU before step 4)
            out = "/tmp/micro.json"
            rc, _, tail, to = stream([sys.executable, os.path.abspath(__file__),
                                      "--micro-child", "--out", out], env,
                                     "/tmp", min(deadline - time.time(), 15 * 60))
            if rc != 0 or not os.path.exists(out):
                raise RuntimeError(
                    f"micro benchmarks failed (rc={rc}, timeout={to}); last "
                    "lines:\n" + "\n".join(tail))
            micro_res = json.loads(Path(out).read_text())

        ladder, prof, files = [], None, {}
        opts = {"attn": attn, "batches": bl, "quick": quick, "notes": [],
                "profile_files": []}
        if not quick:
            benv = bench_env(env, attn)
            ladder = run_ladder(sys.executable, REMOTE_REPO, benv, attn, bl,
                                deadline)
            pb = pick_profile_batch(ladder, profile_batch) if profile else None
            if not profile:
                opts["profile_note"] = None
            elif pb is None:
                opts["profile_note"] = ("no profile: no ladder batch "
                                        "succeeded")
            else:
                if pb != profile_batch:
                    opts["profile_note"] = (
                        f"--profile-batch {profile_batch} did not succeed in "
                        f"the ladder (or was not in it); profiled b{pb}")
                pdir = f"{REMOTE_PROFILE_ROOT}/{attn}_b{pb}"
                prof = run_one(sys.executable, REMOTE_REPO, benv, attn, pb,
                               deadline, "profile",
                               extra=("--profile-dir", pdir))
                trace, xplane = find_profile_files(pdir)
                if trace is None:
                    opts["profile_files"].append(
                        f"no perfetto_trace.json.gz / *.trace.json.gz under "
                        f"{pdir}")
                for p in (trace, xplane):
                    if p is None:
                        continue
                    sz = p.stat().st_size
                    if sz > MAX_RETURN_BYTES:
                        opts["profile_files"].append(
                            f"{p} is {sz / 2**20:.0f} MiB > "
                            f"{MAX_RETURN_BYTES // 2**20} MiB: NOT returned "
                            "(lost when the container exits)")
                        continue
                    files[p.name] = p.read_bytes()
                    opts["profile_files"].append(
                        f"returned {p.name} ({sz / 2**20:.1f} MiB; remote {p})")

        report = build_report(micro_res, ladder, prof, env_extra, opts)
        print("\n" + report, flush=True)
        runs = ladder + ([prof] if prof else [])
        return {"report": report, "runs": runs, "micro": micro_res,
                "files": files}

    @app.local_entrypoint()
    def main(attn: Literal["ref", "cudnn"] = "cudnn",
             batches: str = DEFAULT_BATCHES, micro: bool = False,
             quick: bool = False, profile: bool = True,
             profile_batch: int = DEFAULT_PROFILE_BATCH):
        if attn not in ATTN_CHOICES:
            raise SystemExit(f"--attn must be one of {ATTN_CHOICES}")
        parse_batches(batches)
        if not quick:
            check_local_bench_step(attn, profile)
        res = bench.remote(attn=attn, batches=batches, micro=micro,
                           quick=quick, profile=profile,
                           profile_batch=profile_batch)
        print("\n" + "=" * 72 + "\n" + res["report"])
        out_dir, paths = write_local_results(res)
        print(f"\nwrote {out_dir}:")
        for p in paths:
            print(f"  {p}  ({p.stat().st_size / 2**20:.2f} MiB)")


def _has_dist(metadata, name):
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


if __name__ == "__main__" and "--micro-child" in sys.argv:
    sys.exit(micro_main(sys.argv[1:]))
