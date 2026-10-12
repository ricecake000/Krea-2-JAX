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

Copy-storm probe (--copies; ONE B300 container, variants run sequentially):

    modal run scripts/modal_bench_b300.py --copies
    modal run scripts/modal_bench_b300.py --copies --variants base,unroll --batch 4
    modal run scripts/modal_bench_b300.py --copies --xla-flags=--xla_gpu_autotune_level=0

  Each variant = bench_step --random-weights --mode fwdbwd --img-tokens 4096
  --text-rows 512 (1024^2, txt512) --remat-policy dots --act-sharding on
  --attn $ATTN --qchunk 512 --steps 4 --warmup 1 --profile-dir DIR, its own
  cold compile cache, and XLA_FLAGS += --xla_dump_to=/tmp/hlo_<variant>
  --xla_dump_hlo_as_text --xla_dump_hlo_module_re=fwdbwd (+ --xla-flags):
    base        layer loop as lax.scan (current default)
    dsf         + --xla_gpu_enable_dynamic_slice_fusion=true
    unroll      bench_step --layer-loop unroll (per-layer params, no stacking)
    unroll_dsf  both
  Options: --variants (default base,dsf,unroll,unroll_dsf), --batch (4),
  --steps (4: 2 profiled + 2 timed), --xla-flags "..." (every variant; use
  the --xla-flags=... form), --timeout-min (default 40, max 60).
  Local outputs: .../results/modal/<ts>/copies/report.md + variants.jsonl and
  per variant copies/<variant>/: perfetto_trace.json.gz, *.xplane.pb, the
  jit_fwdbwd *after_optimizations.txt.gz, hlo_copy_report.{md,json}
  (scripts/hlo_copy_report.py), trace_copy_stats.json (memcpy and
  slice/DUS fusion kernels/step, kept apart, plus their copy-like sum).
  Standalone trace check (no Modal):
    python scripts/modal_bench_b300.py --parse-trace perfetto_trace.json.gz

Local CPU smoke test of steps 1-3 (no Modal, tiny sizes):
    JAX_PLATFORMS=cpu python scripts/modal_bench_b300.py --micro-child --allow-cpu \\
        --sizes 256,512 --attn-shape 1,2,256,64 --reps 2 --out /tmp/micro.json
"""

import argparse
import collections
import gzip
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Literal

if __name__ == "__main__" and ("--micro-child" in sys.argv
                               or "--parse-trace" in sys.argv):
    modal = None   # the micro child (also run inside the container) and the
    #                standalone trace parser never need Modal
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
COPIES_MAX_TIMEOUT_MIN = 60   # hard Modal timeout of the copies probe
DEFAULT_BATCHES = "2,4,8"
DEFAULT_PROFILE_BATCH = 4
ATTN_CHOICES = ("ref", "cudnn")
FUSED_ATTN = {"cudnn", "aiter", "splash"}
REMOTE_PROFILE_ROOT = "/tmp/prof"
MAX_RETURN_BYTES = 200 * 2**20   # per returned file
ATTN_SHAPE = (2, 24, 4608, 128)   # B, H, L, D
MATMUL_SIZES = (4096, 8192, 16384)

# --copies probe: name -> (extra bench_step args, extra XLA_FLAGS)
COPIES_VARIANTS = {
    "base": ((), ""),
    "dsf": ((), "--xla_gpu_enable_dynamic_slice_fusion=true"),
    "unroll": (("--layer-loop", "unroll"), ""),
    "unroll_dsf": (("--layer-loop", "unroll"),
                   "--xla_gpu_enable_dynamic_slice_fusion=true"),
}
DEFAULT_COPIES_VARIANTS = "base,dsf,unroll,unroll_dsf"
DEFAULT_COPIES_BATCH = 4
DEFAULT_COPIES_STEPS = 4
PROFILE_STEPS = 2          # bench_step profiles the first 2 timed steps
REMOTE_COPIES_PROF = "/tmp/prof_copies"
HLO_DUMP_FLAGS = ("--xla_dump_hlo_as_text "
                  "--xla_dump_hlo_module_re=fwdbwd")
# GPU copy kernels: CUDA memcpy128 / memcpy32_post / MemcpyD2D, ROCm
# __amd_rocclr_copyBuffer, XLA *copy* fusions
COPY_KERNEL_RE = r"memcpy|copybuffer|copy"
# slice / dynamic-update-slice fusions XLA lowered to a real (non-memcpy)
# kernel, e.g. loop_dynamic_update_slice_fusion on B300: the same copies, so a
# variant that only turned memcpys into these kernels must not look fixed.
# Matched on the kernel name, only for kernels the memcpy regex did not take
# (a kernel is never counted in both categories).
SLICE_FUSION_KERNEL_RE = (r"dynamic_slice_fusion|dynamic_update_slice_fusion"
                          r"|dynamic-slice|dynamic-update-slice")

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


def bench_cmd(python, repo, attn, batch, extra=(), steps=5):
    # 1024^2 = 64x64 patches = 4096 image tokens; 512 text rows
    return [python, f"{repo}/scripts/bench_step.py", "--random-weights",
            "--mode", "fwdbwd", "--img-tokens", "4096", "--text-rows", "512",
            "--remat-policy", "dots", "--attn", attn,
            "--qchunk", "512", "--act-sharding", "on", "--steps", str(steps),
            "--warmup", "1", *extra, "--batch-size", str(batch)]


def bench_env(base_env, attn):
    return {**base_env, **BENCH_ENV,
            "JAX_COMPILATION_CACHE_DIR": f"/tmp/jax_cache_{attn}"}


def run_one(python, repo, env, attn, b, deadline, kind, extra=(), steps=5):
    """One bench_step subprocess -> result dict (never raises)."""
    left = deadline - time.time()
    if left < 120:
        r = {"kind": kind, "attn": attn, "batch": b,
             "error": f"skipped: only {left:.0f}s left before the function "
                      f"timeout"}
        print(r["error"], flush=True)
        return r
    t0 = time.time()
    rc, lines, tail, to = stream(bench_cmd(python, repo, attn, b, extra,
                                           steps), env, repo, left - 60)
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


def check_local_bench_step(attn, profile, need=()):
    """Fail before spending credits if the local checkout (what /repo will
    contain) lacks the bench_step options this run needs (`need`: extra
    option strings such as "--layer-loop")."""
    src = (LOCAL_REPO / "scripts" / "bench_step.py").read_text()
    missing = [o for o in ("--img-tokens", "--text-rows", *need)
               if f'"{o}"' not in src]
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


def _new_out_dir(out_root):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir, i = Path(out_root) / stamp, 1
    while out_dir.exists():   # never overwrite an earlier run's results
        out_dir, i = Path(out_root) / f"{stamp}-{i}", i + 1
    out_dir.mkdir(parents=True)
    return out_dir


def write_local_results(res, out_root=LOCAL_RESULTS):
    out_dir = _new_out_dir(out_root)
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


# ------------------------------------------------------- copies probe helpers
def parse_variants(s):
    out = [v for v in str(s).replace(" ", "").split(",") if v]
    bad = [v for v in out if v not in COPIES_VARIANTS]
    if not out or bad:
        raise SystemExit(f"--variants must be a comma-separated subset of "
                         f"{list(COPIES_VARIANTS)}, got {s!r}")
    if len(set(out)) != len(out):
        raise SystemExit(f"--variants has duplicates: {s!r}")
    return out


def merge_xla_flags(*parts):
    """Concatenate XLA_FLAGS fragments (never overwrite)."""
    return " ".join(p.strip() for p in parts if p and p.strip())


def copies_cache_dir(probe_id, name):
    """Compile cache dir of one variant of one copies_probe invocation:
    unique per probe so a second probe in a warm container compiles cold
    (and dumps fresh HLO) instead of reusing the first probe's executables."""
    return f"/tmp/jax_cache_copies_{probe_id}_{name}"


def new_probe_id():
    import uuid
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def copies_variant_env(base_env, attn, name, user_flags="", probe_id=None):
    """bench_env + a per-variant, per-probe cold compile cache + XLA_FLAGS =
    existing flags + the variant's flags + --xla-flags + the HLO dump
    flags. probe_id: chosen once per copies_probe call (new_probe_id())."""
    env = bench_env(base_env, attn)
    env["JAX_COMPILATION_CACHE_DIR"] = copies_cache_dir(
        probe_id or new_probe_id(), name)
    env["XLA_FLAGS"] = merge_xla_flags(
        env.get("XLA_FLAGS", ""), COPIES_VARIANTS[name][1], user_flags,
        f"--xla_dump_to=/tmp/hlo_{name} {HLO_DUMP_FLAGS}")
    return env


def find_hlo_file(root):
    """Newest jit_fwdbwd *after_optimizations.txt under an XLA dump dir."""
    root = Path(root)
    if not root.exists():
        return None
    hs = [p for p in root.rglob("*after_optimizations.txt")
          if p.is_file() and "jit_fwdbwd" in p.name]
    return max(hs, key=lambda p: p.stat().st_mtime) if hs else None


def _union_ms(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for a, b in sorted(intervals):
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1e3


def trace_copy_stats(path, copy_regex=COPY_KERNEL_RE,
                     step_regex=r"^bench_step$",
                     default_steps=PROFILE_STEPS,
                     slice_regex=SLICE_FUSION_KERNEL_RE):
    """Per-step copy kernel count/time and GPU busy time from a jax.profiler
    perfetto trace (*.json.gz / *.json, or a profile dir -> newest trace).

    Conventions as in krea2-ops/mi350x/tools/trace_breakdown.py: GPU
    processes = process_name containing "gpu"/"device"; kernel tracks = their
    threads named "Stream ..." (fallback "XLA Ops" if no stream track); step
    span = [first start, last end] of the host `bench_step` annotations
    (StepTraceAnnotation), extended to the end of a kernel straddling it; the
    divisor = number of annotations (distinct step_num), else default_steps
    with the whole kernel span as the step span. Two disjoint copy
    categories over the complete ("X") kernel events overlapping the span,
    by kernel name (case-insensitive): "memcpy" = matches copy_regex;
    "slice_fusion" = does not match copy_regex but matches slice_regex
    (slice / DUS fusions lowered to a real kernel); "copy_like" = their sum.
    Raw durations are summed. gpu busy = union of kernel intervals clipped
    to the span. Per-step numbers are per device. The copy_* keys are the
    memcpy category (kept for older readers)."""
    p = Path(path)
    if p.is_dir():
        p = find_profile_files(p)[0]
        if p is None:
            raise FileNotFoundError(f"no trace under {path}")
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "rt") as f:
        d = json.load(f)
    ev = d.get("traceEvents", []) if isinstance(d, dict) else d
    pname, tname = {}, {}
    for e in ev:
        if e.get("ph") == "M" and isinstance(e.get("args"), dict):
            if e.get("name") == "process_name":
                pname[e.get("pid")] = str(e["args"].get("name", ""))
            elif e.get("name") == "thread_name":
                tname[(e.get("pid"), e.get("tid"))] = str(
                    e["args"].get("name", ""))
    gpu_pids = {pid for pid, n in pname.items()
                if re.search(r"gpu|device", n, re.I)}
    tracks = {k for k, n in tname.items()
              if k[0] in gpu_pids and re.search(r"stream", n, re.I)}
    track_kind = "stream"
    if not tracks:
        tracks = {k for k, n in tname.items()
                  if k[0] in gpu_pids and re.search(r"xla ops", n, re.I)}
        track_kind = "xla ops (op-level fallback)"
    kernels = []
    for e in ev:
        if e.get("ph") != "X" or (e.get("pid"), e.get("tid")) not in tracks:
            continue
        try:
            ts, dur = float(e["ts"]), float(e.get("dur", 0))
        except (KeyError, TypeError, ValueError):
            continue
        args = e.get("args") if isinstance(e.get("args"), dict) else {}
        kernels.append((ts, ts + dur, str(e.get("name", "")),
                        str(args.get("hlo_op", "")), e.get("pid")))
    srx = re.compile(step_regex)
    steps = [e for e in ev if e.get("ph") == "X" and e.get("pid") not in
             gpu_pids and srx.search(str(e.get("name", ""))) and "ts" in e]
    if steps:
        nums = [(e.get("args") or {}).get("step_num") for e in steps]
        n_steps = (len(set(nums)) if all(n is not None for n in nums)
                   else len(steps))
        s0 = min(float(e["ts"]) for e in steps)
        s1 = max(float(e["ts"]) + float(e.get("dur", 0)) for e in steps)
        late = [k[1] for k in kernels if k[0] < s1 < k[1]]
        if late:
            s1 = max(s1, max(late))
        step_source = f"{len(steps)} bench_step annotations"
    else:
        n_steps = default_steps
        s0 = min((k[0] for k in kernels), default=0.0)
        s1 = max((k[1] for k in kernels), default=0.0)
        step_source = (f"no step annotations: whole kernel span / "
                       f"{default_steps} steps")
    n_dev = len({k[0] for k in tracks}) or 1
    div = max(n_steps, 1) * n_dev
    crx = re.compile(copy_regex, re.I)
    srx_slice = re.compile(slice_regex, re.I)

    def category(name):
        if crx.search(name):
            return "memcpy"
        if srx_slice.search(name):
            return "slice_fusion"
        return None

    inspan = [k for k in kernels if k[0] < s1 and k[1] > s0]
    copies = [k for k in inspan if category(k[2]) == "memcpy"]
    slices = [k for k in inspan if category(k[2]) == "slice_fusion"]
    busy = 0.0
    for pid in {k[4] for k in inspan}:
        busy += _union_ms([(max(k[0], s0), min(k[1], s1))
                           for k in inspan if k[4] == pid])

    def breakdown(ks):
        by_name = collections.defaultdict(lambda: [0, 0.0])
        by_op = collections.defaultdict(lambda: [0, 0.0])
        for k in ks:
            by_name[k[2]][0] += 1
            by_name[k[2]][1] += (k[1] - k[0]) / 1e3
            if k[3]:
                op = re.sub(r"\.\d+$", "", k[3])
                by_op[op][0] += 1
                by_op[op][1] += (k[1] - k[0]) / 1e3
        return per_step(by_name), dict(list(per_step(by_op).items())[:12])

    def per_step(dd):
        return {n: {"count_per_step": round(c / div, 2),
                    "ms_per_step": round(ms / div, 3)}
                for n, (c, ms) in sorted(dd.items(), key=lambda kv: -kv[1][1])}

    def ms_of(ks):
        return sum(k[1] - k[0] for k in ks) / 1e3

    copy_by_name, copy_by_op = breakdown(copies)
    slice_by_name, slice_by_op = breakdown(slices)
    return {
        "trace": str(p), "track_kind": track_kind, "n_tracks": len(tracks),
        "n_devices": n_dev, "steps": n_steps, "step_source": step_source,
        "step_span_ms_per_step": round((s1 - s0) / 1e3 / max(n_steps, 1), 3),
        "kernels_per_step": round(len(inspan) / div, 2),
        "gpu_busy_ms_per_step": round(busy / div, 3),
        # memcpy kernels (copy_* = the same numbers, older key names)
        "memcpy_regex": copy_regex,
        "memcpy_kernels_per_step": round(len(copies) / div, 2),
        "memcpy_ms_per_step": round(ms_of(copies) / div, 3),
        "copy_regex": copy_regex,
        "copy_kernels_per_step": round(len(copies) / div, 2),
        "copy_ms_per_step": round(ms_of(copies) / div, 3),
        "copy_kernels_in_trace": sum(1 for k in kernels
                                     if category(k[2]) == "memcpy"),
        "copy_by_name": copy_by_name,
        "copy_by_hlo_op": copy_by_op,
        # slice / DUS fusion kernels (not memcpy-named)
        "slice_fusion_regex": slice_regex,
        "slice_fusion_kernels_per_step": round(len(slices) / div, 2),
        "slice_fusion_ms_per_step": round(ms_of(slices) / div, 3),
        "slice_fusion_kernels_in_trace": sum(
            1 for k in kernels if category(k[2]) == "slice_fusion"),
        "slice_fusion_by_name": slice_by_name,
        "slice_fusion_by_hlo_op": slice_by_op,
        # memcpy + slice/DUS fusion
        "copy_like_kernels_per_step": round((len(copies) + len(slices))
                                            / div, 2),
        "copy_like_ms_per_step": round((ms_of(copies) + ms_of(slices))
                                       / div, 3),
    }


def build_copies_report(rows, opts):
    """rows: one dict per variant (run fields + 'trace' stats dict + 'hlo'
    totals dict + 'hlo_summary' line); opts: attn/batch/steps/xla_flags/
    env_extra/notes. Returns markdown."""
    out = [f"# Copy-storm probe on B300 (bench_step fwdbwd, attn="
           f"{opts.get('attn')}, batch {opts.get('batch')})", ""]
    for k, v in (opts.get("env_extra") or {}).items():
        out.append(f"- {k}: {v}")
    out.append(f"- bench_step --random-weights --mode fwdbwd --img-tokens 4096 "
               f"--text-rows 512 --remat-policy dots --act-sharding on --qchunk "
               f"512 --steps {opts.get('steps')} --warmup 1 --profile-dir "
               f"(first {PROFILE_STEPS} timed steps profiled, the rest timed); "
               "each variant compiles cold (own cache dir, unique per probe, "
               "removed first) with an HLO dump")
    if opts.get("xla_flags"):
        out.append(f"- --xla-flags on every variant: `{opts['xla_flags']}`")
    for n in opts.get("notes", []):
        out.append(f"- {n}")
    out += ["", "| variant | step ms | vs base | samples/s | compile s | "
            "memcpy kernels/step | memcpy ms/step | slice/DUS fusion "
            "kernels/step | slice/DUS fusion ms/step | copy-like total "
            "kernels/step | copy-like total ms/step | GPU busy ms/step "
            "(stream union) | HLO copy-like dyn | HLO dyn (krea2_jax) | "
            "HLO MiB/exec | status |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base_ms = next((r.get("ms") for r in rows
                    if r.get("variant") == "base" and r.get("ms")), None)
    for r in rows:
        t = r.get("trace") or {}
        h = r.get("hlo") or {}
        vs = ("n/a" if not base_ms or not r.get("ms") else
              f"{100 * (r['ms'] / base_ms - 1):+.1f}%")
        mib = h.get("bytes_per_exec")
        out.append(
            f"| {r.get('variant')} | {_fmt(r.get('ms'), 1)} | {vs} | "
            f"{_fmt(r.get('sps'), 3)} | {_fmt(r.get('compile_s'))} | "
            f"{_fmt(t.get('memcpy_kernels_per_step'), 1)} | "
            f"{_fmt(t.get('memcpy_ms_per_step'), 2)} | "
            f"{_fmt(t.get('slice_fusion_kernels_per_step'), 1)} | "
            f"{_fmt(t.get('slice_fusion_ms_per_step'), 2)} | "
            f"{_fmt(t.get('copy_like_kernels_per_step'), 1)} | "
            f"{_fmt(t.get('copy_like_ms_per_step'), 2)} | "
            f"{_fmt(t.get('gpu_busy_ms_per_step'), 1)} | "
            f"{h.get('dynamic', 'n/a')} | {h.get('dynamic_krea2', 'n/a')} | "
            f"{'n/a' if mib is None else f'{mib / 2**20:.1f}'} | "
            f"{r.get('error', 'ok')} |")
    out += ["", "Variants: " + "; ".join(
        f"{n} = bench_step {' '.join(a) or '(scan)'}"
        + (f" + XLA_FLAGS {f}" if f else "")
        for n, (a, f) in COPIES_VARIANTS.items()
        if n in {r.get('variant') for r in rows})]
    out += ["", "## hlo_copy_report summaries", ""]
    for r in rows:
        out.append(f"- {r.get('variant')}: "
                   f"{r.get('hlo_summary') or 'n/a (no HLO)'}")
    for title, key in (("Memcpy kernels", "copy_by_name"),
                       ("Slice/DUS fusion kernels", "slice_fusion_by_name")):
        out += ["", f"## {title} by name (per step)", ""]
        for r in rows:
            t = r.get("trace") or {}
            names = list((t.get(key) or {}).items())[:6]
            out.append(f"- {r.get('variant')}: " + (", ".join(
                f"{n} {v['count_per_step']} x / {v['ms_per_step']} ms"
                for n, v in names) or ("none" if t else "n/a")))
    for r in rows:
        if r.get("notes"):
            out.append(f"- {r.get('variant')} notes: " + "; ".join(r["notes"]))
    out += ["", "Notes:",
            "- memcpy kernels/step, ms/step: perfetto trace kernels on GPU "
            f"stream tracks whose name matches /{COPY_KERNEL_RE}/i inside "
            "the bench_step step span, / number of step annotations "
            "(scripts/modal_bench_b300.py trace_copy_stats).",
            "- slice/DUS fusion kernels/step, ms/step: the other kernels "
            f"whose name matches /{SLICE_FUSION_KERNEL_RE}/i (slice / "
            "dynamic-update-slice fusions lowered to a real kernel instead "
            "of a memcpy; never also counted as memcpy). copy-like total = "
            "memcpy + slice/DUS fusion: a variant that only changed the "
            "copies' kernel type does not lower it.",
            "- HLO columns: scripts/hlo_copy_report.py on the optimized "
            "jit_fwdbwd HLO: copy-like instructions (copy, slice, DS, DUS and "
            "fusions rooted in them) x while trip counts per step; "
            "'krea2_jax' = source_file under krea2_jax/; MiB/exec counts DUS "
            "by update size.",
            "- step ms is the median of the non-profiled timed steps only "
            "(few steps: compare with the ladder for precise numbers); "
            "compile s = bench_step 'first step (compile + run)', includes "
            "writing the HLO dump."]
    return "\n".join(out)


def analyze_copies_variant(vdir, python=sys.executable):
    """Run hlo_copy_report.py + trace_copy_stats on a local variant dir.
    Returns (trace stats or None, hlo totals or None, summary line, notes)."""
    vdir = Path(vdir)
    notes, trace, hlo, summary = [], None, None, None
    tr = vdir / "perfetto_trace.json.gz"
    if not tr.exists():
        cands = sorted(vdir.glob("*.trace.json.gz"))
        tr = cands[0] if cands else None
    if tr is not None:
        try:
            trace = trace_copy_stats(tr)
            (vdir / "trace_copy_stats.json").write_text(
                json.dumps(trace, indent=1))
        except Exception as e:  # noqa: BLE001
            notes.append(f"trace parse failed: {type(e).__name__}: {e}")
    hl = sorted(vdir.glob("*after_optimizations.txt.gz"))
    if hl:
        js = vdir / "hlo_copy_report.json"
        try:
            pr = subprocess.run(
                [python, str(LOCAL_REPO / "scripts" / "hlo_copy_report.py"),
                 str(hl[0]), "--json", str(js), "--top", "40"],
                capture_output=True, text=True, timeout=1800)
            (vdir / "hlo_copy_report.md").write_text(pr.stdout)
            if pr.returncode != 0:
                notes.append(f"hlo_copy_report rc={pr.returncode}: "
                             f"{pr.stderr.strip()[-300:]}")
            lines = [ln for ln in pr.stdout.splitlines()
                     if ln.startswith("copies:")]
            summary = lines[-1] if lines else None
            if js.exists():
                data = json.loads(js.read_text())
                hlo = dict(data["totals"])
                hlo["dynamic_krea2"] = sum(
                    it["multiplier"] for it in data["instructions"]
                    if "krea2_jax" in it.get("source_file", ""))
        except Exception as e:  # noqa: BLE001
            notes.append(f"hlo_copy_report failed: {type(e).__name__}: {e}")
    return trace, hlo, summary, notes


def write_copies_results(res, out_root=LOCAL_RESULTS):
    """Save the copies probe's files under <ts>/copies/<variant>/, analyze
    them, write <ts>/copies/report.md + variants.jsonl; returns (dir, md)."""
    cdir = _new_out_dir(out_root) / "copies"
    cdir.mkdir()
    rows = []
    for r in res.get("variants", []):
        name = r["variant"]
        vdir = cdir / name
        vdir.mkdir()
        for fname, data in (res.get("files", {}).get(name) or {}).items():
            (vdir / Path(fname).name).write_bytes(data)
        if r.get("tail"):
            (vdir / "stdout_tail.txt").write_text("\n".join(r["tail"]) + "\n")
        trace, hlo, summary, notes = analyze_copies_variant(vdir)
        rows.append({**r, "trace": trace, "hlo": hlo, "hlo_summary": summary,
                     "notes": list(r.get("notes", [])) + notes})
    md = build_copies_report(rows, res.get("opts", {}))
    (cdir / "report.md").write_text(md + "\n")
    (cdir / "variants.jsonl").write_text("".join(
        json.dumps({k: v for k, v in r.items() if k != "trace"}) + "\n"
        for r in rows))
    return cdir, md


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
        env_extra = collect_env_extra()

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

    @app.function(gpu="B300", timeout=COPIES_MAX_TIMEOUT_MIN * 60, cpu=8.0,
                  memory=32 * 1024)
    def copies_probe(attn: str = "cudnn", batch: int = DEFAULT_COPIES_BATCH,
                     variants: str = DEFAULT_COPIES_VARIANTS,
                     xla_flags: str = "", steps: int = DEFAULT_COPIES_STEPS,
                     timeout_min: int = FUNCTION_TIMEOUT_S // 60) -> dict:
        # internal deadline from --timeout-min (<= the hard Modal timeout)
        deadline = time.time() + min(timeout_min,
                                     COPIES_MAX_TIMEOUT_MIN) * 60 - 90
        env = {**os.environ, **GPU_ENV}
        env_extra = collect_env_extra()
        runs, files = [], {}
        probe_id = new_probe_id()  # fresh compile caches for this probe
        print(f"copies probe id {probe_id}", flush=True)
        for name in parse_variants(variants):
            r, f = run_copies_variant(sys.executable, REMOTE_REPO, env, attn,
                                      batch, name, steps, xla_flags, deadline,
                                      probe_id)
            runs.append(r)
            files[name] = f
        opts = {"attn": attn, "batch": batch, "steps": steps,
                "xla_flags": xla_flags, "env_extra": env_extra, "notes": [],
                "probe_id": probe_id}
        return {"variants": runs, "files": files, "opts": opts}

    @app.local_entrypoint()
    def main(attn: Literal["ref", "cudnn"] = "cudnn",
             batches: str = DEFAULT_BATCHES, micro: bool = False,
             quick: bool = False, profile: bool = True,
             profile_batch: int = DEFAULT_PROFILE_BATCH,
             copies: bool = False, batch: int = DEFAULT_COPIES_BATCH,
             variants: str = DEFAULT_COPIES_VARIANTS, xla_flags: str = "",
             steps: int = DEFAULT_COPIES_STEPS,
             timeout_min: int = FUNCTION_TIMEOUT_S // 60):
        if attn not in ATTN_CHOICES:
            raise SystemExit(f"--attn must be one of {ATTN_CHOICES}")
        if copies:
            vl = parse_variants(variants)
            if batch <= 0:
                raise SystemExit(f"--batch must be positive, got {batch}")
            if steps <= PROFILE_STEPS:
                raise SystemExit(f"--steps must be > {PROFILE_STEPS} (the "
                                 "profiled steps are not timed)")
            if not 1 <= timeout_min <= COPIES_MAX_TIMEOUT_MIN:
                raise SystemExit(f"--timeout-min must be in 1.."
                                 f"{COPIES_MAX_TIMEOUT_MIN}")
            need = ({"--layer-loop"} if any(COPIES_VARIANTS[v][0] for v in vl)
                    else set())
            check_local_bench_step(attn, True, need=tuple(need))
            res = copies_probe.remote(attn=attn, batch=batch,
                                      variants=",".join(vl),
                                      xla_flags=xla_flags, steps=steps,
                                      timeout_min=timeout_min)
            cdir, md = write_copies_results(res)
            print("\n" + "=" * 72 + "\n" + md)
            print(f"\nwrote {cdir}:")
            for p in sorted(cdir.rglob("*")):
                if p.is_file():
                    print(f"  {p}  ({p.stat().st_size / 2**20:.2f} MiB)")
            return
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


def collect_env_extra():
    """nvidia-smi + wheel versions (inside the container); printed."""
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
    return env_extra


def run_copies_variant(python, repo, base_env, attn, batch, name, steps,
                       user_flags, deadline, probe_id):
    """One copies-probe variant in the container -> (result dict, files).
    probe_id (one per copies_probe call) makes the compile cache dir unique,
    and the dir is removed first: every variant compiles cold."""
    import shutil
    extra_args, vflags = COPIES_VARIANTS[name]
    pdir, hdir = f"{REMOTE_COPIES_PROF}/{name}", f"/tmp/hlo_{name}"
    cdir = copies_cache_dir(probe_id, name)
    for d in (pdir, hdir, cdir):
        shutil.rmtree(d, ignore_errors=True)
    env = copies_variant_env(base_env, attn, name, user_flags, probe_id)
    assert env["JAX_COMPILATION_CACHE_DIR"] == cdir
    print(f"\n=== variant {name}: XLA_FLAGS={env['XLA_FLAGS']!r}", flush=True)
    r = run_one(python, repo, env, attn, batch, deadline, f"copies:{name}",
                extra=("--profile-dir", pdir, *extra_args), steps=steps)
    r.update({"variant": name, "args": list(extra_args),
              "xla_flags": env["XLA_FLAGS"], "compile_cache_dir": cdir,
              "probe_id": probe_id, "notes": []})
    files = {}
    trace, xplane = find_profile_files(pdir)
    if trace is None and not r.get("error", "").startswith("skipped"):
        r["notes"].append(f"no perfetto trace under {pdir}")
    for p in (trace, xplane):
        if p is None:
            continue
        sz = p.stat().st_size
        if sz > MAX_RETURN_BYTES:
            r["notes"].append(f"{p.name} is {sz / 2**20:.0f} MiB > "
                              f"{MAX_RETURN_BYTES // 2**20} MiB: not returned")
            continue
        files[p.name] = p.read_bytes()
    hlo = find_hlo_file(hdir)
    if hlo is None:
        if not r.get("error", "").startswith("skipped"):
            r["notes"].append(f"no jit_fwdbwd *after_optimizations.txt under "
                              f"{hdir}")
    else:
        raw = hlo.read_bytes()
        gz = gzip.compress(raw, compresslevel=6)
        r["hlo_file"] = hlo.name
        r["hlo_mib"] = round(len(raw) / 2**20, 2)
        if len(gz) > MAX_RETURN_BYTES:
            r["notes"].append(f"{hlo.name} is {len(gz) / 2**20:.0f} MiB "
                              f"gzipped > {MAX_RETURN_BYTES // 2**20} MiB: "
                              "not returned")
        else:
            files[hlo.name + ".gz"] = gz
    return r, files


def _has_dist(metadata, name):
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


if __name__ == "__main__" and "--micro-child" in sys.argv:
    sys.exit(micro_main(sys.argv[1:]))
elif __name__ == "__main__" and "--parse-trace" in sys.argv:
    _ap = argparse.ArgumentParser(
        description="per-step copy kernel stats of a perfetto trace")
    _ap.add_argument("--parse-trace", required=True, metavar="TRACE_OR_DIR")
    _ap.add_argument("--copy-regex", default=COPY_KERNEL_RE,
                     help="memcpy kernel-name regex")
    _ap.add_argument("--slice-regex", default=SLICE_FUSION_KERNEL_RE,
                     help="slice/DUS fusion kernel-name regex (kernels not "
                          "matched by --copy-regex)")
    _a = _ap.parse_args()
    _st = trace_copy_stats(_a.parse_trace, _a.copy_regex,
                           slice_regex=_a.slice_regex)
    print(json.dumps(_st, indent=1))
    print(f"\nper step: memcpy kernels {_st['memcpy_kernels_per_step']} "
          f"({_st['memcpy_ms_per_step']} ms) | slice/DUS fusion kernels "
          f"{_st['slice_fusion_kernels_per_step']} "
          f"({_st['slice_fusion_ms_per_step']} ms) | copy-like total "
          f"{_st['copy_like_kernels_per_step']} "
          f"({_st['copy_like_ms_per_step']} ms)")
