"""GPU profile of the latent-cache VAE encoder (scripts/vae_fast.py): where
the GPU time of one bucket goes and whether cheap variants help. Runs inside
the Modal profile container:

    modal run scripts/modal_cache_latents.py --profile [--gpu B200]

Inputs are seeded uniform [-1, 1] images (compute is input-independent; no
dataset download). Sections (each one records its own error and the rest
still run; a deadline skips sections once `budget_s` is used up):
  flops      FlopCounterMode on the eager 2d encoder, one image, per stage
             (Encoder2d.stages(), the list forward itself runs)
  stages     eager 2d at the VAE dtype, `batch` images on the device: CUDA
             event time per stage, % of total, achieved TFLOP/s
  parity     check_impls (scripts/modal_cache_latents.py) of the variants
             2d-compile, 2d-compile-cl (vae_fast: module + input
             channels_last), 2d on
             2 seeded images: fp32 err < CHECK_TOL, bf16 err vs the ref's own
  profiler   torch.profiler over `prof_iters` production steps of the
             parity-checked 2d-compile (CPU stack + H2D + encode + bf16 D2H):
             kernel time by category, top kernels, GPU busy vs wall; plus the
             device-resident compiled forward time; chrome trace (gzip)
  sdpa       the mid attention's exact q/k/v shape (captured from the eager
             encoder): default dispatch and each SDPBackend, ms and TFLOP/s;
             which backend the default runs (from its kernel names)
  throughput run_ladder at `batch` for every variant that passed parity
  max_autotune  parity + ladder of 2d-compile under torch.compile mode=
             "max-autotune-no-cudagraphs" (2d-compile-mat): last, only when
             it can start before `mat_start_by`
`checkpoint(res)` runs after every section (but the last).
On CPU every section but flops records "skipped: no CUDA" unless cpu_ok
(smoke tests of the code paths; the numbers mean nothing there).

Importable without torch (torch is imported inside the functions); the
markdown renderer works on the plain result dict.
"""

import contextlib
import gzip
import math
import os
import re
import tempfile
import time
import traceback
import warnings

SEED = 20261102
CHECK_IMAGES = 2
HAND_TFLOP_PER_IMG = 2.7      # hand count at 832x1216 (convs ~86 %, attn ~14 %)
HAND_PIXELS = 832 * 1216
# dense bf16 tensor-core peak (no sparsity), by device-name substring:
# NVIDIA HGX B200 datasheet 2.25 PFLOPS; H100/H200 SXM 989 TFLOPS
PEAK_BF16_TFLOPS = {"B200": 2250.0, "H200": 989.0, "H100": 989.0}
VARIANTS = ("2d-compile", "2d-compile-cl", "2d")
BASELINE = "2d-compile"
# 2d-compile under mode="max-autotune-no-cudagraphs": run last, only when it
# can start before the caller's `mat_start_by` (long compiles)
MAX_AUTOTUNE = "2d-compile-mat"
# kernel-name categories, first match wins (attention before conv/gemm:
# cuDNN attention kernels contain "cudnn")
KERNEL_CATEGORIES = (
    ("attention", r"fmha|flash|attention|attn|sdpa|softmax"),
    ("copy", r"memcpy|memset"),
    ("triton", r"^triton_"),
    ("layout", r"nchwtonhwc|nhwctonchw|transpose|permute"),
    ("conv_gemm", r"conv|gemm|cudnn|nvjet|cutlass|xmma|sm1\d\d_|sm90|sm80|"
                  r"cublas|implicit|fprop|dgrad|wgrad"),
    ("aten_eager", r"at::native|elementwise|reduce|vectorized|"
                   r"distribution|philox|randn"),
)
STEP_LABEL = "encode_step"
SDPA_BACKENDS = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION",
                 "MATH")


def kernel_category(name):
    for cat, pat in KERNEL_CATEGORIES:
        if re.search(pat, name, re.IGNORECASE):
            return cat
    return "other"


def run_stages(net, x, wrap=None):
    """Encoder2d.forward through net.stages(), with `wrap(name)` (a context
    manager) around each stage; == net(x) bitwise (tests/test_vae_fast.py)."""
    for name, fns in net.stages():
        with (wrap(name) if wrap else contextlib.nullcontext()):
            for f in fns:
                x = f(x)
    return x


def union_us(intervals, lo=None, hi=None):
    """Total length of the union of [start, end) intervals, clipped to
    [lo, hi]."""
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        s = s if lo is None else max(s, lo)
        e = e if hi is None else min(e, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def summarize_kernels(kernels, window, top_n=25):
    """kernels [(name, start_us, end_us)], window (start_us, end_us) of the
    profiled steps -> per-category time, top kernels, GPU busy vs wall."""
    by_name, cats = {}, {}
    for name, s, e in kernels:
        d = by_name.setdefault(name, [0.0, 0])
        d[0] += e - s
        d[1] += 1
    total = sum(d[0] for d in by_name.values())
    for name, (us, n) in by_name.items():
        c = cats.setdefault(kernel_category(name), {"us": 0.0, "count": 0})
        c["us"] += us
        c["count"] += n
    for c in cats.values():
        c["pct"] = 100 * c["us"] / total if total else None
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])
    top = [{"name": n[:160], "category": kernel_category(n), "us": us,
            "count": k, "pct": 100 * us / total if total else None}
           for n, (us, k) in ranked[:top_n]]
    other = [n[:160] for n, _ in ranked if kernel_category(n) == "other"][:10]
    wall = window[1] - window[0]
    busy = union_us([(s, e) for _, s, e in kernels], *window)
    return {"kernel_us": total, "busy_us": busy, "wall_us": wall,
            "idle_pct": 100 * (1 - busy / wall) if wall > 0 else None,
            "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1]["us"])),
            "top": top, "other_top_names": other}


def sdpa_backend_from_kernels(names):
    """Kernel names of one SDPA call -> backend label (cuDNN first: its
    kernel names also contain 'flash')."""
    s = " ".join(names).lower()
    if re.search(r"cudnn|fort_native_sdpa", s):
        return "CUDNN_ATTENTION"
    if "flash" in s:
        return "FLASH_ATTENTION"
    if re.search(r"fmha|efficient|mem_eff|cutlass", s):
        return "EFFICIENT_ATTENTION"
    if re.search(r"gemm|bmm|softmax|nvjet", s):
        return "MATH"
    return None


def max_autotune_encoder(vae):
    """2d-compile with torch.compile(mode="max-autotune-no-cudagraphs")
    (same dynamic=True)."""
    import torch
    import vae_fast

    enc = vae_fast.make_encoder(vae, "2d")
    enc.forward = torch.compile(enc.net, dynamic=True,
                                mode="max-autotune-no-cudagraphs")
    return enc


class _Ctx:
    pass


def _sync(c):
    if c.cuda:
        c.torch.cuda.synchronize()


def _mark(c):
    if c.cuda:
        e = c.torch.cuda.Event(enable_timing=True)
        e.record()
        return e
    return time.perf_counter()


def _ms(c, a, b):
    return a.elapsed_time(b) if c.cuda else (b - a) * 1e3


def _tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12 if flops and ms else None


def _sdpa_shapes(c):
    """q/k/v shape + dtype of every SDPA call of one eager forward."""
    torch = c.torch
    F = torch.nn.functional
    calls = []

    class Spy(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func is F.scaled_dot_product_attention:
                calls.append({"shape": [list(a.shape) for a in args[:3]],
                              "dtype": str(args[0].dtype)})
            return func(*args, **(kwargs or {}))

    with Spy():
        c.net(c.x)
    return calls


def _sdpa_flops(shapes):
    """4 * B*H * N_q * N_kv * D (QK^T and PV) of one SDPA call."""
    q, k = shapes[0], shapes[1]
    return 4 * math.prod(q[:-1]) * k[-2] * q[-1]


def sec_flops(c):
    """Per-stage FlopCounterMode on one image. Each SDPA call is measured by
    the counter's growth across the call: an SDPA rule (fused kernels) or
    the math path's decomposed bmm both count there (sdpa_ops). Only a call
    that added 0 FLOPs (e.g. the CPU flash kernel has no formula) gets the
    analytic 4*N^2*D added (sdpa_added)."""
    from torch.utils.flop_counter import FlopCounterMode

    torch = c.torch
    F = torch.nn.functional
    counters, sdpa, counted, sdpa_ops, added, extra = {}, {}, {}, {}, [], {}

    def snapshot(fc):
        return {str(op): v for d in fc.get_flop_counts().values()
                for op, v in d.items()}

    class Spy(torch.overrides.TorchFunctionMode):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func is not F.scaled_dot_product_attention:
                return func(*args, **(kwargs or {}))
            n, fc = self.name, counters[self.name]
            ana = _sdpa_flops([a.shape for a in args[:3]])
            sdpa[n] = sdpa.get(n, 0) + ana
            f0, ops0 = fc.get_total_flops(), snapshot(fc)
            out = func(*args, **(kwargs or {}))
            got = fc.get_total_flops() - f0
            counted[n] = counted.get(n, 0) + got
            sdpa_ops.setdefault(n, set()).update(
                op for op, v in snapshot(fc).items() if v != ops0.get(op, 0))
            if got == 0:   # no counter formula for this kernel: analytic
                extra[n] = extra.get(n, 0) + ana
                counted[n] += ana
                added.append(n)
            return out

    @contextlib.contextmanager
    def wrap(name):
        counters[name] = FlopCounterMode(display=False)
        with counters[name], Spy(name):
            yield

    run_stages(c.net, c.x[:1], wrap)
    per = [(n, fc.get_total_flops() + extra.get(n, 0))
           for n, fc in counters.items()]
    total = sum(f for _, f in per)
    hand = HAND_TFLOP_PER_IMG * 1e12 * c.h * c.w / HAND_PIXELS
    c.stage_flops = dict(per)
    c.flops_img = total
    return {"per_image_flops": total, "hand_flops_scaled": hand,
            "ratio_vs_hand": total / hand if hand else None,
            "sdpa_flops": sum(sdpa.values()),
            "sdpa_counted_flops": sum(counted.values()),
            "sdpa_ops": sorted({o for v in sdpa_ops.values() for o in v}),
            "sdpa_added": sorted(set(added)),
            "per_stage": [{"stage": n, "flops": f,
                           "pct": 100 * f / total if total else None}
                          for n, f in per]}


def sec_stages(c):
    for _ in range(c.warmup):
        run_stages(c.net, c.x)
    _sync(c)
    runs = []
    for _ in range(c.stage_iters):
        spans = []

        @contextlib.contextmanager
        def wrap(name, spans=spans):
            a = _mark(c)
            yield
            spans.append((name, a, _mark(c)))
        run_stages(c.net, c.x, wrap)
        runs.append(spans)
    _sync(c)
    names = [n for n, _, _ in runs[0]]
    ms = {n: sum(_ms(c, a, b) for r in runs for m, a, b in r if m == n)
          / len(runs) for n in names}
    total = sum(ms.values())
    flops = getattr(c, "stage_flops", {})
    rows = []
    for n in names:
        f = flops.get(n)
        fb = f * c.batch if f is not None else None
        rows.append({"stage": n, "ms": ms[n], "pct": 100 * ms[n] / total,
                     "flops_batch": fb, "tflops": _tflops(fb, ms[n])})
    tf = (_tflops(c.flops_img * c.batch, total)
          if getattr(c, "flops_img", None) else None)
    return {"iters": c.stage_iters, "warmup": c.warmup, "total_ms": total,
            "img_s": c.batch / (total * 1e-3), "tflops": tf, "rows": rows}


def sec_parity(c):
    x = c.x_cpu[:CHECK_IMAGES]
    checks, c.encoders = c.check_impls(
        c.vae, list(VARIANTS), x, cuda=c.torch.cuda if c.cuda else None,
        log=c.log)
    return {"checks": checks}


def _step_fn(c, enc):
    torch, imgs, b = c.torch, c.imgs, c.batch

    def step(k):
        x = torch.stack([imgs[(k * b + j) % len(imgs)] for j in range(b)])
        return enc(x).to(torch.bfloat16).cpu()
    return step


def sec_profiler(c):
    torch = c.torch
    out = {}
    enc = getattr(c, "encoders", {}).get(BASELINE)
    if enc is None:
        import vae_fast
        enc = vae_fast.make_encoder(c.vae, BASELINE)
        out["note"] = f"{BASELINE} failed or skipped parity: fresh, ungated"
    step = _step_fn(c, enc)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # as production (OMP_NUM_THREADS=1) / ladder
    try:
        return _profile(c, enc, step, out)
    finally:
        torch.set_num_threads(threads)


def _profile(c, enc, step, out):
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, record_function

    for k in range(c.warmup):
        step(k)
    # the compiled forward alone, input already on the device
    with torch.no_grad():
        enc.forward(c.x)
        _sync(c)
        a = _mark(c)
        for _ in range(c.stage_iters):
            enc.forward(c.x)
        b = _mark(c)
        _sync(c)
    fwd = _ms(c, a, b) / c.stage_iters
    out.update(fwd_ms=fwd, fwd_img_s=c.batch / (fwd * 1e-3),
               fwd_tflops=_tflops(getattr(c, "flops_img", None)
                                  and c.flops_img * c.batch, fwd))
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if c.cuda else [])
    with profile(activities=acts) as prof:
        for k in range(c.prof_iters):
            with record_function(STEP_LABEL):
                step(k)
        _sync(c)
    ev = prof.events()
    steps = [e for e in ev if e.name == STEP_LABEL
             and e.device_type == DeviceType.CPU]
    window = (min(e.time_range.start for e in steps),
              max(e.time_range.end for e in steps))
    if c.cuda:
        # device-side user annotations (record_function, torch.compile
        # regions) span whole steps: not kernels
        kern = [e for e in ev if e.device_type == DeviceType.CUDA
                and e.name != STEP_LABEL
                and not getattr(e, "is_user_annotation", False)]
    else:   # smoke only: leaf CPU ops stand in for kernels
        kern = [e for e in ev if e.device_type == DeviceType.CPU
                and e.name != STEP_LABEL and not e.cpu_children]
    out.update(summarize_kernels(
        [(e.name, e.time_range.start, e.time_range.end) for e in kern],
        window))
    out.update(iters=c.prof_iters, step_img_s=c.prof_iters * c.batch
               / ((window[1] - window[0]) * 1e-6))
    if c.trace_path:
        with tempfile.TemporaryDirectory() as d:
            raw = os.path.join(d, "trace.json")
            prof.export_chrome_trace(raw)
            with open(raw, "rb") as f, gzip.open(c.trace_path, "wb") as g:
                g.write(f.read())
        out["trace"] = os.path.basename(c.trace_path)
        out["trace_bytes"] = os.path.getsize(c.trace_path)
    return out


def sec_sdpa(c):
    import torch
    from torch.autograd import DeviceType
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    F = torch.nn.functional
    calls = _sdpa_shapes(c)
    out = {"calls": calls}
    if not calls:
        out["error"] = "no scaled_dot_product_attention call in the encoder"
        return out
    shape = calls[0]["shape"][0]
    dtype = c.x.dtype
    torch.manual_seed(SEED)
    q, k, v = (torch.randn(*s, device=c.x.device, dtype=dtype)
               for s in calls[0]["shape"])
    bh, n, d = math.prod(shape[:-2]), shape[-2], shape[-1]
    n_kv = calls[0]["shape"][1][-2]
    flops = 4 * bh * n * n_kv * d
    out.update(shape=shape, dtype=str(dtype), flops=flops)
    rows = []
    for label in ("default",) + SDPA_BACKENDS:
        row = {"backend": label}
        ctx = (contextlib.nullcontext() if label == "default"
               else sdpa_kernel(getattr(SDPBackend, label)))
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            try:
                with ctx, torch.no_grad():
                    for _ in range(2):
                        F.scaled_dot_product_attention(q, k, v)
                    _sync(c)
                    a = _mark(c)
                    for _ in range(c.sdpa_iters):
                        F.scaled_dot_product_attention(q, k, v)
                    b = _mark(c)
                    _sync(c)
                row["ms"] = _ms(c, a, b) / c.sdpa_iters
                row["tflops"] = _tflops(flops, row["ms"])
            except Exception as e:  # noqa: BLE001 - unsupported backend
                msgs = list(dict.fromkeys(str(w.message) for w in wl))[:4]
                row["error"] = " | ".join([f"{type(e).__name__}: {e}"]
                                          + msgs)[:1200]
        rows.append(row)
        if c.cuda:
            torch.cuda.empty_cache()
    out["rows"] = rows
    # which backend the default runs: its kernel names under the profiler
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if c.cuda else [])
    with profile(activities=acts) as prof, torch.no_grad():
        F.scaled_dot_product_attention(q, k, v)
        _sync(c)
    dev = DeviceType.CUDA if c.cuda else DeviceType.CPU
    names = list(dict.fromkeys(
        e.name for e in prof.events() if e.device_type == dev
        and not getattr(e, "is_user_annotation", False)))
    out["default_kernels"] = [nm[:160] for nm in names[:8]]
    out["default_by_kernel"] = sdpa_backend_from_kernels(names)
    d_ms = rows[0].get("ms")
    ok = [r for r in rows[1:] if r.get("ms")]
    if d_ms and ok:
        out["default_by_timing"] = min(
            ok, key=lambda r: abs(r["ms"] - d_ms))["backend"]
    return out


def sec_throughput(c):
    checks = (c.res["sections"].get("parity") or {}).get("checks") or {}
    encs = getattr(c, "encoders", {})
    z = c.vae.vae.config.z_dim
    rows = {}
    for v in VARIANTS:
        chk = checks.get(v) or {}
        r = {"ok": bool(chk.get("ok")),
             "fail_reasons": chk.get("fail_reasons"),
             "bf16_mean_max_rel": chk.get("bf16_mean_max_rel"),
             "fp32_mean_max_rel": chk.get("fp32_mean_max_rel")}
        if v not in encs:
            r["skipped"] = "failed or skipped parity: unusable"
        else:
            c.torch.backends.cudnn.benchmark = True
            r["ladder"] = c.run_ladder(
                encs[v], c.imgs, (c.batch,), (z, c.h // 8, c.w // 8),
                cuda=c.torch.cuda if c.cuda else None,
                min_images=c.min_images, min_seconds=c.min_seconds,
                log=c.log, intra_threads=1)
            ok = [x for x in r["ladder"] if "img_s" in x]
            if ok:
                r["img_s"] = ok[0]["img_s"]
                r["tflops"] = (r["img_s"] * c.flops_img / 1e12
                               if getattr(c, "flops_img", None) else None)
        rows[v] = r
    return {"variants": rows}


def sec_max_autotune(c):
    """Parity gate + ladder of MAX_AUTOTUNE. compile_s = its first bf16 call
    (the fp32 parity compile comes on top: parity_s)."""
    t0 = time.monotonic()
    checks, encs = c.check_impls(
        c.vae, [MAX_AUTOTUNE], c.x_cpu[:CHECK_IMAGES],
        cuda=c.torch.cuda if c.cuda else None, log=c.log,
        factory={MAX_AUTOTUNE: max_autotune_encoder})
    chk = checks[MAX_AUTOTUNE]
    r = {"check": chk, "ok": bool(chk.get("ok")),
         "compile_s": chk.get("first_call_s"),
         "parity_s": time.monotonic() - t0}
    if MAX_AUTOTUNE not in encs:
        r["skipped"] = "failed parity: unusable"
        return r
    z = c.vae.vae.config.z_dim
    r["ladder"] = c.run_ladder(
        encs[MAX_AUTOTUNE], c.imgs, (c.batch,), (z, c.h // 8, c.w // 8),
        cuda=c.torch.cuda if c.cuda else None, min_images=c.min_images,
        min_seconds=c.min_seconds, log=c.log, intra_threads=1)
    ok = [x for x in r["ladder"] if "img_s" in x]
    if ok:
        r["img_s"] = ok[0]["img_s"]
        r["tflops"] = (r["img_s"] * c.flops_img / 1e12
                       if getattr(c, "flops_img", None) else None)
    return r


SECTIONS = (("flops", sec_flops, False), ("stages", sec_stages, True),
            ("parity", sec_parity, True), ("profiler", sec_profiler, True),
            ("sdpa", sec_sdpa, True), ("throughput", sec_throughput, True),
            ("max_autotune", sec_max_autotune, True))
NEEDS_HELPERS = ("parity", "throughput", "max_autotune")


def profile_impl(vae, device="cuda", *, h=1216, w=832, batch=4,
                 budget_s=600.0, warmup=3, stage_iters=10, prof_iters=3,
                 sdpa_iters=5, n_imgs=8, min_images=64, min_seconds=10.0,
                 trace_path=None, check_impls=None, run_ladder=None,
                 mat_start_by=None, checkpoint=None, cpu_ok=False,
                 log=print):
    """Profile `vae` (cache_dataset.VaeEncoder-like, at its own dtype) on
    `device` at (h, w), batch `batch`. check_impls / run_ladder: the bench
    helpers of scripts/modal_cache_latents.py (injected; check_impls must take
    factory=). budget_s bounds the sections up to throughput; the
    max_autotune section ignores it and starts only while time.monotonic()
    <= mat_start_by (None = never). checkpoint(res) runs after every
    section but the last (save the report so far: a long compile may
    outlive the caller's timeout). -> plain
    result dict (render_markdown). Never raises."""
    t0 = time.monotonic()
    c = _Ctx()
    res = {"params": {"h": h, "w": w, "batch": batch, "budget_s": budget_s,
                      "warmup": warmup, "stage_iters": stage_iters,
                      "prof_iters": prof_iters, "sdpa_iters": sdpa_iters,
                      "seed": SEED, "variants": list(VARIANTS)},
           "sections": {}}
    try:
        import torch
        import vae_fast

        c.torch, c.vae, c.log, c.res = torch, vae, log, res
        c.cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        c.h, c.w, c.batch, c.warmup = h, w, batch, warmup
        c.stage_iters, c.prof_iters, c.sdpa_iters = (stage_iters, prof_iters,
                                                    sdpa_iters)
        c.min_images, c.min_seconds = min_images, min_seconds
        c.trace_path, c.check_impls, c.run_ladder = (trace_path, check_impls,
                                                     run_ladder)
        env = {"device": str(device), "torch": torch.__version__,
               "cuda": torch.version.cuda, "dtype": str(vae.dtype)}
        if c.cuda:
            env.update(device_name=torch.cuda.get_device_name(0),
                       cudnn=torch.backends.cudnn.version())
            env["peak_bf16_tflops"] = next(
                (p for k, p in PEAK_BF16_TFLOPS.items()
                 if k in env["device_name"]), None)
        res["env"] = env
        g = torch.Generator().manual_seed(SEED)
        c.x_cpu = torch.rand(max(batch, CHECK_IMAGES), 3, h, w,
                             generator=g) * 2 - 1
        c.imgs = list(torch.rand(n_imgs, 3, h, w, generator=g) * 2 - 1)
        c.x = c.x_cpu[:batch].to(device, vae.dtype)
        c.net = vae_fast.make_encoder(vae, "2d").net   # eager, cudnn.benchmark
        # every compiled variant x dtype is another dynamo cache entry of
        # Encoder2d.forward (up to 6 here); past the limit dynamo would fall
        # back to eager silently and a variant would look slow
        import torch._dynamo
        for k in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, k):
                setattr(torch._dynamo.config, k,
                        max(64, getattr(torch._dynamo.config, k)))
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-3000:]
        return res
    for name, fn, needs_cuda in SECTIONS:
        if needs_cuda and not c.cuda and not cpu_ok:
            res["sections"][name] = {"skipped": "no CUDA"}
            continue
        if name in NEEDS_HELPERS and not (check_impls and run_ladder):
            res["sections"][name] = {"skipped": "no bench helpers"}
            continue
        used = time.monotonic() - t0
        if name == "max_autotune":
            if mat_start_by is None or time.monotonic() > mat_start_by:
                res["sections"][name] = {"skipped": (
                    "not requested" if mat_start_by is None else
                    f"would not finish inside the container timeout "
                    f"({used:.0f}s used)")}
                continue
        elif used > budget_s:
            res["sections"][name] = {"skipped": f"budget ({used:.0f}s used)"}
            continue
        log(f"=== {name} ({used:.0f}s used)")
        ts = time.monotonic()
        try:
            with torch.no_grad():
                out = fn(c)
        except Exception:  # noqa: BLE001 - one section never stops the rest
            out = {"error": traceback.format_exc()[-3000:]}
            log(out["error"])
        out["seconds"] = time.monotonic() - ts
        res["sections"][name] = out
        if c.cuda:
            torch.cuda.empty_cache()
        res["seconds"] = time.monotonic() - t0
        if checkpoint and name != SECTIONS[-1][0]:
            try:   # a later section may run into the caller's timeout
                checkpoint(res)
            except Exception as e:  # noqa: BLE001
                log(f"checkpoint failed: {type(e).__name__}: {e}")
    res["seconds"] = time.monotonic() - t0
    return res


# ------------------------------------------------------------- markdown
def _f(x, spec=".2f", none="-"):
    return none if x is None else format(x, spec)


def _table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(str(v) for v in r) + " |" for r in rows]


def _last_line(err):
    lines = [ln for ln in str(err).strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else "?"


def render_markdown(res, title="VAE encoder profile"):
    """Plain result dict (profile_impl) -> markdown report (no torch)."""
    p, env, s = res.get("params", {}), res.get("env", {}), res.get("sections",
                                                                   {})
    out = [f"# {title} ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    out += [f"{env.get('device_name', env.get('device'))}, torch "
            f"{env.get('torch')} (cuda {env.get('cuda')}, cudnn "
            f"{env.get('cudnn')}), dtype {env.get('dtype')}; image "
            f"{p.get('h')}x{p.get('w')} (HxW), batch {p.get('batch')}, seeded "
            f"uniform [-1, 1] inputs; total {_f(res.get('seconds'), '.0f')} s "
            f"(budget {p.get('budget_s')} s).", ""]
    if res.get("error"):
        out += ["**setup FAILED**: `" + _last_line(res["error"]) + "`", ""]
    peak = env.get("peak_bf16_tflops")
    pk = (lambda t: f" ({100 * t / peak:.1f} % of {peak:g} peak)"
          if t and peak else "")
    fl = s.get("flops") or {}
    thr = ((s.get("throughput") or {}).get("variants") or {}).get(BASELINE, {})
    prof = s.get("profiler") or {}
    out += ["## Headline", ""]
    if fl.get("per_image_flops"):
        out.append(f"- FLOPs/image: {fl['per_image_flops'] / 1e12:.3f} TFLOP "
                   f"(hand count scaled to this size "
                   f"{fl['hand_flops_scaled'] / 1e12:.3f}; ratio "
                   f"{_f(fl.get('ratio_vs_hand'), '.3f')})")
    if thr.get("img_s"):
        out.append(f"- {BASELINE} production loop (run_ladder, batch "
                   f"{p.get('batch')}): {thr['img_s']:.2f} img/s = "
                   f"{_f(thr.get('tflops'), '.1f')} TFLOP/s{pk(thr.get('tflops'))}")
    if prof.get("fwd_ms"):
        out.append(f"- {BASELINE} forward only (input on device): "
                   f"{prof['fwd_ms']:.2f} ms/batch = {prof['fwd_img_s']:.2f} "
                   f"img/s = {_f(prof.get('fwd_tflops'), '.1f')} TFLOP/s"
                   f"{pk(prof.get('fwd_tflops'))}")
    st = s.get("stages") or {}
    if st.get("total_ms"):
        out.append(f"- 2d eager forward (stage sum): {st['total_ms']:.2f} "
                   f"ms/batch = {st['img_s']:.2f} img/s = "
                   f"{_f(st.get('tflops'), '.1f')} TFLOP/s{pk(st.get('tflops'))}")
    if prof.get("idle_pct") is not None:
        out.append(f"- GPU idle during the profiled production steps: "
                   f"{prof['idle_pct']:.1f} % (profiler overhead included)")
    out.append("")

    def status(name):
        d = s.get(name)
        if d is None:
            return ["(not run)", ""]
        if d.get("skipped"):
            return [f"skipped: {d['skipped']}", ""]
        if d.get("error"):
            return [f"**FAILED**: `{_last_line(d['error'])}`", ""]
        return None

    out += ["## a. FLOPs (FlopCounterMode, eager 2d, 1 image)", ""]
    out += status("flops") or _table(
        ["stage", "GFLOP/img", "%"],
        [[r["stage"], _f(r["flops"] / 1e9, ".1f"), _f(r["pct"], ".1f")]
         for r in fl["per_stage"]]) + [
        "", f"Total {fl['per_image_flops'] / 1e12:.3f} TFLOP/img; the user's "
        f"hand count {HAND_TFLOP_PER_IMG} TFLOP at 832x1216 scales to "
        f"{fl['hand_flops_scaled'] / 1e12:.3f}. Convs/matmuls/SDPA only "
        "(elementwise ops are not counted). SDPA: analytic 4*N^2*D "
        f"{fl['sdpa_flops'] / 1e9:.3f} GFLOP/img; in the total "
        f"{fl['sdpa_counted_flops'] / 1e9:.3f} GFLOP/img, "
        + (f"analytic, added for {', '.join(fl['sdpa_added'])} (the counter "
           "has no formula for that kernel)" if fl.get("sdpa_added") else
           "counted by FlopCounterMode inside the SDPA call(s) via "
           + (", ".join(f"`{o}`" for o in fl.get("sdpa_ops") or []) or "-"))
        + ".", ""]
    out += ["## b. Stage timing (eager 2d, CUDA events, device-resident "
            "input)", ""]
    out += status("stages") or _table(
        ["stage", "ms/batch", "% time", "GFLOP/batch", "TFLOP/s"],
        [[r["stage"], _f(r["ms"], ".3f"), _f(r["pct"], ".1f"),
          _f(r["flops_batch"] and r["flops_batch"] / 1e9, ".1f"),
          _f(r["tflops"], ".1f")] for r in st["rows"]]) + [
        "", f"Mean of {st['iters']} iterations after {st['warmup']} warmup; "
        f"total {st['total_ms']:.2f} ms/batch.", ""]
    par = s.get("parity") or {}
    out += ["## Parity (check_impls on 2 seeded images)", ""]
    out += status("parity") or _table(
        ["variant", "ok", "fp32 mean/std/sample", "bf16 mean/std (ref)",
         "first call s", "fail reasons"],
        [[v, c.get("ok"),
          "/".join(_f(c.get(f"fp32_{q}_max_rel"), ".1e")
                   for q in ("mean", "std", "sample")),
          "/".join(_f(c.get(f"bf16_{q}_max_rel"), ".1e")
                   for q in ("mean", "std"))
          + f" ({_f(c.get('ref_bf16_mean_max_rel'), '.1e')})",
          _f(c.get("first_call_s"), ".1f"),
          "; ".join(c.get("fail_reasons") or []) or "-"]
         for v, c in par["checks"].items()]) + [""]
    out += [f"## c. torch.profiler ({BASELINE}, production steps)", ""]
    sc = status("profiler")
    if sc:
        out += sc
    else:
        if prof.get("note"):
            out += [f"Note: {prof['note']}", ""]
        out += [f"{prof['iters']} steps (stack + H2D + encode + bf16 D2H): "
                f"wall {prof['wall_us'] / 1e3:.1f} ms, GPU busy "
                f"{prof['busy_us'] / 1e3:.1f} ms, idle "
                f"{_f(prof['idle_pct'], '.1f')} %, kernel sum "
                f"{prof['kernel_us'] / 1e3:.1f} ms; "
                f"{prof['step_img_s']:.2f} img/s under the profiler. Trace: "
                f"{prof.get('trace', '-')} ({_f(prof.get('trace_bytes'), 'd')}"
                " bytes, gzip).", ""]
        out += _table(["category", "ms", "% kernel time", "kernels"],
                      [[k, _f(v["us"] / 1e3, ".2f"), _f(v["pct"], ".1f"),
                        v["count"]] for k, v in prof["categories"].items()])
        out += ["", "Top kernels:", ""]
        out += _table(["kernel", "cat", "ms", "%", "calls"],
                      [["`" + t["name"].replace("|", "/") + "`",
                        t["category"], _f(t["us"] / 1e3, ".3f"),
                        _f(t["pct"], ".1f"), t["count"]]
                       for t in prof["top"]])
        if prof.get("other_top_names"):
            out += ["", "Uncategorized (other): "
                    + ", ".join(f"`{n}`" for n in prof["other_top_names"])]
        out.append("")
    sd = s.get("sdpa") or {}
    out += ["## d. SDPA backends (mid attention shape)", ""]
    sc = status("sdpa")
    if sc:
        out += sc
    else:
        out += [f"q/k/v {sd['shape']} {sd['dtype']} ({len(sd['calls'])} SDPA "
                f"call(s) per forward, captured from the eager encoder); "
                f"{sd['flops'] / 1e12:.3f} TFLOP per call (4 * B*H * N^2 * "
                "D).", ""]
        out += _table(["backend", "ms", "TFLOP/s", "error"],
                      [[r["backend"], _f(r.get("ms"), ".3f"),
                        _f(r.get("tflops"), ".1f"),
                        (r.get("error") or "-").replace("|", "/")[:400]]
                       for r in sd["rows"]])
        out += ["", f"Default dispatch: {sd.get('default_by_kernel')} by its "
                f"kernel names under torch.profiler "
                f"({', '.join('`' + k + '`' for k in sd['default_kernels'])}); "
                f"closest timing: {sd.get('default_by_timing', '-')}.", ""]
    th = (s.get("throughput") or {}).get("variants") or {}
    out += [f"## e. Variant throughput (run_ladder, batch {p.get('batch')})",
            ""]
    out += status("throughput") or _table(
        ["variant", "parity ok", "img/s", "TFLOP/s", "bf16 mean err",
         "note"],
        [[v, r["ok"], _f(r.get("img_s")), _f(r.get("tflops"), ".1f"),
          _f(r.get("bf16_mean_max_rel"), ".1e"),
          r.get("skipped") or "; ".join(
              x.get("error", "") for x in r.get("ladder") or []
              if x.get("error")) or "-"]
         for v, r in th.items()]) + [""]
    ma = s.get("max_autotune") or {}
    out += [f"## e2. {MAX_AUTOTUNE}: torch.compile mode="
            "max-autotune-no-cudagraphs (run last, if time allowed)", ""]
    sc = status("max_autotune")
    if sc and not ma.get("check"):
        out += sc
    else:
        chk = ma.get("check") or {}
        out += [f"- parity ok: {ma.get('ok')} (fp32 mean/std/sample "
                + "/".join(_f(chk.get(f"fp32_{q}_max_rel"), ".1e")
                           for q in ("mean", "std", "sample"))
                + ", bf16 mean/std "
                + "/".join(_f(chk.get(f"bf16_{q}_max_rel"), ".1e")
                           for q in ("mean", "std"))
                + (f"; FAILED: {'; '.join(chk.get('fail_reasons') or [])}"
                   if chk and not ma.get("ok") else "") + ")",
                f"- compile: {_f(ma.get('compile_s'), '.1f')} s (first bf16 "
                f"call); whole parity incl. the fp32 compile "
                f"{_f(ma.get('parity_s'), '.1f')} s",
                f"- steady state (run_ladder, batch {p.get('batch')}): "
                f"{_f(ma.get('img_s'))} img/s = "
                f"{_f(ma.get('tflops'), '.1f')} TFLOP/s"
                + (f" (vs {BASELINE} {_f(thr.get('img_s'))})"
                   if thr.get("img_s") else "")
                + "".join(f"; ladder error {x['error']}"
                          for x in ma.get("ladder") or [] if x.get("error")),
                ""]
    return "\n".join(out)
