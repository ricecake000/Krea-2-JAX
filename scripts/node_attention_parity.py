"""Attention parity + timing for the jax-aiter path (gfx950), CPU-testable.

Kernel level: random q/k/v (B, H, L, D) and random PREFIX key masks (per
sample valid length; the layout dit_forward produces for attn_impl="aiter"
after its per-sample token reorder). Compares

  * test:  aiter_attention(kernel=None)                     (--kernel aiter)
           aiter_attention(kernel=varlen_reference_kernel)  (--kernel emul)
  * ref:   scaled_dot_product_attention(keymask) with padded query rows zeroed
           (the kernel contract), run in the same dtype as the test

against an fp32 "truth" (the ref math on the same dtype-rounded inputs
upcast to fp32), forward and grads (dq, dk, dv). Prints max abs error and
max abs / max |truth| for each (the ref-in-dtype row is the rounding
baseline the kernel should be comparable to) plus median per-call ms for
the jitted forward and forward+backward. Exits non-zero when any test error
(max abs / max |truth|) exceeds --tol (default 2e-2 bf16, 1e-4 f32).
TFLOP/s: full-query-equivalent = 4*Hq*L*sum(lens)*D, logical = 4*Hq*sum(lens^2)*D (fwd; bwd ~2.5x).

Optional DiT level (--base-ckpt real weights, or --random-weights with a tiny
--model-config): dit_forward attn_impl="ref" vs "aiter" on random inputs at
--width/--height with random text padding (padding in the middle of the
sequence, so the aiter path's token reorder is exercised); per-sample
relative-norm error of the image-token velocities vs --dit-tol; --dit-grad
also compares d(loss)/d(img tokens) under remat.

On the node:
    python scripts/node_attention_parity.py --kernel aiter
    python scripts/node_attention_parity.py --kernel aiter --base-ckpt raw.safetensors

CPU smoke:
    JAX_PLATFORMS=cpu python scripts/node_attention_parity.py --kernel emul \
        --dtype f32 --seq 512 --heads 4 --kv-heads 2 --head-dim 32 --iters 2
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

from parity_full import relnorm  # the single float64-safe ||a-b|| / ||ref||


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=48)
    ap.add_argument("--kv-heads", type=int, default=12)
    ap.add_argument("--seq", type=int, default=4608)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--kernel", choices=["aiter", "emul", "emul-nomask"],
                    default="aiter",
                    help="aiter = jax_aiter (gfx950); emul = "
                         "model.varlen_reference_kernel; emul-nomask = negative "
                         "control that ignores the logical lengths (must FAIL)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--qchunk", type=int, default=512,
                    help="query chunk of the reference attention")
    ap.add_argument("--tol", type=float, default=None,
                    help="default 2e-2 for bf16, 1e-4 for f32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", action="store_true",
                    help="shard the batch over all devices (exercises the "
                         "shard_map wrapper; --batch must divide evenly)")
    ap.add_argument("--emulate-devices", type=int, default=0,
                    help="CPU: fake N host devices (sets XLA_FLAGS before jax)")
    ap.add_argument("--skip-kernel", action="store_true",
                    help="skip the kernel-level check (only the DiT check)")
    # DiT-level check
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--base-ckpt", default=None,
                     help="official .safetensors: also run the DiT-level check")
    src.add_argument("--random-weights", action="store_true",
                     help="DiT-level check with random N(0, 0.02) weights")
    ap.add_argument("--model-config", default=None,
                    help="JSON dict of Krea2Config overrides (DiT check)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--txt-len", type=int, default=77)
    ap.add_argument("--dit-batch", type=int, default=None,
                    help="DiT-check batch (default --batch)")
    ap.add_argument("--act-sharding", choices=["on", "off"], default="off",
                    help="DiT check: batch-shard activations")
    ap.add_argument("--dit-grad", action="store_true",
                    help="DiT check: also compare grads wrt the image tokens")
    ap.add_argument("--dit-tol", type=float, default=None,
                    help="max per-sample relnorm (default 5e-2 bf16, 1e-4 f32)")
    return ap.parse_args(argv)


def timeit(fn, args, iters, warmup):
    import jax

    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)) if ts else float("nan")


def errs(a, truth):
    a = np.asarray(a, np.float64)
    truth = np.asarray(truth, np.float64)
    if not np.all(np.isfinite(a)):
        return float("inf"), float("inf")
    mx = float(np.abs(a - truth).max())
    return mx, mx / (float(np.abs(truth).max()) + 1e-30)


def get_kernel(name):
    if name == "emul":
        from krea2_jax.model import varlen_reference_kernel
        return varlen_reference_kernel
    if name == "emul-nomask":
        from krea2_jax.model import varlen_reference_kernel

        def nomask(q3, k3, v3, sq, sk, cu_q, cu_k, *a, **kw):
            full = sq  # physical offsets as logical: every token "valid"
            return varlen_reference_kernel(q3, k3, v3, sq, sk, full, full, *a, **kw)
        return nomask
    return None  # aiter_attention imports jax_aiter.mha.flash_attn_varlen


def make_mesh_for(batch, shard):
    import jax
    from jax.sharding import Mesh

    if not shard:
        return None
    devs = jax.devices()
    n = max(d for d in range(1, len(devs) + 1) if batch % d == 0)
    return Mesh(np.array(devs[:n]), axis_names=("fsdp",))


def kernel_check(args, tol):
    import jax
    import ml_dtypes
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from krea2_jax.model import aiter_attention, scaled_dot_product_attention

    B, Hq, Hk, L, D = args.batch, args.heads, args.kv_heads, args.seq, args.head_dim
    if Hq % Hk:
        raise SystemExit("--heads must be a multiple of --kv-heads")
    rng = np.random.default_rng(args.seed)
    lens = rng.integers(max(1, L // 4), L, size=B) if L > 1 else np.ones(B, int)
    if B > 1:
        lens[-1] = L  # one fully valid sample
    keymask_np = np.arange(L)[None, :] < lens[:, None]

    mesh = make_mesh_for(B, args.shard)
    # inputs are built and cast on the host (numpy / ml_dtypes) and placed
    # straight into their target sharding: no device holds the global batch
    sharding = None if mesh is None else NamedSharding(mesh, P("fsdp"))
    put = lambda a: jax.device_put(a, sharding)
    np_dt = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float32

    q = put(rng.standard_normal((B, Hq, L, D)).astype(np.float32).astype(np_dt))
    k = put(rng.standard_normal((B, Hk, L, D)).astype(np.float32).astype(np_dt))
    v = put(rng.standard_normal((B, Hk, L, D)).astype(np.float32).astype(np_dt))
    w = put(rng.standard_normal((B, L, Hq * D)).astype(np.float32))
    km = put(keymask_np)
    kern = get_kernel(args.kernel)

    print(f"kernel check: B={B} Hq={Hq} Hk={Hk} L={L} D={D} dtype={args.dtype} "
          f"kernel={args.kernel} mesh={'none' if mesh is None else mesh.size} "
          f"valid lens={lens.tolist()}")

    def test_fn(q, k, v, km):
        return aiter_attention(q, k, v, keymask=km, mesh=mesh, kernel=kern)

    def ref_fn(q, k, v, km):
        out = scaled_dot_product_attention(q, k, v, keymask=km, qchunk=args.qchunk)
        return out * km[:, :, None].astype(out.dtype)

    def grads_of(f):
        def g(q, k, v, km, w):
            loss = lambda q, k, v: jnp.sum(f(q, k, v, km).astype(jnp.float32) * w)
            return jax.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v)
        return jax.jit(g)

    test_f, ref_f = jax.jit(test_fn), jax.jit(ref_fn)
    test_g, ref_g = grads_of(test_fn), grads_of(ref_fn)

    q32, k32, v32 = (a.astype(jnp.float32) for a in (q, k, v))
    truth_out = np.asarray(ref_f(q32, k32, v32, km))
    _, truth_g = ref_g(q32, k32, v32, km, w)
    truth_g = [np.asarray(a) for a in truth_g]

    out_t = np.asarray(test_f(q, k, v, km).astype(jnp.float32))
    out_r = np.asarray(ref_f(q, k, v, km).astype(jnp.float32))
    _, g_t = test_g(q, k, v, km, w)
    _, g_r = ref_g(q, k, v, km, w)

    pad_rows = ~keymask_np
    pad_max = float(np.abs(out_t[pad_rows]).max()) if pad_rows.any() else 0.0
    print(f"  test output on padded query rows: max |out| = {pad_max:.3e} "
          f"(kernel contract: zeros)")

    print(f"  {'quantity':<8} {'impl':<12} {'max_abs':>10} {'max_rel':>10}")
    ok = True
    rows = [("out", out_t, out_r, truth_out)] + [
        (f"d{n}", np.asarray(a.astype(jnp.float32)),
         np.asarray(b.astype(jnp.float32)), c)
        for n, a, b, c in zip("qkv", g_t, g_r, truth_g)
    ]
    for name, t_arr, r_arr, tr in rows:
        ta, trl = errs(t_arr, tr)
        ra, rrl = errs(r_arr, tr)
        bad = not trl <= tol
        ok &= not bad
        print(f"  {name:<8} {'test':<12} {ta:>10.3e} {trl:>10.3e}"
              f"{'  <-- FAIL' if bad else ''}")
        print(f"  {name:<8} {'ref(' + args.dtype + ')':<12} {ra:>10.3e} {rrl:>10.3e}")

    if args.iters > 0:
        ms = {
            "test fwd": timeit(test_f, (q, k, v, km), args.iters, args.warmup),
            "ref fwd": timeit(ref_f, (q, k, v, km), args.iters, args.warmup),
            "test fwd+bwd": timeit(test_g, (q, k, v, km, w), args.iters, args.warmup),
            "ref fwd+bwd": timeit(ref_g, (q, k, v, km, w), args.iters, args.warmup),
        }
        # non-causal attention FLOPs, fwd (bwd ~2.5x fwd, so fwd+bwd 3.5x):
        #   full-query-equivalent: every one of the L query rows (padded rows
        #     included) over the valid keys, 4*Hq*L*sum(lens)*D — what the
        #     dense ref computes;
        #   logical: only valid queries over valid keys, 4*Hq*sum(lens^2)*D —
        #     the work a varlen kernel actually has to do.
        flops_fq = 4.0 * Hq * L * float(lens.sum()) * D
        flops_lg = 4.0 * Hq * float((lens.astype(np.float64) ** 2).sum()) * D
        for name, t in ms.items():
            mult = 3.5 if "bwd" in name else 1.0
            tf = lambda fl: fl * mult / (t * 1e-3) / 1e12
            print(f"  {name:<14} {t:>9.3f} ms/call  (~{tf(flops_fq):.2f} "
                  f"full-query-equivalent TFLOP/s, ~{tf(flops_lg):.2f} logical "
                  f"TFLOP/s)")
    print(f"kernel check {'PASSED' if ok else 'FAILED'} (tol {tol:.1e} on "
          f"max_abs / max|truth|)")
    return ok


def dit_report(name, x, y, tol):
    """Per-sample relnorm of x vs reference y (parity_full.relnorm: float64
    before any norm, non-finite -> inf). Prints one line per sample plus the
    verdict; returns True iff every sample's relnorm is finite and <= tol."""
    worst = 0.0
    for b in range(x.shape[0]):
        xb = np.asarray(x[b], np.float64)
        yb = np.asarray(y[b], np.float64)
        rn = relnorm(xb, yb)
        d = np.abs(xb - yb).max()
        mr = float(d / (np.abs(yb).max() + 1e-30))
        worst = max(worst, rn)
        print(f"  {name} sample {b}: relnorm {rn:.3e}, max_rel {mr:.3e}")
    bad = not (math.isfinite(worst) and worst <= tol)
    print(f"  {name}: worst relnorm {worst:.3e} {'> ' if bad else '<= '}"
          f"tol {tol:.1e}{'  <-- FAIL' if bad else ''}")
    return not bad


def random_params(cfg, dtype, seed):
    import jax
    import jax.numpy as jnp

    from krea2_jax.checkpoint import abstract_krea2_params

    abstract = abstract_krea2_params(cfg, dtype=jnp.float32)
    leaves, treedef = jax.tree.flatten(abstract)
    key = jax.random.PRNGKey(seed + 1)
    leaves = [
        (jax.random.normal(jax.random.fold_in(key, i), s.shape, jnp.float32) * 0.02
         ).astype(dtype)
        for i, s in enumerate(leaves)
    ]
    return jax.tree.unflatten(treedef, leaves)


def dit_check(args, dit_tol):
    import jax
    import ml_dtypes
    import jax.numpy as jnp
    from jax.sharding import Mesh

    from krea2_jax.checkpoint import batch_sharding, load_krea2_params
    from krea2_jax.config import KREA2_CONFIG
    from krea2_jax.model import dit_forward
    from parity_full import make_inputs

    cfg = KREA2_CONFIG
    if args.model_config:
        cfg = dataclasses.replace(cfg, **json.loads(args.model_config))
    B = args.dit_batch or args.batch
    dt = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    devs = jax.devices()
    n = max(d for d in range(1, len(devs) + 1) if B % d == 0)
    mesh = Mesh(np.array(devs[:n]), axis_names=("fsdp",))

    t0 = time.perf_counter()
    if args.base_ckpt:
        params = load_krea2_params(args.base_ckpt, mesh=mesh, dtype=dt)
    else:
        from krea2_jax.checkpoint import param_shardings
        params = random_params(cfg, dt, args.seed)
        params = jax.device_put(params, param_shardings(params, mesh))
    print(f"dit check: params ready in {time.perf_counter() - t0:.1f}s "
          f"(features={cfg.features} layers={cfg.layers}), mesh of {n} device(s)")

    ns = argparse.Namespace(width=args.width, height=args.height, batch=B,
                            txt_len=args.txt_len, seed=args.seed)
    inp = make_inputs(cfg, ns)
    shard = batch_sharding(mesh)
    # host-side numpy cast, then straight into the batch sharding
    np_dt = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float32
    put = lambda a, d=None: jax.device_put(np.asarray(a, d), shard)
    img, context = put(inp["img"], np_dt), put(inp["context"], np_dt)
    t, pos, mask = put(inp["t"], np.float32), put(inp["pos"], np.float32), put(inp["mask"])
    print(f"  {inp['htok'] * inp['wtok']} img tokens + {args.txt_len} txt, "
          f"txt valid lens {inp['lens'].tolist()}, act_sharding={args.act_sharding}")

    common = dict(qchunk=args.qchunk)
    if args.act_sharding == "on":
        common["act_sharding"] = shard
    aiter_kw = dict(attn_impl="aiter", attn_mesh=mesh)
    kern = get_kernel(args.kernel)
    if kern is not None:
        aiter_kw["attn_kernel"] = kern

    def fwd(kw):
        return jax.jit(lambda p, *a: dit_forward(p, cfg, *a, remat=False, **common, **kw))

    f_ref, f_ai = fwd(dict(attn_impl="ref")), fwd(aiter_kw)
    a = (params, img, context, t, pos, mask)
    out_r = np.asarray(f_ref(*a).astype(jnp.float32))
    out_a = np.asarray(f_ai(*a).astype(jnp.float32))

    ok = dit_report("aiter-vs-ref fwd", out_a, out_r, dit_tol)

    if args.dit_grad:
        rng = np.random.default_rng(args.seed + 7)
        wcot = put(rng.standard_normal(out_r.shape).astype(np.float32))

        def gfn(kw):
            def loss(x, p, c, t, pos, mask):
                o = dit_forward(p, cfg, x, c, t, pos, mask, remat="full",
                                **common, **kw)
                return jnp.sum(o.astype(jnp.float32) * wcot)
            return jax.jit(jax.grad(loss))

        ga = (img, params, context, t, pos, mask)
        g_r = np.asarray(gfn(dict(attn_impl="ref"))(*ga).astype(jnp.float32))
        g_a = np.asarray(gfn(aiter_kw)(*ga).astype(jnp.float32))
        ok &= dit_report("aiter-vs-ref d/d(img)", g_a, g_r, dit_tol)

    if args.iters > 0:
        it = max(1, min(args.iters, 5))
        print(f"  dit fwd ms: ref {timeit(f_ref, a, it, 1):.1f}, "
              f"aiter {timeit(f_ai, a, it, 1):.1f} (median of {it})")
    print(f"dit check {'PASSED' if ok else 'FAILED'}")
    return ok


def main(argv=None):
    args = parse_args(argv)
    if args.emulate_devices:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={args.emulate_devices}"
        )
    import jax

    tol = args.tol if args.tol is not None else (2e-2 if args.dtype == "bf16" else 1e-4)
    dit_tol = args.dit_tol if args.dit_tol is not None else (
        5e-2 if args.dtype == "bf16" else 1e-4)
    print(f"jax {jax.__version__}, platform {jax.devices()[0].platform}, "
          f"{len(jax.devices())} device(s): {jax.devices()[0].device_kind}")

    ok = True
    if not args.skip_kernel:
        ok &= kernel_check(args, tol)
    if args.base_ckpt or args.random_weights:
        ok &= dit_check(args, dit_tol)
    print("ALL ATTENTION PARITY CHECKS PASSED" if ok else "ATTENTION PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
