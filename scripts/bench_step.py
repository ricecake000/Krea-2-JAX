"""Step-time benchmark for the Krea 2 DiT: forward, forward+backward, or the
full fine-tuning train step, on random inputs (no dataset cache needed).

Modes:
    fwd     jitted dit_forward on bf16 weights
    fwdbwd  jitted value_and_grad of the flow-matching loss w.r.t. the bf16
            weights (grads cast to --grad-dtype; what a frozen-base / LoRA-like
            step costs without the optimizer)
    train   the real scripts/train_full.py step (train_full.make_train_step:
            fp32 master weights, AdamW, EMA, donation)

Reports the median step time, samples/s, analytic TFLOP/s per device
(6 * n_params * padded_tokens * batch, x4/3 for --remat-policy full, x1.1
for dots; 2 * ... for fwd) and the peak device memory when the backend
reports it.

    python scripts/bench_step.py --random-weights --mode train \
        --batch-size 64 --width 1024 --height 1024 --steps 10

    # CPU smoke run with a tiny model
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python scripts/bench_step.py --random-weights --mode fwdbwd \
        --model-config '{"features":128,"tdim":32,"txtdim":64,"heads":4,
        "kvheads":2,"multiplier":4,"layers":2,"patch":2,"channels":4,
        "txtheads":4,"txtkvheads":4,"txtlayers":3}' \
        --batch-size 8 --width 128 --height 128 --txt-len 16 --steps 2

--profile-dir DIR captures a jax.profiler trace (xplane.pb + perfetto
trace.json.gz under DIR/plugins/profile/<run>/) of the first PROFILE_STEPS
timed steps; those steps are excluded from the reported timing, so --steps
must exceed PROFILE_STEPS.
"""

import argparse
import contextlib
import dataclasses
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

import train_full as tf
from krea2_jax.checkpoint import (
    abstract_krea2_params, batch_sharding, load_krea2_params, make_mesh,
    param_shardings,
)
from krea2_jax.config import KREA2_CONFIG
from krea2_jax.model import build_positions_and_mask, dit_forward, patchify
from krea2_jax.sampling import shift_time

REMAT_FLOP_FACTOR = {"full": 4.0 / 3.0, "dots": 1.1, "none": 1.0}
PROFILE_STEPS = 2


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    w = ap.add_mutually_exclusive_group(required=True)
    w.add_argument("--base-ckpt", default=None, help="raw.safetensors path")
    w.add_argument("--random-weights", action="store_true",
                   help="random weights generated on device (no checkpoint)")
    ap.add_argument("--mode", choices=["fwd", "fwdbwd", "train"], default="train")
    ap.add_argument("--batch-size", type=int, default=8, help="global batch")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--txt-len", type=int, default=512)
    ap.add_argument("--remat-policy", choices=["full", "dots", "none"],
                    default="dots")
    ap.add_argument("--attn", choices=["ref", "splash", "aiter"], default="ref")
    ap.add_argument("--attn-block", type=int, default=256)
    ap.add_argument("--act-sharding", choices=["on", "off"], default="on")
    ap.add_argument("--base-sharding", choices=["fsdp", "replicate"],
                    default="fsdp",
                    help="weight placement (train mode always keeps the fp32 "
                         "state FSDP-sharded unless this is 'replicate')")
    ap.add_argument("--grad-dtype", choices=["bf16", "f32"], default="bf16",
                    help="fwdbwd: dtype the weight gradients are cast to")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="train: 0 drops the EMA tree")
    ap.add_argument("--steps", type=int, default=10, help="timed steps")
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed steps after the compile step")
    ap.add_argument("--qchunk", type=int, default=512)
    ap.add_argument("--model-config", default=None,
                    help="JSON object of Krea2Config field overrides")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile-dir", default=None,
                    help=f"profile the first {PROFILE_STEPS} timed steps into "
                         "this directory (jax.profiler, perfetto trace); they "
                         "are excluded from the reported timing")
    args = ap.parse_args(argv)
    if args.profile_dir is not None and args.steps <= PROFILE_STEPS:
        ap.error(f"--profile-dir needs --steps > {PROFILE_STEPS} (the "
                 f"{PROFILE_STEPS} profiled steps are not timed)")
    return args


def random_params(abstract, shardings, seed, dtype):
    """Gaussian(0, 0.02) weights generated directly into their shardings
    (the full tree never exists on the host)."""
    leaves, treedef = jax.tree.flatten(abstract)
    sh_leaves = jax.tree.leaves(shardings)

    def init(key):
        keys = jax.random.split(key, len(leaves))
        return jax.tree.unflatten(treedef, [
            (0.02 * jax.random.normal(k, s.shape, jnp.float32)).astype(dtype)
            for k, s in zip(keys, leaves)
        ])

    return jax.jit(init, out_shardings=jax.tree.unflatten(treedef, sh_leaves))(
        jax.random.PRNGKey(seed))


def random_batch(rng, cfg, B, lat_h, lat_w, txt_len, sharding):
    """Random latents / text states and random valid text lengths."""
    lat = rng.standard_normal((B, cfg.channels, lat_h, lat_w), np.float32)
    txt = rng.standard_normal((B, txt_len, cfg.txtlayers, cfg.txtdim), np.float32)
    lens = rng.integers(1, txt_len + 1, size=B)
    mask = np.arange(txt_len)[None, :] < lens[:, None]
    txt = txt * mask[:, :, None, None]
    # cast on the host and place straight into the batch sharding (no device
    # ever holds the global batch)
    put = lambda a, dt: jax.device_put(np.asarray(a, dt), sharding)
    return (put(lat, ml_dtypes.bfloat16), put(txt, ml_dtypes.bfloat16),
            put(mask, np.bool_))


def peak_memory():
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    for k in ("peak_bytes_in_use", "bytes_in_use"):
        if k in stats:
            return k, stats[k], stats.get("bytes_limit")
    return None


def start_profile(log_dir):
    """jax.profiler.start_trace with a perfetto trace.json.gz when this jax
    supports create_perfetto_trace (falls back to the plain xplane/trace)."""
    try:
        jax.profiler.start_trace(log_dir, create_perfetto_trace=True)
    except TypeError:
        print("note: this jax has no create_perfetto_trace; writing the "
              "default profiler output only")
        jax.profiler.start_trace(log_dir)


def profile_files(log_dir, since):
    """Files the profiler wrote under log_dir (modified at/after `since`)."""
    root = Path(log_dir)
    return sorted(str(p) for p in root.rglob("*")
                  if p.is_file() and p.stat().st_mtime >= since - 1.0)


def main(argv=None):
    args = parse_args(argv)
    cfg = KREA2_CONFIG
    if args.model_config:
        cfg = dataclasses.replace(cfg, **json.loads(args.model_config))
    mesh = make_mesh()
    n_dev = mesh.size
    B = args.batch_size
    if B % n_dev:
        raise SystemExit(f"--batch-size {B} is not divisible by {n_dev} devices")
    comp = cfg.vae_compression
    if args.width % (comp * cfg.patch) or args.height % (comp * cfg.patch):
        raise SystemExit(f"width/height must be multiples of {comp * cfg.patch}")
    lat_h, lat_w = args.height // comp, args.width // comp
    h_tok, w_tok = lat_h // cfg.patch, lat_w // cfg.patch
    img_seq = h_tok * w_tok
    padded = -(-(img_seq + args.txt_len) // cfg.seq_pad_multiple) * cfg.seq_pad_multiple
    mu = cfg.mu_for_seq_len(img_seq)
    print(f"devices: {n_dev} x {jax.devices()[0].platform} | mode {args.mode} | "
          f"batch {B} | {args.width}x{args.height} ({img_seq} img + "
          f"{args.txt_len} txt -> {padded} tokens) | remat {args.remat_policy} | "
          f"attn {args.attn} | act-sharding {args.act_sharding} | "
          f"base {args.base_sharding}")

    # --- weights ---
    wdtype = jnp.float32 if args.mode == "train" else jnp.bfloat16
    t0 = time.time()
    if args.random_weights:
        abstract = abstract_krea2_params(cfg, dtype=wdtype)
        pshard = param_shardings(abstract, mesh, args.base_sharding)
        params = random_params(abstract, pshard, args.seed, wdtype)
    else:
        params = load_krea2_params(args.base_ckpt, mesh=mesh, dtype=wdtype,
                                   sharding=args.base_sharding)
        pshard = param_shardings(params, mesh, args.base_sharding)
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))
    print(f"weights ({jnp.dtype(wdtype).name}, {args.base_sharding}): "
          f"{n_params / 1e9:.3f}B params in {time.time() - t0:.1f}s")

    ds = batch_sharding(mesh)
    hp = tf.parse_args([
        "--base-ckpt", "-", "--cache-dir", "-", "--output-dir", "-",
        "--remat-policy", args.remat_policy, "--attn", args.attn,
        "--attn-block", str(args.attn_block), "--act-sharding", args.act_sharding,
        "--ema-decay", str(args.ema_decay), "--qchunk", str(args.qchunk),
    ])
    remat = tf.REMAT_MODES[args.remat_policy]
    act = tf.act_sharding_for(hp, mesh)
    fwd_kw = dict(lora=None, qchunk=args.qchunk, remat=remat,
                  attn_impl=args.attn, attn_mesh=mesh,
                  attn_block=args.attn_block, act_sharding=act)

    def make_inputs(lat, txt, txt_mask, key):
        k_t, k_n = jax.random.split(key)
        x0 = patchify(lat, cfg.patch).astype(jnp.float32)
        t = shift_time(jax.nn.sigmoid(jax.random.normal(k_t, (lat.shape[0],))), mu)
        noise = jax.random.normal(k_n, x0.shape, jnp.float32)
        tt = t[:, None, None]
        xt = ((1.0 - tt) * x0 + tt * noise).astype(jnp.bfloat16)
        pos, mask = build_positions_and_mask(txt_mask, h_tok, w_tok)
        return xt, t, pos, mask, noise - x0

    if args.mode == "fwd":
        @jax.jit
        def fwd(params, lat, txt, txt_mask, key):
            xt, t, pos, mask, _ = make_inputs(lat, txt, txt_mask, key)
            return dit_forward(params, cfg, xt, txt, t, pos, mask, **fwd_kw)

        def run(state, batch, key):
            return state, fwd(state, *batch, key)
    elif args.mode == "fwdbwd":
        gdt = jnp.bfloat16 if args.grad_dtype == "bf16" else jnp.float32

        @jax.jit
        def fwdbwd(params, lat, txt, txt_mask, key):
            xt, t, pos, mask, target = make_inputs(lat, txt, txt_mask, key)

            def loss_fn(p):
                v = dit_forward(p, cfg, xt, txt, t, pos, mask, **fwd_kw)
                return jnp.mean((v.astype(jnp.float32) - target) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            grads = jax.tree.map(lambda g: g.astype(gdt), grads)
            grads = jax.lax.with_sharding_constraint(grads, pshard)
            return loss, grads

        def run(state, batch, key):
            return state, fwdbwd(state, *batch, key)
    else:
        use_ema = hp.ema_decay > 0
        tx = tf.make_optimizer(hp, tf.make_schedule(hp))
        abstract_opt = jax.eval_shape(tx.init, params)
        opt_shard = tf.opt_state_shardings(abstract_opt, params, mesh, pshard)
        opt_state = jax.jit(tx.init, in_shardings=(pshard,),
                            out_shardings=opt_shard)(params)
        ema = (jax.jit(lambda p: jax.tree.map(jnp.copy, p),
                       out_shardings=pshard)(params) if use_ema else None)
        train_step = tf.make_train_step(
            cfg, hp, mesh, tx, mu, h_tok, w_tok, use_ema, pshard, opt_shard
        )
        params = (params, opt_state, ema)

        def run(state, batch, key):
            p, o, e, metrics = train_step(*state, *batch, key)
            return (p, o, e), metrics

    rng = np.random.default_rng(args.seed)
    batch = random_batch(rng, cfg, B, lat_h, lat_w, args.txt_len, ds)
    base_key = jax.random.PRNGKey(args.seed + 1)
    state = params
    times = []
    total = 1 + args.warmup + args.steps
    prof_first = 1 + args.warmup  # first timed step index
    prof_last = prof_first + PROFILE_STEPS - 1
    prof_t0 = None
    for i in range(total):
        key = jax.random.fold_in(base_key, i)
        if args.profile_dir is not None and i == prof_first:
            prof_t0 = time.time()
            start_profile(args.profile_dir)
        # profiled steps carry a StepTraceAnnotation so the trace analyzer can
        # bound "the two steps" exactly (profiler start/stop overhead excluded)
        ann = (jax.profiler.StepTraceAnnotation("bench_step", step_num=i)
               if args.profile_dir is not None and prof_first <= i <= prof_last
               else contextlib.nullcontext())
        t0 = time.perf_counter()
        with ann:
            state, out = run(state, batch, key)
            jax.block_until_ready(out)
            if args.mode == "train":
                jax.block_until_ready(state)
        dt = time.perf_counter() - t0
        if args.profile_dir is not None and prof_first <= i <= prof_last:
            if i == prof_last:
                jax.profiler.stop_trace()
            continue
        if i == 0:
            print(f"first step (compile + run): {dt:.1f}s")
        elif i > args.warmup:
            times.append(dt)
    if args.mode == "fwd":
        loss = None
    elif args.mode == "fwdbwd":
        loss = float(out[0])
    else:
        loss = float(out["loss"])

    med = statistics.median(times)
    if args.mode == "fwd":
        flops = 2 * n_params * padded * B
    else:
        flops = 6 * n_params * padded * B * REMAT_FLOP_FACTOR[args.remat_policy]
    tflops = flops / med / n_dev / 1e12
    print(f"step time: median {med * 1e3:.1f} ms over {len(times)} steps "
          f"(min {min(times) * 1e3:.1f}, max {max(times) * 1e3:.1f})")
    print(f"throughput: {B / med:.2f} samples/s | analytic {tflops:.2f} "
          f"TFLOP/s per device ({flops:.3e} flops/step)"
          + (f" | loss {loss:.4f}" if loss is not None else ""))
    pm = peak_memory()
    if pm:
        name, val, limit = pm
        print(f"device 0 memory: {name} {val / 2**30:.2f} GiB"
              + (f" / limit {limit / 2**30:.1f} GiB" if limit else ""))
    else:
        print("device 0 memory: not reported by this backend")
    if args.profile_dir is not None:
        print(f"profile: {PROFILE_STEPS} profiled steps excluded from timing; "
              f"files written under {args.profile_dir}:")
        for f in profile_files(args.profile_dir, prof_t0):
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
