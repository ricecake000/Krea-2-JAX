"""Ahead-of-time analysis of the full fine-tuning step (no weights, no data).

Lowers and compiles the exact `scripts/train_full.py` train step
(`train_full.make_train_step`: fp32 master weights, AdamW, EMA, bf16
forward/backward) for abstract FSDP-sharded shapes, then prints what XLA
decided: per-device memory, cost analysis and every SPMD collective it
emitted, plus a verdict on whether full activations are being moved between
devices (the failure mode `--act-sharding on` exists to prevent).

Run this first on a new node/config: it needs neither the 26 GB checkpoint
nor a dataset cache, and on a machine without accelerators it emulates N CPU
devices so the SPMD partitioning (and its collectives) can be inspected
anywhere.

    python scripts/aot_analysis.py --batch-size 64 --resolution 1024 \
        --txt-len 512 --remat-policy dots --act-sharding on

Verdict rule: a collective whose result has shape[0] == global batch and at
least --act-numel elements (default 1e7) is an activation collective.
"""

import argparse
import collections
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _has_accelerator():
    """Whether jax sees a non-CPU backend, probed in a subprocess so this
    process can still set XLA_FLAGS before its own first jax import."""
    if os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
        return False
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    try:
        proc = subprocess.run(
            [sys.executable, "-c",
             "import jax; print(jax.devices()[0].platform)"],
            env=env, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    lines = proc.stdout.strip().splitlines()
    return proc.returncode == 0 and bool(lines) and lines[-1].strip() != "cpu"


def setup_emulation(n):
    """Force n host-platform (CPU) devices. Must run before importing jax."""
    flags = os.environ.get("XLA_FLAGS", "")
    flags = re.sub(r"--xla_force_host_platform_device_count=\d+", "", flags)
    os.environ["XLA_FLAGS"] = (
        f"{flags} --xla_force_host_platform_device_count={n}".strip()
    )
    os.environ["JAX_PLATFORMS"] = "cpu"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch-size", type=int, default=64, help="global batch")
    ap.add_argument("--resolution", type=int, default=1024,
                    help="square training resolution in pixels")
    ap.add_argument("--width", type=int, default=None,
                    help="pixel width (overrides --resolution)")
    ap.add_argument("--height", type=int, default=None,
                    help="pixel height (overrides --resolution)")
    ap.add_argument("--txt-len", type=int, default=512)
    ap.add_argument("--remat-policy", choices=["full", "dots", "none"],
                    default="dots")
    ap.add_argument("--attn", choices=["ref", "splash", "aiter"], default="ref")
    ap.add_argument("--attn-block", type=int, default=256)
    ap.add_argument("--act-sharding", choices=["on", "off"], default="on")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="0 drops the EMA tree from the step")
    ap.add_argument("--qchunk", type=int, default=512)
    ap.add_argument("--timestep-mu", type=float, default=None)
    ap.add_argument("--model-config", default=None,
                    help="JSON object of Krea2Config field overrides "
                         "(default: the real 12.9B config)")
    ap.add_argument("--emulate-devices", type=int, default=None,
                    help="emulate N CPU devices (sets XLA_FLAGS before jax is "
                         "imported). Default: 8 when no accelerator is "
                         "present, else the real devices; 0 forces real "
                         "devices")
    ap.add_argument("--no-compile", action="store_true",
                    help="stop after lowering (StableHLO only: no memory/cost "
                         "analysis, collectives are not yet partitioned)")
    ap.add_argument("--dump-hlo", default=None,
                    help="write the compiled (or lowered) HLO text here")
    ap.add_argument("--act-numel", type=float, default=1e7,
                    help="minimum result size counted as an activation "
                         "collective by the verdict")
    ap.add_argument("--top", type=int, default=40,
                    help="rows of the collectives table to print")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# HLO collectives
# ---------------------------------------------------------------------------

COLLECTIVE_KINDS = ("all-gather", "all-reduce", "reduce-scatter", "all-to-all",
                    "collective-permute")
# `[ROOT] %name = <result type(s)> <op>[-start|-done](`. Result types carry a
# layout (`f32[64,4608,6144]{2,1,0}`) and may be (nested) tuples with
# `/*index=N*/` comments, so the result is matched non-greedily up to the
# first ` <op>(`; it never contains `%` (operands) or `"` (metadata), which
# keeps a non-collective line from stretching into its operand list.
_INSTR_RE = re.compile(
    r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*(?P<res>[^%\"]+?)\s+"
    r"(?P<op>" + "|".join(COLLECTIVE_KINDS) + r")(?P<phase>-start|-done)?\("
)
_SHAPE_RE = re.compile(r"\b([a-z]+[0-9]*)\[([0-9,]*)\]")


def parse_collectives(hlo_text):
    """[(kind, dtype, shape tuple)] for every collective result in HLO text.

    Async pairs are counted once, at the *-done op: its result holds exactly
    the output(s), whereas a *-start result mixes operands, outputs and
    context scalars (a variadic all-gather-start returns
    `((in0, in1), (out0, out1))`), so *-start lines are skipped. Sync
    collectives keep every result shape (tuples, nested or not, are
    flattened; a variadic all-reduce contributes one entry per operand).
    Shared with tests/test_act_sharding.py.
    """
    out = []
    for line in hlo_text.splitlines():
        m = _INSTR_RE.search(line)
        if not m or m.group("phase") == "-start":
            continue
        kind = m.group("op")
        for dt, dims in _SHAPE_RE.findall(m.group("res")):
            if dt == "token":
                continue
            out.append((kind, dt, tuple(int(d) for d in dims.split(",") if d)))
    return out


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def activation_collectives(colls, batch, min_numel):
    return [c for c in colls
            if c[2] and c[2][0] == batch and _numel(c[2]) >= min_numel]


def print_collectives_table(colls, top=40):
    counts = collections.Counter(colls)
    by_kind = collections.Counter(k for k, _, _ in colls)
    print(f"collectives: {len(colls)} total "
          + (", ".join(f"{k} {n}" for k, n in sorted(by_kind.items())) or "-"))
    if not counts:
        return
    print(f"  {'count':>5}  {'kind':<19} {'dtype':<6} {'shape':<28} {'MiB each':>9}")
    rows = sorted(counts.items(),
                  key=lambda kv: (-_numel(kv[0][2]) * kv[1], kv[0]))
    for (kind, dt, shape), n in rows[:top]:
        itemsize = {"f64": 8, "f32": 4, "s32": 4, "u32": 4, "bf16": 2,
                    "f16": 2, "s8": 1, "u8": 1, "pred": 1}.get(dt, 4)
        mib = _numel(shape) * itemsize / 2**20
        print(f"  {n:5d}  {kind:<19} {dt:<6} {str(list(shape)):<28} {mib:9.2f}")
    if len(rows) > top:
        print(f"  ... {len(rows) - top} more (kind, dtype, shape) groups")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    emulate = args.emulate_devices
    if emulate is None:
        emulate = 0 if _has_accelerator() else 8
    if emulate:
        setup_emulation(emulate)
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    repo = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(repo), str(repo / "scripts")]

    import dataclasses
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np

    import train_full as tf
    from krea2_jax.checkpoint import (
        abstract_krea2_params, batch_sharding, make_mesh, param_shardings,
    )
    from krea2_jax.config import KREA2_CONFIG

    cfg = KREA2_CONFIG
    if args.model_config:
        cfg = dataclasses.replace(cfg, **json.loads(args.model_config))
    platform = jax.devices()[0].platform
    mesh = make_mesh()
    n_dev = mesh.size
    print(f"devices: {n_dev} x {platform}"
          + (f" (emulated: --xla_force_host_platform_device_count={emulate})"
             if emulate else ""))
    if platform == "cpu":
        print("=" * 78)
        print("CPU backend caveats (the SPMD partitioning itself is "
              "representative):\n"
              "  1. reduce-scatter is decomposed into all-reduce + slice, so "
              "gradient\n     reduce-scatters show up as all-reduces;\n"
              "  2. bf16 dots are upcast to f32, so collective dtypes show "
              "f32 where a\n     GPU/TPU moves bf16 (halve those bytes);\n"
              "  3. temp/scratch sizes come from the CPU compiler and are not "
              "representative\n     of GPU/TPU HBM use (arguments/outputs "
              "are).")
        print("=" * 78)
    if args.batch_size % n_dev:
        raise SystemExit(f"--batch-size {args.batch_size} is not divisible by "
                         f"{n_dev} devices")

    width = args.width or args.resolution
    height = args.height or args.resolution
    comp = cfg.vae_compression
    if width % (comp * cfg.patch) or height % (comp * cfg.patch):
        raise SystemExit(f"width/height must be multiples of {comp * cfg.patch}")
    lat_h, lat_w = height // comp, width // comp
    h_tok, w_tok = lat_h // cfg.patch, lat_w // cfg.patch
    img_seq = h_tok * w_tok
    total = img_seq + args.txt_len
    padded = -(-total // cfg.seq_pad_multiple) * cfg.seq_pad_multiple
    mu = args.timestep_mu if args.timestep_mu is not None else cfg.mu_for_seq_len(img_seq)
    B = args.batch_size

    # --- abstract, sharded train state ---
    abs_params = abstract_krea2_params(cfg, dtype=jnp.float32)
    pshard = param_shardings(abs_params, mesh)
    params = jax.tree.map(
        lambda s, sh: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sh),
        abs_params, pshard,
    )
    leaves = jax.tree.leaves(params)
    n_params = sum(int(np.prod(x.shape)) for x in leaves)
    n_split = sum(not sh.is_fully_replicated for sh in jax.tree.leaves(pshard))
    print(f"params: {n_params / 1e9:.3f}B in {len(leaves)} leaves "
          f"({n_split} FSDP-sharded)")
    print(f"sequence: {width}x{height}px -> latent {lat_h}x{lat_w}, "
          f"{img_seq} img tokens + txt_len {args.txt_len} = {total} "
          f"-> padded {padded}; mu {mu:.4f}")
    print(f"step: batch {B} ({B // n_dev}/device) | remat {args.remat_policy} | "
          f"attn {args.attn} | act-sharding {args.act_sharding} | "
          f"qchunk {args.qchunk} | ema {args.ema_decay or 'off'}")

    hp = tf.parse_args([
        "--base-ckpt", "-", "--cache-dir", "-", "--output-dir", "-",
        "--remat-policy", args.remat_policy, "--attn", args.attn,
        "--attn-block", str(args.attn_block), "--act-sharding", args.act_sharding,
        "--ema-decay", str(args.ema_decay), "--qchunk", str(args.qchunk),
    ])
    use_ema = hp.ema_decay > 0
    tx = tf.make_optimizer(hp, tf.make_schedule(hp))
    abstract_opt = jax.eval_shape(tx.init, params)
    opt_shard = tf.opt_state_shardings(abstract_opt, params, mesh, pshard)
    opt_state = jax.tree.map(
        lambda s, sh: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sh),
        abstract_opt, opt_shard,
    )
    ema = params if use_ema else None

    ds = batch_sharding(mesh)
    b_lat = jax.ShapeDtypeStruct((B, cfg.channels, lat_h, lat_w), jnp.bfloat16,
                                 sharding=ds)
    b_txt = jax.ShapeDtypeStruct((B, args.txt_len, cfg.txtlayers, cfg.txtdim),
                                 jnp.bfloat16, sharding=ds)
    b_mask = jax.ShapeDtypeStruct((B, args.txt_len), jnp.bool_, sharding=ds)
    key = jax.eval_shape(lambda: jax.random.PRNGKey(0))

    train_step = tf.make_train_step(
        cfg, hp, mesh, tx, mu, h_tok, w_tok, use_ema, pshard, opt_shard
    )
    t0 = time.time()
    lowered = train_step.lower(params, opt_state, ema, b_lat, b_txt, b_mask, key)
    print(f"lowered in {time.time() - t0:.1f}s")

    GiB = 2**30
    if args.no_compile:
        text = lowered.as_text()
        print("--no-compile: StableHLO before SPMD partitioning; collectives "
              "below are only those written explicitly (shard_map etc.)")
    else:
        t0 = time.time()
        compiled = lowered.compile()
        print(f"compiled ({platform} backend) in {time.time() - t0:.1f}s")
        ma = compiled.memory_analysis()
        if ma is not None:
            print("memory_analysis (per device):")
            for k in ("argument_size_in_bytes", "output_size_in_bytes",
                      "alias_size_in_bytes", "temp_size_in_bytes",
                      "generated_code_size_in_bytes"):
                v = getattr(ma, k, None)
                if v is not None:
                    print(f"  {k:30s} {v / GiB:10.2f} GiB")
            try:
                peak = (ma.argument_size_in_bytes + ma.output_size_in_bytes
                        - ma.alias_size_in_bytes + ma.temp_size_in_bytes)
                print(f"  {'args + out - alias + temp':30s} {peak / GiB:10.2f} GiB"
                      + ("  (temp not representative on cpu)"
                         if platform == "cpu" else ""))
            except (AttributeError, TypeError):
                pass
        ca = compiled.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0] if ca else None
        if ca:
            fl = float(ca.get("flops", float("nan")))
            by = float(ca.get("bytes accessed", float("nan")))
            print(f"cost_analysis (per device): flops {fl:.3e}, bytes accessed "
                  f"{by / GiB:.1f} GiB")
            print("  note: XLA counts a while-loop (lax.scan) body ONCE, so the "
                  f"{cfg.layers}-block scan is under-counted ~{cfg.layers}x; "
                  "use bench_step.py's analytic TFLOP/s instead")
        analytic = 6 * n_params * padded * B / n_dev
        print(f"analytic model flops (6 * params * padded_tokens * batch / "
              f"devices): {analytic:.3e} per device per step (before remat)")
        text = compiled.as_text()
    if args.dump_hlo:
        Path(args.dump_hlo).write_text(text)
        print(f"wrote HLO to {args.dump_hlo}")

    colls = parse_collectives(text)
    print_collectives_table(colls, top=args.top)
    acts = activation_collectives(colls, B, args.act_numel)
    if acts:
        groups = collections.Counter(acts)
        worst = sorted(groups, key=lambda c: -_numel(c[2]))[:3]
        print(f"activation collectives: FOUND {len(acts)} (result shape[0] == "
              f"batch {B}, numel >= {args.act_numel:.0e}; e.g. "
              + ", ".join(f"{k} {dt}{list(s)}" for k, dt, s in worst) + ")")
    else:
        print(f"activation collectives: NONE (no result with shape[0] == "
              f"batch {B} and numel >= {args.act_numel:.0e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
