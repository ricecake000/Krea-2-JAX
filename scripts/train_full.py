"""Full fine-tuning of Krea 2 (RAW) with JAX on a single multi-device host.

Same flow-matching objective as scripts/train_lora.py:
    x_t = (1 - t) * x0 + t * noise,   target v = noise - x0,   t in [0, 1]
with logit-normal timestep sampling shifted by the resolution-dependent mu
used at inference. Unlike the LoRA trainer every DiT weight trains: fp32
master weights, AdamW moments and an optional EMA copy are all FSDP-sharded
across the devices exactly like the base params (nothing but scalars is
replicated); the forward runs in bfloat16 from a per-step cast of the master
weights.

Outputs under --output-dir:
    ckpt/<step>/                   orbax train state (params, opt_state, ema)
    model_stepNNNNNN.safetensors   bf16 official-layout export (--export-every)
    model.safetensors              final bf16 export (EMA weights when enabled)
The exports are plain Krea 2 checkpoints (same keys as raw.safetensors):
usable with scripts/sample.py --base-ckpt, the official code and ComfyUI.

Usage:
    python scripts/train_full.py \
        --base-ckpt /path/to/raw.safetensors \
        --cache-dir ./cache --output-dir ./full_out \
        --steps 20000 --batch-size 32 --lr 1e-5
"""

import argparse
import dataclasses
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding, PartitionSpec as P

from safetensors import safe_open

from krea2_jax import fullft_ckpt
from krea2_jax.checkpoint import (
    batch_sharding,
    load_krea2_params,
    make_mesh,
    param_shardings,
    save_krea2_params,
)
from krea2_jax.config import KREA2_CONFIG
from krea2_jax.data import BatchLoader, CachedDataset
from krea2_jax.model import build_positions_and_mask, dit_forward, patchify
from krea2_jax.sampling import shift_time


# ---------------------------------------------------------------------------
# optimizer-state sharding
# ---------------------------------------------------------------------------


def _param_index(params, shardings=None):
    """keystr(path) -> (shape, sharding) for every param leaf."""
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    if shardings is None:
        shard_leaves = [leaf.sharding for _, leaf in flat]
    else:
        shard_leaves = jax.tree.leaves(shardings)
    assert len(shard_leaves) == len(flat), "params/shardings tree mismatch"
    return {
        jax.tree_util.keystr(path): (tuple(leaf.shape), sh)
        for (path, leaf), sh in zip(flat, shard_leaves)
    }


def _match_param(index, path, shape):
    """The param an optimizer-state leaf mirrors: longest param path that is
    a suffix of the leaf's path (optax nests mu/nu under the state tuple)
    and has the same shape. Returns (param_path, sharding) or None."""
    ks = jax.tree_util.keystr(path)
    best = None
    for pk, (pshape, sh) in index.items():
        if ks.endswith(pk) and pshape == tuple(shape):
            if best is None or len(pk) > len(best[0]):
                best = (pk, sh)
    return best


def opt_state_shardings(abstract_opt_state, params, mesh, shardings=None):
    """Target sharding for every optimizer-state leaf: each non-scalar leaf
    (AdamW mu/nu) takes the sharding of the param it mirrors; scalars (step
    counters) are replicated. Raises if a non-scalar leaf mirrors no param."""
    index = _param_index(params, shardings)
    replicated = NamedSharding(mesh, P())

    def pick(path, leaf):
        if leaf.ndim == 0:
            return replicated
        m = _match_param(index, path, leaf.shape)
        if m is None:
            raise ValueError(
                f"optimizer-state leaf {jax.tree_util.keystr(path)} "
                f"{tuple(leaf.shape)} mirrors no parameter; cannot shard it"
            )
        return m[1]

    return jax.tree_util.tree_map_with_path(pick, abstract_opt_state)


def check_opt_state_sharding(opt_state, params):
    """Fail loudly unless every non-scalar optimizer-state leaf is sharded
    exactly like the parameter it mirrors.

    A replicated AdamW state for a 12.9B model is 103 GB per device — the
    LoRA-era replicate_tree pattern would silently OOM (or worse, fit and
    waste most of the HBM). Returns the number of checked leaves.
    """
    index = _param_index(params)
    problems, checked = [], 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(opt_state)[0]:
        if getattr(leaf, "ndim", 0) == 0:
            continue
        checked += 1
        name = jax.tree_util.keystr(path)
        sh = getattr(leaf, "sharding", None)
        m = _match_param(index, path, leaf.shape)
        if m is None:
            problems.append(f"{name} {tuple(leaf.shape)}: mirrors no parameter")
        elif sh is None:
            problems.append(f"{name}: not a sharded jax.Array")
        elif sh.is_fully_replicated and not m[1].is_fully_replicated:
            problems.append(f"{name}: fully replicated (param {m[0]} is sharded)")
        elif not sh.is_equivalent_to(m[1], leaf.ndim):
            problems.append(f"{name}: sharded as {sh}, param as {m[1]}")
    if problems:
        raise RuntimeError(
            f"optimizer state is not sharded like the params "
            f"({len(problems)}/{checked} leaves): " + "; ".join(problems[:8])
            + (" ..." if len(problems) > 8 else "")
        )
    return checked


# ---------------------------------------------------------------------------
# resume configuration check
# ---------------------------------------------------------------------------

# Changing any of these mid-run silently changes the data order, the RNG
# stream, the training objective or the train-state shape: hard error.
RESUME_HARD_KEYS = (
    "seed", "batch_size", "txt_len", "resolution", "n_samples",
    "dataset_fingerprint", "base_ckpt", "model_config", "timestep_mu",
    "logit_mean", "logit_std",
)
# Changing these is legitimate (e.g. extending a run) but changes the
# optimizer / LR schedule from the resumed step on: warning.
RESUME_SOFT_KEYS = (
    "steps", "lr", "warmup_steps", "cosine_decay", "min_lr", "weight_decay",
    "beta1", "beta2", "eps", "grad_clip", "ema_decay", "act_sharding", "attn",
)
# Soft keys that only change how the step is computed (same math up to
# kernel rounding), not the optimizer or LR schedule.
_COMPUTE_SOFT_KEYS = ("act_sharding", "attn")


def check_resume_config(saved, current, step):
    """Compare a checkpoint's run config with the current one.

    Raises SystemExit listing every differing RESUME_HARD_KEYS entry (and an
    EMA enabled/disabled flip); prints a warning for differing
    RESUME_SOFT_KEYS entries. A checkpoint without a config (older format)
    only warns. Returns the list of warning lines printed.
    """
    if saved is None:
        msg = (f"WARNING: checkpoint step {step} has no saved run configuration "
               "(older checkpoint); cannot verify that this run matches it")
        print(msg, flush=True)
        return [msg]
    hard, soft, unrecorded, soft_changed = [], [], [], []
    for keys, out in ((RESUME_HARD_KEYS, hard), (RESUME_SOFT_KEYS, soft)):
        for k in keys:
            if k not in saved:
                unrecorded.append(k)
            elif saved[k] != current[k]:
                out.append(f"  {k}: checkpoint {saved[k]!r}, now {current[k]!r}")
                if out is soft:
                    soft_changed.append(k)
    old_ema = saved.get("ema_decay")
    if old_ema is not None and (old_ema > 0) != (current["ema_decay"] > 0):
        hard.append(
            f"  ema enabled (ema_decay > 0): checkpoint {old_ema > 0} "
            f"(ema_decay {old_ema!r}), now {current['ema_decay'] > 0} "
            f"(ema_decay {current['ema_decay']!r})"
        )
    if hard:
        raise SystemExit(
            f"--resume: checkpoint step {step} was saved with a different run "
            "configuration. These settings change the data order, RNG, "
            "objective or train-state shape and cannot change mid-run:\n"
            + "\n".join(hard)
            + "\nRerun with the checkpoint's values, or use a fresh --output-dir."
        )
    lines = []
    if soft:
        lines.append(
            f"WARNING: resuming step {step} with changed optimizer/schedule "
            "settings (continuing anyway):"
        )
        lines += soft
        if any(k not in _COMPUTE_SOFT_KEYS for k in soft_changed):
            lines.append(
                "  the LR schedule is rebuilt from the new settings and "
                "evaluated at the resumed step, so its position/shape changes "
                "(e.g. a longer --steps stretches the cosine decay and "
                "changes the next LR)"
            )
    if unrecorded:
        lines.append("WARNING: checkpoint config does not record "
                     f"{', '.join(unrecorded)}; not checked")
    for line in lines:
        print(line, flush=True)
    return lines


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def make_schedule(args):
    """Linear warmup from 0 to --lr, then constant or cosine to --min-lr."""
    if args.cosine_decay:
        return optax.warmup_cosine_decay_schedule(
            0.0, args.lr, args.warmup_steps, max(args.steps, args.warmup_steps + 1),
            end_value=args.min_lr,
        )
    return optax.join_schedules(
        [optax.linear_schedule(0.0, args.lr, max(args.warmup_steps, 1)),
         optax.constant_schedule(args.lr)],
        [args.warmup_steps],
    )


def make_optimizer(args, sched):
    return optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(
            sched, b1=args.beta1, b2=args.beta2, eps=args.eps,
            weight_decay=args.weight_decay,
        ),
    )


REMAT_MODES = {"full": True, "dots": "dots", "none": False}


def act_sharding_for(hp, mesh):
    """The activation sharding dit_forward constrains to: batch sharding
    P("fsdp") with --act-sharding on, None (unconstrained) with off."""
    return batch_sharding(mesh) if hp.act_sharding == "on" else None


def make_train_step(cfg, hp, mesh, tx, mu, h_tok, w_tok, use_ema, pshard,
                    opt_shard):
    """The jitted full fine-tuning step.

    Returns train_step(params, opt_state, ema, lat, txt, mask, key) ->
    (params, opt_state, ema, {"loss", "grad_norm"}); params/opt_state/ema
    are donated and come back with shardings pshard/opt_shard/pshard (ema
    is None when use_ema is False). hp is the argparse namespace; fields
    used: logit_mean, logit_std, qchunk, attn, attn_block, remat_policy,
    ema_decay, act_sharding. Shared by train_full.main, scripts/
    aot_analysis.py (abstract shapes) and scripts/bench_step.py.
    """
    remat = REMAT_MODES[hp.remat_policy]
    decay = hp.ema_decay
    act = act_sharding_for(hp, mesh)

    def step_fn(params, opt_state, ema, batch_lat, batch_txt, batch_mask, key):
        B = batch_lat.shape[0]
        k_t, k_noise = jax.random.split(key)

        x0 = patchify(batch_lat, cfg.patch).astype(jnp.float32)
        u = hp.logit_mean + hp.logit_std * jax.random.normal(k_t, (B,))
        t = shift_time(jax.nn.sigmoid(u), mu)
        noise = jax.random.normal(k_noise, x0.shape, dtype=jnp.float32)
        tt = t[:, None, None]
        xt = ((1.0 - tt) * x0 + tt * noise).astype(jnp.bfloat16)
        target = noise - x0
        pos, mask = build_positions_and_mask(batch_mask, h_tok, w_tok)

        def loss_fn(params):
            p16 = jax.tree.map(lambda a: a.astype(jnp.bfloat16), params)
            v = dit_forward(
                p16, cfg, xt, batch_txt.astype(jnp.bfloat16), t, pos, mask,
                lora=None, qchunk=hp.qchunk, remat=remat,
                attn_impl=hp.attn, attn_mesh=mesh, attn_block=hp.attn_block,
                act_sharding=act,
            )
            return jnp.mean((v.astype(jnp.float32) - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if ema is not None:
            ema = jax.tree.map(lambda e, p: e + (1.0 - decay) * (p - e), ema, params)
        return params, opt_state, ema, {"loss": loss, "grad_norm": grad_norm}

    return jax.jit(
        step_fn,
        donate_argnums=(0, 1, 2),
        out_shardings=(pshard, opt_shard, pshard if use_ema else None, None),
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-ckpt", required=True, help="raw.safetensors path")
    ap.add_argument("--cache-dir", required=True,
                    help="cache written by scripts/cache_dataset.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="global batch; must be divisible by the device count")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--cosine-decay", action="store_true",
                    help="cosine-decay the LR to --min-lr at --steps "
                         "(default: constant after warmup)")
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA of the fp32 weights (exported instead of the "
                         "raw params); 0 disables it and saves 4 B/param")
    ap.add_argument("--txt-len", default="auto", help='"auto" or an int (<=512)')
    ap.add_argument("--timestep-mu", type=float, default=None,
                    help="override the resolution-derived timestep shift")
    ap.add_argument("--logit-mean", type=float, default=0.0)
    ap.add_argument("--logit-std", type=float, default=1.0)
    ap.add_argument("--qchunk", type=int, default=512)
    ap.add_argument("--attn", choices=["ref", "splash", "aiter"], default="ref",
                    help="exact fp32-softmax reference attention, the fused "
                         "Pallas Splash kernel (TPU only), or jax-aiter's "
                         "flash_attn_varlen (ROCm gfx950 only; needs the "
                         "jax-aiter package)")
    ap.add_argument("--attn-block", type=int, default=256,
                    help="splash attention tile size")
    ap.add_argument("--act-sharding", choices=["on", "off"], default="on",
                    help="constrain every linear output and the scan carry "
                         "to batch sharding P('fsdp') (on), or let XLA's "
                         "SPMD partitioner choose (off: it moves full "
                         "activations with all-gathers/all-reduces instead "
                         "of gathering per-layer weights)")
    ap.add_argument("--remat-policy", choices=["full", "dots", "none"],
                    default="dots",
                    help="gradient checkpointing: 'full' recomputes each "
                         "block, 'dots' saves dense-projection outputs, "
                         "'none' saves everything")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock budget for the train loop; on expiry the "
                         "current step is checkpointed + exported and the "
                         "run exits cleanly (continue with --resume)")
    ap.add_argument("--save-every", type=int, default=1000,
                    help="orbax train-state checkpoint interval (steps)")
    ap.add_argument("--keep", type=int, default=3,
                    help="number of orbax checkpoints to keep")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in "
                         "--output-dir/ckpt (starts fresh if there is none)")
    ap.add_argument("--resume-step", type=int, default=None,
                    help="resume from this checkpoint step instead of the "
                         "newest; implies --resume. Newer step directories "
                         "are renamed to <step>.superseded so they no longer "
                         "shadow the resumed run")
    ap.add_argument("--export-every", type=int, default=0,
                    help="also write model_stepNNNNNN.safetensors every N "
                         "steps (0: export only at the end)")
    ap.add_argument("--prefetch", type=int, default=4,
                    help="batches buffered by the background loader")
    ap.add_argument("--loader-workers", type=int, default=4,
                    help="threads reading cache files")
    ap.add_argument("--model-config", default=None,
                    help="JSON object of Krea2Config field overrides (tests / "
                         "debugging with tiny models; default: the real "
                         "12.9B config)")
    return ap.parse_args(argv)


CMD_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="


def _command_buffer_flag(xla_flags):
    """Value of the last --xla_gpu_enable_command_buffer= token in an
    XLA_FLAGS string (XLA lets the last occurrence win), or None if absent."""
    vals = [t[len(CMD_BUFFER_FLAG):] for t in (xla_flags or "").split()
            if t.startswith(CMD_BUFFER_FLAG)]
    return vals[-1] if vals else None


def command_buffers_disabled(xla_flags):
    """True iff XLA_FLAGS disables every HIP/CUDA command buffer type, i.e.
    its last --xla_gpu_enable_command_buffer= token has an empty value."""
    return _command_buffer_flag(xla_flags) == ""


def _export_metadata(path):
    """The safetensors __metadata__ dict of an export ({} when it has none);
    raises if the file is unreadable."""
    with safe_open(str(path), framework="numpy") as f:
        return f.metadata() or {}


def _unreadable_step(step, ckpt_dir, e):
    return SystemExit(
        f"checkpoint step {step} in {ckpt_dir} is incomplete or unreadable "
        f"({e!r}); pass --resume-step <older step> "
        f"(available: {fullft_ckpt.all_steps(str(ckpt_dir))})"
    )


def main(argv=None):
    args = parse_args(argv)
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if (args.attn == "aiter" and jax.default_backend() in ("gpu", "rocm")
            and not command_buffers_disabled(xla_flags)):
        val = _command_buffer_flag(xla_flags)
        raise SystemExit(
            "--attn aiter needs XLA_FLAGS=--xla_gpu_enable_command_buffer= "
            "(HIP command buffers break the aiter FFI kernels: "
            "HIP_ERROR_StreamCaptureInvalidated); set it before launching"
            + (f" (XLA_FLAGS has {CMD_BUFFER_FLAG}{val})" if val is not None
               else "")
        )

    cfg = KREA2_CONFIG
    if args.model_config:
        cfg = dataclasses.replace(cfg, **json.loads(args.model_config))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"
    use_ema = args.ema_decay > 0
    if not 0.0 <= args.ema_decay < 1.0:
        raise SystemExit("--ema-decay must be in [0, 1)")

    resume_step = None
    saved_meta = None  # meta of resume_step, read once it is known
    renames = []  # --resume-step: (src, dst) pairs, performed after validation
    if args.resume_step is not None:
        args.resume = True
        steps = fullft_ckpt.all_steps(str(ckpt_dir))
        if args.resume_step not in steps:
            raise SystemExit(
                f"--resume-step {args.resume_step}: no such checkpoint in "
                f"{ckpt_dir} (available: {steps})"
            )
        resume_step = args.resume_step
        # validate everything before renaming anything: the selected step
        # must be readable and no rename target may exist
        try:
            saved_meta = fullft_ckpt.read_meta_at(str(ckpt_dir), resume_step)
        except Exception as e:
            raise _unreadable_step(resume_step, ckpt_dir, e) from e
        renames = [(ckpt_dir / str(s), ckpt_dir / f"{s}.superseded")
                   for s in steps if s > resume_step]
        # a final export newer than the resumed step (or of unknown step)
        # would otherwise pass for this run's result
        final = out_dir / "model.safetensors"
        if final.exists():
            try:
                exp_step = int(_export_metadata(final)["step"])
            except Exception:
                exp_step = None
            if exp_step is None or exp_step > resume_step:
                renames.append((final, out_dir / "model.safetensors.superseded"))
        clash = [str(dst) for _, dst in renames if os.path.lexists(dst)]
        if clash:
            raise SystemExit(
                f"--resume-step {resume_step}: cannot set the newer "
                "checkpoints aside, these rename targets already exist: "
                + ", ".join(clash) + "; move them out of the way first "
                "(nothing was renamed)"
            )
        # nothing is renamed yet: that happens once the run configuration
        # has been checked against the checkpoint (do_renames below)
    elif ckpt_dir.exists():
        resume_step = fullft_ckpt.latest_step(str(ckpt_dir))
    if resume_step is not None and not args.resume:
        raise SystemExit(
            f"{ckpt_dir} already holds checkpoint step {resume_step}; pass "
            "--resume to continue it, or use a fresh --output-dir"
        )
    if args.resume and resume_step is None:
        print(f"no checkpoint in {ckpt_dir}; starting fresh")

    def do_renames():
        # before make_manager opens the dir: orbax enumerates steps at open
        for src, dst in renames:
            os.rename(src, dst)
            print(f"--resume-step {resume_step}: renamed newer "
                  f"{'checkpoint' if src.parent == ckpt_dir else 'export'} "
                  f"{src} -> {dst}", flush=True)
    if resume_step is not None and resume_step >= args.steps:
        # no run-config check here (it needs the dataset and changes nothing
        # in the exported weights): set newer steps aside and export
        do_renames()
        final = out_dir / "model.safetensors"
        source = "ema" if use_ema else "params"
        if final.exists():
            try:
                emeta = _export_metadata(final)
                exported_step = emeta.get("step")
                exported_src = emeta.get("source")
                why = (f"it is from step {exported_step}" if exported_step
                       and exported_step != str(resume_step) else
                       "it records no step" if not exported_step else
                       f"it is the {exported_src} export, {source} requested"
                       if exported_src != source else None)
            except Exception as e:
                why = f"it is unreadable ({e!r})"
            if why is None:
                print(f"checkpoint step {resume_step} >= --steps {args.steps}; "
                      "nothing to do")
                return
            print(f"checkpoint step {resume_step} >= --steps; training is "
                  f"complete but {final} is stale ({why}): re-exporting "
                  f"step {resume_step}", flush=True)
        else:
            # the run finished training but died before/while exporting
            print(f"checkpoint step {resume_step} >= --steps; training is "
                  "complete but model.safetensors is missing: exporting it "
                  "now", flush=True)
        # export the checkpoint on the host (no mesh, no base load) and stop;
        # provenance comes from the checkpoint's own run config
        meta = saved_meta
        if meta is None:
            try:
                meta = fullft_ckpt.read_meta_at(str(ckpt_dir), resume_step)
            except Exception as e:
                raise _unreadable_step(resume_step, ckpt_dir, e) from e
        if use_ema and not meta.get("has_ema", False):
            raise SystemExit(
                f"checkpoint step {resume_step} has no EMA weights (saved with "
                "EMA disabled) but the restore target expects one; rerun "
                "with --ema-decay 0 to export its params"
            )
        config = meta.get("config") or {}
        res = config.get("resolution")
        if res is None:  # older checkpoint without a run config
            txt_len = args.txt_len if args.txt_len == "auto" else int(args.txt_len)
            res = CachedDataset(args.cache_dir, txt_len).resolution
        base_name = config.get("base_ckpt") or Path(args.base_ckpt).name
        t_exp = time.time()
        try:
            tree, _ = fullft_ckpt.restore_tree_on_host(
                str(ckpt_dir), step=resume_step, item=source
            )
        except Exception as e:
            raise _unreadable_step(resume_step, ckpt_dir, e) from e
        save_krea2_params(
            str(final), tree, dtype=jnp.bfloat16,
            metadata={
                "step": str(resume_step),
                "base": base_name,
                "resolution": str(res),
                "source": source,
            },
        )
        print(f"exported {final} ({source}, bf16) "
              f"in {time.time() - t_exp:.0f}s")
        return

    mesh = make_mesh()
    n_dev = mesh.size
    print(f"devices: {n_dev} ({jax.devices()[0].platform})")
    if args.batch_size % n_dev != 0:
        raise SystemExit(f"--batch-size must be divisible by {n_dev} devices")

    # --- data ---
    txt_len = args.txt_len if args.txt_len == "auto" else int(args.txt_len)
    dataset = CachedDataset(args.cache_dir, txt_len)
    lat_h, lat_w = dataset.lat_hw
    lat_c = dataset[0][0].shape[0]
    if lat_c != cfg.channels:
        raise SystemExit(f"cache latents have {lat_c} channels, model expects "
                         f"{cfg.channels}")
    if dataset.n < args.batch_size:
        raise SystemExit(f"{dataset.n} samples < --batch-size {args.batch_size}")
    h_tok, w_tok = lat_h // cfg.patch, lat_w // cfg.patch
    img_seq = h_tok * w_tok
    mu = args.timestep_mu if args.timestep_mu is not None else cfg.mu_for_seq_len(img_seq)
    res = dataset.resolution
    print(
        f"dataset: {dataset.n} samples @ {res}px (latent {lat_h}x{lat_w}, "
        f"{img_seq} tokens, txt_len {dataset.txt_len}, mu {mu:.4f})"
    )

    # --- run configuration (stored with every checkpoint, checked on resume) ---
    run_config = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "cosine_decay": bool(args.cosine_decay),
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "grad_clip": args.grad_clip,
        "ema_decay": args.ema_decay,
        "txt_len": int(dataset.txt_len),
        "resolution": res,
        "n_samples": int(dataset.n),
        "dataset_fingerprint": dataset.fingerprint(),
        "base_ckpt": Path(args.base_ckpt).name,
        "model_config": json.loads(args.model_config) if args.model_config else None,
        "timestep_mu": float(mu),
        "logit_mean": args.logit_mean,
        "logit_std": args.logit_std,
        "act_sharding": args.act_sharding,
        "attn": args.attn,
    }
    if resume_step is not None:
        # before the (slow) base load and before --resume-step renames
        # anything: fail fast on an incompatible resume
        if saved_meta is None:
            try:
                saved_meta = fullft_ckpt.read_meta_at(str(ckpt_dir), resume_step)
            except Exception as e:
                raise _unreadable_step(resume_step, ckpt_dir, e) from e
        check_resume_config(saved_meta.get("config"), run_config, resume_step)
        do_renames()
    manager = fullft_ckpt.make_manager(str(ckpt_dir), max_to_keep=args.keep)

    # --- model: fp32 master weights, FSDP-sharded ---
    print("loading base checkpoint as fp32 (this can take a few minutes)...")
    t0 = time.time()
    params = load_krea2_params(
        args.base_ckpt, mesh=mesh, dtype=jnp.float32, sharding="fsdp"
    )
    pshard = param_shardings(params, mesh)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"base params loaded in {time.time() - t0:.0f}s: {n_params / 1e9:.3f}B "
          f"params; state {n_params * (16 if use_ema else 12) / 2**30:.1f} GiB "
          f"total (fp32 master + adam{' + ema' if use_ema else ''}), "
          f"{n_params * (16 if use_ema else 12) / 2**30 / n_dev:.1f} GiB/device")
    print(f"attn {args.attn} | remat {args.remat_policy} | "
          f"act-sharding {args.act_sharding} | "
          f"ema {args.ema_decay if use_ema else 'off'}")

    # --- optimizer ---
    sched = make_schedule(args)
    tx = make_optimizer(args, sched)
    abstract_opt = jax.eval_shape(tx.init, params)
    opt_shard = opt_state_shardings(abstract_opt, params, mesh, pshard)

    start_step = 0
    if resume_step is not None:
        def abstract(tree, shardings):
            return jax.tree.map(
                lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
                tree, shardings,
            )
        abstract_state = {
            "params": abstract(params, pshard),
            "opt_state": abstract(abstract_opt, opt_shard),
            "ema": abstract(params, pshard) if use_ema else None,
            "step": resume_step,
        }
        # free the base copy before restoring so the device never holds two
        # full fp32 trees at once
        for a in jax.tree.leaves(params):
            a.delete()
        del params
        state = fullft_ckpt.restore_state(manager, resume_step, abstract_state)
        params, opt_state = state["params"], state["opt_state"]
        ema = state["ema"]  # restore_state raises if EMA is wanted but absent
        start_step = int(state["step"])
        print(f"resumed train state from {ckpt_dir}/{start_step}")
    else:
        opt_state = jax.jit(
            tx.init, in_shardings=(pshard,), out_shardings=opt_shard
        )(params)
        # jnp.copy is a real op (not jit input forwarding), so ema gets its
        # own buffers and donating params and ema together is legal
        ema = (
            jax.jit(lambda p: jax.tree.map(jnp.copy, p), out_shardings=pshard)(params)
            if use_ema else None
        )
    n_checked = check_opt_state_sharding(opt_state, params)
    print(f"optimizer state: {n_checked} non-scalar leaves sharded like params")

    data_sharding = batch_sharding(mesh)
    train_step = make_train_step(
        cfg, args, mesh, tx, mu, h_tok, w_tok, use_ema, pshard, opt_shard
    )

    last_saved = start_step if resume_step is not None else None

    def save(step):
        nonlocal last_saved
        if last_saved == step:
            return
        fullft_ckpt.save_state(
            manager, step,
            {"params": params, "opt_state": opt_state, "ema": ema, "step": step},
            config=run_config,
        )
        last_saved = step
        print(f"saved train state {ckpt_dir}/{step} (async)")

    def export(step, final=False):
        name = "model.safetensors" if final else f"model_step{step:06d}.safetensors"
        source = "ema" if use_ema else "params"
        t_exp = time.time()
        save_krea2_params(
            str(out_dir / name), ema if use_ema else params, dtype=jnp.bfloat16,
            metadata={
                "step": str(step),
                "base": Path(args.base_ckpt).name,
                "resolution": str(res),
                "source": source,
            },
        )
        print(f"exported {out_dir / name} ({source}, bf16) "
              f"in {time.time() - t_exp:.0f}s")

    # --- loop ---
    rng = jax.random.PRNGKey(args.seed)
    loader = BatchLoader(
        dataset, args.batch_size, args.seed, start_step=start_step,
        prefetch=args.prefetch, workers=args.loader_workers,
    )
    batches = iter(loader)
    ema_loss, t_log, n_log = None, time.time(), 0
    t_train, last_step = time.time(), start_step

    # SIGTERM (preemption notice) / SIGINT: finish the current step, then
    # save + export + exit 0 like --max-hours; a second signal aborts hard
    # (the operator sees "<SIG> received; stopping at step N" within a step)
    stop = {"sig": None}

    def on_signal(signum, frame):
        if stop["sig"] is not None:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        # flag only: no I/O in the handler (a print here can raise a
        # reentrant-call or broken-pipe error into the main thread); the
        # loop acknowledges the signal within one step
        stop["sig"] = signal.Signals(signum).name

    def log(msg):
        # best effort: a broken stdout must never skip save(last_step)
        try:
            print(msg, flush=True)
        except (OSError, ValueError):
            pass

    prev_handlers = {
        sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for step in range(start_step + 1, args.steps + 1):
            lat_np, txt_np, mask_np = next(batches)
            batch_lat = jax.device_put(lat_np, data_sharding)
            batch_txt = jax.device_put(txt_np, data_sharding)
            batch_mask = jax.device_put(mask_np, data_sharding)
            key = jax.random.fold_in(rng, step)

            params, opt_state, ema, metrics = train_step(
                params, opt_state, ema, batch_lat, batch_txt, batch_mask, key
            )
            n_log += 1
            last_step = step

            if step % args.log_every == 0 or step == start_step + 1:
                loss = float(metrics["loss"])
                gnorm = float(metrics["grad_norm"])
                ema_loss = loss if ema_loss is None else 0.9 * ema_loss + 0.1 * loss
                dt = time.time() - t_log
                log(
                    f"step {step:6d}/{args.steps}  loss {loss:.6f}  "
                    f"ema_loss {ema_loss:.6f}  gnorm {gnorm:.4f}  "
                    f"lr {float(sched(step - 1)):.2e}  {dt / n_log:.2f}s/it"
                )
                t_log, n_log = time.time(), 0
            # before the periodic save/export: the final save -> wait ->
            # export below must run even if a periodic export would fail
            if stop["sig"] is not None:
                log(f"{stop['sig']} received; stopping at step {last_step} "
                    "(resume with --resume)")
                break
            if args.save_every and step % args.save_every == 0 and step < args.steps:
                save(step)
            if args.export_every and step % args.export_every == 0 and step < args.steps:
                export(step)
            if args.max_hours and time.time() - t_train > args.max_hours * 3600:
                log(f"--max-hours {args.max_hours} reached; "
                    f"stopping at step {last_step}")
                break

        save(last_step)
        # resumable state is on disk before the export can fail (ENOSPC etc.)
        fullft_ckpt.wait(manager)
        export(last_step, final=True)
    finally:
        loader.close()
        try:
            # a pending async save is dropped if the interpreter exits first
            fullft_ckpt.wait(manager)
        except Exception as e:  # never mask the original exception
            print(f"WARNING: waiting for pending checkpoint writes failed: "
                  f"{e!r}", file=sys.stderr, flush=True)
        for sig, h in prev_handlers.items():
            signal.signal(sig, h)
    print("done.")


if __name__ == "__main__":
    main()
