"""Full fine-tuning of Krea 2 (RAW) with JAX on a single multi-device host.

Same flow-matching objective as scripts/train_lora.py:
    x_t = (1 - t) * x0 + t * noise,   target v = noise - x0,   t in [0, 1]
with logit-normal timestep sampling shifted by the resolution-dependent mu
used at inference. Unlike the LoRA trainer every DiT weight trains: fp32
master weights, AdamW moments and an optional EMA copy are all FSDP-sharded
across the devices exactly like the base params (nothing but scalars is
replicated); the forward runs in bfloat16 from a per-step cast of the master
weights.

Data (scripts/cache_latents.py, format krea2-latents-v1; see
krea2_jax/data.py::BucketDataset): aspect-bucketed VAE latents plus the raw
long/medium/short captions (and artist names) of metadata.parquet. Captions
are drawn per epoch (--caption-probs, --artist-prob), tokenized by the
loader, and encoded ONLINE inside the jitted step by the JAX Qwen3-VL text
tower (krea2_jax/qwen3_text.py; bf16, replicated on every device, no
gradient). Every sample occupies a static --img-tokens image rows (its
h_tok x w_tok grid first, zero padding after); padded rows are masked out of
attention and of the loss, which is the per-sample mean squared error over
the valid image tokens, averaged over the batch. The timestep shift mu is
derived per sample from its image-token count (or --timestep-mu for all).
--resume-new-dataset continues a checkpoint's train state on a new dataset
version, whose data order starts at its own epoch 0 (data_step_offset).

Outputs under --output-dir:
    ckpt/<step>/                   orbax train state (params, opt_state, ema)
    model_stepNNNNNN.safetensors   bf16 official-layout export (--export-every,
                                   --export-steps)
    model.safetensors              final bf16 export (EMA weights when enabled)
The exports are plain Krea 2 checkpoints (same keys as raw.safetensors):
usable with scripts/sample.py --base-ckpt, the official code and ComfyUI.

Usage:
    python scripts/train_full.py \
        --base-ckpt /path/to/raw.safetensors \
        --data-dir /scratch/krea2-data \
        --text-encoder Qwen/Qwen3-VL-4B-Instruct \
        --output-dir ./full_out --steps 20000 --batch-size 32 --lr 1e-5
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
import numpy as np
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
from krea2_jax.data import BucketBatchLoader, BucketDataset
from krea2_jax.model import build_positions_and_mask_bucketed, dit_forward
from krea2_jax.qwen3_text import (
    from_hf_config,
    load_qwen3_text_params,
    load_tokenizer,
    param_count as text_param_count,
    resolve_text_encoder,
    text_encoder_identity,
    text_states,
)
from krea2_jax.sampling import shift_time
from krea2_jax.text_template import CaptionTemplate


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
    "seed", "batch_size", "text_rows", "img_tokens", "caption_probs",
    "artist_prob", "n_samples", "dataset_fingerprint", "lat_channels",
    "base_ckpt", "model_config", "text_model_config", "position_mode",
    "timestep_mu", "logit_mean", "logit_std", "text_encoder_id",
    "layer_loop",
)
# Value of a hard key in checkpoints written before the key existed (every
# such checkpoint used it): compared instead of reporting it unrecorded.
# layer_loop: "scan" stacks the blocks, "unroll" keeps them as a per-layer
# list, so params / opt_state / ema trees differ and cannot be restored
# into the other mode.
_RESUME_KEY_DEFAULTS = {"layer_loop": "scan"}
# Changing these is legitimate (e.g. extending a run) but changes the
# optimizer / LR schedule from the resumed step on: warning.
# text_encoder is only the --text-encoder name (a local dir's basename or
# the HF id as given), so the same weights under another name must not be
# fatal; what the encoder IS (tokenizer.json bytes, resolved text config,
# sha256 of the loaded weight files' contents) is the hard key text_encoder_id
# (qwen3_text.text_encoder_identity, path independent).
RESUME_SOFT_KEYS = (
    "steps", "lr", "warmup_steps", "cosine_decay", "min_lr", "weight_decay",
    "beta1", "beta2", "eps", "grad_clip", "ema_decay", "act_sharding", "attn",
    "text_encoder",
)
# Soft keys that do not touch the optimizer or LR schedule (same math up to
# kernel rounding, or a name only).
_COMPUTE_SOFT_KEYS = ("act_sharding", "attn", "text_encoder")
# The hard keys that identify the dataset version: --resume-new-dataset lets
# exactly these differ (every other hard key is still enforced).
DATASET_KEYS = ("dataset_fingerprint", "n_samples")


def check_resume_config(saved, current, step, new_dataset=False):
    """Compare a checkpoint's run config with the current one.

    Raises SystemExit listing every differing RESUME_HARD_KEYS entry (and an
    EMA enabled/disabled flip); prints a warning for differing
    RESUME_SOFT_KEYS entries. A checkpoint without a config (older format)
    only warns, except that the historical defaults of _RESUME_KEY_DEFAULTS
    (keys every such checkpoint used, e.g. layer_loop "scan") are still
    enforced as hard keys. new_dataset (--resume-new-dataset) exempts
    DATASET_KEYS from the comparison (resume_data_position handles them).
    Returns the list of warning lines printed.
    """
    if saved is None:
        hard = [f"  {k}: checkpoint {v!r}, now {current[k]!r}"
                for k, v in _RESUME_KEY_DEFAULTS.items() if current[k] != v]
        if hard:
            raise SystemExit(
                f"--resume: checkpoint step {step} was saved with a different "
                "run configuration (it has no saved run configuration, so it "
                "predates these settings and used their historical defaults). "
                "These settings change the train-state shape and cannot "
                "change mid-run:\n" + "\n".join(hard)
                + "\nRerun with the checkpoint's values, or use a fresh "
                "--output-dir."
            )
        msg = (f"WARNING: checkpoint step {step} has no saved run configuration "
               "(older checkpoint); cannot verify that this run matches it")
        print(msg, flush=True)
        return [msg]
    hard, soft, unrecorded, soft_changed = [], [], [], []
    dataset_changed = False
    for keys, out in ((RESUME_HARD_KEYS, hard), (RESUME_SOFT_KEYS, soft)):
        for k in keys:
            if new_dataset and k in DATASET_KEYS:
                continue
            if k not in saved and k not in _RESUME_KEY_DEFAULTS:
                unrecorded.append(k)
                continue
            old = saved.get(k, _RESUME_KEY_DEFAULTS.get(k))
            if old != current[k]:
                out.append(f"  {k}: checkpoint {old!r}, now {current[k]!r}")
                if out is soft:
                    soft_changed.append(k)
                dataset_changed |= k in DATASET_KEYS
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
            + ("\nTo continue this train state on a new dataset version "
               "(its data order restarts at epoch 0), pass "
               "--resume-new-dataset." if dataset_changed else "")
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


def resume_data_position(saved, current, step, new_dataset):
    """Data position of a resume from checkpoint `step` (after
    check_resume_config passed): returns (data_step_offset, switch).

    The loader starts at `step - data_step_offset`. A normal resume keeps
    the checkpoint's offset (0 for checkpoints that predate the key); with
    new_dataset (--resume-new-dataset) and a checkpoint whose DATASET_KEYS
    differ from `current`, the offset becomes `step`, so the new dataset
    starts at its own epoch 0, and `switch` is the record of that switch
    (else None). new_dataset with the checkpoint's own dataset is a no-op
    (e.g. run2 resuming its own newer checkpoint with the flag still set).
    With new_dataset, prints a NOTE either way.
    """
    if saved is None:
        if new_dataset:
            raise SystemExit(
                f"--resume-new-dataset: checkpoint step {step} has no saved "
                "run configuration, so the dataset it was trained on is "
                "unknown; cannot tell whether this is a new dataset version"
            )
        return 0, None
    offset = int(saved.get("data_step_offset", 0))
    if not new_dataset:
        return offset, None
    if any(k not in saved for k in DATASET_KEYS):
        raise SystemExit(
            f"--resume-new-dataset: checkpoint step {step}'s run "
            "configuration does not record "
            + ", ".join(k for k in DATASET_KEYS if k not in saved)
            + "; cannot tell whether this is a new dataset version"
        )
    if all(saved[k] == current[k] for k in DATASET_KEYS):
        print(f"NOTE: --resume-new-dataset: checkpoint step {step} was "
              "trained on this same dataset (fingerprint "
              f"{current['dataset_fingerprint'][:12]}, "
              f"{current['n_samples']} samples); nothing to switch, the data "
              f"order continues (data_step_offset {offset})", flush=True)
        return offset, None
    switch = {
        "step": int(step),
        "from_fingerprint": saved["dataset_fingerprint"],
        "to_fingerprint": current["dataset_fingerprint"],
        "from_n_samples": saved["n_samples"],
        "to_n_samples": current["n_samples"],
    }
    print(f"NOTE: --resume-new-dataset: continuing the train state of step "
          f"{step} (params, optimizer, EMA, LR schedule position) on a new "
          "dataset version:\n"
          f"  dataset_fingerprint {switch['from_fingerprint'][:12]} -> "
          f"{switch['to_fingerprint'][:12]}\n"
          f"  n_samples {switch['from_n_samples']} -> "
          f"{switch['to_n_samples']}\n"
          f"  its data order starts at epoch 0 (data_step_offset {step})",
          flush=True)
    return int(step), switch


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


def per_sample_mu(cfg, n_img, mu_override=None):
    """(B,) f32 timestep-shift mu: cfg.mu_for_seq_len of each sample's image
    token count n_img (B,) (it is linear in the count, so it vectorizes), or
    mu_override for every sample."""
    if mu_override is not None:
        return jnp.full(n_img.shape, mu_override, dtype=jnp.float32)
    return jnp.asarray(cfg.mu_for_seq_len(n_img.astype(jnp.float32)),
                       dtype=jnp.float32)


def masked_flow_loss(v, target, n_img):
    """Flow-matching loss over bucketed image rows.

    v, target: (B, img_tokens, feat); n_img: (B,) valid rows per sample.
    loss = mean_b [ sum_{i < n_img[b], f} (v - target)[b, i, f]^2
                    / (n_img[b] * feat) ],
    i.e. a per-sample mean over its valid tokens and all features, then a
    batch mean (each sample weighs the same whatever its bucket). Rows
    i >= n_img[b] are selected away (jnp.where) BEFORE squaring, so they
    contribute neither to the loss nor to its gradient, even when they hold
    inf/NaN (selecting after the square would send 0 * inf = NaN cotangents
    back through the padded rows). Equals the plain mean of the squared
    error when no sample is padded."""
    valid = jnp.arange(v.shape[1])[None, :] < n_img[:, None]
    err = jnp.where(valid[..., None], v.astype(jnp.float32) - target, 0.0)
    se = jnp.sum(jnp.square(err), axis=-1)
    denom = jnp.maximum(n_img, 1).astype(jnp.float32) * v.shape[-1]
    return jnp.mean(jnp.sum(se, axis=1) / denom)


def make_train_step(cfg, hp, mesh, tx, use_ema, pshard, opt_shard, txt_cfg,
                    img_tokens, mu_override=None):
    """The jitted full fine-tuning step (bucketed data, online text encoder).

    Returns train_step(params, opt_state, ema, txt_params, img, hw, ids,
    txt_len, key) -> (params, opt_state, ema, {"loss", "grad_norm"}) with
    the batch of BucketBatchLoader: img (B, img_tokens, C*p*p) patchified
    latent tokens (rows >= h*w are padding), hw (B, 2) (h_tok, w_tok), ids
    (B, prefix_len + text_rows) and txt_len (B,) from CaptionTemplate.
    txt_params is the text tower (qwen3_text, normally bf16 + replicated);
    it is NOT donated and gets no gradient: text_states runs before the
    loss, outside value_and_grad. params/opt_state/ema are donated and come
    back with shardings pshard/opt_shard/pshard (ema is None when use_ema is
    False). Timestep shift: per-sample mu from h*w (per_sample_mu) unless
    mu_override. Loss: masked_flow_loss. hp is the argparse namespace;
    fields used: logit_mean, logit_std, qchunk, attn, attn_block,
    remat_policy, ema_decay, act_sharding, layer_loop (params, pshard and
    the opt/ema trees must then hold `blocks` as a per-layer list for
    "unroll", stacked for "scan"). Shared by train_full.main,
    scripts/aot_analysis.py (abstract shapes) and scripts/bench_step.py.
    """
    remat = REMAT_MODES[hp.remat_policy]
    decay = hp.ema_decay
    act = act_sharding_for(hp, mesh)

    def step_fn(params, opt_state, ema, txt_params, img, hw, ids, txt_len,
                key):
        B = img.shape[0]
        if img.shape[1] != img_tokens:
            raise ValueError(f"img has {img.shape[1]} rows, expected "
                             f"img_tokens {img_tokens}")
        k_t, k_noise = jax.random.split(key)

        # online text encoder: frozen, no gradient (outside loss_fn)
        context, txt_mask = text_states(txt_params, txt_cfg, ids, txt_len)
        context = context.astype(jnp.bfloat16)
        if act is not None:
            context = jax.lax.with_sharding_constraint(context, act)

        hw = hw.astype(jnp.int32)
        n_img = hw[:, 0] * hw[:, 1]
        mu = per_sample_mu(cfg, n_img, mu_override)
        x0 = img.astype(jnp.float32)  # already patchified on the host
        u = hp.logit_mean + hp.logit_std * jax.random.normal(k_t, (B,))
        t = shift_time(jax.nn.sigmoid(u), mu)
        noise = jax.random.normal(k_noise, x0.shape, dtype=jnp.float32)
        tt = t[:, None, None]
        xt = ((1.0 - tt) * x0 + tt * noise).astype(jnp.bfloat16)
        target = noise - x0
        pos, mask = build_positions_and_mask_bucketed(txt_mask, hw, img_tokens)

        def loss_fn(params):
            p16 = jax.tree.map(lambda a: a.astype(jnp.bfloat16), params)
            v = dit_forward(
                p16, cfg, xt, context, t, pos, mask,
                lora=None, qchunk=hp.qchunk, remat=remat,
                attn_impl=hp.attn, attn_mesh=mesh, attn_block=hp.attn_block,
                act_sharding=act, layer_loop=hp.layer_loop,
            )
            return masked_flow_loss(v, target, n_img)

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
    ap.add_argument("--data-dir", required=True,
                    help="krea2-latents-v1 dataset dir written by "
                         "scripts/cache_latents.py (metadata.parquet + "
                         "latents/)")
    ap.add_argument("--text-encoder", required=True,
                    help="Qwen3-VL text encoder: a local HF model dir or an "
                         "HF repo id (e.g. Qwen/Qwen3-VL-4B-Instruct; only "
                         "config, tokenizer and safetensors are fetched). "
                         "Runs online in the train step, bf16, replicated")
    ap.add_argument("--text-rows", type=int, default=256,
                    help="caption + suffix rows handed to the DiT (captions "
                         "are truncated to text_rows - 5 tokens)")
    ap.add_argument("--img-tokens", type=int, default=4096,
                    help="static image-token rows per sample; every bucket's "
                         "h_tok * w_tok must fit (rest is masked padding)")
    ap.add_argument("--caption-probs", default="0.5,0.3,0.2",
                    help="probabilities of the long,medium,short caption, "
                         "drawn per sample per epoch")
    ap.add_argument("--artist-prob", type=float, default=0.5,
                    help="probability of appending ', drawn by <artist>' "
                         "(never for an empty artist_name)")
    ap.add_argument("--text-model-config", default=None,
                    help="JSON object of Qwen3TextConfig overrides applied on "
                         "top of the encoder's config.json (tests / tiny "
                         "models, e.g. '{\"select_layers\": [0, 1, 2]}')")
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
    ap.add_argument("--timestep-mu", type=float, default=None,
                    help="use this timestep shift for every sample instead of "
                         "the per-sample one derived from its image-token "
                         "count")
    ap.add_argument("--logit-mean", type=float, default=0.0)
    ap.add_argument("--logit-std", type=float, default=1.0)
    ap.add_argument("--qchunk", type=int, default=256,
                    help="ref attention query chunk; must divide the padded "
                         "sequence (4096 + 256 = 4352 = 17 x 256), else ref "
                         "falls back to dense attention")
    ap.add_argument("--attn", choices=["ref", "splash", "aiter", "cudnn"],
                    default="ref",
                    help="exact fp32-softmax reference attention, the fused "
                         "Pallas Splash kernel (TPU only), jax-aiter's "
                         "flash_attn_varlen (ROCm gfx950 only; needs the "
                         "jax-aiter package), or cuDNN flash attention via "
                         "jax.nn.dot_product_attention (NVIDIA GPU; its "
                         "'xla' path on other backends)")
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
    ap.add_argument("--layer-loop", choices=["scan", "unroll"], default="scan",
                    help="'scan': lax.scan over the 28 stacked blocks; "
                         "'unroll': keep the blocks as per-layer arrays and "
                         "run them in an unrolled loop, which removes the "
                         "per-layer dynamic-slice / dynamic-update-slice "
                         "copies of the scan on GPU (weight slices, saved "
                         "activations, stacked grads) at the cost of a "
                         "longer compile. Checkpoints are not "
                         "interchangeable between the two (resume refuses)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock budget for the train loop; on expiry the "
                         "current step is checkpointed + exported and the "
                         "run exits cleanly (continue with --resume)")
    ap.add_argument("--save-every", type=int, default=1000,
                    help="orbax train-state checkpoint interval (steps)")
    ap.add_argument("--save-steps", default="",
                    help="comma-separated extra steps at which the train "
                         "state is also saved (e.g. an early '20' to prove "
                         "the save path); like --save-every, only steps "
                         "below --steps (the final save always happens)")
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
    ap.add_argument("--resume-new-dataset", action="store_true",
                    help="allow resuming a checkpoint that was trained on a "
                         "different dataset version (only the dataset "
                         "fingerprint and sample count may differ); the full "
                         "train state and the LR schedule position carry "
                         "over and the new dataset starts at its own epoch 0 "
                         "at the resumed step. Needs --resume / --resume-step "
                         "and an existing checkpoint; a no-op when the "
                         "checkpoint already used this dataset")
    ap.add_argument("--export-every", type=int, default=0,
                    help="also write model_stepNNNNNN.safetensors every N "
                         "steps (0: export only at the end)")
    ap.add_argument("--export-steps", default="",
                    help="comma-separated extra steps at which "
                         "model_stepNNNNNN.safetensors is also written, on "
                         "top of --export-every; like --export-every, only "
                         "steps below --steps (the final export always "
                         "happens)")
    ap.add_argument("--verify-latents", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="re-hash every latent shard at startup and refuse "
                         "the run if any differs from its manifest digest "
                         "(rebuilt or transfer-corrupted cache); reads the "
                         "whole cache once. --no-verify-latents trusts the "
                         "manifest")
    ap.add_argument("--prefetch", type=int, default=4,
                    help="batches buffered by the background loader")
    ap.add_argument("--loader-workers", type=int, default=4,
                    help="threads reading + patchifying latents")
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


def parse_caption_probs(text):
    """'0.5,0.3,0.2' -> (0.5, 0.3, 0.2) (long, medium, short); BucketDataset
    validates the values."""
    try:
        probs = tuple(float(x) for x in str(text).split(","))
    except ValueError:
        probs = ()
    if len(probs) != 3:
        raise SystemExit(f"--caption-probs must be three comma-separated "
                         f"numbers (long,medium,short), got {text!r}")
    return probs


def parse_save_steps(text, flag="--save-steps"):
    """'20,500' -> frozenset({20, 500}); '' -> empty. Positive ints only
    (`flag` names the option in the error: --save-steps / --export-steps)."""
    text = str(text).strip()
    try:
        steps = frozenset(int(x) for x in text.split(",")) if text else frozenset()
    except ValueError:
        steps = None
    if steps is None or any(s < 1 for s in steps):
        raise SystemExit(f"{flag} must be comma-separated positive "
                         f"integers, got {text!r}")
    return steps


def resolution_metadata(config, img_tokens=None):
    """The resolution fields of an export's metadata: bucketed runs record
    {"resolution": "bucketed", "img_tokens": N}; a run config from the older
    fixed-resolution path keeps its "resolution"; none known -> "unknown"."""
    config = config or {}
    if config.get("img_tokens") is not None:
        img_tokens = config["img_tokens"]
    elif config.get("resolution") is not None:
        return {"resolution": str(config["resolution"])}
    if img_tokens is None:
        return {"resolution": "unknown"}
    return {"resolution": "bucketed", "img_tokens": str(int(img_tokens))}


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
    save_steps = parse_save_steps(args.save_steps)
    export_steps = parse_save_steps(args.export_steps, "--export-steps")
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
    if args.resume_new_dataset and resume_step is None:
        # the flag means a continuation: never silently start from the base
        raise SystemExit(
            "--resume-new-dataset continues an existing checkpoint, but "
            + (f"there is no checkpoint in {ckpt_dir}" if args.resume else
               "neither --resume nor --resume-step was given")
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
        # needs neither the dataset nor the text encoder: provenance comes
        # from the checkpoint's own run config
        config = meta.get("config") or {}
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
                **resolution_metadata(config),
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

    # --- text encoder config + tokenizer (the weights load further down) ---
    te_dir = resolve_text_encoder(args.text_encoder)
    te_name = (Path(args.text_encoder).resolve().name
               if Path(args.text_encoder).is_dir() else args.text_encoder)
    txt_overrides = (json.loads(args.text_model_config)
                     if args.text_model_config else {})
    txt_cfg = from_hf_config(te_dir, **txt_overrides)
    t_id, te_stats = time.time(), {}
    te_id = text_encoder_identity(te_dir, txt_cfg, stats=te_stats)
    print(f"text encoder identity {te_id[:12]} (hashed "
          f"{te_stats['bytes'] / 2**30:.2f} GiB in {time.time() - t_id:.1f} s)")
    template = CaptionTemplate(load_tokenizer(te_dir), text_rows=args.text_rows)
    problems = []
    if txt_cfg.prefix_len != template.prefix_len:
        problems.append(f"prefix_len {txt_cfg.prefix_len} != the tokenizer's "
                        f"template prefix {template.prefix_len}")
    if txt_cfg.hidden_size != cfg.txtdim:
        problems.append(f"hidden_size {txt_cfg.hidden_size} != DiT txtdim "
                        f"{cfg.txtdim}")
    if len(txt_cfg.select_layers) != cfg.txtlayers:
        problems.append(f"{len(txt_cfg.select_layers)} select_layers != DiT "
                        f"txtlayers {cfg.txtlayers}")
    if problems:
        raise SystemExit(f"--text-encoder {args.text_encoder} does not fit "
                         "the model: " + "; ".join(problems))

    # --- data ---
    caption_probs = parse_caption_probs(args.caption_probs)
    try:
        dataset = BucketDataset(
            args.data_dir, template, img_tokens=args.img_tokens,
            patch=cfg.patch, caption_probs=caption_probs,
            artist_prob=args.artist_prob, verify_digests=args.verify_latents,
        )
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"--data-dir {args.data_dir}: {e}") from e
    if args.verify_latents:
        print(f"latents verified: {dataset.verified_shards} shards, "
              f"{dataset.verified_bytes / 2**30:.2f} GiB in "
              f"{dataset.verify_seconds:.1f} s")
    else:
        print("latents NOT verified (--no-verify-latents): trusting the "
              "manifest digests")
    if dataset.lat_channels != cfg.channels:
        raise SystemExit(f"dataset latents have {dataset.lat_channels} "
                         f"channels, model expects {cfg.channels}")
    if dataset.n < args.batch_size:
        raise SystemExit(f"{dataset.n} samples < --batch-size {args.batch_size}")
    n_img = dataset.hw[:, 0].astype(np.int64) * dataset.hw[:, 1]
    if args.timestep_mu is not None:
        mu_desc = f"mu {args.timestep_mu:.4f} (--timestep-mu, all samples)"
    else:
        mu_desc = (f"mu {cfg.mu_for_seq_len(int(n_img.min())):.4f}.."
                   f"{cfg.mu_for_seq_len(int(n_img.max())):.4f} (per sample)")
    buckets = dataset.buckets()
    top = sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    print(
        f"dataset: {dataset.n} samples in {len(buckets)} buckets (top: "
        + ", ".join(f"{w}x{h} x{c}" for (w, h), c in top)
        + f"); img tokens {int(n_img.min())}..{int(n_img.max())} padded to "
        f"{args.img_tokens}; text_rows {args.text_rows} (prefix "
        f"{template.prefix_len}); {mu_desc}"
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
        "text_rows": int(args.text_rows),
        "img_tokens": int(args.img_tokens),
        "caption_probs": list(dataset.caption_probs),
        "artist_prob": float(dataset.artist_prob),
        "n_samples": int(dataset.n),
        "dataset_fingerprint": dataset.fingerprint(),
        "lat_channels": int(dataset.lat_channels),
        "base_ckpt": Path(args.base_ckpt).name,
        "model_config": json.loads(args.model_config) if args.model_config else None,
        "text_encoder": te_name,
        "text_encoder_id": te_id,
        "text_model_config": txt_overrides or None,
        "position_mode": txt_cfg.position_mode,
        "timestep_mu": (float(args.timestep_mu)
                        if args.timestep_mu is not None else None),
        "logit_mean": args.logit_mean,
        "logit_std": args.logit_std,
        "act_sharding": args.act_sharding,
        "attn": args.attn,
        "layer_loop": args.layer_loop,
        # derived, not flags (never compared on resume): the loader reads
        # data step `step - data_step_offset`, and the dataset-version
        # switches (--resume-new-dataset) that led to this train state
        "data_step_offset": 0,
        "dataset_switches": [],
    }
    if resume_step is not None:
        # before the (slow) base load and before --resume-step renames
        # anything: fail fast on an incompatible resume
        if saved_meta is None:
            try:
                saved_meta = fullft_ckpt.read_meta_at(str(ckpt_dir), resume_step)
            except Exception as e:
                raise _unreadable_step(resume_step, ckpt_dir, e) from e
        saved_config = saved_meta.get("config")
        check_resume_config(saved_config, run_config, resume_step,
                            new_dataset=args.resume_new_dataset)
        run_config["data_step_offset"], switch = resume_data_position(
            saved_config, run_config, resume_step, args.resume_new_dataset)
        run_config["dataset_switches"] = (
            list((saved_config or {}).get("dataset_switches", []))
            + ([switch] if switch else []))
        do_renames()
    manager = fullft_ckpt.make_manager(str(ckpt_dir), max_to_keep=args.keep)

    # --- model: fp32 master weights, FSDP-sharded ---
    print("loading base checkpoint as fp32 (this can take a few minutes)...")
    t0 = time.time()
    params = load_krea2_params(
        args.base_ckpt, mesh=mesh, dtype=jnp.float32, sharding="fsdp",
        unstack=("blocks",) if args.layer_loop == "unroll" else (),
    )
    pshard = param_shardings(params, mesh)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"base params loaded in {time.time() - t0:.0f}s: {n_params / 1e9:.3f}B "
          f"params; state {n_params * (16 if use_ema else 12) / 2**30:.1f} GiB "
          f"total (fp32 master + adam{' + ema' if use_ema else ''}), "
          f"{n_params * (16 if use_ema else 12) / 2**30 / n_dev:.1f} GiB/device")
    print(f"attn {args.attn} | remat {args.remat_policy} | "
          f"act-sharding {args.act_sharding} | "
          f"layer-loop {args.layer_loop} | "
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

    # --- text encoder weights: bf16, replicated, frozen ---
    print(f"loading text encoder {te_name} (bf16, replicated)...")
    t0 = time.time()
    txt_params = load_qwen3_text_params(te_dir, jnp.bfloat16, mesh=mesh,
                                        cfg=txt_cfg)
    n_txt = text_param_count(txt_params)
    print(f"text encoder loaded in {time.time() - t0:.0f}s: {n_txt / 1e9:.3f}B "
          f"params ({txt_cfg.n_run_layers}/{txt_cfg.num_layers} layers, "
          f"select_layers {list(txt_cfg.select_layers)}, position_mode "
          f"{txt_cfg.position_mode}), {n_txt * 2 / 2**30:.2f} GiB/device")

    data_sharding = batch_sharding(mesh)
    train_step = make_train_step(
        cfg, args, mesh, tx, use_ema, pshard, opt_shard, txt_cfg,
        args.img_tokens, mu_override=args.timestep_mu,
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
                **resolution_metadata(None, args.img_tokens),
                "source": source,
            },
        )
        print(f"exported {out_dir / name} ({source}, bf16) "
              f"in {time.time() - t_exp:.0f}s")

    # --- loop ---
    # the per-step RNG (noise, timesteps) and the LR schedule follow the
    # global step; only the data order (epoch permutation, per-epoch caption
    # and artist draws) follows the data step, which restarts at 0 when
    # --resume-new-dataset switches the dataset version
    rng = jax.random.PRNGKey(args.seed)
    data_step = start_step - run_config["data_step_offset"]
    loader = BucketBatchLoader(
        dataset, args.batch_size, args.seed, start_step=data_step,
        prefetch=args.prefetch, workers=args.loader_workers,
    )
    epoch, k = divmod(data_step, loader.steps_per_epoch)
    print(f"data: step {start_step + 1} reads data step {data_step} (epoch "
          f"{epoch}, batch {k}/{loader.steps_per_epoch}; data_step_offset "
          f"{run_config['data_step_offset']})")
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
            img_np, hw_np, ids_np, tl_np = next(batches)
            b_img = jax.device_put(img_np, data_sharding)
            b_hw = jax.device_put(hw_np, data_sharding)
            b_ids = jax.device_put(ids_np, data_sharding)
            b_tl = jax.device_put(tl_np, data_sharding)
            key = jax.random.fold_in(rng, step)

            params, opt_state, ema, metrics = train_step(
                params, opt_state, ema, txt_params, b_img, b_hw, b_ids, b_tl,
                key,
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
            if step < args.steps and (
                    (args.save_every and step % args.save_every == 0)
                    or step in save_steps):
                save(step)
            if step < args.steps and (
                    (args.export_every and step % args.export_every == 0)
                    or step in export_steps):
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
