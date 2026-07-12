"""LoRA fine-tuning of Krea 2 (RAW) on TPU with JAX.

Flow-matching objective, matching the official formulation:
    x_t = (1 - t) * x0 + t * noise,   target v = noise - x0,   t in [0, 1]
with logit-normal timestep sampling shifted by the same resolution-dependent
mu used at inference. Only LoRA parameters train; the 12.9B base stays frozen
in bfloat16, FSDP-sharded across all TPU devices.

Usage:
    python scripts/train_lora.py \
        --base-ckpt /path/to/raw.safetensors \
        --cache-dir ./cache --output-dir ./lora_out \
        --steps 2000 --batch-size 8 --rank 16
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors import safe_open

from krea2_jax.checkpoint import (
    batch_sharding,
    init_lora,
    load_krea2_params,
    load_lora,
    make_mesh,
    replicate_tree,
    save_lora,
)
from krea2_jax.config import KREA2_CONFIG
from krea2_jax.model import build_positions_and_mask, dit_forward, patchify
from krea2_jax.sampling import shift_time


def load_cache(cache_dir, txt_len):
    """Load cached latents/text states into host arrays.

    Returns (latents (N,16,h,w) bf16, txt (N,L,12,2560) bf16,
             txt_mask (N,L) bool, meta).
    """
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    cpu = jax.devices("cpu")[0]

    latents, txts, lens = [], [], []
    with jax.default_device(cpu):
        for s in meta["samples"]:
            with safe_open(str(cache_dir / s["file"]), framework="flax") as f:
                latents.append(np.asarray(f.get_tensor("latent")))
                txts.append(np.asarray(f.get_tensor("txt")))
            lens.append(txts[-1].shape[0])

    if txt_len == "auto":
        txt_len = max(lens)
    txt_len = int(txt_len)

    N = len(latents)
    _, layers, dim = txts[0].shape
    txt = np.zeros((N, txt_len, layers, dim), dtype=txts[0].dtype)
    mask = np.zeros((N, txt_len), dtype=bool)
    for i, h in enumerate(txts):
        n = min(h.shape[0], txt_len)
        txt[i, :n] = h[:n]
        mask[i, :n] = True
    return np.stack(latents), txt, mask, meta, txt_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True, help="raw.safetensors path")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--cosine-decay", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=None, help="default: rank")
    ap.add_argument(
        "--targets", choices=["attn", "attn+mlp", "all"], default="attn"
    )
    ap.add_argument("--txt-len", default="auto", help='"auto" or an int (<=512)')
    ap.add_argument("--timestep-mu", type=float, default=None,
                    help="override the resolution-derived timestep shift")
    ap.add_argument("--logit-mean", type=float, default=0.0)
    ap.add_argument("--logit-std", type=float, default=1.0)
    ap.add_argument("--qchunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", default=None, help="LoRA safetensors to resume from")
    args = ap.parse_args()

    cfg = KREA2_CONFIG
    alpha = args.alpha if args.alpha is not None else float(args.rank)
    lora_scale = alpha / args.rank
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = make_mesh()
    n_dev = mesh.size
    print(f"devices: {n_dev} ({jax.devices()[0].platform})")
    if args.batch_size % n_dev != 0:
        raise SystemExit(f"--batch-size must be divisible by {n_dev} devices")

    # --- data ---
    latents, txt, txt_mask, meta, txt_len = load_cache(args.cache_dir, args.txt_len)
    N, C, lat_h, lat_w = latents.shape
    h_tok, w_tok = lat_h // cfg.patch, lat_w // cfg.patch
    img_seq = h_tok * w_tok
    mu = args.timestep_mu if args.timestep_mu is not None else cfg.mu_for_seq_len(img_seq)
    print(
        f"dataset: {N} samples @ {meta['resolution']}px "
        f"(latent {lat_h}x{lat_w}, {img_seq} tokens, txt_len {txt_len}, mu {mu:.4f})"
    )

    # --- model ---
    print("loading base checkpoint (this can take a few minutes)...")
    t0 = time.time()
    params = load_krea2_params(args.base_ckpt, mesh=mesh, dtype=jnp.bfloat16)
    print(f"base params loaded in {time.time() - t0:.0f}s")

    rng = jax.random.PRNGKey(args.seed)
    rng, lora_rng = jax.random.split(rng)
    if args.resume:
        lora, lmeta = load_lora(args.resume, mesh=mesh)
        print(f"resumed LoRA from {args.resume} (meta: {lmeta})")
    else:
        lora = replicate_tree(
            init_lora(params, lora_rng, rank=args.rank, targets=args.targets),
            mesh,
        )
    n_lora = sum(x.size for x in jax.tree.leaves(lora))
    print(f"LoRA params: {n_lora / 1e6:.1f}M (rank {args.rank}, {args.targets})")

    # --- optimizer ---
    if args.cosine_decay:
        sched = optax.warmup_cosine_decay_schedule(
            0.0, args.lr, args.warmup_steps, args.steps
        )
    else:
        sched = optax.join_schedules(
            [optax.linear_schedule(0.0, args.lr, args.warmup_steps),
             optax.constant_schedule(args.lr)],
            [args.warmup_steps],
        )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(sched, weight_decay=args.weight_decay),
    )
    opt_state = replicate_tree(tx.init(lora), mesh)

    data_sharding = batch_sharding(mesh)

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(lora, opt_state, params, batch_lat, batch_txt, batch_mask, key):
        B = batch_lat.shape[0]
        k_t, k_noise = jax.random.split(key)

        x0 = patchify(batch_lat, cfg.patch).astype(jnp.float32)
        u = args.logit_mean + args.logit_std * jax.random.normal(k_t, (B,))
        t = shift_time(jax.nn.sigmoid(u), mu)
        noise = jax.random.normal(k_noise, x0.shape, dtype=jnp.float32)
        tt = t[:, None, None]
        xt = ((1.0 - tt) * x0 + tt * noise).astype(jnp.bfloat16)
        target = noise - x0

        pos, mask = build_positions_and_mask(batch_mask, h_tok, w_tok)

        def loss_fn(lora):
            v = dit_forward(
                params, cfg, xt, batch_txt, t, pos, mask,
                lora=lora, lora_scale=lora_scale, qchunk=args.qchunk, remat=True,
            )
            return jnp.mean((v.astype(jnp.float32) - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(lora)
        updates, opt_state = tx.update(grads, opt_state, lora)
        lora = optax.apply_updates(lora, updates)
        return lora, opt_state, loss

    def save(step, final=False):
        name = "lora.safetensors" if final else f"lora_step{step:06d}.safetensors"
        save_lora(
            str(out_dir / name), jax.device_get(lora), args.rank, alpha,
            extra_meta={
                "step": step,
                "targets": args.targets,
                "resolution": meta["resolution"],
                "base": Path(args.base_ckpt).name,
            },
        )
        print(f"saved {out_dir / name}")

    # --- loop ---
    np_rng = np.random.default_rng(args.seed)
    ema_loss, t_log = None, time.time()
    for step in range(1, args.steps + 1):
        idx = np_rng.choice(N, size=args.batch_size, replace=N < args.batch_size)
        batch_lat = jax.device_put(latents[idx], data_sharding)
        batch_txt = jax.device_put(txt[idx], data_sharding)
        batch_mask = jax.device_put(txt_mask[idx], data_sharding)
        key = jax.random.fold_in(rng, step)

        lora, opt_state, loss = train_step(
            lora, opt_state, params, batch_lat, batch_txt, batch_mask, key
        )

        if step % args.log_every == 0 or step == 1:
            loss = float(loss)
            ema_loss = loss if ema_loss is None else 0.9 * ema_loss + 0.1 * loss
            dt = time.time() - t_log
            t_log = time.time()
            print(
                f"step {step:6d}/{args.steps}  loss {loss:.4f}  "
                f"ema {ema_loss:.4f}  lr {float(sched(step)):.2e}  "
                f"{dt / args.log_every:.2f}s/it"
            )
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            save(step)

    save(args.steps, final=True)
    print("done.")


if __name__ == "__main__":
    main()
