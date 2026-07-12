"""Flow-matching Euler sampler for the JAX Krea 2 DiT (mirrors sampling.py
from the official repo)."""

import math
from functools import partial

import jax
import jax.numpy as jnp

from krea2_jax.config import Krea2Config
from krea2_jax.model import (
    build_positions_and_mask,
    dit_forward,
    patchify,
    unpatchify,
)


def timestep_schedule(cfg: Krea2Config, img_seq_len, steps, mu=None, sigma=1.0):
    """Resolution-shifted t grid from 1 -> 0 (steps+1 values)."""
    if mu is None:
        mu = cfg.mu_for_seq_len(img_seq_len)
    ts = [1 - i / steps for i in range(steps + 1)]
    out = []
    for t in ts:
        if t <= 0.0:
            out.append(0.0)
        else:
            out.append(math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma))
    return out


def shift_time(t, mu, sigma=1.0):
    """Same shift as `timestep_schedule`, element-wise on an array in (0, 1)."""
    emu = jnp.exp(mu)
    return emu / (emu + (1.0 / t - 1.0) ** sigma)


@partial(
    jax.jit,
    static_argnames=("cfg", "guidance", "lora_scale", "qchunk"),
)
def _euler_step(
    params, lora, cfg, x, txt, t_curr, dt, pos, mask, untxt, unpos, unmask,
    guidance, lora_scale, qchunk,
):
    t = jnp.full((x.shape[0],), t_curr, dtype=x.dtype)
    v = dit_forward(
        params, cfg, x, txt, t, pos, mask,
        lora=lora, lora_scale=lora_scale, qchunk=qchunk, remat=False,
    )
    if guidance:
        uv = dit_forward(
            params, cfg, x, untxt, t, unpos, unmask,
            lora=lora, lora_scale=lora_scale, qchunk=qchunk, remat=False,
        )
        v = v + guidance * (v - uv)
    return x + dt * v


def sample_latents(
    params,
    cfg: Krea2Config,
    txt,          # (B, Ltxt, txtlayers, txtdim) conditional text states
    txt_mask,     # (B, Ltxt) bool
    untxt=None,   # unconditional text states (required when guidance > 0)
    untxt_mask=None,
    lora=None,
    lora_scale=1.0,
    height=1024,
    width=1024,
    steps=28,
    guidance=4.5,
    mu=None,
    seed=0,
    qchunk=512,
):
    """Euler-integrate the flow ODE. Returns latents (B, C, H/8, W/8)."""
    B = txt.shape[0]
    lat_h, lat_w = height // cfg.vae_compression, width // cfg.vae_compression
    h_tok, w_tok = lat_h // cfg.patch, lat_w // cfg.patch

    rng = jax.random.PRNGKey(seed)
    noise = jax.random.normal(
        rng, (B, cfg.channels, lat_h, lat_w), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    x = patchify(noise, cfg.patch)

    pos, mask = build_positions_and_mask(txt_mask, h_tok, w_tok)
    use_cfg = guidance > 0
    if use_cfg:
        assert untxt is not None and untxt_mask is not None
        unpos, unmask = build_positions_and_mask(untxt_mask, h_tok, w_tok)
    else:
        untxt, unpos, unmask = txt, pos, mask  # unused placeholders

    ts = timestep_schedule(cfg, x.shape[1], steps, mu=mu)
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        x = _euler_step(
            params, lora, cfg, x, txt, tcurr, tprev - tcurr, pos, mask,
            untxt, unpos, unmask,
            guidance=float(guidance) if use_cfg else 0.0,
            lora_scale=float(lora_scale), qchunk=qchunk,
        )
    return unpatchify(x, cfg.patch, cfg.channels, lat_h, lat_w)
