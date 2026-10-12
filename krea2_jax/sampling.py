"""Flow-matching Euler sampler for the JAX Krea 2 DiT (mirrors sampling.py
from the official repo)."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from krea2_jax.config import Krea2Config
from krea2_jax.model import (
    build_positions_and_mask,
    dit_forward,
    patchify,
    text_tokens_forward,
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


def compact_text(txt, txt_mask, multiple=64):
    """Drop padded text positions: (B, L, ...) -> (B, L', ...), L' <= L.

    Valid tokens are moved to the front of each row (order kept) and the
    sequence is cut to the longest row, rounded up to `multiple` so only a
    few distinct shapes get compiled. Exact for Krea 2: every text token sits
    at RoPE position 0, text fusion has no positional encoding, and padded
    tokens are masked as keys, so dropping them leaves the image tokens'
    outputs unchanged (up to float summation order).
    """
    mask = np.asarray(txt_mask, dtype=bool)
    B, L = mask.shape
    keep = max(int(mask.sum(axis=1).max()), 1)
    keep = min(-(-keep // multiple) * multiple, L)
    order = np.argsort(~mask, axis=1, kind="stable")[:, :keep]
    rows = np.arange(B)[:, None]
    return (
        jnp.asarray(txt)[rows, order],
        jnp.asarray(mask[rows, order]),
    )


@partial(jax.jit, static_argnames=("cfg", "lora_scale"))
def _text_tokens(params, lora, cfg, txt, txt_mask, lora_scale):
    return text_tokens_forward(
        params, cfg, txt, txt_mask, lora=lora, lora_scale=lora_scale
    )


@partial(
    jax.jit,
    static_argnames=(
        "cfg", "guidance", "lora_scale", "qchunk", "attn_impl", "attn_mesh",
    ),
)
def _euler_step(
    params, lora, cfg, x, tok, t_curr, dt, pos, mask, untok, unpos, unmask,
    guidance, lora_scale, qchunk, attn_impl="ref", attn_mesh=None,
):
    t = jnp.full((x.shape[0],), t_curr, dtype=x.dtype)
    fwd = partial(
        dit_forward, params, cfg, lora=lora, lora_scale=lora_scale,
        qchunk=qchunk, remat=False, attn_impl=attn_impl, attn_mesh=attn_mesh,
        attn_shard="heads",
    )
    v = fwd(x, None, t, pos, mask, txt_tokens=tok)
    if guidance:
        uv = fwd(x, None, t, unpos, unmask, txt_tokens=untok)
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
    attn_impl="ref",   # "ref", "splash" (fused TPU kernel) or "cudnn"
    attn_mesh=None,    # mesh to split splash heads over (multi-device)
    compact=True,      # drop padded text tokens before sampling
):
    """Euler-integrate the flow ODE. Returns latents (B, C, H/8, W/8).

    The text branch (text fusion + txtmlp) is independent of t, so it runs
    once per prompt; each step only runs the image/text transformer blocks.
    """
    align = cfg.vae_compression * cfg.patch
    if height % align or width % align:
        raise ValueError(
            f"height/width must be multiples of {align} "
            f"(VAE f{cfg.vae_compression} x patch {cfg.patch}); "
            f"got {width}x{height}"
        )
    B = txt.shape[0]
    lat_h, lat_w = height // cfg.vae_compression, width // cfg.vae_compression
    h_tok, w_tok = lat_h // cfg.patch, lat_w // cfg.patch

    rng = jax.random.PRNGKey(seed)
    noise = jax.random.normal(
        rng, (B, cfg.channels, lat_h, lat_w), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    x = patchify(noise, cfg.patch)

    lora_scale = float(lora_scale)

    def text_branch(states, states_mask):
        if compact:
            states, states_mask = compact_text(states, states_mask)
        states_mask = jnp.asarray(states_mask)
        tok = _text_tokens(params, lora, cfg, states, states_mask, lora_scale)
        pos, mask = build_positions_and_mask(states_mask, h_tok, w_tok)
        return tok, pos, mask

    tok, pos, mask = text_branch(txt, txt_mask)
    use_cfg = guidance > 0
    if use_cfg:
        assert untxt is not None and untxt_mask is not None
        untok, unpos, unmask = text_branch(untxt, untxt_mask)
    else:
        untok, unpos, unmask = tok, pos, mask  # unused placeholders

    ts = timestep_schedule(cfg, x.shape[1], steps, mu=mu)
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        x = _euler_step(
            params, lora, cfg, x, tok, tcurr, tprev - tcurr, pos, mask,
            untok, unpos, unmask,
            guidance=float(guidance) if use_cfg else 0.0,
            lora_scale=lora_scale, qchunk=qchunk,
            attn_impl=attn_impl, attn_mesh=attn_mesh,
        )
    return unpatchify(x, cfg.patch, cfg.channels, lat_h, lat_w)
