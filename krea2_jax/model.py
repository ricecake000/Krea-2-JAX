"""Pure-functional JAX implementation of the Krea 2 SingleStream DiT.

Parameters live in a nested dict whose structure mirrors the official PyTorch
state_dict (see `checkpoint.py` for the mapping), with the 28 transformer
blocks (and the 2+2 text-fusion blocks) stacked along a leading axis so the
forward pass can `lax.scan` over them with gradient rematerialization.

LoRA is expressed as a second, optional pytree with the same nesting; every
`linear()` call takes an optional LoRA leaf `{"lora_A": (r, in), "lora_B":
(out, r)}` and adds `scale * (x @ A^T) @ B^T` to its output.

Numerical conventions copied from the reference implementation:
  * RMSNorm runs in float32 with a zero-centered scale (weight = scale + 1).
  * Attention logits & softmax run in float32; sigmoid output gating.
  * Fully-masked positions get a large-but-finite negative bias so gradients
    stay finite (the reference relies on sliced-away NaNs instead).
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from krea2_jax.config import Krea2Config

MASK_BIAS = -1e30


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _lget(lora, key):
    return None if lora is None else lora.get(key)


def linear(x, p, lora=None, lora_scale=1.0):
    """x @ W^T (+ b) with an optional LoRA delta, torch Linear layout (out, in)."""
    y = jnp.einsum("...i,oi->...o", x, p["weight"])
    if "bias" in p:
        y = y + p["bias"]
    if lora is not None:
        a, b = lora["lora_A"], lora["lora_B"]
        h = jnp.einsum("...i,ri->...r", x.astype(a.dtype), a)
        y = y + (lora_scale * jnp.einsum("...r,or->...o", h, b)).astype(y.dtype)
    return y


def rms_norm(x, scale, eps=1e-5):
    """float32 RMSNorm with zero-centered scale (matches reference)."""
    t = x.astype(jnp.float32)
    t = t * jax.lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + eps)
    t = t * (scale.astype(jnp.float32) + 1.0)
    return t.astype(x.dtype)


def gelu_tanh(x):
    return jax.nn.gelu(x, approximate=True)


def timestep_embedding(t, dim, period=1e4, tfactor=1e3, dtype=jnp.bfloat16):
    """Sinusoidal embedding of t (B,) -> (B, 1, dim). cos first, then sin."""
    half = dim // 2
    freqs = jnp.exp(
        -math.log(period) * jnp.arange(half, dtype=jnp.float32) / half
    )
    args = (t.astype(jnp.float32) * tfactor)[:, None, None] * freqs
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1).astype(dtype)


def rope_freqs(pos, axdims, theta, ntk=1.0):
    """Rotation matrices for 3-axis RoPE.

    pos: (B, L, 3) float32 -> (B, L, headdim/2, 2, 2) float32 where the last
    two dims hold [[cos, -sin], [sin, cos]] per rotated pair.
    """
    outs = []
    for i, d in enumerate(axdims):
        scale = np.arange(0, d, 2, dtype=np.float64) / d
        omega = jnp.asarray((1.0 / ((theta * ntk) ** scale)).astype(np.float32))
        ang = pos[..., i].astype(jnp.float32)[..., None] * omega  # (B, L, d/2)
        cos, sin = jnp.cos(ang), jnp.sin(ang)
        rot = jnp.stack(
            [jnp.stack([cos, -sin], axis=-1), jnp.stack([sin, cos], axis=-1)],
            axis=-2,
        )
        outs.append(rot)
    return jnp.concatenate(outs, axis=-3)


def rope_apply(x, freqs):
    """x: (B, H, L, D), freqs: (B, L, D/2, 2, 2)."""
    xf = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 1, 2)
    f = freqs[:, None]  # (B, 1, L, D/2, 2, 2)
    out = f[..., 0] * xf[..., 0] + f[..., 1] * xf[..., 1]
    return out.reshape(x.shape).astype(x.dtype)


def scaled_dot_product_attention(q, k, v, keymask=None, qchunk=512):
    """Exact attention with float32 softmax, GQA, and query chunking.

    q: (B, Hq, L, D); k, v: (B, Hk, L, D); keymask: (B, L) bool or None.
    Masked keys get a finite large-negative bias so fully-masked query rows
    produce finite (garbage, later discarded) outputs instead of NaNs.
    Query chunking bounds the (heads, chunk, L) float32 logit buffer.
    """
    B, Hq, L, D = q.shape
    Hk = k.shape[1]
    G = Hq // Hk
    scale = 1.0 / math.sqrt(D)
    qg = q.reshape(B, Hk, G, L, D)

    bias = None
    if keymask is not None:
        bias = jnp.where(keymask, 0.0, MASK_BIAS).astype(jnp.float32)
        bias = bias[:, None, None, None, :]  # (B, 1, 1, 1, L)

    def one_chunk(qc):
        logits = jnp.einsum(
            "bhgcd,bhmd->bhgcm", qc, k, preferred_element_type=jnp.float32
        )
        logits = logits * scale
        if bias is not None:
            logits = logits + bias
        w = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        return jnp.einsum("bhgcm,bhmd->bhgcd", w, v)

    if L <= qchunk or L % qchunk != 0:
        out = one_chunk(qg)
    else:
        n = L // qchunk
        qs = jnp.moveaxis(qg.reshape(B, Hk, G, n, qchunk, D), 3, 0)
        out = jnp.moveaxis(jax.lax.map(one_chunk, qs), 0, 3)
        out = out.reshape(B, Hk, G, L, D)

    out = out.reshape(B, Hq, L, D)
    return out.transpose(0, 2, 1, 3).reshape(B, L, Hq * D)


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------


def attention_forward(
    x, p, heads, kvheads, freqs=None, keymask=None, lora=None, lora_scale=1.0,
    qchunk=512,
):
    """Sigmoid-gated multi-head attention (`Attention` in the reference)."""
    B, L, _ = x.shape
    headdim = p["wq"]["weight"].shape[0] // heads

    q = linear(x, p["wq"], _lget(lora, "wq"), lora_scale)
    k = linear(x, p["wk"], _lget(lora, "wk"), lora_scale)
    v = linear(x, p["wv"], _lget(lora, "wv"), lora_scale)
    gate = linear(x, p["gate"], _lget(lora, "gate"), lora_scale)

    q = q.reshape(B, L, heads, headdim).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, kvheads, headdim).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, kvheads, headdim).transpose(0, 2, 1, 3)

    q = rms_norm(q, p["qknorm"]["qnorm"]["scale"])
    k = rms_norm(k, p["qknorm"]["knorm"]["scale"])
    if freqs is not None:
        q = rope_apply(q, freqs)
        k = rope_apply(k, freqs)

    out = scaled_dot_product_attention(q, k, v, keymask=keymask, qchunk=qchunk)
    out = out * jax.nn.sigmoid(gate)
    return linear(out, p["wo"], _lget(lora, "wo"), lora_scale)


def swiglu_forward(x, p, lora=None, lora_scale=1.0):
    g = linear(x, p["gate"], _lget(lora, "gate"), lora_scale)
    u = linear(x, p["up"], _lget(lora, "up"), lora_scale)
    return linear(jax.nn.silu(g) * u, p["down"], _lget(lora, "down"), lora_scale)


def single_stream_block(x, p, tvec6, freqs, keymask, cfg: Krea2Config,
                        lora=None, lora_scale=1.0, qchunk=512):
    mods = tvec6 + p["mod"]["lin"]  # (B, 1, 6*features)
    prescale, preshift, pregate, postscale, postshift, postgate = jnp.split(
        mods, 6, axis=-1
    )
    h = (1 + prescale) * rms_norm(x, p["prenorm"]["scale"]) + preshift
    x = x + pregate * attention_forward(
        h, p["attn"], cfg.heads, cfg.kvheads, freqs=freqs, keymask=keymask,
        lora=_lget(lora, "attn"), lora_scale=lora_scale, qchunk=qchunk,
    )
    h = (1 + postscale) * rms_norm(x, p["postnorm"]["scale"]) + postshift
    x = x + postgate * swiglu_forward(
        h, p["mlp"], lora=_lget(lora, "mlp"), lora_scale=lora_scale
    )
    return x


def text_fusion_block(x, p, heads, kvheads, keymask=None):
    x = x + attention_forward(
        rms_norm(x, p["prenorm"]["scale"]), p["attn"], heads, kvheads,
        keymask=keymask,
    )
    x = x + swiglu_forward(rms_norm(x, p["postnorm"]["scale"]), p["mlp"])
    return x


def text_fusion_forward(context, p, cfg: Krea2Config, keymask=None):
    """TextFusionTransformer: fuse per-layer encoder states then refine.

    context: (B, L, txtlayers, txtdim) -> (B, L, txtdim)
    """
    B, L, N, D = context.shape
    x = context.reshape(B * L, N, D)
    for i in range(2):  # layerwise blocks attend across the N encoder layers
        blk = jax.tree.map(lambda a: a[i], p["layerwise_blocks"])
        x = text_fusion_block(x, blk, cfg.txtheads, cfg.txtkvheads)
    x = x.reshape(B, L, N, D).transpose(0, 1, 3, 2)  # (B, L, D, N)
    x = jnp.einsum("bldn,on->bldo", x, p["projector"]["weight"])[..., 0]
    for i in range(2):  # refiner blocks attend across the L text tokens
        blk = jax.tree.map(lambda a: a[i], p["refiner_blocks"])
        x = text_fusion_block(x, blk, cfg.txtheads, cfg.txtkvheads, keymask=keymask)
    return x


def last_layer_forward(x, p, tvec):
    """Modulated RMSNorm + projection back to patch pixels."""
    mods = tvec + p["modulation"]["lin"][None]  # (B, 2, features)
    scale, shift = mods[:, 0:1], mods[:, 1:2]
    x = (1 + scale) * rms_norm(x, p["norm"]["scale"]) + shift
    return linear(x, p["linear"])


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------


def dit_forward(
    params,
    cfg: Krea2Config,
    img,        # (B, Limg, patch*patch*channels) patchified latent tokens
    context,    # (B, Ltxt, txtlayers, txtdim) stacked encoder hidden states
    t,          # (B,) flow time in [0, 1] (1 = pure noise)
    pos,        # (B, Ltxt+Limg, 3) RoPE positions (text rows are zeros)
    mask,       # (B, Ltxt+Limg) bool validity mask
    lora=None,
    lora_scale=1.0,
    qchunk=512,
    remat=True,
):
    """Velocity prediction for the flow ODE. Returns (B, Limg, patch²·channels)."""
    B, Limg, _ = img.shape
    Ltxt = context.shape[1]

    x = linear(img, params["first"])

    tv = timestep_embedding(t, cfg.tdim, dtype=x.dtype)
    tv = linear(tv, params["tmlp"]["0"])
    tv = linear(gelu_tanh(tv), params["tmlp"]["2"])          # (B, 1, features)
    tvec6 = linear(gelu_tanh(tv), params["tproj"]["1"])      # (B, 1, 6*features)

    txt = text_fusion_forward(
        context, params["txtfusion"], cfg, keymask=mask[:, :Ltxt]
    )
    txt = rms_norm(txt, params["txtmlp"]["0"]["scale"])
    txt = linear(txt, params["txtmlp"]["1"])
    txt = linear(gelu_tanh(txt), params["txtmlp"]["3"])

    combined = jnp.concatenate([txt, x], axis=1)

    # Pad the combined sequence to a multiple of 256 (static shapes under jit).
    full = combined.shape[1]
    padlen = (-full) % cfg.seq_pad_multiple
    if padlen:
        combined = jnp.pad(combined, ((0, 0), (0, padlen), (0, 0)))
        mask = jnp.pad(mask, ((0, 0), (0, padlen)))
        pos = jnp.pad(pos, ((0, 0), (0, padlen), (0, 0)))

    freqs = rope_freqs(pos, cfg.axdims, cfg.theta)

    blocks = params["blocks"]
    block_fn = partial(
        single_stream_block, cfg=cfg, lora_scale=lora_scale, qchunk=qchunk
    )

    if lora is not None and "blocks" in lora:
        def body(carry, layer):
            p, l = layer
            return block_fn(carry, p, tvec6, freqs, mask, lora=l), None
        xs = (blocks, lora["blocks"])
    else:
        def body(carry, p):
            return block_fn(carry, p, tvec6, freqs, mask), None
        xs = blocks

    if remat:
        body = jax.checkpoint(body, prevent_cse=False)
    combined, _ = jax.lax.scan(body, combined, xs)

    out = last_layer_forward(combined, params["last"], tv)
    return out[:, Ltxt : Ltxt + Limg]


# ---------------------------------------------------------------------------
# latent <-> token helpers
# ---------------------------------------------------------------------------


def patchify(lat, patch):
    """(B, C, H, W) -> (B, (H/p)*(W/p), C*p*p) matching the reference layout."""
    B, C, H, W = lat.shape
    h, w = H // patch, W // patch
    x = lat.reshape(B, C, h, patch, w, patch)
    x = x.transpose(0, 2, 4, 1, 3, 5)  # B h w C p p
    return x.reshape(B, h * w, C * patch * patch)


def unpatchify(tokens, patch, channels, H, W):
    """Inverse of `patchify`. H, W are latent-grid sizes."""
    B = tokens.shape[0]
    h, w = H // patch, W // patch
    x = tokens.reshape(B, h, w, channels, patch, patch)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    return x.reshape(B, channels, H, W)


def image_positions(h_tokens, w_tokens, batch, dtype=jnp.float32):
    """(B, h*w, 3) grid positions: axis0 = 0, axis1 = row, axis2 = col."""
    ids = np.zeros((h_tokens, w_tokens, 3), dtype=np.float32)
    ids[..., 1] = np.arange(h_tokens)[:, None]
    ids[..., 2] = np.arange(w_tokens)[None, :]
    ids = ids.reshape(1, h_tokens * w_tokens, 3)
    return jnp.asarray(np.broadcast_to(ids, (batch, h_tokens * w_tokens, 3)), dtype)


def build_positions_and_mask(txt_mask, h_tokens, w_tokens):
    """Combined (pos, mask) for [text, image] sequences.

    txt_mask: (B, Ltxt) bool -> pos (B, Ltxt+Limg, 3), mask (B, Ltxt+Limg).
    Text tokens sit at position (0, 0, 0); images on a (0, y, x) grid.
    """
    B, Ltxt = txt_mask.shape
    imgpos = image_positions(h_tokens, w_tokens, B)
    txtpos = jnp.zeros((B, Ltxt, 3), dtype=imgpos.dtype)
    pos = jnp.concatenate([txtpos, imgpos], axis=1)
    imgmask = jnp.ones((B, h_tokens * w_tokens), dtype=bool)
    mask = jnp.concatenate([txt_mask, imgmask], axis=1)
    return pos, mask
