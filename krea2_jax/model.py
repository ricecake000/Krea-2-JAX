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
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from krea2_jax.config import Krea2Config

MASK_BIAS = -1e30


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _lget(lora, *keys):
    node = lora
    for k in keys:
        node = None if node is None else node.get(k)
    return node


def _lindex(lora, i):
    """Select layer i from a stacked LoRA subtree (None passes through)."""
    return None if lora is None else jax.tree.map(lambda a: a[i], lora)


def _constrain(x, act_sharding):
    """with_sharding_constraint when a sharding is given (None: identity)."""
    if act_sharding is None:
        return x
    return jax.lax.with_sharding_constraint(x, act_sharding)


def _shard_map(f, mesh, in_specs, out_specs):
    """shard_map with replication checks off, across JAX versions
    (`jax.shard_map(check_vma=)` on new JAX, the experimental
    `shard_map(check_rep=)` on old)."""
    sm = getattr(jax, "shard_map", None)
    if sm is not None:
        try:
            return sm(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                      check_vma=False)
        except TypeError:
            pass
    from jax.experimental.shard_map import shard_map

    return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                     check_rep=False)


def linear(x, p, lora=None, lora_scale=1.0, act_sharding=None):
    """x @ W^T (+ b) with an optional LoRA delta, torch Linear layout (out, in).

    act_sharding: optional `jax.sharding.Sharding` applied to the output with
    `with_sharding_constraint` (e.g. NamedSharding(mesh, P("fsdp")) keeps
    activations batch-sharded so the SPMD partitioner all-gathers weights
    instead of activations). None leaves the output unconstrained.
    """
    y = jnp.einsum("...i,oi->...o", x, p["weight"])
    if "bias" in p:
        y = y + p["bias"]
    if lora is not None:
        a, b = lora["lora_A"], lora["lora_B"]
        h = jnp.einsum("...i,ri->...r", x.astype(a.dtype), a)
        y = y + (lora_scale * jnp.einsum("...r,or->...o", h, b)).astype(y.dtype)
    return _constrain(y, act_sharding)


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
    Query chunking bounds the (heads, chunk, L) float32 logit buffer.

    Mask semantics: the reference builds an outer-product (query AND key)
    mask and relies on the fused CUDA kernels returning zeros for fully
    masked query rows. We mask keys only, with a finite large-negative bias:
    valid queries see exactly the same key set (identical outputs, verified
    to float32 rounding in tests/test_parity.py), while invalid query rows
    produce finite garbage instead of the reference's zeros. Those rows are
    masked as keys downstream and sliced away at the output, so they never
    influence results — and keeping the bias finite (rather than -inf on
    empty rows) keeps softmax gradients NaN-free for LoRA training.
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
        if L > qchunk:
            warnings.warn(
                f"qchunk={qchunk} does not divide sequence length {L}; "
                "falling back to unchunked attention (use a divisor of "
                f"{L}, or qchunk >= {L} to silence this)",
                stacklevel=2,
            )
        out = one_chunk(qg)
    else:
        n = L // qchunk
        qs = jnp.moveaxis(qg.reshape(B, Hk, G, n, qchunk, D), 3, 0)
        out = jnp.moveaxis(jax.lax.map(one_chunk, qs), 0, 3)
        out = out.reshape(B, Hk, G, L, D)

    out = out.reshape(B, Hq, L, D)
    return out.transpose(0, 2, 1, 3).reshape(B, L, Hq * D)


def splash_attention(
    q, k, v, keymask=None, mesh=None, block=256, interpret=False,
    shard="batch",
):
    """Fused TPU (Pallas "Splash") attention, numerically matching
    `scaled_dot_product_attention` to bf16/online-softmax rounding.

    q: (B, Hq, L, D); k, v: (B, Hk, L, D); keymask: (B, L) bool or None.
    The L x L float32 logit buffer never materializes in HBM: tiles are
    computed in VMEM with an online softmax, forward and backward.

    Key-only mask semantics are reproduced with segment ids: every query
    (padded or not) gets segment 0 and only valid keys get segment 0, so
    valid queries see exactly the valid key set and padded query rows still
    produce finite values (never an empty softmax — image tokens are always
    valid keys). Requires L to be a multiple of `block`.

    mesh: on a multi-device mesh the kernel must run under shard_map (GSPMD
    cannot partition a Pallas call). With mesh=None the kernel runs on the
    local device (single chip, tests).
    shard: how the shard_map splits the work across `mesh`:
      "batch" — the batch is sharded over the mesh axis (training).
      "heads" — attention heads are split over the mesh; for sampling, where
        the batch (often 1) is replicated. KV heads are split alongside their
        query groups when the mesh divides them, otherwise KV is expanded to
        one head per query head first; if neither divides, every device
        computes all heads.
    interpret: run the kernel in Pallas interpret mode (CPU-testable).
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        BlockSizes,
        FullMask,
        MultiHeadMask,
        SegmentIds,
        make_splash_mha,
    )
    from jax.sharding import PartitionSpec as P

    B, Hq, L, D = q.shape
    if block <= 0 or block % 128 != 0:
        raise ValueError(
            f"splash attention block ({block}) must be a positive multiple "
            "of 128"
        )
    if L % block != 0:
        raise ValueError(f"splash attention needs L ({L}) % block ({block}) == 0")

    if shard not in ("batch", "heads"):
        raise ValueError(f"unknown splash shard mode: {shard!r}")

    # The kernel applies no softmax scale; fold 1/sqrt(D) into q.
    q = q * jnp.asarray(1.0 / math.sqrt(D), q.dtype)

    head_spec = P()
    local_hq = Hq
    if mesh is not None and shard == "heads":
        n = mesh.size
        Hk = k.shape[1]
        if Hk % n == 0:
            head_spec = P(None, "fsdp")
        elif Hq % n == 0:
            # query head h uses KV head h // G: expanding KV to Hq heads keeps
            # that pairing after an even split of the head axis
            k = jnp.repeat(k, Hq // Hk, axis=1)
            v = jnp.repeat(v, Hq // Hk, axis=1)
            head_spec = P(None, "fsdp")
        if head_spec != P():
            local_hq = Hq // n

    kernel = make_splash_mha(
        MultiHeadMask([FullMask((L, L)) for _ in range(local_hq)]),
        head_shards=1,
        q_seq_shards=1,
        block_sizes=BlockSizes(
            block_q=block, block_kv=block, block_kv_compute=block,
            block_q_dkv=block, block_kv_dkv=block, block_kv_dkv_compute=block,
            block_q_dq=block, block_kv_dq=block,
        ),
        interpret=interpret,
    )

    q_seg = jnp.zeros((B, L), dtype=jnp.int32)
    kv_seg = (
        jnp.where(keymask, 0, 1).astype(jnp.int32)
        if keymask is not None
        else jnp.zeros((B, L), dtype=jnp.int32)
    )

    def batched(q, k, v, q_seg, kv_seg):
        return jax.vmap(
            lambda q1, k1, v1, s1, s2: kernel(q1, k1, v1, SegmentIds(s1, s2))
        )(q, k, v, q_seg, kv_seg)

    if mesh is not None and shard == "heads":
        batched = _shard_map(
            batched,
            mesh=mesh,
            in_specs=(head_spec, head_spec, head_spec, P(), P()),
            out_specs=head_spec,
        )
    elif mesh is not None:
        batched = _shard_map(
            batched,
            mesh=mesh,
            in_specs=(
                P("fsdp"), P("fsdp"), P("fsdp"), P("fsdp"), P("fsdp"),
            ),
            out_specs=P("fsdp"),
        )

    out = batched(q, k, v, q_seg, kv_seg)  # (B, Hq, L, D)
    return out.transpose(0, 2, 1, 3).reshape(B, L, Hq * D)


def varlen_reference_kernel(q3, k3, v3, seqstart_q, seqstart_k, cu_q, cu_k,
                            max_seqlen_q, max_seqlen_k, softmax_scale=None,
                            causal=False, deterministic=False, **_):
    """Pure-JAX emulation of jax-aiter `flash_attn_varlen` in padded group
    mode, restricted to FIXED-STRIDE segments (seqstart = i * L).

    q3: (B*L, Hq, D); k3, v3: (B*L, Hk, D). B = len(seqstart_q) - 1 and
    L = total // B are static (from shapes). Segment i's real tokens are the
    first cu_*[i+1] - cu_*[i] rows of its physical span. Each segment runs
    non-causal attention of its logical query prefix over its logical key
    prefix (fp32 softmax, GQA); padding query rows are zero. Returns
    `(out,)` with out (B*L, Hq, D) in q3's dtype, like the kernel.

    The emulation only covers the layout `aiter_attention` produces, and it
    checks its arguments rather than ignoring them: Python-int max_seqlen_*
    must equal L (ValueError otherwise); seqstart_q / seqstart_k must equal
    arange(B+1)*L and cu_q must equal cu_k — checked on traced values
    (jit-safe), a mismatch turns the whole output into NaN so a wrong call
    fails parity checks loudly instead of silently computing something else.
    """
    del deterministic
    if causal:
        raise NotImplementedError("varlen_reference_kernel: causal=True")
    total, Hq, D = q3.shape
    Hk = k3.shape[1]
    B = seqstart_q.shape[0] - 1
    if B <= 0 or total % B:
        raise ValueError(
            f"varlen_reference_kernel: {total} tokens do not split into "
            f"{B} fixed-stride segments"
        )
    L = total // B
    for name, m in (("max_seqlen_q", max_seqlen_q),
                    ("max_seqlen_k", max_seqlen_k)):
        if isinstance(m, (int, np.integer)) and int(m) != L:
            raise ValueError(
                f"varlen_reference_kernel: {name}={int(m)} but the fixed "
                f"segment stride is L={L} (only max_seqlen == L is emulated)"
            )
    if seqstart_k.shape != seqstart_q.shape or cu_q.shape != seqstart_q.shape \
            or cu_k.shape != seqstart_q.shape:
        raise ValueError(
            "varlen_reference_kernel: seqstart_q/seqstart_k/cu_q/cu_k must all "
            f"have shape ({B + 1},); got {seqstart_q.shape}, "
            f"{seqstart_k.shape}, {cu_q.shape}, {cu_k.shape}"
        )
    expected = jnp.arange(B + 1, dtype=jnp.int32) * L
    mismatch = (
        jnp.any(seqstart_q.astype(jnp.int32) != expected)
        | jnp.any(seqstart_k.astype(jnp.int32) != expected)
        | jnp.any(cu_q.astype(jnp.int32) != cu_k.astype(jnp.int32))
    )
    q = q3.reshape(B, L, Hq, D).transpose(0, 2, 1, 3)
    k = k3.reshape(B, L, Hk, D).transpose(0, 2, 1, 3)
    v = v3.reshape(B, L, Hk, v3.shape[-1]).transpose(0, 2, 1, 3)
    if softmax_scale is not None and not math.isclose(
        float(softmax_scale), 1.0 / math.sqrt(D), rel_tol=1e-12
    ):
        # scaled_dot_product_attention applies 1/sqrt(D); fold the rest into q
        q = q * jnp.asarray(float(softmax_scale) * math.sqrt(D), q.dtype)
    ar = jnp.arange(L, dtype=jnp.int32)[None, :]
    len_q = (cu_q[1:] - cu_q[:-1]).astype(jnp.int32)
    len_k = (cu_k[1:] - cu_k[:-1]).astype(jnp.int32)
    keymask = ar < len_k[:, None]                      # (B, L)
    qvalid = ar < len_q[:, None]                       # (B, L)
    qchunk = 512 if L % 512 == 0 else L
    out = scaled_dot_product_attention(q, k, v, keymask=keymask, qchunk=qchunk)
    out = jnp.where(qvalid[:, :, None], out, jnp.zeros((), out.dtype))
    # unsupported seqstart / cu_q != cu_k: poison everything (see docstring)
    out = jnp.where(mismatch, jnp.asarray(jnp.nan, out.dtype), out)
    return (out.reshape(total, Hq, v3.shape[-1]).astype(q3.dtype),)


def _load_aiter_kernel(sharded):
    """jax-aiter varlen flash attention adapted to the `aiter_attention`
    kernel calling convention. Imported lazily: jax-aiter only exists on
    ROCm gfx950 nodes. Under shard_map (sharded=True) the raw entry point
    is used — the public `flash_attn_varlen` wraps its FFI calls in
    custom_partitioning, which conflicts with shard_map's manual mode."""
    try:
        from jax_aiter.mha import flash_attn_varlen, flash_attn_varlen_raw
    except ImportError as e:  # includes ModuleNotFoundError
        raise ImportError(
            "attn_impl='aiter' needs the jax-aiter package (ROCm/jax-aiter, "
            "gfx950 / MI355X only, with its flash-attention libraries: "
            "`jax-aiter-fetch-mha`); it is not importable here: "
            f"{e}. Use attn_impl='ref' or pass "
            "kernel=varlen_reference_kernel for a CPU emulation."
        ) from e
    if not sharded:
        return flash_attn_varlen

    def raw_kernel(q3, k3, v3, seqstart_q, seqstart_k, cu_q, cu_k,
                   max_seqlen_q, max_seqlen_k, softmax_scale=None,
                   causal=False, deterministic=False, **_):
        if softmax_scale is None:
            softmax_scale = q3.shape[-1] ** -0.5
        return flash_attn_varlen_raw(
            q3, k3, v3, seqstart_q, seqstart_k, cu_q, cu_k,
            int(max_seqlen_q), int(max_seqlen_k), 0.0, float(softmax_scale),
            bool(causal), (-1, -1),
        )

    return raw_kernel


def aiter_attention(q, k, v, keymask=None, mesh=None, kernel=None):
    """Varlen flash attention (jax-aiter `flash_attn_varlen`, padded group
    mode) for PREFIX key masks.

    q: (B, Hq, L, D); k, v: (B, Hk, L, D); keymask: (B, L) bool whose valid
    entries form a prefix of each row (dit_forward reorders tokens so this
    holds); None = all valid. The batch is packed to (B*L, H, D) with
    fixed-stride physical segments (seqstart = i*L) and cumulative logical
    lengths from keymask, so each sample attends over its valid prefix only.
    Valid queries see exactly the valid key set (same as the key-only mask of
    `scaled_dot_product_attention`); padded query rows come out as zeros.
    Returns (B, L, Hq*D).

    mesh: when the batch is sharded, the packed call runs under shard_map over
    the "fsdp" batch axis (per-device local cu_seqlens), like
    `splash_attention`. kernel: callable with the `flash_attn_varlen`
    signature (e.g. `varlen_reference_kernel`); None imports jax-aiter.
    """
    from jax.sharding import PartitionSpec as P

    B, Hq, L, D = q.shape
    Hk = k.shape[1]
    if kernel is None:
        kernel = _load_aiter_kernel(sharded=mesh is not None)
    if keymask is None:
        keymask = jnp.ones((B, L), dtype=bool)
    scale = 1.0 / math.sqrt(D)

    def packed(q, k, v, keymask):
        b = q.shape[0]
        q3 = q.transpose(0, 2, 1, 3).reshape(b * L, Hq, D)
        k3 = k.transpose(0, 2, 1, 3).reshape(b * L, Hk, D)
        v3 = v.transpose(0, 2, 1, 3).reshape(b * L, Hk, v.shape[-1])
        seqstart = jnp.arange(b + 1, dtype=jnp.int32) * L
        cu_logical = jnp.concatenate([
            jnp.zeros((1,), jnp.int32),
            jnp.cumsum(keymask.astype(jnp.int32).sum(-1)).astype(jnp.int32),
        ])
        res = kernel(q3, k3, v3, seqstart, seqstart, cu_logical, cu_logical,
                     L, L, softmax_scale=scale, causal=False,
                     deterministic=False)
        out = res[0] if isinstance(res, (tuple, list)) else res
        return out.reshape(b, L, Hq * v.shape[-1])

    if mesh is not None:
        packed = _shard_map(
            packed,
            mesh=mesh,
            in_specs=(P("fsdp"), P("fsdp"), P("fsdp"), P("fsdp")),
            out_specs=P("fsdp"),
        )
    return packed(q, k, v, keymask)


def cudnn_attention(q, k, v, keymask=None, mesh=None, implementation=None):
    """Fused attention through `jax.nn.dot_product_attention` (cuDNN flash
    attention on NVIDIA GPUs) for PREFIX key masks — the NVIDIA counterpart
    of `aiter_attention`, with the same semantics.

    q: (B, Hq, L, D); k, v: (B, Hk, L, D) (GQA when Hk < Hq); keymask: (B, L)
    bool whose valid entries form a prefix of each row (dit_forward reorders
    tokens so this holds); None = all valid. The per-sample valid count goes
    to the kernel as query_seq_lengths AND key_value_seq_lengths (int32), so
    valid queries attend over exactly the valid key set (same as the key-only
    mask of `scaled_dot_product_attention`) and padded query rows come out as
    zeros (also enforced explicitly here). Returns (B, L, Hq*D).

    implementation: passed to dot_product_attention. None picks "cudnn" when
    `jax.default_backend() == "gpu"` and "xla" otherwise — the XLA einsum
    path with the same masking semantics (unchunked (B, H, L, L) fp32
    logits), which is what CPU tests run. cuDNN itself needs bf16/fp16
    inputs and head_dim <= 128 (<= 256 on Hopper+) and a multiple of 8; it
    raises otherwise (and "cuDNN is not detected" on non-CUDA GPUs, e.g.
    ROCm: use attn_impl="aiter" there).

    mesh: when given and the batch divides the "fsdp" axis, q/k/v (BTNH) and
    the lengths are constrained to batch sharding so the cuDNN call's
    custom_partitioning sees consistently batch-sharded operands (it allows
    batch/head sharding only). No shard_map is used.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P

    B, Hq, L, D = q.shape
    if implementation is None:
        implementation = "cudnn" if jax.default_backend() == "gpu" else "xla"
    qt, kt, vt = (a.transpose(0, 2, 1, 3) for a in (q, k, v))  # BNTH -> BTNH
    valid = None
    if keymask is not None:
        valid = keymask.astype(jnp.int32).sum(-1).astype(jnp.int32)  # (B,)
    if mesh is not None and B % mesh.shape["fsdp"] == 0:
        bsh = NamedSharding(mesh, P("fsdp"))
        qt, kt, vt = (jax.lax.with_sharding_constraint(a, bsh)
                      for a in (qt, kt, vt))
        if valid is not None:
            valid = jax.lax.with_sharding_constraint(valid, bsh)
    lengths = {}
    if valid is not None:
        lengths = dict(query_seq_lengths=valid, key_value_seq_lengths=valid)
    out = jax.nn.dot_product_attention(
        qt, kt, vt, scale=1.0 / math.sqrt(D), is_causal=False,
        implementation=implementation, **lengths,
    )                                                    # (B, L, Hq, D)
    if valid is not None:
        qvalid = jnp.arange(L, dtype=jnp.int32)[None, :] < valid[:, None]
        out = jnp.where(qvalid[:, :, None, None], out, jnp.zeros((), out.dtype))
    return out.reshape(B, L, Hq * v.shape[-1])


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------


def attention_forward(
    x, p, heads, kvheads, freqs=None, keymask=None, lora=None, lora_scale=1.0,
    qchunk=512, attn_impl="ref", attn_mesh=None, attn_block=256,
    attn_interpret=False, attn_shard="batch", act_sharding=None,
    attn_kernel=None,
):
    """Sigmoid-gated multi-head attention (`Attention` in the reference)."""
    B, L, _ = x.shape
    headdim = p["wq"]["weight"].shape[0] // heads
    sh = act_sharding

    q = linear(x, p["wq"], _lget(lora, "wq"), lora_scale, act_sharding=sh)
    k = linear(x, p["wk"], _lget(lora, "wk"), lora_scale, act_sharding=sh)
    v = linear(x, p["wv"], _lget(lora, "wv"), lora_scale, act_sharding=sh)
    gate = linear(x, p["gate"], _lget(lora, "gate"), lora_scale, act_sharding=sh)

    q = q.reshape(B, L, heads, headdim).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, kvheads, headdim).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, kvheads, headdim).transpose(0, 2, 1, 3)

    q = rms_norm(q, p["qknorm"]["qnorm"]["scale"])
    k = rms_norm(k, p["qknorm"]["knorm"]["scale"])
    if freqs is not None:
        q = rope_apply(q, freqs)
        k = rope_apply(k, freqs)

    if attn_impl == "splash":
        out = splash_attention(
            q, k, v, keymask=keymask, mesh=attn_mesh, block=attn_block,
            interpret=attn_interpret, shard=attn_shard,
        )
    elif attn_impl == "aiter":
        out = aiter_attention(
            q, k, v, keymask=keymask, mesh=attn_mesh, kernel=attn_kernel
        )
    elif attn_impl == "cudnn":
        out = cudnn_attention(q, k, v, keymask=keymask, mesh=attn_mesh)
    elif attn_impl == "ref":
        out = scaled_dot_product_attention(q, k, v, keymask=keymask, qchunk=qchunk)
    else:
        raise ValueError(f"unknown attn_impl: {attn_impl!r}")
    out = out * jax.nn.sigmoid(gate)
    return linear(out, p["wo"], _lget(lora, "wo"), lora_scale, act_sharding=sh)


def swiglu_forward(x, p, lora=None, lora_scale=1.0, act_sharding=None):
    sh = act_sharding
    g = linear(x, p["gate"], _lget(lora, "gate"), lora_scale, act_sharding=sh)
    u = linear(x, p["up"], _lget(lora, "up"), lora_scale, act_sharding=sh)
    return linear(jax.nn.silu(g) * u, p["down"], _lget(lora, "down"),
                  lora_scale, act_sharding=sh)


def single_stream_block(x, p, tvec6, freqs, keymask, cfg: Krea2Config,
                        lora=None, lora_scale=1.0, qchunk=512,
                        attn_impl="ref", attn_mesh=None, attn_block=256,
                        attn_interpret=False, attn_shard="batch",
                        act_sharding=None, attn_kernel=None):
    mods = tvec6 + p["mod"]["lin"]  # (B, 1, 6*features)
    prescale, preshift, pregate, postscale, postshift, postgate = jnp.split(
        mods, 6, axis=-1
    )
    h = (1 + prescale) * rms_norm(x, p["prenorm"]["scale"]) + preshift
    x = x + pregate * attention_forward(
        h, p["attn"], cfg.heads, cfg.kvheads, freqs=freqs, keymask=keymask,
        lora=_lget(lora, "attn"), lora_scale=lora_scale, qchunk=qchunk,
        attn_impl=attn_impl, attn_mesh=attn_mesh, attn_block=attn_block,
        attn_interpret=attn_interpret, attn_shard=attn_shard,
        act_sharding=act_sharding, attn_kernel=attn_kernel,
    )
    h = (1 + postscale) * rms_norm(x, p["postnorm"]["scale"]) + postshift
    x = x + postgate * swiglu_forward(
        h, p["mlp"], lora=_lget(lora, "mlp"), lora_scale=lora_scale,
        act_sharding=act_sharding,
    )
    return x


def text_fusion_block(x, p, heads, kvheads, keymask=None, lora=None,
                      lora_scale=1.0, act_sharding=None):
    x = x + attention_forward(
        rms_norm(x, p["prenorm"]["scale"]), p["attn"], heads, kvheads,
        keymask=keymask, lora=_lget(lora, "attn"), lora_scale=lora_scale,
        act_sharding=act_sharding,
    )
    x = x + swiglu_forward(
        rms_norm(x, p["postnorm"]["scale"]), p["mlp"],
        lora=_lget(lora, "mlp"), lora_scale=lora_scale,
        act_sharding=act_sharding,
    )
    return x


def text_fusion_forward(context, p, cfg: Krea2Config, keymask=None,
                        lora=None, lora_scale=1.0, act_sharding=None):
    """TextFusionTransformer: fuse per-layer encoder states then refine.

    context: (B, L, txtlayers, txtdim) -> (B, L, txtdim)
    """
    B, L, N, D = context.shape
    x = context.reshape(B * L, N, D)
    for i in range(2):  # layerwise blocks attend across the N encoder layers
        blk = jax.tree.map(lambda a: a[i], p["layerwise_blocks"])
        x = text_fusion_block(
            x, blk, cfg.txtheads, cfg.txtkvheads,
            lora=_lindex(_lget(lora, "layerwise_blocks"), i),
            lora_scale=lora_scale, act_sharding=act_sharding,
        )
    x = x.reshape(B, L, N, D).transpose(0, 1, 3, 2)  # (B, L, D, N)
    x = linear(x, p["projector"], _lget(lora, "projector"), lora_scale,
               act_sharding=act_sharding)[..., 0]
    for i in range(2):  # refiner blocks attend across the L text tokens
        blk = jax.tree.map(lambda a: a[i], p["refiner_blocks"])
        x = text_fusion_block(
            x, blk, cfg.txtheads, cfg.txtkvheads, keymask=keymask,
            lora=_lindex(_lget(lora, "refiner_blocks"), i),
            lora_scale=lora_scale, act_sharding=act_sharding,
        )
    return x


def text_tokens_forward(params, cfg: Krea2Config, context, keymask=None,
                        lora=None, lora_scale=1.0, act_sharding=None):
    """Text branch of the DiT: text fusion + txtmlp -> (B, L, features).

    Independent of the timestep and the image, so it can be computed once
    per prompt and reused across sampling steps (`dit_forward(txt_tokens=)`).
    act_sharding: as in `dit_forward` (applied to every linear output).
    """
    sh = act_sharding
    txt = text_fusion_forward(
        context, params["txtfusion"], cfg, keymask=keymask,
        lora=_lget(lora, "txtfusion"), lora_scale=lora_scale,
        act_sharding=sh,
    )
    txt = rms_norm(txt, params["txtmlp"]["0"]["scale"])
    txt = linear(txt, params["txtmlp"]["1"], _lget(lora, "txtmlp", "1"),
                 lora_scale, act_sharding=sh)
    return linear(
        gelu_tanh(txt), params["txtmlp"]["3"], _lget(lora, "txtmlp", "3"),
        lora_scale, act_sharding=sh,
    )


def last_layer_forward(x, p, tvec, lora=None, lora_scale=1.0,
                       act_sharding=None):
    """Modulated RMSNorm + projection back to patch pixels."""
    mods = tvec + p["modulation"]["lin"][None]  # (B, 2, features)
    scale, shift = mods[:, 0:1], mods[:, 1:2]
    x = (1 + scale) * rms_norm(x, p["norm"]["scale"]) + shift
    return linear(x, p["linear"], _lget(lora, "linear"), lora_scale,
                  act_sharding=act_sharding)


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
    attn_impl="ref",     # "ref" (exact fp32 softmax), "splash" (fused TPU),
                         # "aiter" (jax-aiter varlen flash attn, ROCm) or
                         # "cudnn" (jax.nn.dot_product_attention: cuDNN
                         # flash attn on GPU, its "xla" path elsewhere)
    attn_mesh=None,      # mesh for splash/aiter (shard_map) and cudnn
                         # (batch sharding constraint); see attn_shard
    attn_block=256,      # splash tile size; must divide the padded seq len
    attn_interpret=False,  # run splash in Pallas interpret mode (CPU tests)
    attn_shard="batch",  # splash under attn_mesh: shard "batch" or "heads"
    act_sharding=None,   # Sharding for linear outputs + the scan carry
    attn_kernel=None,    # aiter only: flash_attn_varlen-compatible callable
    txt_tokens=None,     # precomputed text_tokens_forward(...); skips context
):
    """Velocity prediction for the flow ODE. Returns (B, Limg, patch²·channels).

    txt_tokens: the output of `text_tokens_forward` for this context. The
    text branch does not depend on t, so a sampler computes it once per
    prompt and passes it to every step (`context` is then unused).

    remat: True/"full" checkpoints each scanned block (recompute everything in
    backward), "dots" uses the dots_with_no_batch_dims_saveable policy (dense
    projection outputs are saved, attention still rematerializes — spends HBM
    to skip recomputing the big matmuls), False/"none" disables remat.

    act_sharding: optional Sharding (normally NamedSharding(mesh, P("fsdp")))
    applied to every linear output and to the scan carry at block entry and
    exit, so activations stay batch-sharded under FSDP. None: unconstrained.

    attn_impl="aiter": tokens are reordered per sample (after the padding to
    a multiple of 256) so valid tokens form a prefix — stable, so text then
    image order is kept — as the varlen kernel needs; positions and mask are
    permuted with them and the original order is restored after the blocks,
    so outputs are positionally identical to the other impls. The reorder
    acts on the combined sequence, so it is the same with `txt_tokens`.
    attn_shard only applies to splash; aiter under attn_mesh always shards
    the batch (it cannot head-shard a batch-1 sampler).

    attn_impl="cudnn" uses the same reorder (the per-sample valid count is
    passed as query/key sequence lengths, which also describe a prefix) and
    `cudnn_attention`; on non-GPU backends it runs dot_product_attention's
    "xla" implementation with identical semantics. Under attn_mesh it
    constrains the batch to the "fsdp" axis when the batch divides it and
    otherwise leaves sharding to the partitioner (no shard_map).
    """
    B, Limg, _ = img.shape
    Ltxt = (txt_tokens if txt_tokens is not None else context).shape[1]
    sh = act_sharding

    x = linear(img, params["first"], _lget(lora, "first"), lora_scale,
               act_sharding=sh)

    tv = timestep_embedding(t, cfg.tdim, dtype=x.dtype)
    tv = linear(tv, params["tmlp"]["0"], _lget(lora, "tmlp", "0"), lora_scale,
                act_sharding=sh)
    tv = linear(
        gelu_tanh(tv), params["tmlp"]["2"], _lget(lora, "tmlp", "2"), lora_scale,
        act_sharding=sh,
    )                                                        # (B, 1, features)
    tvec6 = linear(
        gelu_tanh(tv), params["tproj"]["1"], _lget(lora, "tproj", "1"),
        lora_scale, act_sharding=sh,
    )                                                        # (B, 1, 6*features)

    if txt_tokens is None:
        txt = text_tokens_forward(
            params, cfg, context, mask[:, :Ltxt], lora=lora,
            lora_scale=lora_scale, act_sharding=sh,
        )
    else:
        txt = _constrain(txt_tokens.astype(x.dtype), sh)

    combined = jnp.concatenate([txt, x], axis=1)

    # Pad the combined sequence to a multiple of 256 (static shapes under jit).
    full = combined.shape[1]
    padlen = (-full) % cfg.seq_pad_multiple
    if padlen:
        combined = jnp.pad(combined, ((0, 0), (0, padlen), (0, 0)))
        mask = jnp.pad(mask, ((0, 0), (0, padlen)))
        pos = jnp.pad(pos, ((0, 0), (0, padlen), (0, 0)))

    inv_perm = None
    if attn_impl in ("aiter", "cudnn"):
        # valid tokens first (stable): the varlen kernel's logical prefix /
        # the cuDNN padding mask's sequence lengths
        perm = jnp.argsort((~mask).astype(jnp.int32), axis=1, stable=True)
        inv_perm = jnp.argsort(perm, axis=1)
        combined = jnp.take_along_axis(combined, perm[..., None], axis=1)
        mask = jnp.take_along_axis(mask, perm, axis=1)
        pos = jnp.take_along_axis(pos, perm[..., None], axis=1)

    freqs = rope_freqs(pos, cfg.axdims, cfg.theta)

    blocks = params["blocks"]
    block_fn = partial(
        single_stream_block, cfg=cfg, lora_scale=lora_scale, qchunk=qchunk,
        attn_impl=attn_impl, attn_mesh=attn_mesh, attn_block=attn_block,
        attn_interpret=attn_interpret, attn_shard=attn_shard,
        act_sharding=sh, attn_kernel=attn_kernel,
    )

    if lora is not None and "blocks" in lora:
        def body(carry, layer):
            p, l = layer
            out = block_fn(_constrain(carry, sh), p, tvec6, freqs, mask, lora=l)
            return _constrain(out, sh), None
        xs = (blocks, lora["blocks"])
    else:
        def body(carry, p):
            out = block_fn(_constrain(carry, sh), p, tvec6, freqs, mask)
            return _constrain(out, sh), None
        xs = blocks

    if remat is True or remat == "full":
        body = jax.checkpoint(body, prevent_cse=False)
    elif remat == "dots":
        body = jax.checkpoint(
            body,
            policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
            prevent_cse=False,
        )
    elif remat not in (False, None, "none"):
        raise ValueError(f"unknown remat mode: {remat!r}")
    combined, _ = jax.lax.scan(body, combined, xs)

    if inv_perm is not None:
        combined = jnp.take_along_axis(combined, inv_perm[..., None], axis=1)

    out = last_layer_forward(
        combined, params["last"], tv, lora=_lget(lora, "last"),
        lora_scale=lora_scale, act_sharding=sh,
    )
    return out[:, Ltxt : Ltxt + Limg]


# ---------------------------------------------------------------------------
# latent <-> token helpers
# ---------------------------------------------------------------------------


def patchify(lat, patch):
    """(B, C, H, W) -> (B, (H/p)*(W/p), C*p*p) matching the reference layout."""
    B, C, H, W = lat.shape
    if H % patch or W % patch:
        raise ValueError(
            f"latent grid {H}x{W} must be divisible by patch size {patch}"
        )
    h, w = H // patch, W // patch
    x = lat.reshape(B, C, h, patch, w, patch)
    x = x.transpose(0, 2, 4, 1, 3, 5)  # B h w C p p
    return x.reshape(B, h * w, C * patch * patch)


def unpatchify(tokens, patch, channels, H, W):
    """Inverse of `patchify`. H, W are latent-grid sizes."""
    B = tokens.shape[0]
    if H % patch or W % patch:
        raise ValueError(
            f"latent grid {H}x{W} must be divisible by patch size {patch}"
        )
    h, w = H // patch, W // patch
    if tokens.shape[1] != h * w:
        raise ValueError(
            f"got {tokens.shape[1]} tokens for a {h}x{w} ({h * w}-token) grid"
        )
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


def build_positions_and_mask_bucketed(txt_mask, hw, img_tokens):
    """Combined (pos, mask) for [text, image] sequences whose image part is
    a static `img_tokens` rows holding a per-sample (h_b, w_b) token grid
    (aspect-bucketed batches; rows >= h_b * w_b are padding).

    txt_mask: (B, Ltxt) bool; hw: (B, 2) int (h_tokens, w_tokens) per sample
    with h_b * w_b <= img_tokens. Returns pos (B, Ltxt+img_tokens, 3) f32 —
    text rows (0, 0, 0), image token i of sample b at (0, i // w_b, i % w_b)
    for i < h_b * w_b and (0, 0, 0) beyond — and mask (B, Ltxt+img_tokens)
    bool = txt_mask ++ (i < h_b * w_b). Pure jnp (traceable under jit with
    hw a traced array); equals `build_positions_and_mask(txt_mask, h, w)`
    when every sample has (h_b, w_b) == (h, w) and h * w == img_tokens.
    """
    B, Ltxt = txt_mask.shape
    hw = jnp.asarray(hw).astype(jnp.int32)
    h, w = hw[:, 0:1], hw[:, 1:2]                            # (B, 1)
    i = jnp.arange(img_tokens, dtype=jnp.int32)[None, :]     # (1, N)
    valid = i < h * w                                        # (B, N)
    w_safe = jnp.maximum(w, 1)
    row = jnp.where(valid, i // w_safe, 0)
    col = jnp.where(valid, i % w_safe, 0)
    imgpos = jnp.stack([jnp.zeros_like(row), row, col], axis=-1)
    imgpos = imgpos.astype(jnp.float32)
    txtpos = jnp.zeros((B, Ltxt, 3), dtype=jnp.float32)
    pos = jnp.concatenate([txtpos, imgpos], axis=1)
    mask = jnp.concatenate([jnp.asarray(txt_mask).astype(bool), valid], axis=1)
    return pos, mask
