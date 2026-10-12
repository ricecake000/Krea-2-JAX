"""JAX port of the Qwen3-VL text tower (Qwen/Qwen3-VL-4B-Instruct), used as
the ONLINE text encoder of the full fine-tuning path.

Only what the DiT conditioning needs: token embeddings + decoder layers, and
the HF `hidden_states[i]` of `cfg.select_layers` at the rows after the
prompt prefix (see krea2_jax/text_template.py for the input layout
`[prefix | caption | suffix | pad ...]`). No vision tower, no lm_head, no KV
cache. HF semantics reproduced here (transformers Qwen3VLTextModel):

* hidden_states[0] = token embeddings; hidden_states[i] (1 <= i < n_layers)
  = output of decoder layer i-1 (pre final-norm). hidden_states[n_layers]
  is version dependent: transformers >= 5 ties it to `last_hidden_state`
  (final RMSNorm applied), 4.x returns the raw last-layer output
  (`cfg.normed_last_hidden`, default True = 5.x). Irrelevant for the 4B
  select set (2, 5, ..., 35): only layers 1..max(select_layers) are computed
  (and loaded), i.e. 35 of 36 layers, and the final norm is never used.
* RMSNorm: normalize in f32, cast back to the input dtype, then
  `weight * x` (eps 1e-6). q_norm/k_norm are RMSNorms over head_dim applied
  BEFORE RoPE.
* RoPE: interleaved MRoPE (mrope_section [24, 20, 20]); for text-only input
  the three position axes are identical, so it reduces exactly to 1-D RoPE
  over head_dim with the rotate_half convention (tested against HF).
* Causal attention with a key padding mask, GQA, scale head_dim**-0.5;
  SwiGLU MLP (silu(gate) * up -> down), no biases.

Positions (`cfg.position_mode`): the official conditioner
(scripts/cache_dataset.py::TextEncoder) pads PREFIX+caption to a fixed length
and appends the SUFFIX after the padding. How HF assigns RoPE positions to
that layout depends on the transformers version:
* "contiguous" (transformers 4.x): positions = cumsum(attention_mask) - 1,
  so the suffix continues right after the caption. On our compact layout
  this is simply arange(L); the valid rows do not depend on the padding.
* "padded" (transformers >= 5.0: text-only input ignores the mask for
  positions and uses arange over the padded sequence): suffix token j sits
  at position prefix_len + text_rows - suffix_len + j regardless of the
  caption length (equal to TextEncoder(max_length=text_rows) under
  transformers 5). Caption rows are identical in both modes.

Numerics: matmuls take the param dtype with f32 accumulation
(`preferred_element_type=f32`) and cast back; softmax, RMSNorm and RoPE in
f32; the residual stream stays in the param dtype (f32 params -> all f32).

This module must not import the DiT (krea2_jax.model).
"""

import dataclasses
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from krea2_jax.hashing import file_sha256
from krea2_jax.text_template import (
    QWEN3_PREFIX_LEN,
    QWEN3_SUFFIX_LEN,
    SELECT_LAYERS,
)

POSITION_MODES = ("contiguous", "padded")

# stacked per-layer field -> HF key suffix under `<prefix>layers.{i}.`
LAYER_FIELDS = {
    "input_layernorm": "input_layernorm.weight",
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Qwen3TextConfig:
    """Qwen3-VL text tower hyper-parameters (defaults: Qwen3-VL-4B).

    prefix_len / suffix_len: token counts of the conditioner PREFIX/SUFFIX
    for this tokenizer (34 / 5 for the real one). position_mode: see the
    module docstring. normed_last_hidden: hidden_states[num_layers] is the
    final-normed output (transformers >= 5) or the raw last-layer output
    (4.x); only matters when num_layers is in select_layers. Frozen and
    hashable, so it can be a static jit argument.
    """

    vocab_size: int = 151936
    hidden_size: int = 2560
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5e6
    mrope_section: tuple = (24, 20, 20)
    mrope_interleaved: bool = True
    select_layers: tuple = SELECT_LAYERS
    prefix_len: int = QWEN3_PREFIX_LEN
    suffix_len: int = QWEN3_SUFFIX_LEN
    position_mode: str = "contiguous"
    normed_last_hidden: bool = True

    def __post_init__(self):
        object.__setattr__(self, "select_layers",
                           tuple(int(i) for i in self.select_layers))
        object.__setattr__(self, "mrope_section",
                           tuple(int(i) for i in self.mrope_section))
        if not self.select_layers:
            raise ValueError("select_layers is empty")
        bad = [i for i in self.select_layers if not 0 <= i <= self.num_layers]
        if bad:
            raise ValueError(
                f"select_layers {bad} outside [0, {self.num_layers}]")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        if self.position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode must be one of {POSITION_MODES}")

    @property
    def n_run_layers(self):
        """Decoder layers actually computed (and loaded)."""
        return max(self.select_layers)

    @property
    def needs_final_norm(self):
        """The final RMSNorm (`params["norm"]`) is needed."""
        return self.normed_last_hidden and self.num_layers in self.select_layers


def from_hf_config(path_or_dict, **overrides):
    """Qwen3TextConfig from an HF Qwen3-VL config.

    path_or_dict: a model dir (reads config.json; if the dir also has
    tokenizer.json, prefix_len/suffix_len are derived from it), a
    config.json path, or the parsed dict (full Qwen3-VL config with
    `text_config`, or the text config itself). Handles both the 4.x
    (`rope_theta` + `rope_scaling`) and 5.x (`rope_parameters`) layouts.
    overrides: any Qwen3TextConfig field (select_layers, prefix_len, ...).
    """
    derived = {}
    if isinstance(path_or_dict, dict):
        cfgd = path_or_dict
    else:
        p = Path(path_or_dict)
        cfg_path = p / "config.json" if p.is_dir() else p
        cfgd = json.loads(cfg_path.read_text())
        tok = cfg_path.parent / "tokenizer.json"
        if tok.exists() and not {"prefix_len", "suffix_len"} <= set(overrides):
            from krea2_jax.text_template import CaptionTemplate

            t = CaptionTemplate(load_tokenizer(cfg_path.parent))
            derived = {"prefix_len": t.prefix_len,
                       "suffix_len": len(t.suffix_ids)}
    tc = cfgd.get("text_config", cfgd)
    rope = dict(tc.get("rope_parameters") or tc.get("rope_scaling") or {})
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type != "default":
        raise ValueError(f"unsupported rope_type {rope_type!r}")
    if tc.get("attention_bias", False):
        raise ValueError("attention_bias=True is not supported")
    if tc.get("hidden_act", "silu") != "silu":
        raise ValueError(f"unsupported hidden_act {tc.get('hidden_act')!r}")
    heads = int(tc["num_attention_heads"])
    head_dim = tc.get("head_dim") or tc["hidden_size"] // heads
    kw = dict(
        vocab_size=int(tc["vocab_size"]),
        hidden_size=int(tc["hidden_size"]),
        num_layers=int(tc["num_hidden_layers"]),
        num_heads=heads,
        num_kv_heads=int(tc.get("num_key_value_heads", heads)),
        head_dim=int(head_dim),
        intermediate_size=int(tc["intermediate_size"]),
        rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
        rope_theta=float(rope.get("rope_theta", tc.get("rope_theta", 1e6))),
        mrope_section=tuple(rope.get("mrope_section", (24, 20, 20))),
        mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
    )
    kw.update(derived)
    kw.update(overrides)
    return Qwen3TextConfig(**kw)


# ---------------------------------------------------------------------------
# tokenizer / model resolution
# ---------------------------------------------------------------------------


def load_tokenizer(model_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))


def resolve_text_encoder(spec):
    """A local dir is used as is; anything else is an HF repo id, fetched
    (config, tokenizer and safetensors only) with snapshot_download."""
    p = Path(spec)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        spec, allow_patterns=["*.json", "*.safetensors", "merges.txt",
                              "vocab.json"]))


# ---------------------------------------------------------------------------
# params
# ---------------------------------------------------------------------------


def _param_shapes(cfg):
    H, I, D = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    qd, kd = cfg.num_heads * D, cfg.num_kv_heads * D
    layer = {
        "input_layernorm": (H,),
        "post_attention_layernorm": (H,),
        "q_proj": (qd, H),
        "k_proj": (kd, H),
        "v_proj": (kd, H),
        "o_proj": (H, qd),
        "q_norm": (D,),
        "k_norm": (D,),
        "gate_proj": (I, H),
        "up_proj": (I, H),
        "down_proj": (H, I),
    }
    n = cfg.n_run_layers
    shapes = {
        "embed_tokens": (cfg.vocab_size, H),
        "layers": {k: (n,) + s for k, s in layer.items()},
    }
    if cfg.needs_final_norm:
        shapes["norm"] = (H,)
    return shapes


def _is_shape(x):
    return isinstance(x, tuple)


def abstract_qwen3_text_params(cfg, dtype=jnp.bfloat16):
    """ShapeDtypeStruct pytree with exactly the layout
    `load_qwen3_text_params` produces (for AOT analysis)."""
    return jax.tree.map(lambda s: jax.ShapeDtypeStruct(s, dtype),
                        _param_shapes(cfg), is_leaf=_is_shape)


def init_qwen3_text_params(cfg, key, dtype=jnp.float32):
    """Random params (tests/benchmarks): embeddings N(0, 1), linears
    N(0, 1/fan_in), norm weights 1 + 0.1 N(0, 1)."""
    shapes = _param_shapes(cfg)
    leaves, treedef = jax.tree.flatten(shapes, is_leaf=_is_shape)
    paths = [jax.tree_util.keystr(p) for p, _ in
             jax.tree_util.tree_flatten_with_path(shapes, is_leaf=_is_shape)[0]]
    keys = jax.random.split(key, len(leaves))
    out = []
    for path, shape, k in zip(paths, leaves, keys):
        x = jax.random.normal(k, shape, jnp.float32)
        if "norm" in path:
            x = 1.0 + 0.1 * x
        elif "embed" not in path:
            x = x / math.sqrt(shape[-1])
        out.append(x.astype(dtype))
    return jax.tree.unflatten(treedef, out)


def param_count(params):
    return int(sum(math.prod(x.shape) for x in jax.tree.leaves(params)))


def _weight_map(model_dir):
    """{tensor key: safetensors file path} for an HF model dir."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text())["weight_map"]
        return {k: model_dir / f for k, f in wm.items()}
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights in {model_dir}")
    from safetensors import safe_open

    out = {}
    for fp in files:
        with safe_open(str(fp), framework="numpy") as f:
            for k in f.keys():
                out[k] = fp
    return out


def _loader_keys(prefix, cfg):
    """The tensor keys load_qwen3_text_params reads for `cfg`."""
    keys = [prefix + "embed_tokens.weight"]
    if cfg.needs_final_norm:
        keys.append(prefix + "norm.weight")
    for suffix in LAYER_FIELDS.values():
        keys += [f"{prefix}layers.{i}.{suffix}"
                 for i in range(cfg.n_run_layers)]
    return keys


def text_encoder_identity(model_dir, cfg, stats=None):
    """Hex sha256 identifying the text encoder a run conditions on.

    Covers (a) the bytes of tokenizer.json, (b) the resolved
    Qwen3TextConfig `cfg` (every field, incl. select_layers, position_mode,
    normed_last_hidden, prefix/suffix_len) as sorted-key JSON, and (c) the
    full contents of every safetensors file load_qwen3_text_params would
    read for `cfg` (from the index, or the single/globbed files): each file
    is streamed through sha256 in 1 MiB reads and its digest enters the
    hash, in the order of the smallest loader key the file holds. So any
    change to a weight VALUE (same shape/dtype) changes the identity, as does
    a different checkpoint/revision/tokenizer/config. Whole files are hashed,
    so a vision-tower or lm_head edit in a file that also holds used text
    weights changes it too (conservative); files holding no used key are
    not read. Cost: one sequential read of those files (~8 GB for
    Qwen3-VL-4B). No filesystem paths or file names enter the hash: the
    same encoder at another path or under other file names has the same
    identity.

    stats: optional dict, filled with {"files": n, "bytes": total bytes
    hashed} for the caller to report (nothing is printed here)."""
    model_dir = Path(model_dir)
    h = hashlib.sha256()
    h.update(b"tokenizer.json\0")
    h.update(bytes.fromhex(file_sha256(model_dir / "tokenizer.json")))
    h.update(b"config\0")
    h.update(json.dumps(dataclasses.asdict(cfg), sort_keys=True,
                        separators=(",", ":")).encode("utf-8"))
    wm = _weight_map(model_dir)
    used = _loader_keys(_text_prefix(wm), cfg)
    missing = [k for k in used if k not in wm]
    if missing:
        raise KeyError(f"missing weights {missing[:5]} in {model_dir}")
    first_key = {}
    for k in used:
        fp = wm[k]
        first_key[fp] = min(first_key.get(fp, k), k)
    nbytes = 0
    h.update(b"weights\0")
    for fp in sorted(first_key, key=first_key.get):
        fh = hashlib.sha256()
        nbytes += file_sha256(fp, h=fh)
        h.update(fh.digest())
    if stats is not None:
        stats.update(files=len(first_key), bytes=nbytes)
    return h.hexdigest()


def _text_prefix(keys):
    """Key prefix of the text model, e.g. 'model.language_model.'."""
    suffix = "embed_tokens.weight"
    cands = [k[:-len(suffix)] for k in keys
             if k.endswith(suffix) and "visual" not in k]
    if len(cands) != 1:
        raise ValueError(f"cannot locate the text embeddings: {cands}")
    return cands[0]


def load_qwen3_text_params(model_dir, dtype=jnp.bfloat16, mesh=None,
                           cfg=None):
    """Load the text tower of an HF Qwen3-VL checkpoint.

    Reads safetensors with framework="numpy" (bf16 via ml_dtypes), stacks
    layers 0..cfg.n_run_layers-1 along a leading axis, and places each leaf
    on the default device (mesh=None) or REPLICATED on `mesh`
    (NamedSharding(mesh, P())). Fields are assembled and placed one at a
    time, so the host peak is the placed params plus one stacked field.
    cfg defaults to from_hf_config(model_dir). Returns
        {"embed_tokens": (V, H),
         "layers": {input_layernorm (n, H), post_attention_layernorm (n, H),
                    q_proj (n, nh*hd, H), k_proj (n, nkv*hd, H),
                    v_proj (n, nkv*hd, H), o_proj (n, H, nh*hd),
                    q_norm (n, hd), k_norm (n, hd), gate_proj (n, I, H),
                    up_proj (n, I, H), down_proj (n, H, I)},
         "norm": (H,)   # only when cfg.needs_final_norm}
    with torch (out, in) linear layout and n = cfg.n_run_layers.
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    if cfg is None:
        cfg = from_hf_config(model_dir)
    np_dtype = np.dtype(dtype)
    wm = _weight_map(model_dir)
    prefix = _text_prefix(wm)
    shapes = _param_shapes(cfg)
    sharding = NamedSharding(mesh, P()) if mesh is not None else None

    def place(arr):
        if sharding is not None:
            return jax.device_put(arr, sharding)
        return jax.device_put(arr)

    with ExitStack() as stack:
        handles = {}

        def get(key, shape):
            if key not in wm:
                raise KeyError(f"missing weight {key!r} in {model_dir}")
            fp = wm[key]
            if fp not in handles:
                handles[fp] = stack.enter_context(
                    safe_open(str(fp), framework="numpy"))
            t = handles[fp].get_tensor(key)
            if tuple(t.shape) != tuple(shape):
                raise ValueError(
                    f"{key}: shape {tuple(t.shape)} != expected {shape}")
            return t if t.dtype == np_dtype else t.astype(np_dtype)

        params = {"embed_tokens": place(
            get(prefix + "embed_tokens.weight", shapes["embed_tokens"]))}
        if cfg.needs_final_norm:
            params["norm"] = place(get(prefix + "norm.weight", shapes["norm"]))
        layers = {}
        for field, suffix in LAYER_FIELDS.items():
            shape = shapes["layers"][field]
            arr = np.empty(shape, np_dtype)
            for i in range(shape[0]):
                arr[i] = get(f"{prefix}layers.{i}.{suffix}", shape[1:])
            layers[field] = place(arr)
            del arr
        params["layers"] = layers
    return params


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def _linear(x, w):
    """x @ w.T, torch (out, in) layout, f32 accumulation, output in x.dtype."""
    y = jnp.einsum("...i,oi->...o", x, w.astype(x.dtype),
                   preferred_element_type=jnp.float32)
    return y.astype(x.dtype)


def rms_norm(x, weight, eps):
    """HF Qwen3 RMSNorm: f32 normalize, cast back, then weight * x."""
    t = x.astype(jnp.float32)
    t = t * jax.lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + eps)
    return weight.astype(x.dtype) * t.astype(x.dtype)


def rope_inv_freq(cfg):
    """(head_dim/2,) f32, as HF compute_default_rope_parameters."""
    d = cfg.head_dim
    exponent = np.arange(0, d, 2, dtype=np.float32) / np.float32(d)
    return (1.0 / (np.float32(cfg.rope_theta) ** exponent)).astype(np.float32)


def rope_cos_sin(cfg, positions):
    """positions (B, L) int -> cos, sin (B, L, head_dim) f32 (1-D RoPE,
    halves duplicated like HF's cat((freqs, freqs)))."""
    inv = jnp.asarray(rope_inv_freq(cfg))
    freqs = positions.astype(jnp.float32)[..., None] * inv
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x, cos, sin):
    """x (B, L, heads, D); cos/sin (B, L, D) f32; rotate_half convention."""
    xf = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    rot = jnp.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
    out = xf * cos[:, :, None] + rot * sin[:, :, None]
    return out.astype(x.dtype)


def text_positions(cfg, txt_len, total_len):
    """(B, L) int32 RoPE positions for the compact layout (see module doc)."""
    pos = jnp.broadcast_to(jnp.arange(total_len, dtype=jnp.int32),
                           (txt_len.shape[0], total_len))
    if cfg.position_mode == "contiguous":
        return pos
    text_rows = total_len - cfg.prefix_len
    end = cfg.prefix_len + txt_len[:, None]
    is_suffix = (pos >= end - cfg.suffix_len) & (pos < end)
    return jnp.where(is_suffix, pos + (text_rows - txt_len)[:, None], pos)


def _attention(x, p, cos, sin, keymask, cfg):
    B, L, _ = x.shape
    D, nh, nkv = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    rep = nh // nkv
    eps = cfg.rms_norm_eps
    q = rms_norm(_linear(x, p["q_proj"]).reshape(B, L, nh, D), p["q_norm"], eps)
    k = rms_norm(_linear(x, p["k_proj"]).reshape(B, L, nkv, D), p["k_norm"], eps)
    v = _linear(x, p["v_proj"]).reshape(B, L, nkv, D)
    q = apply_rope(q, cos, sin).reshape(B, L, nkv, rep, D)
    k = apply_rope(k, cos, sin)
    s = jnp.einsum("bqgrd,bkgd->bgrqk", q, k,
                   preferred_element_type=jnp.float32) * (D ** -0.5)
    s = jnp.where(keymask[:, None, None], s, -1e30)
    w = jax.nn.softmax(s, axis=-1).astype(x.dtype)
    o = jnp.einsum("bgrqk,bkgd->bqgrd", w, v,
                   preferred_element_type=jnp.float32).astype(x.dtype)
    return _linear(o.reshape(B, L, nh * D), p["o_proj"])


def _decoder_layer(h, p, cos, sin, keymask, cfg):
    eps = cfg.rms_norm_eps
    h = h + _attention(rms_norm(h, p["input_layernorm"], eps), p, cos, sin,
                       keymask, cfg)
    x = rms_norm(h, p["post_attention_layernorm"], eps)
    m = jax.nn.silu(_linear(x, p["gate_proj"]).astype(jnp.float32))
    m = (m.astype(x.dtype) * _linear(x, p["up_proj"]))
    return h + _linear(m, p["down_proj"])


def text_states(params, cfg, ids, txt_len):
    """Qwen3-VL hidden states for the DiT.

    params: pytree from load/init/abstract_qwen3_text_params (dtype = compute
    dtype). ids: (B, prefix_len + text_rows) int32 laid out as
    `[prefix | caption | suffix | pad...]` (CaptionTemplate.encode_captions);
    txt_len: (B,) int32 valid rows after the prefix.
    Returns states (B, text_rows, len(select_layers), hidden) in the param
    dtype = HF hidden_states[i][:, prefix_len:] stacked over select_layers,
    with invalid rows zeroed, and mask (B, text_rows) bool (row < txt_len).
    Gradient-free (stop_gradient). Traceable: jit it with cfg static (cfg
    is hashable); ids/txt_len may be batch-sharded, params replicated.
    """
    ids = jnp.asarray(ids, jnp.int32)
    txt_len = jnp.asarray(txt_len, jnp.int32)
    B, L = ids.shape
    pl = cfg.prefix_len
    text_rows = L - pl
    if text_rows <= 0:
        raise ValueError(f"ids length {L} must exceed prefix_len {pl}")
    n_run = cfg.n_run_layers
    layers = params["layers"]
    n_have = layers["q_proj"].shape[0]
    if n_have < n_run:
        raise ValueError(f"params hold {n_have} layers, need {n_run}")
    if n_have > n_run:
        layers = jax.tree.map(lambda a: a[:n_run], layers)

    dtype = params["embed_tokens"].dtype
    h = jnp.take(params["embed_tokens"], ids, axis=0).astype(dtype)
    pos_idx = jnp.arange(L, dtype=jnp.int32)
    key_valid = pos_idx[None, :] < (pl + txt_len)[:, None]          # (B, L)
    keymask = key_valid[:, None, :] & (pos_idx[None, :, None]
                                       >= pos_idx[None, None, :])    # (B, Lq, Lk)
    cos, sin = rope_cos_sin(cfg, text_positions(cfg, txt_len, L))

    def body(carry, p):
        out = _decoder_layer(carry, p, cos, sin, keymask, cfg)
        return out, out[:, pl:]

    if n_run > 0:
        h_last, ys = jax.lax.scan(body, h, layers)
    else:
        h_last, ys = h, None

    picked = []
    for i in cfg.select_layers:
        if i == 0:
            picked.append(h[:, pl:])
        elif i == cfg.num_layers and cfg.normed_last_hidden:
            picked.append(rms_norm(h_last[:, pl:], params["norm"],
                                   cfg.rms_norm_eps))
        else:
            picked.append(ys[i - 1])
    states = jnp.stack(picked, axis=2)                 # (B, T, S, H)
    mask = jnp.arange(text_rows)[None, :] < txt_len[:, None]
    states = jnp.where(mask[:, :, None, None], states, jnp.zeros((), dtype))
    return jax.lax.stop_gradient(states), mask
