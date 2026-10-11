"""Checkpoint I/O: official safetensors -> sharded JAX pytree, LoRA save/load.

The official `raw.safetensors` / `turbo.safetensors` files store the PyTorch
state_dict of `SingleStreamDiT` in bfloat16. We keep torch's (out, in) Linear
layout and the exact key names, but nest keys by "." and stack repeated blocks
(`blocks.{i}.*`, `txtfusion.layerwise_blocks.{i}.*`,
`txtfusion.refiner_blocks.{i}.*`) along a leading axis for `lax.scan`.
"""

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from safetensors import safe_open

STACKED_PREFIXES = (
    "blocks",
    "txtfusion.layerwise_blocks",
    "txtfusion.refiner_blocks",
)

_STACK_RE = re.compile(
    r"^(blocks|txtfusion\.layerwise_blocks|txtfusion\.refiner_blocks)\.(\d+)\.(.+)$"
)


# ---------------------------------------------------------------------------
# mesh & sharding
# ---------------------------------------------------------------------------


def make_mesh():
    """1D mesh over all local devices; FSDP + data parallel share the axis."""
    return Mesh(np.array(jax.devices()), axis_names=("fsdp",))


def param_partition_spec(shape, mesh_size, stacked):
    """Shard the largest dim divisible by the mesh (skip the stack axis)."""
    if mesh_size <= 1:
        return P()
    best, best_size = None, 0
    for i, d in enumerate(shape):
        if stacked and i == 0:
            continue
        if d % mesh_size == 0 and d >= best_size and d >= mesh_size:
            best, best_size = i, d
    if best is None:
        return P()
    spec = [None] * len(shape)
    spec[best] = "fsdp"
    return P(*spec)


def batch_sharding(mesh):
    return NamedSharding(mesh, P("fsdp"))


def replicated_sharding(mesh):
    return NamedSharding(mesh, P())


# ---------------------------------------------------------------------------
# base checkpoint loading
# ---------------------------------------------------------------------------


def _nest(tree, path, leaf):
    node = tree
    for k in path[:-1]:
        node = node.setdefault(k, {})
    node[path[-1]] = leaf


def load_krea2_params(path, mesh=None, dtype=None, sharding="fsdp"):
    """Load the official DiT safetensors into a (sharded) nested pytree.

    Tensors are read on host CPU, repeated blocks are stacked, and each leaf
    is placed on the mesh. sharding="fsdp" shards each leaf's largest
    divisible dim across the mesh; "replicate" keeps a full copy per device
    (frozen base + pure data parallelism: no weight all-gathers per step, at
    ~24 GiB HBM per device for the 12.9B model). With mesh=None the tree
    stays on the default device, replicated.
    """
    if sharding not in ("fsdp", "replicate"):
        raise ValueError(f"unknown base sharding: {sharding!r}")
    cpu = jax.devices("cpu")[0]
    flat = {}
    stacked = {}  # prefix -> field -> {idx: array}

    with safe_open(path, framework="flax") as f:
        keys = list(f.keys())
        with jax.default_device(cpu):
            for key in keys:
                tensor = f.get_tensor(key)
                if dtype is not None:
                    tensor = tensor.astype(dtype)
                m = _STACK_RE.match(key)
                if m:
                    prefix, idx, rest = m.group(1), int(m.group(2)), m.group(3)
                    stacked.setdefault(prefix, {}).setdefault(rest, {})[idx] = tensor
                else:
                    flat[key] = tensor

    params = {}
    for key, tensor in flat.items():
        _nest(params, key.split("."), _place(tensor, mesh, stacked=False, sharding=sharding))
    for prefix, fields in stacked.items():
        for rest, by_idx in fields.items():
            n = max(by_idx) + 1
            assert len(by_idx) == n, f"missing block index under {prefix}.{rest}"
            with jax.default_device(cpu):
                # pop as we stack: keeping every per-block tensor alive
                # alongside its stacked copy doubles the ~26 GB host peak
                arr = jnp.stack([by_idx.pop(i) for i in range(n)], axis=0)
            _nest(
                params,
                prefix.split(".") + rest.split("."),
                _place(arr, mesh, stacked=True, sharding=sharding),
            )
            del arr
    return params


def abstract_krea2_params(cfg, dtype=jnp.float32):
    """`jax.ShapeDtypeStruct` pytree of the DiT parameters (no weights, no
    torch), with exactly the nesting/stacking `load_krea2_params` produces:
    `blocks`, `txtfusion.layerwise_blocks` and `txtfusion.refiner_blocks`
    stacked with the layer axis first. Unsharded (pair with
    `param_shardings` for AOT compilation).

    Shapes follow the official `SingleStreamDiT` module: torch Linear
    (out, in) layout; `first`, `tmlp.*`, `txtmlp.1/3`, `tproj.1` and
    `last.linear` always carry a bias, the attention/MLP linears (blocks and
    text fusion) only when `cfg.bias`, `txtfusion.projector` never; SwiGLU
    hidden = round_up(int(2*features/3)*multiplier, 128); the text fusion
    always has 2 layerwise + 2 refiner blocks.
    """

    def leaf(*shape):
        return jax.ShapeDtypeStruct(tuple(int(d) for d in shape), dtype)

    def lin(out_f, in_f, bias):
        p = {"weight": leaf(out_f, in_f)}
        if bias:
            p["bias"] = leaf(out_f)
        return p

    def norm(features):
        return {"scale": leaf(features)}

    def swiglu(features):
        hidden = int(2 * features / 3) * cfg.multiplier
        hidden = 128 * ((hidden + 127) // 128)
        return {
            "gate": lin(hidden, features, cfg.bias),
            "up": lin(hidden, features, cfg.bias),
            "down": lin(features, hidden, cfg.bias),
        }

    def attention(dim, heads, kvheads):
        hd = dim // heads
        return {
            "wq": lin(hd * heads, dim, cfg.bias),
            "wk": lin(hd * kvheads, dim, cfg.bias),
            "wv": lin(hd * kvheads, dim, cfg.bias),
            "gate": lin(dim, dim, cfg.bias),
            "wo": lin(dim, dim, cfg.bias),
            "qknorm": {"qnorm": norm(hd), "knorm": norm(hd)},
        }

    def stack(n, tree):
        return jax.tree.map(
            lambda s: jax.ShapeDtypeStruct((n,) + tuple(s.shape), s.dtype), tree
        )

    F, T = cfg.features, cfg.txtdim
    block = {
        "mod": {"lin": leaf(6 * F)},
        "prenorm": norm(F),
        "postnorm": norm(F),
        "attn": attention(F, cfg.heads, cfg.kvheads),
        "mlp": swiglu(F),
    }
    txt_block = {
        "prenorm": norm(T),
        "postnorm": norm(T),
        "attn": attention(T, cfg.txtheads, cfg.txtkvheads),
        "mlp": swiglu(T),
    }
    return {
        "first": lin(F, cfg.channels * cfg.patch ** 2, True),
        "blocks": stack(cfg.layers, block),
        "tmlp": {"0": lin(F, cfg.tdim, True), "2": lin(F, F, True)},
        "txtfusion": {
            "layerwise_blocks": stack(2, txt_block),
            "projector": lin(1, cfg.txtlayers, False),
            "refiner_blocks": stack(2, txt_block),
        },
        "txtmlp": {"0": norm(T), "1": lin(F, T, True), "3": lin(F, F, True)},
        "last": {
            "norm": norm(F),
            "linear": lin(cfg.patch * cfg.patch * cfg.channels, F, True),
            "modulation": {"lin": leaf(2, F)},
        },
        "tproj": {"1": lin(6 * F, F, True)},
    }


def _place(arr, mesh, stacked, sharding="fsdp"):
    if mesh is None:
        return arr
    if sharding == "replicate":
        spec = P()
    else:
        spec = param_partition_spec(arr.shape, mesh.size, stacked)
    return jax.device_put(arr, NamedSharding(mesh, spec))


def _is_stacked_path(keys):
    """True when a params-tree path lies under one of STACKED_PREFIXES."""
    dotted = ".".join(keys)
    return any(dotted.startswith(p + ".") for p in STACKED_PREFIXES)


def _path_keys(path):
    return [str(getattr(k, "key", k)) for k in path]


def param_shardings(params, mesh, sharding="fsdp"):
    """NamedSharding pytree matching how `load_krea2_params` placed each leaf.

    Mirrors `_place`: sharding="fsdp" uses `param_partition_spec` (stacked
    subtrees skip the leading block axis), "replicate" uses P(). Works on
    real arrays or `jax.ShapeDtypeStruct` leaves (only `.shape` is read).
    """
    if sharding not in ("fsdp", "replicate"):
        raise ValueError(f"unknown base sharding: {sharding!r}")

    def one(path, leaf):
        if sharding == "replicate":
            spec = P()
        else:
            spec = param_partition_spec(
                leaf.shape, mesh.size, _is_stacked_path(_path_keys(path))
            )
        return NamedSharding(mesh, spec)

    return jax.tree_util.tree_map_with_path(one, params)


_ST_DTYPES = {
    np.dtype(jnp.bfloat16): "BF16",
    np.dtype(np.float16): "F16",
    np.dtype(np.float32): "F32",
    np.dtype(np.float64): "F64",
}


def _official_entries(params):
    """Unstack a params tree into official safetensors entries.

    Returns (entries, leaves): entries is a list of (official key, leaf
    index, layer index or None, per-tensor shape) sorted by key, leaves the
    flattened leaves. Stacked subtrees (STACKED_PREFIXES) become per-layer
    keys ('blocks.{i}.attn.wq.weight', ...); everything else is dotted
    as-is. Only `.shape` of each leaf is read.
    """
    leaves_with_path = jax.tree_util.tree_flatten_with_path(params)[0]
    entries = []
    leaves = []
    for li, (kp, leaf) in enumerate(leaves_with_path):
        keys = _path_keys(kp)
        dotted = ".".join(keys)
        shape = tuple(int(d) for d in leaf.shape)
        leaves.append(leaf)
        prefix = next(
            (p for p in STACKED_PREFIXES if dotted.startswith(p + ".")), None
        )
        if prefix is None:
            entries.append((dotted, li, None, shape))
        else:
            if len(shape) < 1:
                raise ValueError(f"{dotted}: stacked leaf has no block axis")
            rest = dotted[len(prefix) + 1:]
            for i in range(shape[0]):
                entries.append((f"{prefix}.{i}.{rest}", li, i, shape[1:]))
    names = [e[0] for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("duplicate safetensors keys while unstacking params")
    entries.sort(key=lambda e: e[0])
    return entries, leaves


def official_key_shapes(params):
    """{official safetensors key: shape tuple} that `save_krea2_params`
    would write for `params` (same unstacking; nothing is fetched, so it
    works on sharded arrays or ShapeDtypeStruct leaves)."""
    entries, _ = _official_entries(params)
    return {name: shape for name, _, _, shape in entries}


def save_krea2_params(path, params, dtype=jnp.bfloat16, metadata=None):
    """Write a params tree as an official-layout Krea 2 safetensors file.

    Inverse of `load_krea2_params`: stacked subtrees (STACKED_PREFIXES) are
    unstacked back to per-layer keys ('blocks.{i}.attn.wq.weight', ...),
    everything else is dotted as-is, and every tensor is cast to `dtype`.
    The file is streamed leaf by leaf (one host fetch per leaf, written
    straight to disk), so the host never holds more than one leaf; works for
    sharded or unsharded trees. Leaves are fetched from a temporary cast copy,
    so no host copy stays cached on the live arrays afterwards. Tensors are
    written in sorted key order with a deterministic header, so the same tree
    always gives identical bytes. The write is atomic: data goes to a sibling
    temp file that is fsynced and then renamed onto `path`; on any error the
    temp file is removed and an existing `path` is left untouched.
    """
    import json
    import os
    import struct
    import sys

    if sys.byteorder != "little":  # safetensors data is little-endian
        raise NotImplementedError("save_krea2_params needs a little-endian host")
    out_dtype = np.dtype(dtype)
    if out_dtype not in _ST_DTYPES:
        raise ValueError(f"unsupported safetensors dtype: {out_dtype}")
    itemsize = out_dtype.itemsize

    # (official key, leaf index, layer index or None, per-tensor shape);
    # shared with official_key_shapes so the two cannot drift
    entries, leaves = _official_entries(params)

    header = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    for name, _, _, shape in entries:
        nbytes = int(np.prod(shape, dtype=np.int64)) * itemsize
        header[name] = {
            "dtype": _ST_DTYPES[out_dtype],
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    hbytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    hbytes += b" " * (-len(hbytes) % 8)  # 8-byte align the data section

    # group per-layer writes by leaf so each leaf is fetched once, but keep
    # the on-disk order sorted by key (offsets above)
    by_leaf = {}
    for name, li, idx, _ in entries:
        by_leaf.setdefault(li, []).append((header[name]["data_offsets"][0], idx))
    data_start = 8 + len(hbytes)
    path = os.fspath(path)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", len(hbytes)))
            f.write(hbytes)
            f.truncate(data_start + offset)
            for li in sorted(by_leaf):
                leaf = leaves[li]
                if isinstance(leaf, jax.Array):
                    # Fetch from a short-lived cast copy: np.asarray caches
                    # the gathered host value on the array it reads, so
                    # reading the live leaf would pin the whole tree in host
                    # RAM. astype is a no-op (same object) for a matching
                    # dtype, so force a real copy then.
                    tmp = leaf.astype(out_dtype)
                    if tmp is leaf:
                        tmp = jnp.copy(leaf)
                    arr = np.asarray(tmp)
                else:
                    tmp = None
                    arr = np.asarray(leaf).astype(out_dtype, copy=False)
                for off, idx in by_leaf[li]:
                    part = arr if idx is None else arr[idx]
                    f.seek(data_start + off)
                    f.write(np.ascontiguousarray(part).tobytes())
                    del part
                del leaf, tmp, arr
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

ATTN_TARGETS = ("wq", "wk", "wv", "wo")
ATTN_GATE_TARGETS = ("gate",)
MLP_TARGETS = ("gate", "up", "down")


# Every Linear in the DiT that lives outside the 28 scanned blocks and the
# text-fusion blocks (path into the params tree, in application order).
EXTRA_LINEAR_PATHS = (
    ("txtfusion", "projector"),
    ("first",),
    ("tmlp", "0"),
    ("tmlp", "2"),
    ("tproj", "1"),
    ("txtmlp", "1"),
    ("txtmlp", "3"),
    ("last", "linear"),
)


LORA_TARGET_MODES = ("attn", "attn+mlp", "all", "full")


def _target_lists(targets):
    if targets not in LORA_TARGET_MODES:
        raise ValueError(f"unknown LoRA targets: {targets!r}")
    attn_t = list(ATTN_TARGETS)
    mlp_t = []
    if targets in ("attn+mlp", "all", "full"):
        mlp_t = list(MLP_TARGETS)
    if targets in ("all", "full"):
        attn_t += list(ATTN_GATE_TARGETS)
    return attn_t, mlp_t


def expected_lora_modules(targets):
    """Dotted module paths (layer index collapsed) that `init_lora` adapts."""
    attn_t, mlp_t = _target_lists(targets)
    mods = {f"blocks.attn.{t}" for t in attn_t}
    mods |= {f"blocks.mlp.{t}" for t in mlp_t}
    if targets == "full":
        for grp in ("layerwise_blocks", "refiner_blocks"):
            mods |= {f"txtfusion.{grp}.attn.{t}" for t in attn_t}
            mods |= {f"txtfusion.{grp}.mlp.{t}" for t in mlp_t}
        mods |= {".".join(p) for p in EXTRA_LINEAR_PATHS}
    return mods


def init_lora(params, rng, rank=16, targets="attn", dtype=jnp.float32):
    """LoRA tree mirroring the params nesting.

    targets: "attn" (wq/wk/wv/wo), "attn+mlp" (adds the SwiGLU MLP), "all"
    (adds the attention sigmoid-gate projection), or "full" (every Linear in
    the DiT — blocks plus the text-fusion transformer, the image/timestep/
    text projections, and the final output layer; matches the model authors'
    recommended coverage). lora_A ~ U(±1/√fan_in) (kaiming_uniform with
    a=√5, matching the reference PEFT/kohya init), lora_B = 0, so the model
    starts exactly at the base checkpoint.
    """
    attn_t, mlp_t = _target_lists(targets)

    # Ordered (path, weight, stacked) entries; stacked weights carry a
    # leading block axis. Order fixes RNG key assignment, so the pre-"full"
    # prefix must stay stable for seed reproducibility.
    entries = []
    blocks = params["blocks"]
    for t in attn_t:
        entries.append((("blocks", "attn", t), blocks["attn"][t]["weight"], True))
    for t in mlp_t:
        entries.append((("blocks", "mlp", t), blocks["mlp"][t]["weight"], True))
    if targets == "full":
        tf = params["txtfusion"]
        for grp in ("layerwise_blocks", "refiner_blocks"):
            for t in attn_t:
                entries.append(
                    (("txtfusion", grp, "attn", t), tf[grp]["attn"][t]["weight"], True)
                )
            for t in mlp_t:
                entries.append(
                    (("txtfusion", grp, "mlp", t), tf[grp]["mlp"][t]["weight"], True)
                )
        for path in EXTRA_LINEAR_PATHS:
            node = params
            for k in path:
                node = node[k]
            entries.append((path, node["weight"], False))

    lora = {}
    keys = jax.random.split(rng, len(entries))
    for (path, w, stacked), key in zip(entries, keys):
        if stacked:
            n, out_f, in_f = w.shape
            bound = 1.0 / math.sqrt(in_f)
            a = jax.random.uniform(
                key, (n, rank, in_f), dtype=dtype, minval=-bound, maxval=bound
            )
            b = jnp.zeros((n, out_f, rank), dtype=dtype)
        else:
            out_f, in_f = w.shape
            bound = 1.0 / math.sqrt(in_f)
            a = jax.random.uniform(
                key, (rank, in_f), dtype=dtype, minval=-bound, maxval=bound
            )
            b = jnp.zeros((out_f, rank), dtype=dtype)
        _nest(lora, list(path), {"lora_A": a, "lora_B": b})
    return lora


def replicate_tree(tree, mesh):
    sharding = replicated_sharding(mesh)
    return jax.tree.map(lambda a: jax.device_put(a, sharding), tree)


def _lora_leaves(tree, path=()):
    """Yield (path_tuple, {"lora_A": ..., "lora_B": ...}) for every adapter."""
    if isinstance(tree, dict) and "lora_A" in tree:
        yield path, tree
        return
    for k in sorted(tree):
        yield from _lora_leaves(tree[k], path + (k,))


def save_lora(path, lora, rank, alpha, extra_meta=None):
    """Write LoRA weights as safetensors with per-layer unstacked keys.

    Key format: {module_path}.lora_A.weight (r, in) /
                {module_path}.lora_B.weight (out, r) — plain module paths
    into the official state_dict (e.g. blocks.3.attn.wq, first,
    txtfusion.refiner_blocks.1.mlp.up), PEFT-style A/B naming. Stacked
    subtrees (blocks, text-fusion blocks) are unstacked to per-layer keys.
    """
    from safetensors.numpy import save_file

    flat = {}
    for mpath, ab in _lora_leaves(lora):
        dotted = ".".join(mpath)
        stacked = next(
            (p for p in STACKED_PREFIXES if dotted.startswith(p + ".")), None
        )
        a = np.asarray(jax.device_get(ab["lora_A"])).astype(np.float32)
        b = np.asarray(jax.device_get(ab["lora_B"])).astype(np.float32)
        want_ndim = 3 if stacked else 2
        if a.ndim != want_ndim or b.ndim != want_ndim:
            raise ValueError(
                f"{dotted}: expected {want_ndim}-d lora_A/lora_B, got "
                f"{a.shape} / {b.shape}"
            )
        if stacked and a.shape[0] != b.shape[0]:
            raise ValueError(
                f"{dotted}: lora_A has {a.shape[0]} layers but lora_B has "
                f"{b.shape[0]}"
            )
        if a.shape[-2] != b.shape[-1]:
            raise ValueError(
                f"{dotted}: lora_A rank {a.shape[-2]} != lora_B rank "
                f"{b.shape[-1]}"
            )
        if stacked:
            rest = dotted[len(stacked) + 1:]
            for i in range(a.shape[0]):
                flat[f"{stacked}.{i}.{rest}.lora_A.weight"] = a[i]
                flat[f"{stacked}.{i}.{rest}.lora_B.weight"] = b[i]
        else:
            flat[f"{dotted}.lora_A.weight"] = a
            flat[f"{dotted}.lora_B.weight"] = b
    meta = {"rank": str(rank), "alpha": str(alpha), "format": "krea2-jax-lora"}
    if extra_meta:
        meta.update({k: str(v) for k, v in extra_meta.items()})
    save_file(flat, path, metadata=meta)


def load_lora(path, mesh=None):
    """Load a LoRA safetensors written by `save_lora` back into a stacked tree.

    Validates the file before returning: every module path must belong to the
    known adapter schema, every adapter needs a matching (lora_A, lora_B) pair
    with agreeing rank and layer count, stacked layer counts must be uniform
    per stacked prefix, and when the metadata records rank/targets the tensors
    must be consistent with them. Returns (lora_tree, metadata).
    """
    valid = expected_lora_modules("full")
    per_module = {}  # module_path -> {"lora_A"/"lora_B": {idx: arr} or arr}
    with safe_open(path, framework="numpy") as f:
        meta = f.metadata() or {}
        for key in f.keys():
            m = re.match(r"^(.+)\.(lora_A|lora_B)\.weight$", key)
            if not m:
                raise ValueError(f"{path}: unrecognized LoRA key: {key}")
            module, ab = m.group(1), m.group(2)
            sm = _STACK_RE.match(module)
            if sm:
                mod = f"{sm.group(1)}.{sm.group(3)}"
                per_module.setdefault(mod, {}).setdefault(ab, {})[
                    int(sm.group(2))
                ] = f.get_tensor(key)
            else:
                per_module.setdefault(module, {})[ab] = f.get_tensor(key)

    for module in per_module:
        if module not in valid:
            raise ValueError(
                f"{path}: unknown LoRA module {module!r} (no such Linear in "
                "the DiT — typo or incompatible export?)"
            )
    tmeta = meta.get("targets")
    if tmeta is not None:
        if tmeta not in LORA_TARGET_MODES:
            raise ValueError(f"{path}: unknown targets metadata {tmeta!r}")
        want = expected_lora_modules(tmeta)
        if set(per_module) != want:
            missing = sorted(want - set(per_module))
            extra = sorted(set(per_module) - want)
            raise ValueError(
                f"{path}: modules do not match targets={tmeta!r} "
                f"(missing {missing or 'none'}, unexpected {extra or 'none'})"
            )

    meta_rank = int(meta["rank"]) if "rank" in meta else None
    lora = {}
    stack_counts = {}  # stacked prefix -> (module, n)
    for module, abs_ in per_module.items():
        if set(abs_) != {"lora_A", "lora_B"}:
            missing = {"lora_A", "lora_B"} - set(abs_)
            raise ValueError(f"{path}: {module} is missing {sorted(missing)}")
        stacked = isinstance(abs_["lora_A"], dict)
        pair = {}
        for ab, val in abs_.items():
            if isinstance(val, dict) != stacked:
                raise ValueError(
                    f"{path}: {module} mixes stacked and unstacked lora_A/lora_B"
                )
            if stacked:
                n = max(val) + 1
                if len(val) != n:
                    raise ValueError(
                        f"{path}: missing layer index under {module}.{ab}"
                    )
                pair[ab] = np.stack([val[i] for i in range(n)], axis=0)
            else:
                pair[ab] = val
        a, b = pair["lora_A"], pair["lora_B"]
        want_ndim = 3 if stacked else 2
        if a.ndim != want_ndim or b.ndim != want_ndim:
            raise ValueError(
                f"{path}: {module} has lora_A/lora_B shapes {a.shape} / "
                f"{b.shape}, expected {want_ndim}-d"
            )
        if stacked:
            if a.shape[0] != b.shape[0]:
                raise ValueError(
                    f"{path}: {module} has {a.shape[0]} lora_A layers but "
                    f"{b.shape[0]} lora_B layers"
                )
            prefix = next(
                p for p in STACKED_PREFIXES if module.startswith(p + ".")
            )
            seen = stack_counts.get(prefix)
            if seen is not None and seen[1] != a.shape[0]:
                raise ValueError(
                    f"{path}: {module} has {a.shape[0]} layers but "
                    f"{seen[0]} has {seen[1]}"
                )
            stack_counts[prefix] = (module, a.shape[0])
        if a.shape[-2] != b.shape[-1]:
            raise ValueError(
                f"{path}: {module} lora_A rank {a.shape[-2]} != lora_B rank "
                f"{b.shape[-1]}"
            )
        if meta_rank is not None and a.shape[-2] != meta_rank:
            raise ValueError(
                f"{path}: {module} has rank {a.shape[-2]} but metadata says "
                f"rank {meta_rank}"
            )
        for ab, arr in pair.items():
            _nest(lora, module.split(".") + [ab], jnp.asarray(arr))
    if mesh is not None:
        lora = replicate_tree(lora, mesh)
    return lora, meta


def lora_scale_from_meta(meta, default_alpha=None, default_rank=None):
    alpha = float(meta.get("alpha", default_alpha))
    rank = float(meta.get("rank", default_rank))
    return alpha / rank


# ---------------------------------------------------------------------------
# optimizer / train-state checkpointing
# ---------------------------------------------------------------------------


def save_train_state(path, opt_state, step, extra_meta=None):
    """Write the optimizer state (any optax pytree) plus the step counter.

    Leaves are stored in `jax.tree.leaves` order; the tree structure itself is
    not serialized — `load_train_state` rebuilds it from a freshly
    `tx.init`-ed template, so save and load must use the same optimizer
    configuration.
    """
    from safetensors.numpy import save_file

    leaves = jax.tree.leaves(opt_state)
    flat = {
        f"leaf.{i:05d}": np.asarray(jax.device_get(a)) for i, a in enumerate(leaves)
    }
    meta = {
        "step": str(step),
        "n_leaves": str(len(leaves)),
        "format": "krea2-jax-train-state",
    }
    if extra_meta:
        meta.update({k: str(v) for k, v in extra_meta.items()})
    save_file(flat, path, metadata=meta)


def load_train_state(path, opt_state_template, mesh=None):
    """Restore an opt_state saved by `save_train_state`.

    opt_state_template: a freshly `tx.init`-ed state with the same optimizer,
    LoRA rank, and targets as the saved run; it supplies the tree structure
    and expected leaf shapes/dtypes. Returns (opt_state, step).
    """
    template_leaves, treedef = jax.tree.flatten(opt_state_template)
    with safe_open(path, framework="numpy") as f:
        meta = f.metadata() or {}
        n = int(meta["n_leaves"])
        if n != len(template_leaves):
            raise ValueError(
                f"{path}: {n} optimizer-state leaves, expected "
                f"{len(template_leaves)} (different optimizer/--rank/--targets?)"
            )
        leaves = [f.get_tensor(f"leaf.{i:05d}") for i in range(n)]
    for i, (got, want) in enumerate(zip(leaves, template_leaves)):
        if got.shape != tuple(want.shape):
            raise ValueError(
                f"{path}: leaf {i} has shape {got.shape}, expected "
                f"{tuple(want.shape)} (different --rank/--targets?)"
            )
    leaves = [
        jnp.asarray(got, dtype=want.dtype)
        for got, want in zip(leaves, template_leaves)
    ]
    opt_state = jax.tree.unflatten(treedef, leaves)
    if mesh is not None:
        opt_state = replicate_tree(opt_state, mesh)
    return opt_state, int(meta["step"])
