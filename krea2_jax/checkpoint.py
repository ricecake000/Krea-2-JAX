"""Checkpoint I/O: official safetensors -> sharded JAX pytree, LoRA save/load.

The official `raw.safetensors` / `turbo.safetensors` files store the PyTorch
state_dict of `SingleStreamDiT` in bfloat16. We keep torch's (out, in) Linear
layout and the exact key names, but nest keys by "." and stack repeated blocks
(`blocks.{i}.*`, `txtfusion.layerwise_blocks.{i}.*`,
`txtfusion.refiner_blocks.{i}.*`) along a leading axis for `lax.scan`.
"""

import json
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


def load_krea2_params(path, mesh=None, dtype=None):
    """Load the official DiT safetensors into a (sharded) nested pytree.

    Tensors are read on host CPU, repeated blocks are stacked, and each leaf
    is placed on the mesh with an FSDP-style sharding. With mesh=None the
    tree stays on the default device, replicated.
    """
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
        _nest(params, key.split("."), _place(tensor, mesh, stacked=False))
    for prefix, fields in stacked.items():
        for rest, by_idx in fields.items():
            n = max(by_idx) + 1
            assert len(by_idx) == n, f"missing block index under {prefix}.{rest}"
            with jax.default_device(cpu):
                arr = jnp.stack([by_idx[i] for i in range(n)], axis=0)
            _nest(
                params,
                prefix.split(".") + rest.split("."),
                _place(arr, mesh, stacked=True),
            )
    return params


def _place(arr, mesh, stacked):
    if mesh is None:
        return arr
    spec = param_partition_spec(arr.shape, mesh.size, stacked)
    return jax.device_put(arr, NamedSharding(mesh, spec))


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

ATTN_TARGETS = ("wq", "wk", "wv", "wo")
ATTN_GATE_TARGETS = ("gate",)
MLP_TARGETS = ("gate", "up", "down")


def init_lora(params, rng, rank=16, targets="attn", dtype=jnp.float32):
    """LoRA tree for the stacked `blocks` subtree.

    targets: "attn" (wq/wk/wv/wo), "attn+mlp", or "all" (adds the attention
    sigmoid-gate projection). lora_A ~ N(0, 1/rank), lora_B = 0, so the model
    starts exactly at the base checkpoint.
    """
    attn_t = list(ATTN_TARGETS)
    mlp_t = []
    if targets in ("attn+mlp", "all"):
        mlp_t = list(MLP_TARGETS)
    if targets == "all":
        attn_t += list(ATTN_GATE_TARGETS)

    blocks = params["blocks"]
    n_layers = blocks["attn"]["wq"]["weight"].shape[0]
    lora = {"attn": {}, "mlp": {}}
    keys = jax.random.split(rng, len(attn_t) + len(mlp_t))
    ki = 0

    def make(weight_leaf, key):
        n, out_f, in_f = weight_leaf.shape
        a = jax.random.normal(key, (n, rank, in_f), dtype=dtype) / rank
        b = jnp.zeros((n, out_f, rank), dtype=dtype)
        return {"lora_A": a, "lora_B": b}

    for t in attn_t:
        lora["attn"][t] = make(blocks["attn"][t]["weight"], keys[ki]); ki += 1
    for t in mlp_t:
        lora["mlp"][t] = make(blocks["mlp"][t]["weight"], keys[ki]); ki += 1
    if not lora["mlp"]:
        del lora["mlp"]
    assert n_layers == next(iter(lora["attn"].values()))["lora_A"].shape[0]
    return {"blocks": lora}


def replicate_tree(tree, mesh):
    sharding = replicated_sharding(mesh)
    return jax.tree.map(lambda a: jax.device_put(a, sharding), tree)


def save_lora(path, lora, rank, alpha, extra_meta=None):
    """Write LoRA weights as safetensors with per-layer unstacked keys.

    Key format: blocks.{i}.attn.wq.lora_A.weight (r, in) /
                blocks.{i}.attn.wq.lora_B.weight (out, r) — plain module paths
    into the official state_dict, PEFT-style A/B naming.
    """
    from safetensors.numpy import save_file

    flat = {}
    for group, targets in lora["blocks"].items():
        for name, ab in targets.items():
            n = ab["lora_A"].shape[0]
            a = np.asarray(jax.device_get(ab["lora_A"])).astype(np.float32)
            b = np.asarray(jax.device_get(ab["lora_B"])).astype(np.float32)
            for i in range(n):
                flat[f"blocks.{i}.{group}.{name}.lora_A.weight"] = a[i]
                flat[f"blocks.{i}.{group}.{name}.lora_B.weight"] = b[i]
    meta = {"rank": str(rank), "alpha": str(alpha), "format": "krea2-jax-lora"}
    if extra_meta:
        meta.update({k: str(v) for k, v in extra_meta.items()})
    save_file(flat, path, metadata=meta)


def load_lora(path, mesh=None):
    """Load a LoRA safetensors written by `save_lora` back into a stacked tree.

    Returns (lora_tree, metadata).
    """
    per_layer = {}
    with safe_open(path, framework="numpy") as f:
        meta = f.metadata() or {}
        for key in f.keys():
            m = re.match(
                r"^blocks\.(\d+)\.(attn|mlp)\.(\w+)\.(lora_A|lora_B)\.weight$", key
            )
            if not m:
                raise ValueError(f"unrecognized LoRA key: {key}")
            idx, group, name, ab = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            per_layer.setdefault((group, name, ab), {})[idx] = f.get_tensor(key)

    lora = {}
    for (group, name, ab), by_idx in per_layer.items():
        n = max(by_idx) + 1
        arr = jnp.asarray(np.stack([by_idx[i] for i in range(n)], axis=0))
        lora.setdefault(group, {}).setdefault(name, {})[ab] = arr
    lora = {"blocks": lora}
    if mesh is not None:
        lora = replicate_tree(lora, mesh)
    return lora, meta


def lora_scale_from_meta(meta, default_alpha=None, default_rank=None):
    alpha = float(meta.get("alpha", default_alpha))
    rank = float(meta.get("rank", default_rank))
    return alpha / rank


def save_train_state_meta(path, **kwargs):
    with open(path, "w") as f:
        json.dump(kwargs, f, indent=2)
