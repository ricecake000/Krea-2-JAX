"""dit_forward(layer_loop="unroll") with per-layer list blocks, on CPU with 8
emulated devices (FSDP-sharded params, activation sharding on):

  a. unroll (list blocks) == scan (stacked blocks): forward outputs and the
     parameter gradients (compared after stack_blocks), remat "dots",
     "full" and "none"; also with all three repeated-block prefixes
     unstacked, with a (stacked or list) LoRA tree, and with a LoRA tree
     whose "blocks" entry is None
  b. unroll over STACKED blocks (static per-layer index) == the list form
  c. checkpoint round trip: load_krea2_params(unstack=("blocks",)) has the
     structure of abstract_krea2_params(unstack=...), stack_blocks of it
     equals the stacked load; save_krea2_params of the list tree writes the
     same keys, shapes and bytes as the stacked tree; official_key_shapes
     agree; unstack_blocks / stack_blocks round trip (arrays, numpy and
     ShapeDtypeStruct leaves, nested prefixes, idempotent, bad input)
  d. param_shardings of the unstacked tree == the stacked spec minus the
     layer axis for every leaf (tiny tree + full-size 1-D/2-D/3-D shapes)
  e. HLO (the acceptance criterion): the compiled forward+backward of the
     unroll + list configuration contains zero dynamic-slice /
     dynamic-update-slice / while instructions from model.py; the scan
     configuration contains some (the SPMD partitioner's batch-shard slices
     at sharding constraints outside the loop are counted apart and must
     match between the two)
  f. error paths: scan with list blocks -> TypeError, unknown layer_loop ->
     ValueError, init_lora / save_lora on list blocks -> ValueError

    python tests/test_layer_loop.py
"""

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import re
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from krea2_jax.checkpoint import (
    STACKED_PREFIXES, abstract_krea2_params, batch_sharding, init_lora,
    load_krea2_params, make_mesh, official_key_shapes, param_partition_spec,
    param_shardings, save_krea2_params, save_lora, stack_blocks,
    unstack_blocks,
)
from krea2_jax.config import Krea2Config
from krea2_jax.model import build_positions_and_mask, dit_forward, patchify

assert len(jax.devices()) == 8, jax.devices()

TINY = dict(features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
            layers=3, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3)
cfg = Krea2Config(**TINY)
mesh = make_mesh()
act = batch_sharding(mesh)
UNSTACK = ("blocks",)
ALL_PREFIXES = STACKED_PREFIXES


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def tree_rel_err(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    return max(rel_err(x, y) for x, y in zip(la, lb))


def normspec(spec, ndim):
    s = tuple(spec)
    return s + (None,) * (ndim - len(s))


# ---------------------------------------------------------------------------
# tiny random official checkpoint (written by save_krea2_params)
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp(prefix="test_layer_loop_"))
abstract = abstract_krea2_params(cfg, dtype=jnp.float32)
rng = np.random.default_rng(0)
host_stacked = jax.tree.map(
    lambda s: (0.05 * rng.standard_normal(s.shape)).astype(np.float32), abstract)
ckpt = tmp / "base.safetensors"
save_krea2_params(str(ckpt), host_stacked, dtype=jnp.float32)

params_st = load_krea2_params(str(ckpt), mesh=mesh, dtype=jnp.float32)
params_ls = load_krea2_params(str(ckpt), mesh=mesh, dtype=jnp.float32,
                              unstack=UNSTACK)

# ---------------------------------------------------------------------------
# c. checkpoint round trip
# ---------------------------------------------------------------------------
abs_ls = abstract_krea2_params(cfg, dtype=jnp.float32, unstack=UNSTACK)
assert isinstance(params_ls["blocks"], list) and len(params_ls["blocks"]) == 3
assert jax.tree.structure(params_ls) == jax.tree.structure(abs_ls)
for a, s in zip(jax.tree.leaves(params_ls), jax.tree.leaves(abs_ls)):
    assert tuple(a.shape) == tuple(s.shape) and a.dtype == s.dtype
assert jax.tree.structure(params_st) == jax.tree.structure(abstract)
restacked = stack_blocks(params_ls)
assert jax.tree.structure(restacked) == jax.tree.structure(params_st)
for a, b in zip(jax.tree.leaves(restacked), jax.tree.leaves(params_st)):
    assert np.array_equal(np.asarray(a), np.asarray(b))
# every per-layer leaf was placed on its own with the unstacked spec
for kp, a in jax.tree_util.tree_flatten_with_path(params_ls["blocks"])[0]:
    want = param_partition_spec(a.shape, mesh.size, stacked=False)
    assert normspec(a.sharding.spec, a.ndim) == normspec(want, a.ndim), kp

# all three prefixes unstacked
params_all = load_krea2_params(str(ckpt), mesh=mesh, dtype=jnp.float32,
                               unstack=ALL_PREFIXES)
abs_all = abstract_krea2_params(cfg, dtype=jnp.float32, unstack=ALL_PREFIXES)
assert jax.tree.structure(params_all) == jax.tree.structure(abs_all)
assert isinstance(params_all["txtfusion"]["refiner_blocks"], list)
assert "projector" in params_all["txtfusion"]
for a, b in zip(jax.tree.leaves(stack_blocks(params_all, ALL_PREFIXES)),
                jax.tree.leaves(params_st)):
    assert np.array_equal(np.asarray(a), np.asarray(b))

# save: list tree == stacked tree, key for key and byte for byte
out_st, out_ls, out_all = (tmp / f"{n}.safetensors" for n in ("st", "ls", "all"))
save_krea2_params(str(out_st), params_st, metadata={"x": "1"})
save_krea2_params(str(out_ls), params_ls, metadata={"x": "1"})
save_krea2_params(str(out_all), params_all, metadata={"x": "1"})
assert out_st.read_bytes() == out_ls.read_bytes() == out_all.read_bytes()
ks_st = official_key_shapes(params_st)
assert ks_st == official_key_shapes(params_ls) == official_key_shapes(abs_ls)
assert ks_st == official_key_shapes(params_all) == official_key_shapes(abs_all)
assert "blocks.2.attn.wq.weight" in ks_st and ks_st["blocks.2.attn.wq.weight"] == (128, 128)
# duplicate-key check still applies to a mixed tree (list + stray dotted key)
bad = dict(params_ls)
bad["blocks.0"] = {"attn": {"wq": {"weight": params_ls["blocks"][0]["attn"]["wq"]["weight"]}}}
try:
    official_key_shapes(bad)
except ValueError as e:
    assert "duplicate" in str(e), e
else:
    raise AssertionError("duplicate official keys not detected")

# converters: ShapeDtypeStruct leaves, numpy leaves, nested prefixes
u = unstack_blocks(abstract)
assert jax.tree.structure(u) == jax.tree.structure(abs_ls)
assert all(tuple(x.shape) == tuple(y.shape) and x.dtype == y.dtype
           for x, y in zip(jax.tree.leaves(u), jax.tree.leaves(abs_ls)))
assert unstack_blocks(u) is u or jax.tree.structure(unstack_blocks(u)) == jax.tree.structure(u)
s = stack_blocks(abs_ls)
assert jax.tree.structure(s) == jax.tree.structure(abstract)
assert all(tuple(x.shape) == tuple(y.shape) and x.sharding is None
           for x, y in zip(jax.tree.leaves(s), jax.tree.leaves(abstract)))
assert jax.tree.structure(stack_blocks(abstract)) == jax.tree.structure(abstract)
u_all = unstack_blocks(abstract, ALL_PREFIXES)
assert jax.tree.structure(u_all) == jax.tree.structure(abs_all)
u_np = unstack_blocks(host_stacked, ("txtfusion.layerwise_blocks",))
assert isinstance(u_np["txtfusion"]["layerwise_blocks"], list)
assert not isinstance(u_np["blocks"], list)
assert isinstance(u_np["txtfusion"]["layerwise_blocks"][1]["attn"]["wq"]["weight"], np.ndarray)
back = stack_blocks(u_np, "txtfusion.layerwise_blocks")
assert isinstance(back["txtfusion"]["layerwise_blocks"]["attn"]["wq"]["weight"], np.ndarray)
for a, b in zip(jax.tree.leaves(back), jax.tree.leaves(host_stacked)):
    assert np.array_equal(a, b)
assert not isinstance(host_stacked["txtfusion"]["layerwise_blocks"], list)  # no mutation
for fn, arg, what in (
    (unstack_blocks, ("nope",), "unknown prefix"),
    (stack_blocks, ("tmlp",), "non-block prefix"),
):
    try:
        fn(abstract, arg)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{what} accepted")
mixed = dict(abstract)
mixed["blocks"] = dict(abstract["blocks"])
mixed["blocks"]["prenorm"] = {"scale": jax.ShapeDtypeStruct((2, 128), jnp.float32)}
try:
    unstack_blocks(mixed)
except ValueError as e:
    assert "layer count" in str(e), e
else:
    raise AssertionError("mixed layer counts accepted")
ragged = dict(abs_ls)
ragged["blocks"] = list(abs_ls["blocks"])
ragged["blocks"][1] = {"mod": abs_ls["blocks"][1]["mod"]}
try:
    stack_blocks(ragged)
except ValueError as e:
    assert "structure" in str(e), e
else:
    raise AssertionError("per-layer trees of different structure stacked")
print("c. checkpoint round trip: unstacked load == abstract structure, "
      "stack_blocks == stacked load, identical export bytes/keys, "
      "converters: ok")

# ---------------------------------------------------------------------------
# d. param_shardings: unstacked spec == stacked spec minus axis 0
# ---------------------------------------------------------------------------
def check_specs(stacked_tree, list_tree, prefixes):
    sh_st = param_shardings(stacked_tree, mesh)
    sh_ls = param_shardings(list_tree, mesh)
    n_checked = n_split = 0
    for prefix in prefixes:
        node_st, node_ls = sh_st, sh_ls
        leaf_st, leaf_ls = stacked_tree, list_tree
        for k in prefix.split("."):
            node_st, node_ls = node_st[k], node_ls[k]
            leaf_st, leaf_ls = leaf_st[k], leaf_ls[k]
        flat_st = jax.tree_util.tree_flatten_with_path(node_st)[0]
        shapes_st = jax.tree.leaves(leaf_st)
        for i, (layer_sh, layer_leaf) in enumerate(zip(node_ls, leaf_ls)):
            flat_ls = jax.tree_util.tree_flatten_with_path(layer_sh)[0]
            shapes_ls = jax.tree.leaves(layer_leaf)
            assert len(flat_ls) == len(flat_st)
            for (kp_s, s_st), (kp_l, s_ls), x_st, x_ls in zip(
                    flat_st, flat_ls, shapes_st, shapes_ls):
                assert kp_s == kp_l
                assert tuple(x_st.shape[1:]) == tuple(x_ls.shape)
                want = normspec(s_st.spec, len(x_st.shape))[1:]
                got = normspec(s_ls.spec, len(x_ls.shape))
                assert got == want, (prefix, i, kp_l, x_ls.shape, got, want)
                n_checked += 1
                n_split += "fsdp" in got
    # leaves outside the prefixes are untouched
    rest_st = {k: v for k, v in sh_st.items() if k != "blocks"}
    rest_ls = {k: v for k, v in sh_ls.items() if k != "blocks"}
    if prefixes == UNSTACK:
        assert jax.tree.leaves(rest_st) == jax.tree.leaves(rest_ls)
    return n_checked, n_split


n1, s1 = check_specs(abstract, abs_ls, UNSTACK)
n2, s2 = check_specs(abstract, abs_all, ALL_PREFIXES)
assert s1 > 0 and s2 > s1
# full-size shapes (12.9B config) and odd ones: 1-D, 2-D, 3-D
F = 6144
shapes = [(6 * F,), (F,), (128,), (7,), (1536, F), (F, F), (6 * F, F),
          (16384, F), (F, 16384), (12, 20), (4, 1536, F), (3, 5, 7), (8, 3, 16)]
st_tree = {"blocks": {f"w{j}": jax.ShapeDtypeStruct((28,) + s, jnp.float32)
                      for j, s in enumerate(shapes)}}
ls_tree = unstack_blocks(st_tree)
assert len(ls_tree["blocks"]) == 28
n3, s3 = check_specs(st_tree, ls_tree, UNSTACK)
assert n3 == 28 * len(shapes)
for s in shapes:  # direct param_partition_spec check
    a = normspec(param_partition_spec((28,) + s, 8, stacked=True), len(s) + 1)
    b = normspec(param_partition_spec(s, 8, stacked=False), len(s))
    assert a[0] is None and a[1:] == b, (s, a, b)
# per-layer [6144, 6144] must shard axis 1 (tie -> later dim), as the stacked
assert normspec(param_shardings(ls_tree, mesh)["blocks"][0]["w5"].spec, 2) == (None, "fsdp")
assert normspec(param_shardings(ls_tree, mesh)["blocks"][0]["w4"].spec, 2) == (None, "fsdp")
assert normspec(param_shardings(ls_tree, mesh)["blocks"][3]["w6"].spec, 2) == ("fsdp", None)
print(f"d. param_shardings: {n1 + n2 + n3} unstacked leaves == stacked spec "
      "minus the layer axis: ok")

# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
B, ltxt, latH, latW = 8, 6, 16, 16
htok, wtok = latH // 2, latW // 2
txt_mask = np.ones((B, ltxt), dtype=bool)
txt_mask[0, 4:] = False
pos, mask = build_positions_and_mask(jnp.asarray(txt_mask), htok, wtok)
x0 = patchify(jnp.asarray(rng.normal(size=(B, 4, latH, latW)).astype(np.float32)), 2)
ctx = jnp.asarray(rng.normal(size=(B, ltxt, 3, 64)).astype(np.float32))
t = jnp.asarray(rng.uniform(0.1, 0.9, size=(B,)).astype(np.float32))
put = lambda a: jax.device_put(a, act)
inputs = tuple(put(a) for a in (x0, ctx, t, pos, mask))
wout = put(jnp.asarray(rng.normal(size=(B, htok * wtok, 16)).astype(np.float32)))
# qchunk >= padded length (256): dense attention, no lax.map in the graph
KW = dict(qchunk=512, act_sharding=act)


def make_loss(layer_loop, remat, lora=None):
    def loss(p, x0, ctx, t, pos, mask):
        out = dit_forward(p, cfg, x0, ctx, t, pos, mask, remat=remat,
                          layer_loop=layer_loop, lora=lora, **KW)
        return jnp.mean(out * wout), out
    return loss


def fwd_grad(params, layer_loop, remat, lora=None):
    f = jax.jit(jax.value_and_grad(make_loss(layer_loop, remat, lora), has_aux=True))
    (l, out), g = f(params, *inputs)
    return float(l), np.asarray(out), g


# ---------------------------------------------------------------------------
# a. unroll(list) == scan(stacked), forward + grads, three remat modes
# ---------------------------------------------------------------------------
for remat in ("dots", True, False):
    _, out_s, g_s = fwd_grad(params_st, "scan", remat)
    _, out_u, g_u = fwd_grad(params_ls, "unroll", remat)
    assert isinstance(g_u["blocks"], list)
    e_out = rel_err(out_u, out_s)
    e_grad = tree_rel_err(stack_blocks(g_u), g_s)
    print(f"a. remat {remat!s:5s}: unroll(list) vs scan(stacked) forward max "
          f"rel err {e_out:.3e}, grads {e_grad:.3e}")
    assert e_out < 1e-5 and e_grad < 1e-5, (remat, e_out, e_grad)
    if remat == "dots":
        out_dots_scan, g_dots_scan = out_s, g_s

# all three repeated-block prefixes as lists
_, out_a, g_a = fwd_grad(params_all, "unroll", "dots")
e_out, e_grad = rel_err(out_a, out_dots_scan), tree_rel_err(
    stack_blocks(g_a, ALL_PREFIXES), g_dots_scan)
print(f"a. all prefixes unstacked: forward {e_out:.3e}, grads {e_grad:.3e}")
assert e_out < 1e-5 and e_grad < 1e-5

# LoRA (stacked lora["blocks"], random lora_B so the delta matters)
lora = init_lora(params_st, jax.random.PRNGKey(3), rank=4, targets="attn+mlp")
lora = jax.tree.map(lambda a: a + 0.01, lora)
_, out_ls_l, _ = fwd_grad(params_st, "scan", "dots", lora)
_, out_lu_l, _ = fwd_grad(params_ls, "unroll", "dots", lora)
lora_list = {"blocks": unstack_blocks({"blocks": lora["blocks"]})["blocks"]}
_, out_lu_ll, _ = fwd_grad(params_ls, "unroll", "dots", lora_list)
assert rel_err(out_ls_l, out_dots_scan) > 1e-4, "LoRA delta had no effect"
e1, e2 = rel_err(out_lu_l, out_ls_l), rel_err(out_lu_ll, out_ls_l)
print(f"a. LoRA: unroll(list params, stacked/list lora) vs scan: {e1:.3e} / {e2:.3e}")
assert e1 < 1e-5 and e2 < 1e-5

# LoRA tree whose "blocks" subtree is None (the pair body gets (p, None))
lora_nb = {**lora, "blocks": None}
_, out_nb_s, _ = fwd_grad(params_st, "scan", "dots", lora_nb)
_, out_nb_u, _ = fwd_grad(params_ls, "unroll", "dots", lora_nb)
_, out_nb_us, _ = fwd_grad(params_st, "unroll", "dots", lora_nb)
e1, e2 = rel_err(out_nb_u, out_nb_s), rel_err(out_nb_us, out_nb_s)
print(f"a. LoRA with blocks=None: unroll (list / stacked params) vs scan: "
      f"{e1:.3e} / {e2:.3e}")
assert e1 < 1e-5 and e2 < 1e-5

# ---------------------------------------------------------------------------
# b. unroll over stacked blocks (static index) == list form
# ---------------------------------------------------------------------------
_, out_b, g_b = fwd_grad(params_st, "unroll", "dots")
_, out_bl, g_bl = fwd_grad(params_ls, "unroll", "dots")
e_out, e_grad = rel_err(out_b, out_bl), tree_rel_err(g_b, stack_blocks(g_bl))
print(f"b. unroll(stacked, static index) vs unroll(list): forward {e_out:.3e}, "
      f"grads {e_grad:.3e}")
assert e_out < 1e-5 and e_grad < 1e-5

# ---------------------------------------------------------------------------
# e. HLO: no dynamic-slice / dynamic-update-slice / while from model.py
# ---------------------------------------------------------------------------
# the opcode follows whitespace; operand references read "%dynamic-slice.N"
OP_RE = re.compile(r"\s(dynamic-slice|dynamic-update-slice|while)\(")
SRC_RE = re.compile(r'source_file="([^"]*)"')
OPNAME_RE = re.compile(r'op_name="([^"]*)"')


def model_py_ops(params, layer_loop):
    """Counts of dynamic-slice / dynamic-update-slice / while instructions
    attributed to model.py in the compiled fwd+bwd HLO. Dynamic slices the
    SPMD partitioner emits to cut a device's batch shard at a
    with_sharding_constraint (op_name ".../sharding_constraint", indexed by
    the partition id) have nothing to do with the layer loop — they are
    counted separately as "spmd-constraint-slice" and must be identical in
    both modes."""
    f = jax.jit(jax.value_and_grad(make_loss(layer_loop, "dots"), has_aux=True))
    text = f.lower(params, *inputs).compile().as_text()
    counts = {"dynamic-slice": 0, "dynamic-update-slice": 0, "while": 0,
              "spmd-constraint-slice": 0}
    for line in text.splitlines():
        m = OP_RE.search(line)
        if not m:
            continue
        src = SRC_RE.search(line)
        if not (src and src.group(1).endswith("model.py")):
            continue
        name = OPNAME_RE.search(line)
        if (m.group(1) == "dynamic-slice" and name
                and name.group(1).endswith("/sharding_constraint")):
            counts["spmd-constraint-slice"] += 1
        else:
            counts[m.group(1)] += 1
    return counts


c_scan = model_py_ops(params_st, "scan")
c_unroll = model_py_ops(params_ls, "unroll")
print(f"e. compiled fwd+bwd HLO ops from model.py (remat dots, FSDP, 8 dev): "
      f"scan {c_scan} | unroll+list {c_unroll}")
assert c_scan["dynamic-slice"] > 0 and c_scan["dynamic-update-slice"] > 0, c_scan
assert c_scan["while"] > 0, c_scan
assert (c_unroll["dynamic-slice"], c_unroll["dynamic-update-slice"],
        c_unroll["while"]) == (0, 0, 0), c_unroll
assert c_unroll["spmd-constraint-slice"] == c_scan["spmd-constraint-slice"], (
    c_scan, c_unroll)

# ---------------------------------------------------------------------------
# f. error paths
# ---------------------------------------------------------------------------
try:
    dit_forward(params_ls, cfg, *inputs, layer_loop="scan", **KW)
except TypeError as e:
    assert "stack_blocks" in str(e), e
else:
    raise AssertionError("scan accepted list blocks")
for bad in ("none", "while", None):
    try:
        dit_forward(params_st, cfg, *inputs, layer_loop=bad, **KW)
    except ValueError as e:
        assert "layer_loop" in str(e), e
    else:
        raise AssertionError(f"layer_loop={bad!r} accepted")
try:
    init_lora(params_ls, jax.random.PRNGKey(0), rank=4)
except ValueError as e:
    assert "per-layer list" in str(e), e
else:
    raise AssertionError("init_lora accepted list blocks")
try:
    save_lora(str(tmp / "l.safetensors"), lora_list, rank=4, alpha=4)
except ValueError as e:
    assert "per-layer list" in str(e), e
else:
    raise AssertionError("save_lora accepted a list LoRA tree")
# stacked LoRA whose layer axis disagrees with params: unroll must refuse it
# (scan refuses via the scan length check); a list of the wrong length too
_lora_short = {**lora, "blocks": jax.tree.map(lambda a: a[:-1], lora["blocks"])}
for _bad_lora in (_lora_short, {**lora, "blocks": lora_list["blocks"][:-1]}):
    try:
        dit_forward(params_ls, cfg, *inputs, lora=_bad_lora, layer_loop="unroll", **KW)
    except ValueError as e:
        assert "layers" in str(e), e
    else:
        raise AssertionError("unroll accepted a LoRA tree with the wrong layer count")
print("f. error paths (scan + list, bad layer_loop, LoRA on lists, LoRA layer count): ok")

print("ALL LAYER-LOOP TESTS PASSED")
