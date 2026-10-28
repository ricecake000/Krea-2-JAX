"""Full-FT checkpoint I/O on the tiny model: official safetensors export, param
shardings, orbax train-state round trip, scripts/export_full.py."""

import contextlib
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from safetensors import safe_open

from helpers import REPO, TINY
from krea2_jax.checkpoint import (init_lora, load_krea2_params, load_lora, official_key_shapes,
                                  param_shardings, save_krea2_params, save_lora)
from krea2_jax.fullft_ckpt import latest_step, make_manager, restore_state, save_state, wait

NAMES = ("sharded bf16", "sharded fp32", "local fp32")
ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": ""}


def abstract(tree, sharded=True):
    return jax.tree.map(lambda a: jax.ShapeDtypeStruct(
        a.shape, a.dtype, sharding=a.sharding if sharded else None), tree)


def export_cmd(ckpt, out, *extra):
    return [sys.executable, str(REPO / "scripts" / "export_full.py"), "--ckpt-dir", str(ckpt), "--output", str(out), *extra]


@pytest.fixture(scope="module")
def loaded(tiny_ckpt, mesh):
    p = str(tiny_ckpt[2])
    trees = dict(zip(NAMES, (load_krea2_params(p, mesh=mesh),
                             load_krea2_params(p, mesh=mesh, dtype=jnp.float32),
                             load_krea2_params(p, mesh=None, dtype=jnp.float32))))
    assert any(not a.sharding.is_fully_replicated for a in jax.tree.leaves(trees["sharded fp32"]))
    return trees


@pytest.fixture(scope="module")
def train_state(loaded, mesh, tmp_path_factory):
    p0, shard = loaded["sharded fp32"], param_shardings(loaded["sharded fp32"], mesh)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))
    opt = jax.jit(tx.init, in_shardings=(shard,))(p0)
    upd, opt = jax.jit(tx.update)(jax.tree.map(lambda a: jnp.sin(a * 37.0), p0), opt, p0)
    params = optax.apply_updates(p0, upd)
    ema = jax.jit(lambda p, q: jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, p, q), out_shardings=shard)(p0, params)
    state = {"params": params, "opt_state": opt, "ema": ema, "step": 7}
    ckpt = tmp_path_factory.mktemp("orbax") / "ckpt"
    mgr = make_manager(str(ckpt), max_to_keep=2)
    for s in (3, 5, 7):
        save_state(mgr, s, {**state, "step": s})
    wait(mgr)
    mgr.close()
    return ckpt, state, {k: abstract(v) for k, v in state.items() if k != "step"} | {"step": 0}


def assert_trees_equal(want, got):
    assert jax.tree.structure(want) == jax.tree.structure(got)
    for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got)):
        assert a.dtype == b.dtype and a.sharding == b.sharding
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("name", NAMES)
def test_save_reproduces_official_bf16(loaded, tiny_ckpt, ref, name, tmp_path):
    """load -> save reproduces the official bf16 file bit for bit; loads strict into torch."""
    import torch
    from safetensors.torch import load_file
    out = tmp_path / "out.safetensors"
    save_krea2_params(str(out), loaded[name], metadata={"step": 3, "source": "params"})
    base, got = load_file(str(tiny_ckpt[2])), load_file(str(out))
    assert set(got) == set(base)
    for k, v in base.items():
        assert got[k].dtype == torch.bfloat16 and torch.equal(got[k].view(torch.int16), v.view(torch.int16)), k
    with safe_open(str(out), framework="numpy") as f:
        assert f.metadata() == {"step": "3", "source": "params"}
        written = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    assert official_key_shapes(loaded[name]) == written
    assert official_key_shapes(abstract(loaded[name], sharded=False)) == written
    ref.SingleStreamDiT(ref.SingleMMDiTConfig(**TINY)).load_state_dict(got, strict=True)


def test_lora_round_trip(loaded, tmp_path):
    lora = init_lora(loaded["local fp32"], jax.random.PRNGKey(0), rank=4, targets="full")
    keys = iter(jax.random.split(jax.random.PRNGKey(1), len(jax.tree.leaves(lora))))
    lora = jax.tree.map(lambda a: jax.random.normal(next(keys), a.shape, a.dtype), lora)
    save_lora(str(tmp_path / "l.safetensors"), lora, rank=4, alpha=8)
    got, meta = load_lora(str(tmp_path / "l.safetensors"))
    assert (meta["rank"], meta["alpha"]) == ("4", "8") and jax.tree.structure(got) == jax.tree.structure(lora)
    for a, b in zip(jax.tree.leaves(lora), jax.tree.leaves(got)):
        assert np.asarray(a).any() and b.dtype == a.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


WQ, A, B = "blocks.0.attn.wq", np.ones((4, 8), np.float32), np.ones((8, 4), np.float32)


@pytest.mark.parametrize("flat,meta,match", [
    ({f"{WQ}.lora_A.weight": A, f"{WQ}.lora_B.weight": B}, {"rank": "4"}, None),
    ({f"{WQ}.lora_A.weight": A}, None, "missing"),
    ({"blocks.0.attn.projetor.lora_A.weight": A, "blocks.0.attn.projetor.lora_B.weight": B}, None, "unknown LoRA"),
    ({f"{WQ}.lora_A.weight": A, "blocks.1.attn.wq.lora_A.weight": A, f"{WQ}.lora_B.weight": B}, None, "layers"),
    ({f"{WQ}.lora_A.weight": A, f"{WQ}.lora_B.weight": B}, {"targets": "attn"}, "do not match targets"),
    ({f"{WQ}.lora_A.weight": A, f"{WQ}.lora_B.weight": B}, {"rank": "8"}, "metadata says rank 8")])
def test_load_lora_rejects_malformed(flat, meta, match, tmp_path):
    from safetensors.numpy import save_file
    save_file(flat, str(tmp_path / "bad.safetensors"), metadata=meta)
    with pytest.raises(ValueError, match=match) if match else contextlib.nullcontext():
        load_lora(str(tmp_path / "bad.safetensors"))


def test_save_is_atomic(tmp_path):
    good = {"a": np.full((3, 4), 1.5, np.float32), "b": np.ones((2, 3), np.float32)}

    class FailingLeaf:
        shape, dtype = (2, 3), np.dtype(np.float32)
        def __array__(self, dtype=None, copy=None):
            seen_tmp.extend(p.name for p in tmp_path.iterdir() if ".tmp." in p.name)
            raise RuntimeError("injected write failure")

    seen_tmp, dest = [], tmp_path / "model.safetensors"
    save_krea2_params(str(dest), good)
    prev = dest.read_bytes()
    for target in (dest, tmp_path / "fresh.safetensors"):
        with pytest.raises(RuntimeError, match="injected write failure"):
            save_krea2_params(str(target), {**good, "b": FailingLeaf()})
    assert seen_tmp, "the write did not go to a temp file"
    assert dest.read_bytes() == prev
    assert [p.name for p in tmp_path.iterdir()] == ["model.safetensors"]


@pytest.mark.parametrize("sharding", ["fsdp", "replicate"])
def test_param_shardings_match_loader(tiny_ckpt, mesh, sharding):
    tree = load_krea2_params(str(tiny_ckpt[2]), mesh=mesh, dtype=jnp.float32, sharding=sharding)
    got = jax.tree.map(lambda a: a.sharding, tree)
    assert param_shardings(tree, mesh, sharding) == got
    assert param_shardings(abstract(tree, sharded=False), mesh, sharding) == got
    for path, a in jax.tree_util.tree_flatten_with_path(tree)[0]:
        dotted = ".".join(str(k.key) for k in path)
        if dotted.startswith(("blocks.", "txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks.")):
            assert a.sharding.spec[:1] in ((), (None,)), (dotted, a.sharding.spec)


def test_orbax_round_trip(train_state, tmp_path):
    """Values/dtypes/shardings survive; max_to_keep rotates; ema=None round trip, EMA from EMA-less raises."""
    ckpt, state, abs_state = train_state
    assert latest_step(str(ckpt)) == 7
    assert sorted(int(p.name) for p in ckpt.iterdir() if p.name.isdigit()) == [5, 7]
    assert latest_step(str(ckpt.parent / "missing")) is None
    mgr = make_manager(str(ckpt), max_to_keep=2)
    r = restore_state(mgr, 7, abs_state)
    assert r["step"] == 7 and type(r["step"]) is int
    for key in ("params", "opt_state", "ema"):
        assert_trees_equal(state[key], r[key])
    assert restore_state(mgr, 7, {**abs_state, "ema": None})["ema"] is None
    mgr.close()
    mgr = make_manager(str(tmp_path / "noema"))
    save_state(mgr, 2, {**state, "ema": None, "step": 2})
    wait(mgr)
    r = restore_state(mgr, 2, {**abs_state, "ema": None})
    assert r["ema"] is None and r["step"] == 2
    assert_trees_equal(state["params"], r["params"])
    with pytest.raises(ValueError):
        restore_state(mgr, 2, abs_state)
    mgr.close()


@pytest.mark.parametrize("source", ["ema", "params"])
def test_export_full_matches_direct_save(train_state, tiny_ckpt, source, tmp_path):
    ckpt, state, _ = train_state
    out, direct = tmp_path / "out.safetensors", tmp_path / "direct.safetensors"
    subprocess.run(export_cmd(ckpt, out, "--source", source, "--step", "7", "--base-ckpt", str(tiny_ckpt[2])),
                   check=True, env=ENV)
    save_krea2_params(str(direct), state[source], metadata={
        "step": 7, "source": source, "format": "krea2-full-ft", "base": tiny_ckpt[2].name})
    assert out.read_bytes() == direct.read_bytes()


@pytest.mark.parametrize("bad", ["renamed", "reshaped"])
def test_export_full_refuses_mismatched_base(train_state, tiny_ckpt, bad, tmp_path):
    """A --base-ckpt with other keys/shapes is refused before the destination is touched."""
    from safetensors.torch import load_file, save_file
    sd = load_file(str(tiny_ckpt[2]))
    victim = sorted(sd)[0]
    t = sd.pop(victim)
    sd.update({victim + ".renamed": t} if bad == "renamed" else {victim: t.reshape(-1)[:-1].clone()})
    save_file(sd, str(tmp_path / "bad.safetensors"))
    dest = tmp_path / "model.safetensors"
    dest.write_bytes(b"previous export")
    proc = subprocess.run(export_cmd(train_state[0], dest, "--base-ckpt", str(tmp_path / "bad.safetensors")),
                          env=ENV, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "do not match" in proc.stderr and victim in proc.stderr, proc.stderr[-2000:]
    assert dest.read_bytes() == b"previous export"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.safetensors", "model.safetensors"]
