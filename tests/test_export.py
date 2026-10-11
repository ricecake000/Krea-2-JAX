"""Full fine-tuning checkpoint I/O on CPU with a tiny model (8 fake devices):

  1. load_krea2_params -> save_krea2_params reproduces the official
     safetensors bit for bit (bf16), from sharded or local, bf16 or fp32 trees,
     and the result loads strictly into the PyTorch reference model; the
     export leaves no host copy cached on the live arrays, and the write is
     atomic (a failure mid-write keeps the previous file, no temp left over),
  2. param_shardings agrees leaf-for-leaf with the shardings the loader used,
  3. an orbax round trip of {params, opt_state, ema, step} restores identical
     values and shardings; latest_step / max_to_keep / ema=None behave,
  4. scripts/export_full.py (subprocess, 1 CPU device) writes exactly what a
     direct save_krea2_params of the restored tree writes; a --base-ckpt
     whose keys/shapes differ is refused before anything is written (an
     existing destination stays byte-identical). official_key_shapes agrees
     with the keys/shapes save_krea2_params writes.

Requires CPU PyTorch (to fabricate a tiny reference checkpoint) + einops.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from test_parity import get_reference, patch_reference_for_cpu

TINY = dict(
    features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
    layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3,
)


def main():
    import torch

    mmdit = get_reference()
    patch_reference_for_cpu(mmdit, torch)

    import jax
    import jax.numpy as jnp
    import optax
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from krea2_jax.checkpoint import (
        load_krea2_params, make_mesh, official_key_shapes, param_shardings,
        save_krea2_params,
    )
    from krea2_jax.fullft_ckpt import (
        latest_step, make_manager, restore_state, restore_tree_on_host,
        save_state, wait,
    )

    assert len(jax.devices()) == 8, jax.devices()
    tmp = Path(tempfile.mkdtemp())

    # --- tiny official-layout checkpoint (bf16, like raw.safetensors) ---
    torch.manual_seed(0)
    model = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**TINY))
    g = torch.Generator().manual_seed(1)
    sd = {k: (torch.randn(v.shape, generator=g) * 0.02).to(torch.bfloat16)
          for k, v in model.state_dict().items()}
    base_path = tmp / "base.safetensors"
    save_file(sd, str(base_path))

    def assert_same_as_base(path, what):
        got = load_file(str(path))
        assert set(got) == set(sd), (
            f"{what}: key mismatch, missing {sorted(set(sd) - set(got))[:5]}, "
            f"extra {sorted(set(got) - set(sd))[:5]}"
        )
        for k, v in sd.items():
            assert got[k].dtype == torch.bfloat16, (what, k, got[k].dtype)
            assert got[k].shape == v.shape, (what, k, got[k].shape, v.shape)
            assert torch.equal(got[k].view(torch.int16), v.view(torch.int16)), (what, k)

    # --- 1. save_krea2_params inverts load_krea2_params bit for bit ---
    mesh = make_mesh()
    params_bf16 = load_krea2_params(str(base_path), mesh=mesh)
    params_f32 = load_krea2_params(str(base_path), mesh=mesh, dtype=jnp.float32)
    params_local = load_krea2_params(str(base_path), mesh=None, dtype=jnp.float32)
    assert any(not a.sharding.is_fully_replicated for a in jax.tree.leaves(params_f32))

    _missing = object()

    def cached_host_leaves(tree):
        """Number of jax.Array leaves holding a cached host copy (the private
        `_npy_value`), or None when this JAX has no such attribute."""
        vals = [getattr(a, "_npy_value", _missing) for a in jax.tree.leaves(tree)
                if isinstance(a, jax.Array)]
        if any(v is _missing for v in vals):
            return None
        return sum(v is not None for v in vals)

    for tree in (params_bf16, params_f32):
        assert cached_host_leaves(tree) in (None, 0), "host cache set before export"

    for name, tree in (("sharded bf16", params_bf16), ("sharded fp32", params_f32),
                       ("local fp32", params_local)):
        out = tmp / f"rt_{name.replace(' ', '_')}.safetensors"
        save_krea2_params(str(out), tree, metadata={"step": 3, "source": "params"})
        assert_same_as_base(out, name)
    with safe_open(str(out), framework="numpy") as f:
        assert f.metadata() == {"step": "3", "source": "params"}, f.metadata()
    # deterministic bytes: the same tree always gives the same file
    save_krea2_params(str(tmp / "again.safetensors"), params_local,
                      metadata={"source": "params", "step": 3})
    assert (tmp / "again.safetensors").read_bytes() == out.read_bytes()
    # the export loads strictly into the official PyTorch model
    missing, unexpected = model.load_state_dict(load_file(str(out)), strict=True)
    assert not missing and not unexpected
    # fp32 output dtype works too (values exact: bf16 -> fp32 is lossless)
    out32 = tmp / "rt_f32.safetensors"
    save_krea2_params(str(out32), params_f32, dtype=jnp.float32)
    got32 = load_file(str(out32))
    for k, v in sd.items():
        assert got32[k].dtype == torch.float32 and torch.equal(got32[k], v.float()), k
    print("save_krea2_params round-trips the official checkpoint bit for bit: ok")

    # official_key_shapes predicts exactly the keys/shapes save_krea2_params
    # writes (sharded, local and abstract trees)
    def file_key_shapes(p):
        with safe_open(str(p), framework="numpy") as f:
            return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}

    written = file_key_shapes(out)
    assert written == {k: tuple(v.shape) for k, v in sd.items()}
    abstract_f32 = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), params_f32)
    for name, tree in (("sharded fp32", params_f32), ("local fp32", params_local),
                       ("abstract", abstract_f32)):
        ks = official_key_shapes(tree)
        assert ks == written, (name, sorted(set(ks) ^ set(written))[:5])
    print("official_key_shapes matches the written file: ok")

    # exporting must not leave the gathered host copy cached on the live
    # (sharded) arrays: at 12.9B that would pin ~52 GB of host RAM. Covers the
    # casting path (bf16 out of fp32 / bf16 trees) and the same-dtype path
    # (fp32 out of the fp32 tree, just above).
    n_cached = [cached_host_leaves(t) for t in (params_bf16, params_f32)]
    if None in n_cached:
        print("  (skip) this JAX has no Array._npy_value; host-cache check skipped")
    else:
        assert n_cached == [0, 0], f"leaves with a cached host copy: {n_cached}"
        print("export leaves no host copy cached on the live arrays: ok")

    # atomic write: a failure on the 2nd tensor keeps the previous file
    # byte for byte and leaves no temp file behind
    atom = tmp / "atomic"
    atom.mkdir()
    dest = atom / "model.safetensors"
    good = {"a": np.full((3, 4), 1.5, np.float32),
            "b": np.arange(6, dtype=np.float32).reshape(2, 3),
            "c": np.zeros((5,), np.float32)}
    seen_tmp = []

    class FailingLeaf:
        """Leaf that fails when fetched (2nd in flatten/fetch order)."""
        shape = (2, 3)
        dtype = np.dtype(np.float32)

        def __array__(self, dtype=None, copy=None):
            seen_tmp.append(sorted(p.name for p in atom.iterdir()
                                   if ".tmp." in p.name))
            raise RuntimeError("injected write failure")

    def expect_failure(target):
        try:
            save_krea2_params(str(target), {**good, "b": FailingLeaf()})
        except RuntimeError as e:
            assert "injected write failure" in str(e), e
        else:
            raise AssertionError("injected failure did not propagate")

    save_krea2_params(str(dest), good, metadata={"v": 1})
    prev = dest.read_bytes()
    expect_failure(dest)
    assert seen_tmp and seen_tmp[-1], "the write did not go to a temp file"
    assert dest.read_bytes() == prev, "failed export clobbered the previous file"
    assert sorted(p.name for p in atom.iterdir()) == ["model.safetensors"], (
        sorted(p.name for p in atom.iterdir()))
    # no previous file: nothing at all is left behind
    expect_failure(atom / "fresh.safetensors")
    assert sorted(p.name for p in atom.iterdir()) == ["model.safetensors"], (
        sorted(p.name for p in atom.iterdir()))
    print("save_krea2_params is atomic (failure keeps the old file, no temp): ok")

    # --- 2. param_shardings matches what the loader placed ---
    def check_shardings(tree, sharding):
        want = param_shardings(tree, mesh, sharding=sharding)
        got = jax.tree.map(lambda a: a.sharding, tree)
        assert jax.tree.structure(want) == jax.tree.structure(got)
        for w, a in zip(jax.tree.leaves(want), jax.tree.leaves(got)):
            assert w == a, (w, a)
        abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), tree)
        for w, a in zip(jax.tree.leaves(param_shardings(abstract, mesh, sharding)),
                        jax.tree.leaves(got)):
            assert w == a, (w, a)

    check_shardings(params_f32, "fsdp")
    check_shardings(params_bf16, "fsdp")
    check_shardings(
        load_krea2_params(str(base_path), mesh=mesh, sharding="replicate"), "replicate"
    )
    # stacked leaves must never shard the block axis
    for path, a in jax.tree_util.tree_flatten_with_path(params_f32)[0]:
        dotted = ".".join(str(k.key) for k in path)
        if dotted.startswith(("blocks.", "txtfusion.layerwise_blocks.",
                              "txtfusion.refiner_blocks.")):
            assert a.sharding.spec[:1] in ((), (None,)), (dotted, a.sharding.spec)
    print("param_shardings matches the loaded shardings: ok")

    # --- 3. orbax train-state round trip ---
    pshard = param_shardings(params_f32, mesh)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))
    opt_state = jax.jit(tx.init, in_shardings=(pshard,))(params_f32)
    # make the optimizer moments non-trivial so a broken restore can't pass
    grads = jax.tree.map(lambda a: jnp.sin(a * 37.0), params_f32)
    updates, opt_state = jax.jit(tx.update)(grads, opt_state, params_f32)
    params_new = optax.apply_updates(params_f32, updates)
    ema = jax.jit(
        lambda p, q: jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, p, q),
        out_shardings=pshard,
    )(params_f32, params_new)
    state = {"params": params_new, "opt_state": opt_state, "ema": ema, "step": 7}

    ckpt_dir = tmp / "out" / "ckpt"
    assert latest_step(str(ckpt_dir)) is None  # missing dir
    mgr = make_manager(str(ckpt_dir), max_to_keep=2)
    assert latest_step(str(ckpt_dir)) is None  # empty dir
    for s in (3, 5, 7):
        save_state(mgr, s, {**state, "step": s})
    wait(mgr)
    assert latest_step(str(ckpt_dir)) == 7
    kept = sorted(int(p.name) for p in ckpt_dir.iterdir() if p.name.isdigit())
    assert kept == [5, 7], kept

    def abstract(t):
        return jax.tree.map(
            lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=a.sharding), t
        )

    abstract_state = {"params": abstract(params_new), "opt_state": abstract(opt_state),
                      "ema": abstract(ema), "step": 0}
    mgr2 = make_manager(str(ckpt_dir), max_to_keep=2)  # fresh manager, as on resume
    restored = restore_state(mgr2, latest_step(str(ckpt_dir)), abstract_state)
    assert restored["step"] == 7 and type(restored["step"]) is int, restored["step"]
    for key in ("params", "opt_state", "ema"):
        want, got = state[key], restored[key]
        assert jax.tree.structure(want) == jax.tree.structure(got), key
        for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got)):
            assert a.dtype == b.dtype and a.shape == b.shape, key
            assert a.sharding == b.sharding, (key, a.sharding, b.sharding)
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    print("orbax save/restore of params/opt_state/ema/step: ok")

    # ema=None is stored as "no EMA" and comes back as None
    ckpt_noema = tmp / "noema" / "ckpt"
    mgr3 = make_manager(str(ckpt_noema))
    save_state(mgr3, 2, {**state, "ema": None, "step": 2})
    wait(mgr3)
    assert latest_step(str(ckpt_noema)) == 2
    r = restore_state(mgr3, 2, {**abstract_state, "ema": None})
    assert r["ema"] is None and r["step"] == 2
    for a, b in zip(jax.tree.leaves(state["params"]), jax.tree.leaves(r["params"])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    try:
        restore_state(mgr3, 2, abstract_state)
    except ValueError:
        pass
    else:
        raise AssertionError("restoring an EMA from an EMA-less checkpoint must fail")
    # a checkpoint with EMA can still be resumed with EMA disabled
    r = restore_state(mgr2, 7, {**abstract_state, "ema": None})
    assert r["ema"] is None
    print("ema=None round trip: ok")

    # --- 4. scripts/export_full.py on a 1-device CPU process ---
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": ""}

    def export(out, *extra):
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "export_full.py"),
             "--ckpt-dir", str(ckpt_dir), "--output", str(out), *extra],
            check=True, env=env,
        )

    for source, tree in (("ema", ema), ("params", params_new)):
        out = tmp / f"export_{source}.safetensors"
        export(out, "--source", source, "--step", "7",
               "--base-ckpt", str(base_path))
        direct = tmp / f"direct_{source}.safetensors"
        save_krea2_params(str(direct), tree, metadata={
            "step": 7, "source": source, "format": "krea2-full-ft",
            "base": base_path.name,
        })
        assert out.read_bytes() == direct.read_bytes(), f"export {source} differs"
    # a mismatching --base-ckpt is rejected BEFORE writing: the existing
    # destination stays byte-identical and no temp file is left behind
    guard = tmp / "guard"
    guard.mkdir()
    dest = guard / "model.safetensors"
    dest.write_bytes((tmp / "export_ema.safetensors").read_bytes())
    prev = dest.read_bytes()
    victim = sorted(sd)[0]
    renamed = dict(sd)
    renamed[victim + ".renamed"] = renamed.pop(victim)
    reshaped = dict(sd)
    reshaped[victim] = reshaped[victim].reshape(-1)[:-1].clone() \
        if reshaped[victim].numel() > 1 else torch.zeros(2, dtype=torch.bfloat16)
    for what, bad_sd in (("renamed", renamed), ("reshaped", reshaped)):
        bad_base = tmp / f"base_{what}.safetensors"
        save_file(bad_sd, str(bad_base))
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "export_full.py"),
             "--ckpt-dir", str(ckpt_dir), "--output", str(dest),
             "--base-ckpt", str(bad_base)],
            env=env, capture_output=True, text=True,
        )
        assert proc.returncode != 0, f"{what} base was accepted"
        assert "do not match" in proc.stderr, proc.stderr[-2000:]
        assert victim in proc.stderr, proc.stderr[-2000:]
        assert dest.read_bytes() == prev, f"rejected export ({what}) clobbered dest"
        assert sorted(p.name for p in guard.iterdir()) == ["model.safetensors"], (
            sorted(p.name for p in guard.iterdir()))
    print("mismatching --base-ckpt is refused before writing (dest intact): ok")
    # default source = ema when present, default step = newest
    export(tmp / "export_default.safetensors")
    with safe_open(str(tmp / "export_default.safetensors"), framework="numpy") as f:
        assert f.metadata()["source"] == "ema" and f.metadata()["step"] == "7"
    # exporting the unmodified params of a step reproduces the base file
    mgr4 = make_manager(str(tmp / "pristine" / "ckpt"))
    save_state(mgr4, 0, {"params": params_f32, "opt_state": opt_state,
                         "ema": None, "step": 0})
    wait(mgr4)
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export_full.py"),
         "--ckpt-dir", str(tmp / "pristine" / "ckpt"),
         "--output", str(tmp / "pristine.safetensors")],
        check=True, env=env,
    )
    assert_same_as_base(tmp / "pristine.safetensors", "pristine export")
    # host restore helper in-process (8-device process, single-device target)
    tree, meta = restore_tree_on_host(str(ckpt_dir), None)
    assert meta["item"] == "ema" and meta["step"] == 7
    for a, b in zip(jax.tree.leaves(ema), jax.tree.leaves(tree)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    print("export_full.py matches a direct save_krea2_params: ok")

    for m in (mgr, mgr2, mgr3, mgr4):
        m.close()
    print("ALL EXPORT TESTS PASSED")


if __name__ == "__main__":
    main()
