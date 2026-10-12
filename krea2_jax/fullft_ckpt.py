"""Orbax checkpointing for the full fine-tuning train state.

Layout under `ckpt_dir/<step>/`, one orbax item per tree:
  params     fp32 master weights
  opt_state  optax state
  ema        EMA weights (only written when state["ema"] is not None)
  meta       JSON {"step": int, "has_ema": bool, "config": dict (optional)}
Separate items let scripts/export_full.py restore a single tree (params or
ema) without reading the optimizer state.

Saves are asynchronous: the device->host copy finishes inside `save_state`,
so the caller may donate/overwrite the arrays as soon as it returns; call
`wait` before reading the files or exiting (a pending save is lost if the
interpreter exits first). A save killed mid-write leaves a stale
`<step>.orbax-checkpoint-tmp` dir; the writing manager (`make_manager`)
removes such dirs when it opens the checkpoint dir.
"""

import os

import jax
import orbax.checkpoint as ocp

TREE_ITEMS = ("params", "opt_state", "ema")


def _handlers():
    return {
        "params": ocp.StandardCheckpointHandler(),
        "opt_state": ocp.StandardCheckpointHandler(),
        "ema": ocp.StandardCheckpointHandler(),
        "meta": ocp.JsonCheckpointHandler(),
    }


def make_manager(ckpt_dir, max_to_keep=3, async_save=True):
    """CheckpointManager over `ckpt_dir`, keeping the newest `max_to_keep` steps."""
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        enable_async_checkpointing=async_save,
        create=True,
        cleanup_tmp_directories=True,
    )
    return ocp.CheckpointManager(
        os.path.abspath(ckpt_dir), options=options, item_handlers=_handlers()
    )


def save_state(manager, step, state, config=None):
    """Save {"params", "opt_state", "ema" (or None), "step"} as orbax step `step`.

    config: optional JSON-serializable dict (the run configuration) stored in
    the meta item, so a resume can check it continues the same run.
    Returns once the arrays are copied off the devices; the write to disk
    continues in the background (see `wait`).
    """
    step = int(step)
    has_ema = state.get("ema") is not None
    meta = {"step": int(state.get("step", step)), "has_ema": has_ema}
    if config is not None:
        meta["config"] = dict(config)
    items = {
        "params": ocp.args.StandardSave(state["params"]),
        "opt_state": ocp.args.StandardSave(state["opt_state"]),
        "meta": ocp.args.JsonSave(meta),
    }
    if has_ema:
        items["ema"] = ocp.args.StandardSave(state["ema"])
    manager.save(step, args=ocp.args.Composite(**items), force=True)


def latest_step(ckpt_dir):
    """Newest finalized step under `ckpt_dir`, or None (missing dir / no steps)."""
    if not os.path.isdir(ckpt_dir):
        return None
    mgr = ocp.CheckpointManager(
        os.path.abspath(ckpt_dir),
        options=ocp.CheckpointManagerOptions(read_only=True, create=False),
        item_handlers=_handlers(),
    )
    try:
        return mgr.latest_step()
    finally:
        mgr.close()


def all_steps(ckpt_dir):
    """Sorted list of every step orbax sees under `ckpt_dir` ([] if the dir
    is missing). Like `latest_step`, a step dir is listed even if its
    contents are incomplete (e.g. an interrupted copy)."""
    if not os.path.isdir(ckpt_dir):
        return []
    mgr = ocp.CheckpointManager(
        os.path.abspath(ckpt_dir),
        options=ocp.CheckpointManagerOptions(read_only=True, create=False),
        item_handlers=_handlers(),
    )
    try:
        return sorted(int(s) for s in mgr.all_steps())
    finally:
        mgr.close()


def read_meta(manager, step):
    """The JSON meta item of `step`: {"step": int, "has_ema": bool,
    "config": dict or None} ("config" is None for checkpoints saved without
    one)."""
    out = manager.restore(step, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = dict(out["meta"])
    meta.setdefault("config", None)
    return meta


def read_meta_at(ckpt_dir, step):
    """`read_meta` of `step` through a read-only manager over `ckpt_dir`
    (nothing is created or cleaned up). Raises if the step is missing or its
    meta item is incomplete/unreadable."""
    mgr = ocp.CheckpointManager(
        os.path.abspath(ckpt_dir),
        options=ocp.CheckpointManagerOptions(read_only=True, create=False),
        item_handlers=_handlers(),
    )
    try:
        return read_meta(mgr, step)
    finally:
        mgr.close()


def restore_state(manager, step, abstract_state):
    """Restore `step` into arrays laid out like `abstract_state`.

    abstract_state: {"params", "opt_state", "ema" (or None), "step"} with
    jax.ShapeDtypeStruct leaves carrying `.sharding` (the "step" entry is
    ignored). Returns the same dict with real sharded arrays; "step" is a
    python int, and "ema" is None when the abstract "ema" is None. Asking
    for an EMA from a checkpoint saved without one is an error.
    """
    meta = read_meta(manager, step)
    want_ema = abstract_state.get("ema") is not None
    if want_ema and not meta.get("has_ema", False):
        raise ValueError(
            f"checkpoint step {step} has no EMA weights (saved with EMA "
            "disabled) but the restore target expects one"
        )
    items = {
        "params": ocp.args.StandardRestore(abstract_state["params"]),
        "opt_state": ocp.args.StandardRestore(abstract_state["opt_state"]),
    }
    if want_ema:
        items["ema"] = ocp.args.StandardRestore(abstract_state["ema"])
    out = manager.restore(step, args=ocp.args.Composite(**items))
    return {
        "params": out["params"],
        "opt_state": out["opt_state"],
        "ema": out["ema"] if want_ema else None,
        "step": int(meta["step"]),
    }


def restore_tree_on_host(ckpt_dir, step=None, item=None, dtype=None):
    """Restore one tree ("params" or "ema") onto the local CPU, no mesh needed.

    step=None picks the newest step; item=None picks "ema" when the step has
    one, else "params".
    The abstract target is rebuilt from the checkpoint's own metadata with a
    single-CPU-device sharding, so this works on a machine with a different
    device count/platform than the one that trained (e.g. JAX_PLATFORMS=cpu).
    dtype=None keeps the stored dtype. Returns (tree, meta) where meta is
    the step's JSON meta plus "item" (the tree that was restored).
    """
    if item not in (None, "params", "ema"):
        raise ValueError(f"unknown tree item: {item!r}")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")
    mgr = ocp.CheckpointManager(
        os.path.abspath(ckpt_dir),
        options=ocp.CheckpointManagerOptions(read_only=True, create=False),
        item_handlers=_handlers(),
    )
    try:
        if step is None:
            step = mgr.latest_step()
            if step is None:
                raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
        meta = read_meta(mgr, step)
        if item is None:
            item = "ema" if meta.get("has_ema", False) else "params"
        meta["item"] = item
        if item == "ema" and not meta.get("has_ema", False):
            raise ValueError(f"checkpoint step {step} was saved without EMA")
        tree_meta = mgr.item_metadata(step)[item]
        cpu = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
        abstract = jax.tree.map(
            lambda m: jax.ShapeDtypeStruct(
                tuple(m.shape), dtype or m.dtype, sharding=cpu
            ),
            tree_meta,
        )
        out = mgr.restore(
            step, args=ocp.args.Composite(**{item: ocp.args.StandardRestore(abstract)})
        )
        return out[item], meta
    finally:
        mgr.close()


def wait(manager):
    """Block until every pending async save has been written to disk."""
    manager.wait_until_finished()
