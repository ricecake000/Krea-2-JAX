"""CPU tests of scripts/modal_coherence.py without Modal, network or real
weights (tiny train states saved with the repo's own save path):

  * partial restore: a tiny train state (real train_full.make_optimizer /
    make_schedule, real optimizer steps + the train step's EMA update, saved
    with fullft_ckpt.save_state) in both --layer-loop layouts; every leaf
    group restored with ocp.PLACEHOLDER for the rest equals the full
    restore exactly (params / ema via fullft_ckpt.restore_tree_on_host,
    mu / nu via fullft_ckpt.restore_state into the real optax state); the
    groups cover every leaf once; the Adam count; a partial restore of one
    leaf of four (saved FSDP-sharded from 8 devices, like train_full) costs
    ~one leaf of RAM, with no transient copy (subprocesses, VmRSS /
    ru_maxrss)
  * metrics on hand-built arrays with known answers: cos(-u, D),
    cos(-m, D), sign agreement, bias correction, bf16 round-to-nearest-even
    rounding-away count (incl. ties), norm ratio / cos of the bf16 export,
    chunked / threaded sums == one chunk, undefined values -> None
  * analyze() on the tiny state == a direct numpy computation (global,
    per kind); the scan and unroll layouts give identical per-tensor sums;
    official keys / kinds / blocks; memory-budget refusal; every array of
    a leaf group is released (weakref probe, no gc) before the next group
    is restored
  * schedule == train_full.make_schedule (optax) for warmup / constant /
    cosine configs; the simulation's Adam == optax.adam on the same
    gradients; s = 0: cos(-u, D) ~ 10/sqrt(19 T_eff), ||De||/||D|| ~ 0.51
    (T 1513, warmup 150); large s: rms(D)/lr_sum -> 1, cos -> 1; inversion
    (unique, clamped, and "ambiguous" for a non-monotone or flat curve,
    incl. the real run1 ||De||/||D|| tail)
  * coherence_impl end to end against a fake R2 (the S3 fake of
    tests/test_modal_export_full.py, with the s3transfer download
    allowlist): params already on the volume -> only ema / opt_state
    objects downloaded, a re-run downloads nothing, report JSON, then the
    simulation + implied values; selection of opt_state; a step saved
    without EMA is refused before any transfer
  * CLI parsing, --sim-only (local), --help

    python tests/test_modal_coherence.py   # or: pytest tests/test_modal_coherence.py
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import math
import subprocess
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np

import modal_coherence as mc
import modal_export_full as mxf
from test_modal_export_full import FakeS3

TINY = dict(
    features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
    layers=3, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3,
)
STEPS = 4
RUN_CFG = {"lr": 1e-3, "warmup_steps": 2, "cosine_decay": False,
           "steps": 12, "min_lr": 0.0, "beta1": 0.9, "beta2": 0.999,
           "eps": 1e-8, "weight_decay": 0.01, "grad_clip": 1.0,
           "ema_decay": 0.9}


def _base(tmp):
    """Tiny official-layout bf16 base safetensors (cached per tmp dir)."""
    import jax
    import jax.numpy as jnp

    from krea2_jax.checkpoint import abstract_krea2_params, save_krea2_params
    from krea2_jax.config import Krea2Config

    path = Path(tmp) / "base" / "raw.safetensors"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        ab = abstract_krea2_params(Krea2Config(**TINY), dtype=jnp.float32)
        base = jax.tree.map(lambda s: jnp.asarray(
            (0.05 * rng.standard_normal(s.shape)).astype(np.float32)), ab)
        save_krea2_params(str(path), base, dtype=jnp.bfloat16)
    return path


def make_state(tmp, layout, step=STEPS):
    """A tiny train state saved like train_full: params loaded f32 from the
    bf16 base (unrolled `blocks` for layout "unroll"), the real optimizer,
    `step` updates with synthetic gradients (a fixed coherent direction +
    noise, identical in both layouts: drawn per official key), EMA as in
    step_fn. -> (ckpt_dir, base_path, tx, params0)."""
    import jax
    import jax.numpy as jnp
    import optax

    import train_full
    from krea2_jax import fullft_ckpt
    from krea2_jax.checkpoint import _official_entries, load_krea2_params

    base_path = _base(tmp)
    unstack = ("blocks",) if layout == "unroll" else ()
    params = load_krea2_params(str(base_path), dtype=jnp.float32,
                               unstack=unstack)
    params0 = params
    args = SimpleNamespace(**RUN_CFG)
    tx = train_full.make_optimizer(args, train_full.make_schedule(args))
    opt = tx.init(params)
    ema = jax.tree.map(jnp.copy, params)
    entries, _ = _official_entries(params)
    leaves, treedef = jax.tree.flatten(params)

    def grads(t):
        out = [np.zeros(x.shape, np.float32) for x in leaves]
        for name, li, layer, shape in entries:
            seed = sum(map(ord, name)) * 1000 + t
            r = np.random.default_rng(seed)
            g = (np.sin(np.arange(int(np.prod(shape)))).reshape(shape) * 0.3
                 + r.standard_normal(shape)).astype(np.float32) * 1e-4
            # (global norm ~0.1: never clipped, so the update is elementwise
            # and both layouts get bitwise identical states)
            if layer is None:
                out[li] = g
            else:
                out[li][layer] = g
        return jax.tree.unflatten(treedef, [jnp.asarray(x) for x in out])

    decay = RUN_CFG["ema_decay"]
    for t in range(step):
        upd, opt = tx.update(grads(t), opt, params)
        params = optax.apply_updates(params, upd)
        ema = jax.tree.map(lambda e, p: e + (1.0 - decay) * (p - e), ema,
                           params)
    ckpt = Path(tmp) / f"ckpt_{layout}"
    mgr = fullft_ckpt.make_manager(str(ckpt))
    fullft_ckpt.save_state(mgr, step, {"params": params, "opt_state": opt,
                                       "ema": ema, "step": step},
                           config={**RUN_CFG, "layer_loop": layout})
    fullft_ckpt.wait(mgr)
    mgr.close()
    return ckpt, base_path, tx, params0


def item_dirs(ckpt, step=STEPS):
    return [Path(ckpt) / str(step) / t for t in mc.CKPT_TREES]


def full_restore(ckpt, tx, params0, step=STEPS):
    """params, ema (restore_tree_on_host) and the opt_state restored into
    the real optax structure (fullft_ckpt.restore_state) -- the production
    restore paths."""
    import jax
    import orbax.checkpoint as ocp

    from krea2_jax import fullft_ckpt

    params, _ = fullft_ckpt.restore_tree_on_host(str(ckpt), step, "params")
    ema, _ = fullft_ckpt.restore_tree_on_host(str(ckpt), step, "ema")
    cpu = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])

    def ab(t):
        return jax.tree.map(lambda a: jax.ShapeDtypeStruct(
            a.shape, a.dtype, sharding=cpu), t)
    mgr = ocp.CheckpointManager(
        os.path.abspath(ckpt),
        options=ocp.CheckpointManagerOptions(read_only=True, create=False),
        item_handlers=fullft_ckpt._handlers())
    try:
        st = fullft_ckpt.restore_state(mgr, step, {
            "params": ab(params0), "ema": ab(params0),
            "opt_state": ab(jax.eval_shape(tx.init, params0)),
            "step": step})
    finally:
        mgr.close()
    return params, ema, st["opt_state"]


def _flat(tree):
    import jax

    return {mc.path_keys(p): np.asarray(x)
            for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]}


# ------------------------------------------------------------ partial restore
def test_partial_restore_matches_full(tmp_path):
    import optax

    for layout in ("unroll", "scan"):
        ckpt, _, tx, params0 = make_state(tmp_path, layout)
        params, ema, opt = full_restore(ckpt, tx, params0)
        adam = opt[1][0]
        assert isinstance(adam, optax.ScaleByAdamState), type(adam)
        want = {"params": _flat(params), "ema": _flat(ema),
                "mu": _flat(adam.mu), "nu": _flat(adam.nu)}
        st = mc.TrainState(*item_dirs(ckpt))
        assert st.adam == ("1", "0"), st.adam
        assert st.adam_count() == STEPS == int(adam.count)
        counts = st.opt_scalars()
        assert counts == {"1.0.count": STEPS, "1.2.count": STEPS}, counts
        groups = mc.plan_groups(st.leaves)
        seen = [k for ks in groups.values() for k in ks]
        assert sorted(seen) == sorted(st.leaves) and len(seen) == len(
            set(seen)), "groups must cover every leaf exactly once"
        if layout == "unroll":
            assert {f"blocks.{i:03d}" for i in range(3)} <= set(groups)
        else:
            assert "blocks:attn.wq.weight" in groups
            assert groups["blocks:attn.wq.weight"] == [
                ("blocks", "attn", "wq", "weight")]
        n = 0
        for g, keys in groups.items():
            got = st.restore_group(keys)
            for tree in ("params", "ema", "mu", "nu"):
                assert set(got[tree]) == set(keys), (g, tree)
                for k in keys:
                    x, y = got[tree][k], want[tree][k]
                    assert x.dtype == y.dtype == np.float32
                    assert x.shape == y.shape, (k, x.shape, y.shape)
                    assert np.array_equal(x, y), (layout, tree, k)
                    n += 1
        assert n == 4 * len(st.leaves)
        # the state moved (else equality would be trivial) and differs per tree
        k = ("first", "weight")
        assert not np.array_equal(want["params"][k], want["ema"][k])
        assert np.abs(want["mu"][k]).max() > 0
        print(f"    {layout}: {len(groups)} groups, {len(st.leaves)} leaves x "
              "4 trees partial == full restore (exact)")


MEM_SCRIPT = textwrap.dedent("""
    import os, resource, sys
    os.environ["JAX_PLATFORMS"] = "cpu"
    mode, d = sys.argv[2], sys.argv[3]
    if mode == "save":   # saved from an 8-device FSDP mesh, like train_full
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/scripts")
    import numpy as np
    N = 24 * 2**20   # float32 elements per leaf (96 MiB)
    if mode == "save":
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P
        from krea2_jax import fullft_ckpt
        sh = NamedSharding(jax.make_mesh((8,), ("fsdp",)), P("fsdp"))
        tree = {k: jax.device_put(np.full((4096, N // 4096), i + 1,
                                          np.float32), sh)
                for i, k in enumerate("abcd")}
        tree["s"] = np.ones(8, np.float32)
        mgr = fullft_ckpt.make_manager(d)
        fullft_ckpt.save_state(mgr, 1, {"params": tree, "ema": tree,
                                        "opt_state": {"z": np.zeros(1)}})
        fullft_ckpt.wait(mgr); mgr.close(); sys.exit(0)
    import modal_coherence as mc
    item = d + "/1/params"
    meta = mc.tree_metadata(item)
    def status(key):   # bytes
        with open("/proc/self/status") as f:
            return 1024 * int(next(l for l in f if l.startswith(key))
                              .split()[1])
    mc.restore_leaves(item, [("s",)], meta)    # warm-up (thread pools ...)
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")                           # reset the peak RSS (VmHWM)
    r0 = status("VmRSS")
    keys = [("a",)] if mode == "one" else [(k,) for k in "abcd"]
    got = mc.restore_leaves(item, keys, meta)
    assert all(float(v.flat[0]) == "abcd".index(k[0]) + 1 and
               float(v.flat[-1]) == "abcd".index(k[0]) + 1
               for k, v in got.items())
    held = status("VmRSS") - r0                # the restored arrays alive
    peak = status("VmHWM") - r0                # peak during the restore
    print(held, peak)
""")


def test_partial_restore_memory(tmp_path):
    script = tmp_path / "mem.py"
    script.write_text(MEM_SCRIPT)
    d = str(tmp_path / "ckpt")

    def run(mode):
        p = subprocess.run([sys.executable, str(script), str(REPO), mode, d],
                           capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, p.stderr[-3000:]
        return [int(x) for x in p.stdout.split()[-2:]] if mode != "save" \
            else None
    run("save")
    leaf = 96 * 2**20
    one, four = run("one"), run("four")
    print(f"    RSS growth (held / peak during the restore): 1 of 4 leaves "
          f"{[x // 2**20 for x in one]} MiB, all 4 "
          f"{[x // 2**20 for x in four]} MiB (leaf {leaf // 2**20} MiB)")
    # (+ ~50-100 MiB of tensorstore cache / allocator slack on top)
    assert four[0] >= 3.5 * leaf, four
    assert leaf <= one[0] <= leaf + 2**27 and one[1] <= leaf + 2**27, one
    assert four[1] <= 4 * leaf + 2**27, four   # no transient array copies


# ------------------------------------------------------------ metrics
def test_chunk_metrics_known(tmp_path=None):
    f4 = np.float32
    # bf16 grid at 1.0: ulp 2^-7 above, 2^-8 below
    b = np.array([1, 1, 1, 1, 1, 1, 2], f4)
    p = np.array([1 + 0.003, 1 + 0.005, 1 - 0.001, 1, 1 + 2**-8,
                  1 + 3 * 2**-8, 2], f4)
    #    lost      kept      lost       D=0  tie->1.0 (lost) tie->1+2^-6, D=0
    e = b.copy()
    e[1] = 1 + 0.02          # one EMA change, survives bf16
    m = np.array([-1, -2, 1, 5, -1, 1, 0], f4)
    v = np.ones(7, f4)
    s = mc.chunk_sums(p, e, m, v, b, 1.0, 1.0, 0.0)
    d = p.astype(np.float64) - b
    u = m.astype(np.float64)
    assert s["n"] == 7 and s["nz"] == 5 and s["nz_e"] == 1
    assert math.isclose(s["dd"], float(d @ d), rel_tol=1e-12)
    assert math.isclose(s["ud"], float(-u @ d), rel_tol=1e-12)
    # sign(-u) == sign(D) where D != 0: elements 0, 1, 2, 4 agree; 5 not
    assert s["agree"] == 4, s
    pb = np.array([1, 1 + 2**-7, 1, 1, 1, 1 + 2**-6, 2]) - b
    assert s["lost_p"] == 3 and s["chg_p"] == 2, s
    assert math.isclose(s["pbd2"], float(pb @ pb), rel_tol=1e-12)
    assert math.isclose(s["pbd"], float(pb @ d), rel_tol=1e-12)
    assert s["lost_e"] == 0 and s["chg_e"] == 1
    sched = {"lr_sum": 2.0, "lr_l2": 0.5}
    o = mc.derive(s, sched)
    assert math.isclose(o["cos_u"], float(-u @ d) / math.sqrt(
        float(u @ u) * float(d @ d)), rel_tol=1e-12)
    assert math.isclose(o["sign_agree"], 4 / 5)
    assert math.isclose(o["bf16_params"]["lost_frac"], 3 / 5)
    assert math.isclose(o["rms_over_lrsum"], math.sqrt(float(d @ d) / 7) / 2)
    assert math.isclose(o["rms_over_lrl2"], math.sqrt(float(d @ d) / 7) / .5)
    # exact directions: u = -D -> cos 1 / sign 1; u = D -> -1 / 0
    d2 = np.array([1e-3, -2e-3, 5e-4], f4)
    b2 = np.array([0.5, 0.25, -1.0], f4)
    for sign, cos, agree in ((-1, 1.0, 1.0), (1, -1.0, 0.0)):
        mm = (sign * (b2 + d2 - b2)).astype(f4)   # the f32 D itself
        oo = mc.derive(mc.chunk_sums(b2 + d2, b2, mm, np.ones(3, f4), b2,
                                     1.0, 1.0, 0.0), sched)
        assert math.isclose(oo["cos_u"], cos, rel_tol=1e-12), oo["cos_u"]
        assert math.isclose(oo["cos_m"], cos, rel_tol=1e-12)
        assert oo["sign_agree"] == agree
        assert oo["ema_ratio"] == 0.0 and oo["cos_ema"] is None
    # bias correction + eps: u = (m c1) / (sqrt(v c2) + eps)
    sb = mc.chunk_sums(np.array([1.5], f4), np.array([1.0], f4),
                       np.array([0.2], f4), np.array([0.09], f4),
                       np.array([1.0], f4), 10.0, 4.0, 0.5)
    uu = (np.float64(f4(0.2)) * 10) / (math.sqrt(np.float64(f4(0.09)) * 4)
                                       + 0.5)
    assert math.isclose(sb["uu"], uu * uu, rel_tol=1e-12)
    # undefined -> None
    z = mc.derive(mc.chunk_sums(b, b, m, v, b, 1.0, 1.0, 0.0), sched)
    assert z["cos_u"] is None and z["sign_agree"] is None
    assert z["drift_norm"] == 0.0 and z["bf16_params"]["lost_frac"] is None
    # chunked + threaded == one chunk
    rng = np.random.default_rng(1)
    arrs = [rng.standard_normal(1000).astype(f4) for _ in range(5)]
    arrs[3] = np.abs(arrs[3])
    one = mc.tensor_sums(*arrs, 1.1, 1.3, 1e-8, chunk=10**6)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(3) as pool:
        many = mc.tensor_sums(*arrs, 1.1, 1.3, 1e-8, pool=pool, chunk=37)
    for k in mc.SUM_KEYS:
        assert math.isclose(one[k], many[k], rel_tol=1e-9, abs_tol=1e-12), k
    print("    metrics: cos / sign / bias correction / bf16 RNE rounding-"
          "away (ties) / chunked sums")


def _direct(ckpt, tx, params0, base_path, T):
    """Global + per-kind observables computed straight from the full
    restore (unrolled -> official keys via save layout)."""
    import ml_dtypes
    from safetensors.numpy import load_file

    from krea2_jax.checkpoint import _official_entries

    params, ema, opt = full_restore(ckpt, tx, params0)
    adam = opt[1][0]
    base = load_file(str(base_path))
    b1, b2, eps = RUN_CFG["beta1"], RUN_CFG["beta2"], RUN_CFG["eps"]
    trees = {}
    for name, tree in (("p", params), ("e", ema), ("m", adam.mu),
                       ("v", adam.nu)):
        entries, leaves = _official_entries(tree)
        trees[name] = {
            key: np.asarray(leaves[li] if layer is None else
                            leaves[li][layer]).astype(np.float64).ravel()
            for key, li, layer, _ in entries}
    out = {}
    for scope in ["global"] + list(mxf.KINDS):
        keys = sorted(k for k in base if scope == "global"
                      or mxf.tensor_kind(k) == scope)
        if not keys:
            continue
        cat = {n: np.concatenate([trees[n][k] for k in keys]) for n in trees}
        bb = np.concatenate([base[k].astype(np.float64).ravel()
                             for k in keys])
        d, de = cat["p"] - bb, cat["e"] - bb
        u = (cat["m"] / (1 - b1 ** T)) / (np.sqrt(cat["v"] / (1 - b2 ** T))
                                          + eps)
        pb = cat["p"].astype(np.float32).astype(ml_dtypes.bfloat16).astype(
            np.float64) - bb
        out[scope] = {
            "cos_u": float(-u @ d / np.sqrt((u @ u) * (d @ d))),
            "cos_m": float(-cat["m"] @ d / np.sqrt((cat["m"] @ cat["m"])
                                                    * (d @ d))),
            "ema_ratio": float(np.sqrt((de @ de) / (d @ d))),
            "cos_ema": float(de @ d / np.sqrt((de @ de) * (d @ d))),
            "drift_rel": float(np.sqrt((d @ d) / (bb @ bb))),
            "sign_agree": float(np.mean((np.sign(-u) == np.sign(d))[d != 0])),
            "lost_p": float(np.mean((pb == 0)[d != 0])),
        }
    return out


def test_analyze(tmp_path):
    reps = {}
    for layout in ("unroll", "scan"):
        ckpt, base_path, tx, params0 = make_state(tmp_path, layout)
        st = mc.TrainState(*item_dirs(ckpt))
        T = st.adam_count()
        rep = mc.analyze(st, base_path, RUN_CFG, T, workers=3,
                         log=lambda *_: None)
        json.dumps(rep)
        reps[layout] = rep
        want = _direct(ckpt, tx, params0, base_path, T)
        for scope, w in want.items():
            got = rep["global"] if scope == "global" else rep["by_kind"][scope]
            for k, x in w.items():
                y = (got["bf16_params"]["lost_frac"] if k == "lost_p"
                     else got[k])
                assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12), (
                    layout, scope, k, x, y)
        g = rep["global"]
        assert g["cos_u"] > 0.5, g["cos_u"]   # coherent synthetic gradients
        sched = mc.schedule_sums(RUN_CFG, T)
        assert rep["schedule"] == sched
        assert math.isclose(sched["lr_sum"], 0 + 5e-4 + 1e-3 + 1e-3)
        assert math.isclose(g["rms_over_lrsum"], g["drift_rms"]
                            / sched["lr_sum"])
        assert set(rep["by_block"]) == {"block00", "block01", "block02",
                                        "txtfusion", "other"}
        assert set(rep["by_kind"]) == set(mxf.KINDS)
    # identical values in both layouts -> identical per-tensor sums
    tu, ts = reps["unroll"]["tensors"], reps["scan"]["tensors"]
    assert sorted(tu) == sorted(ts) and len(tu) > 50
    for key in tu:
        for k in mc.SUM_KEYS:
            assert math.isclose(tu[key][k], ts[key][k], rel_tol=1e-12,
                                abs_tol=1e-30), (key, k)
    # a group's arrays are all released before the next group is restored
    # (weakref probe on every array restore_group hands out)
    import weakref

    class Probe:
        def __init__(self, st):
            self.st, self.leaves = st, st.leaves
            self.params_tree = st.params_tree
            self.prev, self.calls, self.alive = [], 0, []

        def restore_group(self, keys):
            # no gc.collect(): the arrays must go by refcount alone
            self.alive.append(sum(r() is not None for r in self.prev))
            got = self.st.restore_group(keys)
            self.prev = [weakref.ref(a) for t in got.values()
                         for a in t.values()]
            self.calls += 1
            return got
    probe = Probe(mc.TrainState(*item_dirs(tmp_path / "ckpt_scan")))
    rep = mc.analyze(probe, _base(tmp_path), RUN_CFG, STEPS, workers=2,
                     log=lambda *_: None)
    assert probe.calls == len(mc.plan_groups(probe.leaves)) > 1
    assert probe.alive == [0] * probe.calls, probe.alive
    assert rep["tensors"] == reps["scan"]["tensors"]
    # memory budget refusal (before anything is read)
    st = mc.TrainState(*item_dirs(tmp_path / "ckpt_scan"))
    try:
        mc.analyze(st, _base(tmp_path), RUN_CFG, STEPS, mem_budget_gib=1e-3,
                   log=lambda *_: None)
    except RuntimeError as e:
        assert "budget" in str(e), e
    else:
        raise AssertionError("memory budget not enforced")
    print(f"    analyze == direct numpy (global + {len(mxf.KINDS)} kinds), "
          f"scan == unroll per tensor; global cos(-u,D) "
          f"{reps['unroll']['global']['cos_u']:.3f}")


# ------------------------------------------------------------ schedule / sim
def test_schedule_matches_train_full(tmp_path=None):
    import train_full

    for cfg in (RUN_CFG,
                {**RUN_CFG, "lr": 1e-5, "warmup_steps": 150, "steps": 1513},
                {**RUN_CFG, "warmup_steps": 0},
                {**RUN_CFG, "cosine_decay": True, "min_lr": 1e-4,
                 "steps": 30, "warmup_steps": 5},
                {**RUN_CFG, "cosine_decay": True, "min_lr": 0.0, "steps": 3,
                 "warmup_steps": 5},
                {**RUN_CFG, "cosine_decay": True, "warmup_steps": 0,
                 "steps": 20}):
        T = 60 if cfg["steps"] < 100 else 1600
        sched = train_full.make_schedule(SimpleNamespace(**cfg))
        want = np.array([float(sched(c)) for c in range(T)])
        got = mc.lr_schedule(cfg, T)
        assert np.allclose(got, want, rtol=2e-6, atol=1e-12), (
            cfg, np.abs(got - want).max())
    run1 = mc.lr_schedule(mc.RUN1_SIM, 1513)
    assert run1[0] == 0.0 and math.isclose(run1[1], 1e-5 / 150)
    assert math.isclose(run1[149], 1e-5 * 149 / 150) and run1[150] == 1e-5
    s = mc.schedule_sums(mc.RUN1_SIM, 1513)
    assert math.isclose(s["lr_sum"], 1e-5 * (sum(range(150)) / 150 + 1363))
    print("    lr schedule == train_full.make_schedule (optax), 6 configs")


def test_sim_adam_matches_optax(tmp_path=None):
    import jax
    import jax.numpy as jnp
    import optax

    cfg = {**mc.RUN1_SIM, "lr": 1e-3, "warmup_steps": 3}
    T, n, snrs = 12, 5, (0.0, 0.5)
    lrs = mc.lr_schedule(cfg, T)
    rows = mc.simulate(snrs, lrs, b1=0.9, b2=0.999, eps=0.0, ema_decay=0.9,
                       n=n, seed=3)
    # the same gradient draws through optax.adam (no weight decay, eps 0)
    import train_full

    tx = optax.adam(train_full.make_schedule(SimpleNamespace(
        **{**cfg, "min_lr": 0.0})), b1=0.9, b2=0.999, eps=0.0)
    mu = np.repeat(np.asarray(snrs), n)
    rng = np.random.default_rng(3)
    p = jnp.zeros(mu.size, jnp.float32)
    e = np.zeros(mu.size)
    st = tx.init(p)
    for t in range(T):
        g = jnp.asarray((mu + rng.standard_normal(mu.size)).astype(np.float32))
        upd, st = tx.update(g, st, p)
        p = optax.apply_updates(p, upd)
        e += 0.1 * (np.asarray(p, np.float64) - e)
    p = np.asarray(p, np.float64)
    adam = st[0]
    for i, r in enumerate(rows):
        sl = slice(i * n, (i + 1) * n)
        u = (np.asarray(adam.mu, np.float64)[sl] / (1 - 0.9 ** T)) / np.sqrt(
            np.asarray(adam.nu, np.float64)[sl] / (1 - 0.999 ** T))
        d = p[sl]
        cos = float(-u @ d / np.sqrt((u @ u) * (d @ d)))
        assert math.isclose(r["cos_u"], cos, rel_tol=1e-4), (r["cos_u"], cos)
        ratio = math.sqrt(float(e[sl] @ e[sl]) / float(d @ d))
        assert math.isclose(r["ema_ratio"], ratio, rel_tol=1e-4)
        rms = math.sqrt(float(d @ d) / n)
        assert math.isclose(r["rms_over_lrsum"], rms / lrs.sum(),
                            rel_tol=1e-4)
    del jax
    print("    simulate() Adam / EMA == optax.adam on the same draws")


def test_sim_limits(tmp_path=None):
    lrs = mc.lr_schedule(mc.RUN1_SIM, 1513)
    rows = mc.simulate((0.0,), lrs, n=40000, seed=1)
    r0 = rows[0]
    t_eff = float(lrs @ lrs) / lrs[-1] ** 2
    analytic = 10 / math.sqrt(19 * t_eff)
    assert math.isclose(mc.null_cos_u(lrs), analytic, rel_tol=1e-12)
    print(f"    s=0: cos(-u,D) {r0['cos_u']:.4f} (analytic {analytic:.4f}), "
          f"|De|/|D| {r0['ema_ratio']:.4f}, sign {r0['sign_agree']:.4f}, "
          f"coherent {r0['coherent_frac']:.2e}")
    assert abs(r0["cos_u"] - analytic) < 0.25 * analytic, r0["cos_u"]
    assert abs(r0["ema_ratio"] - 0.51) < 0.02, r0["ema_ratio"]
    assert r0["coherent_frac"] < 0.005 and abs(r0["sign_agree"] - 0.5) < 0.04
    big = mc.simulate((10.0,), lrs, n=2000, seed=2)[0]
    print(f"    s=10: rms/lr_sum {big['rms_over_lrsum']:.4f}, cos "
          f"{big['cos_u']:.5f}, coherent {big['coherent_frac']:.5f}")
    assert 0.98 < big["rms_over_lrsum"] <= 1.0 + 1e-9
    assert big["cos_u"] > 0.999 and big["sign_agree"] == 1.0
    assert big["coherent_frac"] > 0.999
    # inversion on a monotone grid
    grid = [{"s": 0.0, "cos_u": 0.05, "ema_ratio": 0.51,
             "coherent_frac": 0.0},
            {"s": 0.1, "cos_u": 0.4, "ema_ratio": 0.47,
             "coherent_frac": 0.9},
            {"s": 1.0, "cos_u": 0.97, "ema_ratio": 0.47,
             "coherent_frac": 1.0}]
    r = mc.invert(grid, "cos_u", 0.225)
    assert r["range"] == "inside" and math.isclose(r["s"], 0.05)
    assert math.isclose(r["coherent_frac"], 0.45)
    r = mc.invert(grid, "ema_ratio", 0.49)
    assert math.isclose(r["s"], 0.05) and r["range"] == "inside"
    r = mc.invert(grid, "cos_u", 0.4)   # on a grid point: one solution
    assert r["range"] == "inside" and math.isclose(r["s"], 0.1)
    # ambiguous: the flat ||De||/||D|| tail equals x for every s in it
    r = mc.invert(grid, "ema_ratio", 0.47)
    assert r["range"] == "ambiguous" and r["s"] is None, r
    assert (r["s_min"], r["s_max"]) == (0.1, 1.0)
    assert (r["coherent_frac_min"], r["coherent_frac_max"]) == (0.9, 1.0)
    # ambiguous: the real (non-monotone) curve of the run1 --sim-only table;
    # the s = 1.0 row's own ||De||/||D|| is also reached near s = 0.45
    sim = [(0.0, 0.0545, 0.5148, 0.0), (0.003, 0.0668, 0.5112, 0.0122),
           (0.01, 0.0733, 0.5056, 0.1314), (0.02, 0.0977, 0.4987, 0.3690),
           (0.03, 0.1342, 0.4903, 0.5644), (0.05, 0.2162, 0.4790, 0.7884),
           (0.1, 0.4055, 0.4724, 0.9377), (0.2, 0.6605, 0.4707, 0.9838),
           (0.5, 0.9113, 0.4701, 0.9979), (1.0, 0.9748, 0.4702, 0.9996)]
    rows = [{"s": s, "cos_u": c, "ema_ratio": er, "coherent_frac": f}
            for s, c, er, f in sim]
    r = mc.invert(rows, "ema_ratio", 0.4702)
    assert r["range"] == "ambiguous" and r["s"] is None, r
    assert len(r["candidates"]) == 2 and r["s_max"] == 1.0
    assert 0.2 < r["s_min"] < 0.5, r
    assert mc.invert(rows, "ema_ratio", 0.49)["range"] == "inside"
    r = mc.invert(rows, "cos_u", 0.9748)    # monotone metric: unique
    assert r["range"] == "inside" and r["s"] == 1.0
    table = mc.implied_table({"note": "n", "global": {
        "cos_u": mc.invert(rows, "cos_u", 0.3),
        "ema_ratio": mc.invert(rows, "ema_ratio", 0.4702)}})
    assert "?0.4" in table and "-1.0000" in table, table
    assert mc.invert(grid, "cos_u", 0.01)["range"] == "below"
    assert mc.invert(grid, "cos_u", 0.01)["s"] == 0.0
    assert mc.invert(grid, "ema_ratio", 0.6)["range"] == "above"
    assert mc.invert(grid, "cos_u", None) is None
    imp = mc.implied({"global": {"cos_u": 0.4, "ema_ratio": 0.47},
                      "by_kind": {"mlp": {"cos_u": 0.05,
                                          "ema_ratio": None}}}, grid)
    assert math.isclose(imp["global"]["cos_u"]["s"], 0.1)
    assert imp["mlp"]["ema_ratio"] is None and "approximate" in imp["note"]
    print("    sim limits + inversion")


# ------------------------------------------------------------ flow
def test_coherence_impl(tmp_path):
    ckpt, base_path, tx, params0 = make_state(tmp_path, "unroll")
    s3 = FakeS3(page=5)
    pre = mxf.step_prefix("krea2-fullft", "run1", STEPS)
    s3.put_dir(ckpt / str(STEPS), pre)
    vol = tmp_path / "vol"
    (vol / "base").mkdir(parents=True)
    (vol / mxf.BASE_REL).write_bytes(base_path.read_bytes())
    # selection: opt_state only with trees=CKPT_TREES; the default refuses
    objs = mxf.list_objects(s3, "bkt", pre)
    sel = mxf.select_ckpt_objects(objs, pre, "opt_state", trees=mc.CKPT_TREES)
    assert sel and all(r.split("/")[0] in ("opt_state", "meta")
                       or "/" not in r for r in sel), sorted(sel)
    assert any(r.startswith("opt_state/") for r in sel)
    try:
        mxf.select_ckpt_objects(objs, pre, "opt_state")
    except ValueError as e:
        assert "unknown tree" in str(e)
    else:
        raise AssertionError("opt_state accepted by the export selection")
    # the params tree is already on the volume (an earlier export)
    lp = mxf.local_paths(vol, "run1", STEPS, "params")
    mxf.download_ckpt_tree(s3, "bkt", pre, lp["step_dir"], "params",
                           lp["marker"], transfer_config="cfg",
                           log=lambda *_: None)
    n0 = len(s3.downloads)
    kw = dict(env={"R2_BUCKET": "bkt"}, vol=vol, client=s3,
              transfer_config="cfg", workers=2, sim_n=200,
              log=lambda *_: None)
    rep = mc.coherence_impl("run1", STEPS, **kw)
    new = s3.downloads[n0:]
    assert new and not any("/params/" in k for k in new), new
    assert any("/opt_state/" in k for k in new)
    assert any("/ema/" in k for k in new)
    assert rep["downloads"]["params"]["downloaded"] == 0
    paths = mc.coherence_paths(vol, "run1", STEPS)
    for t in mc.CKPT_TREES:   # one orbax dir per tree
        others = set(mc.CKPT_TREES) - {t}
        assert (paths[t]["step_dir"] / t).is_dir()
        assert not any((paths[t]["step_dir"] / o).exists() for o in others)
    assert rep["adam"]["T"] == STEPS and rep["meta_step"] == STEPS
    assert rep["config"]["layer_loop"] == "unroll"
    assert "warning" not in rep
    saved = json.loads(paths["report"].read_text())
    assert saved["global"] == rep["global"]
    assert "bkt" not in json.dumps(rep)
    direct = _direct(ckpt, tx, params0, base_path, STEPS)
    assert math.isclose(rep["global"]["cos_u"], direct["global"]["cos_u"],
                        rel_tol=1e-9)
    # re-run: nothing downloaded
    n1 = len(s3.downloads)
    rep2 = mc.coherence_impl("run1", STEPS, **kw)
    assert len(s3.downloads) == n1
    assert rep2["ckpt_fingerprint"] == rep["ckpt_fingerprint"]
    # simulation with the run's own settings + implied values
    assert rep["sim"]["config"]["T"] == STEPS and rep["sim"]["config"]["n"] \
        == 200
    assert rep["sim"]["config"]["lr"] == RUN_CFG["lr"]
    assert rep["sim"]["config"]["ema_decay"] == RUN_CFG["ema_decay"]
    assert len(rep["sim"]["rows"]) == len(mc.SIM_SNRS)
    assert set(rep["implied"]) >= {"global", "note", "mlp"}
    assert saved["sim"] == rep["sim"] and saved["implied"] == rep["implied"]
    again = mc.add_simulation(rep, sim_n=200, sim_seed=0)   # deterministic
    assert again["sim"] == rep["sim"]
    for table in (mc.report_table(rep), mc.sim_table(rep["sim"]),
                  mc.implied_table(rep["implied"])):
        assert table and "None" not in table, table
    assert "block00" in mc.report_table(rep)
    out = mc.write_local(rep, str(tmp_path / "c.json"))
    assert json.loads(out.read_text())["run"] == "run1"
    # refusals: no base on the volume; a step saved without EMA (before
    # any transfer)
    try:
        mc.coherence_impl("run1", STEPS, **{**kw, "vol": tmp_path / "empty"})
    except RuntimeError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("missing base not refused")
    for k in [k for k in s3.objects if k.startswith(pre)
              and not k.startswith(pre + "ema/")]:
        s3.put(mxf.step_prefix("krea2-fullft", "run1", 9) + k[len(pre):],
               s3.objects[k])
    n2 = len(s3.downloads)
    try:
        mc.coherence_impl("run1", 9, **kw)
    except ValueError as e:
        assert "no ema/ objects" in str(e), e
    else:
        raise AssertionError("a step without EMA not refused")
    assert len(s3.downloads) == n2
    print("    coherence_impl: fake R2, params reused, ema + opt_state "
          "downloaded, idempotent, report + sim, refusals")


def test_cli(tmp_path=None):
    import contextlib
    import io

    a = mc.parse_args(["--step", "1513"])
    assert (a.run, a.step, a.prefix, a.sim_only, a.sim_n) == (
        "run1", 1513, "krea2-fullft", False, mc.SIM_N)
    a = mc.parse_args(["--sim-only"])
    assert a.step is None and a.lr == 1e-5 and a.warmup_steps == 150
    for argv in ([], ["--step", "-1"], ["--step", "1", "--run", "../x"],
                 ["--step", "1", "--prefix", "a//b"],
                 ["--sim-only", "--sim-n", "0"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                mc.parse_args(argv)
        except SystemExit as e:
            assert e.code == 2, (argv, e.code)
        else:
            raise AssertionError(f"{argv} accepted")
    try:
        mc.local_main(["--step", "3"])
    except SystemExit as e:
        assert "modal run" in str(e)
    else:
        raise AssertionError("local_main did not exit")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        sim = mc.local_main(["--sim-only", "--step", "40", "--sim-n", "50",
                             "--warmup-steps", "5"])
    assert sim["config"]["T"] == 40 and sim["config"]["warmup_steps"] == 5
    assert "cos(-u,D)" in buf.getvalue()
    p = subprocess.run([sys.executable, str(REPO / "scripts" /
                                            "modal_coherence.py"), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "--sim-only" in p.stdout, p.stderr
    print("    CLI: defaults, validation, --sim-only, --help")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
    print(f"all {len(tests)} tests passed")
