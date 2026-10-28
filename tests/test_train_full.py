"""scripts/train_full.py: train-step units (loss masking, padding, mu, opt-state
sharding guard, resume-config checks) and end-to-end runs on the tiny DiT /
fake bucketed cache (loss, export, resume, refusals, SIGTERM, layer-loop)."""

import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import train_full
from helpers import REPO, STEP_RE, TINY, run_train, train_cmd

REFUSED = "different run configuration"


# --- units --------------------------------------------------------------------

@pytest.fixture(scope="module")
def state(tiny_ckpt, mesh):
    import jax
    import jax.numpy as jnp
    from krea2_jax.checkpoint import load_krea2_params, param_shardings
    params = load_krea2_params(str(tiny_ckpt[1]), mesh=mesh, dtype=jnp.float32)
    pshard = param_shardings(params, mesh)
    a = SimpleNamespace(lr=1e-3, warmup_steps=2, cosine_decay=False, min_lr=0.0,
                        steps=12, grad_clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8,
                        weight_decay=0.01)
    tx = train_full.make_optimizer(a, train_full.make_schedule(a))
    oshard = train_full.opt_state_shardings(jax.eval_shape(tx.init, params),
                                            params, mesh, pshard)
    opt = jax.jit(tx.init, in_shardings=(pshard,), out_shardings=oshard)(params)
    return SimpleNamespace(params=params, pshard=pshard, tx=tx, oshard=oshard, opt=opt)


def test_opt_state_sharding_guard(state, mesh):
    import jax
    from krea2_jax.checkpoint import replicate_tree
    check = train_full.check_opt_state_sharding
    assert check(state.opt, state.params) == 2 * len(jax.tree.leaves(state.params))
    leaves, tdef = jax.tree.flatten(state.opt)
    i = next(i for i, a in enumerate(leaves)
             if a.ndim and not a.sharding.is_fully_replicated)
    leaves[i] = replicate_tree(leaves[i], mesh)
    for bad in (replicate_tree(state.opt, mesh), jax.tree.unflatten(tdef, leaves)):
        with pytest.raises(RuntimeError, match="not sharded like the params"):
            check(bad, state.params)


def test_masked_flow_loss():
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(5)
    v, tg = rng.standard_normal((2, 3, 6, 4)).astype(np.float32)
    n_img = np.asarray([6, 4, 1], np.int32)
    exp = np.mean([np.mean((v[b, :n] - tg[b, :n]) ** 2) for b, n in enumerate(n_img)])
    got = float(train_full.masked_flow_loss(jnp.asarray(v), jnp.asarray(tg),
                                            jnp.asarray(n_img)))
    assert abs(got - exp) <= 1e-6 * exp
    lg = jax.jit(jax.value_and_grad(lambda x: train_full.masked_flow_loss(
        x, jnp.zeros_like(x), jnp.asarray([1], jnp.int32))))
    for bad in (np.inf, np.nan):
        loss, grad = lg(jnp.asarray([[[2.0], [bad]]], jnp.float32))
        assert float(loss) == 4.0 and np.array_equal(np.asarray(grad), [[[4.0], [0.0]]])


def test_per_sample_mu():
    import jax.numpy as jnp
    from krea2_jax.config import Krea2Config
    cfg = Krea2Config(**TINY)
    n = jnp.asarray([12, 16, 4096], jnp.int32)
    mu = np.asarray(train_full.per_sample_mu(cfg, n))
    assert mu.dtype == np.float32
    np.testing.assert_allclose(mu, [cfg.mu_for_seq_len(k) for k in (12, 16, 4096)],
                               rtol=1e-6)
    assert np.array_equal(np.asarray(train_full.per_sample_mu(cfg, n, 0.8)),
                          np.full(3, 0.8, np.float32))


def test_padded_rows_inert(state, mesh, text_encoder):
    """Padded image rows change nothing (bitwise); a valid row does."""
    import jax
    import jax.numpy as jnp
    from krea2_jax.checkpoint import batch_sharding
    from krea2_jax.config import Krea2Config
    from krea2_jax.qwen3_text import load_qwen3_text_params, load_tokenizer
    from krea2_jax.text_template import CaptionTemplate
    te, txt_cfg = text_encoder
    cfg, N, B = Krea2Config(**TINY), 16, 8
    template = CaptionTemplate(load_tokenizer(te), text_rows=32)
    txt_params = load_qwen3_text_params(te, jnp.bfloat16, mesh=mesh, cfg=txt_cfg)
    hp = train_full.parse_args(["--base-ckpt", "-", "--data-dir", "-",
                                "--text-encoder", "-", "--output-dir", "-",
                                "--qchunk", "256", "--ema-decay", "0"])
    rng = np.random.default_rng(0)
    hw = np.asarray([[4, 3], [4, 4]] * (B // 2), np.int32)
    valid = (np.arange(N)[None] < (hw[:, 0] * hw[:, 1])[:, None])[..., None]
    img = rng.standard_normal((B, N, cfg.channels * cfg.patch ** 2)).astype(np.float32)
    clean = np.where(valid, img, 0.0)
    garbage = np.where(valid, img, 50.0 * rng.standard_normal(img.shape))
    changed = clean.copy()
    changed[0, 3] += 1.0
    ids, txt_len = template.encode_captions([f"caption {i}, a girl" for i in range(B)])
    ds = batch_sharding(mesh)
    copy_p = jax.jit(lambda t: jax.tree.map(jnp.copy, t), out_shardings=state.pshard)
    copy_o = jax.jit(lambda t: jax.tree.map(jnp.copy, t), out_shardings=state.oshard)
    step = train_full.make_train_step(cfg, hp, mesh, state.tx, False, state.pshard,
                                      state.oshard, txt_cfg, N)

    def run(x):
        batch = [jax.device_put(np.asarray(a, dt), ds) for a, dt in (
            (x, jnp.bfloat16), (hw, np.int32), (ids, np.int32), (txt_len, np.int32))]
        p, o, _, m = step(copy_p(state.params), copy_o(state.opt), None, txt_params,
                          *batch, jax.random.PRNGKey(3))
        return (float(m["loss"]), float(m["grad_norm"]),
                [np.asarray(t) for t in jax.tree.leaves((p, o))])

    a, g, c = run(clean), run(garbage), run(changed)
    assert np.isfinite(a[:2]).all() and a[:2] == g[:2]
    assert all(x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(a[2], g[2]))
    assert c[0] != a[0] and c[1] != a[1]


def test_positions_bucketed():
    import jax
    import jax.numpy as jnp
    from krea2_jax.model import build_positions_and_mask, build_positions_and_mask_bucketed
    rng = np.random.default_rng(0)
    B, L, N = 3, 5, 16
    tmask = jnp.asarray(np.arange(L)[None] < rng.integers(1, L + 1, size=(B, 1)))
    for h, w in ((4, 4), (2, 8), (8, 2), (1, 16)):
        hw = jnp.asarray(np.tile([[h, w]], (B, 1)), jnp.int32)
        got, exp = build_positions_and_mask_bucketed(tmask, hw, h * w), \
            build_positions_and_mask(tmask, h, w)
        for x, y in zip(got, exp):
            assert x.dtype == y.dtype and np.array_equal(x, y), (h, w)
    hw = np.asarray([[4, 3], [2, 5], [4, 4]], np.int32)
    pos, mask = map(np.asarray, jax.jit(build_positions_and_mask_bucketed,
                                        static_argnums=2)(tmask, jnp.asarray(hw), N))
    assert pos.shape == (B, L + N, 3) and pos.dtype == np.float32 and mask.dtype == bool
    assert not pos[:, :L].any() and np.array_equal(mask[:, :L], np.asarray(tmask))
    for b, (h, w) in enumerate(hw.tolist()):
        exp = np.zeros((N, 3), np.float32)
        exp[:h * w, 1], exp[:h * w, 2] = np.arange(h * w) // w, np.arange(h * w) % w
        assert np.array_equal(pos[b, L:], exp) and np.array_equal(mask[b, L:],
                                                                  np.arange(N) < h * w)


def _cfg(**kw):
    c = dict.fromkeys(train_full.RESUME_HARD_KEYS + train_full.RESUME_SOFT_KEYS, 0)
    c.update(layer_loop="scan", ema_decay=0.9, dataset_fingerprint="a" * 64,
             n_samples=20)
    return {**c, **kw}


def test_check_resume_config():
    check, saved = train_full.check_resume_config, _cfg()
    assert check(saved, _cfg(), 6) == []
    w = check(None, saved, 6)
    assert len(w) == 1 and "no saved run configuration" in w[0]
    with pytest.raises(SystemExit, match="ema enabled"):
        check(saved, _cfg(ema_decay=0.0), 6)
    assert any("ema_decay: checkpoint 0.9, now 0.99" in line
               for line in check(saved, _cfg(ema_decay=0.99), 6))
    old = {k: v for k, v in saved.items() if k != "layer_loop"}
    assert check(old, saved, 6) == []
    for s in (old, None):
        with pytest.raises(SystemExit, match="layer_loop: checkpoint 'scan', now 'unroll'"):
            check(s, _cfg(layer_loop="unroll"), 6)


def test_check_resume_config_new_dataset():
    check = train_full.check_resume_config
    saved, new = _cfg(), _cfg(dataset_fingerprint="b" * 64, n_samples=17)
    with pytest.raises(SystemExit, match="--resume-new-dataset") as e:
        check(saved, new, 3)
    assert "dataset_fingerprint: checkpoint" in str(e.value) and "n_samples: " in str(e.value)
    assert check(saved, new, 3, new_dataset=True) == []
    with pytest.raises(SystemExit, match="seed: checkpoint") as e:
        check(saved, {**new, "seed": 1}, 3, new_dataset=True)
    assert "--resume-new-dataset" not in str(e.value)
    assert "dataset_fingerprint" not in str(e.value)
    assert check(_cfg(data_step_offset=3, dataset_switches=[{"step": 3}]),
                 _cfg(data_step_offset=0, dataset_switches=[]), 9) == []


def test_resume_data_position():
    pos = train_full.resume_data_position
    a, b = _cfg(), _cfg(dataset_fingerprint="b" * 64, n_samples=17)
    assert pos(a, a, 5, False) == pos(None, a, 5, False) == (0, None)
    assert pos({**b, "data_step_offset": 3}, b, 9, False) == (3, None)
    assert pos(a, b, 1513, True) == (1513, {
        "step": 1513, "from_fingerprint": "a" * 64, "to_fingerprint": "b" * 64,
        "from_n_samples": 20, "to_n_samples": 17})
    assert pos(a, _cfg(n_samples=19), 7, True)[0] == 7
    assert pos(a, _cfg(dataset_fingerprint="c" * 64), 7, True)[0] == 7
    assert pos({**b, "data_step_offset": 1513}, b, 2000, True) == (1513, None)
    with pytest.raises(SystemExit, match="no saved run configuration"):
        pos(None, b, 5, True)
    with pytest.raises(SystemExit, match="dataset_fingerprint"):
        pos({k: v for k, v in a.items() if k != "dataset_fingerprint"}, b, 5, True)


# --- end to end -----------------------------------------------------------------

@pytest.fixture(scope="module")
def job(tiny_ckpt, bucket_cache, text_encoder):
    return tiny_ckpt[1], bucket_cache, text_encoder[0]


@pytest.fixture(scope="module")
def fresh(job, tmp_path_factory):
    out = tmp_path_factory.mktemp("fresh") / "run"
    steps, stdout, _ = run_train(*job, out)
    return SimpleNamespace(steps=steps, stdout=stdout, out=out)


def from_step6(fresh, d):
    shutil.copytree(fresh.out / "ckpt" / "6", d / "ckpt" / "6")
    return d


def tensor_shapes(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def test_fresh_run_lowers_loss(fresh):
    assert sorted(fresh.steps) == list(range(1, 13))
    losses = [fresh.steps[s][0] for s in range(1, 13)]
    assert np.isfinite(losses).all() and np.isfinite([g for _, g in fresh.steps.values()]).all()
    assert losses[-1] < losses[0] and np.mean(losses[-3:]) < np.mean(losses[:3])


def test_export(fresh, tiny_ckpt, mesh, tmp_path):
    """Official keys, bf16, metadata, sharded load, torch parity, EMA tree."""
    import jax
    import jax.numpy as jnp
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from krea2_jax.checkpoint import load_krea2_params
    from krea2_jax.config import Krea2Config
    from krea2_jax.model import build_positions_and_mask, dit_forward, patchify
    sd, path = tiny_ckpt[0], fresh.out / "model.safetensors"
    ft = load_file(str(path))
    assert set(ft) == set(sd)
    assert all(v.dtype == torch.bfloat16 and v.shape == sd[k].shape for k, v in ft.items())
    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata()
    assert meta["step"] == "12" and meta["source"] == "ema"
    sharded = load_krea2_params(str(path), mesh=mesh, dtype=jnp.float32)
    assert any(not a.sharding.is_fully_replicated for a in jax.tree.leaves(sharded))

    model = copy.deepcopy(tiny_ckpt[3])
    model.load_state_dict({k: v.float() for k, v in ft.items()})
    g = torch.Generator().manual_seed(0)
    B, ltxt, C, hh, ww = 2, 7, TINY["channels"], 8, 12
    lat = torch.randn(B, C, 2 * hh, 2 * ww, generator=g)
    ctx = torch.randn(B, ltxt, TINY["txtlayers"], TINY["txtdim"], generator=g)
    t = torch.tensor([0.3, 0.8])
    tmask = torch.ones(B, ltxt, dtype=torch.bool)
    tmask[0, 5:] = False
    img = lat.reshape(B, C, hh, 2, ww, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, hh * ww, -1)
    ids = torch.zeros(hh, ww, 3)
    ids[..., 1], ids[..., 2] = torch.arange(hh)[:, None], torch.arange(ww)[None, :]
    pos = torch.cat([torch.zeros(B, ltxt, 3), ids.reshape(1, -1, 3).expand(B, -1, -1)], 1)
    mask = torch.cat([tmask, torch.ones(B, hh * ww, dtype=torch.bool)], 1)
    with torch.no_grad():
        ref = model(img=img, context=ctx, t=t, pos=pos, mask=mask).numpy()
    jpos, jmask = build_positions_and_mask(jnp.asarray(tmask.numpy()), hh, ww)
    out = np.asarray(dit_forward(
        load_krea2_params(str(path), mesh=None, dtype=jnp.float32), Krea2Config(**TINY),
        patchify(jnp.asarray(lat.numpy()), 2), jnp.asarray(ctx.numpy()),
        jnp.asarray(t.numpy()), jpos, jmask, remat=False, qchunk=64))
    assert np.abs(out - ref).max() / np.abs(ref).max() < 2e-3

    raw_path = tmp_path / "raw.safetensors"
    subprocess.run([sys.executable, str(REPO / "scripts" / "export_full.py"),
                    "--ckpt-dir", str(fresh.out / "ckpt"), "--step", "12",
                    "--output", str(raw_path), "--source", "params"],
                   check=True, capture_output=True)
    raw = load_file(str(raw_path))
    dist = lambda x: sum(float((x[k].float() - sd[k]).norm() ** 2) for k in sd) ** 0.5
    gap = max(float((ft[k].float() - raw[k].float()).abs().max()) for k in sd)
    assert 0 < dist(ft) < dist(raw) and gap > 1e-3


def test_resume_reproduces_run(fresh, job, tmp_path):
    from safetensors.torch import load_file
    d = from_step6(fresh, tmp_path / "run")
    steps, _, _ = run_train(*job, d, extra=("--resume",))
    assert sorted(steps) == list(range(7, 13))
    assert max(abs(steps[s][0] - fresh.steps[s][0]) for s in steps) <= 1e-5
    a, b = load_file(str(d / "model.safetensors")), load_file(str(fresh.out / "model.safetensors"))
    assert max(float((a[k].float() - b[k].float()).abs().max()) for k in b) <= 1e-3


def test_rerun_without_resume_refused(fresh, job):
    _, _, err = run_train(*job, fresh.out, ok=False)
    assert "already holds checkpoint step 12" in err and "--resume" in err


@pytest.mark.parametrize("flag,val,key", [("--seed", "1", "seed"),
                                          ("--batch-size", "16", "batch_size"),
                                          ("--layer-loop", "unroll", "layer_loop")])
def test_resume_hard_key_refused(fresh, job, tmp_path, flag, val, key):
    _, out, err = run_train(*job, from_step6(fresh, tmp_path / "run"),
                            extra=("--resume", flag, val), ok=False)
    assert REFUSED in err and re.search(rf"{key}: checkpoint \S+, now '?{val}'?$", err, re.M)
    assert "resumed train state" not in out


def test_resume_edited_caption_refused(fresh, job, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cache = shutil.copytree(job[1], tmp_path / "data")
    table = pq.read_table(cache / "metadata.parquet")
    longs = table.column("long").to_pylist()
    longs[3] += " (edited)"
    table = table.set_column(table.schema.get_field_index("long"), "long",
                             pa.array(longs, type=pa.string()))
    pq.write_table(table, cache / "metadata.parquet")
    _, _, err = run_train(job[0], cache, job[2], from_step6(fresh, tmp_path / "run"),
                          extra=("--resume",), ok=False)
    assert REFUSED in err and "dataset_fingerprint: checkpoint" in err
    assert "n_samples:" not in err


def test_resume_modified_tokenizer_refused(fresh, job, tmp_path):
    te = shutil.copytree(job[2], tmp_path / "te" / job[2].name)
    tj = json.loads((te / "tokenizer.json").read_text())
    (te / "tokenizer.json").write_text(json.dumps(tj, indent=1))
    _, _, err = run_train(*job, from_step6(fresh, tmp_path / "run"),
                          extra=("--resume", "--text-encoder", str(te)), ok=False)
    assert REFUSED in err and "text_encoder_id: checkpoint" in err
    assert "dataset_fingerprint" not in err


def test_resume_lr_change_warns(fresh, job, tmp_path):
    steps, out, _ = run_train(*job, from_step6(fresh, tmp_path / "run"),
                              extra=("--resume", "--lr", "2e-3", "--steps", "7"))
    assert "lr: checkpoint 0.001, now 0.002" in out and sorted(steps) == [7]


def test_sigterm_stops_then_resumes(fresh, job, tmp_path):
    out = tmp_path / "run"
    proc = subprocess.Popen(
        train_cmd(*job, out, extra=("--steps", "40", "--save-every", "100")),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines = []

    def reader():
        for line in proc.stdout:
            lines.append(line)
            m = STEP_RE.match(line)
            if m and int(m.group(1)) == 3:
                proc.send_signal(signal.SIGTERM)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        rc = proc.wait(timeout=600)
    finally:
        proc.kill()
        th.join(30)
    text = "".join(lines)
    assert rc == 0, text[-4000:]
    ckpts = [int(p.name) for p in (out / "ckpt").iterdir() if p.name.isdigit()]
    logged = [int(s) for s, *_ in STEP_RE.findall(text)]
    assert len(ckpts) == 1 and 3 <= ckpts[0] == max(logged) < 12, (ckpts, text[-3000:])
    stop = ckpts[0]
    from safetensors import safe_open
    with safe_open(str(out / "model.safetensors"), framework="numpy") as f:
        assert f.metadata()["step"] == str(stop)
    steps, _, _ = run_train(*job, out, extra=("--resume",))
    assert sorted(steps) == list(range(stop + 1, 13))
    assert max(abs(steps[s][0] - fresh.steps[s][0]) for s in steps) <= 1e-5


def test_corrupt_latent_shard(job, tmp_path):
    cache = shutil.copytree(job[1], tmp_path / "data")
    shard = cache / "latents" / "train-0001.safetensors"
    b = bytearray(shard.read_bytes())
    b[-1] ^= 0x01
    shard.write_bytes(bytes(b))
    out = tmp_path / "run"
    _, so, se = run_train(job[0], cache, job[2], out, extra=("--steps", "2"), ok=False)
    assert "train-0001.safetensors: expected" in se
    assert "train-0000.safetensors: expected" not in se
    assert "loading base checkpoint" not in so and not (out / "ckpt").exists()
    steps, _, _ = run_train(job[0], cache, job[2], tmp_path / "nv",
                            extra=("--steps", "2", "--no-verify-latents"))
    assert sorted(steps) == [1, 2]


def test_layer_loop_unroll_matches_scan(fresh, job, tmp_path):
    out = tmp_path / "run"
    steps, _, _ = run_train(*job, out, extra=("--layer-loop", "unroll", "--steps", "2"))
    assert sorted(steps) == [1, 2]
    assert max(abs(steps[s][0] - fresh.steps[s][0]) for s in steps) <= 1e-4
    assert tensor_shapes(out / "model.safetensors") == \
        tensor_shapes(fresh.out / "model.safetensors")
