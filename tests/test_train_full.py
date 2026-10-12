"""End-to-end test of scripts/train_full.py on CPU with 8 emulated devices,
a tiny DiT, a tiny Qwen3-VL text encoder (tests/test_qwen3_text.py; run
online in the step) and a fake aspect-bucketed latent cache (20 samples in
two buckets, 12 and 16 image tokens, --img-tokens 16, so half the samples
carry padded image rows):

  a. a short full fine-tuning run (fp32 master, AdamW, EMA, orbax saves)
     lowers the flow-matching loss,
  b. the exported model.safetensors has exactly the official key set, loads
     through load_krea2_params (sharded), and runs in the official PyTorch
     reference with the same output as our JAX forward; the export is the
     EMA tree (between the base and the raw params),
  c. --resume from the step-6 orbax checkpoint reproduces steps 7..12 of the
     uninterrupted run (data order, RNG, LR schedule, optimizer state),
  d. the optimizer-state sharding guard accepts the sharded state and fires
     on a replicated one,
  e. --resume validates the run configuration: a different --seed or
     --batch-size, an edited caption in metadata.parquet (dataset
     fingerprint), or a text encoder with different tokenizer.json bytes
     (text_encoder_id) is a hard error; the same encoder copied to another
     path resumes cleanly; a different --lr warns and continues,
  f. SIGTERM / SIGINT stop the run gracefully (checkpoint + export, exit 0)
     and --resume continues it exactly,
  g. --resume-step N resumes an older step and sets newer ones aside,
  h. a run that reached --steps but lost its final export (or holds a stale
     one from an earlier stop) re-exports it on --resume without training,
  i. an incomplete newest step dir makes --resume fail with a pointer to
     --resume-step, which then works,
  j. stale <step>.orbax-checkpoint-tmp dirs are removed when the writing
     checkpoint manager opens,
  k. the --attn aiter XLA_FLAGS check parses the command-buffer flag
     (last occurrence wins, value must be empty),
  l. build_positions_and_mask_bucketed equals build_positions_and_mask
     when nothing is padded, and has the documented per-sample layout when
     it is (eager and jitted),
  m. a padded sample's loss and gradients do not depend on the values in
     its padded image rows (direct make_train_step call: loss, grad norm,
     updated params and AdamW moments bitwise equal), while a valid row
     does change them; masked_flow_loss matches its numpy formula and
     keeps inf/NaN in padded rows out of the loss and its gradient,
  n. per-sample mu (from h*w) and a --timestep-mu override both train
     (unit + a short CLI run), and the run config records the new keys.

Requires CPU PyTorch + einops (tiny reference checkpoint, parity check).
"""

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
import math
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

from test_data import make_fake_latent_cache
from test_parity import get_reference, patch_reference_for_cpu
from test_qwen3_text import make_tiny_text_encoder_dir

TINY = dict(
    features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
    layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3,
)
# Qwen3TextConfig overrides for the tiny encoder: 3 select layers = txtlayers
TEXT_OVERRIDES = {"select_layers": [0, 1, 2]}
TEXT_ROWS = 32
IMG_TOKENS = 16
# (bucket_w, bucket_h) px -> 4-channel latents (8, 6) / (8, 8) -> 4x3 = 12
# and 4x4 = 16 image tokens
BUCKETS = ((48, 64), (64, 64))
TEXT_ENCODER = None  # set by main(): the tiny encoder dir
STEP_RE = re.compile(
    r"^step\s+(\d+)/\d+\s+loss\s+(\S+)\s+ema_loss\s+\S+\s+gnorm\s+(\S+)", re.M
)


def train_cmd(base, cache, out, extra=()):
    return [
        sys.executable, str(REPO / "scripts" / "train_full.py"),
        "--base-ckpt", str(base), "--data-dir", str(cache),
        "--text-encoder", str(TEXT_ENCODER),
        "--text-model-config", json.dumps(TEXT_OVERRIDES),
        "--text-rows", str(TEXT_ROWS), "--img-tokens", str(IMG_TOKENS),
        "--output-dir", str(out), "--model-config", json.dumps(TINY),
        "--steps", "12", "--save-every", "6", "--export-every", "6",
        "--lr", "1e-3", "--warmup-steps", "2", "--batch-size", "8",
        "--ema-decay", "0.9", "--log-every", "1", "--qchunk", "256",
        "--remat-policy", "dots", "--loader-workers", "2", *extra,
    ]


def run_train(base, cache, out, extra=()):
    cmd = train_cmd(base, cache, out, extra)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=900)
    out_text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(out_text)
        raise AssertionError(f"train_full.py failed (exit {proc.returncode})")
    print(f"  train_full.py {' '.join(extra) or '(fresh)'}: "
          f"{time.time() - t0:.0f}s")
    steps = {int(s): (float(l), float(g)) for s, l, g in STEP_RE.findall(proc.stdout)}
    assert steps, f"no step lines parsed:\n{out_text}"
    return steps, proc.stdout


def check_positions_helper():
    """l. bucketed positions/mask vs the fixed-shape helper."""
    import jax
    import jax.numpy as jnp

    from krea2_jax.model import (
        build_positions_and_mask, build_positions_and_mask_bucketed,
    )

    rng = np.random.default_rng(0)
    B, Ltxt = 3, 5
    txt_mask = jnp.asarray(np.arange(Ltxt)[None, :]
                           < rng.integers(1, Ltxt + 1, size=(B, 1)))
    for h, w in ((4, 4), (2, 8), (8, 2), (1, 16)):
        hw = jnp.asarray(np.tile([[h, w]], (B, 1)), jnp.int32)
        pos_b, mask_b = build_positions_and_mask_bucketed(txt_mask, hw, h * w)
        pos_f, mask_f = build_positions_and_mask(txt_mask, h, w)
        assert pos_b.dtype == pos_f.dtype and mask_b.dtype == mask_f.dtype
        assert np.array_equal(np.asarray(pos_b), np.asarray(pos_f)), (h, w)
        assert np.array_equal(np.asarray(mask_b), np.asarray(mask_f)), (h, w)
    hw_np = np.asarray([[4, 3], [2, 5], [4, 4]], np.int32)
    N = 16
    pos, mask = build_positions_and_mask_bucketed(txt_mask, hw_np, N)
    pos_j, mask_j = jax.jit(build_positions_and_mask_bucketed,
                            static_argnums=2)(txt_mask, jnp.asarray(hw_np), N)
    assert np.array_equal(np.asarray(pos), np.asarray(pos_j))
    assert np.array_equal(np.asarray(mask), np.asarray(mask_j))
    pos, mask = np.asarray(pos), np.asarray(mask)
    assert pos.shape == (B, Ltxt + N, 3) and pos.dtype == np.float32
    assert mask.shape == (B, Ltxt + N) and mask.dtype == bool
    assert not pos[:, :Ltxt].any()
    assert np.array_equal(mask[:, :Ltxt], np.asarray(txt_mask))
    for b, (h, w) in enumerate(hw_np.tolist()):
        n = h * w
        exp = np.zeros((N, 3), np.float32)
        exp[:n, 1] = np.arange(n) // w
        exp[:n, 2] = np.arange(n) % w
        assert np.array_equal(pos[b, Ltxt:], exp), (b, pos[b, Ltxt:])
        assert np.array_equal(mask[b, Ltxt:], np.arange(N) < n), b
    print("l. build_positions_and_mask_bucketed == fixed-shape helper when "
          "unpadded; per-sample grid + zero/False padding (eager == jit): ok")


def check_step_padding_and_mu(cfg, mesh, params, pshard, tx, opt_state,
                              opt_shard):
    """m + n (units): padded image rows are inert; per-sample mu vs
    override; masked_flow_loss formula."""
    import jax
    import jax.numpy as jnp

    import train_full
    from krea2_jax.checkpoint import batch_sharding
    from krea2_jax.qwen3_text import (
        from_hf_config, load_qwen3_text_params, load_tokenizer,
    )
    from krea2_jax.text_template import CaptionTemplate

    # masked_flow_loss == numpy per-sample mean over valid rows, batch mean
    rng = np.random.default_rng(5)
    v = rng.standard_normal((3, 6, 4)).astype(np.float32)
    tg = rng.standard_normal((3, 6, 4)).astype(np.float32)
    n_img = np.asarray([6, 4, 1], np.int32)
    exp = np.mean([np.mean((v[b, :n] - tg[b, :n]) ** 2)
                   for b, n in enumerate(n_img)])
    got = float(train_full.masked_flow_loss(jnp.asarray(v), jnp.asarray(tg),
                                            jnp.asarray(n_img)))
    assert abs(got - exp) <= 1e-6 * abs(exp), (got, exp)
    full = float(train_full.masked_flow_loss(
        jnp.asarray(v), jnp.asarray(tg), jnp.full((3,), 6, jnp.int32)))
    assert abs(full - float(np.mean((v - tg) ** 2))) <= 1e-6 * full
    # per-sample mu: cfg.mu_for_seq_len per sample; override: constant
    n = jnp.asarray([12, 16, 4096], jnp.int32)
    mu = np.asarray(train_full.per_sample_mu(cfg, n))
    ref = np.asarray([cfg.mu_for_seq_len(int(k)) for k in (12, 16, 4096)])
    assert mu.dtype == np.float32 and np.allclose(mu, ref, rtol=1e-6), (mu, ref)
    assert np.array_equal(np.asarray(train_full.per_sample_mu(cfg, n, 0.8)),
                          np.full(3, 0.8, np.float32))
    # padded rows holding inf / NaN: select-before-square keeps the loss and
    # its gradient finite (loss 4 = 2^2 over the single valid row)
    lg = jax.jit(jax.value_and_grad(
        lambda v_: train_full.masked_flow_loss(v_, jnp.zeros_like(v_),
                                               jnp.asarray([1], jnp.int32))))
    for bad in (np.inf, np.nan):
        lv, gv = lg(jnp.asarray([[[2.0], [bad]]], jnp.float32))
        assert float(lv) == 4.0, (bad, lv)
        assert np.array_equal(np.asarray(gv), [[[4.0], [0.0]]]), (bad, gv)
    print(f"m/n. masked_flow_loss == numpy formula, inf/NaN padded rows give "
          f"loss 4.0 and grad [4, 0]; per_sample_mu {mu.round(4)} "
          "== mu_for_seq_len per sample, override constant: ok")

    te = Path(TEXT_ENCODER)
    txt_cfg = from_hf_config(te, **TEXT_OVERRIDES)
    template = CaptionTemplate(load_tokenizer(te), text_rows=TEXT_ROWS)
    txt_params = load_qwen3_text_params(te, jnp.bfloat16, mesh=mesh,
                                        cfg=txt_cfg)
    hp = train_full.parse_args([
        "--base-ckpt", "-", "--data-dir", "-", "--text-encoder", "-",
        "--output-dir", "-", "--qchunk", "256", "--ema-decay", "0",
    ])
    B, feat = 8, cfg.channels * cfg.patch * cfg.patch
    hw = np.asarray([[4, 3], [4, 4]] * (B // 2), np.int32)
    n_img = hw[:, 0] * hw[:, 1]
    valid = np.arange(IMG_TOKENS)[None, :] < n_img[:, None]
    img = rng.standard_normal((B, IMG_TOKENS, feat)).astype(np.float32)
    clean = np.where(valid[..., None], img, 0.0)
    garbage = np.where(valid[..., None], img,
                       50.0 * rng.standard_normal(img.shape))
    changed = clean.copy()
    changed[0, 3] += 1.0  # a VALID row of a padded sample
    ids, txt_len = template.encode_captions(
        [f"caption {i}, a girl with long hair" for i in range(B)])
    ds = batch_sharding(mesh)
    key = jax.random.PRNGKey(3)
    copy_p = jax.jit(lambda t: jax.tree.map(jnp.copy, t), out_shardings=pshard)
    copy_o = jax.jit(lambda t: jax.tree.map(jnp.copy, t),
                     out_shardings=opt_shard)

    def run(step, img_np):
        batch = [jax.device_put(np.asarray(a, dt), ds) for a, dt in (
            (img_np, jnp.bfloat16), (hw, np.int32), (ids, np.int32),
            (txt_len, np.int32))]
        p, o, _, m = step(copy_p(params), copy_o(opt_state), None, txt_params,
                          *batch, key)
        return (float(m["loss"]), float(m["grad_norm"]),
                [np.asarray(x) for x in jax.tree.leaves(p)],
                [np.asarray(x) for x in jax.tree.leaves(o)])

    step = train_full.make_train_step(cfg, hp, mesh, tx, False, pshard,
                                      opt_shard, txt_cfg, IMG_TOKENS)
    a, g, c = run(step, clean), run(step, garbage), run(step, changed)
    assert np.isfinite(a[0]) and np.isfinite(a[1]), a[:2]
    assert a[0] == g[0] and a[1] == g[1], (a[:2], g[:2])
    for x, y in zip(a[2] + a[3], g[2] + g[3]):
        assert x.dtype == y.dtype and np.array_equal(x, y), "padding leaked"
    assert c[0] != a[0] and c[1] != a[1], "a valid row did not matter"
    print(f"m. padded rows (zeros vs 50*N(0,1) garbage): loss {a[0]:.6f}, "
          f"gnorm {a[1]:.6f}, params and AdamW moments bitwise equal; a "
          f"changed valid row moves the loss to {c[0]:.6f}: ok")

    step_mu = train_full.make_train_step(cfg, hp, mesh, tx, False, pshard,
                                         opt_shard, txt_cfg, IMG_TOKENS,
                                         mu_override=0.8)
    o = run(step_mu, clean)
    assert np.isfinite(o[0]) and np.isfinite(o[1]), o[:2]
    assert o[0] != a[0], "mu override did not change the timesteps"
    print(f"n. per-sample mu loss {a[0]:.6f} vs --timestep-mu 0.8 loss "
          f"{o[0]:.6f} (same key/batch, both finite): ok")


def main():
    global TEXT_ENCODER
    import torch

    mmdit = get_reference()
    patch_reference_for_cpu(mmdit, torch)

    import jax
    import jax.numpy as jnp
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from krea2_jax.checkpoint import (
        load_krea2_params, make_mesh, param_shardings, replicate_tree,
    )
    from krea2_jax.config import Krea2Config
    from krea2_jax.model import build_positions_and_mask, dit_forward, patchify
    import train_full

    assert len(jax.devices()) == 8, jax.devices()
    cfg = Krea2Config(**TINY)
    tmp = Path(tempfile.mkdtemp(prefix="test_train_full_"))

    # --- tiny base checkpoint (reference state-dict layout) + fake cache ---
    torch.manual_seed(0)
    model = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**TINY)).eval().float()
    g = torch.Generator().manual_seed(1)
    sd = {k: torch.randn(v.shape, generator=g) * 0.02
          for k, v in model.state_dict().items()}
    base = tmp / "base.safetensors"
    save_file(sd, str(base))
    cache = tmp / "data"
    # 20 samples / batch 8: 2 steps per epoch (tail of 4 dropped), so the
    # resume at step 6 lands on an epoch boundary mid-run; two buckets (12
    # and 16 image tokens at --img-tokens 16) in two shards
    ids = np.random.default_rng(7).choice(10_000, size=20, replace=False)
    rows = [{"id": int(i), "bucket_w": BUCKETS[k % 2][0],
             "bucket_h": BUCKETS[k % 2][1],
             "artist_name": "" if k % 5 == 0 else f"artist {k % 3}",
             "tar": f"data/train-{k % 2:04d}.tar"}
            for k, i in enumerate(ids.tolist())]
    make_fake_latent_cache(cache, rows, lat_channels=TINY["channels"])
    te_dir, _ = make_tiny_text_encoder_dir(tmp / "te")
    TEXT_ENCODER = te_dir

    # --- d. optimizer-state sharding guard (unit) ---
    mesh = make_mesh()
    params = load_krea2_params(str(base), mesh=mesh, dtype=jnp.float32)
    pshard = param_shardings(params, mesh)
    args = SimpleNamespace(
        lr=1e-3, warmup_steps=2, cosine_decay=False, min_lr=0.0, steps=12,
        grad_clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
    )
    tx = train_full.make_optimizer(args, train_full.make_schedule(args))
    abstract_opt = jax.eval_shape(tx.init, params)
    opt_shard = train_full.opt_state_shardings(abstract_opt, params, mesh, pshard)
    opt_state = jax.jit(tx.init, in_shardings=(pshard,),
                        out_shardings=opt_shard)(params)
    n_param_leaves = len(jax.tree.leaves(params))
    n = train_full.check_opt_state_sharding(opt_state, params)
    assert n == 2 * n_param_leaves, (n, n_param_leaves)  # adam mu + nu
    n_sharded = sum(not a.sharding.is_fully_replicated
                    for a in jax.tree.leaves(opt_state) if a.ndim)
    assert n_sharded > 0
    print(f"opt-state guard: {n} leaves sharded like params "
          f"({n_sharded} actually split): ok")

    def expect_guard_failure(state, why):
        try:
            train_full.check_opt_state_sharding(state, params)
        except RuntimeError as e:
            assert "not sharded like the params" in str(e), e
            return
        raise AssertionError(f"sharding guard accepted {why}")

    expect_guard_failure(replicate_tree(opt_state, mesh), "a replicated state")
    leaves, treedef = jax.tree.flatten(opt_state)
    i_bad = next(i for i, a in enumerate(leaves)
                 if a.ndim and not a.sharding.is_fully_replicated)
    leaves[i_bad] = replicate_tree(leaves[i_bad], mesh)
    expect_guard_failure(jax.tree.unflatten(treedef, leaves),
                         "a single replicated moment")
    print("opt-state guard fires on replicated leaves: ok")

    # --- l, m, n (units on the tiny model) ---
    check_positions_helper()
    check_step_padding_and_mu(cfg, mesh, params, pshard, tx, opt_state,
                              opt_shard)
    # informational: does plain jit(tx.init, in_shardings=...) shard mu/nu?
    plain = jax.jit(tx.init, in_shardings=(pshard,))(params)
    try:
        train_full.check_opt_state_sharding(plain, params)
        print("  (info) jit(tx.init) without out_shardings also shards: yes")
    except RuntimeError:
        print("  (info) jit(tx.init) without out_shardings replicates "
              "moments — explicit out_shardings required")
    del params, opt_state, plain, leaves

    # --- a. fresh 12-step run ---
    out1 = tmp / "run1"
    steps1, log1 = run_train(base, cache, out1)
    assert sorted(steps1) == list(range(1, 13)), sorted(steps1)
    losses = [steps1[s][0] for s in range(1, 13)]
    print(f"losses: {', '.join(f'{l:.4f}' for l in losses)}")
    assert all(np.isfinite(losses)), losses
    assert all(np.isfinite([steps1[s][1] for s in steps1]))
    assert losses[-1] < losses[0], "loss at step 12 not below step 1"
    assert np.mean(losses[-3:]) < np.mean(losses[:3]), "loss did not decrease"
    assert "optimizer state:" in log1 and "sharded like params" in log1
    assert re.search(r"dataset: 20 samples in 2 buckets \(top: 48x64 x10, "
                     r"64x64 x10\); img tokens 12\.\.16 padded to 16; "
                     r"text_rows 32 ", log1), log1
    assert "text encoder loaded" in log1 and "position_mode contiguous" in log1
    from krea2_jax import fullft_ckpt as _fc
    rc = _fc.read_meta_at(str(out1 / "ckpt"), 12)["config"]
    for k, val in (("text_rows", TEXT_ROWS), ("img_tokens", IMG_TOKENS),
                   ("caption_probs", [0.5, 0.3, 0.2]), ("artist_prob", 0.5),
                   ("text_encoder", "te"), ("position_mode", "contiguous"),
                   ("lat_channels", 4), ("n_samples", 20),
                   ("timestep_mu", None),
                   ("text_model_config", TEXT_OVERRIDES)):
        assert rc.get(k, "<missing>") == val, (k, rc.get(k, "<missing>"))
    assert "txt_len" not in rc and "resolution" not in rc, rc
    assert len(rc["dataset_fingerprint"]) == 64, rc
    from krea2_jax.qwen3_text import from_hf_config, text_encoder_identity
    assert rc["text_encoder_id"] == text_encoder_identity(
        TEXT_ENCODER, from_hf_config(TEXT_ENCODER, **TEXT_OVERRIDES)), rc
    print("a. full fine-tuning lowers the loss; run config records the "
          "bucketed/text-encoder keys: ok")

    # --- b. exported weights ---
    exported = out1 / "model.safetensors"
    assert exported.exists() and (out1 / "model_step000006.safetensors").exists()
    with safe_open(str(exported), framework="numpy") as f:
        meta = f.metadata()
        keys = set(f.keys())
    assert keys == set(sd), (sorted(set(sd) - keys), sorted(keys - set(sd)))
    assert meta["step"] == "12" and meta["source"] == "ema", meta
    assert meta["base"] == "base.safetensors", meta
    assert meta["resolution"] == "bucketed" and meta["img_tokens"] == "16", meta
    with safe_open(str(out1 / "model_step000006.safetensors"), framework="numpy") as f:
        assert f.metadata()["step"] == "6"
    ft = load_file(str(exported))
    assert all(v.dtype == torch.bfloat16 for v in ft.values())
    assert all(ft[k].shape == sd[k].shape for k in sd)
    moved = max(float((ft[k].float() - sd[k]).abs().max()) for k in sd)
    assert moved > 1e-3, f"exported weights barely differ from base ({moved})"

    sharded_ft = load_krea2_params(str(exported), mesh=mesh, dtype=jnp.float32)
    assert any(not a.sharding.is_fully_replicated
               for a in jax.tree.leaves(sharded_ft))
    del sharded_ft

    # torch reference forward on the exported weights vs our JAX forward
    model.load_state_dict({k: v.float() for k, v in ft.items()})
    B, ltxt, latH, latW = 2, 7, 16, 24
    htok, wtok = latH // 2, latW // 2
    lat = torch.randn(B, TINY["channels"], latH, latW, generator=g)
    context = torch.randn(B, ltxt, TINY["txtlayers"], TINY["txtdim"], generator=g)
    t = torch.tensor([0.3, 0.8])
    txtmask = torch.ones(B, ltxt, dtype=torch.bool)
    txtmask[0, 5:] = False
    img_tokens = (
        lat.reshape(B, TINY["channels"], htok, 2, wtok, 2)
        .permute(0, 2, 4, 1, 3, 5).reshape(B, htok * wtok, -1)
    )
    imgids = torch.zeros(htok, wtok, 3)
    imgids[..., 1] = torch.arange(htok)[:, None]
    imgids[..., 2] = torch.arange(wtok)[None, :]
    pos = torch.cat(
        [torch.zeros(B, ltxt, 3), imgids.reshape(1, -1, 3).expand(B, -1, -1)], dim=1
    )
    mask = torch.cat([txtmask, torch.ones(B, htok * wtok, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        ref_out = model(img=img_tokens, context=context, t=t, pos=pos,
                        mask=mask).numpy()

    jparams = load_krea2_params(str(exported), mesh=None, dtype=jnp.float32)
    jpos, jmask = build_positions_and_mask(jnp.asarray(txtmask.numpy()), htok, wtok)
    jx_out = np.asarray(dit_forward(
        jparams, cfg, patchify(jnp.asarray(lat.numpy()), 2),
        jnp.asarray(context.numpy()), jnp.asarray(t.numpy()), jpos, jmask,
        remat=False, qchunk=64,
    ))
    err = np.abs(jx_out - ref_out).max() / (np.abs(ref_out).max() + 1e-8)
    print(f"exported weights: torch-reference vs JAX forward rel err {err:.3e}")
    assert err < 2e-3, "exported checkpoint does not run identically in torch"

    # EMA export sits between the base and the raw trained params
    raw_path = tmp / "raw_params.safetensors"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export_full.py"),
         "--ckpt-dir", str(out1 / "ckpt"), "--step", "12",
         "--output", str(raw_path), "--source", "params"],
        check=True, env={**os.environ}, capture_output=True,
    )
    raw = load_file(str(raw_path))
    d_ema = sum(float((ft[k].float() - sd[k]).norm() ** 2) for k in sd) ** 0.5
    d_raw = sum(float((raw[k].float() - sd[k]).norm() ** 2) for k in sd) ** 0.5
    d_gap = max(float((ft[k].float() - raw[k].float()).abs().max()) for k in sd)
    print(f"|ema - base| {d_ema:.4f}  |params - base| {d_raw:.4f}  "
          f"max|ema - params| {d_gap:.4f}")
    assert 0 < d_ema < d_raw and d_gap > 1e-3, "export is not the EMA tree"
    print("b. export: official keys, bf16, sharded load, torch parity, EMA: ok")

    # --- c. resume from the step-6 checkpoint ---
    out2 = tmp / "run2"
    (out2 / "ckpt").mkdir(parents=True)
    assert (out1 / "ckpt" / "6").is_dir(), sorted(os.listdir(out1 / "ckpt"))
    shutil.copytree(out1 / "ckpt" / "6", out2 / "ckpt" / "6")
    steps2, log2 = run_train(base, cache, out2, extra=("--resume",))
    assert "resumed train state" in log2, log2
    assert "WARNING" not in log2, "same-config resume printed a config warning"
    assert sorted(steps2) == list(range(7, 13)), sorted(steps2)
    diffs = [abs(steps2[s][0] - steps1[s][0]) for s in range(7, 13)]
    print("resumed vs uninterrupted loss diffs (steps 7..12): "
          + ", ".join(f"{d:.1e}" for d in diffs))
    assert all(math.isfinite(steps2[s][0]) for s in range(7, 13)), steps2
    assert all(math.isfinite(d) for d in diffs), diffs
    assert max(diffs) <= 1e-5, diffs
    ft2 = load_file(str(out2 / "model.safetensors"))
    wdiff = max(float((ft2[k].float() - ft[k].float()).abs().max()) for k in sd)
    print(f"resumed vs uninterrupted final export: max abs diff {wdiff:.2e}")
    assert wdiff <= 1e-3, wdiff
    # refusing to clobber an existing run without --resume
    proc = subprocess.run(
        train_cmd(base, cache, out2, ("--steps", "20")),
        env={**os.environ}, capture_output=True, text=True,
    )
    assert proc.returncode != 0 and "--resume" in proc.stderr, proc.stderr[-2000:]
    print("c. --resume reproduces the uninterrupted run: ok")

    # --- e. resume configuration validation ---
    def fresh_copy(name):
        d = tmp / name
        (d / "ckpt").mkdir(parents=True)
        shutil.copytree(out1 / "ckpt" / "6", d / "ckpt" / "6")
        return d

    for flag, val in (("--seed", "1"), ("--batch-size", "16")):
        d = fresh_copy(f"bad{flag.strip('-')}")
        proc = subprocess.run(
            train_cmd(base, cache, d, ("--resume", flag, val)),
            env={**os.environ}, capture_output=True, text=True, timeout=900,
        )
        key = flag.strip("-").replace("-", "_")
        err = proc.stderr
        assert proc.returncode != 0, f"{flag} {val} resume did not fail"
        assert "different run configuration" in err, err[-2000:]
        assert f"{key}: checkpoint" in err and f"now {val}" in err, err[-2000:]
        assert "resumed train state" not in proc.stdout
        print(f"  resume with {flag} {val}: refused ({key} mismatch)")
    # same samples and latents, one caption edited in metadata.parquet: the
    # training text changes, so the dataset fingerprint must refuse the resume
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq_path = cache / "metadata.parquet"
    pq_orig = pq_path.read_bytes()
    try:
        table = pq.read_table(pq_path)
        longs = table.column("long").to_pylist()
        longs[3] = longs[3] + " (edited)"
        table = table.set_column(table.schema.get_field_index("long"), "long",
                                 pa.array(longs, type=pa.string()))
        pq.write_table(table, pq_path)
        d = fresh_copy("badcaption")
        proc = subprocess.run(
            train_cmd(base, cache, d, ("--resume",)),
            env={**os.environ}, capture_output=True, text=True, timeout=900,
        )
    finally:
        pq_path.write_bytes(pq_orig)
    err = proc.stderr
    assert proc.returncode != 0, "resume on an edited caption did not fail"
    assert "different run configuration" in err, err[-2000:]
    assert "dataset_fingerprint: checkpoint" in err, err[-2000:]
    assert "n_samples:" not in err, err[-2000:]
    assert "resumed train state" not in proc.stdout
    print("  resume with one caption edited in metadata.parquet: refused "
          "(dataset_fingerprint mismatch)")
    # text encoder identity: a tokenizer.json with other bytes (same tokens,
    # so the dataset fingerprint is unchanged) is refused; the same encoder
    # copied to another path (same basename) resumes without a warning
    te_mod = shutil.copytree(TEXT_ENCODER, tmp / "te_modified" / "te")
    tj = json.loads((te_mod / "tokenizer.json").read_text())
    (te_mod / "tokenizer.json").write_text(json.dumps(tj, indent=1))
    d = fresh_copy("badtokenizer")
    proc = subprocess.run(
        train_cmd(base, cache, d, ("--resume", "--text-encoder", str(te_mod))),
        env={**os.environ}, capture_output=True, text=True, timeout=900,
    )
    err = proc.stderr
    assert proc.returncode != 0, "resume with another tokenizer.json ran"
    assert "different run configuration" in err, err[-2000:]
    assert "text_encoder_id: checkpoint" in err, err[-2000:]
    assert "dataset_fingerprint" not in err, err[-2000:]
    assert "resumed train state" not in proc.stdout
    print("  resume with a modified tokenizer.json: refused (text_encoder_id "
          "mismatch)")
    te_copy = shutil.copytree(TEXT_ENCODER, tmp / "te_elsewhere" / "te")
    d = fresh_copy("copiedte")
    stepsT, logT = run_train(base, cache, d,
                             extra=("--resume", "--text-encoder", str(te_copy)))
    assert "resumed train state" in logT and sorted(stepsT) == list(range(7, 13))
    assert "WARNING" not in logT, logT
    tdiffs = [abs(stepsT[k][0] - steps1[k][0]) for k in range(7, 13)]
    assert max(tdiffs) <= 1e-5, tdiffs  # same encoder -> same run
    print("  resume with the same encoder copied to another path: accepted")
    d = fresh_copy("newlr")
    steps3, log3 = run_train(base, cache, d, extra=("--resume", "--lr", "2e-3"))
    assert "WARNING: resuming step 6 with changed optimizer/schedule" in log3, log3
    assert "lr: checkpoint 0.001, now 0.002" in log3, log3
    assert "LR schedule" in log3
    assert "resumed train state" in log3 and sorted(steps3) == list(range(7, 13))
    # unit: older checkpoints (no config) warn; flipping EMA on/off is fatal
    saved = json.loads(json.dumps({k: 0 for k in train_full.RESUME_HARD_KEYS
                                   + train_full.RESUME_SOFT_KEYS}))
    saved["ema_decay"] = 0.9
    w = train_full.check_resume_config(None, saved, 6)
    assert len(w) == 1 and "no saved run configuration" in w[0], w
    assert train_full.check_resume_config(saved, dict(saved), 6) == []
    try:
        train_full.check_resume_config(saved, {**saved, "ema_decay": 0.0}, 6)
    except SystemExit as e:
        assert "ema enabled" in str(e), e
    else:
        raise AssertionError("EMA on -> off resume was not refused")
    w = train_full.check_resume_config(saved, {**saved, "ema_decay": 0.99}, 6)
    assert any("ema_decay: checkpoint 0.9, now 0.99" in l for l in w), w
    print("e. --resume config validation (hard errors / warnings): ok")

    def numeric_steps(ckpt):
        return sorted(int(p.name) for p in ckpt.iterdir() if p.name.isdigit())

    def check_losses(steps, lo, what, last=12):
        """steps must be exactly lo..last; losses on the overlap with the
        uninterrupted 12-step run (lo..12) must match it."""
        assert sorted(steps) == list(range(lo, last + 1)), sorted(steps)
        assert all(math.isfinite(steps[s][0]) for s in steps), steps
        diffs = [abs(steps[s][0] - steps1[s][0]) for s in range(lo, 13)]
        print(f"  {what} vs uninterrupted loss diffs (steps {lo}..12): "
              + ", ".join(f"{d:.1e}" for d in diffs))
        assert diffs and all(math.isfinite(d) for d in diffs), diffs
        assert max(diffs) <= 1e-5, diffs

    def run_and_signal(cmd, sig, at_step, timeout=300):
        """Run cmd, send `sig` when the step-`at_step` line appears; returns
        (returncode, output). Bounded: kills the process on timeout."""
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        lines, state = [], {"sent": False}

        def reader():
            for line in proc.stdout:
                lines.append(line)
                m = STEP_RE.match(line)
                if not state["sent"] and m and int(m.group(1)) == at_step:
                    proc.send_signal(sig)
                    state["sent"] = True

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            th.join(timeout=10)
            raise AssertionError(f"train_full.py did not exit within "
                                 f"{timeout}s:\n{''.join(lines)}")
        th.join(timeout=30)
        text = "".join(lines)
        assert state["sent"], (f"process exited (code {proc.returncode}) "
                               f"before step {at_step} was seen:\n{text}")
        return proc.returncode, text

    # --- f. graceful stop on SIGTERM / SIGINT, then --resume ---
    # --steps 40 (constant LR after warmup, so steps 1..12 are the same as
    # run1's) leaves plenty of slack between the signal and the end of the run
    long_run = ("--steps", "40", "--save-every", "100")
    stop_re = re.compile(r"^(\w+) received; stopping at step (\d+) ", re.M)
    resumed = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        name = sig.name
        outS = tmp / f"run_{name.lower()}"
        rc, text = run_and_signal(train_cmd(base, cache, outS, long_run), sig, 3)
        assert rc == 0, f"exit {rc}:\n{text}"
        m = stop_re.search(text)
        assert m and m.group(1) == name, f"no '{name} received; stopping at " \
            f"step' line:\n{text}"
        stopped = int(m.group(2))
        assert 3 <= stopped < 40, (stopped, text)
        ckpt_steps = numeric_steps(outS / "ckpt")
        assert ckpt_steps == [stopped], (ckpt_steps, stopped, text)
        assert not list((outS / "ckpt").glob("*.orbax-checkpoint-tmp*")), \
            sorted(os.listdir(outS / "ckpt"))
        assert (outS / "model.safetensors").exists()
        with safe_open(str(outS / "model.safetensors"), framework="numpy") as f:
            assert f.metadata()["step"] == str(stopped), f.metadata()
        stepsS, logS = run_train(base, cache, outS, (*long_run, "--resume"))
        assert f"resumed train state from {outS / 'ckpt'}/{stopped}" in logS, logS
        assert "WARNING" not in logS, logS
        assert sorted(stepsS) == list(range(stopped + 1, 41)), sorted(stepsS)
        assert all(math.isfinite(v[0]) for v in stepsS.values()), stepsS
        if stopped < 12:
            check_losses(stepsS, stopped + 1, f"{name} resume", last=40)
        else:
            print(f"  WARNING: signal landed at step {stopped} >= 12 on this "
                  "machine; overlap comparison with the 12-step reference "
                  "skipped")
        resumed[name] = stepsS
        print(f"f. {name}: stopped at step {stopped} with checkpoint + export, "
              "exit 0; --resume continues: ok")
    common = sorted(set(resumed["SIGTERM"]) & set(resumed["SIGINT"]))
    assert common, "SIGTERM/SIGINT resumed runs share no logged step"
    sdiffs = [abs(resumed["SIGTERM"][s][0] - resumed["SIGINT"][s][0])
              for s in common]
    assert all(math.isfinite(d) for d in sdiffs), sdiffs
    assert max(sdiffs) <= 1e-5, sdiffs
    print(f"f. SIGTERM vs SIGINT resumed runs agree on steps {common[0]}.."
          f"{common[-1]} (max diff {max(sdiffs):.1e}): ok")

    # --- g. --resume-step picks an older step, sets newer ones aside ---
    outR = tmp / "run_resume_step"
    (outR / "ckpt").mkdir(parents=True)
    for st in ("6", "12"):
        shutil.copytree(out1 / "ckpt" / st, outR / "ckpt" / st)
    shutil.copy(out1 / "model.safetensors", outR / "model.safetensors")  # step 12
    stepsR, logR = run_train(base, cache, outR, ("--resume-step", "6"))
    assert (outR / "ckpt" / "12.superseded").is_dir()
    assert "renamed newer checkpoint" in logR, logR
    assert "renamed newer export" in logR, logR
    with safe_open(str(outR / "model.safetensors.superseded"),
                   framework="numpy") as f:
        assert f.metadata()["step"] == "12", f.metadata()
    assert f"resumed train state from {outR / 'ckpt'}/6" in logR, logR
    check_losses(stepsR, 7, "--resume-step 6")
    assert (outR / "ckpt" / "12").is_dir(), sorted(os.listdir(outR / "ckpt"))
    assert (outR / "model.safetensors").exists()
    print("g. --resume-step 6: newer step + step-12 export renamed to "
          ".superseded, steps 7..12 rerun and re-saved: ok")
    # again: 12.superseded now exists, so nothing may be renamed
    before = sorted(os.listdir(outR / "ckpt"))
    proc = subprocess.run(
        train_cmd(base, cache, outR, ("--resume-step", "6")),
        env={**os.environ}, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode != 0, "rename clash was not refused"
    assert "12.superseded" in proc.stderr and "nothing was renamed" in proc.stderr, \
        proc.stderr[-2000:]
    assert sorted(os.listdir(outR / "ckpt")) == before, os.listdir(outR / "ckpt")
    assert (outR / "model.safetensors").exists()
    print("g. --resume-step with an existing .superseded target: refused, "
          "nothing renamed: ok")
    # an incompatible run config is refused before anything is renamed
    outB = tmp / "run_resume_step_badseed"
    (outB / "ckpt").mkdir(parents=True)
    for st in ("6", "12"):
        shutil.copytree(out1 / "ckpt" / st, outB / "ckpt" / st)
    shutil.copy(out1 / "model.safetensors", outB / "model.safetensors")
    proc = subprocess.run(
        train_cmd(base, cache, outB, ("--resume-step", "6", "--seed", "1")),
        env={**os.environ}, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode != 0, "--resume-step with a different seed ran"
    assert "different run configuration" in proc.stderr, proc.stderr[-2000:]
    assert (outB / "ckpt" / "12").is_dir() and (outB / "model.safetensors").exists()
    assert not list(outB.rglob("*.superseded")), sorted(outB.rglob("*"))
    print("g. --resume-step 6 --seed 1: config mismatch refused before any "
          "rename: ok")

    # --- h. export-only rerun when the final export is missing ---
    outE = tmp / "run_export_only"
    (outE / "ckpt").mkdir(parents=True)
    shutil.copytree(out1 / "ckpt" / "12", outE / "ckpt" / "12")
    # export-only needs neither the dataset nor the text encoder (a missing
    # --text-encoder dir would otherwise be looked up on the HF hub)
    proc = subprocess.run(
        train_cmd(base, tmp / "no_such_data", outE,
                  ("--resume", "--text-encoder", str(tmp / "no_such_te"))),
        env={**os.environ}, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "model.safetensors is missing: exporting" in proc.stdout, proc.stdout
    assert "resumed train state" not in proc.stdout
    assert not STEP_RE.search(proc.stdout), proc.stdout
    ftE = load_file(str(outE / "model.safetensors"))
    assert set(ftE) == set(ft)
    assert all(torch.equal(ftE[k], ft[k]) for k in ft), "export-only != run1 export"
    with safe_open(str(outE / "model.safetensors"), framework="numpy") as f:
        assert f.metadata() == meta, (f.metadata(), meta)
    # rerunning once the export exists is a clean no-op
    proc = subprocess.run(
        train_cmd(base, cache, outE, ("--resume",)),
        env={**os.environ}, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0 and "nothing to do" in proc.stdout, \
        proc.stdout[-2000:] + proc.stderr[-2000:]
    # a stale model.safetensors from an earlier stop (step 6) must not hide
    # the missing step-12 export
    outE2 = tmp / "run_export_stale"
    (outE2 / "ckpt").mkdir(parents=True)
    shutil.copytree(out1 / "ckpt" / "12", outE2 / "ckpt" / "12")
    shutil.copy(out1 / "model_step000006.safetensors", outE2 / "model.safetensors")
    proc = subprocess.run(
        train_cmd(base, cache, outE2, ("--resume",)),
        env={**os.environ}, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "is stale (it is from step 6): re-exporting step 12" in proc.stdout, \
        proc.stdout
    assert "nothing to do" not in proc.stdout and not STEP_RE.search(proc.stdout)
    with safe_open(str(outE2 / "model.safetensors"), framework="numpy") as f:
        assert f.metadata() == meta, (f.metadata(), meta)
    ftE2 = load_file(str(outE2 / "model.safetensors"))
    assert set(ftE2) == set(ft) and all(torch.equal(ftE2[k], ft[k]) for k in ft)
    # an export of the wrong tree (ema export, --ema-decay 0 asks for params)
    # is stale too: re-exported from the params
    proc = subprocess.run(
        train_cmd(base, cache, outE2, ("--resume", "--ema-decay", "0")),
        env={**os.environ}, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "(it is the ema export, params requested)" in proc.stdout, proc.stdout
    with safe_open(str(outE2 / "model.safetensors"), framework="numpy") as f:
        assert f.metadata() == {**meta, "source": "params"}, f.metadata()
    ftP = load_file(str(outE2 / "model.safetensors"))
    assert all(torch.equal(ftP[k], raw[k]) for k in raw), "params re-export"
    print("h. reached --steps without a current model.safetensors (missing, stale "
          "step-6 file or wrong source): export-only rerun, bitwise equal to the "
          "original export: ok")

    # --- i. incomplete newest step dir ---
    outI = tmp / "run_incomplete"
    (outI / "ckpt").mkdir(parents=True)
    shutil.copytree(out1 / "ckpt" / "6", outI / "ckpt" / "6")
    (outI / "ckpt" / "18").mkdir()
    proc = subprocess.run(
        train_cmd(base, cache, outI, ("--resume",)),
        env={**os.environ}, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode != 0, "resume from an empty newest step did not fail"
    assert "--resume-step" in proc.stderr, proc.stderr[-2000:]
    assert "incomplete or unreadable" in proc.stderr, proc.stderr[-2000:]
    stepsI, logI = run_train(base, cache, outI, ("--resume-step", "6"))
    assert "resumed train state" in logI, logI
    assert (outI / "ckpt" / "18.superseded").is_dir()
    assert not (outI / "ckpt" / "18").exists()
    check_losses(stepsI, 7, "--resume-step past an incomplete step")
    print("i. incomplete newest step: --resume refuses with a --resume-step "
          "hint, --resume-step 6 proceeds: ok")

    # --- j. stale orbax tmp dirs are cleaned when the manager opens (unit) ---
    from krea2_jax import fullft_ckpt
    d = tmp / "tmpclean"
    tiny = {
        "params": {"w": jnp.arange(16.0).reshape(4, 4)},
        "opt_state": {"m": jnp.zeros((4, 4))},
        "ema": None,
    }
    mgr = fullft_ckpt.make_manager(str(d))
    fullft_ckpt.save_state(mgr, 1, {**tiny, "step": 1})
    fullft_ckpt.wait(mgr)
    mgr.close()
    (d / "7.orbax-checkpoint-tmp").mkdir()
    mgr = fullft_ckpt.make_manager(str(d))
    fullft_ckpt.save_state(mgr, 2, {**tiny, "step": 2})
    fullft_ckpt.wait(mgr)
    mgr.close()
    assert not (d / "7.orbax-checkpoint-tmp").exists(), sorted(os.listdir(d))
    assert fullft_ckpt.all_steps(str(d)) == [1, 2], fullft_ckpt.all_steps(str(d))
    print("j. stale .orbax-checkpoint-tmp removed on manager open: ok")

    # --- k. XLA_FLAGS command-buffer check (unit) ---
    cbd = train_full.command_buffers_disabled
    flag = "--xla_gpu_enable_command_buffer="
    assert not cbd("")
    assert cbd(flag)
    assert not cbd(flag + "FUSION")
    assert cbd(f"--foo {flag}FUSION {flag}")
    assert not cbd(f"--foo {flag} {flag}FUSION")
    assert not cbd("--xla_force_host_platform_device_count=8 --foo=1")
    print("k. command_buffers_disabled parses XLA_FLAGS (last token wins): ok")

    # --- n. CLI run with a --timestep-mu override ---
    outM = tmp / "run_mu"
    stepsM, logM = run_train(base, cache, outM,
                             ("--steps", "2", "--timestep-mu", "0.8"))
    assert sorted(stepsM) == [1, 2], sorted(stepsM)
    assert all(math.isfinite(v[0]) for v in stepsM.values()), stepsM
    assert "mu 0.8000 (--timestep-mu, all samples)" in logM, logM
    assert "(per sample)" in log1, log1
    cfgM = fullft_ckpt.read_meta_at(str(outM / "ckpt"), 2)["config"]
    assert cfgM["timestep_mu"] == 0.8, cfgM
    print(f"n. --timestep-mu 0.8 CLI run: losses "
          f"{', '.join(f'{stepsM[s][0]:.4f}' for s in (1, 2))}, recorded in "
          "the run config: ok")

    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL TRAIN-FULL TESTS PASSED")


if __name__ == "__main__":
    main()
