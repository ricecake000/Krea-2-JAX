"""End-to-end test of scripts/train_full.py on CPU with 8 emulated devices
and a tiny model:

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
     --batch-size, or a reordered cache (dataset fingerprint), is a hard
     error, a different --lr warns and continues,
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
     (last occurrence wins, value must be empty).

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

from test_data import make_fake_cache
from test_parity import get_reference, patch_reference_for_cpu

TINY = dict(
    features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
    layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3,
)
STEP_RE = re.compile(
    r"^step\s+(\d+)/\d+\s+loss\s+(\S+)\s+ema_loss\s+\S+\s+gnorm\s+(\S+)", re.M
)


def train_cmd(base, cache, out, extra=()):
    return [
        sys.executable, str(REPO / "scripts" / "train_full.py"),
        "--base-ckpt", str(base), "--cache-dir", str(cache),
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


def main():
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
    cache = tmp / "cache"
    # 20 samples / batch 8: 2 steps per epoch (tail of 4 dropped), so the
    # resume at step 6 lands on an epoch boundary mid-run
    make_fake_cache(cache, 20, lat_shape=(4, 8, 8), txt_layers=3, txt_dim=64)

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
    print("a. full fine-tuning lowers the loss: ok")

    # --- b. exported weights ---
    exported = out1 / "model.safetensors"
    assert exported.exists() and (out1 / "model_step000006.safetensors").exists()
    with safe_open(str(exported), framework="numpy") as f:
        meta = f.metadata()
        keys = set(f.keys())
    assert keys == set(sd), (sorted(set(sd) - keys), sorted(keys - set(sd)))
    assert meta["step"] == "12" and meta["source"] == "ema", meta
    assert meta["base"] == "base.safetensors", meta
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
        [sys.executable, str(REPO / "scripts" / "train_full.py"),
         "--base-ckpt", str(base), "--cache-dir", str(cache),
         "--output-dir", str(out2), "--model-config", json.dumps(TINY),
         "--steps", "20", "--batch-size", "8"],
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
    # same sample count, two samples swapped in meta.json: the data order
    # changes, so the dataset fingerprint must refuse the resume
    meta_path = cache / "meta.json"
    meta_orig = meta_path.read_text()
    try:
        m = json.loads(meta_orig)
        m["samples"][0], m["samples"][1] = m["samples"][1], m["samples"][0]
        meta_path.write_text(json.dumps(m, indent=2))
        d = fresh_copy("badswap")
        proc = subprocess.run(
            train_cmd(base, cache, d, ("--resume",)),
            env={**os.environ}, capture_output=True, text=True, timeout=900,
        )
    finally:
        meta_path.write_text(meta_orig)
    err = proc.stderr
    assert proc.returncode != 0, "resume on a reordered cache did not fail"
    assert "different run configuration" in err, err[-2000:]
    assert "dataset_fingerprint: checkpoint" in err, err[-2000:]
    assert "n_samples:" not in err, err[-2000:]
    assert "resumed train state" not in proc.stdout
    print("  resume with two cache samples swapped: refused "
          "(dataset_fingerprint mismatch)")
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
    proc = subprocess.run(
        train_cmd(base, cache, outE, ("--resume",)),
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

    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL TRAIN-FULL TESTS PASSED")


if __name__ == "__main__":
    main()
