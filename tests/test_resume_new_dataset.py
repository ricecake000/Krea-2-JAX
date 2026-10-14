"""scripts/train_full.py --resume-new-dataset: continue a full fine-tune's
train state (params, AdamW state, EMA, step / LR schedule position) on a new
dataset version, whose data order starts at its own epoch 0.

Units (in-process, no devices): check_resume_config / resume_data_position
on run-config dicts.

End to end (main(), run by test_cli in a subprocess on 8 emulated CPU
devices; reuses tests/test_train_full.py's tiny DiT, tiny text encoder and
fake bucketed latent cache; every train run goes through this file's
--train driver, which records the dataset ids of every batch the loader
schedules):

  1. dataset A (20 samples), 3 steps, save; the checkpoint config is then
     stripped of data_step_offset / dataset_switches (the format of the
     checkpoints written before them, e.g. run1),
  2. dataset B (A minus 3 samples, one caption edited) resumed WITHOUT the
     flag: refused, the message names --resume-new-dataset; WITH the flag
     and another hard key changed (--seed): still refused; the flag without
     a checkpoint (or without --resume): refused,
  3. B WITH the flag: steps 4..7 read B's data steps 0..3 (the first batch
     is a fresh B loader's step 0: same ids, epoch 0), the NOTE names the
     old -> new fingerprint / n_samples, and every saved config records
     data_step_offset 3 and the switch,
  4. run2's own step-4 checkpoint resumed WITH the flag (no-op) and WITHOUT
     it: both read the same batches (ids + epochs) and reproduce the losses
     of the uninterrupted run 3,
  5. the old-format checkpoint of 1 resumed on A without the flag continues
     at data step 3 (data_step_offset 0), no "does not record" warning.

Run: python tests/test_resume_new_dataset.py (or via pytest: test_cli).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

RECORD_ENV = "KREA2_TEST_RECORD_BATCH_IDS"


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------


def _config(**kw):
    import train_full
    cfg = {k: 0 for k in train_full.RESUME_HARD_KEYS + train_full.RESUME_SOFT_KEYS}
    cfg.update(layer_loop="scan", ema_decay=0.9, dataset_fingerprint="a" * 64,
               n_samples=20)
    cfg.update(kw)
    return cfg


def _expect_exit(fn, *needles, absent=()):
    try:
        fn()
    except SystemExit as e:
        msg = str(e)
        for n in needles:
            assert n in msg, (n, msg)
        for n in absent:
            assert n not in msg, (n, msg)
        return msg
    raise AssertionError("no SystemExit")


def test_check_resume_config_new_dataset(capsys):
    import train_full
    check = train_full.check_resume_config
    saved = _config()
    new = _config(dataset_fingerprint="b" * 64, n_samples=17)
    # without the flag a new dataset is a hard error that names the flag
    _expect_exit(lambda: check(saved, new, 3), "dataset_fingerprint: checkpoint",
                 "n_samples: checkpoint", "--resume-new-dataset")
    # only the fingerprint differs (recaptioned): same, with the hint
    _expect_exit(lambda: check(saved, _config(dataset_fingerprint="c" * 64), 3),
                 "dataset_fingerprint: checkpoint", "--resume-new-dataset",
                 absent=("n_samples:",))
    # with the flag the dataset keys are exempt ...
    assert check(saved, new, 3, new_dataset=True) == []
    # ... but no other hard key, and that error does not suggest the flag
    _expect_exit(lambda: check(saved, {**new, "seed": 1}, 3, new_dataset=True),
                 "seed: checkpoint",
                 absent=("dataset_fingerprint:", "--resume-new-dataset"))
    _expect_exit(lambda: check(saved, {**new, "batch_size": 16}, 3,
                               new_dataset=True), "batch_size: checkpoint")
    # a non-dataset hard error alone does not mention the flag either
    _expect_exit(lambda: check(saved, _config(seed=1), 3),
                 absent=("--resume-new-dataset",))
    # data_step_offset / dataset_switches are derived: never compared, and
    # a checkpoint without them (older format) does not warn
    assert check(_config(data_step_offset=3, dataset_switches=[{"step": 3}]),
                 _config(data_step_offset=0, dataset_switches=[]), 9) == []
    assert "WARNING" not in capsys.readouterr().out


def test_resume_data_position(capsys):
    import train_full
    pos = train_full.resume_data_position
    old_a = _config()  # older format: no data_step_offset
    b = _config(dataset_fingerprint="b" * 64, n_samples=17)
    # normal resumes: the checkpoint's offset (0 when unrecorded / no config)
    assert pos(old_a, _config(), 5, False) == (0, None)
    assert pos(None, _config(), 5, False) == (0, None)
    assert pos({**b, "data_step_offset": 3}, b, 9, False) == (3, None)
    # the switch: offset = the resumed step, with its record
    off, sw = pos(old_a, b, 1513, True)
    assert off == 1513 and sw == {
        "step": 1513, "from_fingerprint": "a" * 64, "to_fingerprint": "b" * 64,
        "from_n_samples": 20, "to_n_samples": 17}, sw
    out = capsys.readouterr().out
    assert "NOTE: --resume-new-dataset" in out and "aaaaaaaaaaaa -> bbbbbbbbbbbb" \
        in out and "n_samples 20 -> 17" in out, out
    # only n_samples or only the fingerprint differing is a switch too
    assert pos(old_a, _config(n_samples=19), 7, True)[0] == 7
    assert pos(old_a, _config(dataset_fingerprint="c" * 64), 7, True)[0] == 7
    # flag with the checkpoint's own dataset: no-op, keeps its offset
    assert pos({**b, "data_step_offset": 1513}, b, 2000, True) == (1513, None)
    assert "nothing to switch" in capsys.readouterr().out
    # a checkpoint that cannot say what it was trained on
    _expect_exit(lambda: pos(None, b, 5, True), "no saved run configuration")
    no_fp = {k: v for k, v in old_a.items() if k != "dataset_fingerprint"}
    _expect_exit(lambda: pos(no_fp, b, 5, True), "dataset_fingerprint")


def test_cli():
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
           "PYTHONUNBUFFERED": "1"}
    r = subprocess.run([sys.executable, __file__], env=env,
                       capture_output=True, text=True, timeout=3600)
    sys.stdout.write(r.stdout[-6000:])
    if r.returncode or "ALL RESUME-NEW-DATASET TESTS PASSED" not in r.stdout:
        sys.stdout.write(r.stderr[-6000:])
        raise AssertionError("resume-new-dataset end-to-end test failed")


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def train_recording(argv):
    """train_full.main(argv) with every batch the loader schedules appended
    to $KREA2_TEST_RECORD_BATCH_IDS as {"epoch", "ids"} (dataset ids). The
    loader schedules (and records) batches in the order they are consumed,
    prefetching a few past the last step."""
    import train_full
    path = os.environ[RECORD_ENV]

    class Recording(train_full.BucketBatchLoader):
        def _schedule(self, idx, epoch):
            with open(path, "a") as f:
                f.write(json.dumps({"epoch": int(epoch), "ids": [
                    int(i) for i in self.dataset.ids[idx]]}) + "\n")
            return super()._schedule(idx, epoch)

    train_full.BucketBatchLoader = Recording
    train_full.main(argv)


def main():
    import numpy as np

    import test_train_full as T
    from test_data import make_fake_latent_cache
    from test_parity import get_reference
    from test_qwen3_text import make_tiny_text_encoder_dir
    from krea2_jax import fullft_ckpt
    from krea2_jax.data import BucketBatchLoader, BucketDataset
    from krea2_jax.qwen3_text import load_tokenizer
    from krea2_jax.text_template import CaptionTemplate

    import torch
    from safetensors.torch import save_file

    tmp = Path(tempfile.mkdtemp(prefix="test_resume_new_dataset_"))
    mmdit = get_reference()
    torch.manual_seed(0)
    model = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**T.TINY))
    g = torch.Generator().manual_seed(1)
    sd = {k: torch.randn(v.shape, generator=g) * 0.02
          for k, v in model.state_dict().items()}
    base = tmp / "base.safetensors"
    save_file(sd, str(base))
    te_dir, _ = make_tiny_text_encoder_dir(tmp / "te")
    T.TEXT_ENCODER = te_dir

    # A: test_train_full's 20 samples (2 steps per epoch at batch 8);
    # B: a filtered copy (3 samples removed, one caption edited), 17
    # samples, also 2 steps per epoch
    ids = np.random.default_rng(7).choice(10_000, size=20, replace=False)
    rows = [{"id": int(i), "bucket_w": T.BUCKETS[k % 2][0],
             "bucket_h": T.BUCKETS[k % 2][1],
             "artist_name": "" if k % 5 == 0 else f"artist {k % 3}",
             "tar": f"data/train-{k % 2:04d}.tar"}
            for k, i in enumerate(ids.tolist())]
    cache_a, cache_b = tmp / "data_a", tmp / "data_b"
    make_fake_latent_cache(cache_a, rows, lat_channels=T.TINY["channels"])
    rows_b = [dict(r) for k, r in enumerate(rows) if k not in (2, 9, 15)]
    rows_b[4]["long"] = f"long caption of {rows_b[4]['id']} (recaptioned)"
    make_fake_latent_cache(cache_b, rows_b, lat_channels=T.TINY["channels"])

    template = CaptionTemplate(load_tokenizer(te_dir), text_rows=T.TEXT_ROWS)

    def data_steps(cache, steps):
        """[{"epoch", "ids"}] of data steps `steps` of a fresh loader."""
        ds = BucketDataset(cache, template, img_tokens=T.IMG_TOKENS, patch=2,
                           caption_probs=(0.5, 0.3, 0.2), artist_prob=0.5,
                           verify_digests=False)
        loader = BucketBatchLoader(ds, 8, 0)
        try:
            return ds, [{"epoch": s // loader.steps_per_epoch,
                         "ids": [int(i) for i in ds.ids[loader.indices(s)]]}
                        for s in steps]
        finally:
            loader.close()

    ds_a, a_steps = data_steps(cache_a, range(6))
    ds_b, b_steps = data_steps(cache_b, range(6))
    assert ds_b.n == 17 and ds_a.fingerprint() != ds_b.fingerprint()
    # B's data step 0 differs from its step 3 (what the switch run would
    # read without the offset), so the checks below can tell them apart
    assert b_steps[0] != b_steps[3], b_steps

    short = ("--save-every", "2", "--export-every", "0")

    def run(cache, out, extra, ok=True):
        rec = tmp / f"{out.name}.{len(list(tmp.glob(out.name + '.*.ids')))}.ids"
        cmd = T.train_cmd(base, cache, out, (*short, *extra))
        cmd = [sys.executable, __file__, "--train", *cmd[2:]]
        proc = subprocess.run(
            cmd, env={**os.environ, "PYTHONUNBUFFERED": "1", RECORD_ENV: str(rec)},
            capture_output=True, text=True, timeout=900)
        if ok and proc.returncode != 0:
            print(proc.stdout + proc.stderr)
            raise AssertionError(f"train_full.py {extra} failed")
        if not ok:
            assert proc.returncode != 0, proc.stdout + proc.stderr
            assert "resumed train state" not in proc.stdout, proc.stdout
            return proc.stderr
        steps = {int(s): (float(l), float(gn))
                 for s, l, gn in T.STEP_RE.findall(proc.stdout)}
        batches = [json.loads(line) for line in rec.read_text().splitlines()]
        return steps, batches[:len(steps)], proc.stdout

    def config(out, step):
        return fullft_ckpt.read_meta_at(str(out / "ckpt"), step)["config"]

    # --- 1. run1 on A, 3 steps; then make its checkpoint config old-format ---
    out1 = tmp / "run1"
    steps1, batches1, _ = run(cache_a, out1, ("--steps", "3"))
    assert sorted(steps1) == [1, 2, 3] and batches1 == a_steps[:3], batches1
    c1 = config(out1, 3)
    assert c1["data_step_offset"] == 0 and c1["dataset_switches"] == [], c1
    meta_file = out1 / "ckpt" / "3" / "meta" / "metadata"
    meta = json.loads(meta_file.read_text())
    del meta["config"]["data_step_offset"], meta["config"]["dataset_switches"]
    meta_file.write_text(json.dumps(meta))
    assert "data_step_offset" not in config(out1, 3)
    print("1. run1 on dataset A (3 steps), checkpoint made old-format: ok")

    def copy_ckpt(src, step, name):
        out = tmp / name
        shutil.copytree(src / "ckpt" / str(step), out / "ckpt" / str(step))
        return out

    # --- 2. refusals (all before the base checkpoint load) ---
    err = run(cache_b, copy_ckpt(out1, 3, "noflag"), ("--resume", "--steps", "7"),
              ok=False)
    for needle in ("dataset_fingerprint: checkpoint", "n_samples: checkpoint 20, "
                   "now 17", "--resume-new-dataset"):
        assert needle in err, (needle, err[-2000:])
    err = run(cache_b, copy_ckpt(out1, 3, "flagseed"),
              ("--resume", "--resume-new-dataset", "--steps", "7", "--seed", "1"),
              ok=False)
    assert "seed: checkpoint 0, now 1" in err, err[-2000:]
    assert "dataset_fingerprint:" not in err, err[-2000:]
    err = run(cache_b, tmp / "nockpt", ("--resume", "--resume-new-dataset"),
              ok=False)
    assert "--resume-new-dataset" in err and "no checkpoint" in err, err[-2000:]
    assert not (tmp / "nockpt" / "ckpt").exists()
    err = run(cache_b, tmp / "noresume", ("--resume-new-dataset",), ok=False)
    assert "neither --resume nor --resume-step" in err, err[-2000:]
    print("2. B without the flag, flag + --seed change, flag without a "
          "checkpoint / --resume: refused: ok")

    # --- 3. run2: the switch onto B at step 3 ---
    out2 = copy_ckpt(out1, 3, "run2")
    steps2, batches2, log2 = run(cache_b, out2, (
        "--resume", "--resume-new-dataset", "--steps", "7"))
    assert sorted(steps2) == [4, 5, 6, 7], sorted(steps2)
    assert batches2[0] == b_steps[0], (batches2[0], b_steps[0])
    assert batches2 == b_steps[:4], batches2
    fa, fb = ds_a.fingerprint(), ds_b.fingerprint()
    for needle in ("NOTE: --resume-new-dataset: continuing the train state of "
                   "step 3", f"dataset_fingerprint {fa[:12]} -> {fb[:12]}",
                   "n_samples 20 -> 17", "data_step_offset 3",
                   "data: step 4 reads data step 0 (epoch 0, batch 0/2; "
                   "data_step_offset 3)"):
        assert needle in log2, (needle, log2)
    # the only warning is the longer --steps (run1 ran 3): the old-format
    # config's missing keys are not reported as unrecorded
    assert log2.count("WARNING") == 1 and "  steps: checkpoint 3, now 7" in log2
    assert "does not record" not in log2, log2
    switch = {"step": 3, "from_fingerprint": fa, "to_fingerprint": fb,
              "from_n_samples": 20, "to_n_samples": 17}
    for s in (4, 6, 7):
        c = config(out2, s)
        assert c["data_step_offset"] == 3 and c["dataset_switches"] == [switch], c
        assert c["dataset_fingerprint"] == fb and c["n_samples"] == 17, c
    print("3. switch onto B at step 3: steps 4..7 read B's data steps 0..3 "
          "(first batch == fresh B loader step 0), configs record "
          "data_step_offset 3 + the switch: ok")

    # --- 4. run2 resumes its own step-4 checkpoint with / without the flag ---
    for name, flag in (("run2_noop", ("--resume-new-dataset",)),
                       ("run2_plain", ())):
        out = copy_ckpt(out2, 4, name)
        steps, batches, log = run(cache_b, out, ("--resume", *flag, "--steps", "7"))
        assert sorted(steps) == [5, 6, 7], sorted(steps)
        assert batches == batches2[1:], (name, batches, batches2)
        assert "data: step 5 reads data step 1 (epoch 0, batch 1/2; " \
               "data_step_offset 3)" in log, log
        assert ("nothing to switch" in log) == bool(flag), log
        assert "continuing the train state" not in log and "WARNING" not in log
        diffs = [abs(steps[s][0] - steps2[s][0]) for s in (5, 6, 7)]
        assert max(diffs) <= 1e-5, (name, diffs)
        c = config(out, 7)
        assert c["data_step_offset"] == 3 and c["dataset_switches"] == [switch], c
        print(f"4. run2 step 4 resumed {'with' if flag else 'without'} the "
              f"flag: same batches as the uninterrupted run2, loss diffs "
              + ", ".join(f"{d:.1e}" for d in diffs) + ": ok")

    # --- 5. the old-format checkpoint continues on A at offset 0 ---
    out5 = copy_ckpt(out1, 3, "old_format")
    steps5, batches5, log5 = run(cache_a, out5, ("--resume", "--steps", "5"))
    assert sorted(steps5) == [4, 5] and batches5 == a_steps[3:5], batches5
    assert "data: step 4 reads data step 3 (epoch 1, batch 1/2; " \
           "data_step_offset 0)" in log5, log5
    assert log5.count("WARNING") == 1 and "  steps: checkpoint 3, now 5" in log5
    assert "does not record" not in log5, log5
    assert config(out5, 5)["data_step_offset"] == 0
    print("5. old-format checkpoint (no data_step_offset) resumes on A at "
          "data step 3, no unrecorded-key warning: ok")

    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL RESUME-NEW-DATASET TESTS PASSED")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    if sys.argv[1:2] == ["--train"]:
        train_recording(sys.argv[2:])
    else:
        main()
