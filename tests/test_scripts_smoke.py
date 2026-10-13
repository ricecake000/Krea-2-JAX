"""CPU smoke tests for the node-side scripts, run through subprocess with a
tiny model config on 8 emulated devices:

  aot_analysis_verdict   scripts/aot_analysis.py: the activation-collective
                         verdict is NONE with --act-sharding on and FOUND
                         with it off (the partitioner moves activations)
  bench_step_modes       scripts/bench_step.py --random-weights in fwd,
                         fwdbwd and train modes (2 timed steps each), plus
                         train with --attn cudnn and train with
                         --layer-loop unroll (per-layer list blocks)
  modal_text_parity_flow scripts/modal_text_parity.py control flow (exit
                         codes, skip-on-tokenizer-failure, exit 1) with a
                         mocked subprocess runner, no Modal SDK
  modal_cache_latents_flow scripts/modal_cache_latents.py pure parts, no
                         Modal SDK: cost model vs hand-computed numbers, GPU
                         selection (5% tie -> higher img/s, failed GPUs and
                         OOM/error ladder rows ignored), batch ladder (stub
                         encoder), HF token write check, worker command +
                         env + timeout, the commit-per-shard stream on a
                         real stub cache_latents run, upload filter + remote
                         content check (sha256 / git blob id)

Structure: every case is a function registered with @case (name = function
name); `main` runs them all, or only those named on the command line:

    python tests/test_scripts_smoke.py                    # all cases
    python tests/test_scripts_smoke.py bench_step_modes   # one case

To add a case, define a new @case function ABOVE the runner marker at the
bottom of this file (cases run in definition order).
"""

import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

TINY = dict(
    features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
    layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3,
)
TINY_JSON = json.dumps(TINY)
# Qwen3TextConfig overrides matching TINY (hidden 64 = txtdim, 3 select
# layers = txtlayers); same shape as tests/test_qwen3_text.py's tiny encoder
TINY_TEXT = dict(
    vocab_size=600, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
    head_dim=16, intermediate_size=96, mrope_section=[4, 2, 2],
    select_layers=[0, 1, 2],
)
TINY_TEXT_JSON = json.dumps(TINY_TEXT)

CASES = []


def case(fn):
    """Register a smoke-test case (runs in definition order)."""
    CASES.append(fn)
    return fn


def cpu_env(n_devices=8):
    """Environment for a subprocess on n emulated CPU devices."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_devices}"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    return env


def run_script(script, *args, env=None, timeout=900, expect_ok=True):
    """Run scripts/<script> with args; returns (returncode, stdout+stderr).
    Raises with the output tail when the exit status is not as expected."""
    cmd = [sys.executable, str(REPO / "scripts" / script), *map(str, args)]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env or cpu_env(), capture_output=True,
                          text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    ok = proc.returncode == 0
    if ok != expect_ok:
        print(out[-6000:])
        raise AssertionError(
            f"{script} {' '.join(map(str, args))}: exit {proc.returncode} "
            f"(expected {'0' if expect_ok else 'non-zero'})")
    shown = ["<json>" if str(a).startswith("{") else str(a) for a in args]
    print(f"    {script} {' '.join(shown[:8])}{' ...' if len(shown) > 8 else ''}: "
          f"{time.time() - t0:.0f}s")
    return proc.returncode, out


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

AOT_ARGS = (
    "--model-config", TINY_JSON, "--text-model-config", TINY_TEXT_JSON,
    "--batch-size", 16, "--img-tokens", 256, "--bucket", "192x320",
    "--text-rows", 64, "--qchunk", 128, "--emulate-devices", 8,
    "--remat-policy", "dots", "--act-numel", 2e5,
)
VERDICT_RE = re.compile(r"^activation collectives: (NONE|FOUND (\d+))", re.M)


@case
def aot_analysis_verdict():
    verdicts = {}
    for mode in ("on", "off"):
        _, out = run_script("aot_analysis.py", *AOT_ARGS, "--act-sharding", mode)
        for needle in ("params: ", "sequence: bucket 192x320px -> 20x12 = 240 "
                       "valid of 256 img rows", "text tower: ", "lowered in",
                       "compiled (cpu", "memory_analysis", "cost_analysis",
                       "analytic text-tower flops", "collectives: ",
                       "CPU backend caveats"):
            assert needle in out, f"--act-sharding {mode}: no {needle!r}\n{out}"
        m = VERDICT_RE.search(out)
        assert m, f"no verdict line:\n{out}"
        verdicts[mode] = m.group(1)
        print(f"    --act-sharding {mode}: activation collectives: {m.group(1)}")
    assert verdicts["on"] == "NONE", verdicts
    assert verdicts["off"].startswith("FOUND"), verdicts


@case
def bench_step_modes():
    common = ("--random-weights", "--model-config", TINY_JSON,
              "--text-model-config", TINY_TEXT_JSON, "--batch-size", 8,
              "--img-tokens", 64, "--bucket", "96x160", "--text-rows", 16,
              "--qchunk", 128, "--steps", 2, "--warmup", 1)
    for mode in ("fwd", "fwdbwd", "train"):
        _, out = run_script("bench_step.py", *common, "--mode", mode)
        assert "bucket 96x160px (60 of 64 img rows" in out, out
        if mode == "train":
            assert "text tower (random bf16, replicated)" in out, out
            assert "+ text tower forward" in out, out
        m = re.search(r"step time: median ([\d.]+) ms over (\d+) steps", out)
        assert m and int(m.group(2)) == 2, out
        assert re.search(r"analytic [\d.]+ TFLOP/s per device", out), out
        assert "samples/s" in out, out
        if mode != "fwd":
            loss = float(re.search(r"loss ([\d.eE+-]+|nan|inf)", out).group(1))
            assert loss == loss and loss < 1e3, f"{mode}: bad loss {loss}"
        print(f"    bench_step --mode {mode}: median {m.group(1)} ms")
    # --attn cudnn (jax.nn.dot_product_attention; its "xla" path on CPU)
    _, out = run_script("bench_step.py", *common, "--mode", "train",
                        "--attn", "cudnn")
    assert "attn cudnn" in out, out
    m = re.search(r"step time: median ([\d.]+) ms over (\d+) steps", out)
    assert m and int(m.group(2)) == 2, out
    loss = float(re.search(r"loss ([\d.eE+-]+|nan|inf)", out).group(1))
    assert loss == loss and loss < 1e3, f"cudnn: bad loss {loss}"
    print(f"    bench_step --mode train --attn cudnn: median {m.group(1)} ms")
    # --layer-loop unroll: blocks as a per-layer list, unrolled layer loop
    _, out = run_script("bench_step.py", *common, "--mode", "train",
                        "--layer-loop", "unroll")
    assert "layer-loop unroll" in out, out
    m = re.search(r"step time: median ([\d.]+) ms over (\d+) steps", out)
    assert m and int(m.group(2)) == 2, out
    loss = float(re.search(r"loss ([\d.eE+-]+|nan|inf)", out).group(1))
    assert loss == loss and loss < 1e3, f"unroll: bad loss {loss}"
    print(f"    bench_step --mode train --layer-loop unroll: median "
          f"{m.group(1)} ms")


PARITY_ARGS = (
    "--random-weights", "--model-config", TINY_JSON, "--width", 128,
    "--height", 192, "--txt-len", 7, "--batch", 4, "--torch-device", "cpu",
    "--seed", 3,
)
RELNORM_RE = re.compile(r"worst per-sample relnorm ([\d.eE+-]+|inf|nan)")


@case
def parity_full_tiny():
    """scripts/parity_full.py: official torch (math attention) vs JAX on a
    fabricated tiny checkpoint, ref and aiter-emulation attention, with and
    without activation sharding; bf16 within the default tolerance; and a
    negative control (JAX ignores the text mask) that must fail."""
    # relnorm self-check: float64 before any norm (a float32 norm of 1e20s
    # overflows and used to report 0), non-finite -> inf -> failure
    import numpy as np
    sys.path.insert(0, str(REPO / "scripts"))
    from parity_full import compare, relnorm

    ref = np.full((4, 64, 16), 1e20, np.float32)
    rn = relnorm(2 * ref, ref)
    assert abs(rn - 1.0) < 1e-6, rn
    bad = ref.copy()
    bad[1, 3, 5] = np.nan
    assert relnorm(bad, ref) == float("inf")
    assert relnorm(ref, bad) == float("inf")
    assert not compare(bad, ref, np.zeros(4, int), 1e9)
    print(f"    relnorm(2*ref, ref) with ref=1e20 f32: {rn:.6f}; NaN -> inf (FAILED)")
    runs = [
        ("f32", "ref", "off", "1e-4"),
        ("f32", "aiter", "on", "1e-4"),
        ("bf16", "ref", "on", "5e-2"),
    ]
    for dtype, attn, act, tol in runs:
        extra = ("--aiter-kernel", "emul") if attn == "aiter" else ()
        _, out = run_script("parity_full.py", *PARITY_ARGS, "--dtype", dtype,
                            "--attn", attn, "--act-sharding", act, "--tol", tol,
                            "--sequential", *extra)
        assert "PARITY PASSED" in out, out
        assert "attention=math (patch_reference_for_cpu)" in out, out
        rn = float(RELNORM_RE.search(out).group(1))
        if dtype == "f32":
            assert rn < 1e-5, (dtype, attn, act, rn)
        print(f"    parity_full {dtype} attn={attn} act={act}: relnorm {rn:.2e}")
    _, out = run_script("parity_full.py", *PARITY_ARGS, "--dtype", "f32",
                        "--tol", "1e-5", "--negative-control", expect_ok=False)
    assert "PARITY FAILED" in out, out
    print(f"    negative control: relnorm "
          f"{float(RELNORM_RE.search(out).group(1)):.2e} (FAILED as expected)")


NODE_ATTN_ARGS = (
    "--seq", 512, "--heads", 4, "--kv-heads", 2, "--head-dim", 32,
    "--batch", 4, "--iters", 2, "--warmup", 1,
)


@case
def node_attention_parity_emul():
    """scripts/node_attention_parity.py --kernel emul: kernel-level fwd+grads
    vs fp32 truth (f32 unsharded, bf16 under shard_map), the DiT-level
    ref-vs-aiter check with random tiny weights, the --dit-truth f32 mode
    (ref-bf16 and emul-bf16 vs a ref-f32 truth; rejects --dtype f32, a
    non-positive --dit-truth-ratio and a missing weight source), and a
    negative-control kernel that ignores the logical lengths (must fail)."""
    # dit_report self-check (in-process): relnorm via parity_full.relnorm,
    # float64 before any norm (a float32 norm of 1e20s overflowed and the
    # 100%-error case used to report relnorm 0 / PASS); NaN -> inf -> FAIL
    import contextlib
    import io

    import numpy as np
    sys.path.insert(0, str(REPO / "scripts"))
    from node_attention_parity import dit_report, truth_report

    ref = np.full((2, 64, 16), 1e20, np.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        passed = dit_report("big", 2 * ref, ref, tol=1e9)
        assert dit_report("big", 2 * ref, ref, tol=0.5) is False
    rns = [float(x) for x in re.findall(r"sample \d+: relnorm ([\d.eE+-]+)", buf.getvalue())]
    assert passed and len(rns) == 4 and all(abs(r - 1.0) < 1e-6 for r in rns), buf.getvalue()
    bad = ref.copy()
    bad[1, 3, 5] = np.nan
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert dit_report("nan", bad, ref, tol=1e9) is False
        assert dit_report("nan", ref, bad, tol=1e9) is False
    assert "worst relnorm inf" in buf.getvalue() and "FAIL" in buf.getvalue(), buf.getvalue()
    print(f"    dit_report: 2*ref vs ref=1e20 f32 -> relnorm {rns[0]:.6f}; "
          "NaN -> inf (FAIL)")
    # truth_report self-check: truth = ones, x = truth * (1 + e) has relnorm
    # e per sample; PASS iff every sample has test_err <= ratio*ref_err + 1e-3
    truth = np.ones((2, 64, 16), np.float64)

    def scaled(*e):
        return truth * (1.0 + np.asarray(e, np.float64))[:, None, None]

    def tr(e_ref, e_test, ratio=1.5):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = truth_report("fwd", scaled(*e_ref), scaled(*e_test), truth, ratio)
        return res, buf.getvalue()

    res, out = tr((1e-2, 2e-2), (1.2e-2, 2.9e-2))  # ratios 1.2, 1.45
    assert res is True and "FAIL" not in out and "verdict PASS" in out, out
    assert re.search(r"truth fwd: worst ratio 1\.45 ", out), out
    # sample 1 over its allowance 1.5*2e-2 + 1e-3 = 3.1e-2; sample 0 fine
    res, out = tr((1e-2, 2e-2), (1.2e-2, 3.3e-2))
    lines = out.splitlines()
    assert res is False and "verdict FAIL" in out, out
    assert "FAIL" not in lines[0] and "FAIL" in lines[1], out
    assert "allowance 3.10e-02" in lines[1], out
    nan_test = scaled(1e-2, 2e-2)
    nan_test[1, 3, 5] = np.nan
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = truth_report("fwd", scaled(1e-2, 2e-2), nan_test, truth, 1.5)
    out = buf.getvalue()
    assert res is False and "non-finite" in out and "test relnorm inf" in out, out
    # 1e-3 absolute floor: ref_err 0 -> allowance 1e-3
    res, out = tr((0.0, 0.0), (5e-4, 5e-4))
    assert res is True and "verdict PASS" in out, out
    res, out = tr((0.0, 0.0), (5e-4, 2e-3))
    assert res is False and "FAIL" in out.splitlines()[1], out
    print("    truth_report: within ratio PASS, sample 1 over allowance FAIL, "
          "NaN FAIL, 1e-3 floor (5e-4 PASS / 2e-3 FAIL)")
    _, out = run_script("node_attention_parity.py", *NODE_ATTN_ARGS,
                        "--kernel", "emul", "--dtype", "f32")
    assert "kernel check PASSED" in out and "ALL ATTENTION PARITY CHECKS PASSED" in out, out
    for q in ("out", "dq", "dk", "dv"):
        assert re.search(rf"^  {q} +test ", out, re.M), (q, out)
    assert re.search(r"test fwd\+bwd +[\d.]+ ms/call", out), out
    _, out = run_script("node_attention_parity.py", *NODE_ATTN_ARGS,
                        "--kernel", "emul", "--dtype", "bf16", "--shard")
    assert "mesh=4" in out and "kernel check PASSED" in out, out
    _, out = run_script("node_attention_parity.py", *NODE_ATTN_ARGS,
                        "--kernel", "emul", "--dtype", "f32", "--skip-kernel",
                        "--random-weights", "--model-config", TINY_JSON,
                        "--width", 128, "--height", 192, "--txt-len", 7,
                        "--act-sharding", "on", "--dit-grad")
    assert "dit check PASSED" in out and "d/d(img)" in out, out
    _, out = run_script("node_attention_parity.py", *NODE_ATTN_ARGS,
                        "--kernel", "emul-nomask", "--dtype", "f32",
                        "--iters", 0, expect_ok=False)
    assert "kernel check FAILED" in out, out
    # f32 truth: ratio 4 because emul and ref are both bf16 with a different
    # op order; this checks the code path, not aiter numerics
    _, out = run_script("node_attention_parity.py", *NODE_ATTN_ARGS,
                        "--kernel", "emul", "--dtype", "bf16", "--skip-kernel",
                        "--random-weights", "--model-config", TINY_JSON,
                        "--width", 128, "--height", 192, "--txt-len", 7,
                        "--act-sharding", "on", "--dit-grad",
                        "--dit-truth", "f32", "--dit-truth-ratio", 4)
    assert "truth check PASSED" in out and "dit check PASSED" in out, out
    assert re.search(r"truth fwd sample \d+: ref\(bf16\) relnorm", out), out
    assert re.search(r"truth d/d\(img\) sample", out), out
    assert "(info)" in out, out
    _, out = run_script("node_attention_parity.py", *NODE_ATTN_ARGS,
                        "--kernel", "emul", "--dtype", "f32", "--skip-kernel",
                        "--random-weights", "--model-config", TINY_JSON,
                        "--dit-truth", "f32", expect_ok=False)
    assert "needs --dtype bf16" in out, out
    _, out = run_script("node_attention_parity.py", *NODE_ATTN_ARGS,
                        "--kernel", "emul", "--dtype", "bf16", "--skip-kernel",
                        "--random-weights", "--model-config", TINY_JSON,
                        "--dit-truth", "f32", "--dit-truth-ratio", 0,
                        expect_ok=False)
    assert "--dit-truth-ratio must be finite and > 0" in out, out
    _, out = run_script("node_attention_parity.py", *NODE_ATTN_ARGS,
                        "--kernel", "emul", "--dtype", "bf16", "--skip-kernel",
                        "--dit-truth", "f32", expect_ok=False)
    assert "pass --base-ckpt or --random-weights" in out, out
    print("    node_attention_parity: emul passes (f32, bf16+shard_map, dit, "
          "dit f32 truth), truth rejects --dtype f32 / ratio 0 / no weights, "
          "negative control fails")


@case
def modal_text_parity_flow():
    """scripts/modal_text_parity.py without Modal (the import is guarded):
    run_checks aggregates the subprocess exit codes, a failed tokenizer-only
    check skips the model checks, parity_impl reports a failed download, and
    finish() writes the log, then exits 1 with the one-line reason."""
    import tempfile
    from types import SimpleNamespace

    sys.path.insert(0, str(REPO / "scripts"))
    import modal_text_parity as mtp

    def fake_runner(codes):
        calls = []

        def runner(cmd, **kw):
            assert kw == {"cwd": mtp.REMOTE_REPO, "capture_output": True,
                          "text": True}, kw
            calls.append(cmd)
            tag = "tok" if "--tokenizer-only" in cmd else cmd[
                cmd.index("--dtype") + 1]
            return SimpleNamespace(returncode=codes.get(tag, 0),
                                   stdout=f"out {tag}", stderr="")
        return runner, calls

    runner, calls = fake_runner({})
    lines, failures = mtp.run_checks("/w", 8, ["f32", "bf16"], 4, runner=runner)
    assert failures == [] and len(calls) == 3, (failures, calls)
    assert "[exit 0]\n" in lines and "out bf16" in lines, lines

    runner, calls = fake_runner({"bf16": 3})
    _, failures = mtp.run_checks("/w", 8, ["f32", "bf16"], 4, runner=runner)
    assert failures == ["bf16 parity exit 3"] and len(calls) == 3, failures

    runner, calls = fake_runner({"tok": 1})
    lines, failures = mtp.run_checks("/w", 8, ["f32", "bf16"], 4,
                                     runner=runner)
    assert len(calls) == 1 and "--tokenizer-only" in calls[0], calls
    assert failures == ["tokenizer-only check exit 1",
                        "model checks skipped (f32,bf16)"], failures

    runner, _ = fake_runner({})
    text, ok = mtp.parity_impl(8, ["f32"], 4, download=lambda: "/w",
                               runner=runner)
    assert ok and text.splitlines()[-1] == "parity result: PASS", text
    runner, _ = fake_runner({"tok": 2})
    text, ok = mtp.parity_impl(8, ["f32"], 4, download=lambda: "/w",
                               runner=runner)
    assert not ok and text.splitlines()[-1] == (
        "parity result: FAIL (tokenizer-only check exit 2; model checks "
        "skipped (f32))"), text

    def broken():
        raise OSError("no network")
    text, ok = mtp.parity_impl(8, ["f32"], 4, download=broken, runner=runner)
    assert not ok and "OSError: no network" in text, text
    assert text.endswith("parity result: FAIL (weight download failed)"), text

    import contextlib
    import io
    with tempfile.TemporaryDirectory() as d:
        with contextlib.redirect_stdout(io.StringIO()):
            path = mtp.finish("all good\nparity result: PASS", True,
                              results_dir=d)
        assert path.read_text().endswith("PASS"), path
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mtp.finish(text, False, results_dir=Path(d) / "f")
        except SystemExit as e:
            msg = str(e)
            assert msg.startswith("parity result: FAIL (weight download "
                                  "failed) (log: "), msg
            assert "\n" not in msg, msg
        else:
            raise AssertionError("finish() did not exit on failure")
        logs = list((Path(d) / "f").glob("text_parity_*.log"))
        assert len(logs) == 1 and logs[0].read_text() == text, logs
    print("    modal_text_parity: exit codes aggregated, tokenizer failure "
          "skips model checks, download failure reported, log written "
          "before exit 1")


@case
def modal_cache_latents_flow():
    """scripts/modal_cache_latents.py without Modal (the import is guarded):
    pricing/cost math against hand-computed numbers, the selection rule,
    the ladder's OOM handling, token parsing, the worker command, the
    stream that commits the volume after every shard (real cache_latents
    --vae-stub run on a tiny dataset), the upload filter and finalize with
    a fake HfApi (write check, privacy, content check)."""
    import contextlib
    import hashlib
    import io
    import tempfile
    from math import isclose
    from types import SimpleNamespace

    import torch

    sys.path.insert(0, str(REPO / "scripts"))
    import modal_cache_latents as mcl

    def close(a, b):
        return isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

    # GPU strings / prices
    assert mcl.gpu_key("h100!") == "H100" and mcl.gpu_key("B200+") == "B200"
    assert mcl.parse_gpus(" l4, A10 ,H100!,l4") == ["L4", "A10", "H100!"]
    for bad in ("T4", "B200:2", "A10G"):
        try:
            mcl.gpu_key(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"gpu_key accepted {bad!r}")

    # cores_needed = max(4, ceil(1.5 * img_s / decode) + 1)
    assert mcl.cores_needed(2.0, 1.0) == 4        # ceil(3) + 1 = 4
    assert mcl.cores_needed(1.0, 10.0) == 4       # floor of 4
    assert mcl.cores_needed(10.0, 0.5) == 31      # ceil(30) + 1
    assert mcl.cores_needed(3.1, 1.0) == 6        # ceil(4.65) + 1

    # cost: L4 at 2 img/s, 1 img/s/thread decode -> 4 cores
    # hourly = 0.80 + 4 * 0.0472 + 32 * 0.00799 = 1.24448 $/h
    # $/1k = 1.24448 / 7200 * 1000 = 0.172844...; hours = 96848 / 7200
    # 1 ctr: 1.24448 * (13.451111 + 0.05) = 16.8018627...
    # 4 ctr: 1.24448 * (13.451111 + 0.2) = 16.9885347..., wall 3.4127777 h
    res_l4 = {"gpu_requested": "L4", "device_name": "NVIDIA L4",
              "decode_img_s_per_thread": 1.0,
              "ladder": [{"batch": 4, "img_s": 1.5, "peak_gib": 6.0},
                         {"batch": 8, "img_s": 2.0, "peak_gib": 11.0},
                         {"batch": 16, "error": "OOM"}]}
    best = mcl.best_of(res_l4["ladder"])
    assert best == {"best_batch": 8, "best_img_s": 2.0,
                    "best_peak_gib": 11.0}, best
    res_l4.update(best)
    r = mcl.cost_row(res_l4)
    assert r["cores"] == 4 and r["batch"] == 8, r
    assert close(r["usd_h"], 1.24448), r
    assert close(r["usd_per_1k"], 0.17284444444444444), r
    assert close(r["hours_1"], 13.45111111111111), r
    assert close(r["proj"][1]["usd"], 16.801862755555558), r
    assert close(r["proj"][4]["usd"], 16.988534755555555), r
    assert close(r["proj"][4]["wall_h"], 3.4127777777777775), r
    # worker timeout: ceil(1.5 * (13.451111 / 4 + 0.05) + 0.5) = ceil(5.619)
    assert mcl.worker_timeout_h(r) == 6 and mcl.worker_timeout_h(r, 32) == 2
    fast = dict(r, hours_1=0.1)                  # ceil(0.6125) -> min 1
    assert mcl.worker_timeout_h(fast) == 1
    # B200 at 10 img/s, 0.5 img/s/thread -> 31 cores:
    # 6.25 + 31 * 0.0472 + 0.25568 = 7.96888 $/h -> 0.2213577.. $/1k
    r = mcl.cost_row({"gpu_requested": "B200", "best_img_s": 10.0,
                      "best_batch": 16, "decode_img_s_per_thread": 0.5})
    assert r["cores"] == 31 and close(r["usd_h"], 7.96888), r
    assert close(r["usd_per_1k"], 0.2213577777777778), r
    # unusable results -> None (never selected)
    assert mcl.cost_row({"gpu_requested": "L4", "error": "boom",
                         "best_img_s": 99.0,
                         "decode_img_s_per_thread": 1.0}) is None
    all_oom = {"gpu_requested": "L4", "decode_img_s_per_thread": 1.0,
               "ladder": [{"batch": 4, "error": "OOM"}]}
    all_oom.update(mcl.best_of(all_oom["ladder"]))
    assert mcl.cost_row(all_oom) is None, all_oom

    # selection: lowest $/img; within 5% is a tie -> higher img/s wins
    def row(gpu, usd, img_s):
        return {"gpu": gpu, "usd_per_1k": usd, "img_s": img_s}
    rows = [row("A", 1.00, 5.0), row("B", 1.04, 20.0), row("C", 1.06, 50.0),
            None]
    assert mcl.select_gpu(rows)["gpu"] == "B"          # C is 6% off
    assert mcl.select_gpu([row("A", 1.0, 5.0), row("B", 1.05, 6.0)])[
        "gpu"] == "B"                                    # exactly 5%: tie
    assert mcl.select_gpu([row("A", 1.0, 5.0), row("B", 1.2, 60.0)])[
        "gpu"] == "A"
    assert mcl.select_gpu([None, None]) is None

    # report from bench results (one usable, one OOM-only, one error)
    results = [res_l4, all_oom | {"gpu_requested": "A10"},
               {"gpu_requested": "B300", "error": "Traceback\nRuntimeError: "
                "no capacity"}]
    rows = [mcl.cost_row(x) for x in results]
    choice = mcl.select_gpu(rows)
    md = mcl.render_bench_report(results, rows, choice,
                                 prep={"whoami": "ricecake",
                                       "can_write": True,
                                       "write_reason": "token role write"})
    cmd = ("modal run --detach scripts/modal_cache_latents.py --build --gpu "
           "L4 --containers 4 --batch 8 --cpu 4 --worker-timeout-h 6")
    assert mcl.build_command(choice) == cmd and cmd in md, md
    # single-GPU bench (--gpus B200): its measured batch/cores in the command
    one = {"gpu_requested": "B200", "device_name": "NVIDIA B200",
           "decode_img_s_per_thread": 0.5, "ladder": [
               {"batch": 16, "img_s": 10.0, "peak_gib": 40.0},
               {"batch": 32, "error": "OOM"}]}
    one.update(mcl.best_of(one["ladder"]))
    rows1 = [mcl.cost_row(one)]
    md1 = mcl.render_bench_report([one], rows1, mcl.select_gpu(rows1))
    # 96848 / 10 / 3600 / 4 + 0.05 = 0.7225 h -> ceil(1.584) = 2
    assert ("--build --gpu B200 --containers 4 --batch 16 --cpu 31 "
            "--worker-timeout-h 2") in md1, md1
    assert "**Chosen: L4**" in md and "8:2.00 16:OOM" in md, md
    assert "- A10 " in md and "- B300 " in md and "no capacity" in md, md

    # ladder: stub encoder, OOM at batch 8 stops it, wrong shape -> error
    class OutOfMemoryError(RuntimeError):
        pass
    stub = __import__("cache_latents").StubEncoder()

    def enc(x):
        if x.shape[0] >= 8:
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate")
        return stub(x)
    imgs = [torch.rand(3, 64, 32) * 2 - 1 for _ in range(5)]
    quiet = contextlib.redirect_stdout(io.StringIO())
    with quiet:
        lad = mcl.run_ladder(enc, imgs, (2, 4, 8, 16), (16, 8, 4),
                             min_images=4, min_seconds=0.0)
    assert [x["batch"] for x in lad] == [2, 4, 8], lad
    assert lad[2] == {"batch": 8, "error": "OOM"}, lad
    assert all(x["img_s"] > 0 and x["images"] >= max(4, 3 * x["batch"])
               for x in lad[:2]), lad
    with contextlib.redirect_stdout(io.StringIO()):
        lad = mcl.run_ladder(stub, imgs, (2,), (16, 4, 4), min_images=2,
                             min_seconds=0.0)
    assert lad[0]["error"].startswith("ValueError: output (2, 16, 8, 4)"), lad

    # HF token
    assert mcl.pick_hf_token({"HUGGINGFACE_TOKEN": "c",
                              "HUGGING_FACE_HUB_TOKEN": "b"}) == (
        "HUGGING_FACE_HUB_TOKEN", "b")
    env = {"HUGGINGFACE_TOKEN": "c"}
    assert mcl.setup_hf_token(env) == ("HUGGINGFACE_TOKEN", "c")
    assert env["HF_TOKEN"] == "c"
    assert mcl.pick_hf_token({"HF_TOKEN": ""}) == (None, None)

    def who(role, name="ricecake", scoped=None):
        tok = {"role": role}
        if scoped is not None:
            tok["fineGrained"] = {"scoped": scoped, "global": []}
        return {"name": name, "auth": {"accessToken": tok}}
    assert mcl.token_can_write(who("write"))[0] is True
    assert mcl.token_can_write(who("read"))[0] is False
    assert mcl.token_can_write(who("write", name="other"))[0] is False
    assert mcl.token_can_write(who("fineGrained", scoped=[
        {"entity": {"type": "user", "name": "ricecake"},
         "permissions": ["repo.content.read", "repo.write"]}]))[0] is True
    assert mcl.token_can_write(who("fineGrained", scoped=[
        {"entity": {"type": "user", "name": "ricecake"},
         "permissions": ["repo.content.read"]}]))[0] is False
    # write on single repos only: unknown, must not block (create_repo decides)
    assert mcl.token_can_write(who("fineGrained", scoped=[
        {"entity": {"type": "dataset",
                    "name": "ricecake/danbooru-curated-v2-latents"},
         "permissions": ["repo.content.read", "repo.write"]}]))[0] is None
    assert mcl.token_can_write(who("mystery"))[0] is None
    assert mcl.token_can_write({})[0] is None

    # worker / finalize commands
    assert mcl.worker_cmd(2, 4, 16, 12, python="py") == [
        "py", "scripts/cache_latents.py", "--out", "/vol/krea2",
        "--num-workers", "4", "--worker-index", "2", "--batch", "16",
        "--decode-threads", "12", "--download-dir", "/tmp/dl/w2"]
    assert mcl.finalize_cmd(python="py") == [
        "py", "scripts/cache_latents.py", "--out", "/vol/krea2",
        "--write-manifest", "--download-dir", "/tmp/dl/manifest"]

    env = mcl.worker_env({"PATH": "/bin", "OMP_NUM_THREADS": "64"})
    assert env == {"PATH": "/bin", "PYTHONUNBUFFERED": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, env

    # upload filter (same semantics as huggingface_hub's ignore_patterns)
    paths = ["metadata.parquet", "latents/manifest.json",
             "latents/train-0000.safetensors",
             "latents/train-0000.safetensors.tmp", "latents/manifest.json.tmp",
             "_download/manifest/metadata.parquet",
             "_download/w0/data/train-0001.tar",
             ".cache/huggingface/upload/latents/x.metadata"]
    kept = [p for p in paths if mcl.should_upload(p)]
    assert kept == paths[:3], kept
    try:
        from huggingface_hub.utils import filter_repo_objects
    except ImportError:
        pass
    else:
        assert list(filter_repo_objects(
            paths, ignore_patterns=mcl.UPLOAD_IGNORE)) == kept

    # commit after every shard: real cache_latents --vae-stub run, 1 worker
    from test_cache_latents import make_fake_dataset

    class FakeVol:
        def __init__(self, fail_first=False):
            self.commits, self.fail = 0, fail_first

        def commit(self):
            self.commits += 1
            if self.fail:
                self.fail = False
                raise RuntimeError("commit conflict")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_fake_dataset(d / "src")
        vol, logs = FakeVol(fail_first=True), []
        cmd = mcl.worker_cmd(0, 1, 2, 2, python=sys.executable,
                             out=str(d / "out"), dl_root=str(d / "dl")) + [
            "--tar-dir", str(d / "src"), "--vae-stub", "--device", "cpu"]
        buf = io.StringIO()
        rc, tail, n = mcl.stream_worker(
            cmd, lambda: mcl.commit_volume(vol, logs.append), cwd=str(REPO),
            env=mcl.worker_env(os.environ), out=buf)
        shard_lines = [ln for ln in buf.getvalue().splitlines()
                       if mcl.is_shard_line(ln)]
        assert rc == 0 and n == 2 and len(shard_lines) == 2, buf.getvalue()
        assert vol.commits == 2 and len(logs) == 1, (vol.commits, logs)
        assert "commit conflict" in logs[0], logs
        assert any("done: 6 images" in ln for ln in tail), tail
        ok, lines = mcl.summarize_workers([
            {"worker": 1, "rc": 0, "seconds": 60},
            {"worker": 0, "rc": rc, "shards_written": n, "seconds": 1}])
        assert ok and lines[0].startswith("w0: ok"), lines
        ok, lines = mcl.summarize_workers([
            {"worker": 0, "rc": 0}, {"worker": 1, "rc": None,
                                     "error": "Traceback\nOSError: x"}])
        assert not ok and "FAILED" in lines[1] and "OSError: x" in lines[1]

        # manifest pass, then the remote content check
        subprocess.run([sys.executable, "scripts/cache_latents.py",
                        "--tar-dir", str(d / "src"), "--out",
                        str(d / "out"), "--write-manifest"], cwd=str(REPO),
                       check=True, capture_output=True)
        out = d / "out"
        expected = mcl.expected_upload(out)
        assert sorted(expected) == [
            "latents/manifest.json", "latents/train-0000.safetensors",
            "latents/train-0001.safetensors", "metadata.parquet"], expected
        manifest = json.loads((out / "latents/manifest.json").read_text())
        for p, e in expected.items():
            data = (out / p).read_bytes()
            assert e["size"] == len(data), (p, e)
            assert e["sha256"] == hashlib.sha256(data).hexdigest(), (p, e)
            if p.endswith(".safetensors"):   # from the manifest, not re-read
                assert e["sha256"] == manifest["digests"][Path(p).name]
            else:                            # git blob id = `git hash-object`
                git = subprocess.run(["git", "hash-object", str(out / p)],
                                     capture_output=True, text=True)
                if git.returncode == 0:
                    assert e["git_sha1"] == git.stdout.strip(), (p, e)
                assert e["git_sha1"] == hashlib.sha1(
                    b"blob %d\0" % len(data) + data).hexdigest()

        # what list_repo_tree returns: LFS shards + parquet, a non-LFS
        # manifest (git blob id only) and a folder
        def tree(exp, lfs_manifest=False):
            items = [SimpleNamespace(path="latents", tree_id="t")]
            for p, e in exp.items():
                lfs = None
                if p != "latents/manifest.json" or lfs_manifest:
                    lfs = SimpleNamespace(sha256=e["sha256"], size=e["size"])
                items.append(SimpleNamespace(
                    path=p, size=e["size"], lfs=lfs,
                    blob_id=e.get("git_sha1") or "f" * 40))
            return items
        remote = mcl.remote_entries(tree(expected))
        assert "latents" not in remote and len(remote) == 4, remote
        assert remote["latents/manifest.json"]["sha256"] is None, remote
        assert mcl.verify_upload(expected, remote) == []
        assert mcl.verify_upload(
            expected, mcl.remote_entries(tree(expected, True))) == []

        bad = {p: dict(e) for p, e in remote.items()}
        bad.pop("latents/train-0001.safetensors")
        bad["metadata.parquet"]["size"] += 1
        bad["latents/train-0000.safetensors"]["sha256"] = "0" * 64
        bad["latents/manifest.json"]["blob_id"] = "1" * 40
        exp_unv = dict(expected, **{"latents/x.safetensors": {
            "size": 3, "sha256": "a" * 64}})
        bad["latents/x.safetensors"] = {"size": 3, "sha256": None,
                                        "blob_id": "2" * 40}
        probs = mcl.verify_upload(exp_unv, bad)
        e = expected
        assert probs == [
            f"git blob latents/manifest.json: remote {'1' * 40} != local "
            f"{e['latents/manifest.json']['git_sha1']}",
            f"sha256 latents/train-0000.safetensors: remote {'0' * 64} != "
            f"expected {e['latents/train-0000.safetensors']['sha256']}",
            "missing latents/train-0001.safetensors",
            "unverifiable latents/x.safetensors: no LFS sha256 on the remote",
            f"size metadata.parquet: remote {e['metadata.parquet']['size'] + 1}"
            f" != local {e['metadata.parquet']['size']}"], probs
        assert "/vol/krea2/.cache/huggingface/upload" in mcl.CONTENT_HELP

        # finalize_impl with a fake HfApi: write check, privacy, upload,
        # content check at the commit read after the upload
        class FakeApi:
            def __init__(self, who_=None, exists=False, private=True,
                         auth_ok=True, corrupt=None):
                self.who = who_ or who("write")
                self.exists, self.private = exists, private
                self.auth_ok, self.corrupt = auth_ok, corrupt
                self.calls, self.uploaded = [], {}

            def whoami(self):
                self.calls.append("whoami")
                return self.who

            def repo_exists(self, repo, repo_type):
                self.calls.append("repo_exists")
                return self.exists

            def auth_check(self, repo, repo_type, write):
                self.calls.append(f"auth_check write={write}")
                if not self.auth_ok:
                    raise PermissionError("403 Forbidden")

            def create_repo(self, repo, repo_type, private, exist_ok):
                assert (repo, repo_type, private, exist_ok) == (
                    mcl.OUT_REPO, "dataset", True, True)
                self.calls.append("create_repo")
                if not self.exists:
                    self.exists, self.private = True, True

            def repo_info(self, repo, repo_type):
                self.calls.append("repo_info")
                return SimpleNamespace(private=self.private, sha="c0ffee")

            def upload_large_folder(self, repo, folder, **kw):
                assert kw["ignore_patterns"] == mcl.UPLOAD_IGNORE, kw
                assert kw["private"] is True, kw
                self.calls.append("upload")
                self.uploaded = mcl.expected_upload(folder)

            def list_repo_tree(self, repo, repo_type, recursive, expand,
                               revision):
                assert (recursive, expand, revision) == (True, True,
                                                         "c0ffee")
                self.calls.append("list_repo_tree")
                items = tree(self.uploaded)
                if self.corrupt:
                    for it in items:
                        if it.path == self.corrupt:
                            it.lfs.sha256 = "e" * 64
                return items

        man_cmds = []

        def runner(cmd, cwd):
            assert cmd == mcl.finalize_cmd(python="py", out=str(out)), cmd
            man_cmds.append(cmd)
            return SimpleNamespace(returncode=0)   # manifest already written

        def fin(api):
            with contextlib.redirect_stdout(io.StringIO()):
                return mcl.finalize_impl(FakeVol(), api=api, out_dir=out,
                                         runner=runner, python="py")

        api = FakeApi()
        r = fin(api)
        assert r["ok"] and r["problems"] == [] and r["error"] is None, r
        assert r["revision"] == "c0ffee" and r["n_files"] == 4, r
        assert api.calls == ["whoami", "repo_exists", "create_repo",
                             "repo_info", "upload", "repo_info",
                             "list_repo_tree"], api.calls
        api = FakeApi(exists=True)                 # existing private repo
        assert fin(api)["ok"] and api.calls[:3] == [
            "whoami", "repo_exists", "auth_check write=True"], api.calls
        api = FakeApi(exists=True, private=False)  # public: never upload
        r = fin(api)
        assert not r["ok"] and "not private" in r["error"], r
        assert "upload" not in api.calls, api.calls
        api = FakeApi(exists=True, auth_ok=False)
        r = fin(api)
        assert not r["ok"] and "cannot write" in r["error"], r
        assert "403 Forbidden" in r["error"], r
        assert "create_repo" not in api.calls, api.calls
        api = FakeApi(who_=who("read"))
        r = fin(api)
        assert "token role read" in r["error"] and api.calls == ["whoami"]
        api = FakeApi(who_=who("fineGrained", scoped=[
            {"entity": {"type": "dataset", "name": mcl.OUT_REPO},
             "permissions": ["repo.write"]}]))     # unknown: not blocked
        assert fin(api)["ok"], api.calls
        api = FakeApi(corrupt="latents/train-0001.safetensors")
        r = fin(api)
        assert not r["ok"] and r["problems"][0].startswith(
            "sha256 latents/train-0001.safetensors: remote eeee"), r
        assert "--finalize-only" in r["error"], r
        assert len(man_cmds) == 4, man_cmds        # read/auth/public stop early
    print("    modal_cache_latents: cost math, 5% tie rule, OOM ladder, "
          "token check, worker command/env/timeout, commit per shard, upload "
          "filter + content check")


# === add new @case functions above this line ===============================


def main(argv=None):
    names = sys.argv[1:] if argv is None else argv
    known = {fn.__name__: fn for fn in CASES}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; known: {sorted(known)}")
    selected = [known[n] for n in names] if names else CASES
    failed = []
    for fn in selected:
        print(f"[{fn.__name__}]")
        t0 = time.time()
        try:
            fn()
        except Exception:
            traceback.print_exc()
            failed.append(fn.__name__)
            print(f"[{fn.__name__}] FAILED ({time.time() - t0:.0f}s)")
        else:
            print(f"[{fn.__name__}] ok ({time.time() - t0:.0f}s)")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"ALL SCRIPT SMOKE TESTS PASSED ({len(selected)} cases)")


if __name__ == "__main__":
    main()
