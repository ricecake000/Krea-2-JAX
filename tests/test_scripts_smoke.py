"""CPU smoke tests for the node-side scripts, run through subprocess with a
tiny model config on 8 emulated devices:

  aot_analysis_verdict   scripts/aot_analysis.py: the activation-collective
                         verdict is NONE with --act-sharding on and FOUND
                         with it off (the partitioner moves activations)
  bench_step_modes       scripts/bench_step.py --random-weights in fwd,
                         fwdbwd and train modes (2 timed steps each), plus
                         train with --attn cudnn

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
    "--model-config", TINY_JSON, "--batch-size", 16, "--resolution", 256,
    "--txt-len", 64, "--qchunk", 128, "--emulate-devices", 8,
    "--remat-policy", "dots", "--act-numel", 2e5,
)
VERDICT_RE = re.compile(r"^activation collectives: (NONE|FOUND (\d+))", re.M)


@case
def aot_analysis_verdict():
    verdicts = {}
    for mode in ("on", "off"):
        _, out = run_script("aot_analysis.py", *AOT_ARGS, "--act-sharding", mode)
        for needle in ("params: ", "sequence: ", "lowered in", "compiled (cpu",
                       "memory_analysis", "cost_analysis", "collectives: ",
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
              "--batch-size", 8, "--width", 128, "--height", 128,
              "--txt-len", 16, "--qchunk", 128, "--steps", 2, "--warmup", 1)
    for mode in ("fwd", "fwdbwd", "train"):
        _, out = run_script("bench_step.py", *common, "--mode", mode)
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
