"""Modal app: real-weight parity of the JAX Qwen3-VL-4B text tower against the
official torch conditioner (scripts/text_encoder_parity.py) on a CPU container
with enough RAM (the laptop cannot hold the 4B weights twice).

    modal run scripts/modal_text_parity.py                 # f32 then bf16, 16 captions
    modal run scripts/modal_text_parity.py --n 8 --dtypes bf16

The container pins transformers==4.57.1 (the official krea-2 inference repo's
pin, whose text-only RoPE positions are the "contiguous" mode) and torch CPU.
The repo's krea2_jax/ and scripts/ plus the dataset parquet are attached as
local files; Qwen/Qwen3-VL-4B-Instruct (public) is downloaded in the
container. Output: the parity script's stdout, also written to
/home/mj/krea2-ops/mi350x/results/modal/text_parity_<ts>.log; the run exits
1 with a one-line reason when any check failed (a failed tokenizer-only
check skips the model checks). Importable without the Modal SDK (the app is
only defined when `modal` imports); run_checks / parity_impl / finish are
plain functions (tests/test_scripts_smoke.py::modal_text_parity_flow).
"""

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

try:
    import modal
except ImportError:  # importable (pyflakes, tests) without the Modal SDK
    modal = None

LOCAL_REPO = Path(__file__).resolve().parent.parent
# the dataset's metadata.parquet (download it from the HF dataset repo
# ricecake/danbooru-curated-v2 or set KREA2_PARQUET)
LOCAL_PARQUET = Path(os.environ.get(
    "KREA2_PARQUET", "/home/mj/krea2-ops/mi350x/data/metadata.parquet"))
LOCAL_RESULTS = Path("/home/mj/krea2-ops/mi350x/results/modal")
REMOTE_REPO = "/repo"
RESULT_PREFIX = "parity result: "


def run_checks(model_dir, n, dtypes, batch, runner=subprocess.run,
               cwd=REMOTE_REPO, python=sys.executable):
    """Run the tokenizer-only check, then one model parity run per dtype,
    through `runner` (subprocess.run signature). If the tokenizer check
    fails, the model checks are skipped. Returns (log lines, failures) where
    failures lists one short description per failed or skipped check."""
    out, failures = [], []

    def run(cmd):
        out.append("$ " + " ".join(cmd))
        p = runner(cmd, cwd=cwd, capture_output=True, text=True)
        out.append(p.stdout or "")
        if p.stderr:
            # keep only the informative tail of stderr (progress bars are noise)
            out.append("[stderr tail]\n" + p.stderr[-3000:])
        out.append(f"[exit {p.returncode}]\n")
        return p.returncode

    rc = run([python, "scripts/text_encoder_parity.py", "--tokenizer-only",
              "--text-encoder", model_dir, "--parquet", "metadata.parquet"])
    if rc != 0:
        failures.append(f"tokenizer-only check exit {rc}")
        if dtypes:
            failures.append(f"model checks skipped ({','.join(dtypes)})")
            out.append("tokenizer-only check failed: skipping the model "
                       "parity runs\n")
        return out, failures
    for dt in dtypes:
        rc = run([python, "scripts/text_encoder_parity.py",
                  "--text-encoder", model_dir, "--parquet", "metadata.parquet",
                  "--n", str(n), "--batch", str(batch), "--dtype", dt,
                  "--torch-dtype", dt, "--torch-device", "cpu",
                  "--position-mode", "contiguous"])
        if rc != 0:
            failures.append(f"{dt} parity exit {rc}")
    return out, failures


def parity_impl(n, dtypes, batch, download=None, runner=subprocess.run):
    """Download the weights and run every check. Returns (text, ok); the
    text ends with a 'parity result: PASS' / 'parity result: FAIL (...)'
    line. A download failure is reported the same way (no exception)."""
    out = []
    t0 = time.time()
    try:
        if download is None:
            import huggingface_hub

            def download():
                return huggingface_hub.snapshot_download(
                    "Qwen/Qwen3-VL-4B-Instruct",
                    allow_patterns=["*.json", "*.safetensors", "merges.txt",
                                    "vocab.json"])
        model_dir = download()
    except Exception:
        out.append(traceback.format_exc())
        failures = ["weight download failed"]
    else:
        out.append(f"weights downloaded in {time.time() - t0:.0f}s -> "
                   f"{model_dir}\n")
        lines, failures = run_checks(model_dir, n, dtypes, batch,
                                     runner=runner)
        out += lines
    ok = not failures
    out.append(RESULT_PREFIX + ("PASS" if ok
                                else "FAIL (" + "; ".join(failures) + ")"))
    return "\n".join(out), ok


def finish(text, ok, results_dir=LOCAL_RESULTS):
    """Print and save the log, then raise SystemExit(1) with the one-line
    result when any required check failed. Returns the log path."""
    print(text)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"text_parity_{time.strftime('%Y%m%d-%H%M%S')}.log"
    path.write_text(text)
    print(f"saved {path}")
    if not ok:
        reason = next((ln for ln in reversed(text.splitlines())
                       if ln.startswith(RESULT_PREFIX)),
                      RESULT_PREFIX + "FAIL")
        raise SystemExit(f"{reason} (log: {path})")
    return path


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.10.0",
                     extra_index_url="https://download.pytorch.org/whl/cpu")
        .pip_install(
            "transformers==4.57.1", "jax[cpu]==0.6.2", "safetensors>=0.4.5",
            "ml_dtypes>=0.4.0", "tokenizers", "pyarrow", "numpy", "pillow",
            "huggingface_hub", "accelerate",
        )
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax")
        .add_local_dir(LOCAL_REPO / "scripts", f"{REMOTE_REPO}/scripts")
        .add_local_file(LOCAL_PARQUET, f"{REMOTE_REPO}/metadata.parquet")
    )

    app = modal.App("krea2-text-parity", image=image)

    @app.function(cpu=16, memory=65536, timeout=60 * 60)
    def parity(n: int, dtypes: list, batch: int):
        return parity_impl(n, dtypes, batch)

    @app.local_entrypoint()
    def main(n: int = 16, dtypes: str = "f32,bf16", batch: int = 4):
        text, ok = parity.remote(n, dtypes.split(","), batch)
        finish(text, ok)
