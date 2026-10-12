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
/home/mj/krea2-ops/mi350x/results/modal/text_parity_<ts>.log
"""

import subprocess
import sys
import time
from pathlib import Path

import modal

LOCAL_REPO = Path(__file__).resolve().parent.parent
LOCAL_PARQUET = Path(
    "/tmp/claude-1000/-home-mj-Krea-2-LoRA-JAX/"
    "c32a3b35-21c9-49e1-a93d-8c87239100a6/scratchpad/ds/metadata.parquet")
LOCAL_RESULTS = Path("/home/mj/krea2-ops/mi350x/results/modal")
REMOTE_REPO = "/repo"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", extra_index_url="https://download.pytorch.org/whl/cpu")
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
def parity(n: int, dtypes: list, batch: int) -> str:
    import huggingface_hub

    out = []

    def run(cmd):
        out.append("$ " + " ".join(cmd))
        p = subprocess.run(cmd, cwd=REMOTE_REPO, capture_output=True, text=True)
        out.append(p.stdout)
        if p.stderr:
            # keep only the informative tail of stderr (progress bars are noise)
            out.append("[stderr tail]\n" + p.stderr[-3000:])
        out.append(f"[exit {p.returncode}]\n")
        return p.returncode

    t0 = time.time()
    model_dir = huggingface_hub.snapshot_download(
        "Qwen/Qwen3-VL-4B-Instruct",
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
    out.append(f"weights downloaded in {time.time() - t0:.0f}s -> {model_dir}\n")

    run([sys.executable, "scripts/text_encoder_parity.py", "--tokenizer-only",
         "--text-encoder", model_dir, "--parquet", "metadata.parquet"])
    for dt in dtypes:
        run([sys.executable, "scripts/text_encoder_parity.py",
             "--text-encoder", model_dir, "--parquet", "metadata.parquet",
             "--n", str(n), "--batch", str(batch), "--dtype", dt,
             "--torch-dtype", dt, "--torch-device", "cpu",
             "--position-mode", "contiguous"])
    return "\n".join(out)


@app.local_entrypoint()
def main(n: int = 16, dtypes: str = "f32,bf16", batch: int = 4):
    text = parity.remote(n, dtypes.split(","), batch)
    print(text)
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    path = LOCAL_RESULTS / f"text_parity_{time.strftime('%Y%m%d-%H%M%S')}.log"
    path.write_text(text)
    print(f"saved {path}")
