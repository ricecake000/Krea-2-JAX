# Krea-2-LoRA-JAX

LoRA fine-tuning of [Krea 2](https://www.krea.ai/krea-2-open-source) on **TPU**
with **JAX**.

Krea 2 is a 12.9B-parameter single-stream Diffusion Transformer (flow
matching, Qwen3-VL-4B text conditioning, Qwen-Image f8/16ch VAE) released with
open weights as **Krea 2 RAW** (base, for training) and **Krea 2 Turbo**
(8-step distilled, for inference). This repo re-implements the DiT in pure
functional JAX — numerically verified against the
[official PyTorch implementation](https://github.com/krea-ai/krea-2) down to
float32 rounding (measured ~4e-7 max relative error; tests assert < 2e-4 to
absorb platform/BLAS variance) — and trains LoRA adapters on TPU:

- **FSDP sharding** of the frozen bf16 base across all TPU devices
  (`lax.scan` over blocks + gradient rematerialization, query-chunked
  exact attention), so a v4-8 / v5e-8 / v6e-8 single host is enough.
- **Flow-matching objective** matching the official formulation:
  `x_t = (1-t)·x0 + t·ε`, target `v = ε − x0`, logit-normal timesteps with
  the same resolution-dependent shift (`mu`) the official sampler uses.
- The frozen **text encoder and VAE run once, offline** (PyTorch, CPU or
  GPU): latents and Qwen3-VL hidden states are cached to disk, and the TPU
  loop touches only the DiT.
- LoRA export as safetensors (PEFT-style `lora_A`/`lora_B` on official
  state-dict paths) plus a **merge script** producing a plain full-weight
  checkpoint usable by the official inference code / ComfyUI.

Train on **RAW**, deploy on **Turbo** — LoRAs trained on RAW are designed to
express strongly on Turbo (official recommendation).

## Setup (single-host TPU VM)

```bash
git clone https://github.com/ricecake000/Krea-2-LoRA-JAX
cd Krea-2-LoRA-JAX

pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements-tpu.txt

# host-side preprocessing deps (CPU torch is fine on a TPU VM)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-torch.txt

# base checkpoint (~26 GB, bf16)
hf download krea/Krea-2-Raw raw.safetensors --local-dir ./ckpt
```

## 1. Prepare your dataset

A folder of images with same-basename caption files:

```
data/
  dog_001.jpg   dog_001.txt   # "a photo of sks dog sitting on grass"
  dog_002.png   dog_002.txt
  ...
```

Encode everything once (VAE latents + Qwen3-VL hidden states → safetensors
cache). Runs on CPU; pass `--device cuda` if the machine has a GPU:

```bash
python scripts/cache_dataset.py \
  --data-dir ./data --cache-dir ./cache --resolution 1024
```

Non-square datasets: pass an explicit `--width`/`--height` (multiples of 16)
matching your dataset's aspect ratio instead of `--resolution`, e.g.
`--width 832 --height 1216` for ~2:3 portraits. All images are cover-resized
and center-cropped to that one fixed shape, so a run still compiles a single
XLA graph — if your images share (roughly) one aspect ratio there is no need
for bucketing.

## 2. Train the LoRA (TPU)

```bash
python scripts/train_lora.py \
  --base-ckpt ./ckpt/raw.safetensors \
  --cache-dir ./cache \
  --output-dir ./lora_out \
  --steps 2000 --batch-size 8 --lr 1e-4 \
  --rank 16 --targets attn
```

Useful knobs:

| flag | default | notes |
|---|---|---|
| `--rank` / `--alpha` | 16 / rank | LoRA capacity; scale = alpha/rank |
| `--targets` | `attn` | `attn` (wq/wk/wv/wo), `attn+mlp`, `all` (+attention gate) |
| `--batch-size` | 8 | must be divisible by the TPU device count |
| `--lr` | 1e-4 | linear warmup (`--warmup-steps 100`), `--cosine-decay` optional |
| `--txt-len` | auto | pad text to the longest caption; set 512 to mirror the official pipeline exactly (mathematically equivalent, just slower) |
| `--timestep-mu` | from resolution | logit-normal timestep shift override |
| `--qchunk` | 512 | attention query-chunk size (memory knob) |

Checkpointing: every `--save-every` steps the script writes
`lora_stepNNNNNN.safetensors` plus a companion
`train_state_stepNNNNNN.safetensors` (AdamW moments + step counter).
A bare `--resume` picks the newest checkpoint in `--output-dir` (starting
fresh if there is none, so it is safe to always pass it); `--resume <dir>`
searches that directory instead, and `--resume <file>` selects a checkpoint
explicitly. Either way the companion train_state file is picked up
automatically and training continues from that step (same LR schedule
position) up to `--steps` — useful on preemptible or time-limited machines
such as Kaggle's 9-hour TPU sessions. Resuming requires the same `--rank`/`--targets`; if the
train-state file is missing, only the LoRA weights are restored and the
optimizer starts fresh (the old behavior).

Memory: base 12.9B bf16 ≈ 26 GB FSDP-sharded → ~3.3 GB/device on 8 devices,
plus LoRA/optimizer (replicated, tiny) and rematerialized activations.
Batch 8 @ 1024px trains comfortably on v4-8 / v5e-8.

## 3. Sample with the LoRA (TPU)

```bash
python scripts/sample.py \
  --base-ckpt ./ckpt/raw.safetensors \
  --lora ./lora_out/lora.safetensors \
  --prompt "a photo of sks dog on the beach" \
  --steps 28 --cfg 4.5 --seed 0
```

Prompt encoding and VAE decoding run on host CPU via PyTorch; the DiT
sampling loop runs on TPU. For the Turbo checkpoint use
`--steps 8 --cfg 0 --mu 1.15`.

## 4. Deploy: merge into Turbo

```bash
hf download krea/Krea-2-Turbo turbo.safetensors --local-dir ./ckpt
python scripts/merge_lora.py \
  --base ./ckpt/turbo.safetensors \
  --lora ./lora_out/lora.safetensors \
  --output ./ckpt/turbo_mylora.safetensors
```

The merged file is a plain Krea 2 checkpoint — drop it into the official
inference repo or ComfyUI.

## Tests

CPU-only numerical verification against the official PyTorch code
(downloads `mmdit.py` from the official repo on first run):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install einops optax
python tests/test_parity.py        # forward parity (~4e-7, asserts <2e-4), LoRA == merged
python tests/test_train_and_io.py  # train step, save/load, merge script
python tests/test_sharding.py      # 8-device FSDP parity (emulated on CPU)
```

## Repo layout

```
krea2_jax/
  config.py      # architecture + conditioning constants (single_mmdit_large_wide)
  model.py       # pure-JAX SingleStreamDiT (scan + remat, GQA, gated attention)
  checkpoint.py  # safetensors -> sharded pytree, LoRA init/save/load, mesh/FSDP
  sampling.py    # Euler flow sampler + resolution-shifted timestep schedule
scripts/
  cache_dataset.py  # torch: VAE latents + Qwen3-VL states -> cache
  train_lora.py     # jax: TPU LoRA training
  sample.py         # jax DiT + torch encode/decode
  merge_lora.py     # torch: bake LoRA into a full checkpoint
tests/              # parity / training / sharding verification
```

## Notes & limitations

- Single-host TPU slices (≤8 devices visible to one process). Multi-host
  pods would need per-host data sharding and `jax.make_array_from_*` when
  loading — not implemented yet.
- One fixed training resolution per run — square or rectangular via
  `--width`/`--height` (no mixed-aspect-ratio bucketing yet).
- Krea 2 weights are distributed under the **Krea 2 Community License**;
  review it before commercial use.
