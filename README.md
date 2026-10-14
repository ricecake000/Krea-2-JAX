# Krea-2-JAX

[Krea 2](https://www.krea.ai/krea-2-open-source) in pure functional **JAX**:
**LoRA fine-tuning on TPU** and **full fine-tuning on multi-GPU hosts
(8x AMD MI355X / ROCm)**, with sampling and export back to the official
checkpoint format.

Krea 2 is a 12.9B-parameter single-stream Diffusion Transformer (flow
matching, Qwen3-VL-4B text conditioning, Qwen-Image f8/16ch VAE) released with
open weights as **Krea 2 RAW** (base, for training) and **Krea 2 Turbo**
(8-step distilled, for inference). This repo re-implements the DiT in pure
functional JAX — numerically verified against the
[official PyTorch implementation](https://github.com/krea-ai/krea-2) down to
float32 rounding (measured ~4e-7 max relative error; tests assert < 2e-4 to
absorb platform/BLAS variance). Two training paths share it:

- `scripts/train_lora.py`: LoRA adapters on a single-host TPU (v4-8 / v5e-8 /
  v6e-8), frozen bf16 base FSDP-sharded across the devices.
- `scripts/train_full.py`: every DiT weight, fp32 master + sharded AdamW +
  EMA, on one host with 8 accelerators (developed for 8x MI355X; see
  [Full fine-tuning](#full-fine-tuning-8x-mi355x--rocm)).

Common to both:

- **FSDP sharding** across all devices (`lax.scan` over blocks + gradient
  rematerialization, batch-sharded activations), with exact query-chunked
  attention everywhere plus fused kernels on TPU (Splash) and MI355X (jax-aiter).
- **Flow-matching objective** matching the official formulation:
  `x_t = (1-t)·x0 + t·ε`, target `v = ε − x0`, logit-normal timesteps with
  the same resolution-dependent shift (`mu`) the official sampler uses.
- The frozen **text encoder and VAE run once, offline** (PyTorch, CPU or
  GPU): latents and Qwen3-VL hidden states are cached to disk, and the training
  loop touches only the DiT.
- LoRA export as safetensors (PEFT-style `lora_A`/`lora_B` on official
  state-dict paths) plus a **merge script** producing a plain full-weight
  checkpoint usable by the official inference code / ComfyUI.

Train on **RAW**, deploy on **Turbo** — LoRAs trained on RAW are designed to
express strongly on Turbo (official recommendation).

## Setup (single-host TPU VM)

```bash
git clone https://github.com/ricecake000/Krea-2-JAX
cd Krea-2-JAX

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

Sampling-side optimizations (outputs unchanged, see `tests/test_sampling.py`):

- `--attn splash` (default on TPU): fused Pallas attention, so the
  float32 `(heads, chunk, L)` logits never round-trip through HBM. On more
  than one device the heads are split across chips (no duplicated attention).
  `--attn ref` restores the query-chunked reference path.
- Padded text tokens are dropped before sampling (text sits at RoPE position
  0 and padding is masked, so the image output is identical) — a short
  prompt runs ~4.35k instead of 4.6k tokens at 1024px. `--no-compact-text`
  keeps the padded length.
- The text branch (text fusion + txtmlp) does not depend on the timestep and
  runs once per prompt instead of every step.

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

## Full fine-tuning (8x MI355X / ROCm)

`scripts/train_full.py` trains **every** DiT weight instead of a LoRA, on one
host with 8 accelerators (developed for 8x AMD MI355X, 288 GB HBM each; any
single-host JAX backend works). Same flow-matching objective and
logit-normal timestep sampling as the LoRA trainer; differences:

- fp32 **master weights**, AdamW moments and an optional **EMA** copy, all
  FSDP-sharded across the 8 devices exactly like the base params (only
  scalars are replicated — the script asserts this at startup). The forward
  and backward run in bf16 from a per-step cast of the master weights.
- **Aspect-bucketed data**: the curated dataset (`ricecake/danbooru-curated-v2`,
  96,848 images pre-cropped into 19 buckets of 3,840-4,096 image tokens)
  is cached as **VAE latents only** (`scripts/cache_latents.py`); every
  sample occupies a static `--img-tokens` (4096) rows — its own grid first,
  masked padding after — and its timestep shift `mu` comes from its own
  token count. The loss is a per-sample mean over the valid image tokens,
  averaged over the batch (padded rows contribute nothing).
- **Online text encoder**: captions are drawn per epoch (long / medium /
  short with `--caption-probs`, `", drawn by <artist>"` with
  `--artist-prob`), tokenized by the loader and encoded **inside the train
  step** by a JAX port of the Qwen3-VL-4B text tower
  (`krea2_jax/qwen3_text.py`; bf16, replicated, frozen, no gradient) —
  caching 12 x 2560 hidden states per token for three caption variants would
  be multi-TB.
- The latents are **streamed** from disk (`krea2_jax/data.py`
  `BucketBatchLoader`: a background thread pool with a bounded prefetch
  queue), so ~100k samples never have to fit in host RAM.
- Train state is checkpointed with **orbax** (async) under
  `<output-dir>/ckpt/<step>/`; `--resume` continues the newest step and
  reproduces the uninterrupted run (same data order, RNG and LR schedule).
- Exports are plain **official-layout bf16 safetensors** (the same keys as
  `raw.safetensors`) — no LoRA files anywhere in this path.

### Install (ROCm 7, gfx950)

Follow AMD's JAX-on-ROCm instructions for your ROCm version; for ROCm 7:

```bash
pip install jax jax-rocm7-plugin     # pulls the matching jax-rocm7-pjrt
pip install -r requirements-rocm.txt
python -c "import jax; print(jax.devices())"   # expect 8 ROCm devices

# preprocessing on the same box: ROCm build of PyTorch (see pytorch.org)
pip install -r requirements-torch.txt
```

(`requirements-rocm.txt` leaves `jax` unpinned: the `--attn splash` path is
TPU-only and keeps the TPU pin in `requirements-tpu.txt`. It pins `optax` and
`orbax-checkpoint`, since a train state saved on one lease is restored on the
next install. On MI355X train with `--attn aiter` and the pinned jax-aiter
stack below — ~2.5x the throughput of `--attn ref` (stage 2: 2.2 vs 0.89
samples/s/GPU); `ref` is only the portable default.)

Required environment on the node:

```bash
# the base checkpoint loader and export_full use the CPU backend; with `rocm`
# alone load_krea2_params fails at jax.devices("cpu")
export JAX_PLATFORMS=rocm,cpu
# HIP command buffers capture the aiter FFI kernels and the first train step
# dies with HIP_ERROR_StreamCaptureInvalidated; train_full.py refuses to
# start `--attn aiter` on a GPU backend without this
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
```

### Activation sharding (`--act-sharding`, default `on`)

With FSDP the weights are sharded but every activation is batch-sharded, and
XLA's SPMD partitioner is free to pick any layout in between. Left alone it
gets this wrong for the DiT: an AOT compile of the real train step (batch 64,
1024px, 8 devices; `scripts/aot_analysis.py --act-sharding off`) moves **full
activations** — `[64, 4608, 6144]` all-gathers and all-reduces, about a dozen
per scanned block — instead of all-gathering each layer's weights, i.e.
terabytes of inter-GPU traffic per step. `--act-sharding on` passes
`act_sharding=NamedSharding(mesh, P("fsdp"))` into `dit_forward`, which
applies `with_sharding_constraint` to every `linear` output and to the scan
carry at block entry/exit; with it the block body contains only per-layer
weight all-gathers and gradient reductions (verdict line
`activation collectives: NONE`). It changes no math (outputs and gradients
match `off` to fp32 rounding; `tests/test_aiter_attention.py`), so keep it
on; `off` exists for A/B comparisons. Both `--act-sharding` and `--attn` are
recorded in the checkpoint's run config and only warn when changed on
`--resume`.

### Unrolled layer loop (`--layer-loop`, default `scan`)

`scan` runs the 28 blocks with `lax.scan` over weights stacked on a layer
axis. On GPU every scan iteration costs ~42 device-to-device copies:
dynamic-slices of the stacked weight shards feeding the FSDP all-gathers,
dynamic-update-slices of the remat-saved GEMM outputs into stacked residual
buffers and of the reduce-scattered weight grads into the stacked grads
(~60 ms/step on MI350X, same count on B300). `--layer-loop unroll` (in
`train_full.py`, `bench_step.py` and `aot_analysis.py`) loads the blocks as
a list of 28 per-layer arrays (`load_krea2_params(unstack=("blocks",))`)
and runs them in an unrolled Python loop, so the graph has no slice, DUS or
while loop at all (numerics identical; `tests/test_layer_loop.py` checks
the compiled HLO). The cost is compile time (28 copies of the block body).
Exports are the same official keys and bytes either way, but orbax
checkpoints are not interchangeable between modes (the train-state tree
differs): `layer_loop` is a hard resume key. LoRA training stays on `scan`.

### jax-aiter flash attention (`--attn aiter`, MI355X only)

[jax-aiter](https://github.com/ROCm/jax-aiter) exposes AMD's AITER kernels
to JAX via XLA FFI; `--attn aiter` uses its fused `flash_attn_varlen`
(forward + backward through `custom_vjp`, GQA native, bf16, head dim 128)
instead of the reference attention, whose `[heads, qchunk, L]` fp32 logits
dominate activation memory and bandwidth. It is **gfx950-only** and
supported on exactly one stack (jax-aiter 0.1.0a2):

| component | version |
|---|---|
| image | `ghcr.io/rocm/jax-base-ubu24.therock-7.14:7.14` (TheRock 7.14 GA, no `/opt/rocm`) |
| Python | 3.12 |
| `jax` / `jaxlib` | 0.11.0 |
| `jax-rocm7-plugin` / `jax-rocm7-pjrt` | 0.11.0.post1 |
| jax-aiter | `jax_aiter-0.1.0a2+full-...whl` (`nightly-wheel` CI artifact; no GitHub release) |

```bash
pip install "jax==0.11.0" "jaxlib==0.11.0" \
  "jax-rocm7-plugin==0.11.0.post1" "jax-rocm7-pjrt==0.11.0.post1"
pip install ./jax_aiter-0.1.0a2+full-cp312-cp312-manylinux_2_39_x86_64.whl
python -c "from jax_aiter.mha import flash_attn_varlen; print('MHA ready')"
```

There is no GitHub release for this wheel. The `+full` wheel (MHA libraries
bundled, ~511 MB) is the jax-aiter `nightly-wheel` GitHub Actions artifact
(downloading it needs `gh auth`); our runs use a mirror on the public HF
dataset `ricecake/krea2-ops-assets`. The lite `0.1.0a2` wheel omits the MHA
libraries; after installing it run

```bash
jax-aiter-fetch-mha --tag jit-libs-85d614f12c412443
python -c "from jax_aiter.mha import flash_attn_varlen; print('MHA ready')"
```

The `--tag` is required: the wheel's own `RELEASE_TAG` points at a manifest
whose `aiter_sha` does not match the wheel, and the symptom is a
`named symbol not found` at the first aiter call, after the base load and
compile. jax-aiter is imported lazily, only when `--attn aiter` is used, so
nothing else needs it.

How it maps onto our masking: the DiT masks keys only (padded text tokens),
and the padding sits in the *middle* of each sample's sequence
(`[valid text, text padding, image, pad-to-256]`). `dit_forward` therefore
reorders each sample's tokens with a stable argsort so the valid tokens form
a prefix (positions and the mask travel with the tokens; text RoPE positions
are all zero, so this is exact), runs the blocks, and restores the original
order before the output layer. The attention call packs q/k/v as
`(B*L, H, D)` and uses the kernel's *padded group mode*: physical
`cu_seqlens = [0, L, 2L, ...]` and logical `cu_seqlens_logical` = cumulative
valid lengths, so each segment attends over its valid prefix and the padding
rows (of the output and of dq/dk/dv) come back zero. Under a sharded batch it
runs inside `shard_map` over `P("fsdp")` (per-device segment offsets), like
the Splash path. Check it on the node with `scripts/node_attention_parity.py`
and `scripts/parity_full.py --attn aiter` before training with it.

### Dataset: cache the latents on all 8 GPUs

`train_full.py --data-dir` reads the `krea2-latents-v1` layout written by
`scripts/cache_latents.py`:

```
<data-dir>/
  metadata.parquet              # verbatim copy of the dataset's parquet: id, long/medium/short
                                # captions, artist_name, bucket_w/bucket_h, tar, ...
  latents/manifest.json         # {"format": "krea2-latents-v1", "vae", "n", "shards", "digests"}
  latents/train-XXXX.safetensors  # one per source tar: "<id>" -> (16, bucket_h/8, bucket_w/8) bf16,
                                  # VAE-normalized like cache_dataset.py
```

Only the latents are cached; the captions stay text in `metadata.parquet`
and go through the text encoder online. The cache builder downloads each
tar of the HF dataset (private: set `HF_TOKEN`; `--repo` defaults to
`ricecake/danbooru-curated-v2`, `--tar-dir` reads local tars instead),
VAE-encodes it per bucket, writes the shard atomically and deletes the tar
(`--keep-tars` keeps it). One process per GPU over interleaved shards, then
a manifest pass that copies the parquet, checks every id has exactly one
latent of its bucket's shape and records the sha256 of every shard file in
`manifest.json` (`"digests"`; this reads all ~49 GB once). The trainer
requires the digests (an older manifest without them, or a null/malformed
digest, is refused with a pointer to `--write-manifest`), re-hashes every
shard against them at startup (`--verify-latents`, default on) and folds
them into the dataset fingerprint.
Rerunning the manifest pass over a manifest that already has digests
re-hashes and fails if any shard changed; `--no-verify-digests` accepts the
new content:

```bash
OUT=/scratch/krea2-data
for i in $(seq 0 7); do
  HIP_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 python scripts/cache_latents.py \
    --out $OUT --num-workers 8 --worker-index $i --batch 8 > cache_w$i.log 2>&1 &
done
wait
python scripts/cache_latents.py --out $OUT --write-manifest
```

(ROCm PyTorch exposes the GPUs as `cuda`.) Finished shards are skipped, so
an interrupted run is simply restarted. Disk: the latents are ~49 GB bf16
(~480 MB per shard of ~950 images) plus one in-flight tar per worker; there
is no text cache.

The text encoder (`--text-encoder`, a local HF dir or a repo id such as
`Qwen/Qwen3-VL-4B-Instruct`, ~8.8 GB; only config, tokenizer and
safetensors are fetched) is loaded by the trainer itself. To check the JAX
port against the production torch conditioner on real weights:

```bash
python scripts/text_encoder_parity.py --tokenizer-only \
  --text-encoder Qwen/Qwen3-VL-4B-Instruct --parquet $OUT/metadata.parquet
python scripts/text_encoder_parity.py --text-encoder Qwen/Qwen3-VL-4B-Instruct \
  --parquet $OUT/metadata.parquet --n 32 --dtype bf16 --torch-device cuda
```

### Bring-up on a new node

Run these before the first real training job; none needs the dataset cache
or the text-encoder weights (the text tower is abstract / random).

1. **`scripts/aot_analysis.py`** — compiles the exact train step for
   abstract shapes (no weights, no torch) and prints parameter count,
   sequence settings, compile time, XLA's per-device `memory_analysis()`,
   `cost_analysis()` (note: XLA counts the 28-block scan body once), every
   SPMD collective grouped by (kind, dtype, shape), and the verdict line
   `activation collectives: NONE` / `FOUND n` (a collective whose result has
   `shape[0] == batch` and >= 1e7 elements). Without accelerators it emulates
   8 CPU devices (`--emulate-devices N`), so it also runs on a laptop; the
   CPU backend prints three caveats (reduce-scatter shows as all-reduce,
   bf16 collectives show as f32, temp sizes are not representative).

   ```bash
   python scripts/aot_analysis.py --batch-size 64 --img-tokens 4096 \
     --text-rows 256 --remat-policy dots --attn ref --act-sharding on
   ```

   The step includes the online text tower (default: the real Qwen3-VL-4B
   config, abstract bf16 params replicated on every device — 3.92B params,
   7.3 GiB/device of the arguments); its parameter count, bytes and analytic
   forward flops are printed next to the DiT's. Other flags: `--bucket WxH`
   (pixel bucket of every sample; default the largest square grid fitting
   `--img-tokens`, i.e. 1024x1024), `--text-model-config JSON`
   (Qwen3TextConfig overrides), `--timestep-mu`, `--ema-decay`, `--qchunk`,
   `--model-config JSON`, `--no-compile` (lower only), `--dump-hlo PATH`.

2. **`scripts/bench_step.py`** — step-time benchmark on random inputs:
   `--mode fwd` (forward), `fwdbwd` (value_and_grad w.r.t. bf16 weights,
   grads cast to `--grad-dtype`) or `train` (the real `train_full` step:
   fp32 master + AdamW + EMA + the online text tower with random bf16
   weights, replicated). DiT weights come from `--base-ckpt` or
   `--random-weights` (generated on device). Inputs are a bucketed batch
   (`--img-tokens` rows, every sample on the `--bucket` grid, random caption
   ids / text states). Prints median step time, samples/s, analytic TFLOP/s
   per device (6 x params x padded tokens x batch; x4/3 for
   `--remat-policy full`, x1.1 for `dots`; 2x for `fwd`; `train` adds the
   text tower's forward) and peak HBM when the backend reports it.

   ```bash
   python scripts/bench_step.py --random-weights --mode train \
     --batch-size 64 --img-tokens 4096 --text-rows 256 --remat-policy dots \
     --attn aiter --act-sharding on --steps 10 --warmup 2
   ```

   Also `--bucket WxH`, `--text-model-config`, `--timestep-mu`,
   `--base-sharding fsdp|replicate`, `--qchunk`, `--model-config`, `--seed`.

3. **`scripts/node_attention_parity.py`** — kernel-level check of
   `aiter_attention` against the reference attention (padded query rows
   zeroed, the kernel contract), forward and gradients (dq, dk, dv), on random
   prefix masks, compared to an fp32 truth, plus median per-call ms of the
   jitted forward and forward+backward. Exits non-zero above `--tol` (2e-2
   bf16, 1e-4 f32). `--shard` shards the batch over all devices (exercises
   the `shard_map` wrapper); `--base-ckpt` (or `--random-weights` with a tiny
   `--model-config`) adds a DiT-level `attn_impl="ref"` vs `"aiter"` check at
   `--width/--height` with mid-sequence text padding (`--dit-grad`: also the
   gradients; `--dit-truth f32` instead measures ref-bf16 and aiter-bf16
   against a ref-f32 truth and passes iff aiter's error is at most
   `--dit-truth-ratio` (1.5) × ref-bf16's + 1e-3). `--kernel emul` swaps in the pure-JAX emulation of the padded
   group mode, so it runs on CPU.

   ```bash
   python scripts/node_attention_parity.py --kernel aiter --batch 2 --heads 48 \
     --kv-heads 12 --seq 4608 --head-dim 128 --dtype bf16 --iters 20 --shard
   python scripts/node_attention_parity.py --kernel aiter \
     --base-ckpt ./ckpt/raw.safetensors --width 1024 --height 1024 --dit-grad \
     --qchunk 256   # 4352 padded tokens at --txt-len 77; 512 would go dense
   ```

4. **`scripts/parity_full.py`** — full-size numerical parity of the official
   PyTorch model vs our JAX model on the same weights: random latents, text
   states with random valid lengths (padding in the middle of the combined
   sequence), random t; compares image-token velocities per sample (max abs,
   max rel, relative-norm error) and exits non-zero when a sample's
   relative-norm error exceeds `--tol` (default 5e-2).

   ```bash
   python scripts/parity_full.py --base-ckpt ./ckpt/raw.safetensors \
     --width 1024 --height 1024 --batch 2 --txt-len 77 \
     --attn aiter --act-sharding on --torch-device cuda --dtype bf16 --sequential
   ```

   `--ref-dir` points at the official repo (default `$KREA2_REF_DIR`, else
   `mmdit.py` is downloaded like the tests do). The official attention forces
   cuDNN SDPA, which does not exist on ROCm, so `--torch-attn auto` uses the
   exact math attention on CPU/ROCm (ROCm PyTorch still calls its device
   `cuda`). `--sequential` frees the torch model before the JAX run when HBM
   is tight; `--random-weights --model-config JSON --aiter-kernel emul` is
   the CPU smoke setup, and `--negative-control` (JAX ignores the text mask)
   must fail.

### Train

```bash
# --attn aiter: ~2.5x the throughput of `ref` on MI355X (stage 2: 2.2 vs
# 0.89 samples/s/GPU); `ref` is only the portable default
python scripts/train_full.py \
  --base-ckpt ./ckpt/raw.safetensors \
  --data-dir /scratch/krea2-data --text-encoder Qwen/Qwen3-VL-4B-Instruct \
  --output-dir ./full_out \
  --steps 20000 --batch-size 32 --lr 1e-5 --warmup-steps 500 \
  --cosine-decay --min-lr 1e-6 --ema-decay 0.999 \
  --attn aiter --act-sharding on \
  --save-every 1000 --keep 3 --export-every 5000 --resume
```

Export `JAX_PLATFORMS` and `XLA_FLAGS` first (see the install section).

Useful knobs:

| flag | default | notes |
|---|---|---|
| `--data-dir` | – | `krea2-latents-v1` dir from `scripts/cache_latents.py` (required) |
| `--text-encoder` | – | Qwen3-VL text encoder: local HF dir or repo id (required); bf16, replicated (~7.8 GB/GPU), frozen |
| `--text-rows` | 256 | caption + 5 suffix rows handed to the DiT; captions are truncated to 251 tokens |
| `--img-tokens` | 4096 | static image rows per sample; every bucket's `h_tok * w_tok` must fit (the rest is masked) |
| `--caption-probs` | 0.5,0.3,0.2 | long,medium,short caption probabilities, drawn per sample per epoch |
| `--artist-prob` | 0.5 | probability of appending `", drawn by <artist>"` (never for an empty artist) |
| `--timestep-mu` | per sample | one shift for every sample instead of `mu` from each sample's token count |
| `--qchunk` | 256 | `ref` attention query chunk; must divide the padded sequence (4096 + 256 = 4352 = 17 x 256) |
| `--batch-size` | 8 | global; must be divisible by the device count |
| `--lr` / `--warmup-steps` | 1e-5 / 500 | linear warmup, then constant or `--cosine-decay` to `--min-lr` |
| `--weight-decay`, `--beta1/2`, `--eps` | 0.01, 0.9/0.999, 1e-8 | AdamW; `--grad-clip 1.0` (global norm) |
| `--ema-decay` | 0.999 | EMA of the fp32 weights, exported instead of the raw params; `0` disables |
| `--remat-policy` | dots | `full` saves the most HBM, `none` the least compute |
| `--attn` | ref | `ref` (exact fp32 softmax; portable default), `aiter` (jax-aiter flash attention; use it on MI355X, ~2.5x `ref`), `splash` (TPU) |
| `--act-sharding` | on | constrain activations to batch sharding (see above); `off` only for A/B |
| `--layer-loop` | scan | `unroll`: per-layer weight arrays + unrolled block loop, no per-layer slice/DUS copies on GPU, longer compile (see above); not resumable across modes |
| `--save-every` / `--keep` | 1000 / 3 | orbax checkpoints of params + optimizer + EMA |
| `--save-steps` | – | comma-separated extra save steps on top of `--save-every` (e.g. `20`: an early save that proves the checkpoint path); counted against `--keep`, not part of the resume config |
| `--export-every` | 0 | extra `model_stepNNNNNN.safetensors` exports (0 = only at the end) |
| `--export-steps` | – | comma-separated extra export steps on top of `--export-every` (e.g. `1500,3000`); only steps below `--steps` (the final `model.safetensors` always happens), not part of the resume config |
| `--max-hours` | – | stop cleanly (checkpoint + export) after a wall-clock budget; counts only the train loop (see below) |
| `--resume` | off | continue the newest checkpoint in `--output-dir/ckpt`; without it the script refuses to reuse a directory that already holds checkpoints |
| `--resume-step N` | – | resume from an older step N; newer step dirs are renamed `<step>.superseded` |
| `--verify-latents` | on | re-hash every latent shard at startup (reads the whole ~49 GB cache once) and refuse the run if any shard's bytes differ from its manifest digest; `--no-verify-latents` trusts the manifest |
| `--prefetch` / `--loader-workers` | 4 / 4 | streaming loader queue depth / reader threads |

The step log prints `loss`, a smoothed `ema_loss`, the pre-clip gradient
norm (`gnorm`), the LR and `s/it`.

Stopping, time budget and resuming:

- **SIGTERM/SIGINT**: the first signal finishes the current step, then
  checkpoints, exports `model.safetensors` and exits 0 (the same path as
  `--max-hours`); a second signal aborts immediately. So handle the
  10-minute preemption notice with `kill -TERM <pid>`. A hard `docker stop`
  with its default 10 s grace cannot finish a 205 GB save: use
  `docker stop -t <seconds>`, or send the signal well before the deadline.
- **`--max-hours`** counts only the train loop: the base load, the resume
  restore, the compile and the final save + export are outside it, so leave
  margin (the final save of the 12.9B state is ~205 GB, plus a ~26 GB
  export).
- **`--resume-step N`** resumes from an older step; newer step dirs are
  renamed `<step>.superseded`. Use it when the newest step dir is incomplete
  (e.g. an interrupted copy-back), which a plain `--resume` now reports with
  a hint to use it.
- Rerunning the same command after the run reached `--steps` only writes
  the missing `model.safetensors` (export-only) instead of refusing with
  "nothing to do".
- An interrupted save leaves `<step>.orbax-checkpoint-tmp`; the writing
  checkpoint manager removes stale ones when it opens.

Node-local disk is wiped between leases, so copy `ckpt/<step>` off the node
after each finalized save and copy it back before `--resume`. Build the
latent cache once and transfer it rather than rebuilding it per lease: the
VAE posterior sample is unseeded, so a rebuild changes the latents, its
manifest pass records different shard digests, and `--resume` refuses it
(dataset fingerprint mismatch) unless the digests match the original
cache's. The cache is treated as immutable for a run: with the default
`--verify-latents` the trainer re-hashes every shard at startup (before the
base checkpoint load) and refuses a cache whose bytes differ from its
manifest, so a shard corrupted in transfer, or rebuilt under the old
manifest, is caught instead of silently training on (or resuming with) data
the fingerprint does not describe. `--no-verify-latents` skips the re-hash
and trusts the manifest. A rerun that only needs the final export
(checkpoint step >= `--steps`) reads neither the dataset nor the text
encoder.

Memory (12.9B params, sharded over 8 devices; activations come on top and
scale with batch size, resolution and `--remat-policy`):

| tree | dtype | total | per GPU |
|---|---|---|---|
| master weights | fp32 | 51.6 GB | 6.5 GB |
| AdamW mu + nu | fp32 | 103.2 GB | 12.9 GB |
| EMA (`--ema-decay > 0`) | fp32 | 51.6 GB | 6.5 GB |
| **persistent train state** | | **206 GB** | **25.8 GB** |
| gradients (transient) | fp32 | 51.6 GB | 6.5 GB |
| bf16 compute copy (transient) | bf16 | 25.8 GB | 3.2 GB |
| text encoder (3.92B, replicated, frozen) | bf16 | 7.8 GB x 8 | 7.8 GB |

So roughly 38-44 GB/GPU of state, leaving most of the 288 GB HBM for
activations (the text tower's own activations are small: 290 tokens per
sample, no gradient). Host RAM: loading the base checkpoint peaks at about one fp32
copy (~52 GB) plus the bf16 file. Orbax async saves stage the **whole train
state in host RAM** before `save` returns (~206 GB with EMA, ~155 GB with
`--ema-decay 0`), so host RAM must comfortably exceed that plus the ~52 GB
base-load peak. Checkpoints on disk are the same ~206 GB / ~155 GB each, so
size `--keep` to your disk.

Every checkpoint records the run configuration, and `--resume` validates it
against the current flags. Changing `--seed`, `--batch-size`,
`--text-rows`, `--img-tokens`, `--caption-probs`, `--artist-prob`, the
dataset (fingerprint over ids, buckets, shards and their sha256 digests,
every caption and artist name, the VAE id, the tokenizer's template ids and
the normalized caption probabilities; sample count; latent channels), the
base checkpoint name, `--model-config`, `--text-model-config`, the text
encoder's identity (`text_encoder_id`: sha256 of tokenizer.json, the
resolved text config and the full contents of every safetensors file the
loader reads, so any weight value change counts; path and file-name
independent; hashing ~8 GB takes a few seconds to a minute at startup) and position mode, `--timestep-mu`,
`--logit-mean/--logit-std`, `--layer-loop` (checkpoints from before it was
recorded count as `scan`) or turning the EMA on/off is a hard error: those
would silently change the data order, captions, RNG, objective or
train-state shape. Changing `--steps`, `--lr`, `--warmup-steps`,
`--cosine-decay`, `--min-lr`, `--weight-decay`, `--beta1/2`, `--eps`,
`--grad-clip` or the `--ema-decay` value only prints a warning, since the LR
schedule is rebuilt and evaluated at the resumed step (e.g. extending a
cosine run with a larger `--steps` changes the next LR). `--attn`,
`--act-sharding` and the `--text-encoder` name (a local dir's basename or
the repo id as given, so a local copy of the same weights may be swapped in)
also only warn.
Older checkpoints without a recorded configuration resume with a warning.

### Export and sample

The run writes `<output-dir>/model.safetensors` (EMA weights when enabled,
else the raw params; bf16, official keys; metadata `step`, `base`, `source`,
`resolution: bucketed`, `img_tokens`) at the end, plus
`model_stepNNNNNN.safetensors` every `--export-every` steps and at each
`--export-steps` step. Any orbax step can also be exported afterwards, on
CPU, next to a running job:

```bash
python scripts/export_full.py --ckpt-dir ./full_out/ckpt --step 10000 \
  --source ema --output ./full_out/model_10k.safetensors \
  --base-ckpt ./ckpt/raw.safetensors   # optional: verify keys/shapes
```

The export is a drop-in Krea 2 checkpoint — sample it directly (no `--lora`),
or load it with the official PyTorch code / ComfyUI:

```bash
python scripts/sample.py --base-ckpt ./full_out/model.safetensors \
  --prompt "..." --steps 28 --cfg 4.5 --seed 0
```

### Quality eval on Modal (full FT vs base)

`scripts/modal_eval_full.py` samples the R2 export (`krea2-fullft/run1/
model.safetensors`) and `raw.safetensors` with the same prompt, seed and
sampler settings (28 steps, CFG 4.5, `--attn cudnn`) on one Modal GPU (H100
by default; the DiTs run one after the other), decodes on CPU containers and
writes `NN_{base,ft,pair}.png`, `prompts.json` and an `index.html` contact
sheet to `/home/mj/krea2-ops/eval/run1_<UTC>/`. Prompts: 40 dataset captions
from the 3 most common buckets (formatted like training) + 10 OOD prompts at
1024x1024. Text goes through the training text path (JAX Qwen3-VL tower,
`text_rows` 256).

```bash
modal run scripts/modal_eval_full.py --only-prompts   # prompt table (CPU only)
modal run scripts/modal_eval_full.py --limit 2        # smoke run
modal run scripts/modal_eval_full.py                  # full eval (rough: ~35 min, ~$3 on H100)
```

Any two R2 checkpoints can be compared: `--ft-key KEY` picks the "ft" side
(default `krea2-fullft/run1/model.safetensors`), `--base-key KEY` takes the
"base" side from R2 instead of HF, and `--labels A,B` names the two sides on
the pair images, `index.html` and `prompts.json` (files stay
`NN_base.png` / `NN_ft.png`). Each object is cached on the volume under a
path derived from its key (`krea2-fullft/<rest>` -> `/vol/ft/<rest>`, other
keys -> `/vol/r2/<key>`) with the same ETag/size/LastModified sidecar check:

```bash
modal run scripts/modal_eval_full.py \
  --base-key krea2-fullft/run1/model.safetensors \
  --ft-key krea2-fullft/run1/exports/model_step1513_params.safetensors \
  --labels run1-ema,run1-raw
```

### Export a checkpoint tree on Modal + weight drift

`scripts/modal_export_full.py` (one CPU container, 98 GiB RAM, volume
`krea2-eval-cache`) exports the raw params or the EMA of an orbax step on R2
without the training node: it downloads only what restoring that one tree
needs (`<run>/ckpt/<step>/{<source>/, meta/, top-level files}`, never
`opt_state/`; ETag/size-checked, resumable), runs `scripts/export_full.py`,
uploads `krea2-fullft/<run>/exports/model_step<step>_<source>.safetensors`
(user metadata: checkpoint fingerprint + sha256; a cached export or an
existing R2 object is reused only when made from the same checkpoint
objects, i.e. same keys, sizes and ETags), and prints a weight-drift report vs the base (`||W - W_base|| / ||W_base||`:
global, median / p90 / max over tensors, per DiT block, per tensor kind),
also written to `/home/mj/krea2-ops/eval/drift_<run>_<step>.json`. The base
`raw.safetensors` comes from `modal_eval_full.py`'s prepare (run that once
first).

```bash
modal run scripts/modal_export_full.py --run run1 --step 1513 --source params \
  --compare /vol/ft/run1/model.safetensors        # raw params vs the R2 EMA export
modal run scripts/modal_export_full.py --run run1 --step 1513 --drift-only \
  --compare /vol/ft/run1/model.safetensors        # report only (no R2 transfers)
```

### Update coherence of a checkpoint on Modal

`scripts/modal_coherence.py` (one CPU container, 56 GiB RAM, same volume)
asks whether a run learned coherently or only random-walked, with exact f32
numbers from the train state instead of bf16 exports. It downloads the
`params/`, `ema/` and `opt_state/` trees of `<run>/ckpt/<step>/` (the
download of `modal_export_full.py`; a tree already on the volume is not
fetched again) and reads them one leaf group at a time (a DiT block; orbax
partial restore with `ocp.PLACEHOLDER` for every other leaf), so the four
51 GB trees never sit in RAM together. Per tensor, per kind, per block and
globally, with `D = P - base`, `De = EMA - base` and the current Adam
direction `u = mhat / (sqrt(vhat) + eps)`: drift, `||De||/||D||`,
`cos(De, D)`, `cos(-u, D)`, `cos(-m, D)`, sign agreement, `rms(D)` per unit of
`sum lr` / `sqrt(sum lr^2)`, and what the bf16 export loses (changed
elements rounded back to the base). A null / calibration simulation (Adam on
`g = s + N(0, 1)` with the run's schedule) turns the measured `cos(-u, D)`
and `||De||/||D||` into an effective SNR and the coherent fraction of
`||D||^2`. Report: `/home/mj/krea2-ops/eval/coherence_<run>_<step>.json`.

```bash
modal run scripts/modal_coherence.py --run run1 --step 1513
.venv/bin/python scripts/modal_coherence.py --sim-only   # simulation table only (local)
```

### Dataset filter: inset / border score on Modal

`scripts/modal_score_borders.py` scores every image of
`ricecake/danbooru-curated-v2` for "a picture placed on a flat canvas"
(white or flat margins around a rectangular picture, pillar/letterbox bars,
a frame line + mat) and builds a review page to pick images for removal.
CPU only, Modal default resources (a 950-image shard peaks at ~90 MiB RSS).

Metric (`border_metrics`, numpy, full resolution): canvas colour candidates
are the medians of the 2 px outer ring and of each side strip (only
identical colours merged; a colour covering < 49% of every side strip is
skipped, which provably cannot give a margin); each goes through the full
margin geometry and the best score wins (`bg_src`), so bars on only two
sides work even when the picture's own edges cover more of the ring or sit
within `TOL` of the bars. Canvas pixels are within `TOL` =
18 (max abs channel difference: survives WebP noise/ringing, still separates
near-white picture highlights). Margins are leading rows/columns that are
>= 98% canvas; `fill` = non-canvas fraction of the remaining box; per-side
straightness = the box edge's non-canvas fraction (best of the first 4
lines); `edge_straightness` = their geometric mean over the sides with a
margin (a rectangle gives ~1, a silhouette ~0.1, and one flat cut of a bust
cannot carry three round sides).
`score = margin_area_frac * edge_straightness * min(1, fill / 0.85)` when the
ring is >= 90% canvas or the image is `pillar_or_letter` (margins on exactly
two opposite sides, straightness >= 0.9), else 0. For every candidate whose
margins are a thin frame line (all < 1.5%) the image is also measured
inside the line (`peeled`: frame + mat + inset scores like the inset), and
the final winner is chosen over all peeled / unpeeled results; a frame line
around a full-bleed picture only sets `thin_frame`. `side_pattern` records
which sides carry a band: `4`, `3`, `2-opp` (top+bottom or left+right),
`2-adj`, `1`, `0`; a border is 4, 3 or 2 opposite sides. A side counts when
its margin is >= 3 px (a true k px band measures ~k-1 px after WebP q80;
plain full-bleed WebPs give 0 px) and its straightness is >= 0.25 (a
backdrop panel behind a character gives ~0.26-0.72 depending on how much of
the side it spans, a rounded-corner inset ~0.4, a round silhouette tip
~0.1; a silhouette with one flat cut still has only one such side); this is
separate from the 1.5% `n_sides` threshold the score uses.
All intermediate numbers are stored, so the ranking can
be re-weighted from `scores.parquet` without re-running.

Flow: `prepare` resolves the dataset repo's commit sha once (`--revision`,
default the default branch head); metadata.parquet and both tar passes are
downloaded at that commit (the local parquet copy is used only when
byte-identical to the commit's file), and the sha is recorded in
`summary.json`, the `scores.parquet` schema metadata (`dataset_revision`)
and the review page / its export. Pass 1, one container per tar
(`.starmap`): download, score every
image, thumbnails (512 px JPEG) of the always-review ids. Tar members must
be `<id>.{webp,png,jpg,jpeg,json,txt}` and the image ids of every tar must
equal the parquet's ids for it (missing / extra / duplicate ids, a failed
container or an odd member fail the run with exit != 0 and nothing
written). Then the global selection (top `--top-n` by score among the
border side patterns, every
pillar/letterbox, every `sample watermark` image, every `border` /
`white border` / `outside border` / `pillarboxed` image whose long caption
does not mention border/frame/margin/letterbox/pillarbox/inset, and
`--control-n` seeded random `white background` images); pass 2 fetches
exactly the missing thumbnails from the tars that hold them, and a selected
id without a thumbnail fails the run.

Output in `/home/mj/krea2-ops/eval/datafilter/`: `scores.parquet` (every
image: metrics + tag/caption flags), `thumbs/<id>.jpg`, `summary.json` and
`review/index.html` (self-contained; tabs A top `--top-n` by score among
images with score > 0 and side pattern 4 / 3 / 2 opposite, B sample
watermark, C border tag without a border caption, D white-background
control, E pillar/letterbox; every card carries its side-pattern badge, e.g.
"2면(마주봄)"). A click cycles keep -> 캡션 보완 (caption, amber) -> 자르기
(crop, blue) -> 제거 (remove, red) -> keep; the header shows the legend with
per-state counts. States are kept in localStorage under
`datafilter_review_v2:<review id>` as `{id: state}` (foreign ids and
malformed entries dropped); "export" downloads `datafilter_actions.json`:
`{review_id, generated, dataset_revision, items: [{id, action, tabs, score,
side_pattern}]}` for the non-keep images.

`--rerender DIR` rebuilds only `DIR/review/index.html` of a finished run,
locally and without tars: rows from `scores.parquet` (side patterns are
recomputed under the current rule from the rounded margin / straightness
columns, with the integer px and coverage counts recovered from h/w), the
revision and the run's tab A rule from `summary.json`, captions and list D
from the previous page (a v1 page is first copied to
`review/index.v1.html`); B/C/E are re-derived and must equal the previous
page's lists. By default tab A = every border-pattern row within the run's
score-rank cutoff (`a_rule.rank_cutoff` in `summary.json`; a legacy run
without `a_rule`: its unfiltered top-N, all thumbnailed). The membership is
decided from scores, patterns and the cutoff alone; then any tab id without
a thumbnail (e.g. a larger `--top-n`, or a newer side rule reaching rows the
run never thumbnailed) fails with the count and ids, nothing written. The
rule used (`a_rule`: name, version, thresholds, top_n, rank_cutoff) is
recorded in `summary.json` by new runs and on every page.

```bash
modal run scripts/modal_score_borders.py --shards 2   # smoke run (2 tars)
modal run scripts/modal_score_borders.py              # all 100 tars (rough: a few min, < $0.5)
python scripts/modal_score_borders.py --tar-dir DIR --out OUT   # same, tars already on disk
python scripts/modal_score_borders.py --rerender /home/mj/krea2-ops/eval/datafilter  # page only
```

### Dataset filter: apply the review (patch directory)

`scripts/datafilter_apply.py` turns the review export
(`datafilter_actions.json`) into a **patch directory** that the latent patch
builder consumes; local CPU only, no Modal / HF / network. Inputs:
the actions json, `metadata.parquet`, the review run's `scores.parquet`
(its `dataset_revision` must equal the export's), the originals
(`/home/mj/.cache/krea2-orig/img/<id>.webp`, from the Drive fetch) and the
Qwen3-VL tokenizer of the local HF cache. Unknown / duplicate ids, unknown
actions or a revision mismatch fail before anything is written.

* **crop**: the original (alpha composited on white, a non-sRGB ICC profile
  converted to sRGB) is measured with `border_metrics` (box-downscaled to a
  long side <= 2048 for the measurement, margins mapped back so that no
  canvas survives the downscale, then every side with a margin inset 1 px),
  cropped, put in the bucket of the nearest aspect ratio (log space) among
  the dataset's 19 buckets and cover-resized + center-cropped by
  `cache_dataset.load_image` (Lanczos, the dataset path), saved as WebP q95.
  Border phrases are stripped from long / medium / short. A crop that would
  need upscaling -> `crop_too_small` (no image, old bucket + captions kept,
  planned strip in `notes`); an original whose score is 0 or < 0.5x the
  dataset image's, or a box keeping >= 98% of the original -> `crop_unsure`
  (image produced, flagged); a missing / undecodable original -> `error`.
* **caption**: a phrase from the dataset image's measurement + tags is
  appended to every caption that does not already mention a border:
  `<colour> border` (white / black / simple colour names from the canvas
  colour; `white` from the `white border` tag when nothing was measured;
  `border around the image` without either), `<colour> bars on the left and
  right` / `on the top and bottom` for 2 opposite sides (or the
  `pillarboxed` tag), plus `character extending outside the border` for
  `outside border`.
* **remove**: recorded only.

Stripping is phrase level (comma split, sentences inside a phrase one by
one): a phrase goes as a whole only when every word is border vocabulary
(border, letterbox, pillarbox, "white/thin/... frame", "white/... margins",
"bars at top and bottom", ...) or a colour / modifier / position word
("thin white border framing the image"); a `with` / `framed by` /
`surrounded by` ... tail of pure border words is cut from a real head
("plain grey background with white border" -> "plain grey background");
"framed / bordered background" loses the adjective. Phrases with other
content ("border collie", "mirror bordered with bulbs", "chibi inset",
"signature in the right margin", quoted text) are kept and listed in
`notes` for review. Token counts are recounted like the dataset's own
`*_tok` columns (caption + `, drawn by {artist}`, `CaptionTemplate`); a
caption that an edit pushes past the 251-token budget -> `over_budget`.

Output: `<out>/images/<id>.webp` (crops with an image only),
`<out>/patch.parquet` (id, action, bucket_w/h, long/medium/short + `_old` +
`_tok`, crop_box_orig `[x0, y0, x1, y1]` in original px, orig_w/h, status
`ok | crop_too_small | crop_unsure | over_budget | error`, notes),
`<out>/patch.json` (dataset revision, review id, counts, parameters,
tokenizer sha256, tool versions) and `<out>/report.html` (self-contained:
original with the crop box + final image, caption diffs with removed red /
added green, status badges and filters). Written to a temp dir and renamed;
exit 2 when any row is `error` (patch still written).

```bash
python scripts/datafilter_apply.py --actions ~/다운로드/datafilter_actions.json \
    --out /home/mj/krea2-ops/eval/datafilter/patch    # [--overwrite] [--workers 4]
```

## Tests

CPU-only numerical verification against the official PyTorch code
(downloads `mmdit.py` from the official repo on first run):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install einops optax
python tests/test_parity.py        # forward parity (~4e-7, asserts <2e-4), LoRA == merged
python tests/test_train_and_io.py  # train step, save/load, merge script
python tests/test_sharding.py      # 8-device FSDP parity (emulated on CPU)
python tests/test_train_full.py    # full fine-tuning: loss, export parity, resume,
                                   # padded image rows inert, per-sample mu
python tests/test_export.py        # official-layout export, param_shardings, orbax round trip
python tests/test_data.py          # streaming loaders: fixed-res cache + bucketed latents/captions
python tests/test_qwen3_text.py    # JAX Qwen3-VL text tower (parity parts need transformers)
python tests/test_cache_latents.py # krea2-latents-v1 cache builder (stub VAE)
python tests/test_aiter_attention.py  # padded-group-mode emulation == reference attn;
                                      # dit_forward aiter == ref (outputs + grads);
                                      # act_sharding on == off
python tests/test_act_sharding.py  # real-config AOT: abstract params == torch meta
                                   # state_dict; no activation collectives when on
python tests/test_scripts_smoke.py # aot_analysis / bench_step / parity scripts (tiny model)
python tests/test_sampling.py      # sampler: text caching, text compaction, head-sharded splash
python tests/test_layer_loop.py    # --layer-loop unroll == scan (outputs + grads), list-block
                                   # checkpoint I/O + shardings, no slice/DUS in the HLO
python tests/test_modal_eval_full.py  # Modal eval: prompt selection, caption format, tiny
                                      # generate path, decode/index writing (no Modal)
python tests/test_modal_export_full.py  # Modal export: minimal orbax download (fake R2),
                                        # export_full + upload flow, weight-drift report
python tests/test_modal_coherence.py    # Modal coherence: partial restore == full restore,
                                        # metrics, null-model simulation, fake-R2 flow
python tests/test_score_borders.py      # dataset border score: synthetic insets vs
                                        # silhouettes (WebP), shard id validation, two-pass
                                        # thumbnails, fake-tar run + review page (node),
                                        # side patterns, 4-state review logic, --rerender
python tests/test_datafilter_apply.py   # review -> patch dir: crop geometry on synthetic
                                        # originals, bucket choice, dataset resize reuse,
                                        # caption phrase add / strip, token budget, e2e
```

All of these run on CPU (8 emulated devices where sharding is involved).
The attention and act-sharding tests also pass under JAX 0.11 (the version
the jax-aiter stack pins).

## Repo layout

```
krea2_jax/
  config.py      # architecture + conditioning constants (single_mmdit_large_wide)
  model.py       # pure-JAX SingleStreamDiT (scan + remat, GQA, gated attention)
  checkpoint.py  # safetensors <-> sharded pytree, LoRA init/save/load, mesh/FSDP
  data.py        # streaming loaders: fixed-res cache (LoRA) + bucketed latents (full FT)
  qwen3_text.py  # JAX Qwen3-VL-4B text tower (online text encoder, full FT)
  text_template.py  # Qwen3-VL conditioner prompt template / caption tokenization
  fullft_ckpt.py # orbax train-state checkpoints (full fine-tuning)
  sampling.py    # Euler flow sampler + resolution-shifted timestep schedule
  r2.py          # Cloudflare R2 helpers of the Modal scripts (identity-checked downloads)
scripts/
  cache_dataset.py  # torch: VAE latents + Qwen3-VL states -> cache (LoRA)
  cache_latents.py  # torch: bucketed VAE latents of the HF dataset (full FT)
  text_encoder_parity.py  # torch conditioner vs JAX text tower (real weights)
  train_lora.py     # jax: TPU LoRA training
  train_full.py     # jax: full fine-tuning (fp32 master + AdamW + EMA, FSDP)
  export_full.py    # jax: orbax checkpoint -> official bf16 safetensors
  aot_analysis.py   # jax: compile the train step abstractly; memory + collectives
  bench_step.py     # jax: fwd / fwdbwd / train step-time + TFLOP/s benchmark
  parity_full.py    # torch vs jax full-model parity (node check)
  node_attention_parity.py  # jax-aiter attention vs reference (node check)
  sample.py         # jax DiT + torch encode/decode
  modal_eval_full.py  # Modal: full FT vs base side-by-side images + contact sheet
  modal_export_full.py  # Modal: export a tree of an R2 orbax checkpoint + weight drift
  modal_coherence.py  # Modal: update coherence of an R2 train state (f32) + null model
  modal_score_borders.py  # Modal: inset/border score of the dataset + review page
  datafilter_apply.py  # review actions -> patch dir (crops from originals, captions)
  merge_lora.py     # torch: bake LoRA into a full checkpoint
tests/              # parity / training / sharding verification
```

## Notes & limitations

- Single-host TPU slices (≤8 devices visible to one process). Multi-host
  pods would need per-host data sharding and `jax.make_array_from_*` when
  loading — not implemented yet.
- LoRA training (`train_lora.py`): one fixed training resolution per run —
  square or rectangular via `--width`/`--height` (no aspect bucketing).
  Full fine-tuning (`train_full.py`) trains on aspect buckets with
  per-sample positions, masks and timestep shift.
- Krea 2 weights are distributed under the **Krea 2 Community License**;
  review it before commercial use.
