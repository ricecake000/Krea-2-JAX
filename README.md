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

## Full fine-tuning (8x MI355X / ROCm)

`scripts/train_full.py` trains **every** DiT weight instead of a LoRA, on one
host with 8 accelerators (developed for 8x AMD MI355X, 288 GB HBM each; any
single-host JAX backend works). Same flow-matching objective, timestep
sampling and data cache as the LoRA trainer; differences:

- fp32 **master weights**, AdamW moments and an optional **EMA** copy, all
  FSDP-sharded across the 8 devices exactly like the base params (only
  scalars are replicated — the script asserts this at startup). The forward
  and backward run in bf16 from a per-step cast of the master weights.
- The dataset is **streamed** from the cache (`krea2_jax/data.py`: a
  background thread pool with a bounded prefetch queue), so ~100k samples
  never have to fit in host RAM.
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
TPU-only and keeps the TPU pin in `requirements-tpu.txt`; full fine-tuning on
GPU uses the default `--attn ref`, or `--attn aiter` with the pinned
jax-aiter stack below.)

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

### jax-aiter flash attention (`--attn aiter`, MI355X only)

[jax-aiter](https://github.com/ROCm/jax-aiter) exposes AMD's AITER kernels
to JAX via XLA FFI; `--attn aiter` uses its fused `flash_attn_varlen`
(forward + backward through `custom_vjp`, GQA native, bf16, head dim 128)
instead of the reference attention, whose `[heads, qchunk, L]` fp32 logits
dominate activation memory and bandwidth. It is **gfx950-only** and
supported on exactly one stack (jax-aiter 0.1.0-alpha2):

| component | version |
|---|---|
| image | `ghcr.io/rocm/jax-base-ubu24.therock-7.14:7.14` (TheRock 7.14 GA, no `/opt/rocm`) |
| Python | 3.12 |
| `jax` / `jaxlib` | 0.11.0 |
| `jax-rocm7-plugin` / `jax-rocm7-pjrt` | 0.11.0.post1 |
| jax-aiter | `jax_aiter-0.1.0a2+full-...whl` from the [GitHub release](https://github.com/ROCm/jax-aiter/releases/tag/v0.1.0-alpha2) |

```bash
pip install "jax==0.11.0" "jaxlib==0.11.0" \
  "jax-rocm7-plugin==0.11.0.post1" "jax-rocm7-pjrt==0.11.0.post1"
pip install ./jax_aiter-0.1.0a2+full-cp312-cp312-manylinux_2_39_x86_64.whl
python -c "from jax_aiter.mha import flash_attn_varlen; print('MHA ready')"
```

(The plain `0.1.0a2` wheel omits the MHA libraries; run
`jax-aiter-fetch-mha` after installing it.) jax-aiter is imported lazily,
only when `--attn aiter` is used, so nothing else needs it.

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

### Cache the dataset on all 8 GPUs

Each shard is an independent process on one GPU; file numbering stays
global, so shards never collide. A final unsharded pass reuses every entry
(no model load) and writes `meta.json`:

```bash
for i in $(seq 0 7); do
  HIP_VISIBLE_DEVICES=$i python scripts/cache_dataset.py \
    --data-dir ./data --cache-dir ./cache --resolution 1024 \
    --num-shards 8 --shard-index $i --device cuda &
done
wait
python scripts/cache_dataset.py --data-dir ./data --cache-dir ./cache --resolution 1024
```

(ROCm PyTorch exposes the GPUs as `cuda`.) Budget disk for the cache: a
1024px latent is 0.5 MB and the text states ~60 KB per caption token, so
~100k samples with ~100-token captions are ~650 GB.

### Bring-up on a new node

Run these before the first real training job; none needs the dataset cache.

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
   python scripts/aot_analysis.py --batch-size 64 --resolution 1024 \
     --txt-len 512 --remat-policy dots --attn ref --act-sharding on
   ```

   Other flags: `--width/--height`, `--ema-decay`, `--qchunk`,
   `--model-config JSON`, `--no-compile` (lower only), `--dump-hlo PATH`.

2. **`scripts/bench_step.py`** — step-time benchmark on random inputs:
   `--mode fwd` (forward), `fwdbwd` (value_and_grad w.r.t. bf16 weights,
   grads cast to `--grad-dtype`) or `train` (the real `train_full` step:
   fp32 master + AdamW + EMA). Weights come from `--base-ckpt` or
   `--random-weights` (generated on device). Prints median step time,
   samples/s, analytic TFLOP/s per device (6 x params x padded tokens x
   batch; x4/3 for `--remat-policy full`, x1.1 for `dots`; 2x for `fwd`) and
   peak HBM when the backend reports it.

   ```bash
   python scripts/bench_step.py --random-weights --mode train \
     --batch-size 64 --width 1024 --height 1024 --remat-policy dots \
     --attn aiter --act-sharding on --steps 10 --warmup 2
   ```

   Also `--base-sharding fsdp|replicate`, `--txt-len`, `--qchunk`,
   `--model-config`, `--seed`.

3. **`scripts/node_attention_parity.py`** — kernel-level check of
   `aiter_attention` against the reference attention (padded query rows
   zeroed, the kernel contract), forward and gradients (dq, dk, dv), on random
   prefix masks, compared to an fp32 truth, plus median per-call ms of the
   jitted forward and forward+backward. Exits non-zero above `--tol` (2e-2
   bf16, 1e-4 f32). `--shard` shards the batch over all devices (exercises
   the `shard_map` wrapper); `--base-ckpt` (or `--random-weights` with a tiny
   `--model-config`) adds a DiT-level `attn_impl="ref"` vs `"aiter"` check at
   `--width/--height` with mid-sequence text padding (`--dit-grad`: also the
   gradients). `--kernel emul` swaps in the pure-JAX emulation of the padded
   group mode, so it runs on CPU.

   ```bash
   python scripts/node_attention_parity.py --kernel aiter --batch 2 --heads 48 \
     --kv-heads 12 --seq 4608 --head-dim 128 --dtype bf16 --iters 20 --shard
   python scripts/node_attention_parity.py --kernel aiter \
     --base-ckpt ./ckpt/raw.safetensors --width 1024 --height 1024 --dit-grad
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
python scripts/train_full.py \
  --base-ckpt ./ckpt/raw.safetensors \
  --cache-dir ./cache --output-dir ./full_out \
  --steps 20000 --batch-size 32 --lr 1e-5 --warmup-steps 500 \
  --cosine-decay --min-lr 1e-6 --ema-decay 0.999 \
  --save-every 1000 --keep 3 --export-every 5000 --resume
```

Useful knobs:

| flag | default | notes |
|---|---|---|
| `--batch-size` | 8 | global; must be divisible by the device count |
| `--lr` / `--warmup-steps` | 1e-5 / 500 | linear warmup, then constant or `--cosine-decay` to `--min-lr` |
| `--weight-decay`, `--beta1/2`, `--eps` | 0.01, 0.9/0.999, 1e-8 | AdamW; `--grad-clip 1.0` (global norm) |
| `--ema-decay` | 0.999 | EMA of the fp32 weights, exported instead of the raw params; `0` disables |
| `--remat-policy` | dots | `full` saves the most HBM, `none` the least compute |
| `--attn` | ref | `ref` (exact fp32 softmax), `aiter` (jax-aiter flash attention, MI355X), `splash` (TPU) |
| `--act-sharding` | on | constrain activations to batch sharding (see above); `off` only for A/B |
| `--save-every` / `--keep` | 1000 / 3 | orbax checkpoints of params + optimizer + EMA |
| `--export-every` | 0 | extra `model_stepNNNNNN.safetensors` exports (0 = only at the end) |
| `--max-hours` | – | stop cleanly (checkpoint + export) after a wall-clock budget |
| `--resume` | off | continue the newest checkpoint in `--output-dir/ckpt`; without it the script refuses to reuse a directory that already holds checkpoints |
| `--prefetch` / `--loader-workers` | 4 / 4 | streaming loader queue depth / reader threads |

The step log prints `loss`, a smoothed `ema_loss`, the pre-clip gradient
norm (`gnorm`), the LR and `s/it`.

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

So roughly 30-36 GB/GPU of state, leaving most of the 288 GB HBM for
activations. Host RAM: loading the base checkpoint peaks at about one fp32
copy (~52 GB) plus the bf16 file. Orbax async saves stage the **whole train
state in host RAM** before `save` returns (~206 GB with EMA, ~155 GB with
`--ema-decay 0`), so host RAM must comfortably exceed that plus the ~52 GB
base-load peak. Checkpoints on disk are the same ~206 GB / ~155 GB each, so
size `--keep` to your disk.

Every checkpoint records the run configuration, and `--resume` validates it
against the current flags. Changing `--seed`, `--batch-size`, the resolved
text length, the cache (resolution, sample count), the base checkpoint name,
`--model-config`, `--timestep-mu` (resolved), `--logit-mean/--logit-std` or
turning the EMA on/off is a hard error: those would silently change the data
order, RNG, objective or train-state shape. Changing `--steps`, `--lr`,
`--warmup-steps`, `--cosine-decay`, `--min-lr`, `--weight-decay`,
`--beta1/2`, `--eps`, `--grad-clip` or the `--ema-decay` value only prints a
warning, since the LR schedule is rebuilt and evaluated at the resumed step
(e.g. extending a cosine run with a larger `--steps` changes the next LR).
Older checkpoints without a recorded configuration resume with a warning.

### Export and sample

The run writes `<output-dir>/model.safetensors` (EMA weights when enabled,
else the raw params; bf16, official keys) at the end, plus
`model_stepNNNNNN.safetensors` every `--export-every` steps. Any orbax step
can also be exported afterwards, on CPU, next to a running job:

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

## Tests

CPU-only numerical verification against the official PyTorch code
(downloads `mmdit.py` from the official repo on first run):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install einops optax
python tests/test_parity.py        # forward parity (~4e-7, asserts <2e-4), LoRA == merged
python tests/test_train_and_io.py  # train step, save/load, merge script
python tests/test_sharding.py      # 8-device FSDP parity (emulated on CPU)
python tests/test_train_full.py    # full fine-tuning: loss, export parity, resume
python tests/test_export.py        # official-layout export, param_shardings, orbax round trip
python tests/test_data.py          # streaming cache loader (order, resume, threads)
python tests/test_aiter_attention.py  # padded-group-mode emulation == reference attn;
                                      # dit_forward aiter == ref (outputs + grads);
                                      # act_sharding on == off
python tests/test_act_sharding.py  # real-config AOT: abstract params == torch meta
                                   # state_dict; no activation collectives when on
python tests/test_scripts_smoke.py # aot_analysis / bench_step / parity scripts (tiny model)
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
  data.py        # streaming cache loader (full fine-tuning)
  fullft_ckpt.py # orbax train-state checkpoints (full fine-tuning)
  sampling.py    # Euler flow sampler + resolution-shifted timestep schedule
scripts/
  cache_dataset.py  # torch: VAE latents + Qwen3-VL states -> cache
  train_lora.py     # jax: TPU LoRA training
  train_full.py     # jax: full fine-tuning (fp32 master + AdamW + EMA, FSDP)
  export_full.py    # jax: orbax checkpoint -> official bf16 safetensors
  aot_analysis.py   # jax: compile the train step abstractly; memory + collectives
  bench_step.py     # jax: fwd / fwdbwd / train step-time + TFLOP/s benchmark
  parity_full.py    # torch vs jax full-model parity (node check)
  node_attention_parity.py  # jax-aiter attention vs reference (node check)
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
