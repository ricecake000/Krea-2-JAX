"""Full-size numerical parity: official PyTorch SingleStreamDiT vs our JAX DiT.

Both implementations get the SAME weights (the official safetensors file) and
the SAME random inputs: latents, text hidden states (B, L, txtlayers, txtdim)
with random per-sample valid lengths (so after the [text, image] concatenation
the padding sits in the middle of the sequence), and random flow times. The
image-token velocities are compared per sample (max abs, max abs / max |ref|,
relative-norm error ||jax - torch|| / ||torch||); the script exits non-zero
when any sample's relative-norm error exceeds --tol.

On the node (real weights, GPU torch, JAX on the accelerators):

    python scripts/parity_full.py --base-ckpt raw.safetensors \
        --torch-device cuda --dtype bf16 --attn ref --act-sharding on --sequential
    python scripts/parity_full.py --base-ckpt raw.safetensors \
        --torch-device cuda --dtype bf16 --attn aiter --sequential

CPU smoke (tiny config, fabricated random checkpoint):

    JAX_PLATFORMS=cpu python scripts/parity_full.py --random-weights \
        --model-config '{"features":128,"tdim":32,"txtdim":64,"heads":4,
        "kvheads":2,"layers":2,"channels":4,"txtheads":4,"txtkvheads":4,
        "txtlayers":3}' --width 128 --height 192 --txt-len 7 \
        --torch-device cpu --dtype f32 --tol 1e-3

Torch attention: the official `attention` forces the cuDNN SDPA backend, which
(unlike the math backend) returns zeros for fully-masked query rows. On CPU
(and on ROCm, where cuDNN does not exist) the script swaps in the exact math
version from tests/test_parity.patch_reference_for_cpu (same zero-row
semantics, fp32 logits). On NVIDIA CUDA the official kernel is used untouched.
Override with --torch-attn native|math.
"""

import argparse
import dataclasses
import gc
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # reference uses @torch.compile

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base-ckpt", help="official raw/turbo .safetensors")
    src.add_argument("--random-weights", action="store_true",
                     help="fabricate a random checkpoint (N(0, 0.02)) from the "
                          "reference module; meant for tiny --model-config runs")
    ap.add_argument("--ref-dir", default=None,
                    help="directory holding the official mmdit.py (default: "
                         "$KREA2_REF_DIR, else download like tests/test_parity)")
    ap.add_argument("--model-config", default=None,
                    help="JSON dict of Krea2Config overrides (tiny CPU tests)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--txt-len", type=int, default=77)
    ap.add_argument("--attn", choices=["ref", "aiter"], default="ref")
    ap.add_argument("--aiter-kernel", choices=["native", "emul"], default="native",
                    help="--attn aiter only: native = jax_aiter flash_attn_varlen "
                         "(gfx950), emul = model.varlen_reference_kernel (CPU)")
    ap.add_argument("--act-sharding", choices=["on", "off"], default="off")
    ap.add_argument("--jax-devices", type=int, default=0,
                    help="devices in the JAX mesh (0 = largest count <= "
                         "available that divides --batch)")
    ap.add_argument("--emulate-devices", type=int, default=0,
                    help="CPU: fake N host devices (sets XLA_FLAGS before jax)")
    ap.add_argument("--torch-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--torch-attn", choices=["auto", "native", "math"],
                    default="auto",
                    help="auto: math on cpu/ROCm, official cuDNN SDPA on CUDA")
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--qchunk", type=int, default=512)
    ap.add_argument("--tol", type=float, default=5e-2,
                    help="max allowed per-sample relative-norm error")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--negative-control", action="store_true",
                    help="JAX side ignores the text padding mask (all tokens "
                         "valid); parity must then FAIL — proves the check "
                         "has teeth")
    ap.add_argument("--sequential", action="store_true",
                    help="free the torch model (and CUDA cache) before JAX runs")
    return ap.parse_args(argv)


def import_reference(ref_dir):
    """Import the official mmdit module via tests/test_parity.get_reference
    without letting test_parity force JAX_PLATFORMS=cpu on the node."""
    if ref_dir:
        os.environ["KREA2_REF_DIR"] = ref_dir
    had = "JAX_PLATFORMS" in os.environ
    import test_parity  # noqa: E402  (sets JAX_PLATFORMS=cpu via setdefault)

    if not had:
        os.environ.pop("JAX_PLATFORMS", None)
    return test_parity, test_parity.get_reference()


def ref_config(mmdit, cfg):
    names = {f.name for f in dataclasses.fields(mmdit.SingleMMDiTConfig)}
    full = dataclasses.asdict(cfg)
    return mmdit.SingleMMDiTConfig(**{k: v for k, v in full.items() if k in names})


def fabricate_checkpoint(mmdit, cfg, path, seed):
    import torch
    from safetensors.torch import save_file

    with torch.device("meta"):
        model = mmdit.SingleStreamDiT(ref_config(mmdit, cfg))
    g = torch.Generator().manual_seed(seed + 1)
    sd = {k: torch.randn(v.shape, generator=g) * 0.02
          for k, v in model.state_dict().items()}
    save_file(sd, path)
    n = sum(v.numel() for v in sd.values())
    print(f"fabricated random checkpoint ({n / 1e6:.2f}M params) at {path}")


def make_inputs(cfg, args):
    """Random inputs shared by both frameworks (numpy, float32)."""
    if args.width % 16 or args.height % 16:
        raise SystemExit("width/height must be multiples of 16")
    rng = np.random.default_rng(args.seed)
    B, L = args.batch, args.txt_len
    latH = args.height // cfg.vae_compression
    latW = args.width // cfg.vae_compression
    htok, wtok = latH // cfg.patch, latW // cfg.patch
    lat = rng.standard_normal((B, cfg.channels, latH, latW)).astype(np.float32)
    context = rng.standard_normal((B, L, cfg.txtlayers, cfg.txtdim)).astype(np.float32)
    # random valid lengths, always with some padding in the text region
    # (except the last sample when B > 1, to also cover the no-padding case)
    if L > 1:
        lens = rng.integers(1, L, size=B)
        if B > 1:
            lens[-1] = L
    else:
        lens = np.ones(B, dtype=np.int64)
    txtmask = np.arange(L)[None, :] < lens[:, None]
    t = rng.uniform(0.05, 0.95, size=B).astype(np.float32)

    # reference-style prepare (identical to tests/test_parity.py)
    p = cfg.patch
    img = (lat.reshape(B, cfg.channels, htok, p, wtok, p)
              .transpose(0, 2, 4, 1, 3, 5)
              .reshape(B, htok * wtok, -1))
    imgids = np.zeros((htok, wtok, 3), np.float32)
    imgids[..., 1] = np.arange(htok)[:, None]
    imgids[..., 2] = np.arange(wtok)[None, :]
    pos = np.concatenate(
        [np.zeros((B, L, 3), np.float32),
         np.broadcast_to(imgids.reshape(1, -1, 3), (B, htok * wtok, 3))], axis=1)
    mask = np.concatenate([txtmask, np.ones((B, htok * wtok), bool)], axis=1)
    return dict(lat=lat, img=np.ascontiguousarray(img), context=context, t=t,
                txtmask=txtmask, pos=np.ascontiguousarray(pos), mask=mask,
                lens=lens, htok=htok, wtok=wtok)


def run_torch(test_parity, mmdit, cfg, ckpt, inp, args):
    import torch
    from safetensors.torch import load_file

    device = torch.device(args.torch_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--torch-device cuda but torch.cuda.is_available() is False")
    hip = getattr(torch.version, "hip", None)
    mode = args.torch_attn
    if mode == "auto":
        mode = "math" if (device.type == "cpu" or hip) else "native"
    if mode == "math":
        test_parity.patch_reference_for_cpu(mmdit, torch)
    print(f"torch: device={device}{' (ROCm ' + hip + ')' if hip else ''}, "
          f"attention={'math (patch_reference_for_cpu)' if mode == 'math' else 'official cuDNN SDPA'}")

    tdtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    t0 = time.perf_counter()
    with torch.device("meta"):
        model = mmdit.SingleStreamDiT(ref_config(mmdit, cfg))
    sd = load_file(ckpt, device=str(device))  # straight to the GPU on cuda
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    model = model.to(device=device, dtype=tdtype).eval()
    print(f"torch: model loaded in {time.perf_counter() - t0:.1f}s")

    def T(a, dt=None):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)

    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model(img=T(inp["img"], tdtype), context=T(inp["context"], tdtype),
                    t=T(inp["t"], torch.float32), pos=T(inp["pos"], torch.float32),
                    mask=T(inp["mask"]))
        out = out.float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
    print(f"torch: forward {time.perf_counter() - t0:.1f}s, out {out.shape}")
    if not np.all(np.isfinite(out)):
        print("torch: WARNING non-finite outputs; with --torch-attn native on a "
              "backend that does not zero fully-masked rows NaNs propagate — "
              "try --torch-attn math")
    return out, model


def run_jax(cfg, ckpt, inp, args):
    import jax
    import ml_dtypes
    import jax.numpy as jnp
    from jax.sharding import Mesh

    from krea2_jax.checkpoint import batch_sharding, load_krea2_params
    from krea2_jax.model import dit_forward

    devs = jax.devices()
    n = args.jax_devices or max(d for d in range(1, len(devs) + 1)
                                if args.batch % d == 0)
    if n > len(devs) or args.batch % n:
        raise SystemExit(f"--jax-devices {n}: need <= {len(devs)} and to divide "
                         f"--batch {args.batch}")
    mesh = Mesh(np.array(devs[:n]), axis_names=("fsdp",))
    print(f"jax: {jax.__version__} platform={devs[0].platform}, mesh of {n} "
          f"device(s), attn={args.attn}"
          f"{'/' + args.aiter_kernel if args.attn == 'aiter' else ''}, "
          f"act_sharding={args.act_sharding}")

    jdtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    t0 = time.perf_counter()
    params = load_krea2_params(ckpt, mesh=mesh, dtype=jdtype)
    print(f"jax: params loaded in {time.perf_counter() - t0:.1f}s")

    kw = dict(remat=False, qchunk=args.qchunk, attn_impl=args.attn)
    if args.act_sharding == "on":
        kw["act_sharding"] = batch_sharding(mesh)
    if args.attn == "aiter":
        kw["attn_mesh"] = mesh
        if args.aiter_kernel == "emul":
            from krea2_jax.model import varlen_reference_kernel
            kw["attn_kernel"] = varlen_reference_kernel

    shard = batch_sharding(mesh)
    # cast on the host (numpy / ml_dtypes) and place straight into the batch
    # sharding: no device ever holds the global batch
    np_dt = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float32
    put = lambda a, dt=None: jax.device_put(np.asarray(a, dt), shard)
    img = put(inp["img"], np_dt)
    context = put(inp["context"], np_dt)
    t = put(inp["t"], np.float32)
    pos = put(inp["pos"], np.float32)
    mask = put(np.ones_like(inp["mask"]) if args.negative_control else inp["mask"])
    if args.negative_control:
        print("jax: NEGATIVE CONTROL — text padding mask ignored; expect FAILED")

    fwd = jax.jit(lambda p, *a: dit_forward(p, cfg, *a, **kw))
    t0 = time.perf_counter()
    out = fwd(params, img, context, t, pos, mask)
    out = np.asarray(jax.device_get(out).astype(jnp.float32))
    print(f"jax: compile+forward {time.perf_counter() - t0:.1f}s, out {out.shape}")
    return out


def relnorm(a, ref):
    """||a - ref|| / ||ref||, both operands cast to float64 BEFORE any norm
    (a float32 norm of large values overflows to inf and would turn the
    ratio into a silent 0). Returns inf when the numerator or denominator
    is non-finite, so a NaN/inf output can never pass."""
    a = np.asarray(a, np.float64)
    ref = np.asarray(ref, np.float64)
    num = float(np.linalg.norm(a - ref))
    den = float(np.linalg.norm(ref))
    if not (math.isfinite(num) and math.isfinite(den)):
        return float("inf")
    return num / (den + 1e-12)


def compare(jx, ref, lens, tol):
    print("per-sample image-token velocity error (jax vs torch):")
    print(f"  {'b':>2} {'txt_valid':>9} {'max_abs':>10} {'max_rel':>10} "
          f"{'relnorm':>10} {'ref_absmax':>10}")
    jx = np.asarray(jx, np.float64)
    ref = np.asarray(ref, np.float64)
    worst = 0.0
    for b in range(ref.shape[0]):
        d = jx[b] - ref[b]
        max_abs = float(np.abs(d).max())
        scale = float(np.abs(ref[b]).max())
        max_rel = max_abs / (scale + 1e-12)
        rn = relnorm(jx[b], ref[b])
        worst = max(worst, rn)
        print(f"  {b:>2} {int(lens[b]):>9} {max_abs:>10.3e} {max_rel:>10.3e} "
              f"{rn:>10.3e} {scale:>10.3e}")
    d = jx - ref
    overall = relnorm(jx, ref)
    worst = max(worst, overall)
    print(f"overall: max_abs {np.abs(d).max():.3e}, max_rel "
          f"{np.abs(d).max() / (np.abs(ref).max() + 1e-12):.3e}, relnorm "
          f"{overall:.3e} "
          f"(max_rel = max|a-b| / max|ref|; relnorm = ||a-b|| / ||ref||)")
    if not math.isfinite(worst):
        print("non-finite relnorm: jax or torch output (or their difference) "
              "contains NaN/inf -> failure")
    ok = worst <= tol
    print(f"{'PARITY PASSED' if ok else 'PARITY FAILED'}: worst per-sample "
          f"relnorm {worst:.3e} {'<=' if ok else '>'} tol {tol:.1e}")
    return ok


def main(argv=None):
    args = parse_args(argv)
    if args.emulate_devices:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={args.emulate_devices}"
        )
    if args.torch_device == "cuda":
        # torch and JAX share the GPU(s): no 75% up-front JAX reservation
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    test_parity, mmdit = import_reference(args.ref_dir)

    from krea2_jax.config import KREA2_CONFIG

    cfg = KREA2_CONFIG
    if args.model_config:
        cfg = dataclasses.replace(cfg, **json.loads(args.model_config))
    print(f"config: features={cfg.features} layers={cfg.layers} heads={cfg.heads}/"
          f"{cfg.kvheads} txt={cfg.txtlayers}x{cfg.txtdim}; batch={args.batch} "
          f"{args.width}x{args.height} txt_len={args.txt_len} dtype={args.dtype}")

    tmpdir = None
    ckpt = args.base_ckpt
    if args.random_weights:
        tmpdir = tempfile.TemporaryDirectory()
        ckpt = os.path.join(tmpdir.name, "random.safetensors")
        fabricate_checkpoint(mmdit, cfg, ckpt, args.seed)

    inp = make_inputs(cfg, args)
    full = args.txt_len + inp["htok"] * inp["wtok"]
    print(f"sequence: {inp['htok']}x{inp['wtok']} = {inp['htok'] * inp['wtok']} "
          f"img tokens + {args.txt_len} txt = {full} -> padded "
          f"{full + (-full) % cfg.seq_pad_multiple}; txt valid lens "
          f"{inp['lens'].tolist()}; t {[round(float(x), 3) for x in inp['t']]}")

    ref, model = run_torch(test_parity, mmdit, cfg, ckpt, inp, args)
    if args.sequential:
        import torch

        del model
        gc.collect()
        if args.torch_device == "cuda":
            torch.cuda.empty_cache()
        print("torch: model freed (--sequential)")

    jx = run_jax(cfg, ckpt, inp, args)
    if tmpdir is not None:
        tmpdir.cleanup()
    ok = compare(jx, ref, inp["lens"], args.tol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
