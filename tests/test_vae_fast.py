"""Parity of scripts/vae_fast.py (2D single-frame encode path) against the
diffusers AutoencoderKLQwenImage reference, CPU fp32, on a tiny randomly
initialised VAE with the real structure (dim_mult [1, 2, 4, 4], 2 res blocks,
temperal_downsample [False, True, True], mid attention, z_dim 16).

    python tests/test_vae_fast.py              # skips if diffusers is missing
    KREA2_TEST_COMPILE=1 python tests/test_vae_fast.py   # + 2d-compile

Checks, for 2 non-square bucket-like shapes x batch 1 and 3:
  * posterior mean and std: max |2d - ref| / max |ref| < 1e-5
  * the normalized sample contract: same seed -> same noise draw as
    diffusers' randn_tensor, so make_encoder(.., "2d")(imgs) equals
    cache_latents.vae_encode_batch(imgs) (shape (N, 16, H/8, W/8), (z -
    latents_mean) / latents_std)
  * make_encoder(.., "ref") is vae_encode_batch exactly
  * the bench's correctness gate (modal_cache_latents.check_impls) passes the
    2d impl, rejects a wrong tap, a doubled std and a bf16-only mean offset,
    restores dtype / TF32 flags; bench_gpu_impl end to end on CPU returns
    plain JSON
  * channels_last moments from the 2D net still give contiguous latents
"""

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import torch  # noqa: E402

TOL = 1e-5
SHAPES = [(96, 64), (48, 112)]   # (H, W): portrait / landscape, multiples of 8
BATCHES = (1, 3)


def max_rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def tiny_vae(seed=0):
    from diffusers import AutoencoderKLQwenImage

    torch.manual_seed(seed)
    vae = AutoencoderKLQwenImage(base_dim=8, z_dim=16).eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():   # non-trivial gammas / biases too
        for name, p in vae.named_parameters():
            if name.endswith("gamma"):
                p.copy_(1 + 0.3 * torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    return vae


def stand_in_encoder(vae):
    """A cache_dataset.VaeEncoder without from_pretrained."""
    from cache_dataset import VaeEncoder

    enc = VaeEncoder.__new__(VaeEncoder)
    enc.vae, enc.device, enc.dtype = vae, "cpu", torch.float32
    enc.mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    enc.std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    return enc


def check_impl(enc, impl, log):
    import vae_fast
    from cache_latents import vae_encode_batch

    fast = vae_fast.make_encoder(enc, impl)
    ref = vae_fast.make_encoder(enc, "ref")
    worst = {"mean": 0.0, "std": 0.0, "sample": 0.0}
    for h, w in SHAPES:
        for b in BATCHES:
            g = torch.Generator().manual_seed(h * 1000 + w + b)
            imgs = torch.rand(b, 3, h, w, generator=g) * 2 - 1
            m_ref, s_ref = ref.dist(imgs)
            t0 = time.time()
            m, s = fast.dist(imgs)
            dt = time.time() - t0
            assert m.shape == m_ref.shape == (b, 16, h // 8, w // 8), m.shape
            em, es = max_rel(m, m_ref), max_rel(s, s_ref)
            # sample contract: same seed -> same noise as randn_tensor
            torch.manual_seed(7)
            z_ref = vae_encode_batch(enc, imgs)
            torch.manual_seed(7)
            z = fast(imgs)
            torch.manual_seed(7)
            z_same = ref(imgs)
            assert torch.equal(z_same, z_ref), "ref impl != vae_encode_batch"
            assert z.shape == z_ref.shape == (b, 16, h // 8, w // 8), z.shape
            ez = max_rel(z, z_ref)
            log(f"  {impl:10s} {h}x{w} b{b}: mean {em:.2e}  std {es:.2e}  "
                f"normalized sample {ez:.2e}  ({dt:.2f}s)")
            for k, v in (("mean", em), ("std", es), ("sample", ez)):
                worst[k] = max(worst[k], v)
                assert v < TOL, f"{impl} {h}x{w} b{b}: {k} max rel {v:.3e}"
    # normalization: (sample - latents_mean) / latents_std of the 2D dist
    imgs = torch.rand(2, 3, *SHAPES[0], generator=torch.Generator()
                      .manual_seed(3)) * 2 - 1
    m, s = fast.dist(imgs)
    torch.manual_seed(11)
    noise = torch.randn(m.shape)
    torch.manual_seed(11)
    z = fast(imgs)
    want = (m + s * noise - enc.mean[:, :, 0]) / enc.std[:, :, 0]
    assert max_rel(z, want) < TOL, max_rel(z, want)
    return worst


def check_bench_gate(log):
    """modal_cache_latents.check_impls (the bench's correctness gate) on the
    tiny VAE in bf16 (the production dtype): fp32 phase with TF32 off, then
    bf16 encoders handed to the ladder; a wrong impl is failed and dropped;
    the VAE's dtype and the TF32 flags are restored."""
    import contextlib
    import io

    import modal_cache_latents as mcl
    import vae_fast

    enc = stand_in_encoder(tiny_vae(seed=5))
    enc.to_dtype(torch.bfloat16)
    x = torch.rand(2, 3, *SHAPES[1], generator=torch.Generator()
                   .manual_seed(9)) * 2 - 1
    tf32 = (torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32)
    with contextlib.redirect_stdout(io.StringIO()):
        checks, encs = mcl.check_impls(enc, ["ref", "2d"], x)
    assert enc.dtype == torch.bfloat16 and next(
        enc.vae.parameters()).dtype == torch.bfloat16
    assert (torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32) == tf32
    assert set(encs) == {"ref", "2d"}, checks
    c = checks["2d"]
    assert c["ok"] and c["fp32_mean_max_rel"] < TOL, c
    assert 0 < c["bf16_mean_max_rel"] < 0.1, c   # bf16 vs fp32 ref
    log(f"  bench gate: 2d fp32 {c['fp32_mean_max_rel']:.2e}, bf16 "
        f"{c['bf16_mean_max_rel']:.2e}, ref bf16 "
        f"{checks['ref']['bf16_mean_max_rel']:.2e}")
    with contextlib.redirect_stdout(io.StringIO()):
        lad = mcl.run_ladder(encs["2d"], list(x), (2,),
                             (16, SHAPES[1][0] // 8, SHAPES[1][1] // 8),
                             min_images=2, min_seconds=0.0)
    assert "img_s" in lad[0] and "warmup_s" in lad[0], lad
    # a wrong 2D tap must fail the gate (and not be benchmarked)
    orig = vae_fast._conv2d_from_causal3d

    def wrong_tap(conv):
        c2 = orig(conv)
        if conv.kernel_size[0] == 3:
            c2.weight.data = conv.weight.detach()[:, :, 0].contiguous()
        return c2
    vae_fast._conv2d_from_causal3d = wrong_tap
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            checks, encs = mcl.check_impls(enc, ["2d"], x)
    finally:
        vae_fast._conv2d_from_causal3d = orig
    assert not checks["2d"]["ok"] and encs == {}, checks
    assert checks["2d"]["fp32_mean_max_rel"] > mcl.CHECK_TOL, checks
    log(f"  bench gate: wrong tap rejected (fp32 "
        f"{checks['2d']['fp32_mean_max_rel']:.2e})")

    # a doubled std, and a mean offset only in bf16, must fail the gate
    post = vae_fast.posterior

    def doubled_std(moments):
        m, sd = post(moments)
        return m, 2 * sd

    def bf16_mean_offset(moments):
        m, sd = post(moments)
        if m.dtype == torch.bfloat16:
            m = m + 0.1 * m.abs().max()
        return m, sd
    for patch, keys in ((doubled_std, ["fp32_std_max_rel",
                                       "fp32_sample_max_rel",
                                       "bf16_std_max_rel"]),
                        (bf16_mean_offset, ["bf16_mean_max_rel"])):
        vae_fast.posterior = patch
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                checks, encs = mcl.check_impls(enc, ["ref", "2d"], x)
        finally:
            vae_fast.posterior = post
        c = checks["2d"]
        assert not c["ok"] and "2d" not in encs, c
        assert [r.split()[0] for r in c["fail_reasons"]] == keys, c
        assert checks["ref"]["ok"], checks["ref"]
        log(f"  bench gate: {patch.__name__} rejected: "
            + "; ".join(c["fail_reasons"]))

    # the bench container's code path end to end on CPU (decode, gate,
    # ladder): plain JSON result (it is unpickled where torch is missing)
    import json
    import tempfile

    from test_cache_latents import make_fake_dataset

    enc = stand_in_encoder(tiny_vae(seed=6))
    enc.to_dtype(torch.bfloat16)
    with tempfile.TemporaryDirectory() as d:
        make_fake_dataset(d)
        lines = []
        res = mcl.bench_gpu_impl(
            "L4", 2, ["ref", "2d"], vae=enc, tar=Path(d) / "data/train-0000.tar",
            bucket=(32, 64), device="cpu", ladder=(1, 2), min_images=1,
            min_seconds=0.0, decode_timing_n=2, log=lines.append)
    assert "error" not in res, res["error"]
    assert json.loads(json.dumps(res)) == res
    assert all(res["impls"][i]["check"]["ok"] for i in ("ref", "2d")), res
    rows = [mcl.cost_row(e) for e in mcl.flatten_results([res])]
    assert [r["impl"] for r in rows] == ["ref", "2d"], rows
    log(f"  bench_gpu_impl (CPU, tiny VAE): ref {rows[0]['img_s']:.1f}, 2d "
        f"{rows[1]['img_s']:.1f} img/s; JSON-plain result")


def check_channels_last(log):
    """The 2D net may hand back channels_last moments (cudnn.benchmark /
    inductor): make_encoder's callable still returns contiguous latents
    (safetensors needs them), and the bench gate + ladder do not depend on
    the layout."""
    import contextlib
    import io

    import modal_cache_latents as mcl
    import vae_fast

    enc = stand_in_encoder(tiny_vae(seed=7))
    x = torch.rand(2, 3, *SHAPES[0], generator=torch.Generator()
                   .manual_seed(4)) * 2 - 1
    fwd = vae_fast.Encoder2d.forward

    def channels_last_forward(self, inp):
        return fwd(self, inp).contiguous(memory_format=torch.channels_last)
    vae_fast.Encoder2d.forward = channels_last_forward
    try:
        fast = vae_fast.make_encoder(enc, "2d")
        m, _ = fast.dist(x)
        assert not m.is_contiguous(), m.stride()   # really strided
        z = fast(x)
        assert z.is_contiguous() and z.shape == (2, 16, SHAPES[0][0] // 8,
                                                 SHAPES[0][1] // 8), z.stride()
        torch.manual_seed(3)
        z_ref = vae_fast.make_encoder(enc, "ref")(x)
        torch.manual_seed(3)
        assert max_rel(fast(x), z_ref) < TOL
        with contextlib.redirect_stdout(io.StringIO()):
            checks, encs = mcl.check_impls(enc, ["ref", "2d"], x)
            lad = mcl.run_ladder(encs["2d"], list(x), (2,),
                                 (16, SHAPES[0][0] // 8, SHAPES[0][1] // 8),
                                 min_images=2, min_seconds=0.0)
    finally:
        vae_fast.Encoder2d.forward = fwd
    assert checks["2d"]["ok"], checks["2d"]
    assert "img_s" in lad[0], lad
    log("  channels_last moments: contiguous latents, gate + ladder OK")


def main():
    try:
        import diffusers
    except ImportError:
        print("SKIP test_vae_fast: diffusers is not installed (needs "
              "diffusers>=0.35 with AutoencoderKLQwenImage)")
        return
    torch.set_num_threads(min(4, torch.get_num_threads()))
    t0 = time.time()
    print(f"diffusers {diffusers.__version__}, torch {torch.__version__}")
    enc = stand_in_encoder(tiny_vae())
    impls = ["2d"] + (["2d-compile"] if os.environ.get("KREA2_TEST_COMPILE")
                      else [])
    for impl in impls:
        worst = check_impl(enc, impl, print)
        print(f"  {impl}: worst max rel error mean {worst['mean']:.2e}, std "
              f"{worst['std']:.2e}, sample {worst['sample']:.2e} (< {TOL})")
    check_bench_gate(print)
    check_channels_last(print)
    print(f"ALL VAE_FAST TESTS PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
