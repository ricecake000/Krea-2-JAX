"""8 emulated CPU devices for every test; shared tiny-model fixtures."""

import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "tests", REPO / "scripts"):
    sys.path.insert(0, str(p))

import pytest  # noqa: E402

from helpers import TINY, make_fake_latent_cache, make_tiny_text_encoder_dir  # noqa: E402


@pytest.fixture(scope="session")
def ref():
    """The official mmdit module, patched for CPU attention."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    from test_parity import get_reference, patch_reference_for_cpu
    mmdit = get_reference()
    patch_reference_for_cpu(mmdit, torch)
    return mmdit


@pytest.fixture(scope="session")
def tiny_ckpt(ref, tmp_path_factory):
    """Random tiny official-layout checkpoint: (fp32 state dict, fp32 path,
    bf16 path, torch model)."""
    import torch
    from safetensors.torch import save_file
    model = ref.SingleStreamDiT(ref.SingleMMDiTConfig(**TINY)).eval().float()
    g = torch.Generator().manual_seed(1)
    sd = {k: torch.randn(v.shape, generator=g) * 0.02
          for k, v in model.state_dict().items()}
    model.load_state_dict(sd)
    d = tmp_path_factory.mktemp("ckpt")
    save_file(sd, str(d / "base.safetensors"))
    save_file({k: v.bfloat16() for k, v in sd.items()}, str(d / "base_bf16.safetensors"))
    return sd, d / "base.safetensors", d / "base_bf16.safetensors", model


@pytest.fixture(scope="session")
def text_encoder(tmp_path_factory):
    return make_tiny_text_encoder_dir(tmp_path_factory.mktemp("te") / "te")


@pytest.fixture(scope="session")
def bucket_cache(tmp_path_factory):
    """20 samples in two buckets (12 and 16 image tokens), two shards."""
    import numpy as np
    d = tmp_path_factory.mktemp("cache") / "data"
    ids = np.random.default_rng(7).choice(10_000, size=20, replace=False)
    rows = [{"id": int(i), "bucket_w": (48, 64)[k % 2], "bucket_h": 64,
             "artist_name": "" if k % 5 == 0 else f"artist {k % 3}",
             "tar": f"data/train-{k % 2:04d}.tar"} for k, i in enumerate(ids)]
    make_fake_latent_cache(d, rows, lat_channels=TINY["channels"])
    return d


@pytest.fixture(scope="session")
def mesh():
    from krea2_jax.checkpoint import make_mesh
    return make_mesh()
