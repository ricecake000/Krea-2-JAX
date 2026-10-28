"""scripts/vae_fast.py 2D single-frame encode == diffusers AutoencoderKLQwenImage
on a tiny random VAE with the real structure (CPU fp32)."""

import pytest

diffusers = pytest.importorskip("diffusers")
torch = pytest.importorskip("torch")


def tiny_encoder(seed=0):
    from cache_dataset import VaeEncoder
    torch.manual_seed(seed)
    vae = diffusers.AutoencoderKLQwenImage(base_dim=8, z_dim=16).eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("gamma"):
                p.copy_(1 + 0.3 * torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    enc = VaeEncoder.__new__(VaeEncoder)
    enc.vae, enc.device, enc.dtype = vae, "cpu", torch.float32
    enc.mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    enc.std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    return enc


@pytest.mark.parametrize("h,w", [(96, 64), (48, 112)])
@pytest.mark.parametrize("b", [1, 3])
def test_2d_posterior_matches_reference(h, w, b):
    import vae_fast
    enc = tiny_encoder()
    imgs = torch.rand(b, 3, h, w, generator=torch.Generator().manual_seed(h * 1000 + w + b)) * 2 - 1
    with torch.no_grad():
        m_ref, s_ref = vae_fast.make_encoder(enc, "ref").dist(imgs)
        m, s = vae_fast.make_encoder(enc, "2d").dist(imgs)
    assert m.shape == m_ref.shape == (b, 16, h // 8, w // 8)
    for a, r in ((m, m_ref), (s, s_ref)):
        assert ((a - r).abs().max() / r.abs().max()).item() < 1e-5
