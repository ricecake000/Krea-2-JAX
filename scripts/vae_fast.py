"""Faster single-image encode path for the Qwen-Image VAE
(diffusers AutoencoderKLQwenImage, checked against diffusers 0.40.0).

`vae.encode(x)` with x (B, 3, 1, H, W) runs the video encoder on a
one-frame clip. For T=1 the reference path (`_encode`: clear_cache, one
chunk `x[:, :, :1]`, then quant_conv) does:
  * every QwenImageCausalConv3d with k_t=3 F.pads 2 zero frames in front
    (its feat_cache entry is still None on the first chunk) and convolves
    all 3 taps; only the last tap meets data, so the output equals a 2D
    conv with weight[:, :, -1] (+ bias, same spatial padding/stride).
    k_t=1 convs (residual shortcuts, quant_conv) are weight[:, :, 0].
  * every conv input is cloned into the feat_cache (unused for one frame).
  * downsample3d Resamples only store `x.clone()` in the cache on the first
    chunk: their time_conv never runs, so they are plain downsample2d.
  * the mid block's attention is single-head over H*W tokens, head dim C.
  * RMS norm: F.normalize over channels (fp32 for bf16/fp16 inputs) * sqrt(C)
    * gamma (+ bias, 0.0 here); activations nn.SiLU; dropout is a no-op in
    eval.
  * DiagonalGaussianDistribution: mean, logvar = moments.chunk(2, dim=1),
    logvar.clamp(-30, 20), std = exp(0.5 * logvar), sample = mean + std *
    torch.randn(mean.shape) (same RNG draw as diffusers' randn_tensor with
    no generator, so a seeded 2D sample equals the reference sample).

Encoder2d is a parallel 2D module tree built from the loaded VAE: 2D conv
weights are contiguous copies of the k_t slice (one third of a k_t=3 conv),
everything else (Conv2d modules, RMS gammas) is shared with the VAE. It is
tied to the VAE's device/dtype at build time: rebuild after
VaeEncoder.to_dtype().

make_encoder(vae_encoder, impl) -> callable with the same contract as
cache_latents.vae_encode_batch (imgs (N, 3, H, W) in [-1, 1] -> normalized
latents (N, 16, H/8, W/8)); impl "ref" (diffusers), "2d" or "2d-compile"
(torch.compile(dynamic=True) of the 2D encoder; posterior sampling stays
eager). The 2d variants turn on torch.backends.cudnn.benchmark.
Parity: tests/test_vae_fast.py (tiny random VAE, CPU fp32).
"""

import torch
import torch.nn.functional as F
from torch import nn

IMPLS = ("ref", "2d", "2d-compile")


def _cls(m):
    return type(m).__name__


def _conv2d_from_causal3d(conv):
    """QwenImageCausalConv3d evaluated on one frame with no cache -> the
    equivalent nn.Conv2d (last temporal tap)."""
    kt, kh, kw = conv.kernel_size
    # F.pad order (w_l, w_r, h_t, h_b, t_front, t_back); t_front = 2 * p_t
    front, ph, pw = conv._padding[4], conv._padding[2], conv._padding[0]
    # with T=1 (no cache) the output has one frame iff front + 1 - k_t + 1
    # == 1 at temporal stride 1; the data then meets only the last tap
    if conv.stride[0] != 1 or front != kt - 1 or conv._padding[5] != 0:
        raise ValueError(f"unsupported causal conv for T=1: k={conv.kernel_size}"
                         f" stride={conv.stride} padding={conv._padding}")
    if conv._padding[1] != pw or conv._padding[3] != ph:
        raise ValueError(f"asymmetric spatial padding {conv._padding}")
    out = nn.Conv2d(conv.in_channels, conv.out_channels, (kh, kw),
                    stride=conv.stride[1:], padding=(ph, pw),
                    dilation=conv.dilation[1:], groups=conv.groups,
                    bias=conv.bias is not None,
                    device=conv.weight.device, dtype=conv.weight.dtype)
    out.weight = nn.Parameter(conv.weight.detach()[:, :, -1].contiguous(),
                              requires_grad=False)
    if conv.bias is not None:
        out.bias = nn.Parameter(conv.bias.detach(), requires_grad=False)
    return out


def _check_silu(act):
    if not isinstance(act, nn.SiLU):
        raise ValueError(f"unsupported nonlinearity {_cls(act)} (expected SiLU)")


class RMSNorm2d(nn.Module):
    """QwenImageRMS_norm (channel_first) on (B, C, H, W); shares gamma."""

    def __init__(self, ref):
        super().__init__()
        if not ref.channel_first:
            raise ValueError("channel_first=False RMS norm is not used here")
        self.scale = ref.scale
        self.register_buffer("gamma", ref.gamma.detach().view(-1, 1, 1),
                             persistent=False)
        bias = ref.bias
        if isinstance(bias, torch.Tensor):
            self.register_buffer("bias", bias.detach().view(-1, 1, 1),
                                 persistent=False)
        else:
            self.bias = float(bias)

    def forward(self, x):
        up = x.dtype in (torch.float16, torch.bfloat16)
        normalized = F.normalize(x.float() if up else x, dim=1).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class ResBlock2d(nn.Module):
    def __init__(self, ref):
        super().__init__()
        _check_silu(ref.nonlinearity)
        self.norm1 = RMSNorm2d(ref.norm1)
        self.conv1 = _conv2d_from_causal3d(ref.conv1)
        self.norm2 = RMSNorm2d(ref.norm2)
        self.conv2 = _conv2d_from_causal3d(ref.conv2)
        self.shortcut = (nn.Identity() if isinstance(ref.conv_shortcut,
                                                     nn.Identity)
                         else _conv2d_from_causal3d(ref.conv_shortcut))

    def forward(self, x):
        h = self.shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + h


class Downsample2d(nn.Module):
    """QwenImageResample downsample2d / downsample3d (first chunk: the
    time_conv is skipped by the reference): ZeroPad2d((0, 1, 0, 1)) + the
    shared stride-2 Conv2d."""

    def __init__(self, ref):
        super().__init__()
        if ref.mode not in ("downsample2d", "downsample3d"):
            raise ValueError(f"unsupported Resample mode {ref.mode!r}")
        pad, conv = ref.resample
        if _cls(pad) != "ZeroPad2d" or tuple(pad.padding) != (0, 1, 0, 1):
            raise ValueError(f"unexpected resample padding {pad}")
        self.conv = conv

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Attention2d(nn.Module):
    """QwenImageAttentionBlock with b*t = B: single head over H*W tokens,
    head dim C; shares to_qkv / proj."""

    def __init__(self, ref):
        super().__init__()
        self.norm = RMSNorm2d(ref.norm)
        self.to_qkv = ref.to_qkv
        self.proj = ref.proj

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x))
        qkv = qkv.reshape(b, 1, c * 3, h * w).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        return self.proj(x) + identity


class Encoder2d(nn.Module):
    """vae.encoder + vae.quant_conv for one frame: (B, 3, H, W) -> moments
    (B, 2 * z_dim, H/8, W/8) (before the posterior's chunk/clamp)."""

    def __init__(self, vae):
        super().__init__()
        enc = vae.encoder
        _check_silu(enc.nonlinearity)
        self.conv_in = _conv2d_from_causal3d(enc.conv_in)
        layers = []
        for layer in enc.down_blocks:
            name = _cls(layer)
            if name == "QwenImageResidualBlock":
                layers.append(ResBlock2d(layer))
            elif name == "QwenImageResample":
                layers.append(Downsample2d(layer))
            else:
                # e.g. QwenImageAttentionBlock (attn_scales): the reference
                # calls it as layer(x, feat_cache, feat_idx), which its
                # forward(x) does not accept, so it cannot encode either
                raise ValueError(f"unsupported encoder down block {name}")
        self.down = nn.ModuleList(layers)
        mid = enc.mid_block
        self.mid_res0 = ResBlock2d(mid.resnets[0])
        self.mid = nn.ModuleList()
        for attn, res in zip(mid.attentions, mid.resnets[1:]):
            self.mid.append(Attention2d(attn) if attn is not None
                            else nn.Identity())
            self.mid.append(ResBlock2d(res))
        self.norm_out = RMSNorm2d(enc.norm_out)
        self.conv_out = _conv2d_from_causal3d(enc.conv_out)
        self.quant_conv = _conv2d_from_causal3d(vae.quant_conv)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.down:
            x = layer(x)
        x = self.mid_res0(x)
        for layer in self.mid:
            x = layer(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return self.quant_conv(x)


def posterior(moments):
    """DiagonalGaussianDistribution's (mean, std) of moments (B, 2z, ...)."""
    mean, logvar = torch.chunk(moments, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    return mean, torch.exp(0.5 * logvar)


class FastEncoder:
    """Callable imgs (N, 3, H, W) in [-1, 1] -> normalized sampled latents
    (N, z, H/8, W/8), like cache_latents.vae_encode_batch; dist(imgs) ->
    unnormalized posterior (mean, std) (N, z, H/8, W/8) for parity checks."""

    def __init__(self, vae_encoder, impl="ref"):
        if impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}, got {impl!r}")
        self.vae, self.impl = vae_encoder, impl
        self.mean = vae_encoder.mean.view(1, -1, 1, 1)
        self.std = vae_encoder.std.view(1, -1, 1, 1)
        self.net = None
        if impl != "ref":
            torch.backends.cudnn.benchmark = True
            self.net = Encoder2d(vae_encoder.vae).eval()
            self.forward = self.net
            if impl == "2d-compile":
                self.forward = torch.compile(self.net, dynamic=True)

    @torch.no_grad()
    def dist(self, imgs):
        x = imgs.to(self.vae.device, self.vae.dtype)
        if self.net is None:
            d = self.vae.vae.encode(x.unsqueeze(2)).latent_dist
            return d.mean[:, :, 0], d.std[:, :, 0]
        return posterior(self.forward(x))

    @torch.no_grad()
    def __call__(self, imgs):
        if self.net is None:   # == cache_latents.vae_encode_batch (tested)
            x = imgs.to(self.vae.device, self.vae.dtype).unsqueeze(2)
            z = self.vae.vae.encode(x).latent_dist.sample()
            return ((z - self.vae.mean) / self.vae.std)[:, :, 0]
        mean, std = self.dist(imgs)
        z = mean + std * torch.randn(mean.shape, device=mean.device,
                                     dtype=mean.dtype)
        return (z - self.mean) / self.std


def make_encoder(vae_encoder, impl):
    """cache_dataset.VaeEncoder + impl in IMPLS -> FastEncoder."""
    return FastEncoder(vae_encoder, impl)
