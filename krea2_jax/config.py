"""Model configuration for the Krea 2 SingleStream DiT.

Numbers mirror `single_mmdit_large_wide` from the official inference repo
(github.com/krea-ai/krea-2). Both the RAW and Turbo checkpoints share this
architecture.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Krea2Config:
    features: int = 6144      # transformer width
    tdim: int = 256           # sinusoidal timestep-embedding dim
    txtdim: int = 2560        # Qwen3-VL-4B hidden dim
    heads: int = 48           # query heads
    kvheads: int = 12         # KV heads (GQA)
    multiplier: int = 4       # SwiGLU expansion
    layers: int = 28          # SingleStream blocks
    patch: int = 2            # latent patch size
    channels: int = 16        # VAE latent channels
    bias: bool = False        # attention / MLP linears have no bias
    theta: float = 1e3        # RoPE theta
    txtlayers: int = 12       # number of text-encoder hidden states fused
    txtheads: int = 20        # text-fusion attention heads
    txtkvheads: int = 20      # text-fusion KV heads (no GQA)

    # Text conditioning (matches the official Qwen3VLConditioner).
    text_encoder_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    text_max_length: int = 512
    # Encoder hidden-state layers stacked into `txtlayers` inputs.
    text_select_layers: tuple = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

    # VAE (Qwen-Image VAE, f8 / 16 channels).
    vae_id: str = "Qwen/Qwen-Image"
    vae_compression: int = 8

    # Resolution-aware timestep-shift endpoints (mu interpolation), from the
    # official sampler: mu goes from y1 at (minres)^2 tokens to y2 at (maxres)^2.
    shift_y1: float = 0.5
    shift_y2: float = 1.15
    shift_minres: int = 256
    shift_maxres: int = 1280

    # The combined text+image sequence is padded to a multiple of this.
    seq_pad_multiple: int = 256

    @property
    def headdim(self) -> int:
        return self.features // self.heads

    @property
    def axdims(self) -> tuple:
        """Head-dim split across the 3 RoPE position axes (t, y, x)."""
        hd = self.headdim
        axes = (hd - 12 * (hd // 16), 6 * (hd // 16), 6 * (hd // 16))
        assert sum(axes) == hd and all(a % 2 == 0 for a in axes)
        return axes

    def mu_for_seq_len(self, img_seq_len: int) -> float:
        """Timestep-shift mu, linearly interpolated in image-token count."""
        x1 = (self.shift_minres // (self.vae_compression * self.patch)) ** 2
        x2 = (self.shift_maxres // (self.vae_compression * self.patch)) ** 2
        slope = (self.shift_y2 - self.shift_y1) / (x2 - x1)
        return slope * img_seq_len + (self.shift_y1 - slope * x1)


KREA2_CONFIG = Krea2Config()
