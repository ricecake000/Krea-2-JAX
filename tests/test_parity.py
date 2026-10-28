"""Forward and gradient parity: JAX DiT vs the official PyTorch reference
(github.com/krea-ai/krea-2 mmdit.py, downloaded or $KREA2_REF_DIR) on the
tiny model. Import-safe: scripts/parity_full.py reuses get_reference()."""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # reference uses @torch.compile
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

REF_URL = "https://raw.githubusercontent.com/krea-ai/krea-2/main/mmdit.py"


def get_reference():
    ref_dir = os.environ.get("KREA2_REF_DIR")
    if ref_dir and (Path(ref_dir) / "mmdit.py").exists():
        path = Path(ref_dir)
    else:
        path = Path(tempfile.gettempdir()) / "krea2_ref"
        path.mkdir(exist_ok=True)
        if not (path / "mmdit.py").exists():
            print(f"downloading reference mmdit.py from {REF_URL}")
            urllib.request.urlretrieve(REF_URL, path / "mmdit.py")
    sys.path.insert(0, str(path))
    import mmdit

    return mmdit


def patch_reference_for_cpu(mmdit, torch):
    """Replace the cuDNN-only SDPA with an exact math version that (like the
    fused CUDA kernels) returns zeros for fully-masked query rows."""
    import torch.nn.functional as F
    from einops import rearrange

    def attention(q, k, v, mask=None, scale=None, gqa=False):
        if gqa:
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        d = q.shape[-1]
        s = scale if scale is not None else 1.0 / d**0.5
        logits = (q.float() @ k.float().transpose(-2, -1)) * s
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logits, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)  # fully-masked rows -> zeros
        x = (w @ v.float()).to(v.dtype)
        return rearrange(x, "B H L D -> B L (H D)")

    mmdit.attention = attention
    mmdit.Attention.forward.__globals__["attention"] = attention


# --- tests (pytest is not imported: this module also runs on GPU nodes) -----
_CASE = {}


def pytest_generate_tests(metafunc):
    if "remat" in metafunc.fixturenames:
        metafunc.parametrize("remat", [False, "dots", True])


def rel_err(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def _case(ref, tiny_ckpt):
    """Shared inputs: non-square 8x12 token grid, padded text in both samples."""
    if _CASE:
        return _CASE
    import jax.numpy as jnp
    import torch
    from helpers import TINY
    from krea2_jax.checkpoint import load_krea2_params
    from krea2_jax.model import build_positions_and_mask, patchify

    sd, path, _, _ = tiny_ckpt
    g = torch.Generator().manual_seed(2)
    B, ltxt, h, w, C = 2, 7, 8, 12, TINY["channels"]
    lat = torch.randn(B, C, 2 * h, 2 * w, generator=g)
    txtmask = torch.ones(B, ltxt, dtype=torch.bool)
    txtmask[0, 5:] = txtmask[1, 6:] = False
    imgids = torch.zeros(h, w, 3)
    imgids[..., 1] = torch.arange(h)[:, None]
    imgids[..., 2] = torch.arange(w)[None, :]
    targs = dict(
        img=lat.reshape(B, C, h, 2, w, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, -1),
        context=torch.randn(B, ltxt, TINY["txtlayers"], TINY["txtdim"], generator=g),
        t=torch.tensor([0.3, 0.8]),
        pos=torch.cat([torch.zeros(B, ltxt, 3), imgids.reshape(1, -1, 3).expand(B, -1, -1)], 1),
        mask=torch.cat([txtmask, torch.ones(B, h * w, dtype=torch.bool)], 1))
    jpos, jmask = build_positions_and_mask(jnp.asarray(txtmask.numpy()), h, w)
    _CASE.update(
        targs=targs, lat=lat, txtmask=txtmask, h=h, w=w, jpos=jpos, jmask=jmask,
        target=torch.randn(B, h * w, 4 * C, generator=g),
        params=load_krea2_params(str(path), mesh=None, dtype=jnp.float32),
        jargs=(patchify(jnp.asarray(lat.numpy()), 2), jnp.asarray(targs["context"].numpy()),
               jnp.asarray(targs["t"].numpy()), jpos, jmask))
    return _CASE


def _torch_model(ref, sd):
    from helpers import TINY
    model = ref.SingleStreamDiT(ref.SingleMMDiTConfig(**TINY)).eval().float()
    model.load_state_dict(sd)
    return model


def _jax_forward(c, **kw):
    from helpers import TINY
    from krea2_jax.config import Krea2Config
    from krea2_jax.model import dit_forward
    return dit_forward(c["params"], Krea2Config(**TINY), *c["jargs"], remat=False, **kw)


def test_forward_parity(ref, tiny_ckpt):
    """Layout helpers exact, forward and LoRA (zero-init, merged) match the reference."""
    import jax
    import torch
    from helpers import TINY
    from krea2_jax.checkpoint import init_lora

    c, sd = _case(ref, tiny_ckpt), tiny_ckpt[0]
    np.testing.assert_array_equal(np.asarray(c["jargs"][0]), c["targs"]["img"].numpy())
    np.testing.assert_array_equal(np.asarray(c["jpos"]), c["targs"]["pos"].numpy())
    np.testing.assert_array_equal(np.asarray(c["jmask"]), c["targs"]["mask"].numpy())
    with torch.no_grad():
        want = _torch_model(ref, sd)(**c["targs"]).numpy()
    out = np.asarray(_jax_forward(c, qchunk=64))
    assert rel_err(out, want) < 2e-4

    lora = init_lora(c["params"], jax.random.PRNGKey(0), rank=4, targets="attn+mlp")
    np.testing.assert_allclose(np.asarray(_jax_forward(c, lora=lora, lora_scale=2.0)),
                               out, atol=1e-6)
    lora = jax.tree.map(lambda a: jax.random.normal(jax.random.PRNGKey(2), a.shape) * 0.02, lora)
    merged = dict(sd)
    for group, targets in lora["blocks"].items():
        for name, ab in targets.items():
            a, b = np.asarray(ab["lora_A"]), np.asarray(ab["lora_B"])
            for i in range(TINY["layers"]):
                key = f"blocks.{i}.{group}.{name}.weight"
                merged[key] = merged[key] + 2.0 * torch.from_numpy(b[i] @ a[i])
    with torch.no_grad():
        want_lora = _torch_model(ref, merged)(**c["targs"]).numpy()
    assert rel_err(np.asarray(_jax_forward(c, lora=lora, lora_scale=2.0)), want_lora) < 2e-4


def _torch_grads(ref, tiny_ckpt, c):
    if "tgrads" not in c:
        model = _torch_model(ref, tiny_ckpt[0])
        loss = ((model(**c["targs"]) - c["target"]) ** 2).mean()
        loss.backward()
        c["tloss"] = loss.item()
        c["tgrads"] = {k: p.grad.numpy() for k, p in model.named_parameters()
                       if p.grad is not None}
    return c["tloss"], c["tgrads"]


def test_grad_parity(ref, tiny_ckpt, remat):
    """dLoss/dW for every base weight == torch autograd, with and without remat."""
    import jax
    import jax.numpy as jnp
    from helpers import TINY
    from krea2_jax.checkpoint import _official_entries
    from krea2_jax.config import Krea2Config
    from krea2_jax.model import dit_forward

    c = _case(ref, tiny_ckpt)
    tloss, tgrads = _torch_grads(ref, tiny_ckpt, c)
    target = jnp.asarray(c["target"].numpy())

    def loss_fn(params, *args):
        v = dit_forward(params, Krea2Config(**TINY), *args, remat=remat, qchunk=64)
        return ((v - target) ** 2).mean()

    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(c["params"], *c["jargs"])
    assert abs(float(loss) - tloss) < 1e-6
    entries, leaves = _official_entries(grads)
    got = {k: np.asarray(leaves[li])[i] if i is not None else np.asarray(leaves[li])
           for k, li, i, _ in entries}
    assert set(tgrads) <= set(got)
    worst = max((rel_err(got[k], g), k) for k, g in tgrads.items())
    assert worst[0] < 2e-4, worst
