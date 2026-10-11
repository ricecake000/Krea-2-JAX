"""Activation sharding at the REAL model size, checked ahead of time.

1. `abstract_krea2_params` (no torch, no weights) reproduces the official
   module's state_dict keys/shapes (via the torch reference instantiated on
   the meta device) for the 12.9B config and for tiny configs, and has the
   same tree structure as `load_krea2_params`.
2. Lowers + compiles `value_and_grad` of the flow-matching loss for the full
   12.9B DiT (batch 64, 1024px, 512 text tokens, remat "dots", FSDP params)
   on 8 emulated CPU devices — abstract shapes only, nothing is allocated —
   and inspects the SPMD collectives XLA emitted. Without activation
   sharding constraints the partitioner moves full activations
   ([64, seq, features]-sized collectives); with `act_sharding=
   batch_sharding(mesh)` there must be none (only weight gathers / gradient
   reductions remain).

    python tests/test_act_sharding.py
"""

import os, sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "scripts"))

import collections
import tempfile
import time

import numpy as np
import torch

from test_parity import get_reference
from aot_analysis import parse_collectives

mmdit = get_reference()

import jax
import jax.numpy as jnp
from safetensors.numpy import save_file

from krea2_jax.checkpoint import (
    abstract_krea2_params, batch_sharding, load_krea2_params, make_mesh,
    official_key_shapes, param_shardings, replicated_sharding,
)
from krea2_jax.config import KREA2_CONFIG, Krea2Config
from krea2_jax.model import build_positions_and_mask, dit_forward, patchify
from krea2_jax.sampling import shift_time

assert len(jax.devices()) == 8, jax.devices()

# ---------------------------------------------------------------------------
# 0. HLO collective parser (scripts/aot_analysis.parse_collectives) on
#    hand-written lines: sync results are flattened (nested tuples, index
#    comments), async pairs are counted once at *-done (the *-start result
#    holds the inputs too), token / context scalars are dropped.
# ---------------------------------------------------------------------------

_HLO_UNIT = """
  %all-gather.1 = bf16[64,4608,6144]{2,1,0} all-gather(bf16[8,4608,6144]{2,1,0} %p.1), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="jit(f)/all_gather"}
  %all-reduce.2 = (f32[6144,6144]{1,0}, f32[128]{0}, f32[4]{0}, f32[4]{0}, f32[4]{0}, /*index=5*/f32[512,3]{1,0}) all-reduce(f32[6144,6144]{1,0} %a0, f32[128]{0} %a1, f32[4]{0} %a2, f32[4]{0} %a3, f32[4]{0} %a4, /*index=5*/f32[512,3]{1,0} %a5), channel_id=2, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=%add.3
  %all-gather-start = ((bf16[8,4608,128]{2,1,0}, bf16[8,4608,6144]{2,1,0}), (bf16[64,4608,128]{2,1,0}, bf16[64,4608,6144]{2,1,0})) all-gather-start(bf16[8,4608,128]{2,1,0} %x, bf16[8,4608,6144]{2,1,0} %y), channel_id=3, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true
  %all-gather-done = (bf16[64,4608,128]{2,1,0}, bf16[64,4608,6144]{2,1,0}) all-gather-done(((bf16[8,4608,128]{2,1,0}, bf16[8,4608,6144]{2,1,0}), (bf16[64,4608,128]{2,1,0}, bf16[64,4608,6144]{2,1,0})) %all-gather-start)
  %reduce-scatter.4 = f32[768,6144]{1,0} reduce-scatter(f32[6144,6144]{1,0} %g), channel_id=4, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, to_apply=%add.3
  %collective-permute-start.5 = (bf16[8,16]{1,0}, bf16[8,16]{1,0}, u32[], u32[]) collective-permute-start(bf16[8,16]{1,0} %z), channel_id=5, source_target_pairs={{0,1},{1,0}}
  ROOT %collective-permute-done.5 = bf16[8,16]{1,0} collective-permute-done((bf16[8,16]{1,0}, bf16[8,16]{1,0}, u32[], u32[]) %collective-permute-start.5)
  %fusion.6 = bf16[64,4608,6144]{2,1,0} fusion(bf16[64,4608,6144]{2,1,0} %all-gather-done), kind=kLoop, calls=%fused_computation, metadata={op_name="all-gather(x)"}
"""
_got = parse_collectives(_HLO_UNIT)
_want = [
    ("all-gather", "bf16", (64, 4608, 6144)),
    ("all-reduce", "f32", (6144, 6144)),
    ("all-reduce", "f32", (128,)),
    ("all-reduce", "f32", (4,)),
    ("all-reduce", "f32", (4,)),
    ("all-reduce", "f32", (4,)),
    ("all-reduce", "f32", (512, 3)),
    ("all-gather", "bf16", (64, 4608, 128)),
    ("all-gather", "bf16", (64, 4608, 6144)),
    ("reduce-scatter", "f32", (768, 6144)),
    ("collective-permute", "bf16", (8, 16)),
]
assert _got == _want, _got
print("parse_collectives on hand-written HLO (sync/tuple/async/variadic): ok")

# ---------------------------------------------------------------------------
# 1. abstract params == official module (torch meta device)
# ---------------------------------------------------------------------------

TINY = dict(features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
            layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3)
REF_FIELDS = ("features", "tdim", "txtdim", "heads", "multiplier", "layers",
              "patch", "channels", "bias", "theta", "kvheads", "txtlayers",
              "txtheads", "txtkvheads")


def meta_state_shapes(cfg):
    with torch.device("meta"):
        ref = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(
            **{f: getattr(cfg, f) for f in REF_FIELDS}))
    return {k: tuple(v.shape) for k, v in ref.state_dict().items()}


for name, cfg in (("KREA2", KREA2_CONFIG), ("TINY", Krea2Config(**TINY)),
                  ("TINY+bias", Krea2Config(**TINY, bias=True))):
    want = meta_state_shapes(cfg)
    abstract = abstract_krea2_params(cfg)
    got = official_key_shapes(abstract)
    assert set(got) == set(want), (
        name, sorted(set(got) - set(want)), sorted(set(want) - set(got)))
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    assert not bad, (name, bad)
    assert all(a.dtype == jnp.float32 for a in jax.tree.leaves(abstract))
    n = sum(int(np.prod(s)) for s in want.values())
    print(f"abstract params == torch meta state_dict ({name}): "
          f"{len(want)} tensors, {n / 1e9:.3f}B params")

# same tree (nesting, stacking, shapes) as the real loader
cfg_t = Krea2Config(**TINY)
sd = {k: np.zeros(s, np.float32) for k, s in meta_state_shapes(cfg_t).items()}
with tempfile.TemporaryDirectory() as d:
    save_file(sd, os.path.join(d, "tiny.safetensors"))
    loaded = load_krea2_params(os.path.join(d, "tiny.safetensors"),
                               dtype=jnp.float32)
abstract_t = abstract_krea2_params(cfg_t, dtype=jnp.bfloat16)
assert jax.tree.structure(loaded) == jax.tree.structure(abstract_t)
for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(abstract_t)):
    assert a.shape == b.shape and b.dtype == jnp.bfloat16, (a.shape, b.shape)
print("abstract tree structure == load_krea2_params: ok")

# ---------------------------------------------------------------------------
# 2. AOT compile at the real size: collectives with / without act_sharding
# ---------------------------------------------------------------------------

cfg = KREA2_CONFIG
B, RES, TXT_LEN, REMAT, QCHUNK = 64, 1024, 512, "dots", 512
mesh = make_mesh()
abstract = abstract_krea2_params(cfg)
pshard = param_shardings(abstract, mesh)
params = jax.tree.map(
    lambda s, sh: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sh),
    abstract, pshard,
)
lat = RES // cfg.vae_compression
h_tok = w_tok = lat // cfg.patch
mu = cfg.mu_for_seq_len(h_tok * w_tok)
ds = batch_sharding(mesh)
b_lat = jax.ShapeDtypeStruct((B, cfg.channels, lat, lat), jnp.bfloat16, sharding=ds)
b_txt = jax.ShapeDtypeStruct((B, TXT_LEN, cfg.txtlayers, cfg.txtdim),
                             jnp.bfloat16, sharding=ds)
b_mask = jax.ShapeDtypeStruct((B, TXT_LEN), jnp.bool_, sharding=ds)
key = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=replicated_sharding(mesh))


def make_loss_and_grad(act_sharding):
    def loss_and_grad(params, batch_lat, batch_txt, batch_mask, key):
        k_t, k_noise = jax.random.split(key)
        x0 = patchify(batch_lat, cfg.patch).astype(jnp.float32)
        t = shift_time(jax.nn.sigmoid(jax.random.normal(k_t, (B,))), mu)
        noise = jax.random.normal(k_noise, x0.shape, dtype=jnp.float32)
        tt = t[:, None, None]
        xt = ((1.0 - tt) * x0 + tt * noise).astype(jnp.bfloat16)
        pos, mask = build_positions_and_mask(batch_mask, h_tok, w_tok)

        def loss_fn(params):
            p16 = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
            v = dit_forward(p16, cfg, xt, batch_txt, t, pos, mask, lora=None,
                            qchunk=QCHUNK, remat=REMAT, attn_impl="ref",
                            act_sharding=act_sharding)
            return jnp.mean((v.astype(jnp.float32) - (noise - x0)) ** 2)

        return jax.value_and_grad(loss_fn)(params)

    return jax.jit(loss_and_grad,
                   out_shardings=(replicated_sharding(mesh), pshard))


def collectives(hlo_text):
    """[(kind, dtype, shape)] for every collective result in the HLO (the
    exact parser scripts/aot_analysis.py uses; unit-tested in section 0)."""
    return parse_collectives(hlo_text)


def activation_collectives(colls):
    # rule shared with scripts/aot_analysis.py: batch-leading and big
    return [c for c in colls
            if c[2] and c[2][0] == B and int(np.prod(c[2])) >= 1e7]


results = {}
for label, act in (("unconstrained", None), ("act_sharding", ds)):
    t0 = time.time()
    lowered = make_loss_and_grad(act).lower(params, b_lat, b_txt, b_mask, key)
    t1 = time.time()
    compiled = lowered.compile()
    t2 = time.time()
    colls = collectives(compiled.as_text())
    acts = activation_collectives(colls)
    results[label] = (colls, acts)
    kinds = collections.Counter(c[0] for c in colls)
    print(f"[{label}] lower {t1 - t0:.1f}s compile {t2 - t1:.1f}s; "
          f"collectives {dict(kinds)}; activation collectives: {len(acts)}")
    for (kind, dt, shape), n in collections.Counter(acts).most_common(6):
        print(f"    {n:3d} x {kind} {dt}{list(shape)}")

colls_u, acts_u = results["unconstrained"]
colls_c, acts_c = results["act_sharding"]
assert colls_u and colls_c, "no collectives parsed (HLO format changed?)"
# teeth: the unconstrained program must show the problem the constraints fix
assert acts_u, "expected activation collectives without act_sharding"
assert not acts_c, f"activation collectives with act_sharding: {acts_c[:10]}"
# stronger: with constraints every sizable collective is a weight (gather)
# or a weight gradient (reduction) — shaped like one per-layer tensor
weight_shapes = set(official_key_shapes(abstract).values())
non_weight = [c for c in colls_c
              if int(np.prod(c[2] or (1,))) >= 1e6 and c[2] not in weight_shapes]
assert not non_weight, f"non-weight collectives with act_sharding: {non_weight[:10]}"
assert any(int(np.prod(c[2] or (1,))) >= 1e6 and c[2] not in weight_shapes
           for c in colls_u)
print("act_sharding: every collective >= 1e6 elements is weight-shaped")

print("ALL ACT SHARDING TESTS PASSED")
