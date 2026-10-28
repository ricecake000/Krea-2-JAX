"""krea2_jax/qwen3_text.py pure-JAX parts: config, loader key mapping, padding
invariance, text_encoder_identity (on the tiny HF-layout text tower)."""

import dataclasses
import json
import shutil

import numpy as np
from safetensors.numpy import load_file, save_file

from helpers import BF16
from krea2_jax.qwen3_text import (Qwen3TextConfig, abstract_qwen3_text_params, from_hf_config,
                                  load_qwen3_text_params, load_tokenizer, param_count,
                                  text_encoder_identity, text_states)
from krea2_jax.text_template import CaptionTemplate

REAL = {"attention_bias": False, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2560,
        "intermediate_size": 9728, "num_attention_heads": 32, "num_hidden_layers": 36,
        "num_key_value_heads": 8, "rms_norm_eps": 1e-06, "rope_theta": 5000000,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20],
                         "rope_type": "default"}, "vocab_size": 151936}
PRE = "model.language_model."
CAPS = ["a girl", "a girl with long silver hair and red eyes, wearing a black dress",
        "two cats sitting on a wooden table next to a window, sunlight, drawn by someone"]


def test_real_config_and_param_count():
    cfg = from_hf_config({"text_config": REAL})
    assert cfg == Qwen3TextConfig() and cfg.n_run_layers == 35 and not cfg.needs_final_norm
    tc5 = dict(REAL)
    rp = dict(tc5.pop("rope_scaling"), rope_theta=tc5.pop("rope_theta"))
    assert from_hf_config(dict(tc5, rope_parameters=rp)) == cfg
    n_run = param_count(abstract_qwen3_text_params(cfg))
    full = param_count(abstract_qwen3_text_params(Qwen3TextConfig(select_layers=(36,))))
    assert 3.9e9 < n_run < 4.0e9 and 4.0e9 < full < 4.1e9, (n_run, full)


def test_loader_key_mapping(text_encoder):
    import jax.numpy as jnp
    d, cfg = text_encoder
    raw = load_file(str(d / "model.safetensors"))
    p = load_qwen3_text_params(d, dtype=jnp.float32, cfg=cfg)
    assert cfg.needs_final_norm and p["layers"]["q_proj"].shape[0] == 2
    np.testing.assert_array_equal(p["embed_tokens"], raw[PRE + "embed_tokens.weight"])
    np.testing.assert_array_equal(p["norm"], raw[PRE + "norm.weight"])
    np.testing.assert_array_equal(p["layers"]["down_proj"][1], raw[PRE + "layers.1.mlp.down_proj.weight"])
    np.testing.assert_array_equal(p["layers"]["q_norm"][0], raw[PRE + "layers.0.self_attn.q_norm.weight"])
    p1 = load_qwen3_text_params(d, dtype=jnp.bfloat16, cfg=dataclasses.replace(cfg, select_layers=(0, 1)))
    assert "norm" not in p1 and p1["layers"]["q_proj"].shape == (1, 64, 64)
    np.testing.assert_array_equal(np.asarray(p1["layers"]["k_proj"][0]),
                                  raw[PRE + "layers.0.self_attn.k_proj.weight"].astype(BF16))


def test_padding_invariance(text_encoder):
    import jax
    import jax.numpy as jnp
    d, cfg = text_encoder
    params, tok = load_qwen3_text_params(d, dtype=jnp.float32, cfg=cfg), load_tokenizer(d)
    f, out = jax.jit(text_states, static_argnums=1), {}
    for mode in ("contiguous", "padded"):
        for rows in (8, 32):
            t = CaptionTemplate(tok, text_rows=rows)
            ids, tl = t.encode_captions(CAPS)
            s, m = (np.asarray(x) for x in f(params, dataclasses.replace(cfg, position_mode=mode), ids, tl))
            np.testing.assert_array_equal(m, t.valid_mask(tl))
            assert np.all(s[~m] == 0)
            out[mode, rows] = s, tl
    (s8, tl8), (s32, tl32) = out["contiguous", 8], out["contiguous", 32]
    assert tl8[0] == tl32[0] <= 8
    for b in range(len(CAPS)):
        k = int(tl8[0]) if b == 0 else int(min(tl8[b], tl32[b])) - cfg.suffix_len
        np.testing.assert_allclose(s8[b, :k], s32[b, :k], rtol=0, atol=1e-5)
    for rows in (8, 32):
        (sc, tl), (sp, _) = out["contiguous", rows], out["padded", rows]
        for b in range(len(CAPS)):
            nc = int(tl[b]) - cfg.suffix_len
            np.testing.assert_allclose(sp[b, :nc], sc[b, :nc], rtol=0, atol=1e-5)
            if tl[b] < rows:
                assert np.abs(sp[b, nc:tl[b], 1:] - sc[b, nc:tl[b], 1:]).max() > 1e-4


def test_text_encoder_identity(text_encoder, tmp_path):
    d, cfg = text_encoder
    ident = text_encoder_identity(d, cfg, stats=(stats := {}))
    assert stats == {"files": 1, "bytes": (d / "model.safetensors").stat().st_size}
    other = shutil.copytree(d, tmp_path / "elsewhere" / "other_name")
    assert text_encoder_identity(other, from_hf_config(other, select_layers=cfg.select_layers)) == ident

    def variant(name, edit):
        v = shutil.copytree(d, tmp_path / name)
        edit(v)
        return text_encoder_identity(v, cfg)

    def revalue(v):
        w = load_file(str(v / "model.safetensors"))
        w[PRE + "embed_tokens.weight"][3, 5] += 1.0
        save_file(w, str(v / "model.safetensors"), metadata={"format": "pt"})

    def retok(v):
        (v / "tokenizer.json").write_text(json.dumps(json.loads((v / "tokenizer.json").read_text()), indent=1))

    def vision(v):
        save_file({"model.visual.patch_embed.proj.weight": np.ones((4, 3), np.float32)},
                  str(v / "model-visual.safetensors"))

    changed = [variant("value", revalue), variant("tok", retok),
               text_encoder_identity(d, dataclasses.replace(cfg, select_layers=(0, 1))),
               text_encoder_identity(d, dataclasses.replace(cfg, position_mode="padded"))]
    assert ident not in changed and len(set(changed)) == len(changed)
    assert variant("vision", vision) == ident
