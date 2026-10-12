"""Tests for krea2_jax/qwen3_text.py (JAX Qwen3-VL text tower).

Pure-JAX parts run in the repo .venv (no transformers). The parity parts
(vs transformers and the production conditioner
scripts/cache_dataset.py::TextEncoder) need torch + transformers and are
skipped with a message when transformers is missing; run them with e.g.
/home/mj/krea2-ops/mi350x/venv-jax0110/bin/python tests/test_qwen3_text.py

The 8-device mesh case runs in-process when 8 devices are visible, otherwise
in a subprocess with XLA_FLAGS=--xla_force_host_platform_device_count=8.

Provides `make_tiny_tokenizer(dir)` and `make_tiny_text_encoder_dir(dir)`,
which other tests import (keep their signatures stable). Neither needs
torch or transformers.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from krea2_jax.text_template import PAD_TOKEN, PREFIX, SUFFIX, CaptionTemplate

TINY_CAPTIONS = [
    "",
    "a girl",
    "a girl with long silver hair and red eyes, wearing a black dress",
    "two cats sitting on a wooden table next to a window, sunlight, "
    "drawn by someone",
    "landscape painting of mountains at sunset with a river in the "
    "foreground and pine trees, soft clouds, warm colors, detailed brush "
    "strokes, a small cabin on the left side, smoke rising from the "
    "chimney, birds flying over the lake, reflections on the water",
]
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
# the Qwen2/Qwen3 pre-tokenizer regex (from the real tokenizer.json)
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
              r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# Qwen/Qwen3-VL-4B-Instruct config.json -> text_config (4.x layout)
REAL_TEXT_CONFIG = {
    "attention_bias": False, "head_dim": 128, "hidden_act": "silu",
    "hidden_size": 2560, "intermediate_size": 9728,
    "num_attention_heads": 32, "num_hidden_layers": 36,
    "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
    "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20],
                     "rope_type": "default"},
    "rope_theta": 5000000, "vocab_size": 151936,
}


# ---------------------------------------------------------------------------
# helpers (importable)
# ---------------------------------------------------------------------------


def make_tiny_tokenizer(out_dir, vocab_size=600, extra_corpus=()):
    """Train a tiny byte-level BPE (Qwen pre-tokenizer, NFC) on a few strings
    (plus `extra_corpus`) with the special tokens <|endoftext|> <|im_start|>
    <|im_end|>; writes tokenizer.json + tokenizer_config.json (loadable by
    transformers' AutoTokenizer, pad = <|endoftext|>) into out_dir. Returns
    the `tokenizers.Tokenizer`. Deterministic."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers
    from tokenizers import pre_tokenizers, trainers

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False)
    corpus = [PREFIX, SUFFIX] + TINY_CAPTIONS[:-1] * 3 + list(extra_corpus)
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(str(out_dir / "tokenizer.json"))
    (out_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": PAD_TOKEN, "eos_token": "<|im_end|>",
        "bos_token": None, "unk_token": None, "add_bos_token": False,
        "model_max_length": 4096, "padding_side": "right",
        "clean_up_tokenization_spaces": False,
    }))
    return Tokenizer.from_file(str(out_dir / "tokenizer.json"))


def make_tiny_text_encoder_dir(out_dir, select_layers=(0, 1, 2), seed=0,
                               position_mode="contiguous"):
    """Write a tiny HF-format Qwen3-VL checkpoint dir: config.json (full
    Qwen3VLForConditionalGeneration config, 4.x layout, tiny vision tower),
    tokenizer.json/tokenizer_config.json (make_tiny_tokenizer) and
    model.safetensors with random f32 TEXT weights under
    `model.language_model.*` (all layers + final norm; no vision weights,
    no lm_head: transformers initializes those itself, they are unused).
    Text tower: hidden 64, 2 layers, 4 q heads / 2 kv heads, head_dim 16,
    intermediate 96, vocab = tokenizer vocab. Needs no torch/transformers.
    Returns (Path(out_dir), Qwen3TextConfig) with the given select_layers
    (default (0, 1, 2): embeddings, layer 1, final-normed layer 2) and
    prefix/suffix lengths of the tiny tokenizer."""
    from safetensors.numpy import save_file

    from krea2_jax.qwen3_text import from_hf_config

    out_dir = Path(out_dir)
    tok = make_tiny_tokenizer(out_dir)
    V = tok.get_vocab_size()
    H, n_layers, nh, nkv, D, I = 64, 2, 4, 2, 16, 96
    text_config = {
        "model_type": "qwen3_vl_text", "attention_bias": False,
        "attention_dropout": 0.0, "head_dim": D, "hidden_act": "silu",
        "hidden_size": H, "intermediate_size": I,
        "initializer_range": 0.02, "max_position_embeddings": 4096,
        "num_attention_heads": nh, "num_hidden_layers": n_layers,
        "num_key_value_heads": nkv, "rms_norm_eps": 1e-6,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [4, 2, 2],
                         "rope_type": "default"},
        "rope_theta": 5000000, "tie_word_embeddings": True,
        "use_cache": False, "vocab_size": V, "dtype": "float32",
        "bos_token_id": tok.token_to_id("<|endoftext|>"),
        "eos_token_id": tok.token_to_id("<|im_end|>"),
        "pad_token_id": tok.token_to_id(PAD_TOKEN),
    }
    vision_config = {
        "model_type": "qwen3_vl", "depth": 1, "hidden_size": 32,
        "intermediate_size": 64, "num_heads": 2, "in_channels": 3,
        "patch_size": 4, "spatial_merge_size": 2, "temporal_patch_size": 2,
        "out_hidden_size": H, "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0], "hidden_act": "gelu_pytorch_tanh",
        "initializer_range": 0.02,
    }
    config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl", "text_config": text_config,
        "vision_config": vision_config, "tie_word_embeddings": True,
        # out of the tiny vocab on purpose: text-only inputs never match
        "image_token_id": V + 10, "video_token_id": V + 11,
        "vision_start_token_id": V + 12, "vision_end_token_id": V + 13,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    rng = np.random.default_rng(seed)

    def lin(o, i):
        return (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    pre = "model.language_model."
    w = {pre + "embed_tokens.weight":
         rng.standard_normal((V, H)).astype(np.float32)}
    for i in range(n_layers):
        p = f"{pre}layers.{i}."
        w[p + "input_layernorm.weight"] = norm(H)
        w[p + "post_attention_layernorm.weight"] = norm(H)
        w[p + "self_attn.q_proj.weight"] = lin(nh * D, H)
        w[p + "self_attn.k_proj.weight"] = lin(nkv * D, H)
        w[p + "self_attn.v_proj.weight"] = lin(nkv * D, H)
        w[p + "self_attn.o_proj.weight"] = lin(H, nh * D)
        w[p + "self_attn.q_norm.weight"] = norm(D)
        w[p + "self_attn.k_norm.weight"] = norm(D)
        w[p + "mlp.gate_proj.weight"] = lin(I, H)
        w[p + "mlp.up_proj.weight"] = lin(I, H)
        w[p + "mlp.down_proj.weight"] = lin(H, I)
    w[pre + "norm.weight"] = norm(H)
    save_file(w, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    cfg = from_hf_config(out_dir, select_layers=tuple(select_layers),
                         position_mode=position_mode)
    return out_dir, cfg


def _have_transformers():
    import importlib

    try:
        importlib.import_module("torch")
        tf = importlib.import_module("transformers")
        getattr(tf, "Qwen3VLForConditionalGeneration")
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def _jax():
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    return jax


# ---------------------------------------------------------------------------
# pure-JAX tests
# ---------------------------------------------------------------------------


def test_config_and_param_count():
    import jax.numpy as jnp

    from krea2_jax.qwen3_text import (Qwen3TextConfig,
                                      abstract_qwen3_text_params,
                                      from_hf_config, param_count)

    cfg = from_hf_config({"text_config": REAL_TEXT_CONFIG})
    assert cfg == Qwen3TextConfig(), cfg
    assert cfg.n_run_layers == 35 and not cfg.needs_final_norm
    # 5.x rope_parameters layout gives the same config
    tc5 = dict(REAL_TEXT_CONFIG)
    rp = dict(tc5.pop("rope_scaling"), rope_theta=tc5.pop("rope_theta"))
    assert from_hf_config(dict(tc5, rope_parameters=rp)) == cfg
    n_run = param_count(abstract_qwen3_text_params(cfg, jnp.bfloat16))
    full = param_count(abstract_qwen3_text_params(
        Qwen3TextConfig(select_layers=(36,)), jnp.bfloat16))
    print(f"  real config: {full / 1e9:.3f}B text-tower params (36 layers + "
          f"norm), {n_run / 1e9:.3f}B loaded/computed for select_layers "
          f"(35 layers)")
    assert 3.9e9 < n_run < 4.0e9 and 4.0e9 < full < 4.1e9, (n_run, full)
    for bad in (dict(select_layers=(37,)), dict(position_mode="x"),
                dict(select_layers=())):
        try:
            Qwen3TextConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    print("  config/param count OK")


def test_tiny_dir_and_loader(tmp):
    import jax.numpy as jnp
    import ml_dtypes
    from safetensors.numpy import load_file

    from krea2_jax.qwen3_text import load_qwen3_text_params, load_tokenizer

    d, cfg = make_tiny_text_encoder_dir(Path(tmp) / "te")
    tmpl = CaptionTemplate(load_tokenizer(d), text_rows=24)
    assert (cfg.prefix_len, cfg.suffix_len) == (
        tmpl.prefix_len, len(tmpl.suffix_ids))
    assert cfg.num_layers == 2 and cfg.hidden_size == 64
    assert cfg.n_run_layers == 2 and cfg.needs_final_norm
    tok = tmpl.tokenizer
    # no boundary merge for these captions (encode_captions does not rely on
    # it: it tokenizes PREFIX + caption like the conditioner)
    for c in TINY_CAPTIONS:
        assert tok.encode(PREFIX + c).ids == tmpl.prefix_ids + tok.encode(c).ids
    raw = load_file(str(d / "model.safetensors"))
    params = load_qwen3_text_params(d, dtype=jnp.float32, cfg=cfg)
    pre = "model.language_model."
    np.testing.assert_array_equal(np.asarray(params["embed_tokens"]),
                                  raw[pre + "embed_tokens.weight"])
    np.testing.assert_array_equal(np.asarray(params["norm"]),
                                  raw[pre + "norm.weight"])
    np.testing.assert_array_equal(np.asarray(params["layers"]["down_proj"][1]),
                                  raw[pre + "layers.1.mlp.down_proj.weight"])
    np.testing.assert_array_equal(np.asarray(params["layers"]["q_norm"][0]),
                                  raw[pre + "layers.0.self_attn.q_norm.weight"])
    # select_layers (0, 1): only layer 0 loaded, no final norm
    import dataclasses

    cfg1 = dataclasses.replace(cfg, select_layers=(0, 1))
    p1 = load_qwen3_text_params(d, dtype=jnp.bfloat16, cfg=cfg1)
    assert "norm" not in p1 and p1["layers"]["q_proj"].shape == (1, 64, 64)
    assert p1["embed_tokens"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(p1["layers"]["k_proj"][0]),
        raw[pre + "layers.0.self_attn.k_proj.weight"].astype(ml_dtypes.bfloat16))
    print(f"  tiny dir + loader OK (vocab {cfg.vocab_size}, prefix "
          f"{cfg.prefix_len}, suffix {cfg.suffix_len})")
    return d, cfg


def test_text_encoder_identity(d, cfg, tmp):
    """text_encoder_identity: path and file-name independent; changes with
    a weight VALUE (same shape/dtype), a weight's dtype/shape,
    tokenizer.json bytes or the resolved config; ignores weight files the
    loader does not read (vision tower); reports the bytes it hashed."""
    import dataclasses
    import shutil

    from safetensors.numpy import load_file, save_file

    from krea2_jax.qwen3_text import from_hf_config, text_encoder_identity

    tmp = Path(tmp) / "ident"
    stats = {}
    ident = text_encoder_identity(d, cfg, stats=stats)
    assert isinstance(ident, str) and len(ident) == 64
    assert stats == {"files": 1,
                     "bytes": (d / "model.safetensors").stat().st_size}, stats
    assert text_encoder_identity(d, cfg) == ident
    # the same encoder copied elsewhere under another name: same identity
    other = shutil.copytree(d, tmp / "elsewhere" / "another_name")
    ocfg = from_hf_config(other, select_layers=cfg.select_layers,
                          position_mode=cfg.position_mode)
    assert ocfg == cfg
    assert text_encoder_identity(other, ocfg) == ident

    def variant(name, edit):
        v = shutil.copytree(d, tmp / name)
        edit(v)
        return text_encoder_identity(v, cfg)

    q = "model.language_model.layers.0.self_attn.q_norm.weight"

    def retype(v):  # one used tensor stored as f16 instead of f32
        w = load_file(str(v / "model.safetensors"))
        w[q] = w[q].astype(np.float16)
        save_file(w, str(v / "model.safetensors"), metadata={"format": "pt"})

    def reshape(v):  # same bytes, another shape
        w = load_file(str(v / "model.safetensors"))
        w[q] = w[q].reshape(1, -1)
        save_file(w, str(v / "model.safetensors"), metadata={"format": "pt"})

    def revalue(v):  # one embedding value changed; same shape and dtype
        e = "model.language_model.embed_tokens.weight"
        w = load_file(str(v / "model.safetensors"))
        w[e] = w[e].copy()
        w[e][3, 5] += 1.0
        save_file(w, str(v / "model.safetensors"), metadata={"format": "pt"})

    def retok(v):  # same tokens, different tokenizer.json bytes
        j = json.loads((v / "tokenizer.json").read_text())
        (v / "tokenizer.json").write_text(json.dumps(j, indent=1))

    def vision(v):  # a weight file holding only unused (vision) tensors
        save_file({"model.visual.patch_embed.proj.weight":
                   np.ones((4, 3), np.float32)},
                  str(v / "model-visual.safetensors"))

    changed = {
        "value": variant("value", revalue),
        "dtype": variant("dtype", retype),
        "shape": variant("shape", reshape),
        "tokenizer.json": variant("tok", retok),
        "select_layers": text_encoder_identity(
            d, dataclasses.replace(cfg, select_layers=(0, 1))),
        "position_mode": text_encoder_identity(
            d, dataclasses.replace(cfg, position_mode="padded")),
    }
    for what, x in changed.items():
        assert x != ident, f"{what} did not change the identity"
    assert len(set(changed.values())) == len(changed)
    assert variant("vision", vision) == ident

    # same content under another file name (single file, globbed)
    def rename(v):
        (v / "model.safetensors").rename(v / "weights-renamed.safetensors")
    assert variant("renamed", rename) == ident

    # sharded layout with an index: the same shard contents under other
    # shard file names at another path match; one value edit does not
    def shard(v, names=("model-00001-of-00002.safetensors",
                        "model-00002-of-00002.safetensors")):
        w = load_file(str(v / "model.safetensors"))
        (v / "model.safetensors").unlink()
        keys = sorted(w)
        parts = (keys[:len(keys) // 2], keys[len(keys) // 2:])
        wm = {}
        for name, ks in zip(names, parts):
            save_file({k: w[k] for k in ks}, str(v / name),
                      metadata={"format": "pt"})
            wm.update({k: name for k in ks})
        (v / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": wm}))
    sharded = variant("sharded", shard)
    assert sharded != ident  # different file contents (two files)
    sstats = {}
    rel = shutil.copytree(tmp / "sharded", tmp / "x" / "relocated")
    idx = json.loads((rel / "model.safetensors.index.json").read_text())
    ren = {}
    for old in sorted(set(idx["weight_map"].values())):
        new = "renamed-" + old[::-1]
        (rel / old).rename(rel / new)
        ren[old] = new
    idx["weight_map"] = {k: ren[f] for k, f in idx["weight_map"].items()}
    (rel / "model.safetensors.index.json").write_text(json.dumps(idx))
    assert text_encoder_identity(rel, cfg, stats=sstats) == sharded
    assert sstats["files"] == 2, sstats

    def shard_revalue(v):
        shard(v)
        f = next(iter(sorted(v.glob("model-*.safetensors"))))
        w = load_file(str(f))
        k = sorted(w)[0]
        w[k] = w[k].copy()
        w[k].reshape(-1)[0] += 1.0
        save_file(w, str(f), metadata={"format": "pt"})
    assert variant("sharded_value", shard_revalue) not in (sharded, ident)
    print("  text_encoder_identity: path/file-name independent (single and "
          "sharded); weight value/dtype/shape/tokenizer/select_layers/"
          "position_mode change it; unused weight files do not; stats "
          "report the hashed bytes")


def _parity_main(d, cfg, tmp, mutate_ref=False, mutate_jax=False):
    """Run scripts/text_encoder_parity.py main() on the tiny encoder with
    mocked I/O: captions instead of a parquet, and the reference phase
    replaced by the JAX text tower itself (so an unmutated run is exact).
    Returns (exit code, stdout)."""
    import contextlib
    import io

    import jax
    import jax.numpy as jnp
    from safetensors.numpy import save_file

    import krea2_jax.qwen3_text as q

    sys.path.insert(0, str(REPO / "scripts"))
    import text_encoder_parity as par

    real_ts = q.text_states

    def fake_reference(args, model_dir, tmpl, captions, select_layers, tdir):
        c = q.from_hf_config(model_dir, select_layers=select_layers,
                             prefix_len=tmpl.prefix_len,
                             suffix_len=len(tmpl.suffix_ids),
                             position_mode="padded", normed_last_hidden=True)
        params = q.load_qwen3_text_params(model_dir, jnp.float32, cfg=c)
        ids, tl = tmpl.encode_captions(captions)
        st, m = jax.jit(real_ts, static_argnums=1)(params, c, ids, tl)
        st, m = np.asarray(st), np.asarray(m)
        out = {}
        for b in range(len(captions)):
            h = st[b][m[b]]
            if mutate_ref and b == 1:
                h = h.copy()
                h[0, 0, 0] = np.nan
            out[f"h{b}"], out[f"m{b}"] = h, m[b].astype(np.uint8)
        path = Path(tdir) / "ref.safetensors"
        save_file(out, str(path), metadata={"transformers": "5.0.0"})
        return path

    def nan_states(params, c, ids, txt_len):
        st, m = real_ts(params, c, ids, txt_len)
        return jnp.full_like(st, jnp.nan), m

    tags = [f"tiny:{i}" for i in range(len(TINY_CAPTIONS))]
    patches = [
        (par, "run_reference", fake_reference),
        (par, "pick_captions", lambda parquet, n, seed: TINY_CAPTIONS[:n]),
        (par, "read_captions", lambda parquet: (list(TINY_CAPTIONS), tags)),
        (par, "QWEN3_PREFIX_LEN", cfg.prefix_len),
        (par, "QWEN3_SUFFIX_LEN", cfg.suffix_len),
    ]
    if mutate_jax:
        patches.append((q, "text_states", nan_states))
    saved = [(m, k, getattr(m, k)) for m, k, _ in patches]
    buf = io.StringIO()
    code = 0
    try:
        for m, k, v in patches:
            setattr(m, k, v)
        with contextlib.redirect_stdout(buf):
            try:
                par.main(["--text-encoder", str(d), "--parquet", "unused",
                          "--n", str(len(TINY_CAPTIONS)), "--dtype", "f32",
                          "--select-layers",
                          ",".join(map(str, cfg.select_layers)),
                          "--text-rows", "24", "--batch", "2"])
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
    finally:
        for m, k, v in saved:
            setattr(m, k, v)
    return code, buf.getvalue()


def test_parity_script_nonfinite(d, cfg, tmp):
    """text_encoder_parity.py fails on non-finite states instead of passing
    a NaN rel-norm through the tolerance check."""
    code, out = _parity_main(d, cfg, tmp)
    assert code == 0 and "-> PASS" in out, out
    assert "encode_captions == conditioner layout: 0 mismatches" in out, out
    code, out = _parity_main(d, cfg, tmp, mutate_jax=True)
    assert code != 0 and "FAIL: non-finite JAX states" in out, out
    code, out = _parity_main(d, cfg, tmp, mutate_ref=True)
    assert code != 0 and "FAIL: non-finite reference states" in out, out
    print("  text_encoder_parity.py: exact tiny run PASS; NaN JAX states / "
          "NaN reference -> exit 1")


def test_padding_invariance(d, cfg):
    import dataclasses

    import jax
    import jax.numpy as jnp

    from krea2_jax.qwen3_text import (load_qwen3_text_params, load_tokenizer,
                                      text_states)

    params = load_qwen3_text_params(d, dtype=jnp.float32, cfg=cfg)
    tok = load_tokenizer(d)
    caps = TINY_CAPTIONS[1:4]
    f = jax.jit(text_states, static_argnums=1)
    out = {}
    for mode in ("contiguous", "padded"):
        c = dataclasses.replace(cfg, position_mode=mode)
        for rows in (8, 32):
            t = CaptionTemplate(tok, text_rows=rows)
            ids, tl = t.encode_captions(caps)
            s, m = f(params, c, ids, tl)
            s, m = np.asarray(s), np.asarray(m)
            np.testing.assert_array_equal(m, t.valid_mask(tl))
            assert np.all(s[~m] == 0)
            out[mode, rows] = (s, tl)
    s8, tl8 = out["contiguous", 8]
    s32, tl32 = out["contiguous", 32]
    # caption 1 is short enough to fit in 8 rows untruncated
    assert tl8[0] == tl32[0] and tl8[0] <= 8
    n = int(tl8[0])
    # (equal up to f32 reduction order: the softmax/matmul length differs)
    np.testing.assert_allclose(s8[0, :n], s32[0, :n], rtol=0, atol=1e-5)
    # truncated captions: rows before the suffix still agree
    for b in range(len(caps)):
        k = int(min(tl8[b], tl32[b])) - cfg.suffix_len
        np.testing.assert_allclose(s8[b, :k], s32[b, :k], rtol=0, atol=1e-5)
    # padded mode: caption rows equal contiguous mode, suffix rows move
    for rows in (8, 32):
        sc, tl = out["contiguous", rows]
        sp, _ = out["padded", rows]
        for b in range(len(caps)):
            nc = int(tl[b]) - cfg.suffix_len
            np.testing.assert_allclose(sp[b, :nc], sc[b, :nc], rtol=0,
                                       atol=1e-5)
            if int(tl[b]) < rows:  # suffix not at the end -> positions differ
                assert np.abs(sp[b, nc:tl[b], 1:] - sc[b, nc:tl[b], 1:]).max() > 1e-4
    print("  padding invariance OK (contiguous: valid rows independent of "
          "text_rows; padded: caption rows equal)")


def test_bf16_close(d, cfg):
    import jax
    import jax.numpy as jnp

    from krea2_jax.qwen3_text import (load_qwen3_text_params, load_tokenizer,
                                      text_states)

    t = CaptionTemplate(load_tokenizer(d), text_rows=24)
    ids, tl = t.encode_captions(TINY_CAPTIONS)
    f = jax.jit(text_states, static_argnums=1)
    ref, m = f(load_qwen3_text_params(d, jnp.float32, cfg=cfg), cfg, ids, tl)
    s, _ = f(load_qwen3_text_params(d, jnp.bfloat16, cfg=cfg), cfg, ids, tl)
    assert s.dtype == jnp.bfloat16
    ref, s, m = np.asarray(ref), np.asarray(s, np.float32), np.asarray(m)
    for j in range(len(cfg.select_layers)):
        a, b = s[:, :, j][m], ref[:, :, j][m]
        rel = np.linalg.norm(a - b) / np.linalg.norm(b)
        assert rel < 3e-2, (j, rel)
    print(f"  bf16 vs f32 OK (last layer rel-norm err {rel:.2e})")


def _mesh_check():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from krea2_jax.checkpoint import make_mesh
    from krea2_jax.qwen3_text import (init_qwen3_text_params,
                                      load_qwen3_text_params, load_tokenizer,
                                      text_states)

    assert jax.device_count() == 8, jax.device_count()
    with tempfile.TemporaryDirectory() as tmp:
        d, cfg = make_tiny_text_encoder_dir(Path(tmp) / "te")
        mesh = make_mesh()
        params = load_qwen3_text_params(d, jnp.float32, mesh=mesh, cfg=cfg)
        for leaf in jax.tree.leaves(params):
            assert leaf.sharding.is_fully_replicated, leaf.sharding
        t = CaptionTemplate(load_tokenizer(d), text_rows=24)
        caps = (TINY_CAPTIONS * 2)[:8]
        ids, tl = t.encode_captions(caps)
        f = jax.jit(text_states, static_argnums=1)
        ref, mref = f(load_qwen3_text_params(d, jnp.float32, cfg=cfg),
                      cfg, ids, tl)
        bs = NamedSharding(mesh, P("fsdp"))
        s, m = f(params, cfg, jax.device_put(ids, bs), jax.device_put(tl, bs))
        np.testing.assert_allclose(np.asarray(s), np.asarray(ref), atol=1e-5,
                                   rtol=1e-5)
        np.testing.assert_array_equal(np.asarray(m), np.asarray(mref))
        # init params (bf16) also run sharded
        p0 = init_qwen3_text_params(cfg, jax.random.key(0), jnp.bfloat16)
        p0 = jax.device_put(p0, NamedSharding(mesh, P()))
        s0, _ = f(p0, cfg, jax.device_put(ids, bs), jax.device_put(tl, bs))
        assert np.isfinite(np.asarray(s0, np.float32)).all()
    print(f"  8-device mesh OK (batch-sharded == unsharded, out sharding "
          f"{s.sharding.spec})")


def test_mesh():
    import jax

    if jax.device_count() == 8:
        _mesh_check()
        return
    env = dict(os.environ,
               XLA_FLAGS="--xla_force_host_platform_device_count=8",
               JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, __file__, "--mesh-only"], env=env,
                       capture_output=True, text=True, timeout=600)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stderr.write(r.stderr[-4000:])
        raise AssertionError("mesh subprocess failed")


# ---------------------------------------------------------------------------
# transformers parity
# ---------------------------------------------------------------------------


def test_rope_matches_hf():
    """HF interleaved MRoPE with identical t/h/w positions == our 1-D
    rotate_half RoPE (real 4B config and the tiny one)."""
    import jax.numpy as jnp
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb)

    from krea2_jax.qwen3_text import apply_rope, from_hf_config, rope_cos_sin

    for tc in (REAL_TEXT_CONFIG,
               dict(REAL_TEXT_CONFIG, head_dim=16, num_attention_heads=4,
                    num_key_value_heads=2, hidden_size=64,
                    rope_scaling={"mrope_interleaved": True,
                                  "mrope_section": [4, 2, 2],
                                  "rope_type": "default"})):
        hf = Qwen3VLTextRotaryEmbedding(Qwen3VLTextConfig(**tc))
        cfg = from_hf_config(tc)
        L = 300
        pos = torch.arange(L)[None].expand(3, 2, L)  # identical t/h/w
        x = torch.zeros(2, L, 4)
        cos_t, sin_t = hf(x, pos)
        cos_j, sin_j = rope_cos_sin(cfg, jnp.broadcast_to(jnp.arange(L), (2, L)))
        np.testing.assert_allclose(np.asarray(cos_j), cos_t.numpy(), atol=2e-5)
        np.testing.assert_allclose(np.asarray(sin_j), sin_t.numpy(), atol=2e-5)
        # apply on random q/k: HF layout (B, heads, L, D), ours (B, L, heads, D)
        g = torch.Generator().manual_seed(0)
        q = torch.randn(2, 3, L, cfg.head_dim, generator=g)
        qe, _ = apply_rotary_pos_emb(q, q, cos_t, sin_t)
        ours = apply_rope(jnp.asarray(q.numpy()).transpose(0, 2, 1, 3),
                          cos_j, sin_j)
        np.testing.assert_allclose(np.asarray(ours).transpose(0, 2, 1, 3),
                                   qe.numpy(), atol=1e-4)
    print("  RoPE: HF interleaved MRoPE (t=h=w) == 1-D rotate_half RoPE OK")


def _errs(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max()), float(
        np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def test_parity_vs_text_encoder(d, cfg):
    """JAX f32 vs scripts/cache_dataset.py::TextEncoder (production
    conditioner incl. its mid-sequence padding) and vs a direct HF call on
    our compact layout."""
    import dataclasses

    import jax
    import jax.numpy as jnp
    import torch
    import transformers

    sys.path.insert(0, str(REPO / "scripts"))
    import cache_dataset

    transformers.logging.set_verbosity_error()

    from krea2_jax.qwen3_text import (load_qwen3_text_params, load_tokenizer,
                                      text_states)

    text_rows = 24
    tmpl = CaptionTemplate(load_tokenizer(d), text_rows=text_rows)
    ids, tl = tmpl.encode_captions(TINY_CAPTIONS)
    assert tl[-1] == text_rows, "the long caption must be truncated"
    assert tl[0] == cfg.suffix_len, "the empty caption has only the suffix"
    # transformers >= 5: positions = arange over TextEncoder's mid-padded
    # layout and hidden_states[-1] is final-normed; 4.x: mask-cumsum
    # positions and a raw hidden_states[-1]
    tf5 = int(transformers.__version__.split(".")[0]) >= 5
    mode = "padded" if tf5 else "contiguous"
    cfg = dataclasses.replace(cfg, normed_last_hidden=tf5)

    # production conditioner, constants adapted to the tiny tokenizer/model
    saved = (cache_dataset.PREFIX_IDX, cache_dataset.SUFFIX_START_IDX,
             cache_dataset.SELECT_LAYERS)
    cache_dataset.PREFIX_IDX = cfg.prefix_len
    cache_dataset.SUFFIX_START_IDX = cfg.suffix_len
    cache_dataset.SELECT_LAYERS = cfg.select_layers
    try:
        enc = cache_dataset.TextEncoder(str(d), "cpu", torch.float32,
                                        max_length=text_rows)
        ref = [enc(c) for c in TINY_CAPTIONS]
    finally:
        (cache_dataset.PREFIX_IDX, cache_dataset.SUFFIX_START_IDX,
         cache_dataset.SELECT_LAYERS) = saved

    params = load_qwen3_text_params(d, jnp.float32, cfg=cfg)
    f = jax.jit(text_states, static_argnums=1)
    worst = {}
    for m_name in ("contiguous", "padded"):
        s, m = f(params, dataclasses.replace(cfg, position_mode=m_name), ids, tl)
        worst[m_name] = (np.asarray(s), np.asarray(m))
    s, m = worst[mode]
    max_abs = max_rel = 0.0
    for b, (h, hm) in enumerate(ref):
        h, hm = h.numpy(), hm.numpy()
        assert h.shape == (text_rows, len(cfg.select_layers), cfg.hidden_size)
        n_cap = int(tl[b]) - cfg.suffix_len
        # TextEncoder layout: caption, pads, suffix at the end
        exp_mask = np.zeros(text_rows, bool)
        exp_mask[:n_cap] = True
        exp_mask[text_rows - cfg.suffix_len:] = True
        np.testing.assert_array_equal(hm, exp_mask)
        assert int(hm.sum()) == int(m[b].sum()) == int(tl[b])
        a, r = s[b, :tl[b]], h[hm]
        e_abs, e_rel = _errs(a, r)
        tol = 1e-4 * np.maximum(1.0, np.abs(r))
        assert np.all(np.abs(a - r) <= tol), (b, e_abs, e_rel)
        max_abs, max_rel = max(max_abs, e_abs), max(max_rel, e_rel)
    print(f"  parity vs cache_dataset.TextEncoder (transformers "
          f"{transformers.__version__} -> position_mode={mode!r}, "
          f"normed_last_hidden={tf5}): "
          f"max abs {max_abs:.2e}, rel-norm {max_rel:.2e} over "
          f"{len(TINY_CAPTIONS)} captions x layers {cfg.select_layers}")
    other = "contiguous" if mode == "padded" else "padded"
    so = worst[other][0]
    diff = max(np.abs(so[b, :tl[b]] - ref[b][0].numpy()[ref[b][1].numpy()]).max()
               for b in range(len(ref)) if tl[b] < text_rows)
    print(f"  (position_mode={other!r} vs TextEncoder: max abs {diff:.2e} on "
          f"suffix rows -> the modes really differ)")
    assert diff > 1e-3

    # direct HF call on the compact layout (pads at the end): HF positions
    # (mask-cumsum in 4.x, arange in 5.x) coincide on valid rows -> must
    # equal contiguous mode in every transformers version
    model = enc.model
    am = torch.from_numpy(np.arange(ids.shape[1])[None] <
                          (cfg.prefix_len + tl)[:, None]).long()
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids).long(), attention_mask=am,
                    output_hidden_states=True)
    hs = torch.stack([out.hidden_states[i] for i in cfg.select_layers],
                     dim=2)[:, cfg.prefix_len:].numpy()
    sc, mc = worst["contiguous"]
    e_abs, e_rel = _errs(sc[mc], hs[mc])
    assert np.all(np.abs(sc[mc] - hs[mc]) <= 1e-4 * np.maximum(1, np.abs(hs[mc])))
    print(f"  parity vs HF compact layout (contiguous mode): max abs "
          f"{e_abs:.2e}, rel-norm {e_rel:.2e}")
    del enc, model


def main():
    if "--mesh-only" in sys.argv:
        _jax()
        _mesh_check()
        return
    _jax()
    print("test_config_and_param_count")
    test_config_and_param_count()
    with tempfile.TemporaryDirectory() as tmp:
        print("test_tiny_dir_and_loader")
        d, cfg = test_tiny_dir_and_loader(tmp)
        print("test_text_encoder_identity")
        test_text_encoder_identity(d, cfg, tmp)
        print("test_parity_script_nonfinite")
        test_parity_script_nonfinite(d, cfg, tmp)
        print("test_padding_invariance")
        test_padding_invariance(d, cfg)
        print("test_bf16_close")
        test_bf16_close(d, cfg)
        print("test_mesh")
        test_mesh()
        ok, why = _have_transformers()
        if not ok:
            print(f"SKIP transformers parity tests (transformers/torch not "
                  f"available: {why}); run with a python that has "
                  f"transformers >= 4.57 for test_rope_matches_hf and "
                  f"test_parity_vs_text_encoder")
        else:
            print("test_rope_matches_hf")
            test_rope_matches_hf()
            print("test_parity_vs_text_encoder")
            test_parity_vs_text_encoder(d, cfg)
    print("all qwen3_text tests passed")


if __name__ == "__main__":
    main()
