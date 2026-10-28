"""Shared fixtures-as-functions for the test suite (not collected by pytest).

make_tiny_tokenizer / make_tiny_text_encoder_dir: a tiny Qwen-style BPE and
an HF-layout Qwen3-VL text tower dir; make_fake_latent_cache: a
krea2-latents-v1 cache (scripts/cache_latents.py layout) with a digest
manifest; train_cmd / run_train: scripts/train_full.py on the tiny model.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

REPO = Path(__file__).resolve().parent.parent
BF16 = ml_dtypes.bfloat16
TINY = dict(features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
            layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3)
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
              r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
CORPUS = ["a girl", "a girl with long silver hair and red eyes, wearing a dress",
          "two cats sitting on a wooden table next to a window, drawn by someone"]


def make_tiny_tokenizer(out_dir, vocab_size=600, extra_corpus=()):
    """Byte-level BPE with the Qwen pre-tokenizer and special tokens
    <|endoftext|> <|im_start|> <|im_end|>; writes out_dir/tokenizer.json."""
    from tokenizers import (Regex, Tokenizer, decoders, models, normalizers,
                            pre_tokenizers, trainers)
    from krea2_jax.text_template import PREFIX, SUFFIX

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, show_progress=False,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([PREFIX, SUFFIX] + CORPUS * 3 + list(extra_corpus),
                            trainer=trainer)
    tok.save(str(out_dir / "tokenizer.json"))
    return Tokenizer.from_file(str(out_dir / "tokenizer.json"))


def make_tiny_text_encoder_dir(out_dir, select_layers=(0, 1, 2), seed=0):
    """HF-layout Qwen3-VL dir (config.json, tokenizer.json, f32 text weights
    under model.language_model.*): hidden 64, 2 layers, 4 q / 2 kv heads,
    head_dim 16. Returns (dir, Qwen3TextConfig)."""
    from safetensors.numpy import save_file
    from krea2_jax.qwen3_text import from_hf_config

    out_dir = Path(out_dir)
    tok = make_tiny_tokenizer(out_dir)
    V, H, L, nh, nkv, D, I = tok.get_vocab_size(), 64, 2, 4, 2, 16, 96
    tc = {"model_type": "qwen3_vl_text", "head_dim": D, "hidden_size": H,
          "intermediate_size": I, "num_attention_heads": nh,
          "num_hidden_layers": L, "num_key_value_heads": nkv,
          "rms_norm_eps": 1e-6, "rope_theta": 5000000, "vocab_size": V,
          "rope_scaling": {"mrope_interleaved": True,
                           "mrope_section": [4, 2, 2], "rope_type": "default"}}
    (out_dir / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_vl", "text_config": tc}))
    rng = np.random.default_rng(seed)
    lin = lambda o, i: (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)
    norm = lambda n: (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)
    pre = "model.language_model."
    w = {pre + "embed_tokens.weight": rng.standard_normal((V, H)).astype(np.float32),
         pre + "norm.weight": norm(H)}
    for i in range(L):
        p = f"{pre}layers.{i}."
        w.update({p + "input_layernorm.weight": norm(H),
                  p + "post_attention_layernorm.weight": norm(H),
                  p + "self_attn.q_proj.weight": lin(nh * D, H),
                  p + "self_attn.k_proj.weight": lin(nkv * D, H),
                  p + "self_attn.v_proj.weight": lin(nkv * D, H),
                  p + "self_attn.o_proj.weight": lin(H, nh * D),
                  p + "self_attn.q_norm.weight": norm(D),
                  p + "self_attn.k_norm.weight": norm(D),
                  p + "mlp.gate_proj.weight": lin(I, H),
                  p + "mlp.up_proj.weight": lin(I, H),
                  p + "mlp.down_proj.weight": lin(H, I)})
    save_file(w, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    return out_dir, from_hf_config(out_dir, select_layers=tuple(select_layers))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refresh_digests(data_dir):
    """Recompute manifest.json "digests" from the shards on disk."""
    mpath = Path(data_dir) / "latents" / "manifest.json"
    m = json.loads(mpath.read_text())
    m["digests"] = {n: file_sha256(mpath.parent / n) for n in m["shards"]}
    mpath.write_text(json.dumps(m))
    return m


def make_fake_latent_cache(data_dir, rows, lat_channels=4, seed=0, vae="fake-vae"):
    """krea2-latents-v1 cache: metadata.parquet (all S1 columns, rows in a
    rank order that differs from id order), latents/<tar stem>.safetensors
    with bf16 (C, h/8, w/8) latents keyed by str(id), manifest.json with
    sha256 digests. rows: dicts with at least `id`. Returns {id: latent}."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from safetensors.numpy import save_file

    data_dir = Path(data_dir)
    (data_dir / "latents").mkdir(parents=True, exist_ok=True)
    full = []
    for r in rows:
        i = int(r["id"])
        d = {"id": i, "long": f"long caption of {i}", "medium": f"medium caption {i}",
             "short": f"short {i}", "tags": f"tag{i}, 1girl",
             "character_name": f"char{i % 5}", "series_name": f"series{i % 3}",
             "artist_name": f"artist{i % 7}", "rating": "general",
             "bucket_w": 32, "bucket_h": 32, "score": float(i % 10) / 10.0,
             "tar": "data/train-0000.tar"}
        full.append({**d, **r})
    n = len(full)
    perm = np.random.default_rng([seed, n]).permutation(n)
    if n > 1 and np.array_equal(perm, np.argsort([d["id"] for d in full])):
        perm = perm[::-1]
    ordered = [full[j] for j in perm]
    for rank, d in enumerate(ordered):
        d["rank"] = rank
        for f in ("long", "medium", "short", "tags"):
            d[f + "_tok"] = len(d[f].split())
    ints = {"id", "bucket_w", "bucket_h", "rank", "long_tok", "medium_tok",
            "short_tok", "tags_tok"}
    cols = ["id", "long", "medium", "short", "tags", "character_name",
            "series_name", "artist_name", "rating", "bucket_w", "bucket_h",
            "score", "rank", "long_tok", "medium_tok", "short_tok", "tags_tok", "tar"]
    pq.write_table(pa.table({c: pa.array(
        [d[c] for d in ordered], type=pa.int64() if c in ints else
        pa.float32() if c == "score" else pa.string()) for c in cols}),
        str(data_dir / "metadata.parquet"))
    latents, shards = {}, {}
    for d in ordered:
        i = int(d["id"])
        latents[i] = np.random.default_rng([seed, i]).standard_normal(
            (lat_channels, int(d["bucket_h"]) // 8, int(d["bucket_w"]) // 8)
        ).astype(BF16)
        shards.setdefault(d["tar"], {})[str(i)] = latents[i]
    names = []
    for tar, tensors in sorted(shards.items()):
        names.append(Path(tar).stem + ".safetensors")
        save_file(tensors, str(data_dir / "latents" / names[-1]), metadata={
            "format": "krea2-latents-v1", "vae": vae, "source": tar,
            "count": str(len(tensors))})
    (data_dir / "latents" / "manifest.json").write_text(json.dumps({
        "format": "krea2-latents-v1", "vae": vae, "n": n, "shards": sorted(names)}))
    refresh_digests(data_dir)
    return latents


# --- scripts/train_full.py on the tiny model --------------------------------
TEXT_OVERRIDES = {"select_layers": [0, 1, 2]}
STEP_RE = re.compile(
    r"^step\s+(\d+)/\d+\s+loss\s+(\S+)\s+ema_loss\s+\S+\s+gnorm\s+(\S+)", re.M)


def train_cmd(base, cache, te, out, extra=()):
    return [sys.executable, str(REPO / "scripts" / "train_full.py"),
            "--base-ckpt", str(base), "--data-dir", str(cache),
            "--text-encoder", str(te), "--text-model-config",
            json.dumps(TEXT_OVERRIDES), "--text-rows", "32", "--img-tokens", "16",
            "--output-dir", str(out), "--model-config", json.dumps(TINY),
            "--steps", "12", "--save-every", "6", "--export-every", "6",
            "--lr", "1e-3", "--warmup-steps", "2", "--batch-size", "8",
            "--ema-decay", "0.9", "--log-every", "1", "--qchunk", "256",
            "--remat-policy", "dots", "--loader-workers", "2", *extra]


def run_train(*args, extra=(), ok=True, timeout=900):
    """Run train_full.py; returns ({step: (loss, gnorm)}, stdout, stderr)."""
    proc = subprocess.run(train_cmd(*args, extra=extra), capture_output=True,
                          text=True, timeout=timeout,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if ok:
        assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    else:
        assert proc.returncode != 0, proc.stdout[-2000:]
    steps = {int(s): (float(l), float(g))
             for s, l, g in STEP_RE.findall(proc.stdout)}
    return steps, proc.stdout, proc.stderr
