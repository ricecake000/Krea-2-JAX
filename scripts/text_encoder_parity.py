"""Real-weight parity: production torch conditioner vs the JAX Qwen3-VL text
tower (krea2_jax/qwen3_text.py), on captions from the dataset parquet.

Reference = scripts/cache_dataset.py::TextEncoder(max_length=text_rows) (the
official Qwen3VLConditioner logic, transformers). It runs FIRST in a
subprocess (optionally another interpreter, --torch-python) that writes the
valid hidden-state rows to a temp safetensors file and exits, so torch and
JAX weights never share host RAM (bf16: ~9 GB peak each). Then JAX loads the
same checkpoint and compares, per selected layer: max abs error and
relative-norm error ||jax - ref|| / ||ref|| over all valid rows, split into
caption rows and suffix rows.

Position semantics differ by transformers version (see qwen3_text module
docstring): --position-mode auto picks "padded" when the reference ran under
transformers >= 5 and "contiguous" under 4.x, so the comparison is
like-for-like; the other mode only changes the 5 suffix rows.

Tokenizer checks (always; --tokenizer-only skips the models): prefix_len ==
34, len(suffix_ids) == 5, pad = <|endoftext|>, and for EVERY caption variant
in the parquet (long/medium/short, each with and without
", drawn by {artist}") encode(PREFIX + cap) == prefix_ids + encode(cap)
(the property CaptionTemplate relies on to match the conditioner).

    python scripts/text_encoder_parity.py --tokenizer-only \
        --text-encoder Qwen/Qwen3-VL-4B-Instruct --parquet metadata.parquet
    python scripts/text_encoder_parity.py --text-encoder <dir|hf id> \
        --parquet metadata.parquet --n 32 --dtype bf16 --torch-device cuda
"""

import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from krea2_jax.text_template import (
    PAD_TOKEN,
    PREFIX,
    QWEN3_PREFIX_LEN,
    QWEN3_SUFFIX_LEN,
    SELECT_LAYERS,
    CaptionTemplate,
)

VARIANTS = ("long", "medium", "short")


def read_captions(parquet):
    """All caption variants fed in training: (list[str], list[str] tags)."""
    import pyarrow.parquet as pq

    t = pq.read_table(parquet, columns=["id", *VARIANTS, "artist_name"])
    cols = {k: t.column(k).to_pylist() for k in t.column_names}
    caps, tags = [], []
    for i in range(t.num_rows):
        artist = cols["artist_name"][i] or ""
        for v in VARIANTS:
            c = cols[v][i] or ""
            caps.append(c)
            tags.append(f"{cols['id'][i]}:{v}")
            if artist:
                caps.append(f"{c}, drawn by {artist}")
                tags.append(f"{cols['id'][i]}:{v}+artist")
    return caps, tags


def tokenizer_checks(model_dir, parquet, text_rows):
    from krea2_jax.qwen3_text import load_tokenizer

    tok = load_tokenizer(model_dir)
    t = CaptionTemplate(tok, text_rows=text_rows)
    ok = True
    print(f"prefix_len {t.prefix_len} (expect {QWEN3_PREFIX_LEN}), "
          f"suffix_len {len(t.suffix_ids)} (expect {QWEN3_SUFFIX_LEN}), "
          f"pad {PAD_TOKEN!r} = {t.pad_id}, total_len {t.total_len}")
    if (t.prefix_len, len(t.suffix_ids)) != (QWEN3_PREFIX_LEN, QWEN3_SUFFIX_LEN):
        print("FAIL: prefix/suffix lengths differ from the conditioner's "
              "PREFIX_IDX/SUFFIX_START_IDX")
        ok = False
    if parquet is None:
        return ok, t
    t0 = time.time()
    caps, tags = read_captions(parquet)
    mism, lens = [], np.empty(len(caps), np.int64)
    chunk = 8192  # Encoding objects are heavy: bound host memory
    for s0 in range(0, len(caps), chunk):
        part = caps[s0:s0 + chunk]
        plain = tok.encode_batch(part)
        joined = tok.encode_batch([PREFIX + c for c in part])
        for i, (a, b) in enumerate(zip(joined, plain), start=s0):
            lens[i] = len(b.ids)
            if a.ids != t.prefix_ids + b.ids:
                mism.append(tags[i])
        del plain, joined
    trunc = int((lens > t.max_caption_tokens).sum())
    print(f"concat property: {len(mism)} mismatches over {len(caps)} caption "
          f"variants ({time.time() - t0:.0f}s); tokens max {lens.max()}, "
          f"mean {lens.mean():.1f}; {trunc} exceed max_caption_tokens "
          f"{t.max_caption_tokens} (truncated)")
    if mism:
        print(f"FAIL: first mismatches: {mism[:10]}")
        ok = False
    return ok, t


def pick_captions(parquet, n, seed):
    """The longest long caption (+artist) plus n-1 random variants."""
    import pyarrow.parquet as pq

    t = pq.read_table(parquet, columns=[*VARIANTS, "artist_name", "long_tok"])
    cols = {k: t.column(k).to_pylist() for k in t.column_names}
    rng = np.random.default_rng(seed)
    i0 = int(np.argmax(cols["long_tok"]))
    picks = [(i0, "long", True)]
    for i in rng.choice(t.num_rows, size=max(n - 1, 0), replace=False):
        picks.append((int(i), VARIANTS[rng.integers(3)], bool(rng.random() < .5)))
    out = []
    for i, v, art in picks:
        c = cols[v][i] or ""
        a = cols["artist_name"][i] or ""
        out.append(f"{c}, drawn by {a}" if art and a else c)
    return out


def torch_phase(args):
    """Subprocess: run TextEncoder, save valid rows + masks, exit."""
    import torch
    import transformers
    from safetensors.torch import save_file

    sys.path.insert(0, str(REPO / "scripts"))
    import cache_dataset

    transformers.logging.set_verbosity_error()
    cfg = json.loads(Path(args.torch_phase_in).read_text())
    cache_dataset.PREFIX_IDX = cfg["prefix_len"]
    cache_dataset.SUFFIX_START_IDX = cfg["suffix_len"]
    cache_dataset.SELECT_LAYERS = tuple(cfg["select_layers"])
    dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[cfg["dtype"]]
    t0 = time.time()
    enc = cache_dataset.TextEncoder(cfg["model_dir"], cfg["device"], dtype,
                                    max_length=cfg["text_rows"])
    print(f"[torch] transformers {transformers.__version__}, {cfg['dtype']} on "
          f"{cfg['device']}, loaded in {time.time() - t0:.0f}s", flush=True)
    out = {}
    t0 = time.time()
    for b, c in enumerate(cfg["captions"]):
        h, m = enc(c)
        out[f"h{b}"] = h[m].contiguous().cpu()
        out[f"m{b}"] = m.to(torch.uint8).cpu()
    print(f"[torch] {len(cfg['captions'])} captions in {time.time() - t0:.0f}s",
          flush=True)
    save_file(out, args.torch_phase_out,
              metadata={"transformers": transformers.__version__})


def run_reference(args, model_dir, tmpl, captions, select_layers, tmp):
    spec = {
        "model_dir": str(model_dir), "device": args.torch_device,
        "dtype": args.torch_dtype or args.dtype, "text_rows": args.text_rows,
        "prefix_len": tmpl.prefix_len, "suffix_len": len(tmpl.suffix_ids),
        "select_layers": list(select_layers), "captions": captions,
    }
    spec_path, out_path = Path(tmp) / "spec.json", Path(tmp) / "ref.safetensors"
    spec_path.write_text(json.dumps(spec))
    cmd = [args.torch_python or sys.executable, str(Path(__file__).resolve()),
           "--torch-phase-in", str(spec_path),
           "--torch-phase-out", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--text-encoder", default="Qwen/Qwen3-VL-4B-Instruct",
                    help="local dir or HF repo id")
    ap.add_argument("--parquet", help="dataset metadata.parquet")
    ap.add_argument("--n", type=int, default=32, help="captions to compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-rows", type=int, default=256)
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16",
                    help="JAX param/compute dtype (and torch's by default)")
    ap.add_argument("--torch-dtype", choices=["bf16", "f32"], default=None)
    ap.add_argument("--torch-device", default="cpu", help="cpu|cuda")
    ap.add_argument("--torch-python", default=None,
                    help="interpreter for the torch reference (needs torch + "
                         "transformers); default: this one")
    ap.add_argument("--position-mode", default="auto",
                    choices=["auto", "contiguous", "padded"],
                    help="auto: padded for a transformers>=5 reference, "
                         "contiguous for 4.x")
    ap.add_argument("--select-layers", default=None,
                    help="comma list (default: text_template.SELECT_LAYERS)")
    ap.add_argument("--batch", type=int, default=4, help="JAX batch size")
    ap.add_argument("--tol", type=float, default=None,
                    help="max per-layer rel-norm error (default 1e-3 f32, "
                         "5e-2 bf16)")
    ap.add_argument("--tokenizer-only", action="store_true")
    ap.add_argument("--torch-phase-in", help=argparse.SUPPRESS)
    ap.add_argument("--torch-phase-out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.torch_phase_in:
        torch_phase(args)
        return

    from krea2_jax.qwen3_text import resolve_text_encoder

    model_dir = resolve_text_encoder(args.text_encoder)
    print(f"text encoder: {model_dir}")
    ok, tmpl = tokenizer_checks(model_dir, args.parquet, args.text_rows)
    if args.tokenizer_only:
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if args.parquet is None:
        raise SystemExit("--parquet is required")
    select = (tuple(int(x) for x in args.select_layers.split(","))
              if args.select_layers else SELECT_LAYERS)
    captions = pick_captions(args.parquet, args.n, args.seed)
    ids, txt_len = tmpl.encode_captions(captions)
    print(f"{len(captions)} captions, txt_len min {txt_len.min()} max "
          f"{txt_len.max()} (text_rows {args.text_rows})")

    with tempfile.TemporaryDirectory(prefix="te_parity_") as tmp:
        ref_path = run_reference(args, model_dir, tmpl, captions, select, tmp)
        gc.collect()

        import jax
        import jax.numpy as jnp
        from safetensors import safe_open

        from krea2_jax.qwen3_text import (from_hf_config,
                                          load_qwen3_text_params,
                                          param_count, text_states)

        with safe_open(str(ref_path), framework="numpy") as f:
            tf_version = (f.metadata() or {}).get("transformers", "?")
            refs = [(f.get_tensor(f"h{b}"), f.get_tensor(f"m{b}").astype(bool))
                    for b in range(len(captions))]
        tf5 = int(tf_version.split(".")[0]) >= 5 if tf_version != "?" else True
        mode = args.position_mode
        if mode == "auto":
            mode = "padded" if tf5 else "contiguous"
        cfg = from_hf_config(model_dir, select_layers=select,
                             prefix_len=tmpl.prefix_len,
                             suffix_len=len(tmpl.suffix_ids),
                             position_mode=mode, normed_last_hidden=tf5)
        dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
        t0 = time.time()
        params = load_qwen3_text_params(model_dir, dtype=dtype, cfg=cfg)
        print(f"[jax] {jax.devices()[0].platform}, {param_count(params) / 1e6:.1f}M "
              f"params ({cfg.n_run_layers} layers) loaded in "
              f"{time.time() - t0:.0f}s; reference transformers {tf_version} "
              f"-> position_mode {mode!r}")
        f = jax.jit(text_states, static_argnums=1)
        S, sl = len(select), cfg.suffix_len
        # per layer, per row kind (0 caption, 1 suffix): max abs, sq err, sq ref
        mx = np.zeros((S, 2))
        se = np.zeros((S, 2))
        sr = np.zeros((S, 2))
        worst = []
        t0 = time.time()
        for s0 in range(0, len(captions), args.batch):
            sl_b = slice(s0, s0 + args.batch)
            ib, tb = ids[sl_b], txt_len[sl_b]
            pad = args.batch - len(ib)
            if pad:  # keep one compiled shape
                ib = np.concatenate([ib, np.repeat(ib[-1:], pad, 0)])
                tb = np.concatenate([tb, np.repeat(tb[-1:], pad, 0)])
            st, m = f(params, cfg, ib, tb)
            st, m = np.asarray(st, np.float32), np.asarray(m)
            for j in range(len(ib) - pad):
                b = s0 + j
                h, hm = refs[b]
                n = int(tb[j])
                if int(hm.sum()) != n or int(m[j].sum()) != n:
                    raise SystemExit(f"caption {b}: mask mismatch (torch "
                                     f"{int(hm.sum())}, jax {int(m[j].sum())},"
                                     f" txt_len {n})")
                a, r = st[j, :n], h.astype(np.float32)
                d = a - r
                for k, rows in ((0, slice(0, n - sl)), (1, slice(n - sl, n))):
                    dk, rk = d[rows], r[rows]
                    if dk.size:
                        mx[:, k] = np.maximum(mx[:, k], np.abs(dk).max(axis=(0, 2)))
                        se[:, k] += (dk.astype(np.float64) ** 2).sum(axis=(0, 2))
                        sr[:, k] += (rk.astype(np.float64) ** 2).sum(axis=(0, 2))
                worst.append((float(np.linalg.norm(d) / np.linalg.norm(r)), b))
        print(f"[jax] forward {time.time() - t0:.0f}s")
        rel = np.sqrt(se / np.maximum(sr, 1e-30))
        rel_all = np.sqrt(se.sum(1) / np.maximum(sr.sum(1), 1e-30))
        print(f"{'layer':>6} {'max_abs cap':>12} {'max_abs suf':>12} "
              f"{'rel cap':>10} {'rel suf':>10} {'rel all':>10} {'ref rms':>10}")
        n_rows = np.array([[sum(int(t) - sl for t in txt_len),
                            sl * len(captions)]]) * cfg.hidden_size
        rms = np.sqrt(sr.sum(1) / n_rows.sum())
        for j, i in enumerate(select):
            print(f"{i:>6} {mx[j, 0]:>12.4g} {mx[j, 1]:>12.4g} {rel[j, 0]:>10.3e} "
                  f"{rel[j, 1]:>10.3e} {rel_all[j]:>10.3e} {rms[j]:>10.4g}")
        worst.sort(reverse=True)
        print("worst captions (rel-norm, index): "
              + ", ".join(f"{e:.3e}@{b}" for e, b in worst[:5]))
        tol = args.tol if args.tol is not None else (
            1e-3 if args.dtype == "f32" else 5e-2)
        bad = float(rel_all.max()) > tol
        print(f"max per-layer rel-norm {rel_all.max():.3e} (tol {tol:g}) -> "
              f"{'FAIL' if bad or not ok else 'PASS'}")
        sys.exit(1 if bad or not ok else 0)


if __name__ == "__main__":
    main()
