"""Krea 2 release kit: streaming safetensors I/O, the ComfyUI fp8_scaled scheme, the diffusers re-shard.

Nothing here holds a whole model in RAM: tensors are read one at a time with os.pread and written with os.pwrite
at offsets fixed by the header, which is computed from (key, dtype, shape) alone before any data is written.
Peak RSS ~ the largest tensor in its source dtype + one f32 chunk (CHUNK_ELEMS) + torch: ~1.5 GB.

ComfyUI fp8_scaled (reproduced bit-exactly from Comfy-Org/Krea-2's krea2_{raw,turbo}_fp8_scaled, see fp8_gate.py):
  - quantized: the 256 linear weights of blocks.N / txtfusion.{layerwise,refiner}_blocks.N (attn wq wk wv wo gate,
    mlp up gate down) -> <layer>.weight F8_E4M3 + <layer>.weight_scale F32 scalar.
    scale = f32(amax|w|) * f32(1/448)  (f32 multiply by the f32 reciprocal: what torch's `amax / 448.0` does on CUDA)
    q = float8_e4m3fn(clamp(f32(w) / scale, -448, 448))  (true f32 division, round-to-nearest-even cast)
  - every other tensor (174: norms, mod.lin, first/last, tmlp/tproj/txtmlp, txtfusion.projector) -> BF16 (RNE cast
    from F32, or a byte copy when the source is already BF16).
  - __metadata__: {"_quantization_metadata": json.dumps({"layers": {layer: {"format": "float8_e4m3fn"[,
    "full_precision_matrix_mult": true for attn.wo / attn.gate / mlp.down]}}} sorted), "format": "pt"}.
  - layout as the `safetensors` library writes it: tensors ordered by dtype (F32, BF16, F8_E4M3), then name;
    compact JSON header padded with spaces to a multiple of 8.
"""
import json
import os
import re

import numpy as np

DSIZE = {"F8_E4M3": 1, "BF16": 2, "F32": 4}
# safetensors (Rust) sorts tensors by dtype descending (its Dtype enum order), then by name
DTYPE_RANK = {"F32": 3, "BF16": 2, "F8_E4M3": 1}
FP8_MAX = 448.0
RECIP_448 = np.float32(1.0) / np.float32(FP8_MAX)
assert RECIP_448 == np.float32(1.0 / FP8_MAX)
CHUNK_ELEMS = 1 << 24
EXPECT_KEYS = 430

QUANT_RE = re.compile(r"^(blocks\.\d+|txtfusion\.(?:layerwise|refiner)_blocks\.\d+)\."
                      r"(attn\.(?:wq|wk|wv|wo|gate)|mlp\.(?:up|gate|down))\.weight$")
FPMM_RE = re.compile(r"\.(attn\.wo|attn\.gate|mlp\.down)$")


class Refuse(Exception):
    """A must-fail check: the caller exits non-zero and writes nothing more."""


def need(cond, msg):
    if not cond:
        raise Refuse(msg)


# ---------------- key map official -> diffusers: copied verbatim from krea2-ops/maxdiff/krea2_keymap.py (r2d) ----
SUB = [
 (r'^blocks\.(\d+)\.', r'transformer_blocks.\1.'),
 (r'^txtfusion\.', 'text_fusion.'),
 (r'\.attn\.wq\.', '.attn.to_q.'), (r'\.attn\.wk\.', '.attn.to_k.'), (r'\.attn\.wv\.', '.attn.to_v.'),
 (r'\.attn\.wo\.', '.attn.to_out.0.'), (r'\.attn\.gate\.', '.attn.to_gate.'),
 (r'\.attn\.qknorm\.qnorm\.scale$', '.attn.norm_q.weight'), (r'\.attn\.qknorm\.knorm\.scale$', '.attn.norm_k.weight'),
 (r'\.mlp\.', '.ff.'), (r'\.prenorm\.scale$', '.norm1.weight'), (r'\.postnorm\.scale$', '.norm2.weight'),
 (r'\.mod\.lin$', '.scale_shift_table'),
 (r'^first\.', 'img_in.'), (r'^last\.linear\.', 'final_layer.linear.'), (r'^last\.modulation\.lin$', 'final_layer.scale_shift_table'),
 (r'^last\.norm\.scale$', 'final_layer.norm.weight'),
]
def raw_to_diffusers(k, tmlp, tproj, txtmlp):
  for a,b in SUB: k = re.sub(a,b,k)
  m = re.match(r'^(tmlp|tproj|txtmlp)\.(\d+)\.(\w+)$', k)
  if m:
    tbl={'tmlp':tmlp,'tproj':tproj,'txtmlp':txtmlp}[m[1]]
    return tbl[m[2]] + '.' + ('weight' if m[3]=='scale' else m[3])
  return k
TMLP={'0':'time_embed.linear_1','2':'time_embed.linear_2'}
TPROJ={'1':'time_mod_proj'}
TXTMLP={'0':'txt_in.norm','1':'txt_in.linear_1','3':'txt_in.linear_2'}
def r2d(k): return raw_to_diffusers(k, TMLP, TPROJ, TXTMLP)
# ---------------- end of the copied key map ----------------
MOD_LIN_RE = re.compile(r"^blocks\.\d+\.mod\.lin$")


# ---------------- safetensors headers ----------------
def numel(shape):
    n = 1
    for x in shape:
        n *= int(x)
    return n


def parse_header_bytes(prefix, file_size=None, what="file"):
    """prefix = 8-byte length + header JSON. Returns (header_len, metadata, {key: (dtype, shape, abs_b, abs_e)})."""
    need(len(prefix) >= 8, f"{what}: shorter than 8 bytes")
    n = int.from_bytes(prefix[:8], "little")
    need(0 < n < 100 << 20 and len(prefix) >= 8 + n, f"{what}: header length {n} vs {len(prefix) - 8} bytes read")
    try:
        j = json.loads(bytes(prefix[8:8 + n]).decode("utf-8"))
    except ValueError as e:
        raise Refuse(f"{what}: header is not JSON ({e})")
    need(isinstance(j, dict), f"{what}: header is not an object")
    meta = j.pop("__metadata__", None) or {}
    out, base = {}, 8 + n
    for k, v in j.items():
        need(isinstance(v, dict) and v.get("dtype") in DSIZE and isinstance(v.get("shape"), list)
             and isinstance(v.get("data_offsets"), list) and len(v["data_offsets"]) == 2, f"{what}: bad entry {k!r}")
        b, e = v["data_offsets"]
        need(0 <= b <= e and e - b == numel(v["shape"]) * DSIZE[v["dtype"]], f"{what}: {k} byte size != numel*itemsize")
        out[k] = (v["dtype"], tuple(v["shape"]), base + b, base + e)
    pos = base
    for k, (_, _, b, e) in sorted(out.items(), key=lambda kv: (kv[1][2], kv[1][3])):
        need(b == pos, f"{what}: data region not tiled at {k}")
        pos = e
    if file_size is not None:
        need(pos == file_size, f"{what}: tensors end at {pos} != file size {file_size}")
    return n, meta, out


def read_header(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        need(0 < n < 100 << 20, f"{path}: implausible header length {n}")
        f.seek(0)
        prefix = f.read(8 + n)
    return parse_header_bytes(prefix, size, path)


def build_header(entries, metadata):
    """entries: [(key, dtype, shape)]. Returns (header bytes incl. the 8-byte length, {key: (abs_b, abs_e)}, file size),
    laid out exactly as the safetensors library does (dtype rank desc, then name; spaces pad to 8)."""
    keys = [k for k, _, _ in entries]
    need(len(set(keys)) == len(keys), "duplicate tensor keys")
    order = sorted(entries, key=lambda e: (-DTYPE_RANK[e[1]], e[0]))
    j, rel, pos = {}, {}, 0
    if metadata:
        j["__metadata__"] = {k: metadata[k] for k in sorted(metadata)}
    for k, dt, shape in order:
        nb = numel(shape) * DSIZE[dt]
        j[k] = {"dtype": dt, "shape": [int(x) for x in shape], "data_offsets": [pos, pos + nb]}
        rel[k] = (pos, pos + nb)
        pos += nb
    h = json.dumps(j, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    h += b" " * ((8 - len(h) % 8) % 8)
    base = 8 + len(h)
    return len(h).to_bytes(8, "little") + h, {k: (base + b, base + e) for k, (b, e) in rel.items()}, base + pos


# ---------------- the fp8_scaled plan ----------------
def quant_metadata(layers):
    lay = {}
    for layer in sorted(layers):
        c = {"format": "float8_e4m3fn"}
        if FPMM_RE.search(layer):
            c["full_precision_matrix_mult"] = True
        lay[layer] = c
    return {"_quantization_metadata": json.dumps({"layers": lay}), "format": "pt"}


def fp8_plan(src_tensors):
    """src_tensors: {key: (dtype, shape, ...)} with official keys. Returns (entries, metadata, jobs) where jobs maps
    each source key to ('quant', layer) or ('bf16', key)."""
    entries, jobs, layers = [], {}, []
    for k, v in src_tensors.items():
        dt, shape = v[0], v[1]
        need(dt in ("BF16", "F32"), f"{k}: source dtype {dt} (BF16 or F32 expected)")
        if QUANT_RE.match(k):
            need(dt == "BF16" and len(shape) == 2, f"{k}: a quantized layer must be a 2-D BF16 weight (got {dt} {shape})")
            layer = k[: -len(".weight")]
            entries += [(k, "F8_E4M3", shape), (layer + ".weight_scale", "F32", ())]
            jobs[k] = ("quant", layer)
            layers.append(layer)
        else:
            entries.append((k, "BF16", shape))
            jobs[k] = ("bf16", k)
    return entries, quant_metadata(layers), jobs


# ---------------- tensor math (torch, CPU) ----------------
def _torch():
    import torch
    return torch


def to_bf16_bytes(dtype, data):
    """F32 -> BF16 (RNE, torch's cast); BF16 -> unchanged bytes."""
    if dtype == "BF16":
        return bytes(data)
    need(dtype == "F32", f"cannot cast {dtype} to BF16")
    torch = _torch()
    t = torch.frombuffer(bytearray(data), dtype=torch.float32)
    return t.to(torch.bfloat16).view(torch.int16).numpy().tobytes()


def quantize_bf16(data, what="tensor"):
    """data: raw BF16 bytes of one weight. Returns (fp8 bytes, f32 scale bytes)."""
    torch = _torch()
    w = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
    n = w.numel()
    amax = 0.0
    for s in range(0, n, CHUNK_ELEMS):
        amax = max(amax, float(w[s:s + CHUNK_ELEMS].abs().max().to(torch.float32)))  # bf16 max is exact
    need(np.isfinite(amax) and amax > 0, f"{what}: amax {amax} (finite and > 0 required)")
    scale = np.float32(amax) * RECIP_448
    st = torch.tensor(float(scale), dtype=torch.float32)
    out = np.empty(n, dtype=np.uint8)
    for s in range(0, n, CHUNK_ELEMS):
        q = (w[s:s + CHUNK_ELEMS].to(torch.float32) / st).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        out[s:s + q.numel()] = q.view(torch.uint8).numpy()
    nan = int(np.count_nonzero((out & 0x7F) == 0x7F))
    need(nan == 0, f"{what}: {nan} NaN codes after quantization")
    return out.tobytes(), np.array([scale], dtype=np.float32).tobytes()


# ---------------- streaming builders ----------------
def pread_exact(fd, b, e):
    buf = bytearray(e - b)
    mv, pos = memoryview(buf), 0
    while pos < len(buf):
        got = os.preadv(fd, [mv[pos:pos + (64 << 20)]], b + pos)
        need(got > 0, f"short read at {b + pos}")
        pos += got
    return buf


def pwrite_all(fd, data, off):
    mv, pos = memoryview(data), 0
    while pos < len(mv):
        pos += os.pwrite(fd, mv[pos:pos + (64 << 20)], off + pos)


def _open_out(path, size):
    tmp = path + ".part"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, size)
    return fd, tmp


def build_fp8(src, out, log=print, expect_keys=EXPECT_KEYS):
    """src: official-key safetensors (BF16/F32). Writes out (ComfyUI fp8_scaled) via out.part + rename."""
    _, _, src_t = read_header(src)
    need(expect_keys is None or len(src_t) == expect_keys, f"{src}: {len(src_t)} tensors, expected {expect_keys}")
    entries, meta, jobs = fp8_plan(src_t)
    hdr, offs, size = build_header(entries, meta)
    fd_in = os.open(src, os.O_RDONLY)
    fd, tmp = _open_out(out, size)
    try:
        pwrite_all(fd, hdr, 0)
        nq = 0
        for i, k in enumerate(sorted(src_t, key=lambda k: src_t[k][2])):   # source file order: sequential reads
            dt, shape, b, e = src_t[k]
            data = pread_exact(fd_in, b, e)
            kind, name = jobs[k]
            if kind == "quant":
                q, s = quantize_bf16(data, k)
                pwrite_all(fd, q, offs[k][0]); pwrite_all(fd, s, offs[name + ".weight_scale"][0])
                nq += 1
            else:
                pwrite_all(fd, to_bf16_bytes(dt, data), offs[k][0])
            if (i + 1) % 50 == 0:
                log(f"FP8_PROGRESS {i + 1}/{len(src_t)}")
        os.fsync(fd)
    finally:
        os.close(fd); os.close(fd_in)
    os.replace(tmp, out)
    log(f"FP8_BUILT {out} bytes={size} tensors={len(entries)} quantized={nq}")
    return size


def diffusers_plan(src_t, weight_map):
    """Per shard: [(diffusers key, dtype, shape, official key)] + the index json text."""
    need(len(set(map(r2d, src_t))) == len(src_t), "r2d is not 1:1 on the source keys")
    need({r2d(k) for k in src_t} == set(weight_map),
         "r2d(source keys) != the index weight_map keys: missing %s extra %s" % (
             sorted(set(weight_map) - {r2d(k) for k in src_t})[:3], sorted({r2d(k) for k in src_t} - set(weight_map))[:3]))
    shards, total = {}, 0
    for k, v in src_t.items():
        dt, shape = v[0], v[1]
        dk = r2d(k)
        if MOD_LIN_RE.match(k):
            need(len(shape) == 1 and shape[0] % 6 == 0, f"{k}: shape {shape} cannot become [6, F]")
            shape = (6, shape[0] // 6)
        shards.setdefault(weight_map[dk], []).append((dk, dt, shape, k))
        total += numel(shape) * DSIZE[dt]
    index = {"metadata": {"total_size": total}, "weight_map": dict(weight_map)}
    return shards, json.dumps(index, indent=2, sort_keys=True) + "\n", total


def build_diffusers(src, weight_map, config_bytes, out_dir, log=print):
    """Re-shard an official-key single file into out_dir/{config.json, index json, shards} (pure byte copy)."""
    _, _, src_t = read_header(src)
    shards, index_text, total = diffusers_plan(src_t, weight_map)
    os.makedirs(out_dir, exist_ok=True)
    fd_in = os.open(src, os.O_RDONLY)
    try:
        for name in sorted(shards):
            ents = shards[name]
            hdr, offs, size = build_header([(dk, dt, sh) for dk, dt, sh, _ in ents], {"format": "pt"})
            fd, tmp = _open_out(os.path.join(out_dir, name), size)
            try:
                pwrite_all(fd, hdr, 0)
                for dk, dt, sh, k in sorted(ents, key=lambda x: src_t[x[3]][2]):
                    need(src_t[k][0] == dt, f"{k}: dtype changed")
                    pwrite_all(fd, pread_exact(fd_in, src_t[k][2], src_t[k][3]), offs[dk][0])
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, os.path.join(out_dir, name))
            log(f"SHARD_BUILT {name} bytes={size} tensors={len(ents)}")
    finally:
        os.close(fd_in)
    for fn, data in (("diffusion_pytorch_model.safetensors.index.json", index_text.encode()), ("config.json", config_bytes)):
        with open(os.path.join(out_dir, fn), "wb") as f:
            f.write(data)
    log(f"DIFFUSERS_BUILT {out_dir} total_size={total} shards={len(shards)}")
    return total


# ---------------- header checks against the Comfy / official references ----------------
def header_prefix(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        f.seek(0)
        return f.read(8 + n)


def check_single(path, ref_prefix, all_bf16):
    """Our single file vs Comfy's bf16 reference: key set + shapes equal; dtypes equal, or all BF16 (Raw = as trained)."""
    _, meta, t = read_header(path)
    _, _, r = parse_header_bytes(ref_prefix, None, "comfy bf16 ref")
    need(set(t) == set(r), f"{path}: key set != Comfy's ({len(t)} vs {len(r)})")
    bad = [k for k in t if t[k][1] != r[k][1]]
    need(not bad, f"{path}: shapes differ from Comfy's: {bad[:3]}")
    if all_bf16:
        bad = [k for k in t if t[k][0] != "BF16"]
    else:
        bad = [k for k in t if t[k][0] != r[k][0]]
    need(not bad, f"{path}: dtypes differ ({'all BF16 expected' if all_bf16 else 'vs Comfy'}): {bad[:3]}")
    return f"SINGLE_OK {os.path.basename(path)} keys={len(t)} dtypes={'all BF16' if all_bf16 else 'as Comfy'} metadata={sorted(meta)}"


def check_fp8(path, ref_prefix, ref_size):
    """Our fp8 file: header bytes identical to Comfy's (keys, dtypes, shapes, offsets, metadata) and the same size."""
    size = os.path.getsize(path)
    read_header(path)
    need(header_prefix(path) == bytes(ref_prefix), f"{path}: header bytes != Comfy's fp8_scaled header")
    need(size == ref_size, f"{path}: size {size} != Comfy's {ref_size}")
    return f"FP8_OK {os.path.basename(path)} header == Comfy's ({len(ref_prefix)} B) size={size}"


def check_diffusers(out_dir, weight_map, expect_total, expect_shard_prefixes=None):
    idx = json.load(open(os.path.join(out_dir, "diffusion_pytorch_model.safetensors.index.json")))
    need(idx.get("weight_map") == weight_map, "index weight_map != the official one")
    need(idx.get("metadata", {}).get("total_size") == expect_total, f"index total_size {idx.get('metadata')} != {expect_total}")
    seen, tot = set(), 0
    for name in sorted(set(weight_map.values())):
        p = os.path.join(out_dir, name)
        _, meta, t = read_header(p)
        need(meta == {"format": "pt"}, f"{name}: metadata {meta}")
        need(set(t) == {k for k, v in weight_map.items() if v == name}, f"{name}: keys != the weight_map's")
        seen |= set(t)
        tot += sum(e - b for _, _, b, e in t.values())
        if expect_shard_prefixes is not None:
            need(header_prefix(p) == expect_shard_prefixes[name], f"{name}: header bytes != the reference shard's")
    need(seen == set(weight_map) and tot == expect_total, f"shards hold {len(seen)} keys / {tot} B")
    return f"DIFFUSERS_OK {out_dir} shards={len(set(weight_map.values()))} keys={len(seen)} total_size={tot}" + (
        " shard headers == reference" if expect_shard_prefixes is not None else "")
