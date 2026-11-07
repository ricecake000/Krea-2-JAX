#!/usr/bin/env python3
"""Must-pass gate (laptop, before any billing): our fp8_scaled builder reproduces Comfy-Org's files bit-exactly.

HTTP range reads of the public Comfy-Org/Krea-2 files at a pinned revision (never a whole file):
  1. headers: build_header(fp8_plan(Comfy bf16 header)) == Comfy's fp8_scaled header bytes (keys, dtypes, shapes,
     offsets, __metadata__ incl. _quantization_metadata), raw and turbo; build_header(Comfy bf16 entries) == Comfy's
     bf16 header bytes (the writer); optional: the diffusers re-shard of Comfy's turbo bf16 header == reference shard
     header bytes (--shard-ref DIR holding <shard name>.hdr = 8-byte length + header JSON, e.g. read from Drive).
  2. tensors: a sample covering every layer kind (blocks.N + txtfusion layerwise/refiner attn wq wk wv wo gate,
     mlp up gate down = both full_precision_matrix_mult and plain; first / last / tmlp / tproj / txtmlp / txtfusion
     projector / norms / mod.lin) is copied from Comfy's bf16 file into a mini safetensors file, run through the REAL
     streaming builder (release_lib.build_fp8), and every output tensor (F8 weight, F32 weight_scale, BF16) must equal
     Comfy's fp8_scaled bytes exactly. Raw and turbo samples use different block indices.
Usage: fp8_gate.py --work DIR [--pairs raw,turbo] [--shard-ref DIR] [--headers-out DIR] [--headers-only]
Prints GATE_* lines; last line FP8_GATE_OK ... (exit 0) or FP8_GATE_FAIL <why> (exit 1).
"""
import argparse, json, os, shutil, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_lib as L  # noqa: E402

REPO, REV = "Comfy-Org/Krea-2", "588feaf955b08b9ac4723eeb751752ab045dba71"
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/diffusion_models/" + "{}.safetensors"
FP8_SIZE = 13141730784
KINDS = ["attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate", "mlp.up", "mlp.gate", "mlp.down"]
NONQ = ["first.weight", "first.bias", "last.linear.weight", "last.linear.bias", "last.modulation.lin", "last.norm.scale",
        "tmlp.0.weight", "tmlp.0.bias", "tmlp.2.bias", "tproj.1.bias", "txtmlp.0.scale", "txtmlp.1.weight",
        "txtmlp.3.bias", "txtfusion.projector.weight"]
SAMPLE = {  # (block, layerwise block, refiner block, extra non-quantized keys)
    "raw": (0, 0, 1, ["txtmlp.1.bias"]),
    "turbo": (27, 1, 0, ["tmlp.2.weight"]),
}


def fetch(name, b, e, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(URL.format(name), headers={"Range": f"bytes={b}-{e - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            L.need(len(data) == e - b, f"{name}: short range read {len(data)} != {e - b}")
            return data
        except (OSError, L.Refuse) as ex:
            if i == tries - 1:
                raise L.Refuse(f"{name}: range read {b}-{e} failed: {ex}")
            time.sleep(5 * (i + 1))


def fetch_prefix(name):
    n = int.from_bytes(fetch(name, 0, 8), "little")
    L.need(0 < n < 10 << 20, f"{name}: header length {n}")
    return fetch(name, 0, 8 + n)


def sample_keys(m, bf16_t):
    blk, lw, rf, extra = SAMPLE[m]
    keys = [f"blocks.{blk}.{k}.weight" for k in KINDS]
    keys += [f"txtfusion.layerwise_blocks.{lw}.{k}.weight" for k in KINDS]
    keys += [f"txtfusion.refiner_blocks.{rf}.{k}.weight" for k in KINDS]
    keys += [f"blocks.{blk}.{k}" for k in ("mod.lin", "prenorm.scale", "postnorm.scale", "attn.qknorm.qnorm.scale",
                                           "attn.qknorm.knorm.scale")]
    keys += [f"txtfusion.layerwise_blocks.{lw}.prenorm.scale", f"txtfusion.refiner_blocks.{rf}.attn.qknorm.knorm.scale"]
    keys += NONQ + extra
    miss = [k for k in keys if k not in bf16_t]
    L.need(not miss, f"sample keys missing from Comfy's header: {miss}")
    return keys


def gate_pair(m, work, say, headers_only=False):
    bname, fname = f"krea2_{m}_bf16", f"krea2_{m}_fp8_scaled"
    bp, fp = fetch_prefix(bname), fetch_prefix(fname)
    _, bmeta, bt = L.parse_header_bytes(bp, None, bname)
    _, fmeta, ft = L.parse_header_bytes(fp, None, fname)
    entries, qmeta, jobs = L.fp8_plan(bt)
    hdr, _, size = L.build_header(entries, qmeta)
    L.need(hdr == fp, f"{m}: our fp8 header != Comfy's ({len(hdr)} vs {len(fp)} B)")
    L.need(size == FP8_SIZE, f"{m}: fp8 size {size} != {FP8_SIZE}")
    L.need(fmeta == qmeta, f"{m}: __metadata__ differs")
    hb, _, _ = L.build_header([(k, v[0], v[1]) for k, v in bt.items()], bmeta)
    L.need(hb == bp, f"{m}: our writer does not reproduce Comfy's bf16 header")
    nq = sum(1 for j in jobs.values() if j[0] == "quant")
    say(f"GATE_HEADER {m} fp8 header == Comfy ({len(fp)} B, {len(ft)} keys: {nq} F8 + {nq} scales + {len(bt) - nq} BF16; "
        f"_quantization_metadata {len(fmeta['_quantization_metadata'])} chars equal) size={size}; bf16 writer ok")
    if headers_only:
        say(f"GATE_TENSORS {m} SKIPPED (--headers-only)")
        return bp, fp
    keys = sample_keys(m, bt)
    # the mini source: Comfy's bf16 bytes of the sample, in a safetensors file written by our writer
    sub = {k: bt[k] for k in keys}
    mh, moffs, msize = L.build_header([(k, v[0], v[1]) for k, v in sub.items()], None)
    src, out = os.path.join(work, f"{m}_mini_src.safetensors"), os.path.join(work, f"{m}_mini_fp8.safetensors")
    fd = os.open(src, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.ftruncate(fd, msize); L.pwrite_all(fd, mh, 0)
        got = 0
        for k in keys:
            L.pwrite_all(fd, fetch(bname, bt[k][2], bt[k][3]), moffs[k][0]); got += bt[k][3] - bt[k][2]
    finally:
        os.close(fd)
    say(f"GATE_FETCHED {m} {len(keys)} bf16 tensors ({got / 2**20:.0f} MiB)")
    L.build_fp8(src, out, log=lambda s: None, expect_keys=None)
    os.remove(src)
    _, _, ot = L.read_header(out)
    fd = os.open(out, os.O_RDONLY)
    kinds, nbytes = {}, 0
    try:
        for k in keys:
            outs = [k] + ([k[:-len(".weight")] + ".weight_scale"] if jobs[k][0] == "quant" else [])
            for ok in outs:
                L.need(ot[ok][0] == ft[ok][0] and ot[ok][1] == ft[ok][1], f"{m} {ok}: dtype/shape != Comfy")
                ours = bytes(L.pread_exact(fd, ot[ok][2], ot[ok][3]))
                ref = fetch(fname, ft[ok][2], ft[ok][3])
                if ours != ref:
                    diff = sum(1 for a, b in zip(ours, ref) if a != b) if len(ours) < (1 << 26) else "many"
                    raise L.Refuse(f"{m} {ok}: bytes != Comfy's ({diff} of {len(ref)} bytes differ)")
                nbytes += len(ref)
                kind = ft[ok][0] + ("/fpmm" if ok.endswith(".weight") and ft[ok][0] == "F8_E4M3" and L.FPMM_RE.search(ok[:-7]) else "")
                kinds[kind] = kinds.get(kind, 0) + 1
    finally:
        os.close(fd)
    os.remove(out)
    say(f"GATE_TENSORS {m} {len(keys)} source tensors -> outputs equal Comfy's bytes exactly: {kinds} ({nbytes / 2**20:.0f} MiB compared)")
    return bp, fp


def gate_shards(bp_turbo, wm, ref_dir, say):
    _, _, t = L.parse_header_bytes(bp_turbo, None, "turbo bf16")
    shards, _, _ = L.diffusers_plan(t, wm)
    for name, ents in sorted(shards.items()):
        h, _, _ = L.build_header([(a, b, c) for a, b, c, _ in ents], {"format": "pt"})
        ref = open(os.path.join(ref_dir, name + ".hdr"), "rb").read()
        L.need(h == ref, f"diffusers re-shard of Comfy turbo bf16: {name} header != reference bytes")
    say(f"GATE_SHARDS turbo re-shard headers == reference shard headers ({len(shards)} shards, bytes equal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pairs", default="raw,turbo")
    ap.add_argument("--shard-ref", help="dir with <shard>.hdr reference prefixes (Turbo diffusers on Drive)")
    ap.add_argument("--index", help="official diffusers index json (weight_map), needed with --shard-ref")
    ap.add_argument("--headers-out", help="save Comfy's 4 header prefixes here (<file>.hdr) for the VM job")
    ap.add_argument("--headers-only", action="store_true", help="header gates only (no tensor reads): NOT the full gate")
    a = ap.parse_args()
    say = lambda s: print(s, flush=True)
    work = os.path.join(a.work, "fp8_gate"); os.makedirs(work, exist_ok=True)
    try:
        say(f"GATE_REF {REPO}@{REV}")
        prefixes = {}
        for m in a.pairs.split(","):
            L.need(m in SAMPLE, f"unknown pair {m}")
            prefixes[m] = gate_pair(m, work, say, a.headers_only)
        if a.shard_ref:
            L.need("turbo" in prefixes and a.index, "--shard-ref needs the turbo pair and --index")
            gate_shards(prefixes["turbo"][0], json.load(open(a.index))["weight_map"], a.shard_ref, say)
        if a.headers_out:
            os.makedirs(a.headers_out, exist_ok=True)
            for m, (bp, fp) in prefixes.items():
                for nm, data in ((f"krea2_{m}_bf16", bp), (f"krea2_{m}_fp8_scaled", fp)):
                    with open(os.path.join(a.headers_out, nm + ".hdr"), "wb") as f:
                        f.write(data)
    except L.Refuse as e:
        say(f"FP8_GATE_FAIL {e}")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    say(f"FP8_GATE_OK pairs={a.pairs} ref={REPO}@{REV[:12]}" + (" HEADERS ONLY (tensor gate skipped)" if a.headers_only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
