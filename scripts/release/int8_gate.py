#!/usr/bin/env python3
"""Must-pass gate: our int8_convrot builder reproduces Comfy-Org's krea2_{raw,turbo}_int8_convrot files.

HTTP range reads of the public Comfy-Org/Krea-2 files at the fp8 gate's pinned revision (never a whole file):
  1. headers: build_header(int8_plan(Comfy bf16 header), empty_meta=True) == Comfy's int8_convrot header bytes (keys,
     dtypes, shapes, offsets, "__metadata__":{}), size == INT8_SIZE; the comfy_quant bytes of 2 reference layers ==
     COMFY_QUANT_INT8.
  2. tensors (unless --headers-only): all 8 quantized linears of one block (raw: blocks.0, turbo: blocks.27; --quick:
     attn.wk + mlp.gate only) + non-quantized tensors of every kind (mod.lin, norms, txtfusion linears/projector,
     first/last, tmlp, txtmlp.0.scale) are copied from Comfy's bf16 file into a mini safetensors file, run through the
     REAL streaming builder (release_lib.build_int8), and compared with Comfy's int8_convrot bytes:
       I8 weight: mismatch rate <= 1e-5 of the elements, max |dq| <= 1;  F32 weight_scale: finite, > 0, max relative
       difference <= 1e-6;  comfy_quant U8 / every BF16 and F32 tensor / dtypes / shapes: exactly equal.
     Bit-exactness is not reachable: the rotation's f32 summation order differs per hardware (measured on the laptop
     CPU: scales equal or 1 ULP apart, ~1e-7 of the int8 values off by 1).
Usage: int8_gate.py --work DIR [--pairs raw,turbo] [--headers-out DIR] [--headers-only] [--device cpu|cuda] [--quick]
Prints GATE_* lines; last line INT8_GATE_OK ... (exit 0) or INT8_GATE_FAIL <why> (exit 1).
"""
import argparse, os, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_lib as L  # noqa: E402
from fp8_gate import KINDS, REPO, REV, fetch, fetch_prefix  # noqa: E402

import numpy as np  # noqa: E402

MAX_MISMATCH_RATE, MAX_DQ, MAX_SCALE_REL = 1e-5, 1, 1e-6
QUICK_KINDS = ["attn.wk", "mlp.gate"]
SAMPLE = {  # (block, extra non-quantized keys: a txtfusion layerwise linear and a refiner linear)
    "raw": (0, ["txtfusion.layerwise_blocks.0.attn.wq.weight", "txtfusion.refiner_blocks.1.mlp.down.weight"]),
    "turbo": (27, ["txtfusion.layerwise_blocks.1.mlp.up.weight", "txtfusion.refiner_blocks.0.attn.wo.weight"]),
}
NONQ = ["txtfusion.projector.weight", "first.weight", "last.linear.weight", "last.norm.scale", "txtmlp.0.scale",
        "tmlp.0.weight"]


def sample_keys(m, bf16_t, quick):
    blk, extra = SAMPLE[m]
    keys = [f"blocks.{blk}.{k}.weight" for k in (QUICK_KINDS if quick else KINDS)]
    keys += [f"blocks.{blk}.{k}" for k in ("mod.lin", "prenorm.scale", "attn.qknorm.qnorm.scale")]
    keys += extra + NONQ
    miss = [k for k in keys if k not in bf16_t]
    L.need(not miss, f"sample keys missing from Comfy's header: {miss}")
    return keys


def compare_quant(m, layer, ours_q, ref_q, ours_s, ref_s):
    """Returns (mismatches, elements, max |dq|, max rel scale diff); Refuse when out of tolerance."""
    a, b = np.frombuffer(ours_q, dtype=np.int8), np.frombuffer(ref_q, dtype=np.int8)
    dq = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mism, mdq = int(np.count_nonzero(dq)), int(dq.max())
    so, sr = np.frombuffer(ours_s, dtype=np.float32), np.frombuffer(ref_s, dtype=np.float32)
    L.need(bool(np.isfinite(so).all() and (so > 0).all() and np.isfinite(sr).all() and (sr > 0).all()),
           f"{m} {layer}: a scale is non-finite or <= 0")
    rel = float((np.abs(so.astype(np.float64) - sr) / sr).max())
    L.need(rel <= MAX_SCALE_REL, f"{m} {layer}: max rel scale diff {rel:.3e} > {MAX_SCALE_REL}")
    L.need(mism <= MAX_MISMATCH_RATE * a.size and mdq <= MAX_DQ,
           f"{m} {layer}: int8 mismatches {mism}/{a.size} max|dq| {mdq} (limits {MAX_MISMATCH_RATE:g}, {MAX_DQ})")
    return mism, a.size, mdq, rel


def gate_pair(m, work, say, headers_only=False, quick=False, device="cpu"):
    bname, iname = f"krea2_{m}_bf16", f"krea2_{m}_int8_convrot"
    bp, ip = fetch_prefix(bname), fetch_prefix(iname)
    _, _, bt = L.parse_header_bytes(bp, None, bname)
    _, imeta, it = L.parse_header_bytes(ip, None, iname)
    entries, jobs = L.int8_plan(bt)
    hdr, _, size = L.build_header(entries, None, empty_meta=True)
    L.need(hdr == ip, f"{m}: our int8 header != Comfy's ({len(hdr)} vs {len(ip)} B)")
    L.need(size == L.INT8_SIZE and max(v[3] for v in it.values()) == L.INT8_SIZE,
           f"{m}: int8 size {size} != {L.INT8_SIZE}")
    blk = SAMPLE[m][0]
    cq = [f"blocks.{blk}.attn.wk.comfy_quant", f"blocks.{27 - blk}.mlp.down.comfy_quant"]
    for k in cq:
        L.need(fetch(iname, it[k][2], it[k][3]) == L.COMFY_QUANT_INT8, f"{m} {k}: bytes != COMFY_QUANT_INT8")
    nq = sum(1 for j in jobs.values() if j[0] == "quant")
    nf = sum(1 for j in jobs.values() if j[0] == "f32")
    say(f"GATE_HEADER {m} int8 header == Comfy ({len(ip)} B, {len(it)} keys: {nq} I8 + {nq} scales + {nq} comfy_quant + "
        f"{nf} F32 + {len(bt) - nq - nf} BF16; __metadata__ {imeta}) size={size}; comfy_quant bytes == ours ({', '.join(cq)})")
    if headers_only:
        say(f"GATE_TENSORS {m} SKIPPED (--headers-only)")
        return bp, ip
    keys = sample_keys(m, bt, quick)
    sub = {k: bt[k] for k in keys}
    mh, moffs, msize = L.build_header([(k, v[0], v[1]) for k, v in sub.items()], None)
    src, out = os.path.join(work, f"{m}_mini_src.safetensors"), os.path.join(work, f"{m}_mini_int8.safetensors")
    fd = os.open(src, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.ftruncate(fd, msize); L.pwrite_all(fd, mh, 0)
        got = 0
        for k in keys:
            L.pwrite_all(fd, fetch(bname, bt[k][2], bt[k][3]), moffs[k][0]); got += bt[k][3] - bt[k][2]
    finally:
        os.close(fd)
    say(f"GATE_FETCHED {m} {len(keys)} bf16 tensors ({got / 2**20:.0f} MiB)")
    t0 = time.time()
    built = []
    L.build_int8(src, out, device=device, log=built.append, expect_keys=None)
    os.remove(src)
    say(f"GATE_BUILD {m} device={device} {time.time() - t0:.0f}s: {built[-1]}")
    _, _, ot = L.read_header(out)
    fd = os.open(out, os.O_RDONLY)
    exact, stats, nbytes = {}, [], 0
    try:
        rd = lambda k: bytes(L.pread_exact(fd, ot[k][2], ot[k][3]))
        for k in keys:
            if jobs[k][0] == "quant":
                layer = jobs[k][1]
                outs = [k, layer + ".weight_scale", layer + ".comfy_quant"]
            else:
                outs = [k]
            for ok in outs:
                L.need(ot[ok][0] == it[ok][0] and ot[ok][1] == it[ok][1], f"{m} {ok}: dtype/shape != Comfy")
            if jobs[k][0] == "quant":
                ref_q, ref_s = fetch(iname, it[k][2], it[k][3]), fetch(iname, it[outs[1]][2], it[outs[1]][3])
                mism, n, mdq, rel = compare_quant(m, layer, rd(k), ref_q, rd(outs[1]), ref_s)
                stats.append(f"{layer.split('.', 2)[2]}:{mism}/{n},dq{mdq},srel{rel:.2e}")
                nbytes += len(ref_q) + len(ref_s)
                outs = outs[2:]
            for ok in outs:
                ref = fetch(iname, it[ok][2], it[ok][3])
                ours = rd(ok)
                if ours != ref:
                    diff = sum(1 for a, b in zip(ours, ref) if a != b) if len(ours) < (1 << 26) else "many"
                    raise L.Refuse(f"{m} {ok}: bytes != Comfy's ({diff} of {len(ref)} bytes differ)")
                nbytes += len(ref)
                exact[it[ok][0]] = exact.get(it[ok][0], 0) + 1
    finally:
        os.close(fd)
    os.remove(out)
    say(f"GATE_TENSORS {m} blocks.{SAMPLE[m][0]} quantized (mismatches/elements, max|dq|, max rel scale diff): "
        f"{' '.join(stats)}; byte-exact: {exact} ({nbytes / 2**20:.0f} MiB compared)")
    return bp, ip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pairs", default="raw,turbo")
    ap.add_argument("--headers-out", help="save Comfy's header prefixes here (krea2_{m}_{int8_convrot,bf16}.hdr)")
    ap.add_argument("--headers-only", action="store_true", help="header gates only (no tensor reads): NOT the full gate")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--quick", action="store_true", help="2 quantized kinds per block (attn.wk, mlp.gate) instead of 8")
    a = ap.parse_args()
    say = lambda s: print(s, flush=True)
    if not a.headers_only:
        import torch
        if a.device == "cpu":
            torch.set_num_threads(len(os.sched_getaffinity(0)))
    work = os.path.join(a.work, "int8_gate"); os.makedirs(work, exist_ok=True)
    t0 = time.time()
    try:
        say(f"GATE_REF {REPO}@{REV}")
        prefixes = {}
        for m in a.pairs.split(","):
            L.need(m in SAMPLE, f"unknown pair {m}")
            prefixes[m] = gate_pair(m, work, say, a.headers_only, a.quick, a.device)
        if a.headers_out:
            os.makedirs(a.headers_out, exist_ok=True)
            for m, (bp, ip) in prefixes.items():
                for nm, data in ((f"krea2_{m}_bf16", bp), (f"krea2_{m}_int8_convrot", ip)):
                    with open(os.path.join(a.headers_out, nm + ".hdr"), "wb") as f:
                        f.write(data)
    except Exception as e:  # noqa: BLE001 - Refuse, network, torch (e.g. CUDA OOM): one FAIL line, exit 1
        say(f"INT8_GATE_FAIL {type(e).__name__}: {e}")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    mode = " HEADERS ONLY (tensor gate skipped)" if a.headers_only else (" QUICK" if a.quick else " FULL")
    say(f"INT8_GATE_OK pairs={a.pairs} ref={REPO}@{REV[:12]} device={a.device}{mode} {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
