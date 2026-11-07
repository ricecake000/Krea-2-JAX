#!/usr/bin/env python3
"""VM-side CLI of the Krea 2 release kit (torch + numpy; streams tensors, see release_lib).

  release.py fp8 --src SINGLE --out FP8                         ComfyUI fp8_scaled file from an official-key single file
  release.py diffusers --src SINGLE --index IDX --config CFG --out DIR
                                                                transformer/ (3 shards with IDX's key->shard map)
  release.py check-single --file F --ref-hdr H [--all-bf16]     key set / shapes / dtypes vs Comfy's bf16 header
  release.py check-fp8 --file F --ref-hdr H --ref-size N        header bytes == Comfy's fp8_scaled header, same size
  release.py check-diffusers --dir D --index IDX --expect-total N [--shard-ref DIR]
Every command prints one *_OK line (exit 0) or RELEASE_REFUSED <why> (exit 2).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_lib as L  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("fp8"); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("diffusers")
    for k in ("--src", "--index", "--config", "--out"):
        p.add_argument(k, required=True)
    p = sp.add_parser("check-single"); p.add_argument("--file", required=True); p.add_argument("--ref-hdr", required=True)
    p.add_argument("--all-bf16", action="store_true")
    p = sp.add_parser("check-fp8"); p.add_argument("--file", required=True); p.add_argument("--ref-hdr", required=True)
    p.add_argument("--ref-size", type=int, required=True)
    p = sp.add_parser("check-diffusers"); p.add_argument("--dir", required=True); p.add_argument("--index", required=True)
    p.add_argument("--expect-total", type=int, required=True); p.add_argument("--shard-ref")
    a = ap.parse_args(argv)
    say = lambda s: print(s, flush=True)
    try:
        if a.cmd == "fp8":
            L.build_fp8(a.src, a.out, log=say)
        elif a.cmd == "diffusers":
            wm = json.load(open(a.index))["weight_map"]
            L.build_diffusers(a.src, wm, open(a.config, "rb").read(), a.out, log=say)
        elif a.cmd == "check-single":
            say(L.check_single(a.file, open(a.ref_hdr, "rb").read(), a.all_bf16))
        elif a.cmd == "check-fp8":
            say(L.check_fp8(a.file, open(a.ref_hdr, "rb").read(), a.ref_size))
        else:
            wm = json.load(open(a.index))["weight_map"]
            ref = None
            if a.shard_ref:
                ref = {n: open(os.path.join(a.shard_ref, n + ".hdr"), "rb").read() for n in set(wm.values())}
            say(L.check_diffusers(a.dir, wm, a.expect_total, ref))
    except (L.Refuse, OSError, ValueError, KeyError) as e:
        say(f"RELEASE_REFUSED {a.cmd}: {type(e).__name__}: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
