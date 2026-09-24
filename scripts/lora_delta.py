"""Weight-drift analysis for Krea 2 LoRA checkpoints (CPU, torch).

Two modes:

  # once per base checkpoint: extract per-linear ||W||_F (a few hundred floats)
  python scripts/lora_delta.py base-norms raw.safetensors -o base_norms.json

  # per LoRA file: ||scale*B@A||_F, relative drift, effective rank
  python scripts/lora_delta.py delta lora_out/*.safetensors \
      --base-norms base_norms.json --csv drift.csv

The base checkpoint is only needed for the *relative* number; comparing two
runs with identical rank/targets works off the LoRA files alone, since the
denominator is the same for both.

Effective rank is the participation ratio exp(H) of the singular-value
distribution of B@A, computed without forming the (out, in) product: QR of B
gives BA = Q (R @ A), and R @ A is (r, in), so svdvals is cheap.
"""

import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

# module path -> coarse group, for a breakdown that mirrors the architecture
GROUPS = (
    ("txtfusion", re.compile(r"^txtfusion\.")),
    ("attn", re.compile(r"\.attn\.")),
    ("mlp", re.compile(r"\.mlp\.")),
)


def group_of(module):
    for name, pat in GROUPS:
        if pat.search(module):
            return name
    return "other"


def base_norms(ckpt, out_path):
    """||W||_F for every 2-D .weight tensor, streamed one tensor at a time."""
    norms = {}
    with safe_open(str(ckpt), framework="pt") as f:
        for key in f.keys():
            if not key.endswith(".weight"):
                continue
            t = f.get_tensor(key)
            if t.ndim != 2:
                continue
            norms[key[: -len(".weight")]] = {
                "norm": float(torch.linalg.matrix_norm(t.float())),
                "shape": list(t.shape),
            }
    Path(out_path).write_text(json.dumps(norms, indent=1))
    print(f"wrote {len(norms)} linear norms -> {out_path}")


def eff_rank(delta_small):
    """Participation ratio of the singular values of an (r, in) matrix."""
    s = torch.linalg.svdvals(delta_small.float())
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def delta_one(path, base=None):
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        rank = int(meta.get("rank", 0)) or None
        alpha = float(meta.get("alpha", rank or 1))
        scale = alpha / rank if rank else 1.0
        modules = sorted(
            k[: -len(".lora_A.weight")]
            for k in f.keys()
            if k.endswith(".lora_A.weight")
        )
        rows = []
        for m in modules:
            a = f.get_tensor(f"{m}.lora_A.weight").float()  # (r, in)
            b = f.get_tensor(f"{m}.lora_B.weight").float()  # (out, r)
            _, r_b = torch.linalg.qr(b)
            small = scale * (r_b @ a)  # same singular values as scale*B@A
            fro = float(torch.linalg.matrix_norm(small))
            row = {
                "module": m,
                "group": group_of(m),
                "delta": fro,
                "eff_rank": eff_rank(small),
            }
            if base and m in base:
                row["base"] = base[m]["norm"]
                row["rel"] = fro / base[m]["norm"]
            rows.append(row)
    return meta, rows


def summarize(name, meta, rows):
    total = math.sqrt(sum(r["delta"] ** 2 for r in rows))
    mean_eff = sum(r["eff_rank"] for r in rows) / len(rows)
    rel = [r["rel"] for r in rows if "rel" in r]
    out = {
        "name": name,
        "step": int(meta.get("step", 0)),
        "modules": len(rows),
        "total_delta": total,
        "mean_eff_rank": mean_eff,
    }
    if rel:
        out["mean_rel"] = sum(rel) / len(rel)
        out["max_rel"] = max(rel)
    for g, _ in GROUPS:
        sub = [r["delta"] for r in rows if r["group"] == g]
        if sub:
            out[f"delta_{g}"] = math.sqrt(sum(d * d for d in sub))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    bn = sub.add_parser("base-norms", help="extract ||W||_F from a base ckpt")
    bn.add_argument("ckpt")
    bn.add_argument("-o", "--out", default="base_norms.json")

    dl = sub.add_parser("delta", help="drift of one or more LoRA files")
    dl.add_argument("lora", nargs="+")
    dl.add_argument("--base-norms", default=None)
    dl.add_argument("--csv", default=None)
    dl.add_argument("--per-module", default=None,
                    help="write the full per-module table as JSON")

    args = ap.parse_args()
    if args.cmd == "base-norms":
        return base_norms(args.ckpt, args.out)

    base = json.loads(Path(args.base_norms).read_text()) if args.base_norms else None
    summaries, per_module = [], {}
    for p in sorted(args.lora):
        meta, rows = delta_one(p, base)
        s = summarize(Path(p).stem, meta, rows)
        per_module[Path(p).stem] = rows
        summaries.append(s)
        rel = f"  rel {s['mean_rel'] * 100:.3f}%" if "mean_rel" in s else ""
        print(f"{s['name']:<28} step {s['step']:>5}  "
              f"|dW| {s['total_delta']:>9.2f}{rel}  "
              f"eff_rank {s['mean_eff_rank']:.1f}", flush=True)

    if args.csv:
        cols = sorted({k for s in summaries for k in s})
        cols = ["name", "step"] + [c for c in cols if c not in ("name", "step")]
        lines = [",".join(cols)]
        for s in summaries:
            lines.append(",".join(str(s.get(c, "")) for c in cols))
        Path(args.csv).write_text("\n".join(lines) + "\n")
        print(f"wrote {args.csv}")
    if args.per_module:
        Path(args.per_module).write_text(json.dumps(per_module))
        print(f"wrote {args.per_module}")


if __name__ == "__main__":
    main()
