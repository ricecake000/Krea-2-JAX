"""Release kit: fp8_scaled numerics + header layout (the bit-exact Comfy gate is scripts/release/fp8_gate.py)."""
import json, os, sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "release"))
import release_lib as L  # noqa: E402


def test_quantize_scale_and_bytes():
    w = torch.tensor([0.43359375, -0.2, 1e-3, 0.0], dtype=torch.bfloat16)
    q, s = L.quantize_bf16(w.view(torch.int16).numpy().tobytes())
    scale = np.frombuffer(s, np.float32)[0]
    assert scale == np.float32(0.43359375) * np.float32(1 / 448) != np.float32(0.43359375) / np.float32(448)
    ref = (w.float() / torch.tensor(scale)).to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
    assert q == ref and q[0] == 0x7E  # amax -> 448
    with pytest.raises(L.Refuse):
        L.quantize_bf16(torch.zeros(4, dtype=torch.bfloat16).view(torch.int16).numpy().tobytes())


def test_fp8_plan_layout():
    src = {"blocks.0.attn.wo.weight": ("BF16", (8, 8)), "blocks.0.attn.wq.weight": ("BF16", (8, 8)),
           "blocks.0.mod.lin": ("F32", (12,)), "first.weight": ("F32", (8, 4))}
    ents, meta, _ = L.fp8_plan(src)
    layers = json.loads(meta["_quantization_metadata"])["layers"]
    assert layers == {"blocks.0.attn.wo": {"format": "float8_e4m3fn", "full_precision_matrix_mult": True},
                      "blocks.0.attn.wq": {"format": "float8_e4m3fn"}}
    hdr, offs, size = L.build_header(ents, meta)
    order = [k for k in json.loads(hdr[8:]) if k != "__metadata__"]
    assert order[:2] == ["blocks.0.attn.wo.weight_scale", "blocks.0.attn.wq.weight_scale"]  # F32, BF16, F8
    assert order[-2:] == ["blocks.0.attn.wo.weight", "blocks.0.attn.wq.weight"] and len(hdr) % 8 == 0


def test_strip_metadata_inplace(tmp_path):
    p = str(tmp_path / "m.safetensors")
    hdr, offs, size = L.build_header([("a", "F32", (2,))], {"ft": "/vol/exports/x step 1", "alpha": "1.0", "raw": "krea/K@1"})
    open(p, "wb").write(hdr + b"\x01" * 8)
    assert L.strip_metadata_inplace(p, log=lambda s: None) == ["ft"]
    n, meta, t = L.read_header(p)
    assert meta == {"alpha": "1.0", "raw": "krea/K@1"} and os.path.getsize(p) == size and open(p, "rb").read()[-8:] == b"\x01" * 8
