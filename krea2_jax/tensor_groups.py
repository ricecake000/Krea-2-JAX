"""Kind and block group of an official-layout Krea 2 tensor key (stdlib only:
importable by the Modal scripts' local entrypoints without numpy)."""

import re

ATTN_KINDS = {"wq": "attn_q", "wk": "attn_k", "wv": "attn_v", "wo": "attn_o",
              "gate": "attn_gate", "qknorm": "norm"}
KINDS = ("attn_q", "attn_k", "attn_v", "attn_o", "attn_gate", "mlp", "norm",
         "modulation", "embed_final")
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def tensor_kind(key):
    """Official Krea 2 key -> one of KINDS."""
    parts = key.split(".")
    if "attn" in parts[:-1]:
        kind = ATTN_KINDS.get(parts[parts.index("attn") + 1])
        if kind:
            return kind
    if "mlp" in parts:
        return "mlp"
    if parts[-1] == "scale" or any(p.endswith("norm") for p in parts):
        return "norm"
    if "mod" in parts or "modulation" in parts:
        return "modulation"
    return "embed_final"   # first, last.linear, tmlp, tproj, txtmlp, projector


def tensor_group(key):
    """'blockNN' for DiT block NN, 'txtfusion' for the text fusion blocks,
    'other' for the rest (embeddings, final layer)."""
    m = _BLOCK_RE.match(key)
    if m:
        return f"block{int(m.group(1)):02d}"
    return "txtfusion" if key.startswith("txtfusion.") else "other"
