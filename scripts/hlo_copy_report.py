#!/usr/bin/env python3
"""Count the slice / dynamic-slice / dynamic-update-slice / copy instructions
in an XLA optimized HLO text dump (any backend), weighted by while-loop trip
counts.

    python scripts/hlo_copy_report.py module_0042.jit_fwdbwd.gpu_after_optimizations.txt
    python scripts/hlo_copy_report.py FILE --json out.json --top 30 --source-filter model.py

FILE may be gzip-compressed (*.gz).
Input: a `*after_optimizations.txt` file from
`XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_as_text --xla_dump_hlo_module_re=fwdbwd"`.
Stdlib only; lines that do not parse are skipped.

Copy-like instructions
  * plain ops: copy, copy-start, dynamic-slice, dynamic-update-slice, slice
    (copy-done is not counted, its copy-start is);
  * fusions whose fused computation's ROOT is one of those ops, looking
    through single-operand bitcast / convert / reshape / copy / transpose
    chains (convert(dynamic-slice(..)) is a dynamic-slice fusion) and tuple
    roots (counted when every element is copy-like; each element is
    resolved and classified on its own). Kind is then "fusion/<op>", or for
    a multi-element tuple root e.g. "fusion/dynamic-update-slice x2" /
    "fusion/dynamic-update-slice x2+slice"; fusions that do real compute on
    the slice (multiply,
    reduce, dot, ...) are not counted. On GPU these are the
    `loop_dynamic_slice_fusion.N` / `loop_dynamic_update_slice_fusion.N` /
    `*_copy_fusion` kernels that the profiler shows as memcpy / copyBuffer.
  Instructions inside fused computations are never counted on their own (they
  are part of their fusion).

Numbers
  * out_bytes   = bytes of the instruction's output shape (tuple -> sum;
                  copy-start -> its first tuple element).
  * bytes       = bytes the instruction actually moves: out_bytes, except a
                  dynamic-update-slice (plain or fusion root) counts its UPDATE
                  operand (the op is in-place; its output is the whole stacked
                  buffer, e.g. [layers, ...]). A tuple-rooted fusion sums
                  its elements: DUS elements their update operand, the
                  others their element's output shape.  All totals use this.
  * trip multiplier of a computation: ENTRY = 1; a while body / condition =
    the while's known_trip_count x the multiplier of the computation holding
    the while (nested loops multiply); other called computations inherit the
    caller's multiplier (max over callers, flagged when called from >1 site).
    A while without known_trip_count counts as 1 (flagged).
  * static  = number of instructions in the HLO text;
    dynamic = sum of the trip multipliers (executions per call of the module);
    bytes/exec = sum of bytes x multiplier (per call of the module).
  A training step calls jit_fwdbwd once, so "dynamic" ~ copy kernels per step
  (GPU) when every copy-like instruction lowers to one kernel; XLA may still
  fuse/alias some away (e.g. a DS feeding a dot directly), so treat it as an
  upper bound and compare with the profiler's memcpy count.

The last line is grep-able:
    copies: static=N dynamic=M bytes_per_exec=X MiB

    python scripts/hlo_copy_report.py --self-test   # built-in check, no file
"""

import argparse
import collections
import gzip
import json
import re
import sys

COPY_OPS = ("copy", "copy-start", "dynamic-slice", "dynamic-update-slice",
            "slice")
DTYPE_BYTES = {
    "pred": 1, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4, "u32": 4,
    "s64": 8, "u64": 8, "f16": 2, "bf16": 2, "f32": 4, "f64": 8, "c64": 8,
    "c128": 16, "s4": 0.5, "u4": 0.5, "s2": 0.25, "u2": 0.25, "token": 0,
    "f8e4m3fn": 1, "f8e5m2": 1, "f8e4m3fnuz": 1, "f8e5m2fnuz": 1,
    "f8e4m3b11fnuz": 1, "f8e4m3": 1, "f8e3m4": 1, "f4e2m1fn": 0.5,
    "f8e8m0fnu": 1,
}
CALL_ATTRS = ("calls", "to_apply", "condition", "body", "true_computation",
              "false_computation")
LIST_CALL_ATTRS = ("branch_computations", "called_computations")

COMMENT_RE = re.compile(r"/\*.*?\*/")
HEADER_RE = re.compile(r"^(ENTRY\s+)?%?([\w.\-]+)\s*(\(|\{)")
INSTR_RE = re.compile(r"^\s+(ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$")
OPCODE_RE = re.compile(r"^\s*([\w\-]+)\(")
ARRAY_RE = re.compile(r"\b([a-z]+[0-9a-z]*)\[([^\]]*)\]")
TRIP_RE = re.compile(r'"known_trip_count"\s*:\s*\{\s*"n"\s*:\s*"(\d+)"')
OPNAME_RE = re.compile(r'op_name="([^"]*)"')
SRCFILE_RE = re.compile(r'source_file="([^"]*)"')
SRCLINE_RE = re.compile(r"source_line=(\d+)")
TARGET_RE = re.compile(r'custom_call_target="([^"]*)"')
CALL_RE = re.compile(r"\b(" + "|".join(CALL_ATTRS) + r")=%?([\w.\-]+)")
LIST_CALL_RE = re.compile(r"\b(" + "|".join(LIST_CALL_ATTRS) + r")=\{([^}]*)\}")


# ------------------------------------------------------------------ parsing
def _balanced(s, i):
    """s[i] is '(' -> index just past its matching ')' (or len(s))."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def _split_top(s):
    out, depth, cur = [], 0, []
    for c in s:
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        if c == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def shape_arrays(shape):
    """[(dtype, dims)] of every array in a (possibly tuple) shape string."""
    out = []
    for dt, dims in ARRAY_RE.findall(shape):
        if dt not in DTYPE_BYTES:
            continue
        ds = []
        for d in dims.split(","):
            d = d.strip().lstrip("<=")
            if not d:
                continue
            try:
                ds.append(int(d))
            except ValueError:
                ds = None
                break
        if ds is not None:
            out.append((dt, ds))
    return out


def array_bytes(dt, dims):
    n = 1
    for d in dims:
        n *= d
    return int(n * DTYPE_BYTES.get(dt, 0))


def shape_bytes(shape, first_only=False):
    arrs = shape_arrays(shape)
    if first_only:
        arrs = arrs[:1]
    return sum(array_bytes(dt, ds) for dt, ds in arrs)


def short_shape(shape):
    arrs = shape_arrays(shape)
    if not arrs:
        return shape.split("{")[0][:60]
    s = ", ".join(f"{dt}[{','.join(map(str, ds))}]" for dt, ds in arrs[:3])
    if len(arrs) > 3:
        s += f", ... ({len(arrs)} arrays)"
    return f"({s})" if shape.lstrip().startswith("(") else s


def parse_instr(line):
    m = INSTR_RE.match(line)
    if not m:
        return None
    root, name, rest = bool(m.group(1)), m.group(2), m.group(3)
    rest = rest.strip()
    if rest.startswith("("):
        end = _balanced(rest, 0)
        shape, rest = rest[:end], rest[end:]
    else:
        sp = rest.find(" ")
        if sp < 0:
            return None
        shape, rest = rest[:sp], rest[sp:]
    mo = OPCODE_RE.match(rest)
    if not mo:
        return None
    opcode = mo.group(1)
    p0 = rest.index("(", mo.start(1))
    p1 = _balanced(rest, p0)
    operand_str, attrs = rest[p0 + 1:p1 - 1], rest[p1:]
    operands = []
    for op in _split_top(operand_str):
        mm = re.search(r"%?([\w.\-]+)\s*$", op)
        if mm:
            operands.append(mm.group(1))
    callees = []
    for attr, comp in CALL_RE.findall(attrs):
        callees.append((attr, comp))
    for attr, lst in LIST_CALL_RE.findall(attrs):
        for c in lst.split(","):
            c = c.strip().lstrip("%")
            if c:
                callees.append((attr, c))
    meta = {}
    mm = OPNAME_RE.search(attrs)
    if mm:
        meta["op_name"] = mm.group(1)
    mm = SRCFILE_RE.search(attrs)
    if mm:
        meta["source_file"] = mm.group(1)
    mm = SRCLINE_RE.search(attrs)
    if mm:
        meta["source_line"] = int(mm.group(1))
    trip = None
    if opcode == "while":
        mm = TRIP_RE.search(attrs)
        if mm:
            trip = int(mm.group(1))
    mm = TARGET_RE.search(attrs)
    return {"name": name, "root": root, "shape": shape, "opcode": opcode,
            "operands": operands, "callees": callees, "meta": meta,
            "trip": trip, "target": mm.group(1) if mm else None}


def parse_module(text):
    """-> (computations {name: {"entry", "instrs": [..], "by_name": {..}}},
    order [names in file order], entry name, n_skipped_lines)."""
    comps, order, entry, cur, skipped = {}, [], None, None, 0
    for raw in text.splitlines():
        line = COMMENT_RE.sub("", raw).rstrip()
        if not line.strip():
            continue
        if cur is None:
            if line.startswith("HloModule"):
                continue
            if not line[0].isspace() and line.endswith("{"):
                m = HEADER_RE.match(line)
                if m:
                    name = m.group(2)
                    cur = {"name": name, "entry": bool(m.group(1)),
                           "instrs": [], "by_name": {}}
                    comps[name] = cur
                    order.append(name)
                    if cur["entry"]:
                        entry = name
                    continue
            skipped += 1
            continue
        if line.strip() == "}" and not line[0].isspace():
            cur = None
            continue
        try:
            ins = parse_instr(line)
        except Exception:  # noqa: BLE001 -- never crash on odd lines
            ins = None
        if ins is None:
            skipped += 1
            continue
        ins["comp"] = cur["name"]
        cur["instrs"].append(ins)
        cur["by_name"][ins["name"]] = ins
    if entry is None and order:
        entry = order[-1]
    return comps, order, entry, skipped


# ----------------------------------------------------------------- analysis
def compute_multipliers(comps, entry):
    """-> ({comp: multiplier}, {comp: flags}, fused set, while list)."""
    incoming = collections.defaultdict(list)   # callee -> [(caller, factor)]
    fused, whiles, flags = set(), [], collections.defaultdict(set)
    for cname, c in comps.items():
        for ins in c["instrs"]:
            for attr, callee in ins["callees"]:
                if callee not in comps:
                    continue
                factor = 1
                if ins["opcode"] == "while" and attr in ("body", "condition"):
                    if ins["trip"] is None:
                        flags[callee].add("unknown_trip_count")
                    factor = ins["trip"] if ins["trip"] is not None else 1
                if ins["opcode"] == "fusion" and attr == "calls":
                    fused.add(callee)
                incoming[callee].append((cname, factor))
            if ins["opcode"] == "while":
                whiles.append(ins)
    mult, visiting = {}, set()
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    def get(c):
        if c in mult:
            return mult[c]
        if c == entry:
            mult[c] = 1
            return 1
        if c in visiting:     # malformed (cycle): break it
            return 1
        visiting.add(c)
        calls = incoming.get(c, [])
        if not calls:
            flags[c].add("unreferenced")
            m = 1
        else:
            vals = [get(caller) * f for caller, f in calls]
            m = max(vals)
            if len(calls) > 1:
                flags[c].add(f"called_from_{len(calls)}_sites")
        visiting.discard(c)
        mult[c] = m
        return m

    for c in comps:
        get(c)
    return mult, flags, fused, whiles


def _root(comp):
    for ins in comp["instrs"]:
        if ins["root"]:
            return ins
    return comp["instrs"][-1] if comp["instrs"] else None


PASS_THROUGH = ("bitcast", "convert", "reshape", "copy", "transpose")


def _resolve_parts(comp, ins, depth=0):
    """[(kind, op_instr, elem_instr)] for every copy-like element ins boils
    down to, or None. Walks down single-operand layout/dtype ops (bitcast,
    convert, reshape, copy, transpose): convert(dynamic-slice(..)) is a
    dynamic-slice (the CPU backend wraps every bf16 slice in converts),
    copy(param) a copy. A tuple yields the parts of all its elements (None
    unless every element is copy-like). kind is a COPY_OPS member, op_instr
    the copy-like op itself, elem_instr the top of its pass-through chain
    (whose shape is the element's output)."""
    if ins is None or depth > 16:
        return None
    op = ins["opcode"]
    if op in ("dynamic-slice", "dynamic-update-slice", "slice", "copy-start"):
        return [(op, ins, ins)]
    if op in PASS_THROUGH:
        sub = (_resolve_parts(comp, comp["by_name"].get(ins["operands"][0]),
                              depth + 1) if ins["operands"] else None)
        if sub:
            if len(sub) == 1:
                return [(sub[0][0], sub[0][1], ins)]
            return sub
        return [("copy", ins, ins)] if op == "copy" else None
    if op == "tuple" and ins["operands"]:
        parts = []
        for o in ins["operands"]:
            sub = _resolve_parts(comp, comp["by_name"].get(o), depth + 1)
            if not sub:
                return None
            parts += sub
        return parts
    return None


def _parts_kind(parts):
    """'dynamic-update-slice' (one element), 'dynamic-update-slice x2',
    'dynamic-update-slice x2+slice' (sorted, mixed)."""
    if len(parts) == 1:
        return parts[0][0]
    cnt = collections.Counter(k for k, _, _ in parts)
    return "+".join(f"{k} x{n}" if n > 1 else k for k, n in sorted(cnt.items()))


def _resolve_copy_root(comp, ins, depth=0):
    """The copy-like op(s) a fused computation's ROOT boils down to, or
    None. Returns (kind, first op instr, parts) with parts as in
    _resolve_parts and kind from _parts_kind."""
    parts = _resolve_parts(comp, ins, depth)
    if not parts:
        return None
    return _parts_kind(parts), parts[0][1], parts


def _decorate(comps, ins):
    """Opcode label of a producer/user: fusion -> fusion:<root op>,
    custom-call -> custom-call:<target>."""
    if ins is None:
        return "?"
    op = ins["opcode"]
    if op == "fusion":
        for attr, callee in ins["callees"]:
            if attr == "calls" and callee in comps:
                r = _root(comps[callee])
                return f"fusion:{r['opcode']}" if r else "fusion"
        return "fusion"
    if op == "custom-call" and ins["target"]:
        return f"custom-call:{ins['target']}"
    if op == "get-tuple-element" and ins["operands"]:
        return "get-tuple-element"
    return op


def _dus_update_bytes(comp, dus):
    """Bytes of a dynamic-update-slice's update operand (None if unknown)."""
    if dus is None or len(dus["operands"]) < 2:
        return None
    upd = comp["by_name"].get(dus["operands"][1])
    return shape_bytes(upd["shape"]) if upd is not None else None


def _bytes_moved(comp_of_root, kind, root_ins, out_bytes, parts=None):
    """DUS moves its update operand; everything else its output. With
    parts (a fusion's resolved root elements, _resolve_parts): one element
    keeps the fusion's output bytes (or the DUS update bytes); several are
    summed element by element (DUS: update operand; slice / DS / copy: the
    element's output shape; copy-start: its first tuple element)."""
    if parts is not None and len(parts) > 1:
        total = 0
        for k, op_ins, elem in parts:
            b = (_dus_update_bytes(comp_of_root, op_ins)
                 if k == "dynamic-update-slice" else None)
            if b is None:
                b = shape_bytes(elem["shape"], first_only=(k == "copy-start"))
            total += b
        return total
    if parts is not None:
        kind, root_ins = parts[0][0], parts[0][1]
    if kind == "dynamic-update-slice":
        b = _dus_update_bytes(comp_of_root, root_ins)
        if b is not None:
            return b
    return out_bytes


def jax_op_tail(op_name):
    if not op_name:
        return ""
    parts = op_name.split("/")
    return "/".join(parts[-2:])


def analyze(text, source_filter=None):
    comps, order, entry, skipped = parse_module(text)
    mult, flags, fused, whiles = compute_multipliers(comps, entry)
    users = {}
    for cname, c in comps.items():
        if cname in fused:
            continue
        u = collections.defaultdict(list)
        for ins in c["instrs"]:
            for o in ins["operands"]:
                u[o].append(ins)
        users[cname] = u

    items = []
    for cname in order:
        if cname in fused:
            continue
        c = comps[cname]
        for ins in c["instrs"]:
            op = ins["opcode"]
            kind, root_ins, root_comp, parts = None, None, c, None
            if op in COPY_OPS:
                kind, root_ins = op, ins
            elif op == "fusion":
                callee = next((cc for a, cc in ins["callees"] if a == "calls"),
                              None)
                if callee in comps:
                    res = _resolve_copy_root(comps[callee], _root(comps[callee]))
                    if res:
                        kind = "fusion/" + res[0]
                        root_ins, root_comp = res[1], comps[callee]
                        parts = res[2]
            if kind is None:
                continue
            meta = ins["meta"]
            if "op_name" not in meta and "source_file" not in meta \
                    and root_ins is not None:
                meta = root_ins["meta"]   # CPU fusions carry no metadata
            src = meta.get("source_file", "")
            if source_filter and source_filter not in src:
                continue
            out_b = shape_bytes(ins["shape"], first_only=(op == "copy-start"))
            base_kind = kind.split("/", 1)[-1]
            moved = _bytes_moved(root_comp, base_kind, root_ins, out_b,
                                 parts)
            prod = None
            for o in ins["operands"]:
                p = c["by_name"].get(o)
                if p is not None and shape_arrays(p["shape"]) and \
                        p["opcode"] != "constant":
                    prod = p
                    break
            us = users[cname].get(ins["name"], [])
            m = mult.get(cname, 1)
            items.append({
                "computation": cname, "name": ins["name"], "kind": kind,
                "opcode": op, "shape": short_shape(ins["shape"]),
                "out_bytes": out_b, "bytes": moved, "multiplier": m,
                "bytes_per_exec": moved * m,
                "jax_op": jax_op_tail(meta.get("op_name", "")),
                "source": (f"{src.rsplit('/', 1)[-1]}:"
                           f"{meta.get('source_line', '?')}"
                           if src else ""),
                "source_file": src,
                "producer": _decorate(comps, prod) if prod else "-",
                "users": [_decorate(comps, x) for x in us[:4]]
                         + (["..."] if len(us) > 4 else []),
            })

    # while loops: body (+ condition) and the non-fused, non-loop computations
    # they call transitively
    callees_of = {cname: [(a, cc) for ins in c["instrs"]
                          for a, cc in ins["callees"]
                          if cc in comps and ins["opcode"] != "while"
                          and cc not in fused]
                  for cname, c in comps.items()}
    by_comp = collections.defaultdict(list)
    for it in items:
        by_comp[it["computation"]].append(it)
    loops = []
    for w in whiles:
        if w["comp"] in fused:
            continue
        roots = [cc for a, cc in w["callees"]
                 if a in ("body", "condition") and cc in comps]
        seen, stack = set(), list(roots)
        while stack:
            cc = stack.pop()
            if cc in seen:
                continue
            seen.add(cc)
            stack.extend(x for _, x in callees_of.get(cc, []))
        members = sorted((it for cc in seen for it in by_comp.get(cc, [])),
                         key=lambda it: -it["bytes"])
        body = next((cc for a, cc in w["callees"] if a == "body"), None)
        kinds = collections.Counter(it["kind"] for it in members)
        # raw slice ops anywhere in the iteration, including the ones fused
        # into compute fusions (not counted as copy-like above)
        raw, fseen = collections.Counter(), set()
        fstack = [cc for cc0 in seen for ins in comps[cc0]["instrs"]
                  for a, cc in ins["callees"]
                  if cc in fused and ins["opcode"] != "while"]
        while fstack:
            cc = fstack.pop()
            if cc in fseen:
                continue
            fseen.add(cc)
            fstack.extend(x for ins in comps[cc]["instrs"]
                          for _, x in ins["callees"] if x in comps)
        for cc in seen | fseen:
            for ins in comps[cc]["instrs"]:
                if ins["opcode"] in ("dynamic-slice", "dynamic-update-slice",
                                     "slice"):
                    raw[ins["opcode"]] += 1
        loops.append({
            "name": w["name"], "computation": w["comp"], "body": body,
            "trip": w["trip"], "body_multiplier": mult.get(body, 1),
            "jax_op": jax_op_tail(w["meta"].get("op_name", "")),
            "source": (f"{w['meta'].get('source_file', '?').rsplit('/', 1)[-1]}"
                       f":{w['meta'].get('source_line', '?')}"),
            "source_file": w["meta"].get("source_file", ""),
            "per_iter_count": len(members),
            "per_iter_bytes": sum(it["bytes"] for it in members),
            "bytes_per_exec": sum(it["bytes_per_exec"] for it in members),
            "kinds": dict(kinds), "raw_slice_ops": dict(raw),
            "instructions": members,
        })
    loops.sort(key=lambda lp: -lp["bytes_per_exec"])

    def agg(key):
        d = collections.OrderedDict()
        for it in sorted(items, key=lambda it: -it["bytes_per_exec"]):
            k = it[key]
            a = d.setdefault(k, {"static": 0, "dynamic": 0, "bytes": 0,
                                 "bytes_per_exec": 0})
            a["static"] += 1
            a["dynamic"] += it["multiplier"]
            a["bytes"] += it["bytes"]
            a["bytes_per_exec"] += it["bytes_per_exec"]
        return dict(sorted(d.items(), key=lambda kv: -kv[1]["bytes_per_exec"]))

    by_computation = agg("computation")
    for cname, a in by_computation.items():
        a["multiplier"] = mult.get(cname, 1)
        a["flags"] = sorted(flags.get(cname, ()))
    totals = {
        "static": len(items),
        "dynamic": sum(it["multiplier"] for it in items),
        "bytes": sum(it["bytes"] for it in items),
        "bytes_per_exec": sum(it["bytes_per_exec"] for it in items),
        "out_bytes": sum(it["out_bytes"] for it in items),
    }
    return {
        "entry": entry, "n_computations": len(comps),
        "n_fused_computations": len(fused),
        "n_instructions": sum(len(c["instrs"]) for c in comps.values()),
        "skipped_lines": skipped, "source_filter": source_filter,
        "totals": totals, "by_kind": agg("kind"),
        "by_computation": by_computation, "loops": loops,
        "flags": {c: sorted(f) for c, f in flags.items()
                  if c not in fused and f},
        "instructions": sorted(items, key=lambda it: -it["bytes_per_exec"]),
    }


# ------------------------------------------------------------------- output
def _mib(b):
    return f"{b / 2**20:.3f}"


def summary_line(res):
    t = res["totals"]
    return (f"copies: static={t['static']} dynamic={t['dynamic']} "
            f"bytes_per_exec={t['bytes_per_exec'] / 2**20:.1f} MiB")


def _instr_row(it, with_mult=False):
    users = ", ".join(it["users"]) or "-"
    cols = [it["name"], it["kind"], it["shape"], _mib(it["bytes"])]
    if with_mult:
        cols = [it["computation"], f"x{it['multiplier']}"] + cols \
            + [_mib(it["bytes_per_exec"])]
    cols += [it["jax_op"], it["source"], f"{it['producer']} -> {users}"]
    return "| " + " | ".join(cols) + " |"


def render(res, path="", top=20):
    t = res["totals"]
    out = [f"# HLO copy report: {path}", "",
           f"- entry `{res['entry']}`; {res['n_computations']} computations "
           f"({res['n_fused_computations']} fused), {res['n_instructions']} "
           f"instructions parsed, {res['skipped_lines']} lines skipped",
           "- bytes = bytes moved (DUS: update operand; else output shape); "
           "dynamic / per exec = x while trip counts",
           ]
    if res["source_filter"]:
        out.append(f"- source filter: source_file contains "
                   f"`{res['source_filter']}`")
    out += ["", "## Totals", "",
            "| | static | dynamic (x trip) | MiB static | MiB per exec |",
            "|---|---|---|---|---|",
            f"| all copy-like | {t['static']} | {t['dynamic']} | "
            f"{_mib(t['bytes'])} | {_mib(t['bytes_per_exec'])} |"]
    for k, a in res["by_kind"].items():
        out.append(f"| {k} | {a['static']} | {a['dynamic']} | "
                   f"{_mib(a['bytes'])} | {_mib(a['bytes_per_exec'])} |")
    out += ["", "### By computation", "",
            "| computation | multiplier | static | dynamic | MiB static | "
            "MiB per exec | flags |", "|---|---|---|---|---|---|---|"]
    for cname, a in list(res["by_computation"].items())[:40]:
        out.append(f"| {cname} | x{a['multiplier']} | {a['static']} | "
                   f"{a['dynamic']} | {_mib(a['bytes'])} | "
                   f"{_mib(a['bytes_per_exec'])} | "
                   f"{', '.join(a['flags']) or '-'} |")
    if len(res["by_computation"]) > 40:
        out.append(f"| ... {len(res['by_computation']) - 40} more | | | | | | |")

    out += ["", "## While loops (sorted by MiB per exec)", ""]
    if not res["loops"]:
        out.append("no while loops")
    for lp in res["loops"]:
        kinds = ", ".join(f"{k} {v}" for k, v in
                          sorted(lp["kinds"].items(), key=lambda kv: -kv[1]))
        out += [f"### {lp['name']} ({lp['source']}, {lp['jax_op']})", "",
                f"- in `{lp['computation']}`, body `{lp['body']}`; trip count "
                f"{lp['trip'] if lp['trip'] is not None else 'unknown (1)'}; "
                f"body multiplier x{lp['body_multiplier']}",
                f"- per iteration: {lp['per_iter_count']} copy-like "
                f"({kinds or 'none'}), {_mib(lp['per_iter_bytes'])} MiB; per "
                f"exec {_mib(lp['bytes_per_exec'])} MiB",
                "- raw slice ops per iteration incl. those fused into compute "
                "fusions: " + (", ".join(f"{k} {v}" for k, v in
                                         sorted(lp["raw_slice_ops"].items()))
                               or "none"), ""]
        if lp["instructions"]:
            out += ["| instruction | kind | shape | MiB | jax op | source | "
                    "producer -> users |", "|---|---|---|---|---|---|---|"]
            out += [_instr_row(it) for it in lp["instructions"]]
            out.append("")

    out += ["", f"## Top {top} by MiB per exec", "",
            "| computation | mult | instruction | kind | shape | MiB | "
            "MiB/exec | jax op | source | producer -> users |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    out += [_instr_row(it, with_mult=True) for it in res["instructions"][:top]]
    if res["flags"]:
        out += ["", "Flags: " + "; ".join(
            f"{c}: {', '.join(f)}" for c, f in list(res["flags"].items())[:20])
            + (" ..." if len(res["flags"]) > 20 else "")]
    out += ["", summary_line(res)]
    return "\n".join(out)


SELF_TEST_HLO = """\
HloModule self_test, entry_computation_layout={()->()}

%fused_dus2 (p0: f32[4,8], p1: f32[1,8], p2: f32[4,16], p3: f32[1,16], p4: s32[]) -> (f32[4,8], f32[4,16]) {
  %p0 = f32[4,8]{1,0} parameter(0)
  %p1 = f32[1,8]{1,0} parameter(1)
  %p4 = s32[] parameter(4)
  %c0 = s32[] constant(0)
  %dus.a = f32[4,8]{1,0} dynamic-update-slice(f32[4,8]{1,0} %p0, f32[1,8]{1,0} %p1, s32[] %p4, s32[] %c0)
  %p2 = f32[4,16]{1,0} parameter(2)
  %p3 = f32[1,16]{1,0} parameter(3)
  %dus.b = f32[4,16]{1,0} dynamic-update-slice(f32[4,16]{1,0} %p2, f32[1,16]{1,0} %p3, s32[] %p4, s32[] %c0)
  ROOT %t = (f32[4,8]{1,0}, f32[4,16]{1,0}) tuple(f32[4,8]{1,0} %dus.a, f32[4,16]{1,0} %dus.b)
}

%fused_mixed (q0: f32[4,8], q1: f32[1,8], q2: f32[4,16], q3: s32[]) -> (f32[4,8], bf16[2,16]) {
  %q0 = f32[4,8]{1,0} parameter(0)
  %q1 = f32[1,8]{1,0} parameter(1)
  %q3 = s32[] parameter(3)
  %z = s32[] constant(0)
  %dus.c = f32[4,8]{1,0} dynamic-update-slice(f32[4,8]{1,0} %q0, f32[1,8]{1,0} %q1, s32[] %q3, s32[] %z)
  %q2 = f32[4,16]{1,0} parameter(2)
  %sl = f32[2,16]{1,0} slice(f32[4,16]{1,0} %q2), slice={[0:2], [0:16]}
  %cv = bf16[2,16]{1,0} convert(f32[2,16]{1,0} %sl)
  ROOT %t2 = (f32[4,8]{1,0}, bf16[2,16]{1,0}) tuple(f32[4,8]{1,0} %dus.c, bf16[2,16]{1,0} %cv)
}

ENTRY %main () -> () {
  %a = f32[4,8]{1,0} constant({...})
  %b = f32[1,8]{1,0} constant({...})
  %c = f32[4,16]{1,0} constant({...})
  %d = f32[1,16]{1,0} constant({...})
  %i = s32[] constant(1)
  %f = (f32[4,8]{1,0}, f32[4,16]{1,0}) fusion(f32[4,8]{1,0} %a, f32[1,8]{1,0} %b, f32[4,16]{1,0} %c, f32[1,16]{1,0} %d, s32[] %i), kind=kLoop, calls=%fused_dus2
  %g = (f32[4,8]{1,0}, bf16[2,16]{1,0}) fusion(f32[4,8]{1,0} %a, f32[1,8]{1,0} %b, f32[4,16]{1,0} %c, s32[] %i), kind=kLoop, calls=%fused_mixed
  ROOT %r = () tuple()
}
"""


def self_test():
    """Tuple-rooted fusions: two DUS (32 + 64 update bytes) -> 96 bytes,
    kind fusion/dynamic-update-slice x2; DUS + convert(slice) -> 32 + 64
    (bf16[2,16] element, not the whole output) = 96, kind
    fusion/dynamic-update-slice+slice."""
    res = analyze(SELF_TEST_HLO)
    got = {it["name"]: it for it in res["instructions"]}
    assert set(got) == {"f", "g"}, sorted(got)
    f, g = got["f"], got["g"]
    assert f["kind"] == "fusion/dynamic-update-slice x2", f["kind"]
    assert f["bytes"] == 96, f["bytes"]          # 1x8 f32 + 1x16 f32
    assert f["out_bytes"] == 128 + 256, f["out_bytes"]
    assert g["kind"] == "fusion/dynamic-update-slice+slice", g["kind"]
    assert g["bytes"] == 32 + 64, g["bytes"]     # 1x8 f32 + 2x16 bf16
    assert res["totals"]["bytes"] == 192, res["totals"]
    print("hlo_copy_report self-test: tuple-root DUS x2 = 96 B, "
          "DUS + convert(slice) = 96 B: ok")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", help="*after_optimizations.txt HLO text dump")
    ap.add_argument("--json", default=None, help="write the data here")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--source-filter", default=None,
                    help="only instructions whose source_file contains this")
    a = ap.parse_args(argv)
    opener = gzip.open if a.file.endswith(".gz") else open
    with opener(a.file, "rt", encoding="utf-8", errors="replace") as f:
        text = f.read()
    res = analyze(text, a.source_filter)
    res["file"] = a.file
    print(render(res, a.file, a.top))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
