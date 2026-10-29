#!/usr/bin/env bash
# Laptop driver of the gradient S/N measurement on one Colab G4 (scripts/grad_snr.py, VM side scripts/grad_snr_vm.sh).
# The billed part goes through ~/maxdiffusion-krea2-run/colab/colab_job.sh (3 dead-man switches, its own state and
# endpoint ownership); this script never calls `colab` itself.
#   scripts/grad_snr_launch.sh dry [--n N] [--budget-cu B] [--cap-min M] [-- grad_snr.py args]
#   scripts/grad_snr_launch.sh run --confirm-billed [--n N] [--budget-cu B] [--cap-min M] [-- grad_snr.py args]
#   ONE session, grad_snr.py --mode auto: bins 0 and 7 at N (default 64), the go/no-go gate, and on GO the other
#   bins in the same session (or a fresh measurement at a larger N_m; rule in grad_snr.py). B = the CU allowed after
#   the gate (default 15). The colab_job.sh wall cap -m (the real money guard) = wall_cap_min of COST_GO_MAX, whose
#   max_min adds the pinned post-cap allowances (CJ_ENV below) = the worst-case billed minutes.
# dry: samples (select from the local filtered metadata.parquet, sha256-checked against the dataset commit) -> the
#   memory pre-flight (grad_snr.py --mode aot with the job's args: CPU AOT of the exact per-sample step under jax
#   0.11.0 = the VM's pin, GPU peak estimate calibrated on the G4; DRY_FAIL unless a remat candidate fits 0.92 x the
#   G4 limit; $GRADSNR_STAGE/auto-<ts>/aot.log + aot.json) -> bundle
#   (git archive HEAD + samples.json; refuses a dirty tree: commit first) -> checks (bundle size <= 30 MB, scripts
#   parse, repo metadata + LFS sha256 of the files the VM will fetch, and an authenticated 8-byte Range GET of
#   raw.safetensors, the latents manifest and every shard, i.e. the token can really download them) ->
#   the cost estimate and the exact launch command. DRY_OK / DRY_FAIL.
# run: dry, then colab_job.sh --gpu G4 (BILLED, needs --confirm-billed), then unpacks the download into the stage dir
#   and prints the gate line and both reports. Results: $GRADSNR_STAGE/auto-<ts>/out/{smoke,measure}/ (state.json,
#   report.json, report.txt) + logs.
# Secret: the HF token (~/.cache/huggingface/token) is copied into a fresh 0700 mktemp dir, uploaded as
#   /root/hf_token, deleted locally on exit and on the VM right after the downloads; never printed or in argv.
# Laptop python runs under systemd-run MemoryMax=12G, taskset -c 4-7, nice 10.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
M=(systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 taskset -c 4-7 nice -n 10)
PY=${GRADSNR_PY:-$REPO/.venv/bin/python}
AOT_PY=${GRADSNR_AOT_PY:-$HOME/krea2-ops/mi350x/venv-jax0110/bin/python}   # jax 0.11.0 CPU (--mode aot checks it)
COLAB_JOB=${COLAB_JOB:-$HOME/maxdiffusion-krea2-run/colab/colab_job.sh}
STAGE_ROOT=${GRADSNR_STAGE:-$HOME/colab_upload/krea2-gradsnr}
META=${GRADSNR_METADATA:-/home/mj/krea2-ops/eval/datafilter/latents_v2_patched/metadata.parquet}
MANIFEST=${GRADSNR_MANIFEST:-/home/mj/krea2-ops/eval/datafilter/latents_v2_patched/latents/manifest.json}
DATA_REV=8c8af9b53259d175621b08e6ad1572bd23596eaf
META_SHA=c1d533433f5376cd1a57bd6a5d0508b9361298ac0be4e7114e236e0d93fb7643   # metadata.parquet @ DATA_REV (LFS)
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}
MAX_UPLOAD_B=30000000        # one colab upload = one base64 PUT (~31 MB worked, ~113 MB failed)
CU_H=8.90                    # G4, measured 2026-10-27
# colab_job.sh knobs pinned for this job, so the printed worst case holds: billing can last past the -m wall cap by
# the download phase (job done -> hold -> -d downloads; our tgz is a few MB) and, if the driver dies or overruns, the
# guard's deadline (armed BEFORE the create: wall cap + DOWNLOAD_PHASE_S + GUARD_SLACK_S), its poll and one stop pass
# up to the unassign (lookup + list + unassign, each timeout -k 10 GUARD_API_TIMEOUT_S)
CJ_ENV=(DOWNLOAD_PHASE_S=300 GUARD_SLACK_S=600 GUARD_POLL_S=15 GUARD_API_TIMEOUT_S=120)
die() { echo "ERROR: $*" >&2; [ "${CMD:-dry}" = dry ] && echo DRY_FAIL; exit 1; }

USAGE="usage: grad_snr_launch.sh dry|run [--confirm-billed] [--n N] [--budget-cu B] [--cap-min M] [-- grad_snr.py args]"
CMD=${1:-}; shift || true
N=64; BUDGET=15; CAP=""; CONFIRM=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    1|2) die "the PHASE argument is gone: one session runs the gate and phase 2 (grad_snr.py --mode auto); $USAGE" ;;
    --n) N=$2; shift 2 ;;
    --budget-cu) BUDGET=$2; shift 2 ;;
    --cap-min) CAP=$2; shift 2 ;;
    --confirm-billed) CONFIRM=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) die "unknown argument $1; $USAGE" ;;
  esac
done
case "$CMD" in dry|run) ;; *) die "$USAGE" ;; esac
[[ "$N" =~ ^[0-9]+$ ]] && [ "$N" -ge 16 ] || die "--n must be an integer >= 16 (2 x 8 groups)"
[[ "$BUDGET" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--budget-cu must be a number"
[ -z "$CAP" ] || [[ "$CAP" =~ ^[1-9][0-9]*$ ]] || die "--cap-min must be a positive integer"
MODE=auto
[ -x "$PY" ] || die "no python at $PY"
[ -x "$AOT_PY" ] || die "no jax 0.11.0 python at $AOT_PY (GRADSNR_AOT_PY) for the memory pre-flight"
[ -f "$COLAB_JOB" ] || die "colab_job.sh not found at $COLAB_JOB"
[ -f "$HF_TOKEN_FILE" ] || die "no HF token file $HF_TOKEN_FILE"
git diff --quiet HEAD -- || die "uncommitted changes: the bundle is git archive HEAD, commit first"
HEAD_SHA=$(git rev-parse HEAD)

# ---- cost estimate (assumptions stated; s/sample is the one that matters, the run measures it) ----
{ read -r EST_NOGO; read -r CAP; read -r MAX_MIN; read -r NOGO_LINE; read -r GO_LINE; } < <(python3 -I - "$N" "$BUDGET" \
    "$CU_H" "${CAP:-0}" "${CJ_ENV[@]}" <<'EOF'
import math, sys
n, budget_cu, cu_h, cap_arg = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
cj = {k: int(v) for k, v in (a.split("=") for a in sys.argv[5:])}
S_PER_SAMPLE = 4.0     # assumed: B300 measured 0.86 s/sample (bench_step fwdbwd, ref attention, 4608 tokens);
                       # RTX PRO 6000 ~1/4.5 of its bf16 tensor rate and ~1/4.4 of its HBM bandwidth -> ~4 s
K = 8
CREATE = 2.0           # create + upload (billed from create)
HUB = 0.5              # downloader venv (huggingface_hub, hf_xet), then the background downloads start
DL_IN = 1.5            # latents 1.9 GB + Qwen 8.9 GB at ~150 MB/s (assumed undisturbed by the parallel install)
DL_RAW = 3.0 + 1.0     # Raw 26.3 GB at ~150 MB/s + its sha256
INSTALL = 2.0          # uv venv + jax etc., in parallel with the downloads
COMPILE = 3.0          # ref AOT compile of the unrolled 28-block grad step; the cuDNN candidate compiles on a
                       # second thread at the same time (assumed hidden: >= 2 free cores)
INPUTS = 1.5           # shard sha256, latents, Qwen identity hash + 128 text states (after COMPILE, Qwen ready)
LOAD = 3.0             # Raw 26 GB to the device, once its marker (downloaded + sha256) is there
BIN_END = 0.5          # per bin: Grams + P' transfer on the GPU path (host cross products overlap the next bin)
TEARDOWN = 2.0         # pack + download + unassign
CONT = 1.0             # gate -> phase 2 (reports, memmap re-create on a fresh measurement)
# critical path: the downloads and install -> compile -> inputs run side by side; Raw loads after both
ready = CREATE + max(HUB + DL_IN + DL_RAW, HUB + INSTALL + COMPILE + INPUTS) + LOAD
pre = 1 + 3 + 2 + 4    # warmup + 3 timing samples; cuDNN gradient pair + warmup + 3 timing samples
gate = ready + pre * S_PER_SAMPLE / 60 + 2 * (n * S_PER_SAMPLE / 60 + BIN_END) + TEARDOWN
rest = CONT + (K - 2) * (n * S_PER_SAMPLE / 60 + BIN_END)
gate_cap = int(math.ceil(gate * 1.4 + 10))
budget_min = int(math.ceil(budget_cu / cu_h * 60))
cap = cap_arg or gate_cap + budget_min
# billed past the wall cap at most: download phase + guard slack + guard poll + one stop pass (3 API calls)
stop_pass = 3 * (cj["GUARD_API_TIMEOUT_S"] + 10)
post_s = cj["DOWNLOAD_PHASE_S"] + cj["GUARD_SLACK_S"] + cj["GUARD_POLL_S"] + stop_pass
max_min = cap + int(math.ceil(post_s / 60))
nogo = (f"COST_NOGO est_min={gate:.0f} est_cu={gate / 60 * cu_h:.1f} (setup to Raw on the device {ready:.0f} min = "
        f"create {CREATE:.0f} + max(downloads {HUB + DL_IN + DL_RAW:.0f}, install + compile + inputs "
        f"{HUB + INSTALL + COMPILE + INPUTS:.0f}) + load {LOAD:.0f}; {pre} timing samples + 2 gate bins x N {n} at "
        f"{S_PER_SAMPLE} s/sample assumed = {(pre + 2 * n) * S_PER_SAMPLE / 60:.0f} min; teardown {TEARDOWN:.0f}; "
        f"G4 {cu_h} CU/h)")
go = (f"COST_GO_MAX max_min={max_min} max_cu={max_min / 60 * cu_h:.1f} wall_cap_min={cap} (wall cap -m {cap} = " + (
      f"--cap-min {cap}" if cap_arg else
      f"gate stage ceil({gate:.0f} x 1.4 + 10) = {gate_cap} + post-gate budget ceil({budget_cu:g} CU / {cu_h} x 60) "
      f"= {budget_min}") + f"; + past the cap at most ceil({post_s} s / 60) = {max_min - cap} min: download phase "
      f"{cj['DOWNLOAD_PHASE_S']} s + guard slack {cj['GUARD_SLACK_S']} s + guard poll {cj['GUARD_POLL_S']} s + one "
      f"stop pass to the unassign 3 x ({cj['GUARD_API_TIMEOUT_S']} + 10) s; the create is inside the cap (the guard "
      f"deadline is armed before it); assumes the Colab API answers (the guard retries until it does); expected GO "
      f"with N_m = N {n}: ~{gate + rest:.0f} min / {(gate + rest) / 60 * cu_h:.1f} CU)")
print(f"{gate:.0f}\n{cap}\n{max_min}\n{nogo}\n{go}")
EOF
) || die "cost estimate failed"
[[ "$CAP" =~ ^[1-9][0-9]*$ ]] && [[ "$MAX_MIN" =~ ^[1-9][0-9]*$ ]] || die "bad cap $CAP / $MAX_MIN"

# ---- samples + bundle ----
TS=$(date -u +%Y%m%dT%H%M%SZ)
STAGE=$STAGE_ROOT/auto-$TS
case "$STAGE" in *:*) die "stage path must not contain ':'" ;; esac
mkdir -p "$STAGE"
SAMPLES=$STAGE_ROOT/samples.json
GOT_SHA=$(sha256sum "$META" | cut -d' ' -f1)
[ "$GOT_SHA" = "$META_SHA" ] || die "$META sha256 $GOT_SHA != metadata.parquet @ $DATA_REV ($META_SHA)"
if [ ! -f "$SAMPLES" ]; then
  "${M[@]}" "$PY" -I scripts/grad_snr.py --mode select --metadata "$META" --manifest "$MANIFEST" \
    --revision "$DATA_REV" --out "$SAMPLES"
fi
cp "$SAMPLES" "$STAGE/samples.json"
ARGS=(--bins 8 --n-per-bin "$N" --budget-cu "$BUDGET")
ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

# ---- memory pre-flight (free, ~15-30 s): the step the VM AOT-checks before loading anything, estimated here ----
"${M[@]}" "$AOT_PY" -I scripts/grad_snr.py "${ARGS[@]}" --mode aot --out "$STAGE" --samples "$STAGE/samples.json" \
  > "$STAGE/aot.log" 2>&1 || { tail -n 3 "$STAGE/aot.log" >&2; die "memory pre-flight: no remat candidate fits the G4 (or the pre-flight failed; $STAGE/aot.log)"; }
grep -E '^(aot: |AOT_OK )' "$STAGE/aot.log" || die "memory pre-flight printed no verdict ($STAGE/aot.log)"
git archive --prefix=repo/ -o "$STAGE/gradsnr_in.tar" HEAD
tar -rf "$STAGE/gradsnr_in.tar" -C "$STAGE" samples.json
SZ=$(stat -c %s "$STAGE/gradsnr_in.tar")
[ "$SZ" -le "$MAX_UPLOAD_B" ] || die "bundle $SZ B > $MAX_UPLOAD_B"
bash -n scripts/grad_snr_vm.sh || die "grad_snr_vm.sh does not parse"
"$PY" -I -m py_compile scripts/grad_snr.py || die "grad_snr.py does not compile"
echo "BUNDLE $STAGE/gradsnr_in.tar ($SZ B, HEAD $HEAD_SHA, samples $(python3 -I -c 'import json,sys; s=json.load(open(sys.argv[1])); print(len(s["entries"]), "ids of", s["bucket"], "shards", ",".join(sorted(s["shards"])))' "$STAGE/samples.json"))"

# ---- secret ----
SECRET_DIR=""
cleanup() { if [ -n "$SECRET_DIR" ]; then rm -rf "$SECRET_DIR"; fi; }
trap cleanup EXIT
SECRET_DIR=$(umask 077; mktemp -d "${TMPDIR:-/tmp}/gradsnr_secret.XXXXXX")
(umask 077; tr -d '[:space:]' < "$HF_TOKEN_FILE" > "$SECRET_DIR/hf_token")

# ---- remote checks (metadata only: repo info + file sizes; nothing is downloaded) ----
"${M[@]}" "$PY" -I - "$SECRET_DIR/hf_token" "$STAGE/samples.json" "$DATA_REV" <<'EOF' || die "remote checks failed"
import json, sys
from huggingface_hub import HfApi
tok = open(sys.argv[1]).read().strip()
spec = json.load(open(sys.argv[2]))
api = HfApi(token=tok)
raw = {s.rfilename: s for s in api.repo_info("krea/Krea-2-Raw", revision="6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603",
                                             files_metadata=True).siblings}
assert raw["raw.safetensors"].size == 26283332608 and raw["raw.safetensors"].lfs.sha256.startswith("f99bb0ff"), "raw"
ds = {s.rfilename: s for s in api.repo_info("ricecake/danbooru-curated-v2-latents", repo_type="dataset",
                                            revision=sys.argv[3], files_metadata=True).siblings}
need = ["latents/manifest.json"] + ["latents/" + s for s in spec["shards"]]
miss = [n for n in need if n not in ds]
assert not miss, miss
for s, dig in spec["shards"].items():
    assert ds["latents/" + s].lfs.sha256 == dig, s
te = {s.rfilename: s for s in HfApi().repo_info("Qwen/Qwen3-VL-4B-Instruct",
                                               revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
                                               files_metadata=True).siblings}
# file-level, authenticated: an 8-byte Range GET on the resolve URL of every
# gated / private file the VM fetches (metadata alone does not prove download
# rights). Redirects are NEVER followed (urllib would copy the Authorization
# header to the redirect host): 200/206 = the bytes, a 30x to an https
# Location = authorized (the Location is not contacted; only its host is
# printed); a relative Location stays on huggingface.co and is re-requested;
# 401/403/404 or anything else fails. The token is never printed.
import struct, urllib.error, urllib.parse, urllib.request
from huggingface_hub import hf_hub_url

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None

opener = urllib.request.build_opener(NoRedirect)
hosts = set()
def head8(repo, name, rev, repo_type=None):
    url = hf_hub_url(repo, name, revision=rev, repo_type=repo_type)
    for _ in range(3):
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok, "Range": "bytes=0-7"})
        try:
            with opener.open(req, timeout=60) as r:
                b = r.read(8)
            assert r.status in (200, 206) and len(b) == 8, (repo, name, r.status, len(b))
            return b
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location") or ""
            if e.code not in (301, 302, 303, 307, 308):
                raise SystemExit(f"{repo}/{name}: HTTP {e.code} (no download rights?)")
            if loc.startswith("/"):
                url = "https://huggingface.co" + loc   # same origin
                continue
            u = urllib.parse.urlsplit(loc)
            assert u.scheme == "https" and u.netloc, f"{repo}/{name}: redirect to a non-https Location (host {u.netloc!r})"
            hosts.add(u.netloc)
            return None
    raise SystemExit(f"{repo}/{name}: too many same-origin redirects")
b = head8("krea/Krea-2-Raw", "raw.safetensors", "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603")
n = struct.unpack("<Q", b)[0] if b else None
assert n in (None, 43720), f"raw.safetensors header length {n}"
head8("ricecake/danbooru-curated-v2-latents", "latents/manifest.json", sys.argv[3], "dataset")
for s in sorted(spec["shards"]):
    head8("ricecake/danbooru-curated-v2-latents", "latents/" + s, sys.argv[3], "dataset")
gb = (raw["raw.safetensors"].size + sum(ds[n].size for n in need)
      + sum(s.size for n, s in te.items() if n.endswith((".json", ".safetensors", ".txt")))) / 1e9
print(f"REMOTE_OK Raw downloadable with the token (authenticated Range GET, no redirect followed: "
      f"{'header ' + str(n) + ' B' if n else 'authorized https redirect'}; redirect hosts {sorted(hosts)}), latents manifest + shards downloadable, dataset {sys.argv[3][:8]} has the {len(spec['shards'])} shards "
      f"(LFS sha256 == samples digests), Qwen rev ebb281ec ok; VM downloads {gb:.1f} GB")
EOF

# job.sh on the VM is parsed by bash once: each argument shell-escaped, so the VM's argv == the pre-flight's ARGS
VM_ARGV=$(printf '%q ' "$MODE" "$CAP" "${ARGS[@]}")
JOB="tar xOf /root/gradsnr_in.tar repo/scripts/grad_snr_vm.sh > /root/grad_snr_vm.sh && bash /root/grad_snr_vm.sh ${VM_ARGV% }"
UPS=(-u "$STAGE/gradsnr_in.tar:/root/gradsnr_in.tar" -u "$SECRET_DIR/hf_token:/root/hf_token")
DOWNS=(-d "/content/gradsnr_out.tgz:$STAGE/gradsnr_out.tgz")
echo "$NOGO_LINE"
echo "$GO_LINE"
echo "LAUNCH (billed): env ${CJ_ENV[*]} bash $COLAB_JOB -s gradsnr-auto --gpu G4 -m $CAP -u $STAGE/gradsnr_in.tar:/root/gradsnr_in.tar -u <0700 tmp>/hf_token:/root/hf_token ${DOWNS[*]} -- $(printf '%q' "$JOB")"
if [ "$CMD" = dry ]; then echo DRY_OK; exit 0; fi
[ "$CONFIRM" = 1 ] || die "run is billed (~$EST_NOGO min on NO-GO, at most $MAX_MIN min at $CU_H CU/h): pass --confirm-billed"
set +e
env "${CJ_ENV[@]}" bash "$COLAB_JOB" -s gradsnr-auto --gpu G4 -m "$CAP" "${UPS[@]}" "${DOWNS[@]}" -- "$JOB"
RC=$?
set -e
cleanup; SECRET_DIR=""
echo "colab_job rc=$RC"
if [ -f "$STAGE/gradsnr_out.tgz" ]; then
  tar xzf "$STAGE/gradsnr_out.tgz" -C "$STAGE" && mv "$STAGE/gradsnr_out" "$STAGE/out"
  for s in smoke measure; do
    if [ -f "$STAGE/out/$s/report.txt" ]; then echo "==== $s: $STAGE/out/$s/report.txt"; cat "$STAGE/out/$s/report.txt"; fi
  done
  grep -h '^GRADSNR_GATE ' "$STAGE/out/grad_snr.log" 2> /dev/null || echo "no GRADSNR_GATE line (the gate was not reached)"
else
  echo "no gradsnr_out.tgz; if job.log has GRADSNR_STATE lines: $PY scripts/grad_snr.py --mode report --out . --from-log <run dir>/job.log [--stage smoke|measure]"
fi
exit "$RC"
