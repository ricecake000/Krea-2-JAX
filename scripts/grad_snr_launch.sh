#!/usr/bin/env bash
# Laptop driver of the gradient S/N measurement on one Colab G4 (scripts/grad_snr.py, VM side scripts/grad_snr_vm.sh).
# The billed part goes through ~/maxdiffusion-krea2-run/colab/colab_job.sh (3 dead-man switches, its own state and
# endpoint ownership); this script never calls `colab` itself.
#   scripts/grad_snr_launch.sh dry PHASE [--n N] [--cap-min M] [-- grad_snr.py args]
#   scripts/grad_snr_launch.sh run PHASE --confirm-billed [--n N] [--cap-min M] [-- grad_snr.py args]
#   PHASE 1 = smoke (extreme bins, N 32, go/no-go), 2 = measure (all 8 bins, N 64 unless --n)
# dry: samples (select from the local filtered metadata.parquet, sha256-checked against the dataset commit) -> bundle
#   (git archive HEAD + samples.json; refuses a dirty tree: commit first) -> checks (bundle size <= 30 MB, scripts
#   parse, the HF token reaches the gated Raw repo and the dataset commit, sizes of the files the VM will fetch) ->
#   the cost estimate and the exact launch command. DRY_OK / DRY_FAIL.
# run: dry, then colab_job.sh --gpu G4 (BILLED, needs --confirm-billed), then unpacks the download into the stage dir
#   and prints the report. Results: $GRADSNR_STAGE/phase<P>-<ts>/out/ (state.json, report.json, report.txt, logs).
# Secret: the HF token (~/.cache/huggingface/token) is copied into a fresh 0700 mktemp dir, uploaded as
#   /root/hf_token, deleted locally on exit and on the VM right after the downloads; never printed or in argv.
# Laptop python runs under systemd-run MemoryMax=12G, taskset -c 4-7, nice 10.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
M=(systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 taskset -c 4-7 nice -n 10)
PY=${GRADSNR_PY:-$REPO/.venv/bin/python}
COLAB_JOB=${COLAB_JOB:-$HOME/maxdiffusion-krea2-run/colab/colab_job.sh}
STAGE_ROOT=${GRADSNR_STAGE:-$HOME/colab_upload/krea2-gradsnr}
META=${GRADSNR_METADATA:-/home/mj/krea2-ops/eval/datafilter/latents_v2_patched/metadata.parquet}
MANIFEST=${GRADSNR_MANIFEST:-/home/mj/krea2-ops/eval/datafilter/latents_v2_patched/latents/manifest.json}
DATA_REV=8c8af9b53259d175621b08e6ad1572bd23596eaf
META_SHA=c1d533433f5376cd1a57bd6a5d0508b9361298ac0be4e7114e236e0d93fb7643   # metadata.parquet @ DATA_REV (LFS)
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}
MAX_UPLOAD_B=30000000        # one colab upload = one base64 PUT (~31 MB worked, ~113 MB failed)
CU_H=8.90                    # G4, measured 2026-10-27
die() { echo "ERROR: $*" >&2; echo DRY_FAIL; exit 1; }

CMD=${1:-}; PHASE=${2:-}; shift 2 || true
N=""; CAP=""; CONFIRM=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --n) N=$2; shift 2 ;;
    --cap-min) CAP=$2; shift 2 ;;
    --confirm-billed) CONFIRM=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) die "unknown argument $1" ;;
  esac
done
case "$CMD" in dry|run) ;; *) die "usage: grad_snr_launch.sh dry|run 1|2 [--n N] [--cap-min M] [--confirm-billed] [-- args]" ;; esac
case "$PHASE" in
  1) MODE=smoke; N=${N:-32}; RUNBINS=2; DEF_CAP=60 ;;
  2) MODE=measure; N=${N:-64}; RUNBINS=8; DEF_CAP=0 ;;
  *) die "PHASE must be 1 or 2" ;;
esac
[ -x "$PY" ] || die "no python at $PY"
[ -f "$COLAB_JOB" ] || die "colab_job.sh not found at $COLAB_JOB"
[ -f "$HF_TOKEN_FILE" ] || die "no HF token file $HF_TOKEN_FILE"
git diff --quiet HEAD -- || die "uncommitted changes: the bundle is git archive HEAD, commit first"
HEAD_SHA=$(git rev-parse HEAD)

# ---- cost estimate (assumptions stated; s/sample is the one that matters, phase 1 measures it) ----
read -r EST_MIN EST_CAP COST_LINE < <(python3 -I - "$PHASE" "$N" "$RUNBINS" "$CU_H" "${CAP:-0}" "$DEF_CAP" <<'EOF'
import math, sys
phase, n, nbins, cu_h, cap, def_cap = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
S_PER_SAMPLE = 4.0     # assumed: B300 measured 0.86 s/sample (bench_step fwdbwd, ref attention, 4608 tokens);
                       # RTX PRO 6000 ~1/4.5 of its bf16 tensor rate and ~1/4.4 of its HBM bandwidth -> ~4 s
SETUP = 12.0           # create + upload 2, uv/jax install 2, downloads ~37 GB (Raw 26.3, Qwen 8.9, 4 shards 1.9)
                       # at ~150 MB/s 4.2, raw sha256 1, text-encoder identity hash 0.5, slack 2.3
COMPILE = 3.0          # per AOT compile of the unrolled 28-block grad step
LOAD = 3.0             # Raw 26 GB to the device + text states
BIN_END = 0.5          # per bin: Grams, P' transfer, host cross-bin products
TEARDOWN = 2.0         # pack + download + unassign
if phase == 1:
    compiles, samples = 3, 1 + 3 + 2 + 4 + nbins * n   # 2 remat candidates worst case + cudnn; warmup, timing
else:
    compiles, samples = 2, nbins * n
est = SETUP + compiles * COMPILE + LOAD + samples * S_PER_SAMPLE / 60 + nbins * BIN_END + TEARDOWN
cap = cap or def_cap or int(math.ceil(est * 1.4 + 10))
line = (f"COST phase={phase} est_min={est:.0f} est_cu={est / 60 * cu_h:.1f} cap_min={cap} "
        f"cap_cu={cap / 60 * cu_h:.1f} (G4 {cu_h} CU/h; samples {samples} at {S_PER_SAMPLE} s/sample assumed = "
        f"{samples * S_PER_SAMPLE / 60:.0f} min; setup {SETUP:.0f} + {compiles} compiles x {COMPILE:.0f} + load "
        f"{LOAD:.0f} + bins {nbins} x {BIN_END} + teardown {TEARDOWN:.0f} min)")
print(f"{est:.0f} {cap} {line}")
EOF
)
CAP=$EST_CAP

# ---- samples + bundle ----
TS=$(date -u +%Y%m%dT%H%M%SZ)
STAGE=$STAGE_ROOT/phase$PHASE-$TS
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
gb = (raw["raw.safetensors"].size + sum(ds[n].size for n in need)
      + sum(s.size for n, s in te.items() if n.endswith((".json", ".safetensors", ".txt")))) / 1e9
print(f"REMOTE_OK Raw gated access ok, dataset {sys.argv[3][:8]} has the {len(spec['shards'])} shards "
      f"(LFS sha256 == samples digests), Qwen rev ebb281ec ok; VM downloads {gb:.1f} GB")
EOF

ARGS=(--bins 8 --n-per-bin "$N")
[ "$PHASE" = 1 ] && ARGS+=(--run-bins 0,7)
ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})
JOB="tar xOf /root/gradsnr_in.tar repo/scripts/grad_snr_vm.sh > /root/grad_snr_vm.sh && bash /root/grad_snr_vm.sh $MODE $CAP ${ARGS[*]}"
UPS=(-u "$STAGE/gradsnr_in.tar:/root/gradsnr_in.tar" -u "$SECRET_DIR/hf_token:/root/hf_token")
DOWNS=(-d "/content/gradsnr_out.tgz:$STAGE/gradsnr_out.tgz")
echo "$COST_LINE"
echo "LAUNCH (billed): bash $COLAB_JOB -s gradsnr-p$PHASE --gpu G4 -m $CAP -u $STAGE/gradsnr_in.tar:/root/gradsnr_in.tar -u <0700 tmp>/hf_token:/root/hf_token ${DOWNS[*]} -- '$JOB'"
if [ "$CMD" = dry ]; then echo DRY_OK; exit 0; fi
[ "$CONFIRM" = 1 ] || die "run is billed (~$EST_MIN min, cap $CAP min at $CU_H CU/h): pass --confirm-billed"
set +e
bash "$COLAB_JOB" -s "gradsnr-p$PHASE" --gpu G4 -m "$CAP" "${UPS[@]}" "${DOWNS[@]}" -- "$JOB"
RC=$?
set -e
cleanup; SECRET_DIR=""
echo "colab_job rc=$RC"
if [ -f "$STAGE/gradsnr_out.tgz" ]; then
  tar xzf "$STAGE/gradsnr_out.tgz" -C "$STAGE" && mv "$STAGE/gradsnr_out" "$STAGE/out"
  cat "$STAGE/out/report.txt" 2> /dev/null || true
else
  echo "no gradsnr_out.tgz; if job.log has a GRADSNR_STATE line: $PY scripts/grad_snr.py --mode report --out . --from-log <run dir>/job.log"
fi
exit "$RC"
