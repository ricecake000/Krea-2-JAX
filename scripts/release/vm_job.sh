#!/bin/bash
# Colab GPU job of the Krea 2 Anime-tuned release (one session, both models, sequential; launcher: krea2-ops
# release/launch.sh via colab_job.sh, which provides the dead-man switches). Run as root, cwd /content:
#   RUN_ID=<id> [REL_ALLOW_UPLOAD=1] [REL_MAX_MIN=195] [REL_MODELS="raw turbo"] bash /root/vm_job.sh
# Inputs: /root/rel_in.tar (kit/ = scripts/release/*.py, ref/ = Comfy-Org header prefixes + official diffusers
#   index/config + Drive Turbo shard header prefixes, workflows/*_api_smoke.json, plan.json, run_id, rclone.conf
#   ([gdrive] only), rclone.sha256, hf_token). README.md / LICENSE.pdf / NOTICE are NOT shipped: the launcher validates
#   and commits them from the laptop before the runtime starts (hf_release.py prepare); this job uploads model files only.
#   + /root/rclone.xz. SECRETS (rclone.conf, hf_token) are chmod 600, never printed, deleted right after their last
#   use and on EVERY exit (trap + the wall-cap watchdog).
# Per model (raw, then turbo; everything of a model is deleted before the next):
#   pull (rclone copyto from Drive; rclone verifies size + Drive md5 itself) -> check-single (vs Comfy's bf16 header)
#   -> fp8 (release.py fp8) -> check-fp8 (header bytes == Comfy's, size) -> smoke fp8 render + bf16 render (ComfyUI
#   pinned, comfy_smoke.py gate) -> diffusers (release.py diffusers from the single file; Turbo shard headers must
#   equal the Drive Turbo dir's) -> stage (bf16, fp8, transformer/) == plan.json's model files (names + sizes)
#   -> upload (only with REL_ALLOW_UPLOAD=1; hf_release.py: upload_large_folder of the model files, then remote list
#   + sizes == the plan's whole repo, cards included) -> delete.
# /content/out_rel.tgz (logs, renders, status.json; never weights or secrets) is written on every exit path.
# Last line: REL_DONE run_id=.. +Ns  or  REL_FAIL rc=N stage=S run_id=.. reason=..
# Exit codes: 0 ok; 1 bad input; 2 disk; 3 setup (pip / ComfyUI / TE+VAE); 4 pull; 5 header check; 6 fp8 build;
#   7 smoke render gate; 8 diffusers build/check; 9 wall cap / step timeout; 10 stop file /content/stop_rel;
#   11 upload; 12 plan mismatch; 14 UPLOAD_SKIPPED (REL_ALLOW_UPLOAD!=1, everything else passed); 130/143 signal.
set -uo pipefail
C=/content; R=/root; PYK=python3; DRY=${REL_DRY:-0}
if [ "$DRY" = 1 ]; then   # laptop rehearsal (no GPU, no network): other dirs, an existing python, setup + smoke skipped
  C=$REL_CONTENT; R=$REL_ROOT; PYK=$REL_PY
fi
IN=$C/rel_in; WD=$C/relwork; LOGS=$WD/logs; OUT=$WD/out; PACK=$C/out_rel.tgz; STOP_FILE=$C/stop_rel
TAR_IN=$R/rel_in.tar; RCLONE_XZ=$R/rclone.xz; RCLONE=$IN/rclone; RCONF=$IN/rclone.conf; HFTOK=$IN/hf_token
COMFY=$C/ComfyUI; COMFY_TAG=v0.44.0; COMFY_SHA=2c75dc1b379a59e0880619d029607af5169a62ef
COMFY_REPO=https://github.com/comfyanonymous/ComfyUI.git
REF_REPO=Comfy-Org/Krea-2; REF_REV=588feaf955b08b9ac4723eeb751752ab045dba71
PIP_PINS=("huggingface_hub==1.33.0" "hf_xet==1.7.0")
# smoke budgets (comfy_smoke.py): server startup + render + stop; the outer step timeout is strictly larger
# (+ SMOKE_MARGIN_S), so a slow render ends as comfy_smoke's own rc 3 (resource -> bf16 fallback), never as rc 124
SMOKE_START_S=300; SMOKE_RENDER_S=900; SMOKE_LOAD_S=600; SMOKE_STOP_S=60; SMOKE_MARGIN_S=120
SMOKE_OUTER_S=$((SMOKE_START_S + SMOKE_RENDER_S + SMOKE_STOP_S + SMOKE_MARGIN_S))        # 1380
SMOKE_LOAD_OUTER_S=$((SMOKE_START_S + SMOKE_LOAD_S + SMOKE_STOP_S + SMOKE_MARGIN_S))     # 1080
MAX_MIN=${REL_MAX_MIN:-195}; ALLOW=${REL_ALLOW_UPLOAD:-0}; MODELS=${REL_MODELS:-raw turbo}
PULL_FLAGS=(--multi-thread-streams 16 --multi-thread-cutoff 256M --retries 3 --low-level-retries 10 --stats 60s
            --stats-one-line -v)
T0=$(date +%s); WALL_END=$((T0 + MAX_MIN * 60)); RUN_ID=${RUN_ID:-}
RC=0; FAIL_STAGE=""; FAIL_REASON=""; STAGE_NOW=start; CUR_PID=""; WD_PID=""; UPLOADS=""; PULL_PID=""

el() { echo $(( $(date +%s) - T0 )); }
t() { STAGE_NOW=$1; echo "STAGE $1 $(date -u +%T) +$(el)s"; }
alive() { local st; [ -n "$1" ] && st=$(ps -o stat= -p "$1" 2>/dev/null) && [[ $st != Z* ]]; }
scrub() {
  rm -f "$TAR_IN" "$RCLONE_XZ"
  [ -e "$RCONF" ] && { rm -f "$RCONF"; echo "SECRET_DELETED rclone.conf"; }
  [ -e "$HFTOK" ] && { rm -f "$HFTOK"; echo "SECRET_DELETED hf_token"; }
  return 0
}
pack() {
  mkdir -p "$OUT"; cp -p "$LOGS"/*.log "$OUT/" 2>/dev/null
  python3 -I -c 'import json, sys
k = ["run_id", "rc", "stage", "reason", "uploads", "secs", "allow_upload"]
json.dump(dict(zip(k, sys.argv[2:])), open(sys.argv[1], "w"), indent=1)' "$OUT/status.json" "$RUN_ID" "$RC" \
    "${FAIL_STAGE:-none}" "$FAIL_REASON" "$UPLOADS" "$(el)" "$ALLOW"
  find "$OUT" -type f \( -name '*.safetensors' -o -name 'hf_token' -o -name 'rclone.conf' \) -delete
  tar czf "$PACK.tmp" -C "$WD" out 2>/dev/null && mv -f "$PACK.tmp" "$PACK" && echo "PACK_OK $(stat -c %s "$PACK") bytes"
}
finish() {
  trap - EXIT; trap '' INT TERM
  [ -e "$WD/.wall_cap" ] && [ "$RC" = 0 ] && { RC=9; FAIL_STAGE=wall_cap; FAIL_REASON="REL_MAX_MIN=$MAX_MIN reached"; }
  [ "${1:-}" = sig ] && [ "$RC" = 0 ] && { RC=${2:-143}; FAIL_STAGE=signal; FAIL_REASON="terminated in $STAGE_NOW"; }
  alive "$CUR_PID" && kill -TERM "$CUR_PID" 2>/dev/null
  alive "$PULL_PID" && kill -TERM "$PULL_PID" 2>/dev/null
  reap_comfy
  scrub; pack
  : > "$WD/.finished"; alive "$WD_PID" && kill -TERM "$WD_PID" 2>/dev/null
  if [ "$RC" = 0 ]; then echo "REL_DONE run_id=$RUN_ID uploads=[$UPLOADS] +$(el)s"
  else echo "REL_FAIL rc=$RC stage=$FAIL_STAGE run_id=$RUN_ID reason=$(printf '%s' "$FAIL_REASON" | tr '\n' ' ' | cut -c1-300) +$(el)s"; fi
  exit "$RC"
}
fail() { echo "REL_ERROR rc=$1 stage=$2 $3"; [ "$RC" = 0 ] && { RC=$1; FAIL_STAGE=$2; FAIL_REASON=$3; }; finish; }
# step NAME TIMEOUT CMD...: under `timeout` (capped by the wall cap), output to $LOGS/NAME.log, progress every 60 s
step() {
  local name=$1 to=$2 log=$LOGS/$1.log left rc last; shift 2
  left=$(( WALL_END - 120 - $(date +%s) )); [ "$left" -lt "$to" ] && to=$left
  [ "$to" -ge 1 ] || { echo "STEP_SKIPPED $name (no time left)"; return 124; }
  echo "STEP_START $name timeout=${to}s +$(el)s"
  timeout -k 30 "$to" "$@" >> "$log" 2>&1 < /dev/null &
  CUR_PID=$!; last=$(date +%s)
  while alive "$CUR_PID"; do
    [ -e "$STOP_FILE" ] && { kill -TERM "$CUR_PID"; wait "$CUR_PID"; CUR_PID=""; fail 10 "$name" "stop file"; }
    if [ $(( $(date +%s) - last )) -ge 60 ]; then
      last=$(date +%s); echo "STEP_PROGRESS $name +$(el)s: $(tail -n 1 "$log" | tr '\r' '\n' | tail -n 1 | cut -c1-200)"
    fi
    sleep 2
  done
  wait "$CUR_PID"; rc=$?; CUR_PID=""
  echo "STEP_END $name rc=$rc +$(el)s"
  grep -E '^(FP8_|SHARD_|DIFFUSERS_|SINGLE_|SMOKE_|HF_|RELEASE_|META_)' "$log" | grep -v FP8_PROGRESS | tail -n 12 | sed 's/^/  /'
  return "$rc"
}
must() { # RC STAGE STEP TIMEOUT CMD...
  local code=$1 stage=$2 name=$3 to=$4 rc; shift 4
  step "$name" "$to" "$@"; rc=$?
  case $rc in
    0) return 0 ;;
    124|137) fail 9 "$stage" "STEP_TIMEOUT $name" ;;
    *) echo "=== $name.log tail ==="; tail -n 20 "$LOGS/$name.log" | cut -c1-300; fail "$code" "$stage" "$name rc=$rc" ;;
  esac
}
# reap_comfy: no ComfyUI server of THIS job may outlive its render (a SIGKILLed comfy_smoke.py leaves its detached
# server behind): only the process groups recorded in this job's pidfiles ($WD/comfy_*.pid, written by comfy_smoke.py)
# are signalled -- TERM, then KILL -- and waited for until no member remains (bounded). comfy_smoke.py itself refuses a
# port that is not free before it starts (fail closed), so a leftover never gets talked to.
reap_comfy() {
  local f pg st cur i
  for f in "$WD"/comfy_*.pid; do
    [ -e "$f" ] || continue
    read -r pg st < "$f"; rm -f "$f"
    [[ $pg =~ ^[0-9]+$ ]] && [[ $st =~ ^[0-9]+$ ]] || continue
    # the group leader must still be OUR server: same start time (field 22 of /proc/<pid>/stat); a gone leader =
    # signal nothing (comfy_smoke's own stop covers the members), a recycled pid never matches
    cur=$(sed 's/.*) //' "/proc/$pg/stat" 2>/dev/null | awk '{ print $20 }')
    [ "$cur" = "$st" ] || continue
    kill -TERM -- "-$pg" 2>/dev/null
    for ((i = 0; i < 30; i++)); do pgrep -g "$pg" > /dev/null || break; sleep 1; done
    if pgrep -g "$pg" > /dev/null; then
      kill -KILL -- "-$pg" 2>/dev/null
      for ((i = 0; i < 30; i++)); do pgrep -g "$pg" > /dev/null || break; sleep 1; done
    fi
    pgrep -g "$pg" > /dev/null && echo "COMFY_REAP_WARN process group $pg still has members" || echo "COMFY_REAPED group $pg"
  done
  return 0
}
free_gb() { df -P -B1G "$C" | awk 'NR == 2 { print $4 }'; }
watchdog() {
  trap - EXIT INT TERM
  while [ "$(date +%s)" -lt "$WALL_END" ]; do alive "$1" || exit 0; sleep 5; done
  [ -e "$WD/.finished" ] && exit 0
  echo "REL_WALL_CAP +$(el)s: TERM to the job"; : > "$WD/.wall_cap"; kill -TERM "$1" 2>/dev/null
  sleep 90; alive "$1" || exit 0
  pkill -KILL -P "$1" 2>/dev/null; kill -KILL "$1" 2>/dev/null
  rm -f "$RCONF" "$HFTOK" "$TAR_IN" "$RCLONE_XZ"; echo "SECRET_DELETED (watchdog)"; echo "REL_FAIL rc=9 stage=wall_cap_killed"
  exit 0
}
trap finish EXIT
trap 'finish sig 143' TERM
trap 'finish sig 130' INT

# ---- inputs ----
[ -e "$WD" ] && mv "$WD" "$WD.prev.$(date +%s)"
mkdir -p "$LOGS" "$OUT"; rm -f "$PACK" "$STOP_FILE"
echo "REL_JOB start $(date -u +%FT%TZ) allow_upload=$ALLOW models='$MODELS' cap=${MAX_MIN}min pid=$$"
[[ $MAX_MIN =~ ^[1-9][0-9]*$ ]] || fail 1 args "REL_MAX_MIN='$MAX_MIN'"
case $ALLOW in 0|1) ;; *) fail 1 args "REL_ALLOW_UPLOAD='$ALLOW'" ;; esac
watchdog $$ & WD_PID=$!
{ rm -rf "$IN" && mkdir -p "$IN" && chmod 700 "$IN" && (umask 077 && tar xf "$TAR_IN" -C "$IN"); } || fail 1 inputs "cannot extract $TAR_IN"
rm -f "$TAR_IN"; chmod 600 "$RCONF" "$HFTOK" 2>/dev/null
for f in kit/release_lib.py kit/release.py kit/comfy_smoke.py kit/hf_release.py plan.json run_id rclone.conf \
         rclone.sha256 hf_token ref/index.json ref/config.json; do
  [ -s "$IN/$f" ] || fail 1 inputs "missing $f in rel_in.tar"
done
[ "$(tr -d '[:space:]' < "$IN/run_id")" = "$RUN_ID" ] && [ -n "$RUN_ID" ] || fail 1 inputs "RUN_ID mismatch (env '$RUN_ID')"
PLAN=$IN/plan.json; KIT=$IN/kit
for m in $MODELS; do
  case $m in raw|turbo) ;; *) fail 1 args "REL_MODELS has '$m'" ;; esac
done
t inputs

# ---- rclone (binary pinned by sha256 from the laptop) ----
python3 -I -c 'import lzma, shutil, sys
with lzma.open(sys.argv[1]) as f, open(sys.argv[2], "wb") as o:
    shutil.copyfileobj(f, o, 1 << 20)' "$RCLONE_XZ" "$RCLONE" 2>/dev/null || fail 4 rclone "rclone.xz unreadable"
[ "$(sha256sum < "$RCLONE" | cut -d' ' -f1)" = "$(tr -d '[:space:]' < "$IN/rclone.sha256")" ] || fail 4 rclone "rclone binary hash mismatch"
chmod 755 "$RCLONE"; rm -f "$RCLONE_XZ"
RCL=(env "RCLONE_CONFIG=$RCONF" "$RCLONE")
echo "RCLONE_OK $("$RCLONE" version 2>/dev/null | head -1)"
FREE=$(free_gb)
echo "REL_HOST cpus=$(nproc) mem_total_gib=$(awk '/^MemTotal:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo) disk_free_gib=$FREE gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
NEED_GB=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["disk_need_gib"])' "$PLAN") || fail 1 inputs "plan.json"
[ "$FREE" -ge "$NEED_GB" ] || fail 2 disk "free ${FREE} GiB < plan's ${NEED_GB} GiB"
t rclone

pv() { python3 -I -c 'import json, sys; d = json.load(open(sys.argv[1])); [d := d[k] for k in sys.argv[2:]]; print(d)' "$PLAN" "$@"; }
pull_start() { # model -> background rclone copyto of the Drive single file into the model's stage dir
  local m=$1 dst; dst=$C/rel/$m/stage/$(pv models "$m" single_name)
  mkdir -p "$(dirname "$dst")"
  echo "PULL_START $m $(pv models "$m" drive_src) -> $dst"
  timeout -k 30 3600 "${RCL[@]}" copyto "$(pv models "$m" drive_src_vm)" "$dst" "${PULL_FLAGS[@]}" > "$LOGS/pull_$m.log" 2>&1 < /dev/null &
  PULL_PID=$!
}
pull_wait() {
  local m=$1 rc last; last=$(date +%s)
  while alive "$PULL_PID"; do
    [ -e "$STOP_FILE" ] && fail 10 "pull_$m" "stop file"
    [ "$(date +%s)" -ge "$((WALL_END - 120))" ] && fail 9 "pull_$m" "wall cap during the pull"
    [ $(( $(date +%s) - last )) -ge 60 ] && { last=$(date +%s); echo "PULL_PROGRESS $m +$(el)s: $(tail -n 1 "$LOGS/pull_$m.log" | cut -c1-200)"; }
    sleep 2
  done
  wait "$PULL_PID"; rc=$?; PULL_PID=""
  [ "$rc" = 0 ] || { tail -n 5 "$LOGS/pull_$m.log"; fail 4 "pull_$m" "rclone copyto rc=$rc"; }
  local f; f=$C/rel/$m/stage/$(pv models "$m" single_name)
  [ "$(stat -c %s "$f")" = "$(pv models "$m" single_bytes)" ] || fail 4 "pull_$m" "size $(stat -c %s "$f") != plan"
  echo "PULL_OK $m bytes=$(stat -c %s "$f") +$(el)s"
}

# ---- the first pull runs while the software is set up ----
set -- $MODELS; pull_start "$1"

if [ "$DRY" = 1 ]; then echo "DRY: setup skipped"; else
# ---- setup: pins, ComfyUI (pinned tag + sha), Comfy-Org TE + VAE (public, pinned revision, no token) ----
must 3 setup pip 900 python3 -m pip install -q "${PIP_PINS[@]}"
must 3 setup comfy_clone 300 git clone -q --depth 1 --branch "$COMFY_TAG" "$COMFY_REPO" "$COMFY"
[ "$(git -C "$COMFY" rev-parse HEAD)" = "$COMFY_SHA" ] || fail 3 setup "ComfyUI $COMFY_TAG is not $COMFY_SHA"
must 3 setup comfy_reqs 900 python3 -m pip install -q -r "$COMFY/requirements.txt" "${PIP_PINS[@]}"
must 3 setup te_vae 900 env -u HF_TOKEN python3 -I -c 'import json, os, sys, shutil
from huggingface_hub import hf_hub_download
plan, comfy, repo, rev = sys.argv[1:]
for rel, size in json.load(open(plan))["comfy_assets"].items():
    p = hf_hub_download(repo, rel, revision=rev, token=False, local_dir=os.path.join(comfy, "models", "_dl"))
    assert os.path.getsize(p) == size, (rel, os.path.getsize(p), size)
    dst = os.path.join(comfy, "models", rel.split("/")[0], os.path.basename(rel))
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(p, dst)
    print("ASSET_OK", rel, size)' "$PLAN" "$COMFY" "$REF_REPO" "$REF_REV"
python3 -I -c 'import torch; print("TORCH", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))' || fail 3 setup "torch/cuda unusable"
fi
t setup

# ---- per model ----
i=0
for m in $MODELS; do
  i=$((i + 1)); MD=$C/rel/$m; ST=$MD/stage
  SINGLE=$ST/$(pv models "$m" single_name); FP8=$ST/$(pv models "$m" fp8_name)
  [ -n "$PULL_PID" ] || pull_start "$m"
  pull_wait "$m"; t "pull_$m"
  # published files carry no internal paths: __metadata__ values with absolute paths are dropped in place
  # (header length, offsets, data bytes and file size unchanged)
  must 5 "check_$m" "strip_meta_$m" 300 "$PYK" -I "$KIT/release.py" strip-meta --file "$SINGLE"
  ALLBF=(); [ "$(pv models "$m" all_bf16)" = True ] && ALLBF=(--all-bf16)
  must 5 "check_$m" "check_single_$m" 300 "$PYK" -I "$KIT/release.py" check-single --file "$SINGLE" \
    --ref-hdr "$IN/ref/krea2_${m}_bf16.hdr" "${ALLBF[@]}"
  must 6 "fp8_$m" "fp8_$m" 3600 env OMP_NUM_THREADS="$(nproc)" "$PYK" -I "$KIT/release.py" fp8 --src "$SINGLE" --out "$FP8"
  must 5 "check_$m" "check_fp8_$m" 300 "$PYK" -I "$KIT/release.py" check-fp8 --file "$FP8" \
    --ref-hdr "$IN/ref/krea2_${m}_fp8_scaled.hdr" --ref-size "$(pv comfy_fp8_size)"
  t "fp8_$m"
  DM=$COMFY/models/diffusion_models; mkdir -p "$DM"
  ln -sf "$FP8" "$DM/$(basename "$FP8")"; ln -sf "$SINGLE" "$DM/$(basename "$SINGLE")"
  WF=$IN/workflows/krea2_anime_tuned_${m}_api_smoke.json; PROMPT=$(pv smoke_prompt)
  [ "$DRY" = 1 ] && echo "DRY: smoke renders skipped"
  SMOKE=("$PYK" -I "$KIT/comfy_smoke.py" --comfy "$COMFY" --workflow "$WF" --out "$OUT/renders" --prompt "$PROMPT"
         --seed "$(pv smoke_seed)" --width "$(pv smoke_width)" --height "$(pv smoke_height)"
         --startup-s "$SMOKE_START_S" --stop-s "$SMOKE_STOP_S")
  if [ "$DRY" != 1 ]; then
    # fp8 = the file users run: a hard gate (comfy_smoke rc 1 fatal / 3 resource: both abort). bf16 (> the L4's 24 GB,
    # relies on ComfyUI's offload): a full render; ONLY a resource failure (rc 3 = OOM / allocation error / server
    # SIGKILLed / --timeout-s exceeded) falls back to a load check (1 step, PLAN A.3) whose key + image gates still
    # apply. Key mismatches, image-gate failures and anything else are fatal.
    reap_comfy
    must 7 "smoke_$m" "smoke_${m}_fp8" "$SMOKE_OUTER_S" "${SMOKE[@]}" --timeout-s "$SMOKE_RENDER_S" --port $((8188 + 3 * i)) --pidfile "$WD/comfy_fp8.pid" \
      --render "${m}_fp8:$(basename "$FP8")"
    reap_comfy
    step "smoke_${m}_bf16" "$SMOKE_OUTER_S" "${SMOKE[@]}" --timeout-s "$SMOKE_RENDER_S" --port $((8189 + 3 * i)) --pidfile "$WD/comfy_bf16.pid" \
      --render "${m}_bf16:$(basename "$SINGLE")"; rc=$?
    reap_comfy
    case $rc in
      0) ;;
      3) echo "BF16_RENDER_FALLBACK $m: resource failure of the full bf16 render (see smoke_${m}_bf16.log); load check next"
         must 7 "smoke_$m" "smoke_${m}_bf16_load" "$SMOKE_LOAD_OUTER_S" "${SMOKE[@]}" --timeout-s "$SMOKE_LOAD_S" --port $((8190 + 3 * i)) --pidfile "$WD/comfy_load.pid" \
           --render "${m}_bf16load:$(basename "$SINGLE"):load"
         reap_comfy ;;
      124|137) fail 9 "smoke_$m" "STEP_TIMEOUT smoke_${m}_bf16" ;;
      *) echo "=== smoke_${m}_bf16.log tail ==="; tail -n 20 "$LOGS/smoke_${m}_bf16.log" | cut -c1-300
         fail 7 "smoke_$m" "smoke_${m}_bf16 rc=$rc (fatal: keys / image gate / other)" ;;
    esac
  fi
  rm -f "$DM/$(basename "$FP8")" "$DM/$(basename "$SINGLE")"
  t "smoke_$m"
  SREF=(); [ "$m" = turbo ] && SREF=(--shard-ref "$IN/ref/turbo_shards")
  must 8 "diffusers_$m" "diffusers_$m" 3600 "$PYK" -I "$KIT/release.py" diffusers --src "$SINGLE" \
    --index "$IN/ref/index.json" --config "$IN/ref/config.json" --out "$ST/transformer"
  must 8 "diffusers_$m" "check_diffusers_$m" 300 "$PYK" -I "$KIT/release.py" check-diffusers --dir "$ST/transformer" \
    --index "$IN/ref/index.json" --expect-total "$(pv models "$m" diffusers_total)" "${SREF[@]}"
  t "diffusers_$m"
  python3 -I -c 'import json, os, sys
plan, m, st, man, exp = sys.argv[1:]
full = json.load(open(plan))["models"][m]["files"]
cards = {"README.md", "LICENSE.pdf", "NOTICE"}   # committed from the laptop (launcher), not by this job
want = {k: v for k, v in full.items() if k not in cards}
got = {}
for root, dirs, files in os.walk(st):
    for fn in files:
        p = os.path.join(root, fn); got[os.path.relpath(p, st)] = os.path.getsize(p)
bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
if bad or set(got) != set(want):
    print("PLAN_MISMATCH", bad, sorted(set(got) ^ set(want))); sys.exit(1)
json.dump({"files": got}, open(man, "w"), indent=1)
json.dump({"files": full}, open(exp, "w"), indent=1)
print("STAGE_OK", m, len(got), "model files", sum(got.values()), "bytes == plan (cards: committed by the launcher)")' \
    "$PLAN" "$m" "$ST" "$MD/manifest.json" "$MD/expect.json" \
    || fail 12 "stage_$m" "staged files != plan.json"
  t "stage_$m"
  # start the next model's pull now (disk: plan.json's disk_need_gib covers both models' peaks at once)
  NEXT=$(python3 -I -c 'import sys; ms = sys.argv[1].split(); i = ms.index(sys.argv[2]); print(ms[i + 1] if i + 1 < len(ms) else "")' "$MODELS" "$m")
  [ -n "$NEXT" ] && pull_start "$NEXT"
  if [ "$ALLOW" = 1 ]; then
    must 11 "upload_$m" "upload_$m" 5400 "$PYK" -I "$KIT/hf_release.py" upload --token-file "$HFTOK" \
      --repo "$(pv models "$m" repo)" --folder "$ST" --manifest "$MD/manifest.json" --expect "$MD/expect.json"
    UPLOADS="$UPLOADS $m"
  else
    echo "UPLOAD_SKIPPED $m (REL_ALLOW_UPLOAD!=1)"
  fi
  rm -rf "$MD"; echo "MODEL_DELETED $m +$(el)s"
  t "done_$m"
done
scrub
[ "$ALLOW" = 1 ] || fail 14 upload "UPLOAD_SKIPPED reason=not_allowed (built + checked + rendered, nothing uploaded)"
finish
