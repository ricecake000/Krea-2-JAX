#!/bin/bash
# Colab G4 VM side of the gradient S/N measurement (scripts/grad_snr.py); started by scripts/grad_snr_launch.sh
# through colab_job.sh, never by hand.
#   bash grad_snr_vm.sh auto|smoke|measure CAP_MIN [grad_snr.py args...]
# Inputs: /root/gradsnr_in.tar (repo/ = git archive HEAD, samples.json), /root/hf_token (deleted right after the
# downloads and on every exit path; read inside python from its path, never in argv, env or logs).
# Steps, overlapped: GPU / host / disk report -> a small downloader venv (huggingface_hub, hf_xet) -> BACKGROUND
# downloads (pinned revisions): latents/manifest.json + the shards samples.json uses (sha256 checked by grad_snr.py),
# Qwen3-VL-4B-Instruct (identity checked by grad_snr.py), Raw raw.safetensors (size + sha256 checked here), each
# stage ending in a ready marker (.ready = "ok") or a .failed file with the reason -> meanwhile the main venv with
# jax[cuda13]==0.11.0 (jax[cuda12] when the driver is < CUDA 13) -> grad_snr.py as soon as jax imports, with
# --deadline-min = what is left of CAP_MIN minus a margin; it AOT-compiles while the downloads run and waits on the
# markers before reading each input. Output: /content/gradsnr_out.tgz (out/smoke + out/measure for auto, logs; the
# P' memmap stays in /content/gradsnr_scratch).
set -euo pipefail
MODE=${1:?mode}; CAP_MIN=${2:?cap}; shift 2
case "$MODE" in auto|smoke|measure) ;; *) echo "BAD_MODE $MODE"; exit 2 ;; esac
T0=$(date +%s)
st() { echo "STAGE $1 $(date -u +%T) +$(( $(date +%s) - T0 ))s"; }
IN=/content/gs; OUT=/content/gradsnr_out; SCR=/content/gradsnr_scratch; DATA=/content/gs_data; HFD=/content/gs_hf
TOKEN=/root/hf_token
RAW_REPO=krea/Krea-2-Raw; RAW_REV=6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603
RAW_SHA=f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7; RAW_SIZE=26283332608
TE_REPO=Qwen/Qwen3-VL-4B-Instruct; TE_REV=ebb281ec70b05090aa6165b016eac8ec08e71b17
DATA_REPO=ricecake/danbooru-curated-v2-latents
MARGIN_MIN=${GRADSNR_MARGIN_MIN:-12}   # VM start lags the runner's clock + final report + download
DL_PID=""
pack() { tar czf /content/gradsnr_out.tgz -C /content gradsnr_out 2> /dev/null || true; }
kill_tree() { local c; for c in $(pgrep -P "$1" 2> /dev/null); do kill_tree "$c"; done; kill "$1" 2> /dev/null || true; }
cleanup() { if [ -n "$DL_PID" ]; then kill_tree "$DL_PID"; fi; rm -f "$TOKEN"; pack; }
trap cleanup EXIT
[ -f "$TOKEN" ] && chmod 600 "$TOKEN"
mkdir -p "$IN" "$OUT" "$SCR" "$DATA" "$HFD/te" "$HFD/raw"
tar xf /root/gradsnr_in.tar -C "$IN" && rm -f /root/gradsnr_in.tar
st extracted
{ nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv; nvidia-smi | sed -n 3p
  free -g; df -h /content; echo "nproc $(nproc)"; python3 --version; } 2>&1 | tee "$OUT/vm_env.txt"
# before ~37 GB of downloads: Raw 26.3 + Qwen 8.9 + shards 1.9, then the P' memmap (~29 GB for the default
# --cross-params, refused by grad_snr.py above 0.8 x the free disk; a fresh phase-2 measurement replaces it, never
# two at once) -> need ~80 GB free
FREE_GB=$(( $(df --output=avail -B1 /content | tail -1) / 1000000000 ))
[ "$FREE_GB" -ge "${GRADSNR_MIN_DISK_GB:-80}" ] || { echo "DISK_TOO_SMALL free ${FREE_GB} GB < ${GRADSNR_MIN_DISK_GB:-80}"; exit 5; }
CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\..*/\1/p' | head -1)
PLUG=cuda13; [ "${CUDA_MAJOR:-0}" -ge 13 ] || PLUG=cuda12
echo "driver CUDA ${CUDA_MAJOR:-?} -> jax[$PLUG]==0.11.0"
command -v uv > /dev/null || pip install -q uv
# the downloader gets its own venv: the main install below never swaps files under the running downloader
uv venv -q /content/dlvenv --python 3.12
DLPY=/content/dlvenv/bin/python
uv pip install -q --python "$DLPY" huggingface_hub hf_xet
st downloader_installed

download_all() {
  # every marker not "ok" when this exits (error, kill) gets a .failed: grad_snr.py stops at once
  trap 'for d in "$DATA" "$HFD/te" "$HFD/raw"; do [ -f "$d/.ready" ] || [ -f "$d/.failed" ] || echo "downloader exited before this stage was ready (see the DL lines)" > "$d/.failed"; done; rm -f "$TOKEN"' EXIT
  HF_HUB_DISABLE_PROGRESS_BARS=1 "$DLPY" -I - "$TOKEN" "$IN/samples.json" "$DATA" "$HFD" "$RAW_REPO" "$RAW_REV" \
      "$TE_REPO" "$TE_REV" "$DATA_REPO" <<'EOF'
import json, os, sys, time
from huggingface_hub import hf_hub_download, snapshot_download
tok_path, samples, data, hfd, raw_repo, raw_rev, te_repo, te_rev, data_repo = sys.argv[1:]
with open(tok_path) as f:
    tok = f.read().strip()
spec = json.load(open(samples))

def stage(d, what, fn):
    t0 = time.time()
    try:
        fn()
    except BaseException as e:
        with open(f"{d}/.failed", "w") as f:
            f.write(f"{what}: {e!r}"[:1000])
        raise
    print(f"DL {what} in {time.time() - t0:.0f} s", flush=True)

def mark(d):
    with open(f"{d}/.ready.tmp", "w") as f:
        f.write("ok")
    os.replace(f"{d}/.ready.tmp", f"{d}/.ready")

def latents():
    for name in ["latents/manifest.json"] + ["latents/" + s for s in sorted(spec["shards"])]:
        hf_hub_download(data_repo, name, repo_type="dataset", revision=spec["revision"], local_dir=data, token=tok)

stage(data, f"latents ({len(spec['shards'])} shards)", latents)
mark(data)
stage(f"{hfd}/te", "text encoder", lambda: snapshot_download(
    te_repo, revision=te_rev, local_dir=f"{hfd}/te",
    allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"]))
mark(f"{hfd}/te")
stage(f"{hfd}/raw", "raw.safetensors", lambda: hf_hub_download(
    raw_repo, "raw.safetensors", revision=raw_rev, local_dir=f"{hfd}/raw", token=tok))
EOF
  rm -f "$TOKEN"; echo "SECRET_DELETED hf_token"
  st downloaded
  local size sha
  size=$(stat -c %s "$HFD/raw/raw.safetensors")
  [ "$size" = "$RAW_SIZE" ] || { echo "RAW_SIZE_MISMATCH $size" | tee "$HFD/raw/.failed"; exit 3; }
  sha=$(sha256sum "$HFD/raw/raw.safetensors" | cut -d' ' -f1)
  [ "$sha" = "$RAW_SHA" ] || { echo "RAW_SHA_MISMATCH $sha" | tee "$HFD/raw/.failed"; exit 3; }
  echo ok > "$HFD/raw/.ready.tmp" && mv "$HFD/raw/.ready.tmp" "$HFD/raw/.ready"
  st raw_verified
}
( download_all ) &
DL_PID=$!
echo "downloader pid $DL_PID"

PY=/content/venv/bin/python
uv venv -q /content/venv --python 3.12
uv pip install -q --python "$PY" "jax[$PLUG]==0.11.0" "jaxlib==0.11.0" "optax==0.2.8" "orbax-checkpoint==0.12.7" \
  "safetensors==0.8.0" "ml_dtypes==0.6.0" "numpy==2.5.4" tokenizers pyarrow
"$PY" -c 'import jax; d = jax.devices()[0]; print("jax", jax.__version__, d.platform, d.device_kind, getattr(d, "compute_capability", None))'
st installed
for d in "$DATA" "$HFD/te" "$HFD/raw"; do
  if [ -f "$d/.failed" ]; then echo "DOWNLOAD_FAILED $d: $(cat "$d/.failed")"; exit 3; fi
done
LEFT_MIN=$(( (T0 + CAP_MIN * 60 - $(date +%s)) / 60 - MARGIN_MIN ))
[ "$LEFT_MIN" -gt 5 ] || { echo "NO_TIME_LEFT $LEFT_MIN"; exit 4; }
echo "deadline for grad_snr.py: $LEFT_MIN min"
set +e
env PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 TF_CPP_MIN_LOG_LEVEL=1 "$PY" "$IN/repo/scripts/grad_snr.py" \
  --mode "$MODE" --base-ckpt "$HFD/raw/raw.safetensors" --text-encoder "$HFD/te" --data-dir "$DATA" \
  --samples "$IN/samples.json" --out "$OUT" --scratch "$SCR" --deadline-min "$LEFT_MIN" \
  --wait-inputs "$DATA/.ready" "$HFD/te/.ready" --wait-raw "$HFD/raw/.ready" --downloader-pid "$DL_PID" \
  "$@" 2>&1 | tee "$OUT/grad_snr.log"
RC=${PIPESTATUS[0]}
set -e
st "grad_snr_rc=$RC"
exit "$RC"
