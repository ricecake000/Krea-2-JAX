#!/bin/bash
# Colab G4 VM side of the gradient S/N measurement (scripts/grad_snr.py); started by scripts/grad_snr_launch.sh
# through colab_job.sh, never by hand.
#   bash grad_snr_vm.sh smoke|measure CAP_MIN [grad_snr.py args...]
# Inputs: /root/gradsnr_in.tar (repo/ = git archive HEAD, samples.json), /root/hf_token (deleted right after the
# downloads and on every exit path; read inside python from its path, never in argv, env or logs).
# Steps: GPU / host / disk report -> uv venv with jax[cuda13]==0.11.0 (jax[cuda12] when the driver is < CUDA 13)
# -> downloads (pinned revisions): Raw raw.safetensors (sha256 checked), Qwen3-VL-4B-Instruct (identity checked by
# grad_snr.py), latents/manifest.json + the shards samples.json uses (sha256 checked by grad_snr.py) -> grad_snr.py
# with --deadline-min = what is left of CAP_MIN minus a margin. Output: /content/gradsnr_out.tgz (state.json,
# report.json, report.txt, logs; the P' memmap stays in /content/gradsnr_scratch).
set -euo pipefail
MODE=${1:?mode}; CAP_MIN=${2:?cap}; shift 2
T0=$(date +%s)
st() { echo "STAGE $1 $(date -u +%T) +$(( $(date +%s) - T0 ))s"; }
IN=/content/gs; OUT=/content/gradsnr_out; SCR=/content/gradsnr_scratch; DATA=/content/gs_data; HFD=/content/gs_hf
TOKEN=/root/hf_token
RAW_REPO=krea/Krea-2-Raw; RAW_REV=6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603
RAW_SHA=f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7; RAW_SIZE=26283332608
TE_REPO=Qwen/Qwen3-VL-4B-Instruct; TE_REV=ebb281ec70b05090aa6165b016eac8ec08e71b17
DATA_REPO=ricecake/danbooru-curated-v2-latents
MARGIN_MIN=${GRADSNR_MARGIN_MIN:-12}   # VM start lags the runner's clock + final report + download
pack() { tar czf /content/gradsnr_out.tgz -C /content gradsnr_out 2> /dev/null || true; }
cleanup() { rm -f "$TOKEN"; pack; }
trap cleanup EXIT
[ -f "$TOKEN" ] && chmod 600 "$TOKEN"
mkdir -p "$IN" "$OUT" "$SCR" "$DATA" "$HFD"
tar xf /root/gradsnr_in.tar -C "$IN" && rm -f /root/gradsnr_in.tar
st extracted
{ nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv; nvidia-smi | sed -n 3p
  free -g; df -h /content; echo "nproc $(nproc)"; python3 --version; } 2>&1 | tee "$OUT/vm_env.txt"
# before ~37 GB of downloads: Raw 26.3 + Qwen 8.9 + shards 1.9, then the P' memmap (~29 GB for the default
# --cross-params, refused by grad_snr.py above 0.8 x the free disk) -> need ~80 GB free
FREE_GB=$(( $(df --output=avail -B1 /content | tail -1) / 1000000000 ))
[ "$FREE_GB" -ge "${GRADSNR_MIN_DISK_GB:-80}" ] || { echo "DISK_TOO_SMALL free ${FREE_GB} GB < ${GRADSNR_MIN_DISK_GB:-80}"; exit 5; }
CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\..*/\1/p' | head -1)
PLUG=cuda13; [ "${CUDA_MAJOR:-0}" -ge 13 ] || PLUG=cuda12
echo "driver CUDA ${CUDA_MAJOR:-?} -> jax[$PLUG]==0.11.0"
command -v uv > /dev/null || pip install -q uv
uv venv -q /content/venv --python 3.12
PY=/content/venv/bin/python
uv pip install -q --python "$PY" "jax[$PLUG]==0.11.0" "jaxlib==0.11.0" "optax==0.2.8" "orbax-checkpoint==0.12.7" \
  "safetensors==0.8.0" "ml_dtypes==0.6.0" "numpy==2.5.4" tokenizers pyarrow huggingface_hub hf_xet
"$PY" -c 'import jax; d = jax.devices()[0]; print("jax", jax.__version__, d.platform, d.device_kind, getattr(d, "compute_capability", None))'
st installed
HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" -I - "$TOKEN" "$IN/samples.json" "$DATA" "$HFD" "$RAW_REPO" "$RAW_REV" "$TE_REPO" \
    "$TE_REV" "$DATA_REPO" <<'EOF'
import json, sys, time
from huggingface_hub import hf_hub_download, snapshot_download
tok_path, samples, data, hfd, raw_repo, raw_rev, te_repo, te_rev, data_repo = sys.argv[1:]
with open(tok_path) as f:
    tok = f.read().strip()
spec = json.load(open(samples))
t0 = time.time()
for name in ["latents/manifest.json"] + ["latents/" + s for s in sorted(spec["shards"])]:
    hf_hub_download(data_repo, name, repo_type="dataset", revision=spec["revision"], local_dir=data, token=tok)
print(f"latents: {len(spec['shards'])} shards in {time.time() - t0:.0f} s", flush=True)
t0 = time.time()
snapshot_download(te_repo, revision=te_rev, local_dir=f"{hfd}/te",
                  allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
print(f"text encoder in {time.time() - t0:.0f} s", flush=True)
t0 = time.time()
hf_hub_download(raw_repo, "raw.safetensors", revision=raw_rev, local_dir=f"{hfd}/raw", token=tok)
print(f"raw.safetensors in {time.time() - t0:.0f} s", flush=True)
EOF
rm -f "$TOKEN"; echo "SECRET_DELETED hf_token"
st downloaded
SIZE=$(stat -c %s "$HFD/raw/raw.safetensors")
[ "$SIZE" = "$RAW_SIZE" ] || { echo "RAW_SIZE_MISMATCH $SIZE"; exit 3; }
SHA=$(sha256sum "$HFD/raw/raw.safetensors" | cut -d' ' -f1)
[ "$SHA" = "$RAW_SHA" ] || { echo "RAW_SHA_MISMATCH $SHA"; exit 3; }
st raw_verified
LEFT_MIN=$(( (T0 + CAP_MIN * 60 - $(date +%s)) / 60 - MARGIN_MIN ))
[ "$LEFT_MIN" -gt 5 ] || { echo "NO_TIME_LEFT $LEFT_MIN"; exit 4; }
echo "deadline for grad_snr.py: $LEFT_MIN min"
set +e
env PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 TF_CPP_MIN_LOG_LEVEL=1 "$PY" "$IN/repo/scripts/grad_snr.py" \
  --mode "$MODE" --base-ckpt "$HFD/raw/raw.safetensors" --text-encoder "$HFD/te" --data-dir "$DATA" \
  --samples "$IN/samples.json" --out "$OUT" --scratch "$SCR" --deadline-min "$LEFT_MIN" "$@" 2>&1 | tee "$OUT/grad_snr.log"
RC=${PIPESTATUS[0]}
set -e
st "grad_snr_rc=$RC"
exit "$RC"
