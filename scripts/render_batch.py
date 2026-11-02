"""Mass-render prompts with the JAX Krea 2 DiT into AVIF files (resumable).

    python scripts/render_batch.py --transformer DIR --prompts FILE --out OUTDIR
        [--batch B (1)] [--limit N] [--attn aiter|ref|cudnn (aiter)]
        [--avif-quality Q (70)] [--decode-device gpu|cpu (gpu)]
        [--text-encoder ID] [--vae ID] [--profile-dir DIR [--profile-steps K]]

Turbo defaults: --steps 8 --cfg 0 --mu 1.15 (fixed mu at every resolution,
as the served maxdiffusion Turbo pipeline). Text: the train_full JAX Qwen3-VL
path with --text-rows 512 (= the served max_sequence_length; captions are cut
at 507 tokens), padded text kept (no compaction) so each resolution compiles
the sampler exactly once.

Inputs
  --transformer  a diffusers/maxdiffusion transformer dir (index json + shards,
                 e.g. the f174 serve snapshot) or an official single
                 safetensors. Every tensor keeps its STORED dtype (f174: 174
                 F32 norm/modulation/projection tensors + 256 BF16); the
                 forward runs with act_dtype=bf16 (bf16 residual stream and
                 GEMMs, the F32 linears and modulation tables computed in F32;
                 see krea2_jax/model.py dit_forward).
  --prompts      {"version":1,"prompts":[{"id","prompt","width","height",
                 "seed","aspect"}, ...]}; ids unique, sizes multiples of 16.
  --limit N      only the first N prompts of the file (in file order).

Outputs (OUTDIR)
  img/<id>.avif      written atomically (tmp + rename)
  manifest.jsonl     one line per finished image (id, prompt, w, h, seed,
                     steps, mu, sample_s, decode_s, encode_s, bytes, utc, ...),
                     appended + fsynced only after its .avif is in place
  render.log         `RENDER done=n/N last_id=.. s_per_img=.. ETA=..` per image,
                     plus load/compile/profile lines; final
                     `RENDER_SUMMARY images=.. wall_s=.. img_per_h=.. avg_bytes=..`
  tmp/               latent hand-off files (deleted once decoded)
Resume: ids in manifest.jsonl whose .avif exists are skipped; just rerun the
same command. A manifest record counts only when newline-terminated: a torn
last record (even complete JSON) is cut at startup BEFORE the done set is
computed, so its image renders again. Exit 0 when every (limited) prompt is
done, 3 after a clean SIGTERM/SIGINT stop (pipeline drained, manifest
consistent), 1 on errors.

Clean stop (SIGTERM/SIGINT): no new batch is admitted from that moment. The
DiT batch being sampled is aborted at its next sampler-step boundary (steps
are synchronised for this; a batch whose first step is still compiling ends
after that compile + step) and dropped, as is a finished batch still waiting
for room in the decode queue; everything already handed to the decoder is
decoded, encoded and recorded, then the process exits 3. Worst case after
the signal, with Q = --queue-depth, W = --encode-workers, MP = megapixels of
the largest pending prompt:
    images to decode  (Q + 1) * B            (decode queue + the job in hand)
    images to encode  (2Q + 2) * B + 2W      (+ decode_out queue, the result
                                              the collector holds, the pool)
    DRAIN_BUDGET_S = ceil(1.5 * MP * (B * step_s + (Q + 1) * B * dec_s
                          + ceil(((2Q + 2) * B + 2W) / W) * enc_s) + 10)
with per-megapixel seconds step_s (one sampler step, one sample;
--est-step-s-per-mp 0.5, a conservative estimate, not measured on the node),
dec_s (one image; --est-decode-s-per-mp, 3.0 on gpu = conservative estimate,
60 on cpu = 58 s measured per 1k image) and enc_s (one image, one thread;
--est-encode-s-per-mp 1.0, measured 0.7-0.8 at q70 s6). Printed once at
startup as the verbatim line `DRAIN_BUDGET_S=<n>`: send TERM at least that
long before the deadline (lower it with a smaller --queue-depth or --batch).
It is also a hard limit: once the stop is DRAIN_BUDGET_S old (measured from
the first signal) with work still pending (a hung VAE decode, a stalled AVIF
encode), the drain gives up, logs the verbatim line `DRAIN_TIMEOUT
pending=<ids>` and goes straight into the bounded shutdown below (decoder and
pool SIGKILLed); exit 3 when every image recorded in this run has its .avif
(the pending ids render on resume), else 1. A DRAIN_TIMEOUT on a healthy
pipeline means the --est-* constants are too low: raise them. Without a stop
request the end-of-run drain is NOT time-limited (it waits for every
submitted job; a hang there is the caller's timeout's job, and its TERM then
starts the bounded drain).

Children: the decode worker and the AVIF pool processes ignore SIGINT/SIGTERM
(the main process drains) and each runs a watchdog thread that hard-exits
(os._exit) within ~1 s once the main process is gone (kill -9 included), and
every blocking queue put/get in the decoder is a timeout loop. Shutdown is
bounded on every path: the decoder gets its sentinel when it can still take
it, is joined with a timeout and then SIGKILLed; the pool is waited for (up
to DRAIN_BUDGET_S on a clean path, 5 s after an error), shut down, and every
worker still alive is SIGKILLed, so no child outlives the main process.

Pipeline
  main process (JAX only, never imports torch): text tower + DiT resident on
  ONE device; prompts grouped by (width, height) in file order of first
  appearance, batches of B (a short last batch is padded with copies of its
  last prompt, outputs dropped, so no extra compile); each row's noise comes
  from its own seed (independent of B and of resume order). Latents (f32) go
  to tmp/ and into a bounded queue (--queue-depth batches) to a torch decode
  worker process (spawn) that decodes image by image on --decode-device and
  sends uint8 images back; AVIF encoding runs in a process pool
  (--encode-workers, at most 2x that many images in flight). Every queue is
  bounded, so a slow stage stalls the DiT instead of growing memory.

GPU memory (one MI355X, 288 GB HBM): JAX preallocates
XLA_PYTHON_CLIENT_MEM_FRACTION of the device; with --decode-device gpu this
script sets it to --jax-mem-fraction (default 0.6 -> ~173 GB), overriding any
inherited value (the kit default 0.92 would starve the torch worker).
JAX needs ~26.3 GB DiT (12.50B bf16 + 0.32B f32 params) + ~8 GB Qwen3-VL-4B
text tower (bf16) + activations at 2k (L = 512 text + 16384 image tokens:
~2 GB per sample per block with aiter, e.g. ~16 GB at B=8; ref attention adds
~1.7 GB of f32 logits per sample per query chunk) -> < 60 GB at B=8. The
torch VAE decode (Qwen-Image VAE, f32, one image at a time) measured on CPU:
1024^2 peak 7.9 GiB above the weights; 2048^2 (4x pixels) did not fit a
14 GiB cap -> ~32 GiB projected, plus allocator caching -> the remaining
~115 GB leave ample room. (CPU decode is no option on the node: 58 s per 1k
image on 4 cores.)

Profiling (--profile-dir DIR, --profile-steps K=2): the first pending batch
of the first resolution group renders as warm-up (compile; saved normally);
after the decode queue drained, K sampler steps of the next batch of that
group (or a discarded re-run of the first batch when the group has only one)
are traced with jax.profiler (perfetto trace.json.gz under
DIR/plugins/profile/<run>/; every traced step is a
StepTraceAnnotation("sampler_step", step_num=i) that waits for the step's
device work), and that batch's VAE decode runs alone and is logged as
`PROFILE decode_s=..`. Warm-up and profile phases are excluded from s_per_img,
ETA and img_per_h. Analyse with
    krea2-ops/mi350x/tools/trace_breakdown.py DIR --steps K \
        --step-regex '^sampler_step$'

AVIF default (Pillow >= 11.2 built-in AVIF, libavif; measured 11/03 on 8
native ~2k Krea-2 anime renders, 3.5-4.1 MP, speed 6, 4:2:0, 1 thread):
q50 143 KB (96-179) 38.4 dB PSNR, q60 195 KB (129-251) 40.4 dB, q70 257 KB
(171-340) 42.0 dB, q80 347 KB (232-456) 43.3 dB; ~2.8 s per image per
thread at q70; 4:4:4 costs +5 % bytes for +0.4 dB. q70 (default) keeps the
fine brush/hatching texture that q60 visibly smooths at 2x zoom; no
blocking or ringing at either.

ROCm notes: needs JAX_PLATFORMS=rocm,cpu (the loader stages tensors on the
CPU device; ',cpu' is appended here when missing) and, for --attn aiter,
XLA_FLAGS=--xla_gpu_enable_command_buffer= (forced here, replacing any other
value of that flag).
"""

import argparse
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STEPS, CFG, MU = 8, 0.0, 1.15
TEXT_ROWS = 512
AVIF_QUALITY, AVIF_SPEED, AVIF_SUBSAMPLING = 70, 6, "4:2:0"
JAX_MEM_FRACTION = 0.6
MANIFEST, LOG = "manifest.jsonl", "render.log"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transformer", required=True,
                    help="diffusers transformer dir or official safetensors")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--attn", choices=["aiter", "ref", "cudnn"],
                    default="aiter")
    ap.add_argument("--avif-quality", type=int, default=AVIF_QUALITY)
    ap.add_argument("--avif-speed", type=int, default=AVIF_SPEED,
                    help="libavif speed 0 (slow, small) .. 10")
    ap.add_argument("--avif-subsampling", default=AVIF_SUBSAMPLING,
                    choices=["4:2:0", "4:2:2", "4:4:4"])
    ap.add_argument("--decode-device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--decode-dtype", choices=["float32", "bfloat16"],
                    default="float32")
    ap.add_argument("--text-encoder", default=None,
                    help="HF id or local dir (default: the config's "
                         "Qwen/Qwen3-VL-4B-Instruct)")
    ap.add_argument("--vae", default=None,
                    help="HF id or local dir holding vae/ (default: "
                         "Qwen/Qwen-Image)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cfg", type=float, default=CFG)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--text-rows", type=int, default=TEXT_ROWS)
    ap.add_argument("--compact-text", action="store_true",
                    help="drop padded text rows (fewer tokens, but one "
                         "compile per text-length bucket and resolution)")
    ap.add_argument("--jax-mem-fraction", type=float,
                    default=JAX_MEM_FRACTION,
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION with --decode-device "
                         "gpu (the torch worker shares the GPU)")
    ap.add_argument("--encode-workers", type=int,
                    default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--queue-depth", type=int, default=2,
                    help="latent batches waiting for the decoder")
    ap.add_argument("--est-step-s-per-mp", type=float, default=0.5,
                    help="DRAIN_BUDGET_S: sampler step seconds per sample "
                         "per megapixel")
    ap.add_argument("--est-decode-s-per-mp", type=float, default=None,
                    help="DRAIN_BUDGET_S: VAE decode seconds per megapixel "
                         "(default 3.0 on gpu, 60 on cpu)")
    ap.add_argument("--est-encode-s-per-mp", type=float, default=1.0,
                    help="DRAIN_BUDGET_S: AVIF encode seconds per megapixel")
    ap.add_argument("--profile-dir", default=None)
    ap.add_argument("--profile-steps", type=int, default=2)
    ap.add_argument("--model-config", default=None,
                    help="JSON Krea2Config overrides (tests: tiny model)")
    ap.add_argument("--text-overrides", default=None,
                    help="JSON from_hf_config overrides (tests)")
    args = ap.parse_args(argv)
    if args.batch < 1 or (args.limit is not None and args.limit < 0):
        ap.error("--batch must be >= 1 and --limit >= 0")
    if args.profile_dir and not 1 <= args.profile_steps <= args.steps:
        ap.error("--profile-steps must be in 1..--steps")
    return args


# ------------------------------------------------------------------ plan
def load_prompts(path, limit=None):
    data = json.loads(Path(path).read_text())
    prompts = data["prompts"] if isinstance(data, dict) else data
    if limit is not None:
        prompts = prompts[:limit]
    seen = set()
    for p in prompts:
        pid = str(p["id"])
        if pid in seen or "/" in pid or pid.startswith("."):
            raise ValueError(f"bad or duplicate prompt id {pid!r}")
        seen.add(pid)
        if p["width"] % 16 or p["height"] % 16:
            raise ValueError(f"{pid}: {p['width']}x{p['height']} is not a "
                             "multiple of 16")
    return prompts


def _manifest_bytes(out):
    """(bytes, cut): the manifest and the length of its complete part (a
    record counts only when newline-terminated)."""
    path = Path(out) / MANIFEST
    data = path.read_bytes() if path.exists() else b""
    return data, data.rfind(b"\n") + 1


def repair_manifest(out):
    """Cut a torn last record (killed mid-write), complete JSON or not."""
    data, cut = _manifest_bytes(out)
    if cut != len(data):
        with open(Path(out) / MANIFEST, "r+b") as f:
            f.truncate(cut)
            os.fsync(f.fileno())
    return len(data) - cut


def read_manifest(out):
    """{id: row} of the parseable, newline-terminated manifest records."""
    rows = {}
    data, cut = _manifest_bytes(out)
    for line in data[:cut].decode().splitlines():
        try:
            row = json.loads(line)
            rows[str(row["id"])] = row
        except (ValueError, KeyError, TypeError):
            continue
    return rows


def done_ids(out):
    out = Path(out)
    return {i for i in read_manifest(out)
            if (out / "img" / f"{i}.avif").is_file()}


def plan_batches(prompts, batch):
    """[(width, height, [prompts])]: grouped by resolution in order of first
    appearance, file order within a group, chunks of `batch`."""
    groups = {}
    for p in prompts:
        groups.setdefault((p["width"], p["height"]), []).append(p)
    return [(w, h, ps[i:i + batch]) for (w, h), ps in groups.items()
            for i in range(0, len(ps), batch)]


def drain_budget_s(args, todo):
    """Worst-case seconds from SIGTERM to exit (see the module docstring)."""
    B, Q, W = args.batch, max(1, args.queue_depth), args.encode_workers
    mp_ = max((p["width"] * p["height"] for p in todo), default=0) / 1e6
    dec = args.est_decode_s_per_mp
    if dec is None:
        dec = 3.0 if args.decode_device == "gpu" else 60.0
    t = mp_ * (B * args.est_step_s_per_mp + (Q + 1) * B * dec
               + math.ceil(((2 * Q + 2) * B + 2 * W) / W)
               * args.est_encode_s_per_mp)
    return math.ceil(1.5 * t + 10)


def fmt_hms(s):
    if s is None or not math.isfinite(s):
        return "?"
    s = int(round(s))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


# --------------------------------------------------------- worker side
def encode_avif(img, path, quality, speed, subsampling):
    """uint8 (H, W, 3) -> AVIF at `path` (atomic). Returns (bytes, seconds)."""
    from PIL import Image

    t0 = time.time()
    tmp = f"{path}.tmp.{os.getpid()}"
    Image.fromarray(img).save(tmp, format="AVIF", quality=quality,
                              speed=speed, subsampling=subsampling,
                              max_threads=1)   # parallelism = the pool
    os.replace(tmp, path)
    return os.path.getsize(path), time.time() - t0


def watch_parent(parent, period=1.0):
    """Daemon thread: hard-exit this process once `parent` is gone (no
    finalizers, so a queue feeder blocked on a dead reader cannot hold it)."""
    def run():
        while os.getppid() == parent:
            time.sleep(period)
        os._exit(1)
    threading.Thread(target=run, daemon=True, name="parent-watchdog").start()


def child_init(parent):
    """Children leave SIGINT/SIGTERM to the main process, which drains, and
    die with it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    watch_parent(parent)


def decode_worker(in_q, out_q, vae_id, device, dtype, parent):
    """Torch process: latent files from in_q -> uint8 images to out_q.
    Messages out: ("ready", load_s), ("result", job, images, decode_s list),
    ("error", text), ("done",). Exits on its own when the main process is
    gone (watchdog thread + liveness re-checks in every put/get loop: no
    orphan holding GPU memory after a kill -9 of the main)."""
    child_init(parent)

    def alive():
        if os.getppid() != parent:
            os._exit(1)

    def put(msg):
        while True:
            try:
                out_q.put(msg, timeout=2)
                return
            except queue.Full:
                alive()

    try:
        import numpy as np
        import torch
        from diffusers import AutoencoderKLQwenImage

        t0 = time.time()
        dev = "cuda" if device == "gpu" else "cpu"   # ROCm torch: "cuda"
        tdt = getattr(torch, dtype)
        vae = (AutoencoderKLQwenImage.from_pretrained(
            vae_id, subfolder="vae", torch_dtype=tdt)
            .to(dev).eval().requires_grad_(False))
        mean = torch.tensor(vae.config.latents_mean, dtype=tdt,
                            device=dev).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, dtype=tdt,
                           device=dev).view(1, -1, 1, 1, 1)
        put(("ready", time.time() - t0))
        while True:
            try:
                job = in_q.get(timeout=2)
            except queue.Empty:
                alive()
                continue
            if job is None:
                put(("done",))
                return
            lat = np.load(job["path"])
            images, secs = [], []
            for i in range(lat.shape[0]):
                t0 = time.time()
                # the math of scripts/sample.py decode_latents
                z = torch.from_numpy(lat[i:i + 1]).to(dev, tdt).unsqueeze(2)
                with torch.no_grad():
                    img = vae.decode(z * std + mean).sample[:, :, 0]
                img = (img.float().clamp(-1, 1) * 0.5 + 0.5) * 255.0
                images.append(img.permute(0, 2, 3, 1).byte().cpu().numpy()[0])
                secs.append(time.time() - t0)
            os.remove(job["path"])
            put(("result", job, images, secs))
    except BaseException:  # noqa: BLE001 - reported to the main process
        put(("error", traceback.format_exc()))


# ------------------------------------------------------------ main side
class StopRequested(Exception):
    """SIGTERM/SIGINT seen at a sampler-step boundary: drop this batch."""


class Renderer:
    """Owns the decode worker, the encode pool, the manifest and the log."""

    def __init__(self, args, out, total, already, drain_budget):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self.args, self.out = args, out
        self.total, self.done = total, already
        self.drain_budget = drain_budget
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.fatal = None
        self.stop = False
        self.stop_t = None           # monotonic time of the first signal
        self.timed_out = False       # the stop's drain budget ran out
        self.abandon = False         # close() gave up the remaining encodes
        self.pending = {}            # job n -> ids sent to the decoder
        self.enc_ids = set()         # ids in the AVIF pool
        self.run_ids = set()         # ids recorded in this run
        self.closing = False         # close() started: collector stops
        self.closed = False          # manifest closed: no more records
        self.inflight = 0            # jobs sent to the decoder, not back yet
        self.encoding = 0            # images in the AVIF pool
        self.run_imgs = 0            # images finished in this run
        self.run_bytes = 0
        self.excluded_s = 0.0        # warm-up/profile wall time
        self.excluded_imgs = 0
        self.excluded_ids = set()
        self.t_start = None          # first batch start
        self.logf = open(out / LOG, "a", buffering=1)
        repair_manifest(out)   # main() did already; cheap, keeps the rule
        self.manf = open(out / MANIFEST, "ab")
        ctx = mp.get_context("spawn")
        self.dec_in = ctx.Queue(maxsize=max(1, args.queue_depth))
        self.dec_out = ctx.Queue(maxsize=max(1, args.queue_depth))
        self.worker = ctx.Process(
            target=decode_worker, daemon=True,
            args=(self.dec_in, self.dec_out, args.vae, args.decode_device,
                  args.decode_dtype, os.getpid()))
        self.worker.start()
        self.pool = ProcessPoolExecutor(max_workers=args.encode_workers,
                                        mp_context=ctx,
                                        initializer=child_init,
                                        initargs=(os.getpid(),))
        self.slots = threading.Semaphore(2 * args.encode_workers)
        self.collector = threading.Thread(target=self._collect, daemon=True)
        self.collector.start()

    def log(self, msg):
        """RENDER*/PROFILE/DRAIN lines verbatim (machine-read), others
        stamped."""
        line = msg if msg.startswith(("RENDER", "PROFILE", "DRAIN")) else (
            f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}")
        with self.lock:
            print(line, flush=True)
            self.logf.write(line + "\n")

    def _fail(self, msg):
        with self.cond:
            if self.fatal is None:
                self.fatal = msg
            self.cond.notify_all()
        self.log(f"ERROR {msg}")

    def _quit(self):
        return self.closing or self.fatal is not None

    def _collect(self):
        while not self._quit():
            try:
                msg = self.dec_out.get(timeout=2)
            except queue.Empty:
                if not self.worker.is_alive() and not self._quit():
                    self._fail(f"decode worker died (exit "
                               f"{self.worker.exitcode})")
                    return
                continue
            kind = msg[0]
            if kind == "done":
                return
            if kind == "error":
                self._fail("decode worker:\n" + msg[1])
                return
            if kind == "ready":
                self.log(f"decode worker ready ({self.args.decode_device}, "
                         f"{self.args.decode_dtype}, VAE load {msg[1]:.1f} s)")
                continue
            _, job, images, secs = msg
            if job.get("profile"):
                self.log(f"PROFILE decode_s={sum(secs):.3f} images="
                         f"{len(secs)} (alone on the device, per image "
                         f"{', '.join(f'{s:.3f}' for s in secs)})")
            if not job.get("discard"):
                for rec, img, ds in zip(job["recs"], images, secs):
                    while not self.slots.acquire(timeout=2):
                        if self._quit():
                            return
                    if self._quit():
                        self.slots.release()
                        return
                    path = self.out / "img" / f"{rec['id']}.avif"
                    a = self.args
                    with self.cond:
                        self.encoding += 1
                        self.enc_ids.add(rec["id"])
                    try:
                        fut = self.pool.submit(encode_avif, img, str(path),
                                               a.avif_quality, a.avif_speed,
                                               a.avif_subsampling)
                    except RuntimeError as e:   # pool shut down / broken
                        self._encode_finished(rec["id"])
                        if not self._quit():
                            self._fail(f"AVIF pool: {e!r}")
                        return
                    fut.add_done_callback(
                        lambda f, rec=rec, ds=ds: self._encoded(f, rec, ds))
            with self.cond:
                self.inflight -= 1
                self.pending.pop(job["n"], None)
                self.cond.notify_all()

    def _encode_finished(self, rid):
        self.slots.release()
        with self.cond:
            self.encoding -= 1
            self.enc_ids.discard(rid)
            self.cond.notify_all()

    def _encoded(self, fut, rec, decode_s):
        try:
            self._record(fut, rec, decode_s)
        finally:
            self._encode_finished(rec["id"])

    def _record(self, fut, rec, decode_s):
        try:
            nbytes, enc_s = fut.result()
        except BaseException as e:  # noqa: BLE001
            if not self.abandon:   # else: cancelled/killed by close()
                self._fail(f"AVIF encode of {rec['id']} failed: {e!r}")
            return
        row = dict(rec, decode_s=round(decode_s, 3), encode_s=round(enc_s, 3),
                   bytes=nbytes,
                   utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with self.lock:
            if self.closed:   # gave up waiting; the image renders on resume
                return
            self.manf.write((json.dumps(row) + "\n").encode())
            self.manf.flush()
            os.fsync(self.manf.fileno())
            self.done += 1
            self.run_imgs += 1
            self.run_ids.add(rec["id"])
            self.run_bytes += nbytes
            if rec["id"] in self.excluded_ids:
                self.excluded_imgs += 1
            spi = self.s_per_img()
            n, N = self.done, self.total
        self.log(f"RENDER done={n}/{N} last_id={rec['id']} s_per_img="
                 f"{spi:.2f} ETA={fmt_hms(spi * (N - n))}"
                 if spi is not None else
                 f"RENDER done={n}/{N} last_id={rec['id']} s_per_img=? ETA=?")

    def s_per_img(self):
        """Wall seconds per image this run, warm-up/profile excluded once a
        steady-state image exists (before that: all images, so it is always
        a number once one image is done)."""
        if self.t_start is None or self.run_imgs <= 0:
            return None
        wall = time.time() - self.t_start
        imgs = self.run_imgs - self.excluded_imgs
        if imgs <= 0 or wall - self.excluded_s <= 0:
            return wall / self.run_imgs
        return (wall - self.excluded_s) / imgs

    def check(self):
        if self.fatal is not None:
            raise RuntimeError(self.fatal)

    def request_stop(self):
        """Signal handler side: no new batch; the drain deadline runs from
        the FIRST signal (a repeated signal does not extend it)."""
        if self.stop_t is None:
            self.stop_t = time.monotonic()
        self.stop = True

    def deadline_hit(self):
        """True once a stop was requested DRAIN_BUDGET_S ago; the first
        call that sees it logs `DRAIN_TIMEOUT pending=<ids>`. Never call it
        while holding self.cond (log() takes the same lock)."""
        if self.stop_t is None or (time.monotonic()
                                   < self.stop_t + self.drain_budget):
            return False
        if not self.timed_out:
            self.timed_out = True
            with self.cond:
                ids = sorted({i for v in self.pending.values() for i in v}
                             | self.enc_ids)
            self.log(f"DRAIN_TIMEOUT pending={','.join(ids) or '-'}")
        return True

    def step_ctx(self, _i):
        """Per sampler step (the sampler syncs each step when given one):
        abort the batch on an error or a stop request."""
        import contextlib

        self.check()
        if self.stop:
            raise StopRequested
        return contextlib.nullcontext()

    def submit(self, job):
        """Hand a latent file to the decoder (blocks while the queue is
        full, i.e. backpressure). False (job dropped, latent file deleted)
        when a stop request arrives first."""
        while True:
            self.check()
            if self.stop:
                Path(job["path"]).unlink(missing_ok=True)
                return False
            try:
                with self.cond:
                    self.inflight += 1
                    self.pending[job["n"]] = [r["id"] for r in job["recs"]]
                self.dec_in.put(job, timeout=1)
                return True
            except queue.Full:
                with self.cond:
                    self.inflight -= 1
                    self.pending.pop(job["n"], None)
                if not self.worker.is_alive():
                    self._fail("decode worker died")

    def drain_decoder(self):
        """Wait until every submitted job came back from the decoder.
        Unbounded without a stop request (a hang there is the caller's
        timeout's job); once a stop was requested, StopRequested when its
        drain budget runs out (DRAIN_TIMEOUT, close() then kills)."""
        while True:
            with self.cond:
                if self.inflight <= 0 or self.fatal is not None:
                    break
                self.cond.wait(timeout=1)
            if self.deadline_hit():
                raise StopRequested
        self.check()

    def close(self):
        """Bounded on every path: stop the decoder (sentinel when it can
        still take it, join with a timeout, then SIGKILL), wait for the
        encodes (up to the drain budget; 5 s after an error), shut the pool
        down and SIGKILL any worker left, close the manifest. Once a stop
        was requested every wait also ends at the stop's drain deadline;
        after a DRAIN_TIMEOUT (or an error) the decoder gets no sentinel
        and the kills follow within seconds."""
        def quick():
            return self.fatal is not None or self.deadline_hit()

        if not quick() and self.worker.is_alive():
            try:
                self.dec_in.put(None, timeout=5)
            except queue.Full:
                pass
            else:   # the collector returns on "done" after the last result
                end = time.monotonic() + self.drain_budget
                while (self.collector.is_alive() and not quick()
                       and time.monotonic() < end):
                    self.collector.join(timeout=1)
        self.closing = True
        self.collector.join(timeout=5)
        self.worker.join(timeout=1 if quick() else 5)
        if self.worker.is_alive():
            self.log("killing the decode worker")
            self.worker.kill()
            self.worker.join(timeout=10)
        self.dec_in.cancel_join_thread()   # no exit-time feeder join
        end = time.monotonic() + (self.drain_budget if self.fatal is None
                                  else 5)
        while True:
            with self.cond:
                left = self.encoding
            if not left or quick() or time.monotonic() >= end:
                break
            with self.cond:
                self.cond.wait_for(lambda: self.encoding == 0, timeout=1)
        with self.lock:
            self.abandon = True   # cancelled/killed encodes are no error
        if left:   # log() takes the lock: never call it under self.cond
            self.log(f"giving up on {left} AVIF encodes (render on resume)")
        procs = list((self.pool._processes or {}).values())
        self.pool.shutdown(wait=False, cancel_futures=True)
        end = time.monotonic() + (1 if left or quick() else 5)
        for p in procs:
            p.join(timeout=max(0.0, end - time.monotonic()))
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(timeout=10)
        with self.lock:
            self.closed = True
            self.manf.close()


def check_text_fit(txt_cfg, template, cfg):
    problems = []
    if txt_cfg.prefix_len != template.prefix_len:
        problems.append(f"prefix_len {txt_cfg.prefix_len} != tokenizer "
                        f"prefix {template.prefix_len}")
    if txt_cfg.hidden_size != cfg.txtdim:
        problems.append(f"hidden {txt_cfg.hidden_size} != txtdim {cfg.txtdim}")
    if len(txt_cfg.select_layers) != cfg.txtlayers:
        problems.append(f"{len(txt_cfg.select_layers)} select_layers != "
                        f"txtlayers {cfg.txtlayers}")
    return problems


def setup_env(args):
    """Env that must be set before jax is imported."""
    if args.decode_device == "gpu":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.jax_mem_fraction)
    plats = os.environ.get("JAX_PLATFORMS")
    if plats and "cpu" not in plats.split(","):
        os.environ["JAX_PLATFORMS"] = plats + ",cpu"
    if args.attn == "aiter":   # aiter FFI + HIP graph capture crash
        keep = [f for f in os.environ.get("XLA_FLAGS", "").split()
                if not f.startswith("--xla_gpu_enable_command_buffer")]
        os.environ["XLA_FLAGS"] = " ".join(
            keep + ["--xla_gpu_enable_command_buffer="])


def main(argv=None):
    args = parse_args(argv)
    setup_env(args)   # before anything can import jax
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "tmp").mkdir(exist_ok=True)
    for f in (out / "tmp").glob("*"):
        f.unlink()
    for f in (out / "img").glob("*.avif.tmp.*"):   # killed mid-encode
        f.unlink()
    prompts = load_prompts(args.prompts, args.limit)
    torn = repair_manifest(out)   # BEFORE the done set (record = full line)
    done = done_ids(out)
    todo = [p for p in prompts if str(p["id"]) not in done]
    already = len(prompts) - len(todo)

    import dataclasses
    from krea2_jax.config import KREA2_CONFIG

    cfg = KREA2_CONFIG
    if args.model_config:
        cfg = dataclasses.replace(cfg, **json.loads(args.model_config))
    args.text_encoder = args.text_encoder or cfg.text_encoder_id
    args.vae = args.vae or cfg.vae_id
    if not todo:
        msg = (f"RENDER_SUMMARY images=0 wall_s=0 img_per_h=0 avg_bytes=0 "
               f"done={already}/{len(prompts)} (nothing to do)")
        print(msg)
        with open(out / LOG, "a") as f:
            f.write(msg + "\n")
        return 0

    budget = drain_budget_s(args, todo)
    r = Renderer(args, out, len(prompts), already, budget)
    stop_sig = []

    def on_signal(signum, _frame):
        stop_sig.append(signum)
        r.request_stop()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    r.log(f"start: {len(todo)} to render, {already} already done, "
          f"{len(prompts)} total; batch {args.batch}, attn {args.attn}, "
          f"steps {args.steps}, cfg {args.cfg}, mu {args.mu}, AVIF q"
          f"{args.avif_quality} s{args.avif_speed} {args.avif_subsampling}; "
          f"XLA_PYTHON_CLIENT_MEM_FRACTION="
          f"{os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'unset')}"
          + (f"; cut a torn manifest record ({torn} bytes)" if torn else ""))
    r.log(f"DRAIN_BUDGET_S={budget}")
    rc = 1
    try:
        rc = render(args, cfg, r, todo)
    except StopRequested:   # profile phase (close() drains) or DRAIN_TIMEOUT
        if not r.timed_out:
            r.log("stop requested during profiling: draining")
        rc = 0
    except BaseException as e:  # noqa: BLE001
        if r.fatal is None:
            r.log(f"ERROR {e!r}\n{traceback.format_exc()}")
        rc = 1
    finally:
        try:
            r.close()
        except BaseException as e:  # noqa: BLE001
            r.log(f"ERROR while closing: {e!r}")
            rc = 1
    if r.fatal is not None:
        rc = 1
    elif r.timed_out and rc == 0:   # invariant: record only a written .avif
        bad = sorted(i for i in r.run_ids
                     if not (out / "img" / f"{i}.avif").is_file())
        if bad:
            r.log(f"ERROR recorded without an .avif: {bad}")
        rc = 1 if bad else 3
    final = done_ids(out) & {str(p["id"]) for p in prompts}
    wall = time.time() - r.t_start if r.t_start else 0.0
    spi = r.s_per_img()
    r.log(f"RENDER_SUMMARY images={r.run_imgs} wall_s={wall:.1f} "
          f"img_per_h={3600 / spi if spi else 0:.1f} avg_bytes="
          f"{r.run_bytes // max(r.run_imgs, 1)} done={len(final)}/"
          f"{len(prompts)} excluded_s={r.excluded_s:.1f} "
          f"load_s={getattr(r, 'load_s', 0):.1f}")
    if rc == 0 and len(final) < len(prompts):
        rc = 3 if stop_sig else 1
    r.logf.close()
    return rc


def render(args, cfg, r, todo):
    import contextlib

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    from krea2_jax.checkpoint import load_krea2_params
    from krea2_jax.qwen3_text import (
        from_hf_config, load_qwen3_text_params, load_tokenizer,
        resolve_text_encoder, text_states,
    )
    from krea2_jax.sampling import sample_latents
    from krea2_jax.text_template import CaptionTemplate

    t0 = time.time()
    dev = jax.devices()[0]
    mesh = Mesh(np.array([dev]), axis_names=("fsdp",))
    te_dir = resolve_text_encoder(args.text_encoder)
    txt_cfg = from_hf_config(te_dir, **json.loads(args.text_overrides or "{}"))
    template = CaptionTemplate(load_tokenizer(te_dir), text_rows=args.text_rows)
    problems = check_text_fit(txt_cfg, template, cfg)
    if problems:
        raise RuntimeError(f"text encoder {te_dir} does not fit: {problems}")
    with jax.default_device(dev):
        txt_params = load_qwen3_text_params(te_dir, jnp.bfloat16, cfg=txt_cfg)
    encode = jax.jit(text_states, static_argnums=1)
    params = load_krea2_params(args.transformer, mesh=mesh, dtype=None,
                               cfg=cfg)
    dts = {}
    for leaf in jax.tree.leaves(params):
        dts[str(leaf.dtype)] = dts.get(str(leaf.dtype), 0) + 1
    r.load_s = time.time() - t0
    r.log(f"loaded text tower {te_dir} + DiT {args.transformer} on {dev} in "
          f"{r.load_s:.1f} s; DiT leaf dtypes {dts} (stored dtypes kept, "
          f"act_dtype bf16); backend {jax.default_backend()}")

    B = args.batch
    use_cfg = args.cfg > 0

    def text(captions):
        ids, txt_len = template.encode_captions(captions)
        return encode(txt_params, txt_cfg, ids, txt_len)

    un = text([args.negative_prompt] * B) if use_cfg else (None, None)

    def run_batch(w, h, batch, step_ctx=r.step_ctx):
        """Sample one batch -> (f32 latents of the real rows, seconds)."""
        t0 = time.time()
        pad = batch + [batch[-1]] * (B - len(batch))
        states, mask = text([p["prompt"] for p in pad])
        lat = sample_latents(
            params, cfg, states, mask, untxt=un[0], untxt_mask=un[1],
            height=h, width=w, steps=args.steps, guidance=args.cfg,
            mu=args.mu, seeds=[int(p["seed"]) for p in pad],
            attn_impl=args.attn, attn_mesh=None, compact=args.compact_text,
            cfg_batch=True, act_dtype=jnp.bfloat16, step_ctx=step_ctx)
        arr = np.asarray(jax.device_get(lat).astype(jnp.float32))[:len(batch)]
        return arr, time.time() - t0

    njob = [0]

    def hand_off(w, h, batch, arr, secs, **flags):
        if not np.isfinite(arr).all():
            r.log(f"WARNING non-finite latents in batch {[p['id'] for p in batch]}")
        path = r.out / "tmp" / f"job{njob[0]:06d}.npy"
        np.save(path.with_suffix(".part.npy"), arr)
        os.replace(path.with_suffix(".part.npy"), path)
        recs = [{"id": str(p["id"]), "prompt": p["prompt"], "w": w, "h": h,
                 "seed": int(p["seed"]), "aspect": p.get("aspect"),
                 "steps": args.steps, "mu": args.mu, "cfg": args.cfg,
                 "batch": B, "attn": args.attn,
                 "sample_s": round(secs / len(batch), 3)} for p in batch]
        njob[0] += 1
        if not r.submit({"n": njob[0] - 1, "path": str(path), "recs": recs,
                         **flags}):
            r.log(f"stop requested: dropped the finished batch "
                  f"{[p['id'] for p in batch]} (renders on resume)")

    batches = plan_batches(todo, B)
    r.t_start = time.time()
    first = 0
    if args.profile_dir:
        w, h, b0 = batches[0]
        t_ex = time.time()
        r.excluded_ids |= {str(p["id"]) for p in b0}
        arr, secs = run_batch(w, h, b0)
        r.log(f"PROFILE warm-up batch {w}x{h} ({len(b0)} img, compile + run) "
              f"{secs:.1f} s")
        hand_off(w, h, b0, arr, secs)
        r.drain_decoder()   # the trace sees the DiT alone
        same = len(batches) > 1 and batches[1][:2] == (w, h)
        pb = batches[1][2] if same else b0
        if same:
            r.excluded_ids |= {str(p["id"]) for p in pb}
        K = args.profile_steps
        from contextlib import ExitStack

        def step_ctx(i):
            r.step_ctx(i)   # stop/error check at every step
            if i >= K:
                return contextlib.nullcontext()
            st = ExitStack()
            if i == K - 1:   # LIFO: runs after the annotation has closed
                st.callback(jax.profiler.stop_trace)
            st.enter_context(jax.profiler.StepTraceAnnotation(
                "sampler_step", step_num=i))
            return st

        try:
            jax.profiler.start_trace(args.profile_dir,
                                     create_perfetto_trace=True)
        except TypeError:
            jax.profiler.start_trace(args.profile_dir)
        arr, secs = run_batch(w, h, pb, step_ctx=step_ctx)
        r.log(f"PROFILE written {args.profile_dir} ({K} sampler steps of a "
              f"{w}x{h} batch of {len(pb)}{'' if same else ', re-run, discarded'};"
              f" batch {secs:.1f} s incl. profiler overhead)")
        hand_off(w, h, pb, arr, secs, profile=True, discard=not same)
        r.drain_decoder()   # the decode runs alone, logged as PROFILE
        r.excluded_s = time.time() - t_ex
        first = 2 if same else 1

    shapes_seen = {tuple(batches[0][:2])} if args.profile_dir else set()
    for w, h, batch in batches[first:]:
        r.check()
        if r.stop:
            break
        try:
            arr, secs = run_batch(w, h, batch)
        except StopRequested:
            r.log(f"stop requested: aborted the batch "
                  f"{[p['id'] for p in batch]} (renders on resume)")
            break
        if (w, h) not in shapes_seen:
            shapes_seen.add((w, h))
            r.log(f"first batch at {w}x{h} (incl. compile) {secs:.1f} s")
        hand_off(w, h, batch, arr, secs)
    if r.stop:
        r.log("stop requested: draining the decoder and the AVIF pool")
    r.drain_decoder()
    return 0


if __name__ == "__main__":
    sys.exit(main())
