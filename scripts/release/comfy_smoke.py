#!/usr/bin/env python3
"""ComfyUI smoke render (VM): headless main.py + POST /prompt of API-format workflows, must-fail gate per render.

  comfy_smoke.py --comfy DIR --workflow API_SMOKE_JSON --render NAME:UNET_FILE[:load] ... --out DIR
        [--prompt TEXT] [--seed N] [--port 8188] [--min-std 6.0] [--timeout-s 1800] [--extra-args "..."]
  Each --render patches the workflow (located by class_type: the one UNETLoader, KSampler, SaveImage; prompt text
  into the CLIPTextEncode wired to KSampler.positive) and queues it. UNET_FILE must already be in
  DIR/models/diffusion_models. ':load' = load-only check: the same graph with steps=1 (the file loads and runs one
  step; the image gate is skipped).
Gate (each render): history status success; the ComfyUI log written during it has no "unet missing" /
  "unet unexpected" / "left over keys in diffusion model" / "Could not detect model type" / Traceback line; the
  saved PNG is finite (it decodes) and not constant: pixel std >= --min-std (0..255 scale).
Output: SMOKE_OK name=.. secs=.. std=.. file=.. per render, SMOKE_FAIL name=.. <why>; the PNGs are copied to --out.
Exit 0 all ok, 1 any failure. The server is started once and stopped at the end (also on failure).
"""
import argparse, copy, json, os, shutil, signal, subprocess, sys, time, urllib.request

BAD_LOG = ("unet missing", "unet unexpected", "left over keys in diffusion model", "Could not detect model type",
           "Traceback (most recent call last)", "ERROR UNSUPPORTED")


def http(port, path, data=None, timeout=30):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def one(wf, cls):
    ids = [k for k, v in wf.items() if v.get("class_type") == cls]
    if len(ids) != 1:
        raise SystemExit(f"SMOKE_FAIL workflow has {len(ids)} {cls} nodes (exactly 1 expected)")
    return ids[0]


def patch(wf, unet, prompt, seed, prefix, load_only):
    wf = copy.deepcopy(wf)
    wf[one(wf, "UNETLoader")]["inputs"]["unet_name"] = unet
    ks = wf[one(wf, "KSampler")]["inputs"]
    ks["seed"] = seed
    if load_only:
        ks["steps"] = 1
    pos = ks["positive"][0]
    if wf[pos]["class_type"] != "CLIPTextEncode":
        raise SystemExit(f"SMOKE_FAIL KSampler.positive comes from {wf[pos]['class_type']}, not CLIPTextEncode")
    if prompt is not None:
        wf[pos]["inputs"]["text"] = prompt
    wf[one(wf, "SaveImage")]["inputs"]["filename_prefix"] = prefix
    return wf


def png_std(path):
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return float(a.std()) if np.isfinite(a).all() else float("nan"), a.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True); ap.add_argument("--workflow", required=True)
    ap.add_argument("--render", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prompt"); ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--port", type=int, default=8188); ap.add_argument("--min-std", type=float, default=6.0)
    ap.add_argument("--timeout-s", type=int, default=1800); ap.add_argument("--extra-args", default="")
    a = ap.parse_args()
    wf0 = json.load(open(a.workflow))
    os.makedirs(a.out, exist_ok=True)
    logp = os.path.join(a.out, "comfyui.log")
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(a.port), "--disable-auto-launch",
           "--output-directory", os.path.join(a.out, "comfy_out")] + a.extra_args.split()
    logf = open(logp, "ab")
    srv = subprocess.Popen(cmd, cwd=a.comfy, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    fails = 0
    try:
        t0 = time.time()
        while True:
            try:
                http(a.port, "/system_stats", timeout=5); break
            except OSError:
                if srv.poll() is not None or time.time() - t0 > 300:
                    print(f"SMOKE_FAIL server did not come up (rc={srv.poll()}); see {logp}", flush=True); return 1
                time.sleep(2)
        print(f"SMOKE_SERVER_UP {time.time() - t0:.0f}s", flush=True)
        for spec in a.render:
            parts = spec.split(":")
            name, unet, load_only = parts[0], parts[1], len(parts) > 2 and parts[2] == "load"
            logf.flush(); log_off = os.path.getsize(logp)
            wf = patch(wf0, unet, a.prompt, a.seed, f"smoke_{name}", load_only)
            t1 = time.time()
            try:
                pid = http(a.port, "/prompt", {"prompt": wf, "client_id": "krea2-release-smoke"})["prompt_id"]
            except urllib.error.HTTPError as e:
                print(f"SMOKE_FAIL name={name} /prompt rejected: {e.read()[:2000]!r}", flush=True); fails += 1; continue
            h = None
            while time.time() - t1 < a.timeout_s:
                h = http(a.port, f"/history/{pid}").get(pid)
                if h and h.get("status", {}).get("completed") is not None and (h["status"]["completed"] or h["status"].get("status_str") == "error"):
                    break
                if srv.poll() is not None:
                    break
                time.sleep(3)
            secs = time.time() - t1
            with open(logp, "rb") as f:
                f.seek(log_off); new = f.read().decode("utf-8", "replace")
            bad = [ln for ln in new.splitlines() if any(b in ln for b in BAD_LOG)]
            st = (h or {}).get("status", {})
            if not h or st.get("status_str") != "success":
                print(f"SMOKE_FAIL name={name} status={st.get('status_str')} server_rc={srv.poll()} after {secs:.0f}s; "
                      f"messages={json.dumps(st.get('messages', []))[-1500:]}", flush=True); fails += 1; continue
            if bad:
                print(f"SMOKE_FAIL name={name} load log: {bad[:3]}", flush=True); fails += 1; continue
            imgs = [i for o in h.get("outputs", {}).values() for i in o.get("images", [])]
            if len(imgs) != 1:
                print(f"SMOKE_FAIL name={name} {len(imgs)} images saved (1 expected)", flush=True); fails += 1; continue
            src = os.path.join(a.out, "comfy_out", imgs[0].get("subfolder", ""), imgs[0]["filename"])
            dst = os.path.join(a.out, f"{name}.png"); shutil.copyfile(src, dst)
            std, shape = png_std(dst)
            if not load_only and not std >= a.min_std:
                print(f"SMOKE_FAIL name={name} pixel std {std:.2f} < {a.min_std} (constant / blank image) shape={shape}", flush=True)
                fails += 1; continue
            print(f"SMOKE_OK name={name} unet={unet} {'load-only(1 step)' if load_only else 'render'} secs={secs:.0f} "
                  f"std={std:.1f} shape={shape} file={dst}", flush=True)
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM); srv.wait(30)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except OSError:
                pass
        logf.close()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
