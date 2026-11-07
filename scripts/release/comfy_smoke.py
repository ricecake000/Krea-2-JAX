#!/usr/bin/env python3
"""ComfyUI smoke render (VM): headless main.py + POST /prompt of API-format workflows, must-fail gate per render.

  comfy_smoke.py --comfy DIR --workflow API_SMOKE_JSON --render NAME:UNET_FILE[:load] ... --out DIR
        [--prompt TEXT] [--seed N] [--width W --height H] [--port 8188] [--min-std 6.0] [--timeout-s 1500]
        [--pidfile F] [--extra-args "..."]
  Each --render patches the workflow (located by class_type: the one UNETLoader, KSampler, SaveImage, EmptyLatentImage;
  prompt text into the CLIPTextEncode wired to KSampler.positive; W/H into the nodes feeding EmptyLatentImage, which
  also feed ModelSamplingFlux) and queues it. UNET_FILE must already be in DIR/models/diffusion_models.
  ':load' = load check: the same graph with steps=1 (the file loads and runs one step).
Gate (each render), checked in this order:
  1. keys: the ComfyUI log written during the render has no "unet missing" / "unet unexpected" / "left over keys in
     diffusion model" / "Could not detect model type" / "ERROR UNSUPPORTED" line -> FATAL (also for ':load').
  2. status: history status success. A failure whose messages or log carry an out-of-memory / allocation error, a
     server killed by SIGKILL (host OOM killer), or the render exceeding --timeout-s -> RESOURCE; any other -> FATAL.
  3. image: exactly one PNG; it decodes; spatial variation: max over R,G,B of the per-channel spatial std
     >= --min-std (0..255; ':load' uses 1.0) AND at most 95 % of the pixels near-black (all channels <= 10).
     This rejects every constant-colour image (all per-channel stds 0) and black / mostly black outputs (where NaN
     latents end up after SaveImage's uint8 cast) -> FATAL.
Output: SMOKE_OK name=.. per render, or SMOKE_FAIL kind=fatal|resource name=.. <why>; PNGs copied to --out.
Exit 0 all ok; 3 only RESOURCE failures; 1 any FATAL. The server (its own process group) is stopped and reaped at
the end, on failure and on SIGTERM / SIGINT; its pid is written to --pidfile while it runs.
"""
import argparse, copy, json, os, re, shutil, signal, subprocess, sys, time, urllib.error, urllib.request

KEY_LOG = ("unet missing", "unet unexpected", "left over keys in diffusion model", "Could not detect model type",
           "ERROR UNSUPPORTED")
RESOURCE_RE = re.compile(r"OutOfMemoryError|out of memory|Allocation on device|CUBLAS_STATUS_ALLOC_FAILED|"
                         r"Cannot allocate memory|MemoryError", re.I)


class Stop(Exception):
    pass


def on_signal(signum, _frame):
    raise Stop(signum)


def http(port, path, data=None, timeout=30):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def one(wf, cls):
    ids = [k for k, v in wf.items() if v.get("class_type") == cls]
    if len(ids) != 1:
        raise SystemExit(f"SMOKE_FAIL kind=fatal workflow has {len(ids)} {cls} nodes (exactly 1 expected)")
    return ids[0]


def set_input(wf, node, name, value):
    v = wf[node]["inputs"][name]
    if isinstance(v, list):   # linked: set the source node's value (PrimitiveInt also feeds ModelSamplingFlux)
        wf[v[0]]["inputs"]["value"] = value
    else:
        wf[node]["inputs"][name] = value


def patch(wf, unet, prompt, seed, prefix, load_only, width, height):
    wf = copy.deepcopy(wf)
    wf[one(wf, "UNETLoader")]["inputs"]["unet_name"] = unet
    ks = wf[one(wf, "KSampler")]["inputs"]
    ks["seed"] = seed
    if load_only:
        ks["steps"] = 1
    pos = ks["positive"][0]
    if wf[pos]["class_type"] != "CLIPTextEncode":
        raise SystemExit(f"SMOKE_FAIL kind=fatal KSampler.positive comes from {wf[pos]['class_type']}")
    if prompt is not None:
        wf[pos]["inputs"]["text"] = prompt
    if width:
        lat = one(wf, "EmptyLatentImage")
        set_input(wf, lat, "width", width); set_input(wf, lat, "height", height)
        for k, v in wf.items():   # every node taking width/height must now agree (ModelSamplingFlux = the shift)
            if v["class_type"] == "ModelSamplingFlux":
                set_input(wf, k, "width", width); set_input(wf, k, "height", height)
    wf[one(wf, "SaveImage")]["inputs"]["filename_prefix"] = prefix
    return wf


def image_gate(path, min_std):
    """(ok, detail): max per-channel spatial std >= min_std and <= 95 % near-black pixels."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    stds = a.reshape(-1, 3).std(axis=0)
    dark = float((a <= 10).all(axis=2).mean())
    detail = f"channel_std={[round(float(x), 1) for x in stds]} dark_frac={dark:.3f} shape={a.shape}"
    return bool(stds.max() >= min_std and dark <= 0.95), detail


def stop_server(srv):
    if srv is None or srv.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        try:
            os.killpg(srv.pid, sig)
        except OSError:
            pass
        try:
            srv.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True); ap.add_argument("--workflow", required=True)
    ap.add_argument("--render", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prompt"); ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--width", type=int, default=0); ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--port", type=int, default=8188); ap.add_argument("--min-std", type=float, default=6.0)
    ap.add_argument("--timeout-s", type=int, default=1500); ap.add_argument("--extra-args", default="")
    ap.add_argument("--pidfile")
    a = ap.parse_args()
    if bool(a.width) != bool(a.height) or a.width % 16 or a.height % 16:
        raise SystemExit("SMOKE_FAIL kind=fatal --width/--height: both, multiples of 16")
    for sg in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sg, on_signal)
    wf0 = json.load(open(a.workflow))
    os.makedirs(a.out, exist_ok=True)
    logp = os.path.join(a.out, "comfyui.log")
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(a.port), "--disable-auto-launch",
           "--output-directory", os.path.join(a.out, "comfy_out")] + a.extra_args.split()
    logf = open(logp, "ab")
    srv, fatal, resource = None, 0, 0
    say = lambda s: print(s, flush=True)
    try:
        srv = subprocess.Popen(cmd, cwd=a.comfy, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        if a.pidfile:
            with open(a.pidfile, "w") as f:
                f.write(f"{srv.pid}\n")
        t0 = time.time()
        while True:
            try:
                http(a.port, "/system_stats", timeout=5); break
            except OSError:
                if srv.poll() is not None or time.time() - t0 > 300:
                    say(f"SMOKE_FAIL kind=fatal server did not come up (rc={srv.poll()}); see {logp}"); return 1
                time.sleep(2)
        say(f"SMOKE_SERVER_UP {time.time() - t0:.0f}s port={a.port}")
        for spec in a.render:
            parts = spec.split(":")
            name, unet, load_only = parts[0], parts[1], len(parts) > 2 and parts[2] == "load"
            logf.flush(); log_off = os.path.getsize(logp)
            wf = patch(wf0, unet, a.prompt, a.seed, f"smoke_{name}", load_only, a.width, a.height)
            t1 = time.time()
            try:
                pid = http(a.port, "/prompt", {"prompt": wf, "client_id": "krea2-release-smoke"})["prompt_id"]
            except urllib.error.HTTPError as e:
                say(f"SMOKE_FAIL kind=fatal name={name} /prompt rejected: {e.read()[:2000]!r}"); fatal += 1; continue
            h, timed_out = None, True
            while time.time() - t1 < a.timeout_s:
                try:
                    h = http(a.port, f"/history/{pid}").get(pid)
                except OSError:   # the server is gone (checked below) or busy: poll again
                    h = None
                st = (h or {}).get("status", {})
                if h and (st.get("completed") or st.get("status_str") == "error"):
                    timed_out = False; break
                if srv.poll() is not None:
                    timed_out = False; break
                time.sleep(3)
            secs = time.time() - t1
            with open(logp, "rb") as f:
                f.seek(log_off); new = f.read().decode("utf-8", "replace")
            keys = [ln for ln in new.splitlines() if any(b in ln for b in KEY_LOG)]
            if keys:
                say(f"SMOKE_FAIL kind=fatal name={name} key mismatch in the load log: {keys[:3]}"); fatal += 1; continue
            st = (h or {}).get("status", {})
            if st.get("status_str") != "success":
                msgs = json.dumps(st.get("messages", []))
                why = f"status={st.get('status_str')} server_rc={srv.poll()} after {secs:.0f}s"
                if timed_out:
                    say(f"SMOKE_FAIL kind=resource name={name} render exceeded --timeout-s {a.timeout_s} ({why})"); resource += 1
                elif RESOURCE_RE.search(msgs) or RESOURCE_RE.search(new) or srv.poll() == -signal.SIGKILL:
                    m = RESOURCE_RE.search(msgs) or RESOURCE_RE.search(new)
                    say(f"SMOKE_FAIL kind=resource name={name} {why}: {m.group(0) if m else 'server SIGKILLed'}"); resource += 1
                else:
                    say(f"SMOKE_FAIL kind=fatal name={name} {why}; messages={msgs[-1500:]}"); fatal += 1
                if srv.poll() is not None:
                    break
                continue
            imgs = [i for o in h.get("outputs", {}).values() for i in o.get("images", [])]
            if len(imgs) != 1:
                say(f"SMOKE_FAIL kind=fatal name={name} {len(imgs)} images saved (1 expected)"); fatal += 1; continue
            src = os.path.join(a.out, "comfy_out", imgs[0].get("subfolder", ""), imgs[0]["filename"])
            dst = os.path.join(a.out, f"{name}.png"); shutil.copyfile(src, dst)
            try:
                ok, detail = image_gate(dst, 1.0 if load_only else a.min_std)
            except Exception as e:  # noqa: BLE001 - an undecodable PNG fails the gate
                ok, detail = False, f"PNG does not decode: {e}"
            if not ok:
                say(f"SMOKE_FAIL kind=fatal name={name} image gate: {detail} (constant / blank / mostly black)"); fatal += 1; continue
            say(f"SMOKE_OK name={name} unet={unet} {'load(1 step)' if load_only else 'render'} secs={secs:.0f} {detail} file={dst}")
    except Stop as e:
        say(f"SMOKE_FAIL kind=fatal stopped by signal {e.args[0]}: stopping the server")
        fatal += 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_server(srv)
        logf.close()
        if a.pidfile:
            try:
                os.remove(a.pidfile)
            except OSError:
                pass
    return 1 if fatal else (3 if resource else 0)


if __name__ == "__main__":
    sys.exit(main())
