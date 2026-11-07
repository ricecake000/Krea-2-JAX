#!/usr/bin/env python3
"""ComfyUI smoke render (VM): headless main.py + POST /prompt of API-format workflows, must-fail gate per render.

  comfy_smoke.py --comfy DIR --workflow API_SMOKE_JSON (--render NAME:UNET_FILE[:load] ... | --renders-json F) --out DIR
        [--prompt TEXT] [--seed N] [--width W --height H] [--port 8188] [--min-std 6.0] [--timeout-s 900]
        [--startup-s 300] [--stop-s 60] [--pidfile F] [--extra-args "..."]
  Worst-case wall time = startup-s + timeout-s per --render + stop-s (+ seconds): a caller's outer timeout must exceed it.
  Each --render patches the workflow (located by class_type: the one UNETLoader, KSampler, SaveImage, EmptyLatentImage;
  prompt text into the CLIPTextEncode wired to KSampler.positive; W/H into the nodes feeding EmptyLatentImage, which
  also feed ModelSamplingFlux) and queues it. UNET_FILE must already be in DIR/models/diffusion_models.
  ':load' = load check: the same graph with steps=1 (the file loads and runs one step).
  A graph with KSamplerAdvanced nodes (the Raw -> Turbo hybrid) keeps its own UNETLoaders: pass UNET_FILE '-'; the
  seed goes to every KSamplerAdvanced, the prompt into the positive of the one starting at step 0.
Gate (each render), checked in this order:
  1. keys: the ComfyUI log written during the render has no "unet missing" / "unet unexpected" / "left over keys in
     diffusion model" / "Could not detect model type" / "ERROR UNSUPPORTED" line -> FATAL (also for ':load').
  2. status: history status success. RESOURCE only for: the terminal execution_error's exception_type/message naming
     an out-of-memory / allocation error, our server killed by SIGKILL (host OOM killer), or the render exceeding
     --timeout-s; any other failure -> FATAL.
  Fail closed: the port must be bindable before our server starts, and readiness counts only when /system_stats
  answers AND the LISTEN socket on the port belongs to our server's process group (/proc/net/tcp inode -> fd).
  3. image: exactly one PNG; it decodes; spatial variation: max over R,G,B of the per-channel spatial std
     >= --min-std (0..255; ':load' uses 1.0) AND at most 95 % of the pixels near-black (all channels <= 10).
     This rejects every constant-colour image (all per-channel stds 0) and black / mostly black outputs (where NaN
     latents end up after SaveImage's uint8 cast) -> FATAL.
Output: SMOKE_OK name=.. per render, or SMOKE_FAIL kind=fatal|resource name=.. <why>; PNGs copied to --out.
Exit 0 all ok; 3 only RESOURCE failures; 1 any FATAL. The server (its own process group) is stopped and reaped at
the end, on failure and on SIGTERM / SIGINT; its pid is written to --pidfile while it runs.
"""
import argparse, copy, json, os, re, shutil, signal, socket, subprocess, sys, time, urllib.error, urllib.request

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
    adv = [k for k, v in wf.items() if v.get("class_type") == "KSamplerAdvanced"]
    if adv:   # the hybrid graph: its own two UNETLoaders (UNET_FILE '-'), seed into every KSamplerAdvanced
        if unet != "-" or load_only:
            raise SystemExit("SMOKE_FAIL kind=fatal a KSamplerAdvanced (hybrid) graph takes UNET_FILE '-' and no ':load'")
        for k in adv:
            set_input(wf, k, "noise_seed", seed)
        ks = next(wf[k]["inputs"] for k in adv if wf[k]["inputs"].get("start_at_step") == 0)
    else:
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


def group_members(pgid):
    """pids whose process group is pgid (Linux /proc)."""
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    st = f.read()
                if int(st[st.rindex(")") + 2:].split()[2]) == pgid:
                    out.append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    return out


def stop_server(srv, budget_s=60):
    """TERM, then KILL the server's whole process group; wait (bounded: budget_s in total) until no member remains."""
    if srv is None:
        return
    for sig, wait in ((signal.SIGTERM, budget_s / 2), (signal.SIGKILL, budget_s / 2)):
        try:
            os.killpg(srv.pid, sig)
        except OSError:
            pass
        t = time.time()
        while time.time() - t < wait:
            if srv.poll() is not None and not group_members(srv.pid):
                return
            time.sleep(0.5)
    srv.poll()


def port_free(port):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port)); return True
    except OSError:
        return False
    finally:
        s.close()


def listener_pgid(port):
    """Process group of the process holding the LISTEN socket on 127.0.0.1:port (via /proc/net/tcp inode), or None."""
    inode = None
    with open("/proc/net/tcp") as f:
        for ln in f.readlines()[1:]:
            c = ln.split()
            if c[1] == f"0100007F:{port:04X}" and c[3] == "0A":
                inode = c[9]
    if not inode:
        return None
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                for fd in os.listdir(f"/proc/{d}/fd"):
                    if os.readlink(f"/proc/{d}/fd/{fd}") == f"socket:[{inode}]":
                        return os.getpgid(int(d))
            except OSError:
                pass
    return None


def terminal_error(st):
    """(exception_type, exception_message) of the LAST execution_error in a history status, or None."""
    err = None
    for m in st.get("messages", []) or []:
        if isinstance(m, list) and len(m) == 2 and m[0] == "execution_error" and isinstance(m[1], dict):
            err = (str(m[1].get("exception_type", "")), str(m[1].get("exception_message", "")))
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True); ap.add_argument("--workflow", required=True)
    ap.add_argument("--render", action="append", default=[]); ap.add_argument("--out", required=True)
    ap.add_argument("--renders-json", help="JSON list of {name, unet, prompt, seed, width, height[, load]}: per-render "
                                          "overrides, rendered after the --render specs on the same server")
    ap.add_argument("--prompt"); ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--width", type=int, default=0); ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--port", type=int, default=8188); ap.add_argument("--min-std", type=float, default=6.0)
    ap.add_argument("--timeout-s", type=int, default=900, help="render budget per --render")
    ap.add_argument("--startup-s", type=int, default=300, help="server startup budget")
    ap.add_argument("--stop-s", type=int, default=60, help="server stop budget (TERM half, KILL half)")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--pidfile")
    a = ap.parse_args()
    renders = []
    for spec in a.render:
        parts = spec.split(":")
        renders.append(dict(name=parts[0], unet=parts[1], load=len(parts) > 2 and parts[2] == "load", prompt=a.prompt,
                            seed=a.seed, width=a.width, height=a.height))
    if a.renders_json:
        for r in json.load(open(a.renders_json)):
            renders.append(dict(name=r["name"], unet=r["unet"], load=bool(r.get("load")), prompt=r.get("prompt", a.prompt),
                                seed=int(r.get("seed", a.seed)), width=int(r.get("width", a.width)),
                                height=int(r.get("height", a.height))))
    if not renders:
        raise SystemExit("SMOKE_FAIL kind=fatal no --render / --renders-json")
    if any(bool(r["width"]) != bool(r["height"]) or r["width"] % 16 or r["height"] % 16 for r in renders):
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
        if not port_free(a.port):   # fail closed: never talk to a server that is not ours
            say(f"SMOKE_FAIL kind=fatal port {a.port} is busy before our server started"); return 1
        srv = subprocess.Popen(cmd, cwd=a.comfy, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        if a.pidfile:
            with open(a.pidfile, "w") as f:
                with open(f"/proc/{srv.pid}/stat") as st:   # "<pgid> <starttime>": a recycled pid never matches
                    start = st.read().rsplit(")", 1)[1].split()[19]
                f.write(f"{srv.pid} {start}\n")
        t0 = time.time()
        while True:
            try:
                if srv.poll() is None:
                    http(a.port, "/system_stats", timeout=5)
                    if srv.poll() is None and listener_pgid(a.port) == srv.pid:
                        break
                    say(f"SMOKE_FAIL kind=fatal port {a.port} answers but its listener is not our server"); return 1
                raise OSError("server exited")
            except OSError:
                if srv.poll() is not None or time.time() - t0 > a.startup_s:
                    say(f"SMOKE_FAIL kind=fatal server did not come up (rc={srv.poll()}); see {logp}"); return 1
                time.sleep(2)
        say(f"SMOKE_SERVER_UP {time.time() - t0:.0f}s port={a.port}")
        for r in renders:
            name, unet, load_only = r["name"], r["unet"], r["load"]
            logf.flush(); log_off = os.path.getsize(logp)
            wf = patch(wf0, unet, r["prompt"], r["seed"], f"smoke_{name}", load_only, r["width"], r["height"])
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
                err = terminal_error(st)   # OOM only from the terminal execution_error, never from inputs / old logs
                why = f"status={st.get('status_str')} server_rc={srv.poll()} after {secs:.0f}s"
                if timed_out:
                    say(f"SMOKE_FAIL kind=resource name={name} render exceeded --timeout-s {a.timeout_s} ({why})"); resource += 1
                elif srv.poll() == -signal.SIGKILL or (err and RESOURCE_RE.search(" ".join(err))):
                    say(f"SMOKE_FAIL kind=resource name={name} {why}: "
                        f"{'server SIGKILLed (OOM killer)' if srv.poll() == -signal.SIGKILL else err}"); resource += 1
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
        stop_server(srv, a.stop_s)
        logf.close()
        if a.pidfile:
            try:
                os.remove(a.pidfile)
            except OSError:
                pass
    return 1 if fatal else (3 if resource else 0)


if __name__ == "__main__":
    sys.exit(main())
