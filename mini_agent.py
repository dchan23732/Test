# ~/executor_agent/mini_agent.py
from __future__ import annotations
import base64, io, os, shlex, subprocess, time, typing, shutil, requests
from typing import Optional, Dict, List
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI(title="mini-executor-agent")

DBG_LOG = "/tmp/mini_agent_debug.log"

def dbg(msg: str) -> None:
    try:
        with open(DBG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")
    except Exception:
        pass

# ---------- helpers ----------

def _detect_display() -> str:
    # Pick the first X socket we see (X0, X1, ...)
    try:
        xs = [p for p in sorted(os.listdir("/tmp/.X11-unix")) if p.startswith("X")]
        if xs:
            d = f":{xs[0][1:]}"
            dbg(f"_detect_display -> {d}")
            return d
    except Exception as e:
        dbg(f"_detect_display error: {e}")
    dbg("_detect_display -> :0 (fallback)")
    return ":0"

def _env(extra: Optional[Dict[str,str]] = None) -> Dict[str,str]:
    e = os.environ.copy()
    # IMPORTANT: always set a known-good DISPLAY from the active X socket.
    e["DISPLAY"] = _detect_display()
    e["XDG_RUNTIME_DIR"] = e.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    e["DBUS_SESSION_BUS_ADDRESS"] = e.get("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    e["XAUTHORITY"] = e.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
    if extra:
        e.update({str(k): str(v) for k, v in extra.items()})
    return e

def _visible_terminal(cmd: str) -> None:
    """
    Try multiple emulators; always create /tmp/term.log so we have breadcrumbs.
    If no emulator is found, fall back to xdotool + xterm.
    """
    env = _env()
    quoted = shlex.quote(cmd)
    term = shutil.which("x-terminal-emulator") or shutil.which("gnome-terminal") \
           or shutil.which("kgx") or shutil.which("tilix") or shutil.which("xfce4-terminal") \
           or shutil.which("mate-terminal") or shutil.which("konsole") or shutil.which("xterm")

    dbg(f"_visible_terminal: DISPLAY={env.get('DISPLAY')} term={term} cmd={cmd}")

    if term:
        if os.path.basename(term) in {"xterm","x-terminal-emulator"}:
            launch = f'nohup {term} -hold -e bash -lc {quoted} >/tmp/term.log 2>&1 &'
        else:
            launch = f'nohup {term} -- bash -lc {quoted} >/tmp/term.log 2>&1 &'
        subprocess.Popen(launch, shell=True, env=env, executable="/bin/bash")
        return

    # last resort: use the hotkey and inject command
    if shutil.which("xdotool"):
        try:
            subprocess.Popen("xdotool key ctrl+alt+t", shell=True, env=env, executable="/bin/bash")
            time.sleep(0.6)
            subprocess.Popen(f'xdotool type --delay 1 -- {quoted}', shell=True, env=env, executable="/bin/bash")
            subprocess.Popen('xdotool key Return', shell=True, env=env, executable="/bin/bash")
            dbg("_visible_terminal: used xdotool fallback")
        except Exception as e:
            dbg(f"_visible_terminal xdotool error: {e}")
    else:
        # no emulator and no xdotool
        dbg("_visible_terminal: no emulator and no xdotool")

# ---------- models ----------

class RunReq(BaseModel):
    cmd: str
    timeout: int = 120
    cwd: Optional[str] = None
    env: Optional[Dict[str,str]] = None

class OpenAppReq(BaseModel):
    command: str
    wait: float = 0.7

class OTRReq(BaseModel):
    cmd: str
    wait_open: float = 0.8

class BrowseReq(BaseModel):
    url: str
    selector: Optional[str] = None
    wait: float = 3.0

class FileReq(BaseModel):
    path: str
    content_b64: Optional[str] = None

class TypeReq(BaseModel):
    text: str
    wpm: int = 600

class KeyReq(BaseModel):
    keys: List[str]

# ---------- endpoints ----------

@app.get("/health")
def health():
    dbg("GET /health")
    return {"ok": True}

@app.post("/run")
def run(req: RunReq):
    dbg(f"POST /run cmd={req.cmd!r}")
    try:
        p = subprocess.run(
            req.cmd, shell=True, cwd=req.cwd, env=_env(req.env),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=req.timeout, text=True, executable="/bin/bash"
        )
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired as e:
        return {"returncode": None, "stdout": e.stdout or "", "stderr": "timeout"}

@app.post("/open_app")
def open_app(req: OpenAppReq):
    dbg(f"POST /open_app command={req.command!r}")
    subprocess.Popen(
        f'nohup bash -lc {shlex.quote(req.command)} >/tmp/open_app.log 2>&1 &',
        shell=True, env=_env(), executable="/bin/bash"
    )
    time.sleep(req.wait)
    return {"ok": True}

@app.post("/open_terminal_and_run")
def open_terminal_and_run(req: OTRReq):
    dbg(f"POST /open_terminal_and_run cmd={req.cmd!r}")
    _visible_terminal(req.cmd)
    time.sleep(req.wait_open)
    return {"ok": True}

@app.post("/browse")
def browse(req: BrowseReq):
    dbg(f"POST /browse url={req.url!r}")
    try:
        subprocess.Popen(f'nohup xdg-open {shlex.quote(req.url)} >/dev/null 2>&1 &',
                         shell=True, env=_env(), executable="/bin/bash")
    except Exception as e:
        dbg(f"/browse xdg-open error: {e}")
    text = ""
    try:
        r = requests.get(req.url, timeout=20, headers={"User-Agent":"Mozilla/5.0 (X11; Linux) mini-agent"})
        text = r.text
    except Exception as e:
        text = f"(fetch error: {e})"
    return {"content": text[:200_000]}

@app.post("/file")
def file_rw(req: FileReq):
    dbg(f"POST /file path={req.path!r} write={req.content_b64 is not None}")
    if req.content_b64 is None:
        with open(req.path, "rb") as f:
            b = f.read()
        return {"content_b64": base64.b64encode(b).decode()}
    os.makedirs(os.path.dirname(req.path) or ".", exist_ok=True)
    with open(req.path, "wb") as f:
        f.write(base64.b64decode(req.content_b64))
    return {"wrote": req.path}

@app.get("/screenshot")
def screenshot():
    dbg("GET /screenshot")
    try:
        import mss, PIL.Image  # pillow
        with mss.mss() as sct:
            mon = sct.monitors[0]
            img = sct.grab(mon)
            from PIL import Image  # alias
            im = Image.frombytes("RGB", img.size, img.rgb)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return {"png_b64": base64.b64encode(buf.getvalue()).decode()}
    except Exception as e:
        dbg(f"/screenshot error: {e}")
        return {"png_b64": None, "error": str(e)}

@app.post("/type")
def type_text(req: TypeReq):
    dbg(f"POST /type len={len(req.text)}")
    if shutil.which("xdotool"):
        delay = max(1, int(60000/(req.wpm or 600)))
        subprocess.Popen(f'xdotool type --delay {delay} -- {shlex.quote(req.text)}',
                         shell=True, env=_env(), executable="/bin/bash")
        return {"ok": True, "method":"xdotool"}
    return {"ok": False, "error":"xdotool not installed"}

@app.post("/key")
def keypress(req: KeyReq):
    dbg(f"POST /key keys={req.keys}")
    if shutil.which("xdotool"):
        keys = " ".join(shlex.quote(k) for k in req.keys)
        subprocess.Popen(f'xdotool key {keys}', shell=True, env=_env(), executable="/bin/bash")
        return {"ok": True}
    return {"ok": False, "error":"xdotool not installed"}

# ---------- main ----------

if __name__ == "__main__":
    import argparse, uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    args = ap.parse_args()
    uvicorn.run("mini_agent:app", host="0.0.0.0", port=args.port, reload=False)
