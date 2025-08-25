# ~/executor_agent/mini_agent.py
from __future__ import annotations
import base64, io, os, shlex, subprocess, time, typing, requests
from typing import Optional, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI(title="mini-executor-agent")

# ---------- helpers ----------

def _env(extra: Optional[Dict[str,str]] = None) -> Dict[str,str]:
    e = os.environ.copy()
    # ensure a GUI session context for X11 Budgie
    if "DISPLAY" not in e or not e["DISPLAY"]:
        # try to auto-detect e.g. :0 or :1 from /tmp/.X11-unix
        disp = ":0"
        try:
            xs = [p for p in os.listdir("/tmp/.X11-unix") if p.startswith("X")]
            if xs:
                disp = f":{xs[0][1:]}"
        except Exception:
            pass
        e["DISPLAY"] = disp
    e.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    e.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    e.setdefault("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
    if extra:
        e.update({str(k): str(v) for k, v in extra.items()})
    return e

def _choose_terminal() -> Optional[str]:
    for t in ("x-terminal-emulator","gnome-terminal","kgx","tilix","xfce4-terminal","mate-terminal","konsole","xterm"):
        if shutil.which(t):
            return t
    return None

def _visible_terminal(cmd: str) -> None:
    # try preferred emulator, fall back to xdotool + xterm
    env = _env()
    term = shutil.which("x-terminal-emulator") or shutil.which("gnome-terminal") \
           or shutil.which("kgx") or shutil.which("tilix") or shutil.which("xfce4-terminal") \
           or shutil.which("mate-terminal") or shutil.which("konsole") or shutil.which("xterm")

    quoted = shlex.quote(cmd)
    if term:
        if os.path.basename(term) in {"xterm","x-terminal-emulator"}:
            launch = f'nohup {term} -hold -e bash -lc {quoted} >/tmp/term.log 2>&1 &'
        else:
            launch = f'nohup {term} -- bash -lc {quoted} >/tmp/term.log 2>&1 &'
        subprocess.Popen(launch, shell=True, env=env, executable="/bin/bash")
        return

    # last resort: synthesize the usual GNOME/Budgie hotkey
    if shutil.which("xdotool"):
        subprocess.Popen("xdotool key ctrl+alt+t", shell=True, env=env, executable="/bin/bash")
        time.sleep(0.6)
        subprocess.Popen(f'xdotool type --delay 1 -- {shlex.quote(cmd)}', shell=True, env=env, executable="/bin/bash")
        subprocess.Popen('xdotool key Return', shell=True, env=env, executable="/bin/bash")

import shutil

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
    return {"ok": True}

@app.post("/run")
def run(req: RunReq):
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
    subprocess.Popen(
        f'nohup bash -lc {shlex.quote(req.command)} >/tmp/open_app.log 2>&1 &',
        shell=True, env=_env(), executable="/bin/bash"
    )
    time.sleep(req.wait)
    return {"ok": True}

@app.post("/open_terminal_and_run")
def open_terminal_and_run(req: OTRReq):
    _visible_terminal(req.cmd)
    time.sleep(req.wait_open)
    return {"ok": True}

@app.post("/browse")
def browse(req: BrowseReq):
    # Fire the desktop browser, but also fetch for content summary
    try:
        subprocess.Popen(f'nohup xdg-open {shlex.quote(req.url)} >/dev/null 2>&1 &',
                         shell=True, env=_env(), executable="/bin/bash")
    except Exception:
        pass
    text = ""
    try:
        r = requests.get(req.url, timeout=20, headers={"User-Agent":"Mozilla/5.0 (X11; Linux) mini-agent"})
        text = r.text
    except Exception as e:
        text = f"(fetch error: {e})"
    return {"content": text[:200_000]}

@app.post("/file")
def file_rw(req: FileReq):
    if req.content_b64 is None:
        with open(req.path, "rb") as f:
            b = f.read()
        return {"content_b64": base64.b64encode(b).decode()}
    # write
    os.makedirs(os.path.dirname(req.path) or ".", exist_ok=True)
    with open(req.path, "wb") as f:
        f.write(base64.b64decode(req.content_b64))
    return {"wrote": req.path}

@app.get("/screenshot")
def screenshot():
    try:
        import mss, PIL.Image  # pillow
        with mss.mss() as sct:
            mon = sct.monitors[0]
            img = sct.grab(mon)
            im = PIL.Image.frombytes("RGB", img.size, img.rgb)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return {"png_b64": base64.b64encode(buf.getvalue()).decode()}
    except Exception as e:
        return {"png_b64": None, "error": str(e)}

@app.post("/type")
def type_text(req: TypeReq):
    if shutil.which("xdotool"):
        delay = max(1, int(60000/(req.wpm or 600)))
        subprocess.Popen(f'xdotool type --delay {delay} -- {shlex.quote(req.text)}',
                         shell=True, env=_env(), executable="/bin/bash")
        return {"ok": True, "method":"xdotool"}
    return {"ok": False, "error":"xdotool not installed"}

@app.post("/key")
def keypress(req: KeyReq):
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
