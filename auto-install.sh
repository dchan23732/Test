#!/usr/bin/env bash
set -euo pipefail

# -- Detect the primary non-root user --
if [ "${SUDO_USER:-}" ]; then TARGET_USER="$SUDO_USER"; else TARGET_USER="$(whoami)"; fi
if [ "$TARGET_USER" = "root" ]; then
  echo "This script is best run with: sudo bash ./agent_install.sh (so TARGET_USER becomes your login user)."
  echo "Continuing with root will install the agent for root's home. Ctrl+C to abort."
fi
TARGET_HOME="$(eval echo ~"$TARGET_USER")"
AGENT_DIR="$TARGET_HOME/executor_agent"
VENV_DIR="$TARGET_HOME/agent-venv"
MNT="/mnt/hostshare"

echo "[1/8] apt packages…"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  qemu-guest-agent spice-vdagent \
  gnome-terminal byobu \
  xdotool wmctrl xclip imagemagick x11-xserver-utils \
  python3 python3-venv python3-pip python-is-python3 \
  git curl unzip jq ripgrep \
  build-essential clang clang-tidy cmake pkg-config \
  shellcheck cppcheck \
  podman uidmap slirp4netns fuse-overlayfs \
  firejail bubblewrap \
  nodejs npm \
  libnss3 libnspr4 libatk-bridge2.0-0 libxkbcommon0 \
  libgtk-3-0 libx11-xcb1 libxcb1 libxcb-dri3-0 libdrm2 \
  libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
  fonts-liberation ca-certificates

echo "[2/8] enable qemu-guest-agent…"
sudo systemctl enable --now qemu-guest-agent || true

echo "[3/8] rootless Podman setup…"
sudo loginctl enable-linger "$TARGET_USER" || true
sudo -u "$TARGET_USER" bash -lc 'mkdir -p ~/.config/containers; cat > ~/.config/containers/storage.conf << EOF
[storage]
driver = "overlay"
graphroot = "'"$HOME"'/.local/share/containers/storage"
EOF'

echo "[4/8] Python venv + agent deps…"
sudo -u "$TARGET_USER" bash -lc "python3 -m venv '$VENV_DIR'"
sudo -u "$TARGET_USER" bash -lc "'$VENV_DIR/bin/pip' install --upgrade pip"
sudo -u "$TARGET_USER" bash -lc "'$VENV_DIR/bin/pip' install fastapi uvicorn[standard] pydantic pillow playwright"

echo "[5/8] Playwright browsers (Chromium)…"
sudo -u "$TARGET_USER" bash -lc "'$VENV_DIR/bin/python' -m playwright install chromium"

echo "[6/8] Agent code…"
sudo -u "$TARGET_USER" bash -lc "mkdir -p '$AGENT_DIR'"

cat <<'PY' | sudo tee "$AGENT_DIR/main.py" >/dev/null
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, shlex, os, base64, tempfile, time
from typing import List, Optional

app = FastAPI(title="Executor Agent")

def _run(cmd: str, timeout: int = 60, cwd: Optional[str] = None, env: Optional[dict] = None):
    e = os.environ.copy()
    if env:
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, shell=True, cwd=cwd, env=e,
                          capture_output=True, text=True, timeout=timeout)

@app.get("/health")
def health():
    return {"ok": True}

class RunReq(BaseModel):
    cmd: str
    timeout: int = 120
    cwd: Optional[str] = None
    env: Optional[dict] = None

@app.post("/run")
def run_cmd(req: RunReq):
    r = _run(req.cmd, timeout=req.timeout, cwd=req.cwd, env=req.env)
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

class AppReq(BaseModel):
    command: str
    wait: float = 0.7

@app.post("/open_app")
def open_app(req: AppReq):
    # Example: {"command":"gnome-terminal -- bash -lc 'byobu || bash'"}
    subprocess.Popen(shlex.split(req.command))
    time.sleep(req.wait)
    return {"ok": True}

class TermReq(BaseModel):
    cmd: str
    wait_open: float = 0.8

@app.post("/open_terminal_and_run")
def open_terminal_and_run(req: TermReq):
    # Launch Terminal with Byobu, then type the command + Enter (X11 only)
    subprocess.Popen(["gnome-terminal", "--", "bash", "-lc", "byobu || bash"])
    time.sleep(req.wait_open)
    # Type the command
    t = req.cmd.replace('"', r'\"')
    _run(f'xdotool type --delay 1 "{t}"')
    _run('xdotool key --clearmodifiers Return')
    return {"ok": True}

class TypeReq(BaseModel):
    text: str
    wpm: int = 600

@app.post("/type")
def type_text(req: TypeReq):
    delay = max(1, int(60000 / max(req.wpm, 1)))
    txt = req.text.replace('"', r'\"')
    r = _run(f'xdotool type --delay {delay} "{txt}"')
    return {"ok": r.returncode == 0, "stderr": r.stderr}

class KeysReq(BaseModel):
    keys: List[str]

@app.post("/key")
def keypress(req: KeysReq):
    seq = "+".join(req.keys)
    r = _run(f'xdotool key --clearmodifiers {seq}')
    return {"ok": r.returncode == 0, "stderr": r.stderr}

@app.get("/screenshot")
def screenshot():
    # Fullscreen screenshot (X11): ImageMagick 'import'
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    r = _run(f'import -window root "{path}"')
    if r.returncode != 0:
        return {"error": r.stderr}
    data = open(path, "rb").read()
    os.remove(path)
    return {"png_b64": base64.b64encode(data).decode()}

class FileReq(BaseModel):
    path: str
    content_b64: Optional[str] = None  # write if provided

@app.post("/file")
def file_ops(req: FileReq):
    if req.content_b64 is not None:
        os.makedirs(os.path.dirname(req.path), exist_ok=True)
        with open(req.path, "wb") as f:
            f.write(base64.b64decode(req.content_b64))
        return {"wrote": req.path}
    else:
        with open(req.path, "rb") as f:
            return {"content_b64": base64.b64encode(f.read()).decode()}

class BrowseReq(BaseModel):
    url: str
    selector: Optional[str] = None
    wait: float = 2.0

@app.post("/browse")
def browse(req: BrowseReq):
    # Headed Chromium so you can see it. Requires a running desktop session.
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(req.url)
        if req.selector:
            page.wait_for_selector(req.selector, timeout=int(req.wait*1000))
            text = page.inner_text(req.selector)
        else:
            page.wait_for_timeout(int(req.wait*1000))
            text = page.content()
        browser.close()
    return {"content": text[:200000]}
PY
sudo chown "$TARGET_USER:$TARGET_USER" "$AGENT_DIR/main.py"

echo "[7/8] systemd service…"
/usr/bin/env bash -lc "grep -q executor-agent@ /etc/systemd/system/executor-agent@.service 2>/dev/null || true"
cat <<'UNIT' | sudo tee /etc/systemd/system/executor-agent@.service >/dev/null
[Unit]
Description=Executor Agent (%i)
After=network-online.target
Wants=network-online.target

[Service]
User=%i
WorkingDirectory=/home/%i
Environment=PYTHONUNBUFFERED=1
Environment=PATH=/home/%i/agent-venv/bin:/usr/bin
ExecStart=/home/%i/agent-venv/bin/uvicorn executor_agent.main:app --host 0.0.0.0 --port 8900
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now "executor-agent@${TARGET_USER}"

echo "[8/8] virtiofs mount…"
sudo mkdir -p "$MNT"
if ! grep -q "^hostshare " /etc/fstab; then
  echo "hostshare  $MNT  virtiofs  defaults,_netdev  0  0" | sudo tee -a /etc/fstab >/dev/null
fi
sudo mount -a || true

IP="$(hostname -I | awk '{print $1}')"
echo
echo "✅ Done. Agent running on: http://$IP:8900"
echo "Shared folder (virtiofs) mounted at: $MNT"
echo "Try from host: curl http://$IP:8900/health"
