cat > ~/fix_termux_ubuntu_gui.sh <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
set -e

# Keep Android/Quest from killing Termux
termux-wake-lock || true
# Also: on Quest, set Termux to unrestricted battery/background if available.

# Ensure proot-distro is present
if ! command -v proot-distro >/dev/null 2>&1; then
  pkg update -y
  pkg install -y proot-distro
fi

distro="ubuntu"
if ! proot-distro list | grep -q "$distro"; then
  echo "Installing Ubuntu proot…"
  proot-distro install ubuntu
fi

# All fixes happen inside Ubuntu
proot-distro login "$distro" -- bash -lc '
set -e

# 1) DNS + IPv4 preference
printf "nameserver 1.1.1.1\nnameserver 9.9.9.9\n" > /etc/resolv.conf

if ! grep -q "precedence ::ffff:0:0/96 100" /etc/gai.conf 2>/dev/null; then
  echo "precedence ::ffff:0:0/96 100" >> /etc/gai.conf
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates tzdata locales curl wget \
  tigervnc-standalone-server dbus-x11 lxqt xterm
update-ca-certificates || true

# 2) Locale/time (helps TLS)
sed -i "s/^# *en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen
locale-gen
update-locale LANG=en_US.UTF-8

# 3) Remove display managers (flaky in proot)
apt-get purge -y lightdm sddm gdm3 || true
apt-get autoremove -y || true

# 4) VNC xstartup for LXQt
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup << "EOS"
#!/usr/bin/env bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export DISPLAY=${DISPLAY:-:1}
if [ -z "$DBUS_SESSION_BUS_ADDRESS" ]; then
  eval "$(dbus-launch --sh-syntax)"
fi
unset XDG_RUNTIME_DIR
lxqt-session &
EOS
chmod +x ~/.vnc/xstartup

# 5) vnc-start / vnc-stop helpers
mkdir -p ~/bin
cat > ~/bin/vnc-start << "EOS"
#!/usr/bin/env bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
GEOM="${1:-1280x720}"
vncserver -kill :1 >/dev/null 2>&1 || true
vncserver :1 -localhost no -geometry "$GEOM" -depth 24
echo "Connect to <phone/headset IP>:5901"
EOS
chmod +x ~/bin/vnc-start

cat > ~/bin/vnc-stop << "EOS"
#!/usr/bin/env bash
vncserver -kill :1 || true
EOS
chmod +x ~/bin/vnc-stop

echo "Inside Ubuntu, run:  vnc-start  (or: vnc-start 1920x1080)"
'
echo "All set. Next steps below."
EOF
