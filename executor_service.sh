mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/executor_agent.service <<'EOF'
[Unit]
Description=Executor Agent (desktop session)
After=graphical-session.target

[Service]
Type=simple
# IMPORTANT: give the agent your session env so it can open windows
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=/run/user/%U
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/%U/bus
# If you're on Wayland, uncomment the next two lines and comment DISPLAY above:
# Environment=WAYLAND_DISPLAY=wayland-0
# Environment=XDG_SESSION_TYPE=wayland
WorkingDirectory=/home/dchan/executor_agent
ExecStart=/home/dchan/executor_agent/.venv/bin/python -m executor_agent --port 8900
Restart=on-failure

[Install]
WantedBy=default.target
EOF

# Make sure the variables resolve for your user ID:
id -u    # (note the number, e.g. 1000)

# Start on login and keep running even after logout:
systemctl --user daemon-reload
systemctl --user enable --now executor_agent
loginctl enable-linger "$USER"
