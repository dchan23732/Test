mkdir -p ~/.config/systemd/user
# Edit ExecStart to your actual venv/path:
cat > ~/.config/systemd/user/executor_agent.service <<'EOF'
[Unit]
Description=Executor Agent (desktop session)
After=graphical-session.target

[Service]
Type=simple
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=/run/user/%U
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/%U/bus
WorkingDirectory=/home/dchan/executor_agent
ExecStart=/home/dchan/executor_agent/.venv/bin/python -m executor_agent --port 8900
Restart=on-failure

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now executor_agent
loginctl enable-linger "$USER"
