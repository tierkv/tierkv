"""
ColdVault gRPC server — runs persistently on Mac Air (port 50051).
Start: launchctl load ~/Library/LaunchAgents/com.hailey.coldvault.plist
"""
import sys
import time
import socket

sys.path.insert(0, "/Users/pk/work/EXO_PK/.venv/lib/python3.12/site-packages")

from hailey_core import start_cold_vault_server

PORT = 50051
start_cold_vault_server(PORT)
time.sleep(0.5)

try:
    s = socket.create_connection(("127.0.0.1", PORT), timeout=2)
    s.close()
    print(f"[cold-vault] Listening on :{PORT}", flush=True)
except Exception as e:
    print(f"[cold-vault] Failed to bind: {e}", flush=True)
    sys.exit(1)

print("[cold-vault] Server running — press Ctrl+C to stop", flush=True)
while True:
    time.sleep(60)
