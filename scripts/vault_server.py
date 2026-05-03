"""
ColdVault gRPC server — runs persistently on a cold-tier node (port 50051).

Usage:
    uv run scripts/vault_server.py [PORT]
    # or use the CLI: tierkv vault --port 50051

For auto-start on macOS, install a launchd plist pointing to this script.
"""
import sys
import time
import socket

from tierkv_core import start_cold_vault_server

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
start_cold_vault_server(PORT)
time.sleep(0.5)

try:
    s = socket.create_connection(("127.0.0.1", PORT), timeout=2)
    s.close()
    print(f"[tierkv vault] ColdVaultService listening on :{PORT}", flush=True)
except Exception as e:
    print(f"[tierkv vault] Failed to bind on :{PORT}: {e}", flush=True)
    sys.exit(1)

print("[tierkv vault] Server running — press Ctrl+C to stop", flush=True)
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\n[tierkv vault] Stopped.")
