"""
tierkv — cluster orchestrator entry point.

Run this on any node to:
  1. Start the appropriate gRPC server for this node's role.
  2. Print cluster status from the EXO API.
  3. Confirm the TieredKVCache is reachable end-to-end.

Usage:
    uv run main.py [--role inference|kv_cold|ssm_cold]
    uv run main.py --role kv_cold   # start ColdVault server
"""

import asyncio
import socket
import sys
import httpx

from tierkv.config import load_config

cfg = load_config()

EXO_API        = f"http://{cfg.cluster.kv_cold.host}:52415"
COLD_PORT      = cfg.vault.port
RECOMPUTE_PORT = cfg.cluster.recompute.port


def detect_role() -> str:
    """Auto-detect node role from tierkv.toml or hostname fallback."""
    if cfg.cluster.role != "inference":
        return cfg.cluster.role

    hostname = socket.gethostname().lower()
    if "air" in hostname:
        return "ssm_cold"
    if "pro" in hostname:
        return "kv_cold"
    # Check subnet IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.50.1", 80))
        local_ip = s.getsockname()[0]
        s.close()
        if local_ip == cfg.cluster.ssm_cold.host:
            return "ssm_cold"
        if local_ip == cfg.cluster.kv_cold.host:
            return "kv_cold"
    except Exception:
        pass
    return "inference"


def start_grpc_servers(role: str) -> None:
    """Start the appropriate gRPC server for this node's role."""
    from tierkv_core import start_cold_vault_server, start_recompute_server
    import time

    if role in ("kv_cold", "ssm_cold"):
        start_cold_vault_server(COLD_PORT)
        time.sleep(0.5)
        try:
            s = socket.create_connection(("127.0.0.1", COLD_PORT), timeout=2)
            s.close()
            print(f"[grpc] ColdVault server listening on :{COLD_PORT}")
        except Exception as e:
            print(f"[grpc] ColdVault server failed to bind: {e}")

    elif role == "inference":
        start_recompute_server(RECOMPUTE_PORT)
        time.sleep(0.5)
        try:
            s = socket.create_connection(("127.0.0.1", RECOMPUTE_PORT), timeout=2)
            s.close()
            print(f"[grpc] Recompute server listening on :{RECOMPUTE_PORT}")
        except Exception as e:
            print(f"[grpc] Recompute server failed to bind: {e}")


def check_tierkv_core() -> bool:
    try:
        from tierkv_core import TieredKVCache, TurboQuant  # noqa: F401
        print("[tierkv-core] Rust extension loaded OK")
        return True
    except ImportError as e:
        print(f"[tierkv-core] Not built yet — run: uv sync\n  ({e})")
        return False


async def check_exo_cluster() -> None:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{EXO_API}/state")
            data = r.json()
            nodes = data.get("topology", {}).get("nodes", [])
            count = len(nodes) if isinstance(nodes, list) else len(nodes.keys())
            print(f"[EXO]  API reachable at {EXO_API} — {count} node(s) visible")
    except Exception as e:
        print(f"[EXO]  Not reachable at {EXO_API}\n  ({e})")


def smoke_test_cache() -> None:
    from tierkv import get_cache
    import struct

    dim = 64  # small for smoke test
    cache = get_cache(dim=dim)

    kv = struct.pack(f"{dim}f", *[float(i) for i in range(dim)])
    cache.insert(token_idx=0, layer=0, data=kv)
    result = cache.get(token_idx=0, layer=0)
    stats = cache.stats()

    print(f"[cache] Inserted and retrieved {dim}-dim KV vector")
    print(f"[cache] Stats: {stats}")
    assert stats["hits"] == 1, f"Expected 1 hit, got {stats}"
    print("[cache] Smoke test PASSED")


def main() -> None:
    print("=== tierkv cluster orchestrator ===\n")

    # Determine node role
    role = sys.argv[1] if len(sys.argv) > 1 else detect_role()
    print(f"[role] Running as: {role}\n")

    core_ok = check_tierkv_core()

    if core_ok:
        start_grpc_servers(role)

    asyncio.run(check_exo_cluster())

    if core_ok:
        smoke_test_cache()

    print("\nCluster status summary complete.")


if __name__ == "__main__":
    main()
