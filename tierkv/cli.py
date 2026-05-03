"""
tierkv CLI — manage and operate a tierkv cluster.

Commands:
    tierkv vault   [--port PORT] [--config PATH]   Start ColdVault gRPC server
    tierkv install [--exo-path PATH] [--config PATH]  Install EXO hook
    tierkv status  [--config PATH]                 Check node reachability
    tierkv bench   [--config PATH]                 Run TTFT benchmark
"""

import argparse
import socket
import sys
import time


def cmd_vault(args):
    """Start the ColdVault gRPC server on this node."""
    from tierkv.config import load_config
    cfg = load_config(args.config)
    port = args.port or cfg.vault.port

    from tierkv_core import start_cold_vault_server
    start_cold_vault_server(port)
    time.sleep(0.5)

    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=2)
        s.close()
        print(f"[tierkv vault] ColdVaultService listening on :{port}")
    except Exception as e:
        print(f"[tierkv vault] Failed to bind on :{port}: {e}", file=sys.stderr)
        sys.exit(1)

    print("[tierkv vault] Server running — Ctrl+C to stop")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[tierkv vault] Stopped.")


def cmd_install(args):
    """Install the tierkv hook into an EXO installation."""
    import shutil
    from pathlib import Path
    from tierkv.config import load_config

    cfg = load_config(args.config)
    exo_path = args.exo_path or cfg.inference.exo_path
    if not exo_path:
        print("[tierkv install] Error: --exo-path required (or set inference.exo_path in tierkv.toml)", file=sys.stderr)
        sys.exit(1)

    exo_path = Path(exo_path)
    if not exo_path.exists():
        print(f"[tierkv install] Error: EXO path not found: {exo_path}", file=sys.stderr)
        sys.exit(1)

    # Copy hook.py to EXO's mlx engine directory
    hook_src = Path(__file__).parent / "exo" / "hook.py"
    hook_dst = exo_path / "worker" / "engines" / "mlx" / "tierkv_hook.py"
    shutil.copy(hook_src, hook_dst)
    print(f"[tierkv install] Copied hook → {hook_dst}")

    # Patch memory threshold in EXO's cache.py
    cache_py = exo_path / "worker" / "engines" / "mlx" / "cache.py"
    if cache_py.exists():
        text = cache_py.read_text()
        threshold = cfg.inference.memory_threshold
        # Insert threshold override at top of file if not already present
        marker = "# tierkv: memory threshold patch"
        if marker not in text:
            patch = f"{marker}\n_MEMORY_THRESHOLD = {threshold}\n"
            text = patch + text
            cache_py.write_text(text)
            print(f"[tierkv install] Patched _MEMORY_THRESHOLD={threshold} in {cache_py}")
        else:
            print(f"[tierkv install] cache.py already patched (skipping)")

    # Patch builder.py to install the hook automatically on EXO startup
    builder_py = exo_path / "worker" / "engines" / "mlx" / "builder.py"
    if builder_py.exists():
        text = builder_py.read_text()
        marker = "# tierkv: hook install"
        if marker not in text:
            hook_patch = (
                "\n"
                f"{marker}\n"
                "try:\n"
                "    from tierkv.exo.hook import install_kv_tiering_hook\n"
                "    install_kv_tiering_hook()\n"
                "except Exception as _e:\n"
                "    import logging; logging.getLogger(__name__).warning(f'tierkv hook failed to load: {_e}')\n"
            )
            # Append at end of file so it runs after KVPrefixCache is initialized
            builder_py.write_text(text + hook_patch)
            print(f"[tierkv install] Patched builder.py → hook installs on EXO startup")
        else:
            print(f"[tierkv install] builder.py already patched (skipping)")
    else:
        print(f"[tierkv install] Warning: builder.py not found at {builder_py}")
        print( "                 Add manually: from tierkv.exo.hook import install_kv_tiering_hook")

    print("[tierkv install] Done. Restart EXO to activate.")


def cmd_status(args):
    """Check reachability of all configured cluster nodes."""
    from tierkv.config import load_config
    cfg = load_config(args.config)

    nodes = [
        ("kv_cold",   cfg.cluster.kv_cold.host,   cfg.cluster.kv_cold.port),
        ("ssm_cold",  cfg.cluster.ssm_cold.host,   cfg.cluster.ssm_cold.port),
        ("recompute", cfg.cluster.recompute.host,  cfg.cluster.recompute.port),
    ]

    print(f"[tierkv status] Cluster role: {cfg.cluster.role}")
    print()

    all_ok = True
    for name, host, port in nodes:
        try:
            t0 = time.monotonic()
            s = socket.create_connection((host, port), timeout=3)
            s.close()
            ms = (time.monotonic() - t0) * 1000
            print(f"  {name:12}  {host}:{port}  ✓  {ms:.1f}ms")
        except Exception as e:
            print(f"  {name:12}  {host}:{port}  ✗  ({e})")
            all_ok = False

    print()
    if all_ok:
        print("[tierkv status] All nodes reachable.")
    else:
        print("[tierkv status] Some nodes unreachable — check tierkv.toml and that vault servers are running.")
        sys.exit(1)


def cmd_bench(args):
    """Run a 3-request TTFT benchmark: cold → evict → restore."""
    from tierkv.config import load_config
    cfg = load_config(args.config)

    try:
        import httpx
    except ImportError:
        print("[tierkv bench] Error: httpx required — pip install httpx", file=sys.stderr)
        sys.exit(1)

    # Try to find EXO API — default to standard port on kv_cold host or localhost
    exo_api = args.exo_api if hasattr(args, 'exo_api') and args.exo_api else "http://127.0.0.1:52415"

    LONG_PROMPT = "Explain the significance of distributed KV caching in large language model inference. " * 200
    SHORT_PROMPT = "Briefly define photosynthesis in one sentence."

    def ttft(prompt: str, label: str) -> float:
        import json, urllib.request
        body = json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80,
            "enable_thinking": False,
            "stream": False,
        }).encode()
        req = urllib.request.Request(f"{exo_api}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                elapsed = time.monotonic() - t0
                data = json.loads(resp.read())
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"  {label}: {elapsed:.2f}s  response: {repr(content[:60])}")
                return elapsed
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  {label}: {elapsed:.2f}s  ERROR: {e}")
            return elapsed

    print(f"[tierkv bench] EXO API: {exo_api}")
    print()

    t1 = ttft(LONG_PROMPT, "Request 1 — cold start (TARGET)")
    print("  Waiting 12s for async eviction to complete...")
    time.sleep(12)

    t2 = ttft(SHORT_PROMPT, "Request 2 — evict step (different prompt)")
    print("  Waiting 12s for eviction gRPC to settle...")
    time.sleep(12)

    t3 = ttft(LONG_PROMPT, "Request 3 — restore (TARGET from cold)")

    speedup = t1 / t3 if t3 > 0 else 0
    print()
    print(f"  Speedup (cold → restore): {speedup:.1f}×")
    print(f"  Time saved per request:   {t1 - t3:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        prog="tierkv",
        description="tierkv — 3-tier distributed KV cache for LLM inference",
    )
    parser.add_argument("--config", metavar="PATH", help="Path to tierkv.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    # vault
    p_vault = sub.add_parser("vault", help="Start ColdVault gRPC server on this node")
    p_vault.add_argument("--port", type=int, help="Port to listen on (default: from config or 50051)")

    # install
    p_install = sub.add_parser("install", help="Install tierkv hook into an EXO installation")
    p_install.add_argument("--exo-path", metavar="PATH", help="Path to EXO src/exo directory")

    # status
    sub.add_parser("status", help="Check reachability of all cluster nodes")

    # bench
    p_bench = sub.add_parser("bench", help="Run cold→restore TTFT benchmark")
    p_bench.add_argument("--exo-api", metavar="URL", help="EXO API base URL (default: http://127.0.0.1:52415)")

    args = parser.parse_args()

    dispatch = {
        "vault":   cmd_vault,
        "install": cmd_install,
        "status":  cmd_status,
        "bench":   cmd_bench,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
