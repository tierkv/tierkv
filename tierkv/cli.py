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
    import importlib.metadata
    import os
    import platform
    import re
    import shutil
    import subprocess
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

    # Install tierkv into EXO's venv so the hook can import it
    # EXO venv is expected at {exo_path}/../../.venv (i.e. exo repo root/.venv)
    # exo_path = …/exo/src/exo → parent.parent = …/exo (repo root)
    exo_repo_root = exo_path.parent.parent
    venv_python = None
    for candidate in [
        exo_repo_root / ".venv" / "bin" / "python",
        exo_repo_root / ".venv" / "bin" / "python3",
    ]:
        if candidate.exists():
            venv_python = candidate
            break

    if venv_python:
        tierkv_version = importlib.metadata.version("tierkv-core")
        system = platform.system()
        machine = platform.machine()
        if system == "Darwin":
            wheel_file = f"tierkv_core-{tierkv_version}-cp39-abi3-macosx_11_0_arm64.whl"
        elif machine == "aarch64":
            wheel_file = f"tierkv_core-{tierkv_version}-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
        else:
            wheel_file = f"tierkv_core-{tierkv_version}-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        wheel_url = f"https://github.com/tierkv/tierkv/releases/download/v{tierkv_version}/{wheel_file}"

        # Find uv — check PATH and common install locations
        def _find_uv():
            uv = shutil.which("uv")
            if uv:
                return uv
            for candidate in [
                Path.home() / ".local" / "bin" / "uv",
                Path("/usr/local/bin/uv"),
            ]:
                if candidate.exists():
                    return str(candidate)
            return None

        def _run_pip(python, *install_args):
            result = subprocess.run(
                [str(python), "-m", "pip", "install", "--quiet", "--no-warn-script-location", *install_args],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                uv = _find_uv()
                if uv:
                    result = subprocess.run(
                        [uv, "pip", "install", "--quiet", "--python", str(python), *install_args],
                        capture_output=True, text=True,
                    )
            return result

        r = _run_pip(venv_python, wheel_url)
        if r.returncode == 0:
            print(f"[tierkv install] Installed tierkv-core=={tierkv_version} into EXO venv")
        else:
            print(f"[tierkv install] Warning: could not install tierkv-core into EXO venv:")
            print(f"                 {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'unknown error'}")
            print(f"                 Run manually: {venv_python} -m pip install {wheel_url}")

        # Copy tierkv Python package directly into EXO venv site-packages
        # (tierkv is not on PyPI — direct copy is more reliable than pip install)
        venv_site = subprocess.run(
            [str(venv_python), "-c", "import site; print(site.getsitepackages()[0])"],
            capture_output=True, text=True,
        )
        if venv_site.returncode == 0:
            dst_pkg = Path(venv_site.stdout.strip()) / "tierkv"
            src_pkg = Path(__file__).parent  # the tierkv package dir (contains cli.py etc.)
            if dst_pkg.exists():
                shutil.rmtree(dst_pkg)
            shutil.copytree(src_pkg, dst_pkg)
            print(f"[tierkv install] Installed tierkv into EXO venv")
        else:
            print(f"[tierkv install] Warning: could not determine EXO venv site-packages path")
    else:
        print(f"[tierkv install] Warning: EXO venv not found at {exo_repo_root}/.venv")
        print( "                 Install tierkv manually into EXO's Python environment.")

    # Patch memory threshold in EXO's cache.py
    cache_py = exo_path / "worker" / "engines" / "mlx" / "cache.py"
    if cache_py.exists():
        text = cache_py.read_text()
        threshold = cfg.inference.memory_threshold
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
            builder_py.write_text(text + hook_patch)
            print(f"[tierkv install] Patched builder.py → hook installs on EXO startup")
        else:
            print(f"[tierkv install] builder.py already patched (skipping)")
    else:
        print(f"[tierkv install] Warning: builder.py not found at {builder_py}")
        print( "                 Add manually: from tierkv.exo.hook import install_kv_tiering_hook")

    # Copy tierkv.toml into EXO's working directory so the hook can find it
    # Detect workdir: check systemd service file first, fall back to exo repo root
    exo_workdir = None
    for svc_path in [
        Path("/etc/systemd/system/exo.service"),
        Path.home() / ".config" / "systemd" / "user" / "exo.service",
    ]:
        if svc_path.exists():
            m = re.search(r"WorkingDirectory=(.+)", svc_path.read_text())
            if m:
                exo_workdir = Path(m.group(1).strip())
                break
    if exo_workdir is None:
        exo_workdir = exo_repo_root  # exo_path is .../exo/src/exo → repo root is 2 up

    config_src = Path(args.config) if args.config else None
    if config_src is None:
        env_path = os.environ.get("TIERKV_CONFIG")
        if env_path:
            config_src = Path(env_path)
        elif (Path.cwd() / "tierkv.toml").exists():
            config_src = Path.cwd() / "tierkv.toml"

    dst = exo_workdir / "tierkv.toml"
    if config_src and config_src.resolve() != dst.resolve():
        shutil.copy(config_src, dst)
        print(f"[tierkv install] Copied tierkv.toml → {dst}")
    elif dst.exists():
        print(f"[tierkv install] tierkv.toml already in EXO workdir ({dst})")
    else:
        print(f"[tierkv install] Warning: no tierkv.toml found to copy.")
        print(f"                 Place tierkv.toml at {dst} before starting EXO.")

    print("[tierkv install] Done. Restart EXO to activate.")


def cmd_status(args):
    """Check reachability of all configured cluster nodes."""
    from tierkv.config import load_config
    cfg = load_config(args.config)

    nodes = [
        ("kv_cold",   cfg.cluster.kv_cold.host,   cfg.cluster.kv_cold.port,   True),
        ("ssm_cold",  cfg.cluster.ssm_cold.host,   cfg.cluster.ssm_cold.port,  True),
        ("recompute", cfg.cluster.recompute.host,  cfg.cluster.recompute.port, False),
    ]

    print(f"[tierkv status] Cluster role: {cfg.cluster.role}")
    print()

    all_ok = True
    for name, host, port, required in nodes:
        label = name if required else f"{name} (EXO internal — optional)"
        try:
            t0 = time.monotonic()
            s = socket.create_connection((host, port), timeout=3)
            s.close()
            ms = (time.monotonic() - t0) * 1000
            print(f"  {label:42}  {host}:{port}  ✓  {ms:.1f}ms")
        except Exception as e:
            print(f"  {label:42}  {host}:{port}  ✗  ({e})")
            if required:
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

    # Try to find EXO API — default to standard port on kv_cold host or localhost
    exo_api = args.exo_api if hasattr(args, 'exo_api') and args.exo_api else "http://127.0.0.1:52415"

    if hasattr(args, 'model') and args.model:
        model_id = args.model
    else:
        import json as _json, urllib.request as _urlreq, urllib.error as _urlerr
        try:
            with _urlreq.urlopen(f"{exo_api}/v1/models", timeout=5) as r:
                models = _json.loads(r.read()).get("data", [])
            if not models:
                print("[tierkv bench] Error: no models loaded in EXO — send a /place_instance request first", file=sys.stderr)
                sys.exit(1)
            model_id = models[0]["id"]
            print(f"[tierkv bench] Auto-detected model: {model_id}")
        except _urlerr.URLError as e:
            print(f"[tierkv bench] Error: cannot reach EXO API at {exo_api}: {e}", file=sys.stderr)
            sys.exit(1)

    LONG_PROMPT = "Explain the significance of distributed KV caching in large language model inference. " * 200
    SHORT_PROMPT = "Briefly define photosynthesis in one sentence."

    def ttft(prompt: str, label: str) -> float:
        import json, urllib.request
        body = json.dumps({
            "model": model_id,
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
    p_bench.add_argument("--model", metavar="MODEL_ID", help="Model ID to use (default: Qwen/Qwen3.6-35B-A3B)")

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
