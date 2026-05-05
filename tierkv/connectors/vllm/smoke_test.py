"""
tierkv vLLM smoke test — run before any benchmark.

Usage:
    python -m tierkv.connectors.vllm.smoke_test \
        [--base http://localhost:8000] \
        [--model MODEL_ID] \
        [--toml /path/to/tierkv.toml] \
        [--bench /path/to/bench_script.py]

Checks:
  1. vLLM /health endpoint responds
  2. Target model is loaded and max_model_len >= 20000
  3. Benchmark context fits within max_model_len (tokenize API)
  4. TCP reachability of kv_cold and ssm_cold vault ports
  5. Quick single-token inference completes without error

Exits 0 if all checks pass, 1 on any failure.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

try:
    import requests
except ImportError:
    print("requests not installed — pip install requests", file=sys.stderr)
    sys.exit(1)

GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"

_results: list[bool] = []


def check(name: str, ok: bool, msg: str = "") -> None:
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    suffix = f": {msg}" if msg else ""
    print(f"  [{tag}] {name}{suffix}")
    _results.append(ok)


def run(base: str, model: str, toml: str | None, bench: str | None) -> int:
    print()

    # ── 1. vLLM health ────────────────────────────────────────────────────────
    print("[1] vLLM health")
    try:
        r = requests.get(f"{base}/health", timeout=5)
        check("vLLM /health", r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as e:
        check("vLLM /health", False, str(e))

    # ── 2. Model loaded ───────────────────────────────────────────────────────
    print("[2] Model")
    max_model_len = 0
    try:
        r = requests.get(f"{base}/v1/models", timeout=5)
        data = r.json()
        models = [m["id"] for m in data.get("data", [])]
        loaded = model in models
        check("Model loaded", loaded, models[0] if models else "none")
        if data.get("data"):
            max_model_len = data["data"][0].get("max_model_len", 0)
        check("max_model_len >= 20000", max_model_len >= 20000, str(max_model_len))
    except Exception as e:
        check("Model check", False, str(e))

    # ── 3. Context fit check (only if bench script provided) ─────────────────
    print("[3] Context fit check")
    if bench:
        try:
            with open(bench) as f:
                content = f.read()
            marker = 'EARNINGS_REPORT = """'
            if marker in content:
                start = content.index(marker) + len(marker)
                lines = content[start:].split("\n")
                report_lines = []
                for line in lines:
                    if line.strip() == '"""':
                        break
                    report_lines.append(line)
                ctx = "\n".join(report_lines)
                # Find longest question
                import re
                questions = re.findall(r'"\s*(.*?)\s*"', content[content.find("QUESTIONS"):], re.DOTALL)
                longest_q = max(questions, key=len) if questions else "test"

                r = requests.post(f"{base}/tokenize", json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": ctx},
                        {"role": "user",   "content": longest_q[:500]},
                    ],
                    "add_special_tokens": True,
                }, timeout=10)
                d = r.json()
                count = d.get("count", 0)
                max_len = d.get("max_model_len", max_model_len or 20000)
                headroom = max_len - count
                check("Context fits (with longest Q)", headroom > 0,
                      f"{count} tokens, {headroom} headroom")
                check("Headroom > 100 tokens", headroom > 100,
                      f"{headroom} spare tokens")
            else:
                check("Context fit (EARNINGS_REPORT found)", False,
                      "bench script does not contain EARNINGS_REPORT marker")
        except Exception as e:
            check("Context fit check", False, str(e))
    else:
        print("  [skipped] no --bench script provided")

    # ── 4. Vault connectivity ─────────────────────────────────────────────────
    print("[4] Vault connectivity")
    vaults: list[tuple[str, str, int]] = []

    if toml:
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            with open(toml, "rb") as f:
                cfg = tomllib.load(f)
            cluster = cfg.get("cluster", {})
            kv  = cluster.get("kv_cold",  {})
            ssm = cluster.get("ssm_cold", {})
            vault_section = cfg.get("vault", {})
            if kv:
                vaults.append(("kv_cold",  kv.get("host",  "127.0.0.1"), int(kv.get("port",  50051))))
            if ssm:
                vaults.append(("ssm_cold", ssm.get("host", "127.0.0.1"), int(ssm.get("port", 50051))))
            # Flat [vault] section fallback
            if not vaults and vault_section:
                vaults.append(("kv_cold",
                                vault_section.get("kv_cold_host", "127.0.0.1"),
                                int(vault_section.get("kv_cold_port", 50051))))
        except Exception as e:
            check("TOML parse", False, str(e))
    else:
        vaults = [("kv_cold", "127.0.0.1", 50051)]
        print("  [note] no --toml provided, checking default 127.0.0.1:50051")

    for name, host, port in vaults:
        try:
            s = socket.create_connection((host, port), timeout=3)
            s.close()
            check(f"TCP {name}", True, f"{host}:{port}")
        except Exception as e:
            check(f"TCP {name}", False, f"{host}:{port} — {e}")

    # ── 5. Quick inference ────────────────────────────────────────────────────
    print("[5] Quick inference (1 token)")
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{base}/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with just: OK"}],
            "max_tokens": 1, "temperature": 0.0,
        }, timeout=60)
        elapsed = time.perf_counter() - t0
        ok = r.status_code == 200 and "choices" in r.json()
        first_tok = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        check("Inference responds", ok, f"{elapsed:.2f}s, first token: {first_tok[:30]!r}")
    except Exception as e:
        check("Inference responds", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    passed = sum(_results)
    total  = len(_results)
    ok_all = all(_results)
    tag    = f"{GREEN}PASS{RESET}" if ok_all else f"{RED}FAIL{RESET}"
    print(f"[{tag}] Smoke test: {passed}/{total} checks passed")
    if not ok_all:
        print("     Fix failing checks before running benchmark.")
    return 0 if ok_all else 1


def main() -> None:
    p = argparse.ArgumentParser(description="TierKV vLLM smoke test")
    p.add_argument("--base",  default="http://localhost:8000", help="vLLM base URL")
    p.add_argument("--model", default="/home/pk/.local/share/exo/models/Qwen--Qwen3.6-35B-A3B")
    p.add_argument("--toml",  default=None, help="Path to tierkv.toml")
    p.add_argument("--bench", default=None, help="Path to benchmark script (for context fit check)")
    args = p.parse_args()
    sys.exit(run(args.base, args.model, args.toml, args.bench))


if __name__ == "__main__":
    main()
