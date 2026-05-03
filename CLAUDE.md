# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Project Hailey-Turbo** — a 3-tier KV cache system for LLM inference across a heterogeneous cluster:

| Node | Role | IP |
|------|------|----|
| DGX Spark | Full model inference (Qwen3.6-35B-A3B BF16) + RecomputeService | 192.168.10.227 (WiFi) |
| Mac Pro | KV cold tier (ColdVaultService, TurboQuant compressed) | 192.168.50.11 (LAN) |
| Mac Air | Residual cold tier (this machine) | 192.168.10.174 (WiFi) |

EXO runs on DGX in single-node mode (`hailey-dgx-solo` namespace). The hailey-core Rust extension is monkey-patched into EXO's `KVPrefixCache._evict_if_needed()` to redirect evictions to Mac Pro instead of dropping them.

## Build

```bash
# Build the Rust extension (hailey-core) and install into the venv
uv sync

# Rebuild after Rust changes (faster than uv sync)
cd hailey-core && maturin develop --release
```

The Rust build generates protobuf bindings from `hailey-core/proto/hailey.proto` via `tonic-build` in `build.rs`. Any change to the proto requires rebuilding.

## Run

```bash
# Smoke test the full stack (cache insert/get + EXO cluster check)
uv run main.py

# Start ColdVaultService (Mac Air, port 50051)
uv run scripts/cold_vault_server.py

# Check hailey_core is importable
python3 -c "import hailey_core; print(hailey_core.__file__)"
```

## Architecture

### Rust extension (`hailey-core/src/`)

- **`lib.rs`** — PyO3 module entry point. Exposes `TieredKVCache`, `TurboQuant`, `start_cold_vault_server`, `start_recompute_server`.
- **`tiered_kv.rs`** — Core cache logic. Insert path: `encode → HotTier DashMap → if full, async gRPC evict to cold`. Get path: hot hit → cold `Promote` RPC → recompute `Recompute` RPC → empty (full miss).
- **`turbo_quant.rs`** — Product vector quantizer. Splits f32 vectors into D_SUB=8 sub-vectors, maps each to nearest 256-entry codebook centroid (4-bit code). Two codes packed per byte → ~9.1× compression. **Codebook is random-init** — not trained; reconstruction quality is poor until trained on real KV data.
- **`residual_sync.rs`** — tonic gRPC client/server stubs. `ColdVaultClient.store/promote`, `RecomputeClient.recompute`, plus server impls (`ColdVaultServer` stores in-memory HashMap, `RecomputeServer` is a stub returning empty tensors).

### Python wrapper (`exo_pk/`)

- **`cache.py`** — Thin wrapper. `get_cache(dim, hot_capacity, cold_node, recompute_node)` returns a `TieredKVCache`. Default `cold_node` in the file points to the old IP — always pass it explicitly.

### EXO integration (lives on DGX, not in this repo)

- **`/home/pk/exo/src/exo/worker/engines/mlx/hailey_kv_hook.py`** — Monkey-patches `KVPrefixCache._evict_if_needed`. Installed from `builder.py` line 85. Logs to `/tmp/hailey-dgx.log` on DGX.
- **`/home/pk/exo/src/exo/worker/engines/mlx/cache.py`** — `_MEMORY_THRESHOLD` hardcoded to 0.60 (patched from default 0.80) to trigger evictions earlier.

### gRPC protocol (`hailey-core/proto/hailey.proto`)

Two services:
- `ColdVaultService` — `Store(KVTensor)` and `Promote(token_idx, layer)`. Runs on Mac Pro (port 50051) and Mac Air (port 50051).
- `RecomputeService` — `Recompute(token_idx, layer)`. Stub on DGX (port 50052).

`KVTensor` carries `token_idx`, `layer`, `data` (raw bytes), `rows`, `cols`, `dtype`.

## Deploying to DGX

```bash
# SCP updated hook to DGX (atomic via temp file)
scp -o PubkeyAuthentication=yes -o PasswordAuthentication=no -o IdentityAgent=none \
    -i ~/.ssh/id_ed25519 \
    /tmp/hailey_kv_hook.py pk@192.168.10.227:/tmp/hailey_kv_hook.py.tmp
ssh -o PubkeyAuthentication=yes -o PasswordAuthentication=no -o IdentityAgent=none \
    -i ~/.ssh/id_ed25519 pk@192.168.10.227 \
    'mv /tmp/hailey_kv_hook.py.tmp /home/pk/exo/src/exo/worker/engines/mlx/hailey_kv_hook.py && \
     find /home/pk/exo/src/exo/worker/engines/mlx -name "*.pyc" -delete'

# SSH to DGX
ssh -o PubkeyAuthentication=yes -o PasswordAuthentication=no -o IdentityAgent=none \
    -i ~/.ssh/id_ed25519 pk@192.168.10.227

# SSH to Mac Pro (via DGX as jump host — direct LAN unreliable from Mac Air)
ssh -o PubkeyAuthentication=yes -o PasswordAuthentication=no -o IdentityAgent=none \
    -i ~/.ssh/id_ed25519 -J pk@192.168.10.227 prasannak@192.168.50.11
```

## Known Issues / Gotchas

- **SSH locks out for 2-3 min after every EXO restart** — DGX CPU saturates loading 70GB model via mmap. Ping still works. Just wait; do not assume DGX is down.
- **Never POST `/instance` twice** — old runners revive and compete for GPU memory (each tries to allocate 65GB), causing OOM and another lockout cycle.
- **Stale GPU worker after RunnerFailed** — find PID with `nvidia-smi pmon -s m -c 1`, kill with `kill -9`, then restart EXO.
- **`enable_thinking: false` required** — Qwen3 defaults to thinking mode; without this flag all tokens go to hidden reasoning and `content` is empty.
- **TurboQuant codebook is random** — compression works (9.1×) but reconstruction quality is poor. Train codebook on real Qwen3.6 KV activations before using for accuracy-sensitive workloads.
- **`dim` must be divisible by 8** — TurboQuant D_SUB=8. For Qwen3.6-35B-A3B use `dim=256` (head_dim for full-attention layers).
- **Only 10/40 layers use KV cache** — Qwen3.6-35B-A3B is hybrid MoE; 30 layers use linear attention (ArraysCache). `_serialize_kv_entry` skips non-KVCache types, so `Evicted N layers` will always show N≤10.

## EXO Patches on DGX (must survive upgrades)

| File | Change |
|------|--------|
| `worker/engines/mlx/cache.py` line 46 | `_MEMORY_THRESHOLD = 0.60` (was dynamic from env) |
| `worker/engines/mlx/builder.py` line 83 | installs hailey KV tiering hook after `KVPrefixCache` init |
| `master/placement_utils.py` line ~35 | `if True:` bypasses memory filter in `filter_cycles_by_memory` |
| `master/placement_utils.py` line ~115 | `if False:` bypasses OOM raise in `get_shard_assignments_for_pipeline_parallel` |
| `api/main.py` line ~446 | `if False:` disables memory check for manual placement |
| `worker/main.py` | `_forward_info` catches `BrokenResourceError` |
| `master/main.py` | inactivity timeout 300s |
| `download_utils.py` | `resolve_allow_patterns` caching |
