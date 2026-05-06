# tierkv

[![Build Wheels](https://github.com/tierkv/tierkv/actions/workflows/wheels.yml/badge.svg)](https://github.com/tierkv/tierkv/actions/workflows/wheels.yml)

**3-tier distributed KV cache for LLM inference.**

When your GPU evicts a KV cache entry, tierkv ships it to another machine over gRPC instead of dropping it. On the next request with the same prompt, the KV is fetched back in a single batch call — skipping the expensive prefill entirely.

Tested on Qwen3.6-35B-A3B across a DGX GB10 + Mac Pro + Mac Air cluster:

**EXO integration** (BF16, 8k–15k token prompts):

| Scenario | TTFT | vs Cold |
|---|---|---|
| Cold start, 8,000-token prompt | 30.83s | baseline |
| Restored from cold tier | 4.11s | **7.3× faster** |
| Cold start, 3,707-token prompt | 23.78s | baseline |
| Restored from cold tier | 4.59s | **5.2× faster** |

**vLLM integration** (Apple FY2025 10-K, GB10 GPU, real-world document Q&A):

| Prompt size | Cold prefill | GPU cache hit | Cold restore | Speedup |
|---|---|---|---|---|
| 30k tokens (measured) | 10.75s | 1.19s | **0.52s** | **20×** |
| 60k tokens (projected) | ~26s | ~1.2s | ~1.0s | ~26× |
| 128k tokens (projected) | ~70s | ~1.5s | ~2.0s | ~35× |

Cold vault restore beats GPU cache hit — blocks land directly into the KV cache skipping attention recomputation entirely. The speedup grows with context length because prefill scales super-linearly while restore is near-linear (network transfer). With `turbo_quant = false` in the vault config, output is bit-for-bit identical across all three paths. With TurboQuant INT8 (default), generation is perceptually equivalent but not bitwise identical due to quantization rounding.

---

## How It Works

tierkv supports two inference backends. The cold-storage layer (vault servers, gRPC, TurboQuant) is identical in both cases.

**EXO backend** (monkey-patch):
```
  DGX GB10 — inference only
  ┌─────────────────────────────────┐
  │  EXO + Qwen3.6-35B-A3B (BF16)  │  ← EXO runs HERE only
  │  KVPrefixCache (GPU hot tier)   │
  │         │ evict (60% RAM)       │
  │         ▼                       │
  │   tierkv hook (monkey-patch)    │
  └────┬──────────────┬─────────────┘
       │ KVCache       │ ArraysCache
       │ (10 layers)   │ (30 layers)
       ▼               ▼
  Mac Pro LAN      Mac Air LAN        ← cold storage only, no EXO
  0.5ms RTT        1ms RTT
  tierkv vault     tierkv vault
  (in-memory)      (in-memory)
```

**vLLM backend** (KVConnectorBase_V1 plugin):
```
  DGX GB10 — inference only
  ┌──────────────────────────────────────────┐
  │  vLLM + Qwen3.6-35B-A3B                 │
  │  Paged KV cache (GPU hot tier, 40 blocks)│
  │         │ block evicted                  │
  │         ▼                                │
  │   TierKVConnector (KVConnectorBase_V1)   │
  │   ├─ request_finished  → store to vault  │
  │   ├─ get_num_new_matched_tokens → plan   │
  │   └─ start_load_kv / wait_for_layer_load │
  └────┬──────────────────┬──────────────────┘
       │ full-attention KV │ SSM / linear-attn
       │ (10 layers)       │ (30 layers)
       ▼                   ▼
  Mac Pro LAN          Mac Air LAN     ← cold storage only, no vLLM
  0.5ms RTT            1ms RTT
  tierkv vault         tierkv vault
  (in-memory)          (in-memory)
```

**Three tiers:**

- **Hot** — GPU KV cache on the inference node (EXO's `KVPrefixCache` or vLLM's paged KV cache). Fast, limited by GPU/HBM capacity.
- **Cold KV** — Full-attention layer tensors shipped to a LAN node via gRPC, compressed with TurboQuant INT8 (~3.9× ratio, ≥52 dB SNR).
- **Cold SSM** — Linear-attention / SSM layer states shipped to a second node. Qwen3.6-35B-A3B is a hybrid MoE — 10/40 layers use full attention, 30/40 use linear attention.

On a cache miss, two parallel `BatchPromote` RPCs fetch all blocks in 2 network round-trips, with parallel decode across a thread pool (decode releases the GIL, so N CPU cores work simultaneously).

For vLLM, the `layer_type_map` in `tierkv.toml` routes each layer group to the correct vault. For EXO, layer types are **auto-detected** via `isinstance` checks — no manual configuration needed.

---

## Hardware Requirements

You need at least 2 machines: one running inference, one as cold storage. Three machines lets you split KV and SSM tiers across separate nodes for better throughput.

| Role | What runs on it | Example |
|---|---|---|
| `inference` | EXO **or vLLM** + your model + tierkv | DGX GB10 |
| `kv_cold` | `tierkv vault` only | Mac Pro (32 GB) |
| `ssm_cold` | `tierkv vault` only | Mac Air (16 GB) |

**EXO only runs on the inference node.** The cold-tier machines (Mac Pro, Mac Air) only run the tierkv vault server — a lightweight gRPC process that holds KV data in RAM.

---

## Installation

**EXO compatibility:** tierkv patches EXO's `cache.py` and `builder.py` in-place. Tested with EXO as of May 2026. EXO moves fast — if `tierkv install` errors, check that the patch targets in `cache.py` and `builder.py` still match. EXO version auto-detection is on the [roadmap](#roadmap).

Download the prebuilt wheel for your platform from the [latest release](https://github.com/tierkv/tierkv/releases):

```bash
pip install tierkv
```

**Or build from source** (requires Rust toolchain):

```bash
git clone https://github.com/tierkv/tierkv.git
cd tierkv
cd tierkv-core && maturin develop --release && cd ..
pip install -e .
```

Prebuilt wheels are available for:
- Linux aarch64 (DGX Spark, Jetson, ARM servers)
- Linux x86_64
- macOS arm64 (Apple Silicon — Mac Pro, Mac Air)

---

## Setup — Step by Step

tierkv runs on **all three machines**, but each machine has a different role and a different config. Install `tierkv` on **every node** first — including the cold vault nodes (Mac Pro, Mac Air). The vault server is part of the `tierkv` Python package, not just `tierkv-core`.

```bash
# Run this on DGX, Mac Pro, and Mac Air
pip install tierkv
```

Then configure each machine.

### Step 1 — Configure each machine

Each machine gets its own `tierkv.toml`. The recommended location is `~/.config/tierkv/tierkv.toml` (created automatically by `tierkv install`) or anywhere you prefer — just pass `--config /path/to/tierkv.toml` to every command. You can also set `TIERKV_CONFIG=/path/to/tierkv.toml` in your shell profile to avoid passing it every time.

```bash
mkdir -p ~/.config/tierkv
cp tierkv.toml.example ~/.config/tierkv/tierkv.toml
# edit the file with your IPs and paths
```

**On the inference node (DGX / Linux GPU server)** — set `role = "inference"` and point to the cold nodes:

```toml
[cluster]
role = "inference"

[cluster.kv_cold]
host = "192.168.50.11"      # cold node 1 IP (LAN)
port = 50051

[cluster.ssm_cold]
host = "192.168.50.12"      # cold node 2 IP (LAN)
port = 50051

[cluster.recompute]
host = "127.0.0.1"
port = 50052

[inference]
exo_path = "/home/user/exo/src/exo"   # path to your EXO installation (EXO backend only)
log_file  = "/tmp/tierkv.log"
memory_threshold = 0.60
kv_dim   = 256
```

**On the KV cold node (cold machine 1)** — set `role = "kv_cold"` and set `max_bytes` to ~75% of available RAM:

```toml
[cluster]
role = "kv_cold"

[vault]
port = 50051
max_bytes = 24_000_000_000   # 24 GB — set to ~75% of your machine's RAM
```

**On the SSM cold node (cold machine 2)** — set `role = "ssm_cold"`:

```toml
[cluster]
role = "ssm_cold"

[vault]
port = 50051
max_bytes = 12_000_000_000   # 12 GB — set to ~75% of your machine's RAM
```

> **`max_bytes`** controls LRU eviction — the vault drops oldest blocks when it reaches this limit. Set it to ~75% of available RAM to leave headroom for the OS. If omitted or set to 0, the vault grows without bound until it OOMs.

`tierkv.toml` is gitignored — it contains your private IPs. Only `tierkv.toml.example` is committed.

### Step 2 — Start vault servers on cold nodes

On each **cold node** (not on the inference node):

```bash
tierkv --config ~/.config/tierkv/tierkv.toml vault
```

> **Note:** `--config` is a global flag and must come **before** the subcommand. `tierkv vault --config path` will fail with "unrecognized arguments".

Confirm it's listening before moving on:

```bash
nc -zv localhost 50051 && echo "vault OK"
# Connection to localhost port 50051 [tcp/*] succeeded!
# vault OK
```

> **Firewall:** Port 50051 must be reachable from the inference node. On macOS you'll get a firewall dialog on first launch — click Allow. On Linux: `sudo ufw allow 50051/tcp`

#### Keep it running — macOS launchd

Create `~/Library/LaunchAgents/com.tierkv.vault.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>             <string>com.tierkv.vault</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/tierkv</string>
        <string>--config</string>
        <string>/Users/YOUR_USERNAME/.config/tierkv/tierkv.toml</string>
        <string>vault</string>
    </array>
    <key>RunAtLoad</key>         <true/>
    <key>KeepAlive</key>         <true/>
    <key>StandardOutPath</key>   <string>/tmp/tierkv-vault.log</string>
    <key>StandardErrorPath</key> <string>/tmp/tierkv-vault.log</string>
</dict>
</plist>
```

```bash
# Replace YOUR_USERNAME and adjust the tierkv path (which tierkv)
launchctl load ~/Library/LaunchAgents/com.tierkv.vault.plist
launchctl start com.tierkv.vault
```

#### Keep it running — Linux systemd

```bash
sudo tee /etc/systemd/system/tierkv-vault.service << 'EOF'
[Unit]
Description=tierkv cold vault
After=network.target

[Service]
User=YOUR_USERNAME
ExecStart=/usr/local/bin/tierkv --config /home/YOUR_USERNAME/.config/tierkv/tierkv.toml vault
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now tierkv-vault
sudo systemctl status tierkv-vault
```

#### Flush cached blocks

Useful when switching models or debugging a corrupt restore:

```bash
tierkv --config ~/.config/tierkv/tierkv.toml flush              # both vaults
tierkv --config ~/.config/tierkv/tierkv.toml flush --target kv_cold   # cold node 1 only
tierkv --config ~/.config/tierkv/tierkv.toml flush --target ssm_cold  # cold node 2 only
```

### Step 3 — Install the EXO hook on the inference node

> **Skip this step if you are using the vLLM backend** — see [vLLM Integration](#vllm-integration) instead.

On the **inference node only**:

```bash
tierkv --config ~/.config/tierkv/tierkv.toml install --exo-path /path/to/exo/src/exo
```

This command:
1. Copies the tierkv hook into EXO's engine directory
2. Patches EXO's `cache.py` to set the memory eviction threshold
3. Patches EXO's `builder.py` to auto-load the hook on startup
4. Copies `tierkv.toml` into EXO's working directory

Restart EXO. The hook reads `tierkv.toml` from the working directory and connects to the cold nodes automatically.

### Step 4 — Verify

From the **inference node**, check all nodes are reachable:

> **Note:** Run `tierkv status` from the **inference node** (the one with `role = "inference"` in its config). The cold nodes have minimal configs that don't know the cluster addresses, so `tierkv status` run from a cold node will only check localhost.

```bash
tierkv --config ~/.config/tierkv/tierkv.toml status
```

```
[tierkv status] Cluster role: inference

  kv_cold      192.168.50.11:50051   ✓  0.4ms
  ssm_cold     192.168.10.174:50051  ✓  5.9ms
  recompute    127.0.0.1:50052       ✓  0.1ms

[tierkv status] All nodes reachable.
```

### Step 5 — Benchmark

```bash
tierkv --config ~/.config/tierkv/tierkv.toml bench --exo-api http://192.168.50.11:52415
```

Expected output:

```
[tierkv bench] EXO API: http://192.168.50.11:52415

  Request 1 — cold start (TARGET): 8.41s   response: 'The key advantage of mixture-of-experts…'
  Waiting 12s for async eviction to complete...
  Request 2 — evict step (different prompt): 1.12s   response: 'Sure, here is a short poem…'
  Waiting 12s for eviction gRPC to settle...
  Request 3 — restore (TARGET from cold): 1.62s   response: 'The key advantage of mixture-of-experts…'

  Speedup (cold → restore): 5.2×
  Time saved per request:   6.79s
```

A speedup below **1.5×** means the cold tier isn't being hit — check `tierkv --config ~/.config/tierkv/tierkv.toml status` to confirm the vault servers are running and reachable.

---

## vLLM Integration

tierkv ships a native **vLLM KV Connector** that plugs into vLLM's `KVConnectorBase_V1` API. It uses the same cold vault infrastructure as the EXO hook — the same `tierkv_core` Rust backend, the same `tierkv.toml` config, and the same gRPC vault servers on Mac Pro / Mac Air.

**Tested with:** vLLM 0.20.1, torch 2.11.0+cu130, CUDA 13.0 on DGX GB10 (aarch64). **Requires vLLM ≥ 0.20.1** — earlier versions have an incompatible `KVConnectorBase_V1` API. A runtime warning is emitted if an older version is detected.

### Install vLLM

```bash
# Linux aarch64 (DGX GB10/Spark) — requires Python dev headers for fastsafetensors
sudo apt-get install -y python3.12-dev
pip install "vllm>=0.20.1" tierkv
```

> **Important:** If vLLM runs in its own virtualenv (common with `pip install vllm`), install `tierkv` **inside that same venv** — not just in the system Python. The connector is imported by the vLLM process at runtime.
>
> ```bash
> # Example: vLLM installed in /home/user/vllm-venv
> /home/user/vllm-venv/bin/pip install tierkv
> ```

```bash
# Linux x86_64 / macOS arm64
pip install vllm tierkv
```

### Start vault servers

Same as for EXO — start `tierkv vault` on Mac Pro and Mac Air before launching vLLM.

### Launch vLLM with TierKV

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --kv-transfer-config '{
    "kv_connector": "TierKVConnector",
    "kv_connector_module_path": "tierkv.connectors.vllm.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"config_path": "/path/to/tierkv.toml"}
  }' \
  --enable-prefix-caching \
  --block-size 16 \
  --no-disable-hybrid-kv-cache-manager \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32
```

> **`--no-disable-hybrid-kv-cache-manager` is required** for hybrid models like Qwen3.5 MoE that mix full-attention and SSM/linear-attention layers. vLLM auto-disables HMA when a KV connector is set; this flag re-enables it. TierKVConnector implements `SupportsHMA` so the override is safe.

Or pass config inline without a TOML file:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --kv-transfer-config '{
    "kv_connector": "TierKVConnector",
    "kv_connector_module_path": "tierkv.connectors.vllm.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "kv_cold_host": "192.168.50.11",
      "kv_cold_port": 50051,
      "ssm_cold_host": "192.168.10.174",
      "ssm_cold_port": 50051,
      "kv_dim": 256,
      "turbo_quant": true,
      "block_size": 16,
      "registry_db_path": "/home/user/.local/share/tierkv/registry.db"
    }
  }' \
  --enable-prefix-caching \
  --block-size 16
```

> **Note:** vLLM 0.20+ uses `--kv-transfer-config` (not `--kv-connector` / `--kv-connector-extra-config`). The connector must be specified as `kv_connector` (class name) + `kv_connector_module_path` (module path) — passing the full dotted path as `kv_connector` will fail.

### vLLM Performance

Measured on **DGX Spark (GB10, aarch64)** with **Qwen3.6-35B-A3B** (35B MoE, 40 layers: 10 full-attention + 30 linear-attention), **Apple FY2025 10-K** (30,561-token real document), cold vaults on Mac Pro + Mac Air (5GbE LAN, 1ms RTT):

| Scenario | TTFT | vs Full Prefill | Notes |
|---|---|---|---|
| Full prefill (30k tokens) | 10.75s | 1× baseline | cold GPU cache, no vault |
| **GPU cache hit** | **1.19s** | **9× faster** | same prompt, blocks in GPU |
| **Cold vault restore** | **0.52s** | **20× faster** | blocks from LAN vault, skip attention |

**Cold vault restore beats GPU cache hit** — vault blocks are inserted directly into the KV cache without running attention, so TTFT is pure network + insertion latency. GPU cache hit still runs partial attention over the matched prefix. The gap widens at longer contexts because prefill scales super-linearly while restore is near-linear (network transfer + KV insertion).

**Projected scaling** (Qwen3.6-35B-A3B, 5GbE LAN vault):

| Prompt size | Cold prefill | GPU cache hit | Cold restore | Speedup |
|---|---|---|---|---|
| 30k tokens (measured) | 10.75s | 1.19s | **0.52s** | **20×** |
| 60k tokens (projected) | ~26s | ~1.2s | ~1.0s | ~26× |
| 128k tokens (projected) | ~70s | ~1.5s | ~2.0s | ~35× |

**Answer quality:** With `turbo_quant = false`, cold restore produces bit-for-bit identical output to full prefill. With TurboQuant INT8 (default), output is perceptually equivalent — the model produces a correct and coherent answer but the exact tokens may differ from a full prefill due to quantization rounding in the restored KV tensors. The `tensor_hash` field in each `BlockRecord` detects any in-flight corruption.

```
Cold Prefill TTFT:   10.75s  (30k-token Apple 10-K, no cache)
GPU Cache Hit TTFT:   1.19s  (9× faster — same document, blocks in GPU)
Cold Restore TTFT:    0.52s  (20× faster — blocks in vault, skip attention)
Answer quality:       bit-for-bit identical (turbo_quant=false) or perceptually equivalent (INT8 default)
Vault:                Mac Pro + Mac Air, 5GbE LAN, 1ms RTT
```

**HMA block size:** On hybrid models like Qwen3.6-35B-A3B, vLLM's Hybrid Memory Architecture (HMA) overrides `--block-size` to a larger alignment (e.g. 1056 tokens/block). Cold vault restore only triggers for blocks that have a complete prefix hash, so **prompts must exceed the HMA block size** (≥1056 tokens for Qwen3.6-35B-A3B) to be eligible for vault restore. Short test prompts (<1056 tokens) will never produce a block hash and vault restore will appear inactive.

### Pre-launch smoke test

Run this before any benchmark to catch issues early (context overflow, vault unreachable, vLLM misconfiguration):

```bash
python -m tierkv.connectors.vllm.smoke_test \
  --base http://localhost:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --toml /path/to/tierkv.toml \
  --bench /path/to/bench.py
```

Expected output:
```
[1] vLLM health
  [PASS] vLLM /health: HTTP 200
[2] Model
  [PASS] Model loaded: Qwen/Qwen3.6-35B-A3B
  [PASS] max_model_len >= 20000: 20000
[3] Context fit check
  [PASS] Context fits (with longest Q): 19724 tokens, 276 headroom
  [PASS] Headroom > 100 tokens: 276 tokens to spare
[4] Vault connectivity
  [PASS] TCP kv_cold (Mac Pro): 192.168.50.11:50051
  [PASS] TCP ssm_cold (Mac Air): 192.168.10.174:50051
[5] Quick inference
  [PASS] Inference responds: 8.97s
[PASS] Smoke test: 8/8 checks passed
```

### How it works

The vLLM connector uses a **reactive eviction model** — it intercepts vLLM's block eviction path, not a periodic snapshot:

1. **Eviction** (`request_finished`): vLLM signals that GPU blocks are about to be freed. TierKV reads the KV tensors, quantizes them with TurboQuant INT8, and ships them to the cold vault over gRPC. GPU blocks are freed after the store completes.
2. **Restore** (`get_num_new_matched_tokens` + `start_load_kv`): On the next request with the same prompt prefix, TierKV finds the blocks in the cold registry, fires a `BatchPromote` RPC, dequantizes, and writes directly into vLLM's paged KV buffer.
3. **No-op save** (`save_kv_layer`): vLLM's eager save path is a no-op — eviction is the only trigger.

The connector integrates as a standard vLLM [KV Transfer](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) plugin — no vLLM source changes needed.

### Configuration reference

All fields can be set in `tierkv.toml` under `[tierkv]` or passed via `--kv-connector-extra-config`:

| Field | Default | Description |
|---|---|---|
| `kv_cold_host` | `127.0.0.1` | Cold vault host for full-attention KV layers |
| `kv_cold_port` | `50051` | Cold vault port |
| `ssm_cold_host` | `None` | Cold vault host for SSM/linear-attention layers (uses kv_cold if unset) |
| `ssm_cold_port` | `50052` | SSM vault port |
| `block_size` | `16` | Must match vLLM `--block-size` |
| `kv_dim` | `128` | **Must match model head_dim** — see [Troubleshooting](TROUBLESHOOTING.md#kv_dim--the-silent-corruption-trap) |
| `turbo_quant` | `true` | INT8 compression (~3.9× ratio) |
| `max_inflight_stores` | `8` | Concurrent eviction-to-vault gRPC calls |
| `max_inflight_promotes` | `4` | Concurrent restore-from-vault threads |
| `registry_db_path` | `None` | Path to SQLite registry db (e.g. `~/.local/share/tierkv/registry.db`). When set, the block_hash→vault_key mapping persists across vLLM restarts so vault blocks remain accessible. When unset, the registry is in-memory only — vault blocks become unreachable on restart. |

> **kv_dim is critical.** Wrong value causes silent incorrect compression. Find the right value with:
> ```python
> from transformers import AutoConfig
> cfg = AutoConfig.from_pretrained("your/model")
> print(cfg.hidden_size // cfg.num_attention_heads)
> ```

---

## TurboQuant

tierkv includes a **per-group INT8 quantizer** for KV tensor compression before sending over the network.

- Group size: `kv_dim` floats — must match your model's attention head dimension (default 256 for Qwen3.6-35B-A3B; use 128 for Llama-3, Qwen2.5, Mistral; see `tierkv.toml.example` for how to find the right value for other models)
- Each group gets its own absmax scale: `scale = max(|x|) / 127`
- Wire format: `[scale: f32 LE][i8 × 256]` per group
- Compression ratio: **~3.9×** (BF16 input → INT8 output)
- SNR: **≥52 dB** on real KV distributions (per-group isolates outliers)

```python
from tierkv_core import TurboQuant
q = TurboQuant(dim=256)
compressed = q.encode(f32_bytes)   # ~3.9× smaller
recovered  = q.decode(compressed)  # ≥52 dB SNR
```

---

## Architecture Notes

**Why not standard KV offloading?** Most KV offload systems evict to local SSD or CPU RAM on the same machine. tierkv evicts across the network to separate machines, letting idle hardware on your LAN participate in serving long-context requests.

**Why EXO?** [EXO](https://github.com/exo-explore/exo) provides an OpenAI-compatible API layer and handles model loading across Apple Silicon and CUDA devices. tierkv monkey-patches EXO's `KVPrefixCache` eviction and retrieval paths without modifying EXO's core. EXO runs only on the inference node — cold nodes run only the tierkv vault.

**What about multi-node inference?** EXO supports pipeline-parallel inference (splitting layers across machines). tierkv is currently designed for single-node inference with distributed cold storage. The two can coexist but require separate configuration.

---

## Cluster Tested

| Node | Role | Memory | Network |
|---|---|---|---|
| DGX Spark (GB10, aarch64) | Inference — EXO or vLLM + Qwen3.6-35B-A3B | 128 GB RAM + 96 GB HBM | 5GbE LAN |
| Mac Pro (M2 Pro) | KV cold tier — tierkv vault only | 32 GB | 5GbE LAN (0.5ms to DGX) |
| Mac Air (M2) | SSM cold tier — tierkv vault only | 16 GB | 5GbE LAN (1ms to DGX) |

Over one session (EXO): 227 evictions, 6 cold restores, ~26s saved per restore.
Over one session (vLLM, Apple 10-K 30k tokens): cold restore 0.52s vs 10.75s cold prefill (20×); GPU cache hit 1.19s (9×).

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for documented failures and fixes, including:

- gRPC 4 MB message size limit (silent empty responses)
- `kv_dim` mismatch causing silent incorrect compression
- `KVCache.offset` semantics and garbage output after restore
- Stale semaphores after `kill -9`
- EXO Nack loop and election storm after hard reset
- SSH lockout during model load
- Wrong platform wheel installed on Linux
- vLLM `fastsafetensors` build failure on aarch64
- Linux OOM killer firing during model load (normal; desktop processes may be killed)
- vLLM EngineCore zombie holding GPU memory after `pkill` (use `kill_vllm_clean.sh`)

---

## Roadmap

- Persistent cold storage (SQLite / memory-mapped file — survive reboots)
- TurboQuant codebook training on real KV activations (push SNR higher)
- Longer contexts or slower GPUs where prefill is the bottleneck
- Quantization quality validation with `_kv_offsets` fix in place
- EXO version detection for hook compatibility
- Persistent cold storage (SQLite / memory-mapped file — blocks survive vault restarts)
