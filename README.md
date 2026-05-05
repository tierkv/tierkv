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

**vLLM integration** (19k-token prompt, GB10 H100 GPU):

| Scenario | TTFT | vs Full Prefill |
|---|---|---|
| Full prefill, 19k-token prompt | 31–40s | baseline |
| **GPU cache hit** | **~0.4s** | **80–200× faster** |
| Cold restore, 15k tokens (LAN vault) | ~31s | ~1× (breakeven) |
| Cold restore, 19k tokens (WiFi vault) | 52–80s | 0.6–1.3× |

GPU cache hit is the primary vLLM value driver — ~26–40s saved per hit on long-context prompts. Answer quality is bit-for-bit identical to full prefill across all three paths.

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
  Mac Pro LAN      Mac Air WiFi       ← cold storage only, no EXO
  0.5ms RTT        6ms RTT
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
  Mac Pro LAN          Mac Air WiFi    ← cold storage only, no vLLM
  0.5ms RTT            6ms RTT
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

tierkv runs on **all three machines**, but each machine has a different role and a different config. Install tierkv on every node first, then configure each one.

### Step 1 — Configure each machine

Each machine gets its own `tierkv.toml` with its role and the addresses of the other nodes. Copy the example and edit it:

```bash
cp tierkv.toml.example tierkv.toml
```

**On the inference node (DGX Spark)** — set `role = "inference"` and point to the cold nodes:

```toml
[cluster]
role = "inference"

[cluster.kv_cold]
host = "192.168.50.11"      # Mac Pro LAN IP
port = 50051

[cluster.ssm_cold]
host = "192.168.10.174"     # Mac Air WiFi IP
port = 50051

[cluster.recompute]
host = "127.0.0.1"
port = 50052

[inference]
exo_path = "/home/user/exo/src/exo"   # path to your EXO installation
log_file  = "/tmp/tierkv.log"
memory_threshold = 0.60
kv_dim   = 256
```

**On the KV cold node (Mac Pro)** — set `role = "kv_cold"`, addresses don't matter here:

```toml
[cluster]
role = "kv_cold"

[vault]
port = 50051
```

**On the SSM cold node (Mac Air)** — set `role = "ssm_cold"`:

```toml
[cluster]
role = "ssm_cold"

[vault]
port = 50051
```

`tierkv.toml` is gitignored — it contains your private IPs. Only `tierkv.toml.example` is committed.

### Step 2 — Start vault servers on cold nodes

> **Warning — unbounded RAM growth:** The vault holds all received KV data in RAM and currently has no eviction policy. On a Mac Air (16 GB) running a long session, vault RAM will grow until the process is killed. Monitor with `tierkv status` and restart vault servers between sessions if needed. LRU eviction is on the [roadmap](#roadmap).

On **Mac Pro** and **Mac Air** (not on DGX):

```bash
tierkv vault
```

This starts the ColdVault gRPC server that the inference node will send KV data to. Keep it running as a background service (macOS launchd / Linux systemd).

### Step 3 — Install the EXO hook on the inference node

On **DGX only**:

```bash
tierkv install --exo-path /path/to/exo/src/exo
```

That's it. This command:
1. Copies the tierkv hook into EXO's engine directory
2. Patches EXO's `cache.py` to set the memory eviction threshold
3. Patches EXO's `builder.py` to auto-load the hook on startup

Restart EXO. The hook reads `tierkv.toml` from the working directory and connects to the cold nodes automatically.

### Step 4 — Verify

From the **inference node**, check all nodes are reachable:

```bash
tierkv status
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
tierkv bench --exo-api http://192.168.50.11:52415
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

A speedup below **1.5×** means the cold tier isn't being hit — check `tierkv status` to confirm the vault servers are running and reachable.

---

## vLLM Integration

tierkv ships a native **vLLM KV Connector** that plugs into vLLM's `KVConnectorBase_V1` API. It uses the same cold vault infrastructure as the EXO hook — the same `tierkv_core` Rust backend, the same `tierkv.toml` config, and the same gRPC vault servers on Mac Pro / Mac Air.

**Tested with:** vLLM 0.20.1, torch 2.11.0+cu130, CUDA 13.0 on DGX GB10 (aarch64).

### Install vLLM

```bash
# Linux aarch64 (DGX GB10/Spark) — requires Python dev headers for fastsafetensors
sudo apt-get install -y python3.12-dev
pip install vllm tierkv
```

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
      "block_size": 16
    }
  }' \
  --enable-prefix-caching \
  --block-size 16
```

> **Note:** vLLM 0.20+ uses `--kv-transfer-config` (not `--kv-connector` / `--kv-connector-extra-config`). The connector must be specified as `kv_connector` (class name) + `kv_connector_module_path` (module path) — passing the full dotted path as `kv_connector` will fail.

### vLLM Performance

Measured on **DGX GB10 (GB200 GPU, aarch64)** with **Qwen3.6-35B-A3B** (35B MoE, 40 layers: 10 full-attention + 30 linear-attention), 19,354-token context (earnings report), cold vaults on Mac Pro (10GbE LAN) and Mac Air (WiFi):

| Scenario | TTFT | vs Full Prefill | Notes |
|---|---|---|---|
| Full prefill (19k tokens) | 31–40s | 1× baseline | cold GPU cache, no vault |
| **GPU cache hit** | **~0.4s** | **80–200× faster** | same prompt, blocks in GPU |
| Cold restore (15k tokens) | ~31s | ~1× (breakeven) | LAN + WiFi vault round-trip |
| Cold restore (19k tokens) | 52–80s | 0.6–1.3× | WiFi bottleneck at 27 blocks |

**The primary value driver is GPU cache hit** — TierKV keeps evicted blocks retrievable, so prefix caching stays effective across sessions. When the same prompt prefix is reused after a GPU eviction, TTFT drops from 31–40s to 0.4s.

Cold restore latency at 15k tokens on this hardware matches full prefill (breakeven). At 19k tokens (27 blocks) over WiFi, restore is roughly equal to or slightly worse than prefill; upgrading the Mac Air to 10GbE LAN or using a model with pure quadratic attention (where prefill scales O(n²)) shifts cold restore to a clear win.

**Answer quality:** cold restore produces bit-for-bit identical output to full prefill. TurboQuant INT8 is lossy but per-group quantization preserves KV distributions well enough that the model's token distribution is indistinguishable. The `tensor_hash` field in each `BlockRecord` detects any in-flight corruption.

```
GPU Cache Hit TTFT:  0.40s   (200× vs 31–40s full prefill at 19k tokens)
Cold Restore TTFT:   ~31s    (breakeven at 15k tokens, 60s vault-wait)
Answer quality:      100% identical output across all three paths
Tensor hash check:   every block verified on restore — corrupt → re-prefill
```

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

> **kv_dim is critical.** Wrong value causes silent incorrect compression. Find the right value with:
> ```python
> from transformers import AutoConfig
> cfg = AutoConfig.from_pretrained("your/model")
> print(cfg.hidden_size // cfg.num_attention_heads)
> ```

### SDK hooks

The connector exposes hooks for building proprietary audit or prioritization logic without forking tierkv:

```python
# Attach a callback fired after each successful block store:
connector.register_store_callback(
    lambda context_id, block_pos, vault_key: audit_log(context_id, vault_key)
)

# Tag a request so its blocks are tracked together:
request.extra_metadata = {"tierkv_context_id": "session-abc"}

# Query how many blocks are in cold storage for a session:
count = connector.get_context_stored_count("session-abc")

# Prioritize a context (0=normal, 1=high, 2=critical — higher = evicted last):
connector.set_context_priority("session-abc", priority=2)
```

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
| DGX GB10 (GB200, aarch64) | Inference — EXO or vLLM + Qwen3.6-35B-A3B | 128 GB RAM + 96 GB HBM | WiFi + 10GbE LAN |
| Mac Pro (M2 Pro) | KV cold tier — tierkv vault only | 32 GB | 10GbE LAN (0.5ms to DGX) |
| Mac Air (M2) | SSM cold tier — tierkv vault only | 16 GB | WiFi (6ms to DGX) |

Over one session (EXO): 227 evictions, 6 cold restores, ~26s saved per restore.
Over one session (vLLM, 19k-token context): GPU cache hits at 0.4s vs 31s full prefill (80–200×).

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

---

## Roadmap

- Persistent cold storage (SQLite / memory-mapped file — survive reboots)
- TurboQuant codebook training on real KV activations (push SNR higher)
- Mac Air on LAN (WiFi is the current SSM restore bottleneck)
- Quantization quality validation with `_kv_offsets` fix in place
- EXO version detection for hook compatibility
- LRU eviction inside the cold vault (configurable max capacity — currently vaults grow unbounded in RAM)
