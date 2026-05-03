# tierkv

**3-tier distributed KV cache for LLM inference.**

When your GPU evicts a KV cache entry, tierkv ships it to another machine over gRPC instead of dropping it. On the next request with the same prompt, the KV is fetched back in a single batch call — skipping the expensive prefill entirely.

Tested on Qwen3.6-35B-A3B (70 GB BF16) across a DGX Spark + Mac Pro + Mac Air cluster:

| Scenario | TTFT | vs Cold |
|---|---|---|
| Cold start, 8,000-token prompt | 30.83s | baseline |
| Restored from cold tier | 4.11s | **7.3× faster** |
| Cold start, 3,707-token prompt | 23.78s | baseline |
| Restored from cold tier | 4.59s | **5.2× faster** |

~26 seconds saved per cache hit on an 8K-token prompt.

---

## How It Works

```
  Inference node (DGX)
  ┌─────────────────────────────────┐
  │  EXO + Qwen3.6-35B-A3B (BF16)  │
  │  KVPrefixCache (GPU hot tier)   │
  │         │ evict (60% RAM)       │
  │         ▼                       │
  │   tierkv hook (monkey-patch)    │
  └────┬──────────────┬─────────────┘
       │ KVCache       │ ArraysCache
       │ (10 layers)   │ (30 layers)
       ▼               ▼
  Mac Pro LAN      Mac Air WiFi
  0.5ms RTT        6ms RTT
  ColdVault        ColdVault
  (in-memory)      (in-memory)
```

**Three tiers:**

- **Hot** — EXO's GPU KVPrefixCache. Fast, limited. Evicts at 60% RAM.
- **Cold KV** — Full-attention layers (KVCache/RotatingKVCache) shipped to a LAN node via gRPC, compressed with TurboQuant INT8 (~3.9× ratio, ≥52 dB SNR).
- **Cold SSM** — Linear-attention layers (ArraysCache) shipped to a second node. Qwen3.6-35B-A3B is a hybrid MoE — 10/40 layers use full attention, 30/40 use linear attention.

On a cache miss, two parallel `BatchPromote` RPCs fetch all 40 layers in 2 network round-trips (down from 40 sequential RPCs before batch optimization).

Layer types are **auto-detected** via `isinstance` checks — no manual layer index configuration needed.

---

## Hardware Requirements

You need at least 2 machines: one running inference, one as cold storage. Three machines lets you split KV and SSM tiers across separate nodes for better throughput.

| Role | Requirement | Example |
|---|---|---|
| `inference` | GPU with enough VRAM for your model | DGX Spark |
| `kv_cold` | RAM to hold compressed KV layers | Mac Pro (32 GB) |
| `ssm_cold` | RAM to hold SSM state | Mac Air (16 GB) |

---

## Installation

**Prerequisites:** Rust toolchain, Python 3.9+, EXO installed on the inference node.

```bash
git clone https://github.com/tierkv/tierkv.git
cd tierkv

# Build the Rust extension
cd tierkv-core && maturin develop --release && cd ..

# Install Python package
pip install -e .
```

Or once wheels are available on a release:

```bash
pip install tierkv
```

---

## Configuration

Copy the example config and fill in your cluster IPs:

```bash
cp tierkv.toml.example tierkv.toml
```

```toml
[cluster]
role = "inference"          # "inference" | "kv_cold" | "ssm_cold"

[cluster.kv_cold]
host = "192.168.50.11"      # LAN address of your KV cold node
port = 50051

[cluster.ssm_cold]
host = "192.168.10.174"     # WiFi address of your SSM cold node
port = 50051

[cluster.recompute]
host = "127.0.0.1"          # Usually localhost on inference node
port = 50052

[inference]
exo_path = "/home/user/exo/src/exo"
log_file  = "/tmp/tierkv.log"
memory_threshold = 0.60     # Evict hot KV when RAM exceeds 60%
kv_dim   = 256              # head_dim for TurboQuant (Qwen3.6-35B-A3B = 256)

[vault]
port = 50051
```

`tierkv.toml` is gitignored — it contains your private IPs. Only `tierkv.toml.example` is committed.

---

## Usage

### 1. Start cold vault servers

On each cold-tier machine (`kv_cold` and `ssm_cold`):

```bash
tierkv vault --port 50051
```

Or as a background service (macOS launchd / Linux systemd).

### 2. Install the EXO hook

On the inference node:

```bash
tierkv install --exo-path /path/to/exo/src/exo
```

Then add to EXO's `builder.py` (after `KVPrefixCache` is initialized):

```python
from tierkv.exo.hook import install_kv_tiering_hook
install_kv_tiering_hook()
```

### 3. Check cluster status

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

### 4. Run benchmark

```bash
tierkv bench --exo-api http://192.168.50.11:52415
```

---

## TurboQuant

tierkv includes a **per-group INT8 quantizer** for KV tensor compression before sending over the network.

- Group size: 256 floats (matches Qwen3.6-35B-A3B head_dim)
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

**Why EXO?** [EXO](https://github.com/exo-explore/exo) provides an OpenAI-compatible API layer and handles model loading across Apple Silicon and CUDA devices. tierkv monkey-patches EXO's `KVPrefixCache` eviction and retrieval paths without modifying EXO's core.

**What about multi-node inference?** EXO supports pipeline-parallel inference (splitting layers across machines). tierkv is currently designed for single-node inference with distributed cold storage. The two can coexist but require separate configuration.

---

## Cluster Tested

| Node | Role | Memory | Network |
|---|---|---|---|
| DGX Spark | Inference (Qwen3.6-35B-A3B BF16) | 128 GB | WiFi + 10GbE LAN |
| Mac Pro (M2 Pro) | KV cold tier | 32 GB | 10GbE LAN (0.5ms) |
| Mac Air (M2) | SSM cold tier | 16 GB | WiFi (6ms) |

Over one test session: 227 evictions, 6 successful cold restores, ~26s saved per restore.

---

## Roadmap

- Persistent cold storage (SQLite / memory-mapped file — survive reboots)
- TurboQuant codebook training on real KV activations (push SNR higher)
- Mac Air on LAN (WiFi is the current SSM restore bottleneck)
- Quantization quality validation with `_kv_offsets` fix in place
- EXO version detection for hook compatibility
