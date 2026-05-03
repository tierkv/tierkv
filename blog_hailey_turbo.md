# Building a Distributed AI Inference Cluster with 3-Tier KV Caching: Lessons from Project Hailey-Turbo

## The Goal

We wanted to run a serious large language model — Qwen3.6-35B-A3B, a 70GB BF16 model — on consumer and prosumer hardware we already owned, and make it *fast*. Not just "it runs" fast, but intelligently fast: by keeping the expensive parts of inference (the KV cache) alive across requests, even after the primary GPU memory evicts them.

The result is what we call **Hailey-Turbo**: a 3-tier distributed KV cache system that turns a DGX Spark, a Mac Pro, and a Mac Air into a coordinated AI inference cluster — with a 7.3× reduction in time-to-first-token on repeated prompts.

---

## The Hardware

| Node | Role | Network |
|------|------|---------|
| **DGX Spark** | Primary inference — full 70GB model runs here | WiFi: 192.168.10.227, LAN: 192.168.50.10 |
| **Mac Pro** | KV cold tier — stores key-value cache for full-attention layers | LAN: 192.168.50.11 (0.5ms from DGX) |
| **Mac Air** | SSM cold tier — stores residual/linear-attention cache | WiFi: 192.168.10.174 (6ms from DGX) |

The DGX Spark has 128GB RAM and an NVIDIA GPU with enough headroom to load the full Qwen3.6-35B-A3B BF16 model via memory-mapped NVMe. The Macs serve as extended memory — a kind of hardware swap for the AI's working context.

---

## The Software Foundation: EXO

We built on top of [EXO](https://github.com/exo-explore/exo), an open-source distributed inference framework that handles model sharding, request routing, and the OpenAI-compatible API layer. EXO runs on the DGX in single-node mode ("hailey-dgx-solo" namespace) — it doesn't peer with the Macs for inference, but our custom Rust extension intercepts its internal KV cache eviction events.

The key EXO internals we hooked into:
- **`KVPrefixCache._evict_if_needed()`** — fires when hot RAM fills up (we set the threshold to 60%)
- **`KVPrefixCache` retrieval path** — intercepts cache misses to restore from cold storage before falling through to a full re-prefill

---

## The Architecture: Three Tiers

### Tier 1: Hot (DGX RAM)
The model's internal KV prefix cache — fast, in-process, but limited. At 60% memory utilization, EXO evicts the least-recently-used entry. Without Hailey, that data is simply thrown away.

### Tier 2: Cold KV (Mac Pro via LAN)
When Qwen3.6-35B processes a prompt, 10 of its 40 layers use full-attention (KVCache). These are the most memory-hungry layers — for an 8,000-token prompt, each layer's K+V tensors are ~29MB, totalling ~290MB for all 10 layers. We intercept the eviction and ship these to Mac Pro over a LAN gRPC connection at 0.5ms RTT.

### Tier 3: Cold SSM (Mac Air via WiFi)
The remaining 30 layers use linear attention (ArraysCache). These are smaller per-layer but numerous — ~2.1MB each, totalling ~65MB. These go to Mac Air over WiFi.

The split was dictated by the model architecture: Qwen3.6-35B-A3B is a hybrid MoE model where most layers are linear-attention SSM-style, not traditional transformer attention.

---

## The Rust Core: hailey-core

We wrote the gRPC transport and quantization engine in Rust, exposed to Python via PyO3. The crate lives at `hailey-core/` and compiles to a Python wheel using `maturin`.

### TurboQuant: INT8 Per-Group Quantization

Sending 290MB of BF16 KV data on every eviction would saturate even the LAN. We built **TurboQuant**, a per-group INT8 quantizer:

- Split the tensor into groups of 256 floats (matching Qwen3.6's head dimension)
- Each group gets its own absmax scale: `scale = max(|x|) / 127`
- Wire format: `[scale: f32 LE][quantized: i8 × 256]` per group
- Compression ratio: ~3.9× (1024 bytes → 260 bytes per group)
- SNR: ≥52 dB on typical KV distributions (above the ~40 dB threshold for quality preservation)

The per-group approach was critical. A single outlier value in a tensor (we saw values up to 100× the typical range) would destroy per-tensor quantization SNR. Per-group isolates the outlier to its group while leaving all others clean.

### gRPC: ColdVaultService

Two services defined in protobuf:
- **`ColdVaultService`** — `Store(KVTensor)`, `Promote(token_idx, layer)`, `BatchPromote(token_idx, layers[])`
- **`RecomputeService`** — stub for future on-device recomputation

The Mac Pro and Mac Air each run a `ColdVaultServer` instance — an in-memory HashMap backed by tokio's async RwLock.

---

## Phase Evolution: What We Built, Step by Step

### Phase 1: Get EXO Running on DGX
Simple in theory, brutal in practice. The DGX has both WiFi and a 10GbE LAN. EXO's peer discovery needed to be isolated so it wouldn't try to shard the model across the Macs (which run macOS and can't use the GPU for this model). We patched EXO's `placement_utils.py` to bypass its memory filter and used a custom namespace to prevent cross-node discovery.

**Challenge:** Every EXO restart causes a 2–3 minute SSH lockout. The 70GB model loads via `mmap` from NVMe, saturating the DGX CPU during page-fault resolution. Pings work fine — SSH auth just times out. We learned to wait rather than assume the machine was dead.

### Phase 2: The Eviction Hook
We monkey-patched EXO's `_evict_if_needed()` at import time via `builder.py`. The hook intercepts the eviction, serializes the KV and SSM tensors, and fires async gRPC store calls to Mac Pro and Mac Air without blocking the inference thread.

**Challenge:** The first implementation used fire-and-forget gRPC (`let _ = client.store(kv).await`). This completely hid a critical bug — tonic's default 4MB message size limit was silently dropping our 7.25MB INT8 Store requests. The symptom was just "cold restore always returns empty" with no error logged anywhere. The fix: set 512MiB limits on all four combinations (client encode/decode, server encode/decode).

### Phase 3: The Restore Path
Getting eviction working was half the problem. The other half was restoring the cache on a subsequent request. EXO had to believe the prompt had already been processed.

Four bugs blocked us:

1. **Wrong cache miss detection.** EXO returns a non-None `matched_idx` even for shared system-prompt prefixes. We were checking `matched_idx is None`, which never fired. Fixed by checking whether `len(remaining) == len(prompt_tokens)` — a full miss.

2. **Snapshot structure.** Qwen3.6 requires a `CacheSnapshot` with SSM state to return anything. Injecting just the KV entries left EXO thinking the cache was empty. Fixed by calling `snapshot_ssm_states()` on the restored 40-layer list.

3. **KVCache.offset semantics.** MLX pre-allocates KV buffers to `ceil(n/256)*256`. For a prompt of 8,792 tokens, `keys.shape[2] = 8,960` but only 8,792 positions are valid. Setting `offset = keys.shape[2]` told the attention mechanism there were 168 extra garbage tokens — output became a repetitive loop of `import urllib...`. Fixed: store the actual `entry.offset` at eviction time in a `_kv_offsets` dict.

4. **Empty remaining list.** After restoring, if `remaining = []` (no tokens left to prefill), MLX's batch generator crashed with `IndexError: list index out of range`. Fixed by passing the last token of the prompt as a 1-token prefill.

**Result after Phase 3:** 23.78s cold → 4.59s restored = **5.2× speedup** on a 3,707-token prompt.

### Phase 4: Batch gRPC — 40 RPCs → 2 RPCs
The restore path originally made 40 sequential gRPC calls — one `Promote` RPC per layer. Each call was a network round-trip: 6ms over WiFi × 30 SSM layers = 180ms minimum, plus serialization overhead.

We added a `BatchPromote` RPC that fetches all layers for a given prompt in a single call. Two calls run in parallel via `ThreadPoolExecutor`: one to Mac Pro for all 10 KV layers, one to Mac Air for all 30 SSM layers.

**Challenge:** The 4MB tonic limit again — but this time on the *response*. A `BatchPromoteResponse` containing 30 SSM layers totals ~65MB. Same fix: 512MiB limits everywhere.

**Result after Phase 4:** 30.83s cold → 4.11s restored = **7.3× speedup** on an 8,000-token prompt. Time saved per request: ~26 seconds.

### Phase 5: The Spawn Bug
With the core pipeline working, we noticed EXO's runner process would intermittently fail to spawn on the second attempt after a crash with:

```
FileNotFoundError: [Errno 2] No such file or directory — SemLock._rebuild
```

This was a subtle Python multiprocessing bug. EXO's `supervisor.py` passed the loguru `logger` object directly into `mp.Process(args=(..., logger, ...))`. Loguru's `enqueue=True` handlers internally create `multiprocessing.Queue`, `Event`, and `Lock` objects — each backed by a POSIX named semaphore in `/dev/shm`. When the first runner child process exited, something in the resource cleanup path `sem_unlink`'d one of these semaphores. The second spawn attempt tried to rebuild it via `SemLock._rebuild(name)` → `sem_open(name, 0)` → file not found.

The fix: don't pass the logger to the child at all. Remove it from `mp.Process(args=...)` in `supervisor.py`, and at the top of the child's `entrypoint()` function, call `logger_setup(EXO_LOG, 0)` to create a fresh logger with its own semaphores.

---

## The Numbers

| Scenario | TTFT | vs Cold |
|----------|------|---------|
| Cold start, 8,000-token prompt | 30.83s | baseline |
| Restored from cold tier | 4.11s | **7.3× faster** |
| Cold start, 3,707-token prompt | 23.78s | baseline |
| Restored from cold tier (Phase 3) | 4.59s | **5.2× faster** |

Over the course of our testing session:
- **227 KV evictions** to Mac Pro
- **227 SSM evictions** to Mac Air (paired)
- **6 successful cold restores** (the rest were cache misses from short test prompts)
- **~26 seconds saved per restored request**

---

## What Made This Hard

**The heterogeneous model architecture.** Qwen3.6-35B-A3B is a Mixture-of-Experts hybrid — 10 layers use standard multi-head attention (KVCache), 30 use linear attention (ArraysCache). These have completely different serialization requirements and different cold storage backends. Every component had to handle both paths.

**Silent failures everywhere.** gRPC's default message size limits, fire-and-forget async eviction, Python's multiprocessing semaphore lifecycle — every failure mode was silent. The system appeared to work but produced wrong results. We spent significant time adding instrumentation before we could even see what was failing.

**EXO's internal assumptions.** EXO's prefix cache was designed for in-process use. Hooking into it externally required understanding undocumented invariants about `matched_idx`, `remaining`, snapshot structure, and offset semantics — none of which have public documentation. Every fix required reading CPython source, MLX internals, and EXO's async machinery in parallel.

**The DGX SSH lockout.** Every restart of EXO triggers 2–3 minutes of SSH unavailability while the 70GB model memory-maps from NVMe. Early on we'd assume the machine was down and attempt recovery steps that made things worse. The rule became: if ping works, just wait.

---

## What's Next

- **Quantization quality validation** — TurboQuant's INT8 gives ≥52 dB SNR on synthetic distributions, but we haven't run a formal quality test with the `_kv_offsets` fix in place. That's the next checkpoint.
- **Mac Air on LAN** — The WiFi link to Mac Air (6ms, ~100MB/s) is the bottleneck for SSM restore. Moving Mac Air to the LAN would bring it to 0.5ms parity with Mac Pro.
- **Persistent cold storage** — Currently, Mac Pro and Mac Air hold KV data only in-memory. A restart loses everything. SQLite or a memory-mapped file would make the cold tier durable across reboots.
- **TurboQuant codebook training** — The INT8 approach is solid but a trained codebook (on real Qwen3.6 KV activations) would push compression higher while maintaining SNR.

---

## Closing Thought

The most surprising thing about this project wasn't how hard the distributed systems problems were — it was how much of the difficulty came from *silent failures*. Every major bug looked like "the system works but produces wrong output." Building in observability (explicit error logging, SNR tests, timing instrumentation) at each layer was the only thing that made progress possible.

The final system — a Rust gRPC transport, Python hooks, and three machines coordinating to serve a 70GB model with 7× faster repeat responses — runs stably. The fundamentals work. Now it's about hardening the edges.
