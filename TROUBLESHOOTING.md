# Troubleshooting tierkv

Hard-won lessons from deploying tierkv across a DGX Spark + Mac Pro + Mac Air cluster running Qwen3.6-35B-A3B. These are real failures that cost hours to debug — documented so you don't repeat them.

---

## gRPC / tonic Issues

### BatchPromote returns empty results silently

**Symptom:** `get_batch()` returns `{}` despite confirmed evictions. Hook log shows `restore: missing layer 0`. No exception raised anywhere.

**Root cause:** tonic 0.12's default `max_decoding_message_size` is 4 MB. A `BatchPromoteResponse` for 10 KV layers × ~1.87 MB compressed each = 18.7 MB. tonic silently drops the response.

**Fix:** tierkv already sets 512 MiB limits on both client and server. If you are building a custom gRPC client against tierkv-core, set:
```rust
.max_decoding_message_size(512 * 1024 * 1024)
.max_encoding_message_size(512 * 1024 * 1024)
```
on both client and server, in both directions.

**How to diagnose:** Size bisection — insert N layers, call `get_batch(N)`, find the N where it starts silently failing. Failure at ~3 layers (~5.6 MB compressed) and success at 2 layers (~3.7 MB) points exactly to the 4 MB tonic default.

**Rule:** The 4 MB limit applies to BOTH sides, BOTH directions (client enc/dec + server enc/dec) independently. Fixing only one side is not enough.

---

### Fire-and-forget gRPC hides all errors

**Symptom:** Cold stores appear to succeed (no errors logged), but nothing is in the vault on restore. Hot tier evicts the KV, then `get_batch` returns empty.

**Root cause:** The eviction path uses fire-and-forget (`let _ = client.store(kv).await`) to avoid blocking inference. This makes errors completely invisible.

**Fix:** During initial deployment and debugging, instrument the fire-and-forget path with at least a `tracing::warn!` on failure:
```rust
if let Err(e) = client.store(kv).await {
    tracing::warn!("tierkv store failed: {e}");
}
```
Switch back to silent discard only after correctness is validated end-to-end.

---

## kv_dim — The Silent Corruption Trap

**Symptom:** Restore appears to work (no errors, correct TTFT speedup) but model output is subtly wrong or garbled.

**Root cause:** `kv_dim` in `tierkv.toml` controls TurboQuant's group size for INT8 compression. If it doesn't match your model's attention head dimension (`hidden_size / num_attention_heads`), compression and decompression use different groupings — producing incorrect tensors with no error raised.

**How to find the right value:**
```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B")
print(cfg.hidden_size // cfg.num_attention_heads)  # → 256
```

**Common values:**
| Model | kv_dim |
|---|---|
| Qwen3.6-35B-A3B | 256 |
| Llama-3 8B / 70B | 128 |
| Qwen2.5 7B–72B | 128 |
| Mistral 7B | 128 |
| Phi-3 medium | 96 |

**Rule:** Always verify `kv_dim` matches your model before your first production run. Wrong value = silent incorrect compression with ≥40 dB SNR degradation per group.

---

## KVCache.offset Semantics (EXO)

### Model outputs garbage after restore (repetitive loops, wrong tokens)

**Symptom:** After a cold restore, model starts generating `import urllib...` loops, repeats phrases, or produces off-topic output. The TTFT speedup is correct but generation quality is broken.

**Root cause 1 — Using buffer shape instead of token count:**
MLX pre-allocates key/value buffers to `ceil(ntok / step) * step` (step=256). For 8792 tokens, `keys.shape[2] = 8960` but valid tokens = 8792. Setting `offset = 8960` tells the attention mechanism there are 168 extra garbage tokens.

**Fix:** Set `c.offset = len(prompt_tokens)`, NOT `c.keys.shape[2]`.

**Root cause 2 — Not preserving the original offset at eviction time:**
The prefix cache stores KV for the full tokenized sequence including chat template tokens (`<|im_start|>user\n`, etc.). If you restore with `c.offset = len(user_tokens_only)` instead of the eviction-time offset, the model regenerates the assistant template token it thinks is missing, producing `<|im_start|>` as the first output token.

**Fix:** Store `entry.offset` at eviction time:
```python
# On eviction:
_kv_offsets[token_idx] = int(entry.offset)

# On restore:
c.offset = _kv_offsets.get(token_idx, len(prompt_tokens))
```

---

## Stale Semaphores After kill -9

**Symptom:** After `kill -9` on EXO, the restart immediately exits with `Runner terminated with exitcode=1`. Log shows `FileNotFoundError` in `spawn_main`.

**Root cause:** `kill -9` bypasses Python cleanup hooks. `multiprocessing` leaves named semaphores in `/dev/shm`. On restart, `spawn_main` tries to reopen them and fails.

**Fix — always run before restarting EXO after a hard kill:**
```bash
find /dev/shm -name 'sem.*' -user $USER -delete
```

---

## EXO-Specific Issues (DGX solo mode)

### Nack loop — place_instance accepted but no worker spawns

**Symptom:** `Nack attempt 1: Requesting Event Log from 145332` fills logs at ~1 line/second. `place_instance` returns "Command received" but no worker spawns. `/v1/chat/completions` returns 404 "No instance found".

**Root cause:** EXO persists its event log to `~/.local/share/exo/event_log/`. After a hard reset, EXO tries to fill the gap between the persisted log index and new events. In solo mode (no real EXO peers), the gap never fills and placement commands are swallowed.

**Fix:**
```bash
mv ~/.local/share/exo/event_log ~/.local/share/exo/event_log.old
pkill -9 -f "exo --libp2p"
# systemd restarts EXO clean
```

Do this **before** the first EXO restart after a hard reset.

---

### Election storm — "demoting self" loop

**Symptom:** Log rapidly oscillates `"Node X elected master - demoting self"` → `"Node elected Master - promoting self"` → repeat. No stable master, no worker.

**Root cause:** `--bootstrap-peers` points to Mac Pro/Air which run only tierkv vault (no EXO). These machines are partially reachable at TCP level, triggering `ConnectionMessage` events that restart the election every 3 seconds.

**Fix:** If you run EXO in solo mode, remove `--bootstrap-peers` from your EXO startup command entirely. If your systemd unit hardcodes it, patch the EXO entrypoint to strip it:

```python
# /home/pk/exo/.venv/bin/exo — wrapper
import sys, os
os.environ["EXO_BOOTSTRAP_PEERS"] = ""
sys.argv = [a for a in sys.argv if not a.startswith("--bootstrap-peers")]
from exo.main import main; main()
```

**Warning:** This wrapper is overwritten by `uv sync`. Reapply after EXO upgrades.

---

### Memory threshold not triggering eviction after fresh boot

**Symptom:** Stress test shows high TTFT on second request (expected eviction didn't happen). Hook log shows `is_exact=True cache_entries=1` — hot cache hit instead of cold restore.

**Root cause:** After a fresh boot, the 70 GB model is mmap'd but pages aren't fully faulted in yet. RAM usage reads as ~59%, below the default `memory_threshold = 0.60`. Nothing evicts.

**Fix:** Lower `memory_threshold` in `tierkv.toml` to 0.55 for the first run after boot, then raise it back:
```toml
[inference]
memory_threshold = 0.55
```

---

## SSH Lock on DGX During Model Load

**Symptom:** `ssh` returns "Connection timed out during banner exchange" but `ping` still works. Happens every time EXO restarts and loads Qwen3.6-35B-A3B (70 GB).

**Root cause:** Loading 70 GB of safetensors via mmap saturates CPU memory bandwidth. Linux sshd stalls during key exchange because the kernel is overwhelmed with page faults from the model load. Network layer is fine — TCP is not.

**Workaround:** Always use `ConnectTimeout=45` or higher when SSHing to DGX after an EXO restart:
```bash
ssh -o ConnectTimeout=45 pk@192.168.10.227
```

Wait 2–3 minutes and retry. DGX is not down — SSH auth is just delayed.

---

## Wheel / Installation Issues

### Wrong platform wheel installed on Linux

**Symptom:** `import tierkv_core` fails with `ELF load command address/offset not properly aligned` or similar.

**Root cause:** `pip install --force-reinstall` and `uv pip install --force-reinstall` bypass platform compatibility checks. A `macosx_11_0_arm64` wheel can be force-installed on Linux where it won't load.

**Fix:** Always build wheels on the target platform. DGX (Linux aarch64) requires a `manylinux_2_28_aarch64` wheel — never a macOS wheel:
```bash
# On DGX (aarch64 Linux):
cd tierkv-core && maturin build --release --out /tmp/dist/
uv pip install --force-reinstall /tmp/dist/tierkv_core-*aarch64*.whl
```

The GitHub Actions CI builds the correct wheel per platform. Download it from the release assets.

---

### maturin develop doesn't install to the active venv

**Symptom:** `python3 -c "import tierkv_core"` succeeds from the project directory but fails from `/tmp` or any other path.

**Root cause:** `uv run maturin develop` installs to a temporary location, not the active venv's site-packages.

**Fix:** After `maturin develop --release`, manually verify and copy:
```bash
# Check where it installed:
python3 -c "import tierkv_core; print(tierkv_core.__file__)"

# If not in your venv, copy it:
cp /path/to/tierkv_core.abi3.so \
   .venv/lib/python3.12/site-packages/tierkv_core/tierkv_core.abi3.so
```

---

## Cold Vault — Data Loss on Restart

**Warning:** The vault server stores all KV data **in memory only**. Restarting vault servers wipes all stored KV.

**Rule:** Always restart cold vault servers (`tierkv vault`) **before** the eviction phase of any test — not after. If you restart the vault after eviction, the evicted KV is gone and the next request will cold-prefill instead of restoring.

Order of operations for a clean test:
1. Restart vault servers on Mac Pro and Mac Air
2. Start / restart EXO on DGX
3. Run inference (eviction happens automatically at `memory_threshold`)
4. Run second request (restore from cold)

---

## Quality Testing Pitfalls

### Repeated context text makes model output look broken

**Symptom:** Model responds "It seems like your message contains garbled or repetitive text" even on a cold prefill (no restore involved). Test is useless.

**Root cause:** Padding a prompt with repeated sentences (e.g., 50× "This technological acceleration fundamentally reshaped...") confuses the model regardless of tierkv.

**Fix:** Use genuinely varied narrative text for test prompts. Multi-turn conversation history works well. Never use a while loop that repeats a single sentence.

---

### Short prompts show no speedup

**Symptom:** Cold restore completes but TTFT speedup is only 1.1× or less.

**Root cause:** For prompts under ~1,000 tokens, prefill is fast regardless — there's no expensive computation to skip.

**Rule:** Use prompts of at least **3,000 tokens** for meaningful TTFT speedup measurement. 5,000–8,000 tokens is ideal. The speedup scales with prompt length.

---

### Historical/violent prompt content triggers safety refusals

**Symptom:** Restore succeeds (correct TTFT) but model output is an off-topic safety refusal. Looks like a quality failure but isn't.

**Fix:** Use neutral topics for quality validation prompts. ML papers, software documentation, or fictional dialogue work well.

---

## Logger Passes to multiprocessing.Process

**Symptom:** `Runner terminated with exitcode=1` immediately on second worker spawn. First spawn works fine. Log shows `FileNotFoundError: [Errno 2] No such file or directory` inside `SemLock._rebuild`.

**Root cause:** Loguru handlers with `enqueue=True` hold POSIX semaphores in `/dev/shm`. When a loguru `Logger` instance is pickled and sent to a spawned child, the child calls `sem_open(name, 0)` to reopen the semaphore. Between the first and second spawn, the resource tracker unlinks the semaphore — the second child can't reopen it.

**Fix:** Never pass a loguru logger as an argument to `mp.Process()`. Call `logger_setup()` inside the child's entrypoint function to create a fresh logger:
```python
# Wrong:
mp.Process(target=worker, args=(logger, channel))

# Right:
mp.Process(target=worker, args=(channel,))

# In worker:
def worker(channel):
    logger = logger_setup()  # fresh logger in child
    ...
```

---

## vLLM-Specific

### TierKVConnector import fails without vLLM installed

This is expected. The connector has a graceful fallback:
```python
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
except ImportError:
    KVConnectorBase_V1 = object  # EXO-only installs still work
```
If you see an ImportError for `vllm`, install it: `pip install vllm`.

---

### vLLM install fails on aarch64 Linux (fastsafetensors build error)

**Symptom:** `pip install vllm` fails with `error: command '/usr/bin/aarch64-linux-gnu-gcc' failed — Python.h: No such file or directory`.

**Root cause:** `fastsafetensors` builds a C extension that requires Python development headers.

**Fix:**
```bash
sudo apt-get install -y python3.12-dev   # or python3.X-dev for your version
pip install vllm
```

---

### kv_dim for vLLM connector

The vLLM connector reads `kv_dim` from `tierkv.toml` (under `[tierkv]` or `[inference]`) or from `--kv-connector-extra-config`. Same rule applies as for the EXO hook — must match your model's attention head dimension. See [kv_dim table above](#kv_dim--the-silent-corruption-trap).
