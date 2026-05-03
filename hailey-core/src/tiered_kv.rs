//! TieredKVCache — 3-tier KV cache manager exposed to Python via PyO3.
//!
//! Tiers:
//!   Hot  (Mac Pro RAM)  — DashMap, TurboQuant 3.5-bit, 8K–16K tokens, LRU eviction.
//!   Cold (Mac Air RAM)  — gRPC ColdVaultService, BF16 residuals, ~240K tokens.
//!   Miss (DGX Spark)    — gRPC RecomputeService, recomputes KV from FP4 weights.
//!
//! Python API:
//!   cache = TieredKVCache(hot_capacity, cold_addr, recompute_addr)
//!   cache.insert(token_idx, layer, data_bytes)   # f32 bytes in
//!   data_bytes = cache.get(token_idx, layer)      # f32 bytes out (may block)
//!   stats = cache.stats()                         # {"hits", "misses", "evictions"}

use crate::turbo_quant::TurboQuant;
use dashmap::DashMap;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;

/// Compressed hot-tier entry.
struct HotEntry {
    token_idx: u64,
    layer: u32,
    compressed: Vec<u8>, // TurboQuant-encoded bytes
    dim: usize,          // original f32 dimension (needed for decode)
}

struct HotTier {
    map: DashMap<(u64, u32), Vec<u8>>,    // key → compressed bytes
    dims: DashMap<(u64, u32), usize>,     // key → original dim
    lru: Mutex<VecDeque<(u64, u32)>>,    // LRU order
    capacity: usize,
}

impl HotTier {
    fn new(capacity: usize) -> Self {
        HotTier {
            map: DashMap::new(),
            dims: DashMap::new(),
            lru: Mutex::new(VecDeque::new()),
            capacity,
        }
    }

    /// Insert; returns evicted entry (token_idx, layer, compressed, dim) if over capacity.
    fn insert(&self, key: (u64, u32), compressed: Vec<u8>, dim: usize) -> Option<HotEntry> {
        self.map.insert(key, compressed);
        self.dims.insert(key, dim);

        let mut lru = self.lru.lock().unwrap();
        lru.retain(|k| k != &key);
        lru.push_back(key);

        if lru.len() > self.capacity {
            let evicted_key = lru.pop_front().unwrap();
            let compressed = self.map.remove(&evicted_key).map(|(_, v)| v).unwrap_or_default();
            let dim = self.dims.remove(&evicted_key).map(|(_, v)| v).unwrap_or(0);
            Some(HotEntry {
                token_idx: evicted_key.0,
                layer: evicted_key.1,
                compressed,
                dim,
            })
        } else {
            None
        }
    }

    fn get(&self, key: &(u64, u32)) -> Option<(Vec<u8>, usize)> {
        let compressed = self.map.get(key)?.clone();
        let dim = *self.dims.get(key)?;
        // Promote to MRU
        let mut lru = self.lru.lock().unwrap();
        lru.retain(|k| k != key);
        lru.push_back(*key);
        Some((compressed, dim))
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

struct Stats {
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Python-visible 3-tier KV cache.
#[pyclass]
pub struct TieredKVCache {
    hot: Arc<HotTier>,
    quant: Arc<TurboQuant>,
    cold_addr: String,
    recompute_addr: String,
    rt: Arc<Runtime>,
    stats: Arc<Mutex<Stats>>,
}

#[pymethods]
impl TieredKVCache {
    /// Args:
    ///   hot_capacity    — max token×layer entries in hot tier (default 16384)
    ///   dim             — KV vector dimension (must be divisible by 8)
    ///   cold_addr       — Mac Air gRPC address e.g. "192.168.50.12:50051"
    ///   recompute_addr  — DGX Spark gRPC address e.g. "192.168.50.10:50052"
    #[new]
    pub fn new(
        hot_capacity: usize,
        dim: usize,
        cold_addr: String,
        recompute_addr: String,
    ) -> PyResult<Self> {
        let quant = TurboQuant::new(dim)?;
        let rt = Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("tokio runtime: {e}"))
        })?;
        Ok(TieredKVCache {
            hot: Arc::new(HotTier::new(hot_capacity)),
            quant: Arc::new(quant),
            cold_addr,
            recompute_addr,
            rt: Arc::new(rt),
            stats: Arc::new(Mutex::new(Stats { hits: 0, misses: 0, evictions: 0 })),
        })
    }

    /// Insert a KV tensor (f32 bytes) into the hot tier.
    /// If the hot tier is full, the oldest entry is evicted to cold (Mac Air).
    pub fn insert(&self, token_idx: u64, layer: u32, data: &[u8]) -> PyResult<()> {
        let compressed = self.quant.encode(data)?;
        let dim = data.len() / 4; // f32 = 4 bytes

        if let Some(evicted) = self.hot.insert((token_idx, layer), compressed, dim) {
            // Evict to cold tier asynchronously
            let cold_addr = self.cold_addr.clone();
            let token_idx_e = evicted.token_idx;
            let layer_e = evicted.layer;
            let data_e = evicted.compressed;
            let rows_e = evicted.dim as u32;

            let stats = Arc::clone(&self.stats);
            self.rt.spawn(async move {
                use crate::residual_sync::{hailey::KvTensor, ColdVaultClient};
                if let Ok(mut client) = ColdVaultClient::connect(&cold_addr).await {
                    let kv = KvTensor {
                        token_idx: token_idx_e,
                        layer: layer_e,
                        data: data_e,
                        rows: rows_e,
                        cols: 1,
                        dtype: "f32".into(),
                    };
                    let _ = client.store(kv).await;
                    stats.lock().unwrap().evictions += 1;
                }
            });
        }
        Ok(())
    }

    /// Get a KV tensor (f32 bytes). Checks hot → cold → recompute in order.
    pub fn get(&self, py: Python<'_>, token_idx: u64, layer: u32) -> PyResult<PyObject> {
        let key = (token_idx, layer);

        // ── Hot tier hit ──────────────────────────────────────────────────────
        if let Some((compressed, dim)) = self.hot.get(&key) {
            self.stats.lock().unwrap().hits += 1;
            // Reconstruct f32 bytes from quantized data
            let mut full_data = vec![0u8; dim * 4];
            let decoded = self.quant.decode(&compressed)?;
            let copy_len = full_data.len().min(decoded.len());
            full_data[..copy_len].copy_from_slice(&decoded[..copy_len]);
            return Ok(pyo3::types::PyBytes::new_bound(py, &full_data).into());
        }

        self.stats.lock().unwrap().misses += 1;

        // ── Cold tier promotion ───────────────────────────────────────────────
        let cold_addr = self.cold_addr.clone();
        let recompute_addr = self.recompute_addr.clone();
        let hot = Arc::clone(&self.hot);
        let quant = Arc::clone(&self.quant);

        let data = self.rt.block_on(async move {
            use crate::residual_sync::{hailey::KvTensor, ColdVaultClient, RecomputeClient};

            // Try cold vault first
            if let Ok(mut client) = ColdVaultClient::connect(&cold_addr).await {
                if let Ok(kv) = client.promote(token_idx, layer).await {
                    if !kv.data.is_empty() {
                        // kv.data is TurboQuant-compressed (stored that way by insert()).
                        // Decode before returning f32 bytes to Python, same as hot-hit path.
                        let dim = kv.rows as usize;
                        let _ = hot.insert((token_idx, layer), kv.data.clone(), dim);
                        if let Ok(decoded) = quant.decode(&kv.data) {
                            let mut full = vec![0u8; dim * 4];
                            let n = full.len().min(decoded.len());
                            full[..n].copy_from_slice(&decoded[..n]);
                            return full;
                        }
                    }
                    return vec![];
                }
            }

            // Fall back to recompute on DGX Spark
            if let Ok(mut client) = RecomputeClient::connect(&recompute_addr).await {
                if let Ok(kv) = client.recompute(token_idx, layer).await {
                    if !kv.data.is_empty() {
                        if let Ok(compressed) = quant.encode(&kv.data) {
                            let _ = hot.insert((token_idx, layer), compressed, kv.rows as usize);
                        }
                    }
                    return kv.data;
                }
            }

            vec![] // full miss — caller must handle
        });

        Ok(pyo3::types::PyBytes::new_bound(py, &data).into())
    }

    /// Get multiple KV tensors (f32 bytes) in a single batch RPC.
    ///
    /// Returns a Python list of `(layer: int, data: bytes)` tuples sorted by layer.
    /// Hot hits are decoded inline; cold layers are fetched in one `BatchPromote` RPC.
    /// Missing layers return an empty bytes object.
    pub fn get_batch(
        &self,
        py: Python<'_>,
        token_idx: u64,
        layers: Vec<u32>,
    ) -> PyResult<PyObject> {
        let mut results: Vec<(u32, Vec<u8>)> = Vec::with_capacity(layers.len());
        let mut cold_layers: Vec<u32> = vec![];

        for &layer in &layers {
            let key = (token_idx, layer);
            if let Some((compressed, dim)) = self.hot.get(&key) {
                self.stats.lock().unwrap().hits += 1;
                let decoded = self.quant.decode(&compressed)?;
                let mut full = vec![0u8; dim * 4];
                let n = full.len().min(decoded.len());
                full[..n].copy_from_slice(&decoded[..n]);
                results.push((layer, full));
            } else {
                self.stats.lock().unwrap().misses += 1;
                cold_layers.push(layer);
            }
        }

        if !cold_layers.is_empty() {
            let cold_addr = self.cold_addr.clone();
            let quant = Arc::clone(&self.quant);
            let hot = Arc::clone(&self.hot);

            let cold_results = self.rt.block_on(async move {
                use crate::residual_sync::ColdVaultClient;
                let mut out: Vec<(u32, Vec<u8>)> = vec![];
                if let Ok(mut client) = ColdVaultClient::connect(&cold_addr).await {
                    if let Ok(tensors) = client.batch_promote(token_idx, cold_layers).await {
                        for kv in tensors {
                            if !kv.data.is_empty() {
                                let dim = kv.rows as usize;
                                // Re-insert into hot tier for subsequent accesses
                                let _ = hot.insert((token_idx, kv.layer), kv.data.clone(), dim);
                                if let Ok(decoded) = quant.decode(&kv.data) {
                                    let mut full = vec![0u8; dim * 4];
                                    let n = full.len().min(decoded.len());
                                    full[..n].copy_from_slice(&decoded[..n]);
                                    out.push((kv.layer, full));
                                    continue;
                                }
                            }
                            out.push((kv.layer, vec![])); // missing or empty
                        }
                    }
                }
                out
            });
            results.extend(cold_results);
        }

        results.sort_by_key(|(layer, _)| *layer);
        let list = pyo3::types::PyList::empty_bound(py);
        for (layer, data) in results {
            let tup = pyo3::types::PyTuple::new_bound(
                py,
                &[layer.to_object(py), pyo3::types::PyBytes::new_bound(py, &data).into()],
            );
            list.append(tup)?;
        }
        Ok(list.into())
    }

    /// Returns {"hits", "misses", "evictions", "hot_size", "hit_rate"}.
    pub fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let s = self.stats.lock().unwrap();
        let total = s.hits + s.misses;
        let hit_rate = if total > 0 { s.hits as f64 / total as f64 } else { 0.0 };
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("hits", s.hits)?;
        d.set_item("misses", s.misses)?;
        d.set_item("evictions", s.evictions)?;
        d.set_item("hot_size", self.hot.len())?;
        d.set_item("hit_rate", hit_rate)?;
        Ok(d.into())
    }
}
