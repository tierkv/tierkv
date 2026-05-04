//! ResidualSync — async gRPC client stubs for ColdVaultService and RecomputeService.
//!
//! Both services are defined in proto/tierkv.proto and compiled by build.rs.
//! This module wraps the generated tonic clients in tokio-aware helpers
//! that can be called from the TieredKVCache without blocking the async runtime.

use tonic::transport::Channel;

// Pull in the generated protobuf types and clients.
pub mod tierkv {
    tonic::include_proto!("tierkv");
}

use tierkv::{
    cold_vault_service_client::ColdVaultServiceClient,
    recompute_service_client::RecomputeServiceClient,
    vllm_cold_vault_service_client::VllmColdVaultServiceClient,
    BatchPromoteRequest, KvTensor, PromoteRequest, RecomputeRequest, StoreRequest,
    VllmBlockStoreRequest, VllmBatchPromoteRequest,
};

// ── Cold Vault client ─────────────────────────────────────────────────────────

pub struct ColdVaultClient {
    inner: ColdVaultServiceClient<Channel>,
}

impl ColdVaultClient {
    pub async fn connect(addr: &str) -> Result<Self, tonic::transport::Error> {
        let inner = ColdVaultServiceClient::connect(format!("http://{addr}"))
            .await?
            .max_decoding_message_size(512 * 1024 * 1024) // 512 MiB — batch responses can be >4 MB
            .max_encoding_message_size(512 * 1024 * 1024); // 512 MiB — INT8 Store requests can be >4 MB
        Ok(ColdVaultClient { inner })
    }

    /// Evict a KV tensor to cold storage.
    pub async fn store(&mut self, kv: KvTensor) -> Result<bool, tonic::Status> {
        let req = tonic::Request::new(StoreRequest { kv: Some(kv) });
        let resp = self.inner.store(req).await?;
        Ok(resp.into_inner().ok)
    }

    /// Promote a cold KV tensor back to hot tier.
    pub async fn promote(
        &mut self,
        token_idx: u64,
        layer: u32,
    ) -> Result<KvTensor, tonic::Status> {
        let req = tonic::Request::new(PromoteRequest { token_idx, layer });
        Ok(self.inner.promote(req).await?.into_inner())
    }

    /// Promote multiple layers in a single RPC — one network round-trip for all.
    /// Returns tensors in the same order as `layers`; missing layers have empty data.
    pub async fn batch_promote(
        &mut self,
        token_idx: u64,
        layers: Vec<u32>,
    ) -> Result<Vec<KvTensor>, tonic::Status> {
        let req = tonic::Request::new(BatchPromoteRequest { token_idx, layers });
        Ok(self.inner.batch_promote(req).await?.into_inner().tensors)
    }
}

// ── Recompute client ──────────────────────────────────────────────────────────

pub struct RecomputeClient {
    inner: RecomputeServiceClient<Channel>,
}

impl RecomputeClient {
    pub async fn connect(addr: &str) -> Result<Self, tonic::transport::Error> {
        let inner = RecomputeServiceClient::connect(format!("http://{addr}")).await?;
        Ok(RecomputeClient { inner })
    }

    /// Ask the inference node to recompute a KV tensor from weights (cache miss path).
    pub async fn recompute(
        &mut self,
        token_idx: u64,
        layer: u32,
    ) -> Result<KvTensor, tonic::Status> {
        let req = tonic::Request::new(RecomputeRequest { token_idx, layer });
        Ok(self.inner.recompute(req).await?.into_inner())
    }
}

// ── Cold Vault server ─────────────────────────────────────────────────────────

use tierkv::{
    cold_vault_service_server::{ColdVaultService, ColdVaultServiceServer},
    BatchPromoteResponse, StoreResponse,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Default)]
pub struct ColdVaultServer {
    store: Arc<RwLock<HashMap<(u64, u32), KvTensor>>>,
}

#[tonic::async_trait]
impl ColdVaultService for ColdVaultServer {
    async fn store(
        &self,
        req: tonic::Request<StoreRequest>,
    ) -> Result<tonic::Response<StoreResponse>, tonic::Status> {
        if let Some(kv) = req.into_inner().kv {
            self.store.write().await.insert((kv.token_idx, kv.layer), kv);
            Ok(tonic::Response::new(StoreResponse { ok: true }))
        } else {
            Err(tonic::Status::invalid_argument("missing kv tensor"))
        }
    }

    async fn promote(
        &self,
        req: tonic::Request<PromoteRequest>,
    ) -> Result<tonic::Response<KvTensor>, tonic::Status> {
        let r = req.into_inner();
        let mut guard = self.store.write().await;
        match guard.remove(&(r.token_idx, r.layer)) {
            Some(kv) => Ok(tonic::Response::new(kv)),
            None => Err(tonic::Status::not_found("token not in cold vault")),
        }
    }

    async fn batch_promote(
        &self,
        req: tonic::Request<BatchPromoteRequest>,
    ) -> Result<tonic::Response<BatchPromoteResponse>, tonic::Status> {
        let r = req.into_inner();
        let mut guard = self.store.write().await;
        let tensors = r.layers.iter().map(|&layer| {
            guard.remove(&(r.token_idx, layer)).unwrap_or_else(|| KvTensor {
                token_idx: r.token_idx,
                layer,
                data: vec![],
                rows: 0,
                cols: 0,
                dtype: "f32".into(),
            })
        }).collect();
        Ok(tonic::Response::new(BatchPromoteResponse { tensors }))
    }
}

pub fn cold_vault_server() -> ColdVaultServiceServer<ColdVaultServer> {
    ColdVaultServiceServer::new(ColdVaultServer::default())
        .max_encoding_message_size(512 * 1024 * 1024) // 512 MiB — batch responses can be >4 MB
        .max_decoding_message_size(512 * 1024 * 1024) // 512 MiB — INT8 Store requests can be >4 MB
}

// ── Recompute server stub ─────────────────────────────────────────────────────

use tierkv::{
    recompute_service_server::{RecomputeService, RecomputeServiceServer},
};

#[derive(Default)]
pub struct RecomputeServer;

#[tonic::async_trait]
impl RecomputeService for RecomputeServer {
    async fn recompute(
        &self,
        req: tonic::Request<RecomputeRequest>,
    ) -> Result<tonic::Response<KvTensor>, tonic::Status> {
        let r = req.into_inner();
        // TODO: hook into EXO's shard engine to recompute from weights.
        // For now returns empty tensor so hot-tier tests pass end-to-end.
        Ok(tonic::Response::new(KvTensor {
            token_idx: r.token_idx,
            layer: r.layer,
            data: vec![],
            rows: 0,
            cols: 0,
            dtype: "f32".into(),
        }))
    }
}

pub fn recompute_server() -> RecomputeServiceServer<RecomputeServer> {
    RecomputeServiceServer::new(RecomputeServer::default())
}

// ── vLLM Cold Vault client ───────────────────────────────────────────────────

pub struct VllmColdVaultClient {
    inner: VllmColdVaultServiceClient<Channel>,
}

impl VllmColdVaultClient {
    pub async fn connect(addr: &str) -> Result<Self, tonic::transport::Error> {
        let inner = VllmColdVaultServiceClient::connect(format!("http://{addr}"))
            .await?
            .max_decoding_message_size(512 * 1024 * 1024)
            .max_encoding_message_size(512 * 1024 * 1024);
        Ok(VllmColdVaultClient { inner })
    }

    pub async fn store_block(
        &mut self,
        block_hash_hex: String,
        layer_type: String,
        payload: Vec<u8>,
        is_quantized: bool,
        num_tokens: u32,
    ) -> Result<String, tonic::Status> {
        let req = tonic::Request::new(VllmBlockStoreRequest {
            block_hash_hex,
            layer_type,
            payload,
            is_quantized,
            num_tokens,
        });
        Ok(self.inner.store_block(req).await?.into_inner().vault_key)
    }

    pub async fn batch_promote(
        &mut self,
        vault_keys: Vec<String>,
    ) -> Result<Vec<tierkv::VllmBlockData>, tonic::Status> {
        let req = tonic::Request::new(VllmBatchPromoteRequest { vault_keys });
        Ok(self.inner.batch_promote(req).await?.into_inner().blocks)
    }
}

// ── vLLM Cold Vault server ───────────────────────────────────────────────────

use tierkv::{
    vllm_cold_vault_service_server::{VllmColdVaultService, VllmColdVaultServiceServer},
    VllmBlockData, VllmBlockStoreResponse, VllmBatchPromoteResponse,
};

/// In-memory vLLM block store. Key = vault_key (string), Value = (block_hash_hex, payload, is_quantized).
#[derive(Default)]
pub struct VllmColdVaultServer {
    store: Arc<RwLock<HashMap<String, (String, Vec<u8>, bool)>>>,
    counter: Arc<std::sync::atomic::AtomicU64>,
}

#[tonic::async_trait]
impl VllmColdVaultService for VllmColdVaultServer {
    async fn store_block(
        &self,
        req: tonic::Request<VllmBlockStoreRequest>,
    ) -> Result<tonic::Response<VllmBlockStoreResponse>, tonic::Status> {
        let r = req.into_inner();
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let vault_key = format!("vk-{}-{}", r.block_hash_hex, seq);
        self.store.write().await.insert(
            vault_key.clone(),
            (r.block_hash_hex, r.payload, r.is_quantized),
        );
        Ok(tonic::Response::new(VllmBlockStoreResponse { vault_key }))
    }

    async fn batch_promote(
        &self,
        req: tonic::Request<VllmBatchPromoteRequest>,
    ) -> Result<tonic::Response<VllmBatchPromoteResponse>, tonic::Status> {
        let r = req.into_inner();
        let guard = self.store.read().await;
        let blocks = r.vault_keys.iter().map(|key| {
            match guard.get(key) {
                Some((hash, payload, is_q)) => VllmBlockData {
                    block_hash_hex: hash.clone(),
                    payload: payload.clone(),
                    is_quantized: *is_q,
                },
                None => VllmBlockData {
                    block_hash_hex: String::new(),
                    payload: vec![],
                    is_quantized: false,
                },
            }
        }).collect();
        Ok(tonic::Response::new(VllmBatchPromoteResponse { blocks }))
    }
}

pub fn vllm_cold_vault_server() -> VllmColdVaultServiceServer<VllmColdVaultServer> {
    VllmColdVaultServiceServer::new(VllmColdVaultServer {
        store: Arc::new(RwLock::new(HashMap::new())),
        counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
    })
    .max_encoding_message_size(512 * 1024 * 1024)
    .max_decoding_message_size(512 * 1024 * 1024)
}
