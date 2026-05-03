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
    BatchPromoteRequest, KvTensor, PromoteRequest, RecomputeRequest, StoreRequest,
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
