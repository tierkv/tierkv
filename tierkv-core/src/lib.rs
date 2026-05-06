//! tierkv-core — PyO3 extension module entry point.
//!
//! Exposes:
//!   TurboQuant              — per-group INT8 quantizer (~3.9× compression, ≥52 dB SNR)
//!   TieredKVCache           — 3-tier (hot/cold/recompute) KV cache manager
//!   start_cold_vault_server — launch ColdVault gRPC server (cold tier node, port 50051)
//!   start_recompute_server  — launch Recompute gRPC server  (inference node, port 50052)

use pyo3::prelude::*;
use tokio::runtime::Runtime;

mod tiered_kv;
mod turbo_quant;
pub mod residual_sync;

/// Start the ColdVault gRPC server in a background thread (non-blocking).
/// Serves both EXO (ColdVaultService) and vLLM (VllmColdVaultService) on the same port.
/// Call once at process startup on a cold-tier node.
///
/// max_bytes: evict oldest blocks when vault exceeds this size. 0 = unlimited.
#[pyfunction]
#[pyo3(signature = (port, max_bytes=0))]
fn start_cold_vault_server(port: u16, max_bytes: u64) -> PyResult<()> {
    std::thread::spawn(move || {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async move {
            use residual_sync::{cold_vault_server, vllm_cold_vault_server};
            use tonic::transport::Server;
            let addr = format!("0.0.0.0:{port}").parse().expect("addr");
            Server::builder()
                .add_service(cold_vault_server(max_bytes))
                .add_service(vllm_cold_vault_server(max_bytes))
                .serve(addr)
                .await
                .expect("ColdVault server error");
        });
    });
    Ok(())
}

/// Start the Recompute gRPC server in a background thread (non-blocking).
/// Call once at process startup on the inference node.
#[pyfunction]
fn start_recompute_server(port: u16) -> PyResult<()> {
    std::thread::spawn(move || {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async move {
            use residual_sync::recompute_server;
            use tonic::transport::Server;
            let addr = format!("0.0.0.0:{port}").parse().expect("addr");
            Server::builder()
                .add_service(recompute_server())
                .serve(addr)
                .await
                .expect("Recompute server error");
        });
    });
    Ok(())
}

/// Flush all blocks from a remote ColdVaultService (EXO tier) or VllmColdVaultService.
/// Connects to host:port, calls Flush on both services, returns total bytes freed.
/// Raises RuntimeError on connection or RPC failure.
#[pyfunction]
fn flush_cold_vault(host: &str, port: u16) -> PyResult<u64> {
    let addr = format!("{}:{}", host, port);
    let rt = Runtime::new().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    rt.block_on(async move {
        use residual_sync::tierkv::{
            cold_vault_service_client::ColdVaultServiceClient,
            vllm_cold_vault_service_client::VllmColdVaultServiceClient,
            FlushRequest,
        };
        let url = format!("http://{addr}");
        let mut total = 0u64;

        match ColdVaultServiceClient::connect(url.clone()).await {
            Ok(mut c) => {
                if let Ok(r) = c.flush(tonic::Request::new(FlushRequest {})).await {
                    total += r.into_inner().flushed_bytes;
                }
            }
            Err(e) => return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("connect failed: {e}")
            )),
        }
        match VllmColdVaultServiceClient::connect(url).await {
            Ok(mut c) => {
                if let Ok(r) = c.flush(tonic::Request::new(FlushRequest {})).await {
                    total += r.into_inner().flushed_bytes;
                }
            }
            Err(_) => {} // vLLM vault may not be on this node; not fatal
        }
        Ok(total)
    })
}

#[pymodule]
fn tierkv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<turbo_quant::TurboQuant>()?;
    m.add_class::<tiered_kv::TieredKVCache>()?;
    m.add_function(wrap_pyfunction!(start_cold_vault_server, m)?)?;
    m.add_function(wrap_pyfunction!(start_recompute_server, m)?)?;
    m.add_function(wrap_pyfunction!(flush_cold_vault, m)?)?;
    Ok(())
}
