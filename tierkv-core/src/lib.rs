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
/// Call once at process startup on a cold-tier node.
#[pyfunction]
fn start_cold_vault_server(port: u16) -> PyResult<()> {
    std::thread::spawn(move || {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async move {
            use residual_sync::cold_vault_server;
            use tonic::transport::Server;
            let addr = format!("0.0.0.0:{port}").parse().expect("addr");
            Server::builder()
                .add_service(cold_vault_server())
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

#[pymodule]
fn tierkv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<turbo_quant::TurboQuant>()?;
    m.add_class::<tiered_kv::TieredKVCache>()?;
    m.add_function(wrap_pyfunction!(start_cold_vault_server, m)?)?;
    m.add_function(wrap_pyfunction!(start_recompute_server, m)?)?;
    Ok(())
}
