//! INT8 KV cache quantization — per-tensor absmax scaling.
//!
//! Replaces the previous TurboQuant (Hadamard + NF4) which required a trained
//! codebook. INT8 absmax needs no training and gives ~42 dB SNR on typical
//! LLM KV activations vs effectively negative SNR with an untrained codebook.
//!
//! Wire format per encoded buffer:
//!   [scale: f32 LE (4 bytes)] [quantized: i8 as u8 (n bytes)]
//!
//! Compression: 4 bytes f32 → 1 byte i8 + 4/n bytes overhead ≈ 3.9× for large n.

use pyo3::prelude::*;

/// Python-visible quantizer.
///
/// `dim` sets the group size for per-group INT8 quantization. Each contiguous
/// `dim` floats form one group with its own absmax scale. This isolates outliers
/// to their group, preventing one large value from degrading all others.
///
/// Wire format: repeated groups of [scale: f32 LE (4 bytes)] [quantized: i8 as u8 (dim bytes)].
/// Compression: (dim × 4) / (dim + 4) ≈ 3.9× for dim=256.
#[pyclass]
pub struct TurboQuant {
    dim: usize,
}

#[pymethods]
impl TurboQuant {
    #[new]
    pub fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("dim must be > 0"));
        }
        Ok(TurboQuant { dim })
    }

    /// Encode flat f32 bytes → per-group INT8 compressed bytes.
    pub fn encode(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        let floats = bytes_to_f32(data)?;
        Ok(encode_grouped(&floats, self.dim))
    }

    /// Decode per-group INT8 compressed bytes → flat f32 bytes.
    pub fn decode(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        let floats = decode_grouped(data, self.dim)?;
        Ok(f32_to_bytes(&floats))
    }

    /// Approximate compression ratio (input bytes / output bytes).
    /// Per-group INT8: (dim × 4) / (dim + 4) ≈ 3.9× for dim=256.
    #[getter]
    pub fn ratio(&self) -> f32 {
        let orig = (self.dim * 4) as f32;
        let comp = (self.dim + 4) as f32;
        orig / comp
    }
}

impl TurboQuant {
    /// Encode without going through PyO3 (used internally by TieredKVCache).
    pub fn encode_floats(&self, floats: &[f32]) -> Vec<u8> {
        encode_grouped(floats, self.dim)
    }

    /// Decode without going through PyO3.
    pub fn decode_bytes(&self, data: &[u8]) -> PyResult<Vec<f32>> {
        decode_grouped(data, self.dim)
    }
}

// ── core encode / decode ─────────────────────────────────────────────────────

/// Encode one group of floats: [scale: f32 LE] [quantized: i8 as u8 × n].
fn encode_group(floats: &[f32]) -> Vec<u8> {
    let absmax = floats.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = (absmax / 127.0f32).max(1e-8);
    let mut out = Vec::with_capacity(4 + floats.len());
    out.extend_from_slice(&scale.to_le_bytes());
    for &f in floats {
        let q = (f / scale).round().clamp(-127.0, 127.0) as i8;
        out.push(q as u8);
    }
    out
}

/// Per-group INT8 encode. The input is split into chunks of `group_size` floats;
/// each chunk gets its own absmax scale. This prevents a single outlier from
/// degrading quantization resolution across the whole tensor.
fn encode_grouped(floats: &[f32], group_size: usize) -> Vec<u8> {
    let n_groups = floats.len().div_ceil(group_size);
    let mut out = Vec::with_capacity(n_groups * (4 + group_size));
    for chunk in floats.chunks(group_size) {
        out.extend_from_slice(&encode_group(chunk));
    }
    out
}

/// Decode per-group INT8 bytes back to floats.
/// Each group is [scale: f32 LE (4 bytes)] [quantized: i8 as u8 (group_size bytes)].
fn decode_grouped(data: &[u8], group_size: usize) -> PyResult<Vec<f32>> {
    if data.len() < 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "INT8 encoded data must be at least 4 bytes (scale header)",
        ));
    }
    let stride = 4 + group_size; // bytes per group
    if data.len() % stride != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "encoded data length {} is not a multiple of group stride {} (4 + {})",
            data.len(), stride, group_size
        )));
    }
    let n_groups = data.len() / stride;
    let mut floats = Vec::with_capacity(n_groups * group_size);
    for chunk in data.chunks_exact(stride) {
        let scale = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
        for &b in &chunk[4..] {
            floats.push((b as i8) as f32 * scale);
        }
    }
    Ok(floats)
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn bytes_to_f32(data: &[u8]) -> PyResult<Vec<f32>> {
    if data.len() % 4 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "data length is not a multiple of 4",
        ));
    }
    Ok(data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn f32_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn snr_db(original: &[f32], reconstructed: &[f32]) -> f32 {
        let signal: f32 = original.iter().map(|x| x * x).sum();
        let noise: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        10.0 * (signal / noise.max(1e-12)).log10()
    }

    /// Deterministic stand-in for Gaussian KV activations.
    /// Linear ramp with alternating sign — exercises full [-absmax, +absmax] range.
    fn synthetic_kv(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = (i as f32 / n as f32) * 2.0 - 1.0; // -1..+1
                t * 0.1 // typical KV activation scale
            })
            .collect()
    }

    #[test]
    fn round_trip_quality() {
        let dim = 256;
        let n = 25_600; // [1, 2, 50, 256] KV shape
        let original = synthetic_kv(n);

        let tq = TurboQuant::new(dim).unwrap();
        let encoded = tq.encode_floats(&original);
        let decoded = tq.decode_bytes(&encoded).unwrap();

        assert_eq!(decoded.len(), n);

        let snr = snr_db(&original, &decoded);
        println!(
            "INT8 round-trip SNR: {snr:.1} dB  ({} → {} bytes, {:.1}×)",
            n * 4,
            encoded.len(),
            (n * 4) as f32 / encoded.len() as f32
        );

        // INT8 absmax: SNR ≥ 40 dB for uniform distributions (6 dB/bit × 8 bits).
        assert!(snr >= 40.0, "SNR {snr:.1} dB below 40 dB threshold");
    }

    #[test]
    fn round_trip_zeros() {
        let tq = TurboQuant::new(256).unwrap();
        let zeros = vec![0.0f32; 256];
        let encoded = tq.encode_floats(&zeros);
        let decoded = tq.decode_bytes(&encoded).unwrap();
        // All zeros should round-trip exactly (scale clamps to 1e-8, quantized = 0).
        assert!(decoded.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn compression_ratio() {
        let tq = TurboQuant::new(256).unwrap();
        // (256 × 4) / (256 + 4) = 1024 / 260 ≈ 3.938
        assert!((tq.ratio() - 3.938).abs() < 0.01, "ratio = {}", tq.ratio());
    }

    /// Per-group quantization isolates outliers: one group with a 100× outlier
    /// should not degrade other groups (whole-tensor quantization would fail this).
    #[test]
    fn outlier_isolation() {
        let dim = 256;
        let n_groups = 100;
        let n = dim * n_groups;
        let mut original = synthetic_kv(n); // values in [-0.1, 0.1]
        // Inject a single large outlier in group 0
        original[0] = 100.0;

        let tq = TurboQuant::new(dim).unwrap();
        let encoded = tq.encode_floats(&original);
        let decoded = tq.decode_bytes(&encoded).unwrap();

        // Groups 1..n_groups should have high SNR (outlier confined to group 0)
        let rest_orig = &original[dim..];
        let rest_dec = &decoded[dim..];
        let snr = snr_db(rest_orig, rest_dec);
        assert!(snr >= 40.0, "SNR for non-outlier groups = {snr:.1} dB (should be ≥ 40)");
    }

    #[test]
    fn rejects_zero_dim() {
        assert!(TurboQuant::new(0).is_err());
        assert!(TurboQuant::new(1).is_ok());
        assert!(TurboQuant::new(256).is_ok());
    }

    #[test]
    fn decode_rejects_short_buffer() {
        let tq = TurboQuant::new(256).unwrap();
        assert!(tq.decode_bytes(&[0u8; 3]).is_err());
        assert!(tq.decode_bytes(&[0u8; 4]).is_err()); // 4 bytes not a full group (need 260)
        assert!(tq.decode_bytes(&[0u8; 260]).is_ok()); // one full group (4 scale + 256 values)
    }
}
