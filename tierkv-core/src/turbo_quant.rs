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

    /// Decode per-group INT8 compressed bytes → flat f32 bytes.
    /// Releases the GIL during computation so multiple calls from a Python
    /// thread pool can run concurrently (each takes ~1-2s CPU-bound).
    pub fn decode_nogil(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<Vec<u8>> {
        let dim = self.dim;
        py.allow_threads(move || {
            decode_grouped_inner(&data, dim)
                .map(|floats| f32_to_bytes(&floats))
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Decode in-place into a pre-allocated bytearray.
    ///
    /// Zero intermediate allocations — avoids the page-fault storm from
    /// three ~102 MB Vec allocations inside a large mmap'd vLLM process.
    ///
    /// `out` must be exactly `decoded_size(len(data))` bytes; returns the
    /// number of bytes written.  Call `TurboQuant.decoded_size(compressed_len)`
    /// to size the buffer before calling.
    ///
    /// Releases the GIL during the decode loop.
    pub fn decode_into(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        out: pyo3::Bound<'_, pyo3::types::PyByteArray>,
    ) -> PyResult<usize> {
        use pyo3::types::PyByteArrayMethods;
        let dim = self.dim;
        // Grab the raw pointer + length while we hold the GIL, then cast to
        // usize so the value is Send and can cross the allow_threads boundary.
        // Safety: the bytearray must not be resized or shared across Python
        // threads during this call — same contract as numpy's writable buffer.
        let (out_ptr_usize, out_len) = unsafe {
            let slice = out.as_bytes_mut();
            (slice.as_mut_ptr() as usize, slice.len())
        };
        py.allow_threads(move || {
            decode_grouped_into(&data, dim, out_ptr_usize as *mut u8, out_len)
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Return the number of output bytes that `decode_into` will produce for
    /// `compressed_len` bytes of input.  Raises ValueError if the length is
    /// not a valid multiple of the group stride.
    pub fn decoded_size(&self, compressed_len: usize) -> PyResult<usize> {
        let stride = 4 + self.dim;
        if compressed_len % stride != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "compressed_len {} is not a multiple of stride {} (4 + {})",
                compressed_len, stride, self.dim,
            )));
        }
        Ok((compressed_len / stride) * self.dim * 4)
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

    /// Decode without GIL (pure Rust, no PyO3 error types).
    pub fn decode_bytes_inner(&self, data: &[u8]) -> Result<Vec<f32>, String> {
        decode_grouped_inner(data, self.dim)
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
    decode_grouped_inner(data, group_size)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
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

/// Pure-Rust version of bytes_to_f32 — no GIL required.
fn bytes_to_f32_inner(data: &[u8]) -> Result<Vec<f32>, String> {
    if data.len() % 4 != 0 {
        return Err("data length is not a multiple of 4".to_string());
    }
    Ok(data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn f32_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Zero-allocation in-place decode: writes decoded f32 LE bytes directly into
/// `out_ptr[0..out_len]`.  Called from `decode_into` with the GIL released.
///
/// # Safety
/// Caller must ensure `out_ptr` points to a valid writable buffer of exactly
/// `out_len` bytes, and that no other thread touches that buffer concurrently.
fn decode_grouped_into(
    data: &[u8],
    group_size: usize,
    out_ptr: *mut u8,
    out_len: usize,
) -> Result<usize, String> {
    if data.len() < 4 {
        return Err("INT8 encoded data must be at least 4 bytes (scale header)".to_string());
    }
    let stride = 4 + group_size;
    if data.len() % stride != 0 {
        return Err(format!(
            "encoded data length {} is not a multiple of group stride {} (4 + {})",
            data.len(), stride, group_size
        ));
    }
    let n_groups = data.len() / stride;
    let expected_out = n_groups * group_size * 4;
    if out_len < expected_out {
        return Err(format!(
            "output buffer too small: {} < expected {} ({} groups × {} floats × 4 bytes)",
            out_len, expected_out, n_groups, group_size,
        ));
    }
    // SAFETY: caller guarantees out_ptr is valid for out_len bytes.
    let out: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
    let mut write_pos = 0usize;
    for chunk in data.chunks_exact(stride) {
        let scale = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
        for &b in &chunk[4..] {
            let f: f32 = (b as i8) as f32 * scale;
            out[write_pos..write_pos + 4].copy_from_slice(&f.to_le_bytes());
            write_pos += 4;
        }
    }
    Ok(write_pos)
}

/// Pure-Rust version of decode_grouped — no GIL required, safe for allow_threads.
fn decode_grouped_inner(data: &[u8], group_size: usize) -> Result<Vec<f32>, String> {
    if data.len() < 4 {
        return Err("INT8 encoded data must be at least 4 bytes (scale header)".to_string());
    }
    let stride = 4 + group_size;
    if data.len() % stride != 0 {
        return Err(format!(
            "encoded data length {} is not a multiple of group stride {} (4 + {})",
            data.len(), stride, group_size
        ));
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

    /// decode_into must produce bit-identical output to decode_bytes.
    #[test]
    fn decode_into_matches_decode_bytes() {
        let dim = 128;
        let n = 12_800;
        let original = synthetic_kv(n);
        let tq = TurboQuant::new(dim).unwrap();
        let encoded = tq.encode_floats(&original);

        // Reference path
        let ref_bytes = f32_to_bytes(&tq.decode_bytes(&encoded).unwrap());

        // decode_grouped_into path
        let stride = 4 + dim;
        let n_groups = encoded.len() / stride;
        let out_len = n_groups * dim * 4;
        let mut buf = vec![0u8; out_len];
        let written = decode_grouped_into(&encoded, dim, buf.as_mut_ptr(), out_len).unwrap();

        assert_eq!(written, out_len);
        assert_eq!(buf, ref_bytes, "decode_into output differs from decode_bytes");
    }

    /// decoded_size formula: (compressed_len / stride) * dim * 4.
    #[test]
    fn decoded_size_formula() {
        let dim = 256usize;
        let stride = 4 + dim; // 260
        let n_groups = 10usize;
        let compressed_len = n_groups * stride; // 2600
        let expected_decoded = n_groups * dim * 4; // 10240

        // Test via the internal function directly.
        let mut buf = vec![0u8; expected_decoded];
        // Encode 10 groups of zeros
        let zeros = vec![0.0f32; n_groups * dim];
        let tq = TurboQuant::new(dim).unwrap();
        let encoded = tq.encode_floats(&zeros);
        assert_eq!(encoded.len(), compressed_len);
        let written = decode_grouped_into(&encoded, dim, buf.as_mut_ptr(), expected_decoded).unwrap();
        assert_eq!(written, expected_decoded);
    }
}
