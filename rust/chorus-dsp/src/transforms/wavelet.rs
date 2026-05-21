use std::collections::VecDeque;
use num_complex::Complex;

use crate::estimation::ComplexEstimator;
use crate::filters::{FilterChains, FilterSpec};
use crate::prototypes::{center_prototype_real, surround_prototype_real};
use crate::transforms::Transform;

// Daubechies-4 (db4) decomposition filters (8 taps)
const DB4_LO: [f64; 8] = [
    -0.010597401784997278,
     0.032883011666982945,
     0.030841381835986965,
    -0.18703481171888114,
    -0.027983769416983849,
     0.63088076792959036,
     0.71484657055254153,
     0.23037781330885523,
];
const DB4_HI: [f64; 8] = [
    -0.23037781330885523,
     0.71484657055254153,
    -0.63088076792959036,
    -0.027983769416983849,
     0.18703481171888114,
     0.030841381835986965,
    -0.032883011666982945,
    -0.010597401784997278,
];

// Reconstruction (synthesis) filters: dec_lo / dec_hi reversed
const REC_LO: [f64; 8] = [
     0.23037781330885523,
     0.71484657055254153,
     0.63088076792959036,
    -0.027983769416983849,
    -0.18703481171888114,
     0.030841381835986965,
     0.032883011666982945,
    -0.010597401784997278,
];
const REC_HI: [f64; 8] = [
    -0.010597401784997278,
    -0.032883011666982945,
     0.030841381835986965,
     0.18703481171888114,
    -0.027983769416983849,
    -0.63088076792959036,
     0.71484657055254153,
    -0.23037781330885523,
];

const FILTER_LEN: usize = 8;

/// Decompose `signal` with `filter` using PyWavelets-compatible periodization,
/// then downsample by 2.
///
/// Formula: `y[oi] = sum_k: filter[k] * signal[(2*oi - k + FILTER_LEN/2) mod N]`
///
/// Output length = ceil(N / 2)
fn conv_downsample(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let f_len = filter.len();
    let offset = f_len / 2;
    // output length = ceil(n/2)
    let out_len = (n + 1) / 2;
    let mut out = vec![0.0; out_len];
    for (oi, out_sample) in out.iter_mut().enumerate() {
        let base = 2 * oi;
        let mut acc = 0.0;
        for (k, &fv) in filter.iter().enumerate() {
            // wrapping_sub to handle base - k potentially underflowing usize
            let raw = base + offset + n - k; // always positive in usize domain
            let idx = raw % n;
            acc += fv * signal[idx];
        }
        *out_sample = acc;
    }
    out
}

/// Reconstruct from `approx` and `detail` coefficient arrays using PyWavelets-compatible
/// periodization.
///
/// Python reference formula for output sample `i`:
///   `for k in 0..f_len:`
///     `shifted = i - k + f_len/2 - 1`
///     `if shifted % 2 == 0:`
///       `cidx = (shifted // 2) % m`
///       `acc += REC_LO[k] * approx[cidx] + REC_HI[k] * detail[cidx]`
///
/// where `m = approx.len() = detail.len()`.
fn upsample_conv_add(approx: &[f64], detail: &[f64], out_len: usize) -> Vec<f64> {
    let f_len = FILTER_LEN;
    let m = approx.len(); // = detail.len()
    // offset_r = f_len/2 - 1 = 3
    // (i - k + offset_r) can be negative: min = 0 - (f_len-1) + offset_r = 0 - 7 + 3 = -4
    // Add a large even multiple of m*2 to keep it positive: (f_len + 1) * m * 2
    // This is always even and divisible by m when divided by m after /2.
    let bias = (f_len + 1) * 2 * m; // even, >> 4
    let offset_r = f_len / 2 - 1;
    let mut out = vec![0.0; out_len];
    for i in 0..out_len {
        let mut acc = 0.0;
        for k in 0..f_len {
            // shifted = i + offset_r - k, biased upward to keep positive
            let shifted = i + offset_r + bias - k; // always positive
            if shifted % 2 == 0 {
                let cidx = (shifted / 2) % m;
                acc += REC_LO[k] * approx[cidx];
                acc += REC_HI[k] * detail[cidx];
            }
        }
        out[i] = acc;
    }
    out
}

/// Compute the number of wavelet coefficients produced by wavedec with periodic
/// boundary at the given level for input of length `n`.
/// For each level: new_len = ceil(old_len / 2)
pub fn wavedec_num_coeffs(n: usize, level: usize) -> usize {
    let mut total = 0usize;
    let mut cur = n;
    for _ in 0..level {
        cur = (cur + 1) / 2;
        total += cur;
    }
    total += cur; // final approximation
    total
}

/// Multi-level Daubechies-4 wavelet decomposition (PyWavelets periodization mode).
/// Returns coefficients: `[cA_level, cD_level, cD_{level-1}, ..., cD_1]`
pub fn wavedec(signal: &[f64], level: usize) -> Vec<Vec<f64>> {
    let mut result: Vec<Vec<f64>> = Vec::with_capacity(level + 1);
    let mut approx = signal.to_vec();
    for _ in 0..level {
        let detail = conv_downsample(&approx, &DB4_HI);
        let new_approx = conv_downsample(&approx, &DB4_LO);
        result.push(detail);
        approx = new_approx;
    }
    // result[0] = cD_1, result[1] = cD_2, ..., result[level-1] = cD_level
    // Reverse so: result[0] = cD_level, ..., result[level-1] = cD_1
    result.reverse();
    result.insert(0, approx); // result[0] = cA_level
    result
}

/// Multi-level Daubechies-4 wavelet reconstruction (PyWavelets periodization mode).
/// `coeffs` is `[cA_level, cD_level, cD_{level-1}, ..., cD_1]`
/// Returns reconstructed signal of length `out_len`.
pub fn waverec(coeffs: &[Vec<f64>], out_len: usize) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0; out_len];
    }
    let level = coeffs.len() - 1;
    if level == 0 {
        let mut r = coeffs[0].clone();
        r.truncate(out_len);
        while r.len() < out_len { r.push(0.0); }
        return r;
    }

    // Band lengths from decomposition
    // coeffs: [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    // At each reconstruction step we go from coarse to fine.
    // Level k reconstruction target length = coeffs[level - k + 1].len() * 2
    // at finest level = out_len

    let mut approx = coeffs[0].clone();

    for lev in 0..level {
        let detail = &coeffs[lev + 1];
        // Target length for this step:
        // - At the finest level (last iteration), target is out_len
        // - Otherwise, target is the length of the next detail band (coeffs[lev+2])
        //   because that band has the same length as the next approximation band
        let target_len = if lev == level - 1 {
            out_len
        } else {
            coeffs[lev + 2].len()
        };
        approx = upsample_conv_add(&approx, detail, target_len);
    }

    approx.truncate(out_len);
    while approx.len() < out_len {
        approx.push(0.0);
    }
    approx
}

/// Compute boundary contamination at a given level.
/// = (filter_len - 1) * (2^level - 1)
fn boundary_contamination(level: usize) -> usize {
    (FILTER_LEN - 1) * ((1 << level) - 1)
}

pub struct StreamingWavelet {
    level: usize,
    frame_size: usize,
    overlap: usize,           // = 2 * boundary_contamination
    input_buf: VecDeque<[f64; 2]>,
    output_buf: VecDeque<[f64; 2]>,
    lc_est: ComplexEstimator,
    rc_est: ComplexEstimator,
    ls_est: ComplexEstimator,
    rs_est: ComplexEstimator,
    filter_chains: FilterChains,
}

impl StreamingWavelet {
    pub fn new(
        level: usize,
        frame_size: usize,
        smoothing_alpha: f64,
        epsilon: f64,
        filter_chains: FilterChains,
    ) -> Self {
        let bc = boundary_contamination(level);
        let overlap = 2 * bc;
        let block_size = frame_size + overlap;
        // Compute actual number of wavelet coefficients for the block size
        let num_coeffs = wavedec_num_coeffs(block_size, level);

        let mut input_buf: VecDeque<[f64; 2]> = VecDeque::new();
        // Pre-load overlap zeros
        for _ in 0..overlap {
            input_buf.push_back([0.0, 0.0]);
        }

        Self {
            level,
            frame_size,
            overlap,
            input_buf,
            output_buf: VecDeque::new(),
            lc_est: ComplexEstimator::new(smoothing_alpha, epsilon, num_coeffs),
            rc_est: ComplexEstimator::new(smoothing_alpha, epsilon, num_coeffs),
            ls_est: ComplexEstimator::new(smoothing_alpha, epsilon, num_coeffs),
            rs_est: ComplexEstimator::new(smoothing_alpha, epsilon, num_coeffs),
            filter_chains,
        }
    }

    fn process_block_internal(&mut self) {
        let block_size = self.frame_size + self.overlap;
        let bc = self.overlap / 2;

        // Extract block
        let block: Vec<[f64; 2]> = self.input_buf.iter().take(block_size).copied().collect();

        let left_signal: Vec<f64> = block.iter().map(|s| s[0]).collect();
        let right_signal: Vec<f64> = block.iter().map(|s| s[1]).collect();

        // Wavelet decompose each channel
        let left_coeffs = wavedec(&left_signal, self.level);
        let right_coeffs = wavedec(&right_signal, self.level);

        // Flatten coefficients
        let left_flat: Vec<f64> = left_coeffs.iter().flat_map(|v| v.iter().copied()).collect();
        let right_flat: Vec<f64> = right_coeffs.iter().flat_map(|v| v.iter().copied()).collect();

        // Compute real-valued prototypes
        let center_proto_real = center_prototype_real(&left_flat, &right_flat);
        let surround_proto_real = surround_prototype_real(&left_flat, &right_flat);

        // Wrap as complex for ComplexEstimator
        let left_complex: Vec<Complex<f64>> = left_flat.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let right_complex: Vec<Complex<f64>> = right_flat.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let center_proto: Vec<Complex<f64>> = center_proto_real.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let surround_proto: Vec<Complex<f64>> = surround_proto_real.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Run estimators
        let lc_complex = self.lc_est.estimate(&center_proto, &left_complex).to_vec();
        let rc_complex = self.rc_est.estimate(&center_proto, &right_complex).to_vec();
        let ls_complex = self.ls_est.estimate(&surround_proto, &left_complex).to_vec();
        let rs_complex = self.rs_est.estimate(&surround_proto, &right_complex).to_vec();

        // Convert back to real
        let lc_flat: Vec<f64> = lc_complex.iter().map(|c| c.re).collect();
        let rc_flat: Vec<f64> = rc_complex.iter().map(|c| c.re).collect();
        let ls_flat: Vec<f64> = ls_complex.iter().map(|c| c.re).collect();
        let rs_flat: Vec<f64> = rs_complex.iter().map(|c| c.re).collect();

        // Residuals
        let lo_flat: Vec<f64> = left_flat.iter().zip(lc_flat.iter()).zip(ls_flat.iter())
            .map(|((l, lc), ls)| l - lc - ls).collect();
        let ro_flat: Vec<f64> = right_flat.iter().zip(rc_flat.iter()).zip(rs_flat.iter())
            .map(|((r, rc), rs)| r - rc - rs).collect();

        // Filter chain scalars
        let soloed: Vec<String> = self.filter_chains
            .iter()
            .filter(|(_, chain)| chain.iter().any(|spec| matches!(spec, FilterSpec::Solo)))
            .map(|(name, _)| name.clone())
            .collect();

        let sc_lc = crate::filters::chain_scalar(&self.filter_chains, "Lc", &soloed);
        let sc_rc = crate::filters::chain_scalar(&self.filter_chains, "Rc", &soloed);
        let sc_lo = crate::filters::chain_scalar(&self.filter_chains, "Lo", &soloed);
        let sc_ro = crate::filters::chain_scalar(&self.filter_chains, "Ro", &soloed);
        let sc_ls = crate::filters::chain_scalar(&self.filter_chains, "Ls", &soloed);
        let sc_rs = crate::filters::chain_scalar(&self.filter_chains, "Rs", &soloed);

        let apply_scalar = |v: &[f64], s: f64| -> Vec<f64> { v.iter().map(|x| x * s).collect() };

        let lc_scaled = apply_scalar(&lc_flat, sc_lc);
        let rc_scaled = apply_scalar(&rc_flat, sc_rc);
        let lo_scaled = apply_scalar(&lo_flat, sc_lo);
        let ro_scaled = apply_scalar(&ro_flat, sc_ro);
        let ls_scaled = apply_scalar(&ls_flat, sc_ls);
        let rs_scaled = apply_scalar(&rs_flat, sc_rs);

        // Combine contributions per channel
        let left_out_flat: Vec<f64> = lc_scaled.iter().zip(lo_scaled.iter()).zip(ls_scaled.iter())
            .map(|((lc, lo), ls)| lc + lo + ls).collect();
        let right_out_flat: Vec<f64> = rc_scaled.iter().zip(ro_scaled.iter()).zip(rs_scaled.iter())
            .map(|((rc, ro), rs)| rc + ro + rs).collect();

        // Reshape flat coefficients back into band structure
        let band_lens: Vec<usize> = left_coeffs.iter().map(|v| v.len()).collect();
        let left_bands = unflatten_coeffs(&left_out_flat, &band_lens);
        let right_bands = unflatten_coeffs(&right_out_flat, &band_lens);

        // Reconstruct time-domain signals
        let left_recon = waverec(&left_bands, block_size);
        let right_recon = waverec(&right_bands, block_size);

        // Discard boundary contamination (first bc and last bc samples)
        let valid_start = bc;
        let valid_end = block_size - bc;

        for i in valid_start..valid_end {
            self.output_buf.push_back([left_recon[i], right_recon[i]]);
        }

        // Advance input_buf by frame_size
        for _ in 0..self.frame_size {
            self.input_buf.pop_front();
        }
    }
}

/// Reshape a flat coefficient vector back into bands using the given band lengths.
fn unflatten_coeffs(flat: &[f64], band_lens: &[usize]) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(band_lens.len());
    let mut offset = 0;
    for &len in band_lens {
        out.push(flat[offset..offset + len].to_vec());
        offset += len;
    }
    out
}

impl Transform for StreamingWavelet {
    fn process_block(&mut self, input: &[[f64; 2]]) -> Vec<[f64; 2]> {
        for &sample in input {
            self.input_buf.push_back(sample);
        }

        let block_size = self.frame_size + self.overlap;
        while self.input_buf.len() >= block_size {
            self.process_block_internal();
        }

        self.output_buf.drain(..).collect()
    }

    fn reset(&mut self) {
        self.input_buf.clear();
        self.output_buf.clear();
        // Pre-load overlap zeros
        for _ in 0..self.overlap {
            self.input_buf.push_back([0.0, 0.0]);
        }
        self.lc_est.reset();
        self.rc_est.reset();
        self.ls_est.reset();
        self.rs_est.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{StreamingWavelet, wavedec, waverec};
    use crate::filters::unity_chains;
    use crate::transforms::Transform;

    const LEVEL: usize = 3;
    const FRAME_SIZE: usize = 512;

    fn make_wavelet() -> StreamingWavelet {
        StreamingWavelet::new(LEVEL, FRAME_SIZE, 0.0, 1e-12, unity_chains())
    }

    fn generate_sine(num_samples: usize, freq_hz: f64, sample_rate: f64) -> Vec<[f64; 2]> {
        (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let v = (2.0 * std::f64::consts::PI * freq_hz * t).sin();
                [v, v]
            })
            .collect()
    }

    /// Verify wavedec/waverec is a perfect round-trip for a pure sine
    #[test]
    fn wavedec_waverec_round_trip() {
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / n as f64).sin())
            .collect();
        let coeffs = wavedec(&signal, LEVEL);
        let reconstructed = waverec(&coeffs, n);
        assert_eq!(reconstructed.len(), n);
        for (i, (&r, &orig)) in reconstructed.iter().zip(signal.iter()).enumerate() {
            assert!(
                (r - orig).abs() < 1e-6,
                "wavedec/waverec round-trip mismatch at {i}: got {r}, expected {orig}"
            );
        }
    }

    /// Test 1: Sine wave round-trip with unity bypass
    #[test]
    fn sine_wave_round_trip() {
        let mut wav = make_wavelet();
        let sample_rate = 48000.0;
        let num_samples = 4096;
        let signal = generate_sine(num_samples, 440.0, sample_rate);

        let mut all_output: Vec<[f64; 2]> = Vec::new();
        for chunk in signal.chunks(FRAME_SIZE) {
            let out = wav.process_block(chunk);
            all_output.extend_from_slice(&out);
        }
        // Flush with silence
        let silence = vec![[0.0_f64; 2]; FRAME_SIZE * 4];
        let out = wav.process_block(&silence);
        all_output.extend_from_slice(&out);

        // Overlap-save alignment: the first bc=49 output samples are zeros (startup transient
        // from the pre-loaded zero overlap). After that, output[bc+i] ≈ signal[i].
        // Skip the startup transient and compare with corresponding input.
        let bc = 49; // boundary_contamination(LEVEL) = 7 * 7 = 49
        // Compare output[bc..bc+compare_len] vs signal[0..compare_len]
        let compare_len = num_samples - 2 * FRAME_SIZE;

        assert!(
            all_output.len() >= bc + compare_len,
            "not enough output samples: got {}, need {}",
            all_output.len(),
            bc + compare_len
        );

        for i in 0..compare_len {
            let out_sample = all_output[bc + i];
            let in_sample = signal[i];
            assert!(
                (out_sample[0] - in_sample[0]).abs() < 1e-3,
                "left channel mismatch at sample {}: out={} vs in={}",
                i, out_sample[0], in_sample[0]
            );
            assert!(
                (out_sample[1] - in_sample[1]).abs() < 1e-3,
                "right channel mismatch at sample {}: out={} vs in={}",
                i, out_sample[1], in_sample[1]
            );
        }
    }

    fn run_wav_round_trip(path: &str) {
        let mut reader = hound::WavReader::open(path)
            .unwrap_or_else(|e| panic!("could not open {path}: {e}"));
        let spec = reader.spec();
        let num_channels = spec.channels as usize;
        assert!(num_channels <= 2, "expected mono or stereo WAV");

        let scale = match spec.sample_format {
            hound::SampleFormat::Float => 1.0_f64,
            hound::SampleFormat::Int => {
                1.0 / (1_i64
                    .checked_shl(spec.bits_per_sample as u32 - 1)
                    .unwrap_or(1) as f64)
            }
        };

        let raw_samples: Vec<f64> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| s.expect("read error") as f64)
                .collect(),
            hound::SampleFormat::Int => reader
                .samples::<i32>()
                .map(|s| s.expect("read error") as f64 * scale)
                .collect(),
        };

        let signal: Vec<[f64; 2]> = if num_channels == 2 {
            raw_samples.chunks(2).map(|c| [c[0], c[1]]).collect()
        } else {
            raw_samples.iter().map(|&s| [s, s]).collect()
        };

        let num_samples = signal.len();
        let mut wav = make_wavelet();
        let mut all_output: Vec<[f64; 2]> = Vec::new();

        for chunk in signal.chunks(FRAME_SIZE) {
            let out = wav.process_block(chunk);
            all_output.extend_from_slice(&out);
        }
        let silence = vec![[0.0_f64; 2]; FRAME_SIZE * 4];
        let out = wav.process_block(&silence);
        all_output.extend_from_slice(&out);

        // Overlap-save alignment: output[bc+i] ≈ signal[i] (same 49-sample startup offset).
        let bc = 49; // boundary_contamination(LEVEL) = 7 * 7 = 49
        let compare_len = num_samples.saturating_sub(2 * FRAME_SIZE);

        assert!(
            all_output.len() >= bc + compare_len,
            "not enough output samples: got {}, need {}",
            all_output.len(),
            bc + compare_len
        );

        for i in 0..compare_len {
            let out_sample = all_output[bc + i];
            let in_sample = signal[i];
            assert!(
                (out_sample[0] - in_sample[0]).abs() < 1e-3,
                "left channel WAV mismatch at sample {}: out={}, expected={}",
                i, out_sample[0], in_sample[0]
            );
            assert!(
                (out_sample[1] - in_sample[1]).abs() < 1e-3,
                "right channel WAV mismatch at sample {}: out={}, expected={}",
                i, out_sample[1], in_sample[1]
            );
        }
    }

    /// Test 2a: WAV round-trip (PinkPanther)
    #[test]
    fn wav_round_trip_pinkpanther() {
        run_wav_round_trip("../../tests/test_tracks_wav/PinkPanther.wav");
    }

    /// Test 2b: WAV round-trip (TVSong)
    #[test]
    fn wav_round_trip_tvsong() {
        run_wav_round_trip("../../tests/test_tracks_wav/TVSong.wav");
    }

    /// Test 3: Output length is a multiple of frame_size
    #[test]
    fn output_length_matches_full_frames() {
        let mut wav = make_wavelet();
        let num_samples = FRAME_SIZE * 8;
        let signal = generate_sine(num_samples, 440.0, 48000.0);

        let mut total_out = 0usize;
        for chunk in signal.chunks(FRAME_SIZE) {
            let out = wav.process_block(chunk);
            total_out += out.len();
        }

        assert_eq!(
            total_out % FRAME_SIZE,
            0,
            "output length {} is not a multiple of frame_size {}",
            total_out,
            FRAME_SIZE
        );
    }

    /// Test 4: Streaming consistency — 256-sample vs 512-sample chunks give same output
    #[test]
    fn streaming_consistency() {
        let sample_rate = 48000.0;
        let num_samples = 4096;
        let signal = generate_sine(num_samples, 440.0, sample_rate);
        let flush = vec![[0.0_f64; 2]; FRAME_SIZE * 4];

        // Feed in 256-sample chunks
        let mut wav_256 = make_wavelet();
        let mut out_256: Vec<[f64; 2]> = Vec::new();
        for chunk in signal.chunks(256) {
            out_256.extend(wav_256.process_block(chunk));
        }
        out_256.extend(wav_256.process_block(&flush));

        // Feed in 512-sample chunks
        let mut wav_512 = make_wavelet();
        let mut out_512: Vec<[f64; 2]> = Vec::new();
        for chunk in signal.chunks(512) {
            out_512.extend(wav_512.process_block(chunk));
        }
        out_512.extend(wav_512.process_block(&flush));

        let compare_len = out_256.len().min(out_512.len());
        let compare_len = compare_len.saturating_sub(FRAME_SIZE);

        assert!(compare_len > 0, "no samples to compare");

        for i in 0..compare_len {
            assert!(
                (out_256[i][0] - out_512[i][0]).abs() < 1e-10,
                "left channel streaming inconsistency at {i}: 256={} 512={}",
                out_256[i][0], out_512[i][0]
            );
            assert!(
                (out_256[i][1] - out_512[i][1]).abs() < 1e-10,
                "right channel streaming inconsistency at {i}: 256={} 512={}",
                out_256[i][1], out_512[i][1]
            );
        }
    }
}
