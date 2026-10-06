use std::collections::VecDeque;
use num_complex::Complex;

use crate::estimation::Splitter;
use crate::filters::{FilterChains, CONTRIBUTIONS};
use crate::transforms::Transform;

// Daubechies-4 (db4) decomposition low-pass filter (8 taps)
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
// The other three filters follow from DB4_LO (quadrature mirror relations).
const REC_LO: [f64; 8] = reversed(DB4_LO);
const REC_HI: [f64; 8] = alternate_signs(DB4_LO);
const DB4_HI: [f64; 8] = reversed(REC_HI);

const fn reversed(f: [f64; 8]) -> [f64; 8] {
    let mut out = [0.0; 8];
    let mut k = 0;
    while k < 8 {
        out[k] = f[7 - k];
        k += 1;
    }
    out
}

/// `out[k] = (-1)^k * f[k]`
const fn alternate_signs(f: [f64; 8]) -> [f64; 8] {
    let mut out = f;
    let mut k = 1;
    while k < 8 {
        out[k] = -f[k];
        k += 2;
    }
    out
}

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
    splitter: Splitter,
}

impl StreamingWavelet {
    pub fn new(
        level: usize,
        frame_size: usize,
        sample_rate: u32,
        smoothing_alpha: f64,
        epsilon: f64,
        filter_chains: FilterChains,
    ) -> Self {
        let overlap = 2 * boundary_contamination(level);
        let num_coeffs = wavedec_num_coeffs(frame_size + overlap, level);
        Self {
            level,
            frame_size,
            overlap,
            input_buf: VecDeque::from(vec![[0.0, 0.0]; overlap]), // pre-load overlap zeros
            output_buf: VecDeque::new(),
            splitter: Splitter::new(smoothing_alpha, epsilon, num_coeffs, sample_rate, filter_chains),
        }
    }

    fn process_block_internal(&mut self) {
        let block_size = self.frame_size + self.overlap;
        let bc = self.overlap / 2;

        // Wavelet decompose each channel of the block
        let channel = |c: usize| -> Vec<f64> { self.input_buf.iter().take(block_size).map(|s| s[c]).collect() };
        let left_coeffs = wavedec(&channel(0), self.level);
        let right_coeffs = wavedec(&channel(1), self.level);
        let band_lens: Vec<usize> = left_coeffs.iter().map(Vec::len).collect();

        // Flatten the real coefficients and wrap them as complex (zero imaginary part)
        let as_complex = |bands: &[Vec<f64>]| -> Vec<Complex<f64>> {
            bands.iter().flatten().map(|&x| Complex::new(x, 0.0)).collect()
        };
        let contributions = self.splitter.split(&as_complex(&left_coeffs), &as_complex(&right_coeffs));

        // Reconstruct each contribution as audio, filter it, and sum per channel.
        // Filters act on audio, not on coefficients (an EQ curve has no meaning there).
        let mut recon = [vec![0.0; block_size], vec![0.0; block_size]];
        for (i, (name, coeffs)) in CONTRIBUTIONS.iter().zip(contributions).enumerate() {
            let real: Vec<f64> = coeffs.iter().map(|c| c.re).collect();
            let audio = waverec(&unflatten_coeffs(&real, &band_lens), block_size);
            for (out, s) in recon[i % 2].iter_mut().zip(self.splitter.filter_real(name, &audio)) {
                *out += s;
            }
        }

        // Discard boundary contamination (first bc and last bc samples)
        let [left_recon, right_recon] = recon;
        self.output_buf.extend((bc..block_size - bc).map(|i| [left_recon[i], right_recon[i]]));
        self.input_buf.drain(..self.frame_size);
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
        self.input_buf.extend(input);
        while self.input_buf.len() >= self.frame_size + self.overlap {
            self.process_block_internal();
        }
        self.output_buf.drain(..).collect()
    }

    fn reset(&mut self) {
        self.input_buf = VecDeque::from(vec![[0.0, 0.0]; self.overlap]);
        self.output_buf.clear();
        self.splitter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{StreamingWavelet, wavedec, waverec};
    use crate::filters::unity_chains;
    use crate::transforms::Transform;

    const LEVEL: usize = 3;
    const FRAME_SIZE: usize = 512;

    #[test]
    fn eq_highpass_acts_on_audio_not_coefficients() {
        use crate::filters::FilterSpec;
        // A left-only tone goes to Lo. A 2nd-order 1 kHz high-pass must cut 100 Hz by ~40 dB
        // and pass 5 kHz.
        let gain_db = |freq: f64| {
            let mut chains = unity_chains();
            chains.insert("Lo".into(), vec![FilterSpec::Eq { mode: "highpass".into(), frequency_hz: 1000.0, q: 0.707, gain_db: None }]);
            let mut wav = StreamingWavelet::new(LEVEL, FRAME_SIZE, 48_000, 0.0, 1e-12, chains);
            let input: Vec<[f64; 2]> = (0..48_000)
                .map(|i| [(2.0 * std::f64::consts::PI * freq * i as f64 / 48_000.0).sin(), 0.0])
                .collect();
            let output = wav.process_block(&input);
            let energy = |x: &[[f64; 2]]| x[4096..40_000].iter().map(|s| s[0] * s[0]).sum::<f64>();
            10.0 * (energy(&output) / energy(&input)).log10()
        };
        let low = gain_db(100.0);
        let high = gain_db(5000.0);
        assert!(low < -30.0, "100 Hz gain {low:.1} dB, expected < -30 dB");
        assert!(high.abs() < 1.0, "5 kHz gain {high:.1} dB, expected about 0 dB");
    }

    fn make_wavelet() -> StreamingWavelet {
        StreamingWavelet::new(LEVEL, FRAME_SIZE, 48_000u32, 0.0, 1e-12, unity_chains())
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
                (out_sample[0] - in_sample[0]).abs() < 1e-6,
                "left channel mismatch at sample {}: out={} vs in={}",
                i, out_sample[0], in_sample[0]
            );
            assert!(
                (out_sample[1] - in_sample[1]).abs() < 1e-6,
                "right channel mismatch at sample {}: out={} vs in={}",
                i, out_sample[1], in_sample[1]
            );
        }
    }

    fn run_wav_round_trip(path: &str) {
        let (_, signal) = crate::load_wav_stereo(path).unwrap_or_else(|e| panic!("could not open {path}: {e}"));
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
                (out_sample[0] - in_sample[0]).abs() < 1e-6,
                "left channel WAV mismatch at sample {}: out={}, expected={}",
                i, out_sample[0], in_sample[0]
            );
            assert!(
                (out_sample[1] - in_sample[1]).abs() < 1e-6,
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

    /// Test 3b: Boundary contamination is discarded and does not appear in output
    #[test]
    fn boundary_contamination_discarded() {
        // Create a signal that is zero everywhere except in the trailing boundary region
        // of the first block. The wavelet overlap-save algorithm discards the last `bc`
        // samples of each processed block, so those non-zero samples should NOT appear
        // in the output.
        const FRAME: usize = 512;
        const LEVEL: usize = 3;
        let bc = (8 - 1) * ((1 << LEVEL) - 1); // = 49
        let _overlap = 2 * bc;

        let mut wav = StreamingWavelet::new(LEVEL, FRAME, 48_000u32, 0.0, 1e-9, unity_chains());

        // The input_buf is pre-loaded with `overlap` zeros. After feeding FRAME samples the
        // buf has `overlap + FRAME = block_size` samples, which triggers the first emission.
        // We put non-zero values in the last `bc` positions of the FRAME we feed — these map
        // to the trailing boundary zone of the first processed block and should be discarded.
        let mut input = vec![[0.0_f64; 2]; FRAME];
        for i in (FRAME - bc)..FRAME {
            input[i] = [1.0, 1.0];
        }
        let output = wav.process_block(&input);
        // The first FRAME of input triggers exactly one emission of FRAME output samples.
        assert_eq!(output.len(), FRAME, "expected one frame of output after feeding first frame");
        // The non-zero signal was in the trailing boundary zone that gets discarded —
        // output should be near-zero (small residual may exist due to wavelet spreading).
        for (idx, frame) in output.iter().enumerate() {
            assert!(frame[0].abs() < 0.1, "boundary contamination leaked into output at {}: {}", idx, frame[0]);
            assert!(frame[1].abs() < 0.1, "boundary contamination leaked into output at {}: {}", idx, frame[1]);
        }
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
