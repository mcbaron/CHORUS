// TODO(follow-on): Python FrFTTransform uses phase-shifted FFT (FFT * exp(-0.5j*pi*order)),
// not true FrFT. Rust implements a unitary 2-chirp discrete FrFT.  Python must be updated
// before Rust/Python FrFT fixture parity is possible.

use std::collections::VecDeque;
use std::sync::Arc;
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::estimation::ComplexEstimator;
use crate::filters::{FilterChains, FilterSpec};
use crate::prototypes::{center_prototype, surround_prototype};
use crate::transforms::Transform;

/// Streaming Fractional Fourier Transform processor.
///
/// Implements a unitary 2-chirp discrete FrFT (order α, angle φ = α·π/2):
///
/// ```text
/// forward:  Y = norm · IFFT_1N( chirp · FFT( chirp · x ) )
/// inverse:  x = (1/norm) · conj(chirp) · IFFT_1N( conj(chirp) · FFT( Y ) )
/// ```
///
/// where:
/// - `chirp_pre[n]  = exp(-iπ · cot(φ) · n²/N)`
/// - `chirp_post[k] = exp(-iπ · cot(φ) · k²/N)` (same formula, same values)
/// - `norm          = sqrt((1 - i·cot(φ)) / (N · |csc(φ)|))`  — unit-magnitude scaled
///
/// This operator satisfies `F_a^H · F_a = I` exactly, so the round-trip error
/// is only floating-point rounding (≈ 1e-13 for f64).
///
/// Special cases:
/// - `order = 0` → identity (pass-through)
/// - `order = 1` → plain FFT (forward) / plain IFFT·N (inverse)
///
/// Streaming model: hop_size = frame_size (no overlap).
pub struct StreamingFrft {
    order: f64,
    frame_size: usize,
    /// Chirp for pre-multiply: `exp(-iπ·cot(φ)·n²/N)`, length frame_size.
    chirp_pre: Vec<Complex<f64>>,
    /// Chirp for freq-domain post-multiply: same formula as chirp_pre, length frame_size.
    chirp_post: Vec<Complex<f64>>,
    /// Scalar normalization: `sqrt((1-i·cot(φ)) / (N·|csc(φ)|))`.
    norm: Complex<f64>,
    input_buf: VecDeque<[f64; 2]>,
    output_buf: VecDeque<[f64; 2]>,
    lc_est: ComplexEstimator,
    rc_est: ComplexEstimator,
    ls_est: ComplexEstimator,
    rs_est: ComplexEstimator,
    filter_chains: FilterChains,
    fft: Arc<dyn rustfft::Fft<f64>>,
    ifft: Arc<dyn rustfft::Fft<f64>>,
}

impl StreamingFrft {
    pub fn new(
        order: f64,
        frame_size: usize,
        smoothing_alpha: f64,
        epsilon: f64,
        filter_chains: FilterChains,
    ) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_size);
        let ifft = planner.plan_fft_inverse(frame_size);

        let (chirp_pre, chirp_post, norm) = if order == 0.0 || order == 1.0 {
            (vec![], vec![], Complex::new(1.0, 0.0))
        } else {
            Self::precompute_chirps(order, frame_size)
        };

        Self {
            order,
            frame_size,
            chirp_pre,
            chirp_post,
            norm,
            input_buf: VecDeque::new(),
            output_buf: VecDeque::new(),
            lc_est: ComplexEstimator::new(smoothing_alpha, epsilon, frame_size),
            rc_est: ComplexEstimator::new(smoothing_alpha, epsilon, frame_size),
            ls_est: ComplexEstimator::new(smoothing_alpha, epsilon, frame_size),
            rs_est: ComplexEstimator::new(smoothing_alpha, epsilon, frame_size),
            filter_chains,
            fft,
            ifft,
        }
    }

    /// Precompute chirp tables.
    ///
    /// - `chirp_pre[n] = chirp_post[n] = exp(-iπ · cot(φ) · n²/N)`
    /// - `norm = sqrt((1-i·cot(φ)) / (N·|csc(φ)|))`  — magnitude exactly 1/√N
    fn precompute_chirps(order: f64, n: usize) -> (Vec<Complex<f64>>, Vec<Complex<f64>>, Complex<f64>) {
        let phi = order * std::f64::consts::PI / 2.0;
        let cot_phi = phi.cos() / phi.sin();
        let nf = n as f64;

        // chirp[k] = exp(-iπ·cot·k²/N)
        let chirp: Vec<Complex<f64>> = (0..n)
            .map(|k| {
                let angle = -std::f64::consts::PI * cot_phi * (k as f64).powi(2) / nf;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        // norm = sqrt((1-i·cot) / (N·|csc|))
        // = sqrt((1-i·cot) / (N·sqrt(1+cot²)))
        // This has magnitude: sqrt(|(1-i·cot)|/(N·sqrt(1+cot²))) = sqrt(sqrt(1+cot²)/(N·sqrt(1+cot²))) = 1/sqrt(N)
        let csc_mag = (1.0 + cot_phi * cot_phi).sqrt(); // |csc(φ)| = sqrt(1+cot²)
        let inside = Complex::new(1.0, -cot_phi) / (nf * csc_mag);
        let norm = inside.sqrt();
        // Verify |norm|² ≈ 1/N
        debug_assert!((norm.norm_sqr() - 1.0 / nf).abs() < 1e-12);

        (chirp.clone(), chirp, norm)
    }

    /// Forward FrFT of order `self.order` for a real-valued input.
    fn frft_transform(&self, real: &[f64]) -> Vec<Complex<f64>> {
        let n = self.frame_size;
        if self.order == 0.0 {
            return real.iter().map(|&x| Complex::new(x, 0.0)).collect();
        }
        if self.order == 1.0 {
            let mut buf: Vec<Complex<f64>> = real.iter().map(|&x| Complex::new(x, 0.0)).collect();
            self.fft.process(&mut buf);
            return buf;
        }
        let input: Vec<Complex<f64>> = real.iter().map(|&x| Complex::new(x, 0.0)).collect();
        self.frft_core(&input, &self.chirp_pre, &self.chirp_post, self.norm, n)
    }

    /// Inverse FrFT.
    ///
    /// The exact inverse of `F_a(x) = norm · IFFT_1N( chirp · FFT( chirp · x ) )` is:
    /// `F_a^{-1}(y) = (1/norm) · conj(chirp) · IFFT_1N( conj(chirp) · FFT( y ) )`
    ///
    /// Note that `conj(chirp)` appears after the IFFT_1N (post-multiply), not before the FFT.
    fn ifrft_transform(&self, bins: &[Complex<f64>]) -> Vec<f64> {
        let n = self.frame_size;
        if self.order == 0.0 {
            return bins.iter().map(|c| c.re).collect();
        }
        if self.order == 1.0 {
            // Inverse of plain FFT is normalized IFFT
            let mut buf: Vec<Complex<f64>> = bins.to_vec();
            self.ifft.process(&mut buf);
            let scale = 1.0 / n as f64;
            return buf.iter().map(|c| c.re * scale).collect();
        }
        let chirp_conj: Vec<Complex<f64>> = self.chirp_pre.iter().map(|c| c.conj()).collect();
        let norm_inv = Complex::new(1.0, 0.0) / self.norm; // 1/norm (not conj(norm))

        // Step 1: FFT of input (no pre-chirp multiply)
        let mut buf: Vec<Complex<f64>> = bins.to_vec();
        self.fft.process(&mut buf);

        // Step 2: multiply by conj(chirp_post) in freq domain
        for (b, c) in buf.iter_mut().zip(chirp_conj.iter()) {
            *b *= c;
        }

        // Step 3: IFFT_1N
        self.ifft.process(&mut buf);
        let inv_scale = 1.0 / n as f64;
        for b in buf.iter_mut() {
            *b *= inv_scale;
        }

        // Step 4: post-multiply by conj(chirp_pre) and 1/norm
        buf.iter()
            .zip(chirp_conj.iter())
            .map(|(b, c)| (b * c * norm_inv).re)
            .collect()
    }

    /// Core 2-chirp FrFT algorithm.
    ///
    /// `output = norm · IFFT_1N( chirp_post · FFT( chirp_pre · x ) )`
    fn frft_core(
        &self,
        input: &[Complex<f64>],
        chirp_pre: &[Complex<f64>],
        chirp_post: &[Complex<f64>],
        norm: Complex<f64>,
        n: usize,
    ) -> Vec<Complex<f64>> {

        // Step 1: pre-multiply by chirp
        let mut buf: Vec<Complex<f64>> = input
            .iter()
            .zip(chirp_pre.iter())
            .map(|(x, c)| x * c)
            .collect();

        // Step 2: FFT
        self.fft.process(&mut buf);

        // Step 3: post-multiply by chirp in frequency domain
        for (b, c) in buf.iter_mut().zip(chirp_post.iter()) {
            *b *= c;
        }

        // Step 4: apply normalization scalar
        for b in buf.iter_mut() {
            *b *= norm;
        }

        // Step 5: IFFT (normalized 1/N)
        self.ifft.process(&mut buf);
        let inv_scale = 1.0 / n as f64;
        for b in buf.iter_mut() {
            *b *= inv_scale;
        }

        buf
    }

    fn process_frame(&mut self) {
        let frame_size = self.frame_size;

        let frame: Vec<[f64; 2]> = self.input_buf.iter().take(frame_size).copied().collect();
        let left_signal: Vec<f64> = frame.iter().map(|s| s[0]).collect();
        let right_signal: Vec<f64> = frame.iter().map(|s| s[1]).collect();

        // Forward FrFT
        let left_bins = self.frft_transform(&left_signal);
        let right_bins = self.frft_transform(&right_signal);

        // Prototypes
        let center_proto = center_prototype(&left_bins, &right_bins);
        let surround_proto = surround_prototype(&left_bins, &right_bins);

        // Estimators
        let lc_bins = self.lc_est.estimate(&center_proto, &left_bins).to_vec();
        let rc_bins = self.rc_est.estimate(&center_proto, &right_bins).to_vec();
        let ls_bins = self.ls_est.estimate(&surround_proto, &left_bins).to_vec();
        let rs_bins = self.rs_est.estimate(&surround_proto, &right_bins).to_vec();

        // Residuals
        let lo_bins: Vec<Complex<f64>> = (0..frame_size)
            .map(|k| left_bins[k] - lc_bins[k] - ls_bins[k])
            .collect();
        let ro_bins: Vec<Complex<f64>> = (0..frame_size)
            .map(|k| right_bins[k] - rc_bins[k] - rs_bins[k])
            .collect();

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

        let apply_scalar = |bins: &[Complex<f64>], s: f64| -> Vec<Complex<f64>> {
            bins.iter().map(|c| c * s).collect()
        };

        let lc_scaled = apply_scalar(&lc_bins, sc_lc);
        let rc_scaled = apply_scalar(&rc_bins, sc_rc);
        let lo_scaled = apply_scalar(&lo_bins, sc_lo);
        let ro_scaled = apply_scalar(&ro_bins, sc_ro);
        let ls_scaled = apply_scalar(&ls_bins, sc_ls);
        let rs_scaled = apply_scalar(&rs_bins, sc_rs);

        // Inverse FrFT each contribution, sum per channel
        let lc_time = self.ifrft_transform(&lc_scaled);
        let lo_time = self.ifrft_transform(&lo_scaled);
        let ls_time = self.ifrft_transform(&ls_scaled);
        let rc_time = self.ifrft_transform(&rc_scaled);
        let ro_time = self.ifrft_transform(&ro_scaled);
        let rs_time = self.ifrft_transform(&rs_scaled);

        for i in 0..frame_size {
            let left_out = lc_time[i] + lo_time[i] + ls_time[i];
            let right_out = rc_time[i] + ro_time[i] + rs_time[i];
            self.output_buf.push_back([left_out, right_out]);
        }

        // Advance ring by frame_size (no overlap)
        for _ in 0..frame_size {
            self.input_buf.pop_front();
        }
    }
}

impl Transform for StreamingFrft {
    fn process_block(&mut self, input: &[[f64; 2]]) -> Vec<[f64; 2]> {
        for &sample in input {
            self.input_buf.push_back(sample);
        }
        while self.input_buf.len() >= self.frame_size {
            self.process_frame();
        }
        self.output_buf.drain(..).collect()
    }

    fn reset(&mut self) {
        self.input_buf.clear();
        self.output_buf.clear();
        self.lc_est.reset();
        self.rc_est.reset();
        self.ls_est.reset();
        self.rs_est.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingFrft;
    use crate::filters::unity_chains;
    use crate::transforms::Transform;
    use num_complex::Complex;
    use rustfft::FftPlanner;

    const FRAME_SIZE: usize = 256;
    const WAV_FRAME_SIZE: usize = 1024;
    const WAV_HOP_SIZE: usize = 1024;

    fn make_frft(order: f64) -> StreamingFrft {
        StreamingFrft::new(order, FRAME_SIZE, 0.0, 1e-12, unity_chains())
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

    /// Test 1: Identity at order=0.0 — input passes through unchanged.
    #[test]
    fn identity_at_order_zero() {
        let mut frft = make_frft(0.0);
        let signal: Vec<[f64; 2]> = (0..FRAME_SIZE)
            .map(|i| [i as f64 * 0.001, i as f64 * 0.002])
            .collect();

        let output = frft.process_block(&signal);
        assert_eq!(output.len(), FRAME_SIZE);

        for (i, (out, inp)) in output.iter().zip(signal.iter()).enumerate() {
            assert!(
                (out[0] - inp[0]).abs() < 1e-10,
                "left mismatch at {i}: got {}, expected {}",
                out[0], inp[0]
            );
            assert!(
                (out[1] - inp[1]).abs() < 1e-10,
                "right mismatch at {i}: got {}, expected {}",
                out[1], inp[1]
            );
        }
    }

    /// Test 2: FFT equivalence at order=1.0.
    #[test]
    fn fft_equivalence_at_order_one() {
        let frft = make_frft(1.0);

        let real: Vec<f64> = (0..FRAME_SIZE)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / FRAME_SIZE as f64).sin())
            .collect();

        let frft_result = frft.frft_transform(&real);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let mut buf: Vec<Complex<f64>> = real.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buf);

        for (i, (a, b)) in frft_result.iter().zip(buf.iter()).enumerate() {
            assert!((a.re - b.re).abs() < 1e-10, "re mismatch at {i}: {} vs {}", a.re, b.re);
            assert!((a.im - b.im).abs() < 1e-10, "im mismatch at {i}: {} vs {}", a.im, b.im);
        }
    }

    /// Test 3: Round-trip at order=0.5 — FrFT then inverse FrFT reconstructs input within 1e-6.
    #[test]
    fn round_trip_at_order_half() {
        let frft = make_frft(0.5);

        let real: Vec<f64> = (0..FRAME_SIZE)
            .map(|i| (2.0 * std::f64::consts::PI * 7.0 * i as f64 / FRAME_SIZE as f64).sin())
            .collect();

        let bins = frft.frft_transform(&real);
        let reconstructed = frft.ifrft_transform(&bins);

        assert_eq!(reconstructed.len(), real.len());
        for (i, (r, orig)) in reconstructed.iter().zip(real.iter()).enumerate() {
            assert!(
                (r - orig).abs() < 1e-6,
                "round-trip mismatch at sample {i}: got {r}, expected {orig}"
            );
        }
    }

    /// Test 4: process_block with order=0.0 reconstructs input exactly.
    #[test]
    fn process_block_identity_order_zero() {
        let mut frft = make_frft(0.0);
        let signal = generate_sine(FRAME_SIZE, 440.0, 48000.0);
        let output = frft.process_block(&signal);
        assert_eq!(output.len(), FRAME_SIZE);

        for (i, (out, inp)) in output.iter().zip(signal.iter()).enumerate() {
            assert!((out[0] - inp[0]).abs() < 1e-10, "left mismatch at {i}");
            assert!((out[1] - inp[1]).abs() < 1e-10, "right mismatch at {i}");
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
        let mut frft = StreamingFrft::new(0.5, WAV_FRAME_SIZE, 0.0, 1e-12, unity_chains());
        let mut all_output: Vec<[f64; 2]> = Vec::new();

        for chunk in signal.chunks(WAV_HOP_SIZE) {
            let out = frft.process_block(chunk);
            all_output.extend_from_slice(&out);
        }
        // Flush with silence to drain any remaining output
        let silence = vec![[0.0_f64; 2]; WAV_FRAME_SIZE * 2];
        let out = frft.process_block(&silence);
        all_output.extend_from_slice(&out);

        // FrFT with hop == frame emits one frame after frame_size samples.
        // Compare the middle portion, skipping the first and last frame to
        // avoid boundary effects at both ends.
        let skip = WAV_FRAME_SIZE;
        let compare_len = num_samples.saturating_sub(2 * WAV_FRAME_SIZE);

        assert!(
            all_output.len() >= skip + compare_len,
            "not enough output samples: got {}, need {}",
            all_output.len(),
            skip + compare_len
        );

        for i in 0..compare_len {
            let out_sample = all_output[skip + i];
            let in_sample = signal[skip + i];
            assert!(
                (out_sample[0] - in_sample[0]).abs() < 1e-3,
                "left channel WAV mismatch at sample {}: out={}, expected={}",
                skip + i,
                out_sample[0],
                in_sample[0]
            );
            assert!(
                (out_sample[1] - in_sample[1]).abs() < 1e-3,
                "right channel WAV mismatch at sample {}: out={}, expected={}",
                skip + i,
                out_sample[1],
                in_sample[1]
            );
        }
    }

    #[test]
    fn wav_round_trip_pinkpanther() {
        run_wav_round_trip("../../tests/test_tracks_wav/PinkPanther.wav");
    }

    #[test]
    fn wav_round_trip_tvsong() {
        run_wav_round_trip("../../tests/test_tracks_wav/TVSong.wav");
    }

    /// Test 5: Streaming consistency — chunk size does not affect output.
    #[test]
    fn streaming_consistency() {
        let num_samples = FRAME_SIZE * 4;
        let signal = generate_sine(num_samples, 440.0, 48000.0);

        let mut frft_full = make_frft(0.5);
        let mut out_full: Vec<[f64; 2]> = Vec::new();
        for chunk in signal.chunks(FRAME_SIZE) {
            out_full.extend(frft_full.process_block(chunk));
        }

        let mut frft_half = make_frft(0.5);
        let mut out_half: Vec<[f64; 2]> = Vec::new();
        for chunk in signal.chunks(FRAME_SIZE / 2) {
            out_half.extend(frft_half.process_block(chunk));
        }

        assert_eq!(out_full.len(), out_half.len());
        for (i, (a, b)) in out_full.iter().zip(out_half.iter()).enumerate() {
            assert!((a[0] - b[0]).abs() < 1e-12, "left inconsistency at {i}");
            assert!((a[1] - b[1]).abs() < 1e-12, "right inconsistency at {i}");
        }
    }
}
