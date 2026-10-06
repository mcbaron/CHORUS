use std::collections::VecDeque;
use std::sync::Arc;
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::estimation::Splitter;
use crate::filters::{FilterChains, CONTRIBUTIONS};
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
/// - `chirp[n] = exp(-iπ · cot(φ) · n²/N)`
/// - `norm     = sqrt((1 - i·cot(φ)) / (N · |csc(φ)|))`  — unit-magnitude scaled
///
/// This operator satisfies `F_a^H · F_a = I` exactly, so the round-trip error
/// is only floating-point rounding (≈ 1e-13 for f64).
///
/// Special cases:
/// - `order = 0` → identity (pass-through)
/// - `order = 1` → plain FFT (forward) / plain IFFT·N (inverse)
///
/// Streaming model: hop_size = frame_size (no overlap). Filter chains apply to
/// each contribution in the time domain, after the inverse FrFT.
pub struct StreamingFrft {
    order: f64,
    frame_size: usize,
    chirp: Vec<Complex<f64>>,
    norm: Complex<f64>,
    input_buf: VecDeque<[f64; 2]>,
    output_buf: VecDeque<[f64; 2]>,
    splitter: Splitter,
    fft: Arc<dyn rustfft::Fft<f64>>,
    ifft: Arc<dyn rustfft::Fft<f64>>,
}

impl StreamingFrft {
    pub fn new(
        order: f64,
        frame_size: usize,
        sample_rate: u32,
        smoothing_alpha: f64,
        epsilon: f64,
        filter_chains: FilterChains,
    ) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_size);
        let ifft = planner.plan_fft_inverse(frame_size);

        let (chirp, norm) = if order == 0.0 || order == 1.0 {
            (vec![], Complex::new(1.0, 0.0))
        } else {
            Self::precompute_chirp(order, frame_size)
        };

        Self {
            order,
            frame_size,
            chirp,
            norm,
            input_buf: VecDeque::new(),
            output_buf: VecDeque::new(),
            splitter: Splitter::new(smoothing_alpha, epsilon, frame_size, sample_rate, filter_chains),
            fft,
            ifft,
        }
    }

    /// `chirp[k] = exp(-iπ·cot(φ)·k²/N)` and `norm = sqrt((1-i·cot(φ)) / (N·|csc(φ)|))`,
    /// where `|norm| = 1/√N` because `|csc(φ)| = sqrt(1+cot²)`.
    fn precompute_chirp(order: f64, n: usize) -> (Vec<Complex<f64>>, Complex<f64>) {
        let phi = order * std::f64::consts::PI / 2.0;
        let cot_phi = phi.cos() / phi.sin();
        let nf = n as f64;
        let chirp = (0..n)
            .map(|k| Complex::from_polar(1.0, -std::f64::consts::PI * cot_phi * (k as f64).powi(2) / nf))
            .collect();
        let norm = (Complex::new(1.0, -cot_phi) / (nf * (1.0 + cot_phi * cot_phi).sqrt())).sqrt();
        debug_assert!((norm.norm_sqr() - 1.0 / nf).abs() < 1e-12);
        (chirp, norm)
    }

    /// Forward FrFT of order `self.order` for a real-valued input.
    fn frft_transform(&self, real: &[f64]) -> Vec<Complex<f64>> {
        let mut buf: Vec<Complex<f64>> = real.iter().map(|&x| Complex::new(x, 0.0)).collect();
        if self.order == 0.0 {
            return buf;
        }
        if self.order == 1.0 {
            self.fft.process(&mut buf);
            return buf;
        }
        // norm · IFFT_1N( chirp · FFT( chirp · x ) )
        buf.iter_mut().zip(&self.chirp).for_each(|(b, c)| *b *= c);
        self.fft.process(&mut buf);
        buf.iter_mut().zip(&self.chirp).for_each(|(b, c)| *b = *b * c * self.norm);
        self.ifft.process(&mut buf);
        let inv_scale = 1.0 / self.frame_size as f64;
        buf.iter_mut().for_each(|b| *b *= inv_scale);
        buf
    }

    /// Inverse FrFT: `(1/norm) · conj(chirp) · IFFT_1N( conj(chirp) · FFT( y ) )`.
    fn ifrft_transform(&self, bins: &[Complex<f64>]) -> Vec<f64> {
        let n = self.frame_size;
        if self.order == 0.0 {
            return bins.iter().map(|c| c.re).collect();
        }
        let mut buf: Vec<Complex<f64>> = bins.to_vec();
        let inv_scale = 1.0 / n as f64;
        if self.order == 1.0 {
            self.ifft.process(&mut buf);
            return buf.iter().map(|c| c.re * inv_scale).collect();
        }
        let norm_inv = Complex::new(1.0, 0.0) / self.norm; // 1/norm (not conj(norm))
        self.fft.process(&mut buf);
        buf.iter_mut().zip(&self.chirp).for_each(|(b, c)| *b *= c.conj());
        self.ifft.process(&mut buf);
        buf.iter()
            .zip(&self.chirp)
            .map(|(b, c)| (b * inv_scale * c.conj() * norm_inv).re)
            .collect()
    }

    fn process_frame(&mut self) {
        let n = self.frame_size;
        let left: Vec<f64> = self.input_buf.iter().take(n).map(|s| s[0]).collect();
        let right: Vec<f64> = self.input_buf.iter().take(n).map(|s| s[1]).collect();
        let contributions = self.splitter.split(&self.frft_transform(&left), &self.frft_transform(&right));

        let mut out_frame: Vec<[f64; 2]> = vec![[0.0, 0.0]; n];
        for (i, (name, bins)) in CONTRIBUTIONS.iter().zip(contributions).enumerate() {
            let time_signal = self.splitter.filter_real(name, &self.ifrft_transform(&bins));
            for (out, t) in out_frame.iter_mut().zip(&time_signal) {
                out[i % 2] += t;
            }
        }
        self.output_buf.extend(out_frame);
        self.input_buf.drain(..n);
    }
}

impl Transform for StreamingFrft {
    fn process_block(&mut self, input: &[[f64; 2]]) -> Vec<[f64; 2]> {
        self.input_buf.extend(input);
        while self.input_buf.len() >= self.frame_size {
            self.process_frame();
        }
        self.output_buf.drain(..).collect()
    }

    fn reset(&mut self) {
        self.input_buf.clear();
        self.output_buf.clear();
        self.splitter.reset();
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
        StreamingFrft::new(order, FRAME_SIZE, 48_000, 0.0, 1e-12, unity_chains())
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
    fn eq_chain_runs_in_time_domain() {
        use crate::filters::FilterSpec;
        let mut chains = unity_chains();
        chains.insert("Lo".into(), vec![FilterSpec::Eq { mode: "highpass".into(), frequency_hz: 1000.0, q: 0.707, gain_db: None }]);
        let mut frft = StreamingFrft::new(0.5, FRAME_SIZE, 48_000, 0.0, 1e-12, chains);
        let input: Vec<[f64; 2]> = (0..4 * FRAME_SIZE).map(|i| [(i as f64 * 0.01).sin(), 0.0]).collect();
        let output = frft.process_block(&input);
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|s| s[0].is_finite() && s[1].is_finite()));
    }

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
        let (_, signal) = crate::load_wav_stereo(path).unwrap_or_else(|e| panic!("could not open {path}: {e}"));
        let num_samples = signal.len();
        let mut frft = StreamingFrft::new(0.5, WAV_FRAME_SIZE, 48_000, 0.0, 1e-12, unity_chains());
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
