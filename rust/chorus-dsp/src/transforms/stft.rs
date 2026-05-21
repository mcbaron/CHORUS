use std::collections::VecDeque;
use std::sync::Arc;
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::estimation::ComplexEstimator;
use crate::filters::{FilterChains, FilterSpec};
use crate::prototypes::{center_prototype, surround_prototype};
use crate::transforms::Transform;

pub struct StreamingStft {
    frame_size: usize,
    hop_size: usize,
    window: Vec<f64>,
    input_buf: VecDeque<[f64; 2]>,
    output_buf: VecDeque<[f64; 2]>,
    overlap: Vec<[f64; 2]>,
    lc_est: ComplexEstimator,
    rc_est: ComplexEstimator,
    ls_est: ComplexEstimator,
    rs_est: ComplexEstimator,
    filter_chains: FilterChains,
    fft: Arc<dyn rustfft::Fft<f64>>,
    ifft: Arc<dyn rustfft::Fft<f64>>,
}

impl StreamingStft {
    pub fn new(
        frame_size: usize,
        hop_size: usize,
        smoothing_alpha: f64,
        epsilon: f64,
        filter_chains: FilterChains,
    ) -> Self {
        // Precompute sqrt-Hann window using N (not N-1) denominator for exact OLA
        // at 50% overlap: w[n]^2 + w[n+N/2]^2 = 1 for all n when denominator = N.
        let window: Vec<f64> = (0..frame_size)
            .map(|n| {
                let hann = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / frame_size as f64).cos());
                hann.sqrt()
            })
            .collect();

        let num_bins = frame_size / 2 + 1;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_size);
        let ifft = planner.plan_fft_inverse(frame_size);

        Self {
            frame_size,
            hop_size,
            window,
            input_buf: VecDeque::new(),
            output_buf: VecDeque::new(),
            overlap: vec![[0.0, 0.0]; frame_size - hop_size],
            lc_est: ComplexEstimator::new(smoothing_alpha, epsilon, num_bins),
            rc_est: ComplexEstimator::new(smoothing_alpha, epsilon, num_bins),
            ls_est: ComplexEstimator::new(smoothing_alpha, epsilon, num_bins),
            rs_est: ComplexEstimator::new(smoothing_alpha, epsilon, num_bins),
            filter_chains,
            fft,
            ifft,
        }
    }

    /// Forward FFT: real input → one-sided complex spectrum (frame_size/2+1 bins)
    fn rfft(&self, real: &[f64]) -> Vec<Complex<f64>> {
        let mut buf: Vec<Complex<f64>> = real.iter().map(|&x| Complex::new(x, 0.0)).collect();
        self.fft.process(&mut buf);
        buf.truncate(self.frame_size / 2 + 1);
        buf
    }

    /// Inverse FFT: one-sided complex spectrum → real output
    fn irfft(&self, spectrum: &[Complex<f64>]) -> Vec<f64> {
        let n = self.frame_size;
        let num_bins = n / 2 + 1;
        let mut buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];

        // Copy positive frequencies
        buf[..num_bins].copy_from_slice(spectrum);

        // Mirror for negative frequencies: bin k → bin n-k (conjugate)
        for k in 1..num_bins - 1 {
            buf[n - k] = spectrum[k].conj();
        }

        self.ifft.process(&mut buf);

        // Normalize and take real part
        buf.iter().map(|c| c.re / n as f64).collect()
    }

    /// Compute the effective scalar multiplier for a contribution from its filter chain.
    fn chain_scalar(chains: &FilterChains, name: &str, soloed: &[String]) -> f64 {
        // If any solo exists and this contribution is not soloed, mute it
        if !soloed.is_empty() && !soloed.iter().any(|s| s == name) {
            return 0.0;
        }

        let default_chain = vec![FilterSpec::Unity];
        let chain = chains.get(name).unwrap_or(&default_chain);

        let mut scalar = 1.0_f64;
        for spec in chain {
            match spec {
                FilterSpec::Unity | FilterSpec::Solo => {}
                FilterSpec::Gain { db } => {
                    scalar *= 10.0_f64.powf(db / 20.0);
                }
                FilterSpec::Mute => {
                    scalar = 0.0;
                }
                FilterSpec::Polarity => {
                    scalar = -scalar;
                }
                FilterSpec::Eq { .. } => {
                    unimplemented!("EQ not yet supported");
                }
            }
        }
        scalar
    }

    fn process_hop(&mut self) {
        let frame_size = self.frame_size;
        let hop_size = self.hop_size;
        let num_bins = frame_size / 2 + 1;

        // Extract frame_size samples from input_buf
        let frame: Vec<[f64; 2]> = self.input_buf.iter().take(frame_size).copied().collect();

        // Apply window and separate channels
        let left_windowed: Vec<f64> = frame.iter().enumerate().map(|(i, s)| s[0] * self.window[i]).collect();
        let right_windowed: Vec<f64> = frame.iter().enumerate().map(|(i, s)| s[1] * self.window[i]).collect();

        // Forward FFT per channel
        let left_bins = self.rfft(&left_windowed);
        let right_bins = self.rfft(&right_windowed);

        // Compute prototypes
        let center_proto = center_prototype(&left_bins, &right_bins);
        let surround_proto = surround_prototype(&left_bins, &right_bins);

        // Run estimators
        let lc_bins = self.lc_est.estimate(&center_proto, &left_bins).to_vec();
        let rc_bins = self.rc_est.estimate(&center_proto, &right_bins).to_vec();
        let ls_bins = self.ls_est.estimate(&surround_proto, &left_bins).to_vec();
        let rs_bins = self.rs_est.estimate(&surround_proto, &right_bins).to_vec();

        // Derive Lo, Ro residuals
        let lo_bins: Vec<Complex<f64>> = (0..num_bins)
            .map(|k| left_bins[k] - lc_bins[k] - ls_bins[k])
            .collect();
        let ro_bins: Vec<Complex<f64>> = (0..num_bins)
            .map(|k| right_bins[k] - rc_bins[k] - rs_bins[k])
            .collect();

        // Compute solo list for filter chains
        let soloed: Vec<String> = self.filter_chains
            .iter()
            .filter(|(_, chain)| chain.iter().any(|spec| matches!(spec, FilterSpec::Solo)))
            .map(|(name, _)| name.clone())
            .collect();

        // Apply filter chain scalars
        let sc_lc = Self::chain_scalar(&self.filter_chains, "Lc", &soloed);
        let sc_rc = Self::chain_scalar(&self.filter_chains, "Rc", &soloed);
        let sc_lo = Self::chain_scalar(&self.filter_chains, "Lo", &soloed);
        let sc_ro = Self::chain_scalar(&self.filter_chains, "Ro", &soloed);
        let sc_ls = Self::chain_scalar(&self.filter_chains, "Ls", &soloed);
        let sc_rs = Self::chain_scalar(&self.filter_chains, "Rs", &soloed);

        let apply_scalar = |bins: &[Complex<f64>], s: f64| -> Vec<Complex<f64>> {
            bins.iter().map(|c| c * s).collect()
        };

        let lc_scaled = apply_scalar(&lc_bins, sc_lc);
        let rc_scaled = apply_scalar(&rc_bins, sc_rc);
        let lo_scaled = apply_scalar(&lo_bins, sc_lo);
        let ro_scaled = apply_scalar(&ro_bins, sc_ro);
        let ls_scaled = apply_scalar(&ls_bins, sc_ls);
        let rs_scaled = apply_scalar(&rs_bins, sc_rs);

        // IFFT each contribution separately, apply synthesis window, overlap-add into output frame.
        // Lc, Lo, Ls → left channel (index 0); Rc, Ro, Rs → right channel (index 1).
        let contributions: &[(&[Complex<f64>], usize)] = &[
            (&lc_scaled, 0),
            (&lo_scaled, 0),
            (&ls_scaled, 0),
            (&rc_scaled, 1),
            (&ro_scaled, 1),
            (&rs_scaled, 1),
        ];

        let overlap_len = frame_size - hop_size;

        // Accumulate windowed time-domain signals per channel
        let mut out_frame: Vec<[f64; 2]> = vec![[0.0, 0.0]; frame_size];

        for (bins, ch) in contributions.iter() {
            let time_signal = self.irfft(bins);
            for i in 0..frame_size {
                out_frame[i][*ch] += time_signal[i] * self.window[i];
            }
        }

        // Add overlap tail to the beginning
        for i in 0..overlap_len {
            out_frame[i][0] += self.overlap[i][0];
            out_frame[i][1] += self.overlap[i][1];
        }

        // Emit hop_size samples to output_buf
        for i in 0..hop_size {
            self.output_buf.push_back(out_frame[i]);
        }

        // Save the remaining samples as new overlap
        // overlap_len = frame_size - hop_size
        self.overlap.clear();
        for i in hop_size..frame_size {
            self.overlap.push([out_frame[i][0], out_frame[i][1]]);
        }

        // Advance input ring by hop_size
        for _ in 0..hop_size {
            self.input_buf.pop_front();
        }
    }
}

impl Transform for StreamingStft {
    fn process_block(&mut self, input: &[[f64; 2]]) -> Vec<[f64; 2]> {
        // Push input into buffer
        for &sample in input {
            self.input_buf.push_back(sample);
        }

        // Process all available hops
        while self.input_buf.len() >= self.frame_size {
            self.process_hop();
        }

        // Drain and return output
        self.output_buf.drain(..).collect()
    }

    fn reset(&mut self) {
        self.input_buf.clear();
        self.output_buf.clear();
        for s in &mut self.overlap {
            *s = [0.0, 0.0];
        }
        self.lc_est.reset();
        self.rc_est.reset();
        self.ls_est.reset();
        self.rs_est.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingStft;
    use crate::filters::unity_chains;
    use crate::transforms::Transform;

    const FRAME_SIZE: usize = 1024;
    const HOP_SIZE: usize = 512;

    fn make_stft() -> StreamingStft {
        StreamingStft::new(FRAME_SIZE, HOP_SIZE, 0.0, 1e-12, unity_chains())
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

    #[test]
    fn empty_output_during_warmup() {
        let mut stft = make_stft();
        // Push fewer than frame_size samples → should get empty output
        let input: Vec<[f64; 2]> = vec![[0.1, 0.1]; FRAME_SIZE - 1];
        let output = stft.process_block(&input);
        assert!(output.is_empty(), "expected empty output during warm-up, got {} samples", output.len());
    }

    #[test]
    fn sine_wave_round_trip() {
        let mut stft = make_stft();
        let sample_rate = 48000.0;
        let num_samples = 4096;
        let signal = generate_sine(num_samples, 440.0, sample_rate);

        let mut all_output: Vec<[f64; 2]> = Vec::new();
        // Process in hop-size chunks
        for chunk in signal.chunks(HOP_SIZE) {
            let out = stft.process_block(chunk);
            all_output.extend_from_slice(&out);
        }
        // Flush with silence to drain remaining output
        let silence = vec![[0.0_f64; 2]; FRAME_SIZE * 2];
        let out = stft.process_block(&silence);
        all_output.extend_from_slice(&out);

        // OLA reconstruction: output[j] ≈ input[j] once the overlap has been established.
        // First hop_size samples lack prior overlap contribution → skip them.
        // Compare output[HOP_SIZE .. num_samples-HOP_SIZE] vs the same input indices.
        let skip = HOP_SIZE; // skip first hop (no prior overlap)
        let compare_len = num_samples - 2 * HOP_SIZE; // leave tail margin

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
                "left channel mismatch at sample {}: out={} vs in={}",
                skip + i, out_sample[0], in_sample[0]
            );
            assert!(
                (out_sample[1] - in_sample[1]).abs() < 1e-3,
                "right channel mismatch at sample {}: out={} vs in={}",
                skip + i, out_sample[1], in_sample[1]
            );
        }
    }

    #[test]
    fn wav_round_trip() {
        let mut reader = hound::WavReader::open(
            "../../tests/test_tracks_wav/PinkPanther.wav"
        ).expect("could not open PinkPanther.wav");
        let spec = reader.spec();
        let num_channels = spec.channels as usize;
        assert!(num_channels <= 2, "expected mono or stereo WAV");

        // Normalize samples to [-1.0, 1.0] regardless of bit depth
        let scale = match spec.sample_format {
            hound::SampleFormat::Float => 1.0_f64,
            hound::SampleFormat::Int => 1.0 / (1_i64.checked_shl(spec.bits_per_sample as u32 - 1).unwrap_or(1) as f64),
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

        // Build stereo signal
        let signal: Vec<[f64; 2]> = if num_channels == 2 {
            raw_samples
                .chunks(2)
                .map(|c| [c[0], c[1]])
                .collect()
        } else {
            raw_samples.iter().map(|&s| [s, s]).collect()
        };

        let num_samples = signal.len();
        let mut stft = make_stft();
        let mut all_output: Vec<[f64; 2]> = Vec::new();

        for chunk in signal.chunks(HOP_SIZE) {
            let out = stft.process_block(chunk);
            all_output.extend_from_slice(&out);
        }
        // Flush
        let silence = vec![[0.0_f64; 2]; FRAME_SIZE * 2];
        let out = stft.process_block(&silence);
        all_output.extend_from_slice(&out);

        // Same OLA reconstruction alignment as sine test:
        // output[j] ≈ input[j] for j >= HOP_SIZE.
        let skip = HOP_SIZE;
        let compare_len = num_samples.saturating_sub(2 * HOP_SIZE + skip);

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
                skip + i, out_sample[0], in_sample[0]
            );
            assert!(
                (out_sample[1] - in_sample[1]).abs() < 1e-3,
                "right channel WAV mismatch at sample {}: out={}, expected={}",
                skip + i, out_sample[1], in_sample[1]
            );
        }
    }

    #[test]
    fn streaming_consistency() {
        let sample_rate = 48000.0;
        let num_samples = 4096;
        let signal = generate_sine(num_samples, 440.0, sample_rate);
        let flush = vec![[0.0_f64; 2]; FRAME_SIZE * 2];

        // Feed in 256-sample chunks
        let mut stft_256 = make_stft();
        let mut out_256: Vec<[f64; 2]> = Vec::new();
        for chunk in signal.chunks(256) {
            out_256.extend(stft_256.process_block(chunk));
        }
        out_256.extend(stft_256.process_block(&flush));

        // Feed in 512-sample chunks
        let mut stft_512 = make_stft();
        let mut out_512: Vec<[f64; 2]> = Vec::new();
        for chunk in signal.chunks(512) {
            out_512.extend(stft_512.process_block(chunk));
        }
        out_512.extend(stft_512.process_block(&flush));

        let compare_len = out_256.len().min(out_512.len());
        // Skip the last partial hop to avoid edge differences
        let compare_len = compare_len.saturating_sub(HOP_SIZE);

        for i in 0..compare_len {
            assert!(
                (out_256[i][0] - out_512[i][0]).abs() < 1e-12,
                "left channel streaming inconsistency at {i}: 256={} 512={}",
                out_256[i][0], out_512[i][0]
            );
            assert!(
                (out_256[i][1] - out_512[i][1]).abs() < 1e-12,
                "right channel streaming inconsistency at {i}: 256={} 512={}",
                out_256[i][1], out_512[i][1]
            );
        }
    }
}
