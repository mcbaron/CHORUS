use std::collections::VecDeque;
use std::sync::Arc;
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::estimation::Splitter;
use crate::filters::{FilterChains, CONTRIBUTIONS};
use crate::transforms::{sqrt_hann, Transform};

pub struct StreamingStft {
    frame_size: usize,
    hop_size: usize,
    window: Vec<f64>,
    input_buf: VecDeque<[f64; 2]>,
    output_buf: VecDeque<[f64; 2]>,
    overlap: Vec<[f64; 2]>,
    discard_hop: bool, // true until the first (zero-padded) hop is dropped
    splitter: Splitter,
    fft: Arc<dyn rustfft::Fft<f64>>,
    ifft: Arc<dyn rustfft::Fft<f64>>,
}

impl StreamingStft {
    pub fn new(
        frame_size: usize,
        hop_size: usize,
        sample_rate: u32,
        smoothing_alpha: f64,
        epsilon: f64,
        filter_chains: FilterChains,
    ) -> Self {
        let num_bins = frame_size / 2 + 1;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame_size);
        let ifft = planner.plan_fft_inverse(frame_size);

        Self {
            frame_size,
            hop_size,
            window: sqrt_hann(frame_size),
            // One hop of zeros first, so the first frame starts one hop before the input,
            // as in the Python reference. The first output hop is then dropped.
            input_buf: VecDeque::from(vec![[0.0, 0.0]; hop_size]),
            output_buf: VecDeque::new(),
            overlap: vec![[0.0, 0.0]; frame_size - hop_size],
            discard_hop: true,
            splitter: Splitter::new(smoothing_alpha, epsilon, num_bins, sample_rate, filter_chains),
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

    fn process_hop(&mut self) {
        let frame_size = self.frame_size;
        let hop_size = self.hop_size;

        // Window the first frame_size samples and separate channels
        let left_windowed: Vec<f64> = self.input_buf.iter().zip(&self.window).map(|(s, w)| s[0] * w).collect();
        let right_windowed: Vec<f64> = self.input_buf.iter().zip(&self.window).map(|(s, w)| s[1] * w).collect();
        let contributions = self.splitter.split(&self.rfft(&left_windowed), &self.rfft(&right_windowed));

        // Filter and IFFT each contribution, apply the synthesis window, and sum per channel.
        let mut out_frame: Vec<[f64; 2]> = vec![[0.0, 0.0]; frame_size];
        for (i, (name, bins)) in CONTRIBUTIONS.iter().zip(contributions).enumerate() {
            let time_signal = self.irfft(&self.splitter.filter_bins(name, bins));
            for ((out, t), w) in out_frame.iter_mut().zip(&time_signal).zip(&self.window) {
                out[i % 2] += t * w;
            }
        }

        // Add overlap tail to the beginning
        for (out, tail) in out_frame.iter_mut().zip(&self.overlap) {
            out[0] += tail[0];
            out[1] += tail[1];
        }

        // Emit hop_size samples, keep the rest as the next overlap, and advance the input.
        let start = if std::mem::take(&mut self.discard_hop) { hop_size } else { 0 };
        self.output_buf.extend(&out_frame[start..hop_size]);
        self.overlap = out_frame[hop_size..].to_vec();
        self.input_buf.drain(..hop_size);
    }
}

impl Transform for StreamingStft {
    fn process_block(&mut self, input: &[[f64; 2]]) -> Vec<[f64; 2]> {
        self.input_buf.extend(input);
        while self.input_buf.len() >= self.frame_size {
            self.process_hop();
        }
        self.output_buf.drain(..).collect()
    }

    fn reset(&mut self) {
        self.input_buf = VecDeque::from(vec![[0.0, 0.0]; self.hop_size]);
        self.output_buf.clear();
        self.overlap.fill([0.0, 0.0]);
        self.discard_hop = true;
        self.splitter.reset();
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
        StreamingStft::new(FRAME_SIZE, HOP_SIZE, 48_000u32, 0.0, 1e-12, unity_chains())
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
    fn first_hop_reconstructs_input() {
        // The first frame starts one hop before the input, as in the Python reference,
        // so the first output hop gets both overlapping frames.
        let mut t = make_stft();
        let input: Vec<[f64; 2]> = (0..4096).map(|i| [(i as f64 * 0.01).sin(), (i as f64 * 0.02).cos()]).collect();
        let output = t.process_block(&input);
        for i in 0..512 {
            assert!((output[i][0] - input[i][0]).abs() < 1e-9, "left mismatch at {i}");
            assert!((output[i][1] - input[i][1]).abs() < 1e-9, "right mismatch at {i}");
        }
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

    fn run_wav_round_trip(path: &str) {
        let (_, signal) = crate::load_wav_stereo(path).unwrap_or_else(|e| panic!("could not open {path}: {e}"));
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
    fn eq_highpass_reduces_low_frequency_energy() {
        use crate::filters::{FilterSpec, FilterChains};

        let sample_rate = 48_000.0;
        let num_samples: usize = 4096;
        let signal: Vec<[f64; 2]> = (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let v = (2.0 * std::f64::consts::PI * 440.0 * t).sin();
                [v, v]
            })
            .collect();

        let mut chains: FilterChains = crate::filters::unity_chains();
        chains.insert("Lo".to_string(), vec![
            FilterSpec::Eq { mode: "highpass".to_string(), frequency_hz: 1000.0, q: 0.707, gain_db: Some(0.0) }
        ]);
        chains.insert("Lc".to_string(), vec![FilterSpec::Mute]);
        chains.insert("Ls".to_string(), vec![FilterSpec::Mute]);
        chains.insert("Rc".to_string(), vec![FilterSpec::Mute]);
        chains.insert("Rs".to_string(), vec![FilterSpec::Mute]);
        chains.insert("Ro".to_string(), vec![FilterSpec::Mute]);

        let mut stft = StreamingStft::new(FRAME_SIZE, HOP_SIZE, 48_000u32, 0.0, 1e-12, chains);

        let mut all_output: Vec<[f64; 2]> = Vec::new();
        for chunk in signal.chunks(HOP_SIZE) {
            all_output.extend(stft.process_block(chunk));
        }
        let flush = vec![[0.0_f64; 2]; FRAME_SIZE * 2];
        all_output.extend(stft.process_block(&flush));

        let skip = HOP_SIZE;
        let compare_len = num_samples.saturating_sub(2 * HOP_SIZE);
        let input_energy: f64 = signal[skip..skip + compare_len].iter().map(|s| s[0] * s[0]).sum();
        let output_energy: f64 = all_output[skip..skip + compare_len].iter().map(|s| s[0] * s[0]).sum();

        assert!(all_output.iter().all(|s| s[0].is_finite() && s[1].is_finite()),
            "Output contains non-finite values");
        assert!(
            output_energy < input_energy * 0.1,
            "Expected >10 dB reduction: input_energy={input_energy:.4}, output_energy={output_energy:.4}"
        );
    }

    #[test]
    fn wav_round_trip() {
        run_wav_round_trip("../../tests/test_tracks_wav/PinkPanther.wav");
    }

    #[test]
    fn wav_round_trip_tvsong() {
        run_wav_round_trip("../../tests/test_tracks_wav/TVSong.wav");
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
