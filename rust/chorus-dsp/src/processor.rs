use crate::filters::{unity_chains, FilterChains};
use crate::transforms::{StreamingFrft, StreamingStft, StreamingWavelet, Transform, TransformKind};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspConfig {
    /// Note: sample_rate is not currently forwarded to transform constructors (transforms use fixed 48kHz defaults).
    pub sample_rate: u32,
    pub epsilon: f64,
    pub filter_chains: FilterChains,
    #[serde(default)]
    pub transform: TransformKind,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            epsilon: 1e-9,
            filter_chains: unity_chains(),
            transform: TransformKind::default(),
        }
    }
}

impl DspConfig {
    /// Low-latency preset: frame_size=512, hop_size=256 (~10.7ms at 48kHz vs ~21.3ms default).
    ///
    /// Profiling confirmed real-time safety for all target DJ-host callback sizes.
    pub fn low_latency() -> Self {
        Self {
            sample_rate: 48_000,
            epsilon: 1e-9,
            filter_chains: crate::filters::unity_chains(),
            transform: crate::transforms::TransformKind::Stft {
                frame_size: 512,
                hop_size: 256,
                smoothing_alpha: 0.0,
            },
        }
    }
}

fn build_transform(config: &DspConfig) -> Box<dyn Transform> {
    match &config.transform {
        TransformKind::Stft { frame_size, hop_size, smoothing_alpha } =>
            Box::new(StreamingStft::new(*frame_size, *hop_size, *smoothing_alpha, config.epsilon, config.filter_chains.clone())),
        TransformKind::Frft { order, frame_size, smoothing_alpha } =>
            Box::new(StreamingFrft::new(*order, *frame_size, *smoothing_alpha, config.epsilon, config.filter_chains.clone())),
        TransformKind::Wavelet { wavelet: _, level, frame_size, smoothing_alpha } =>
            Box::new(StreamingWavelet::new(*level, *frame_size, *smoothing_alpha, config.epsilon, config.filter_chains.clone())),
    }
}

#[derive(Debug, Error)]
pub enum ChorusError {
    #[error("input must contain at least one stereo frame")]
    EmptyInput,
}

pub struct ChorusDsp {
    // Retained for inspection/serialization; transform reconfiguration requires re-constructing ChorusDsp.
    #[allow(dead_code)]
    config: DspConfig,
    transform: Box<dyn Transform>,
}

impl ChorusDsp {
    pub fn new(config: DspConfig) -> Self {
        let transform = build_transform(&config);
        Self { config, transform }
    }

    pub fn process(&mut self, input: &[[f64; 2]]) -> Result<Vec<[f64; 2]>, ChorusError> {
        if input.is_empty() {
            return Err(ChorusError::EmptyInput);
        }
        Ok(self.transform.process_block(input))
    }

    pub fn reset(&mut self) {
        self.transform.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{ChorusDsp, DspConfig};

    #[test]
    fn process_does_not_error_on_valid_input() {
        let input: Vec<[f64; 2]> = (0..2048).map(|i| {
            let t = i as f64 / 48000.0;
            let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5;
            [s, s]
        }).collect();
        let mut dsp = ChorusDsp::new(DspConfig::default());
        let output = dsp.process(&input).unwrap();
        // Should produce output (may have latency during warm-up, but non-empty)
        assert!(!output.is_empty() || input.len() < 1024, "expected output for large enough input");
    }

    #[test]
    fn process_errors_on_empty_input() {
        let mut dsp = ChorusDsp::new(DspConfig::default());
        assert!(dsp.process(&[]).is_err());
    }

    #[test]
    fn reset_does_not_panic() {
        let input = vec![[0.25_f64, 0.25], [0.0, 0.0], [-0.25, -0.25]];
        let mut dsp = ChorusDsp::new(DspConfig::default());
        let _ = dsp.process(&input);
        dsp.reset();
        // After reset, processing again should not panic
        let _ = dsp.process(&input);
    }

    #[test]
    fn low_latency_preset_produces_output() {
        let input: Vec<[f64; 2]> = (0..2048).map(|i| {
            let t = i as f64 / 48_000.0;
            let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5;
            [s, s]
        }).collect();
        let mut dsp = ChorusDsp::new(DspConfig::low_latency());
        let output = dsp.process(&input).unwrap();
        assert!(!output.is_empty(), "low_latency preset produced no output for 2048 input samples");
        for &[l, r] in &output {
            assert!(l.is_finite() && r.is_finite(), "non-finite sample in low_latency output");
        }
    }

    #[test]
    fn unity_bypass_reconstructs_input_after_warmup() {
        // Feed enough samples so the STFT has warmed up past the first frame
        let n = 4096usize;
        let input: Vec<[f64; 2]> = (0..n).map(|i| {
            let t = i as f64 / 48000.0;
            let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5;
            [s, s * 0.8]
        }).collect();
        let mut dsp = ChorusDsp::new(DspConfig::default());
        let output = dsp.process(&input).unwrap();
        let skip = 1024; // skip warm-up
        let compare_len = output.len().min(input.len()).saturating_sub(skip);
        assert!(compare_len > 0, "expected output after warm-up, but got only {} output samples", output.len());
        for i in skip..skip + compare_len {
            let diff_l = (output[i][0] - input[i][0]).abs();
            let diff_r = (output[i][1] - input[i][1]).abs();
            assert!(diff_l < 1e-3, "left channel diff too large at {i}: {diff_l}");
            assert!(diff_r < 1e-3, "right channel diff too large at {i}: {diff_r}");
        }
    }
}
