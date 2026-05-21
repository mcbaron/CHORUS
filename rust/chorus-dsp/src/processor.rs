use crate::filters::{unity_chains, FilterChains};
use crate::transforms::{StreamingFrft, StreamingStft, StreamingWavelet, Transform, TransformKind};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspConfig {
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
        if compare_len == 0 {
            return; // not enough output yet
        }
        for i in skip..skip + compare_len {
            let diff_l = (output[i][0] - input[i][0]).abs();
            let diff_r = (output[i][1] - input[i][1]).abs();
            assert!(diff_l < 1e-3, "left channel diff too large at {i}: {diff_l}");
            assert!(diff_r < 1e-3, "right channel diff too large at {i}: {diff_r}");
        }
    }
}
