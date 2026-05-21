use crate::estimation::SmoothedScalarEstimator;
use crate::filters::{apply_chains, unity_chains, FilterChains};
use crate::prototypes::{center_prototype, surround_prototype};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspConfig {
    pub sample_rate: u32,
    pub smoothing_alpha: f64,
    pub epsilon: f64,
    pub filter_chains: FilterChains,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            smoothing_alpha: 0.0,
            epsilon: 1e-9,
            filter_chains: unity_chains(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChorusOutput {
    pub center: Vec<[f64; 2]>,
    pub only: Vec<[f64; 2]>,
    pub surround: Vec<[f64; 2]>,
}

#[derive(Debug, Error)]
pub enum ChorusError {
    #[error("input must contain at least one stereo frame")]
    EmptyInput,
}

pub struct ChorusDsp {
    config: DspConfig,
}

impl ChorusDsp {
    pub fn new(config: DspConfig) -> Self {
        Self { config }
    }

    pub fn process(&mut self, input: &[[f64; 2]]) -> Result<ChorusOutput, ChorusError> {
        if input.is_empty() {
            return Err(ChorusError::EmptyInput);
        }
        let left: Vec<f64> = input.iter().map(|frame| frame[0]).collect();
        let right: Vec<f64> = input.iter().map(|frame| frame[1]).collect();
        let center_proto = center_prototype(&left, &right);
        let surround_proto = surround_prototype(&left, &right);
        let mut left_center_estimator =
            SmoothedScalarEstimator::new(self.config.smoothing_alpha, self.config.epsilon);
        let mut right_center_estimator =
            SmoothedScalarEstimator::new(self.config.smoothing_alpha, self.config.epsilon);
        let mut left_surround_estimator =
            SmoothedScalarEstimator::new(self.config.smoothing_alpha, self.config.epsilon);
        let mut right_surround_estimator =
            SmoothedScalarEstimator::new(self.config.smoothing_alpha, self.config.epsilon);
        let lc = left_center_estimator.estimate(&center_proto, &left);
        let rc = right_center_estimator.estimate(&center_proto, &right);
        let ls = left_surround_estimator.estimate(&surround_proto, &left);
        let mut rs = right_surround_estimator.estimate(&surround_proto, &right);
        for sample in &mut rs {
            *sample = -*sample;
        }
        let lo: Vec<f64> = left.iter().zip(&lc).zip(&ls).map(|((source, c), s)| source - c - s).collect();
        let ro: Vec<f64> = right.iter().zip(&rc).zip(&rs).map(|((source, c), s)| source - c - s).collect();
        let contributions = BTreeMap::from([
            ("Lc".to_string(), lc),
            ("Rc".to_string(), rc),
            ("Lo".to_string(), lo),
            ("Ro".to_string(), ro),
            ("Ls".to_string(), ls),
            ("Rs".to_string(), rs),
        ]);
        let filtered = apply_chains(&contributions, &self.config.filter_chains);
        Ok(ChorusOutput {
            center: pair(&filtered["Lc"], &filtered["Rc"]),
            only: pair(&filtered["Lo"], &filtered["Ro"]),
            surround: pair(&filtered["Ls"], &filtered["Rs"]),
        })
    }
}

fn pair(left: &[f64], right: &[f64]) -> Vec<[f64; 2]> {
    left.iter().zip(right).map(|(l, r)| [*l, *r]).collect()
}

#[cfg(test)]
mod tests {
    use super::{ChorusDsp, DspConfig};

    #[test]
    fn unity_bypass_reconstructs_input() {
        let input = vec![[0.25, 0.25], [0.0, 0.0], [-0.25, -0.25]];
        let mut dsp = ChorusDsp::new(DspConfig::default());
        let output = dsp.process(&input).unwrap();
        for index in 0..input.len() {
            assert!((output.center[index][0] + output.only[index][0] + output.surround[index][0] - input[index][0]).abs() < 1e-9);
            assert!((output.center[index][1] + output.only[index][1] + output.surround[index][1] - input[index][1]).abs() < 1e-9);
        }
    }
}
