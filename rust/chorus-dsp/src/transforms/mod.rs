use serde::{Deserialize, Serialize};

pub mod frft;
pub use frft::StreamingFrft;

pub mod stft;
pub use stft::StreamingStft;

pub mod wavelet;
pub use wavelet::StreamingWavelet;

pub trait Transform: Send {
    /// Push one stereo block. Returns processed stereo output samples
    /// when enough output is available; may return empty vec during warm-up.
    fn process_block(&mut self, input: &[[f64; 2]]) -> Vec<[f64; 2]>;
    fn reset(&mut self);
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum WaveletKind {
    #[default]
    #[serde(rename = "db4")]
    Db4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformKind {
    Stft {
        frame_size: usize,
        hop_size: usize,
        smoothing_alpha: f64,
    },
    Frft {
        order: f64,
        frame_size: usize,
        smoothing_alpha: f64,
    },
    Wavelet {
        wavelet: WaveletKind,
        level: usize,
        frame_size: usize,
        smoothing_alpha: f64,
    },
}

impl Default for TransformKind {
    fn default() -> Self {
        TransformKind::Stft {
            frame_size: 1024,
            hop_size: 512,
            smoothing_alpha: 0.0,
        }
    }
}
