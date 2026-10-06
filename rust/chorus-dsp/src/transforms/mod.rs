use serde::{Deserialize, Serialize};

pub mod frft;
pub use frft::StreamingFrft;

pub mod stft;
pub use stft::StreamingStft;

pub mod wavelet;
pub use wavelet::StreamingWavelet;

/// Periodic sqrt-Hann window: w[n]^2 + w[n+N/2]^2 = 1, so 50% overlap-add is exact.
pub fn sqrt_hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos()).sqrt())
        .collect()
}

pub trait Transform: Send {
    /// Push one stereo block. Returns processed stereo output samples
    /// when enough output is available; may return empty vec during warm-up.
    fn process_block(&mut self, input: &[[f64; 2]]) -> Vec<[f64; 2]>;
    fn reset(&mut self);
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
    /// Daubechies-4 only.
    Wavelet {
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
            smoothing_alpha: 0.9, // same default as the Python CLI
        }
    }
}
