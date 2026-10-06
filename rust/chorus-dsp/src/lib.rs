pub mod estimation;
pub mod filters;
pub mod processor;
pub mod prototypes;
pub mod transforms;

pub use processor::{ChorusDsp, ChorusError, DspConfig};
pub use transforms::{Transform, TransformKind};

/// Read a WAV file as stereo samples in [-1, 1]. A mono file goes to both channels.
pub fn load_wav_stereo(path: impl AsRef<std::path::Path>) -> Result<(u32, Vec<[f64; 2]>), hound::Error> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let raw: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.map(f64::from)).collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1_i64 << (spec.bits_per_sample - 1)) as f64;
            reader.samples::<i32>().map(|s| s.map(|v| v as f64 * scale)).collect::<Result<_, _>>()?
        }
    };
    let stereo = match spec.channels {
        1 => raw.iter().map(|&s| [s, s]).collect(),
        2 => raw.chunks_exact(2).map(|c| [c[0], c[1]]).collect(),
        _ => return Err(hound::Error::Unsupported),
    };
    Ok((spec.sample_rate, stereo))
}
