use chorus_dsp::{ChorusDsp, DspConfig};
use chorus_dsp::transforms::TransformKind;
use chorus_dsp::filters::unity_chains;
use std::path::Path;

/// Load a WAV file as stereo f64 samples normalized to [-1.0, 1.0].
/// Mono files are duplicated to both channels.
/// Truncates to `max_samples` stereo frames if the file is longer.
fn load_wav_stereo(path: &Path, max_samples: usize) -> Vec<[f64; 2]> {
    let mut reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("load_wav_stereo: cannot open {}: {e}", path.display()));
    let spec = reader.spec();
    let num_channels = spec.channels as usize;
    assert!(
        num_channels <= 2,
        "load_wav_stereo: expected mono or stereo, got {} channels in {}",
        num_channels,
        path.display()
    );

    assert!(
        spec.bits_per_sample > 0,
        "load_wav_stereo: bits_per_sample is 0 in {}",
        path.display()
    );

    let scale = match spec.sample_format {
        hound::SampleFormat::Float => 1.0_f64,
        hound::SampleFormat::Int => {
            1.0 / (1_i64
                .checked_shl(spec.bits_per_sample as u32 - 1)
                .unwrap_or(1) as f64)
        }
    };

    let raw: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("read error") as f64)
            .collect(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.expect("read error") as f64 * scale)
            .collect(),
    };

    let stereo: Vec<[f64; 2]> = if num_channels == 2 {
        raw.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    } else {
        raw.iter().map(|&s| [s, s]).collect()
    };

    stereo.into_iter().take(max_samples).collect()
}

fn make_config(frame_size: usize, hop_size: usize) -> DspConfig {
    DspConfig {
        sample_rate: 48_000,
        epsilon: 1e-9,
        filter_chains: unity_chains(),
        transform: TransformKind::Stft {
            frame_size,
            hop_size,
            smoothing_alpha: 0.0,
        },
    }
}

fn sanity_check(frame_size: usize, hop_size: usize) {
    let config = make_config(frame_size, hop_size);
    let mut dsp = ChorusDsp::new(config);

    // Generate 4096 samples of 440 Hz stereo sine at 48 kHz
    let n = 4096usize;
    let input: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let t = i as f64 / 48_000.0;
            let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5;
            [s, s]
        })
        .collect();

    let output = dsp
        .process(&input)
        .unwrap_or_else(|e| panic!("sanity_check: process() failed for frame_size={frame_size}: {e}"));

    assert!(
        !output.is_empty(),
        "sanity_check: no output produced for frame_size={frame_size}, hop_size={hop_size}. \
         DSP may be misconfigured."
    );

    for (i, &[l, r]) in output.iter().enumerate() {
        assert!(
            l.is_finite() && r.is_finite(),
            "sanity_check: non-finite sample at index {i}: [{l}, {r}] \
             (frame_size={frame_size}, hop_size={hop_size})"
        );
    }

    println!(
        "  [sanity OK] frame_size={frame_size} hop_size={hop_size}: {} output samples, all finite",
        output.len()
    );
}

#[derive(Debug)]
struct BenchResult {
    callback_size: usize,
    frame_size: usize,
    hop_size: usize,
    source_name: String,
    mean_us: f64,
    p95_us: f64,
    max_us: f64,
    callback_budget_us: f64,
    samples_until_first_output: usize,
    real_time_safe: bool,
}

/// Feed `audio` through `ChorusDsp` in `callback_size`-sample chunks.
/// Returns timing statistics and the sample index of first non-empty output.
fn measure_combo(
    audio: &[[f64; 2]],
    callback_size: usize,
    frame_size: usize,
    hop_size: usize,
    source_name: &str,
) -> BenchResult {
    let config = make_config(frame_size, hop_size);
    let mut dsp = ChorusDsp::new(config);

    let mut call_times_us: Vec<f64> = Vec::new();
    let mut samples_until_first_output: Option<usize> = None;
    let mut total_input = 0usize;

    for chunk in audio.chunks(callback_size) {
        let t0 = std::time::Instant::now();
        let output = dsp.process(chunk).expect("process() failed during benchmark");
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        call_times_us.push(elapsed_us);
        total_input += chunk.len();

        if samples_until_first_output.is_none() && !output.is_empty() {
            samples_until_first_output = Some(total_input);
        }
    }

    // Sort for percentile calculation (operates on a copy)
    let mut sorted = call_times_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let mean_us = call_times_us.iter().sum::<f64>() / n as f64;
    let p95_us = sorted[(n as f64 * 0.95) as usize];
    let max_us = sorted[n - 1];

    // Budget: samples_in_callback / sample_rate, in microseconds
    let callback_budget_us = callback_size as f64 / 48_000.0 * 1_000_000.0;
    let real_time_safe = p95_us <= callback_budget_us;

    BenchResult {
        callback_size,
        frame_size,
        hop_size,
        source_name: source_name.to_string(),
        mean_us,
        p95_us,
        max_us,
        callback_budget_us,
        samples_until_first_output: samples_until_first_output.unwrap_or(usize::MAX),
        real_time_safe,
    }
}

fn main() {
    println!("=== Sanity Checks ===");
    for &(fs, hs) in &[(256usize, 128usize), (512, 256), (1024, 512)] {
        sanity_check(fs, hs);
    }
    println!("All sanity checks passed.\n");
}
