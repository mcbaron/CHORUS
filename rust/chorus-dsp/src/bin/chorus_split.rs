/// chorus_split: process a stereo WAV through ChorusDsp and write center/only/surround stems.
///
/// Usage: chorus_split <input.wav> <output_dir> [--transform stft|wavelet|frft]
///
/// Runs 3 passes with contribution muting to isolate each stem. The estimators
/// are unaffected by output muting (they depend only on input + prototypes), so
/// all 3 passes produce identical per-contribution estimates — just different subsets summed.
use std::path::{Path, PathBuf};

use chorus_dsp::{ChorusDsp, DspConfig};
use chorus_dsp::filters::{FilterSpec, FilterChains, unity_chains};
use chorus_dsp::transforms::TransformKind;

fn load_wav_stereo(path: &Path) -> (u32, Vec<[f64; 2]>) {
    let mut reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let num_channels = spec.channels as usize;
    assert!(num_channels <= 2, "expected mono or stereo, got {num_channels} channels");

    let scale = match spec.sample_format {
        hound::SampleFormat::Float => 1.0_f64,
        hound::SampleFormat::Int => {
            1.0 / (1_i64.checked_shl(spec.bits_per_sample as u32 - 1).unwrap_or(1) as f64)
        }
    };

    let raw: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>()
            .map(|s| s.expect("read error") as f64)
            .collect(),
        hound::SampleFormat::Int => reader.samples::<i32>()
            .map(|s| s.expect("read error") as f64 * scale)
            .collect(),
    };

    let stereo: Vec<[f64; 2]> = if num_channels == 2 {
        raw.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    } else {
        raw.iter().map(|&s| [s, s]).collect()
    };

    (sample_rate, stereo)
}

fn write_wav_stereo(path: &Path, sample_rate: u32, samples: &[[f64; 2]]) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", path.display()));
    for &[l, r] in samples {
        writer.write_sample(l as f32).expect("write error");
        writer.write_sample(r as f32).expect("write error");
    }
    writer.finalize().expect("finalize error");
}

/// Build filter chains that pass only the listed contributions; all others are muted.
fn chains_for_stem(keep: &[&str]) -> FilterChains {
    let mut chains = unity_chains();
    let all = ["Lc", "Rc", "Lo", "Ro", "Ls", "Rs"];
    for name in all {
        if !keep.contains(&name) {
            chains.insert(name.to_string(), vec![FilterSpec::Mute]);
        }
    }
    chains
}

fn run_pass(input: &[[f64; 2]], config: DspConfig) -> Vec<[f64; 2]> {
    let mut dsp = ChorusDsp::new(config);
    let mut output = dsp.process(input).expect("process failed");
    // Flush latency tail with silence (2× frame_size worth)
    let flush = vec![[0.0_f64; 2]; 2048];
    output.extend(dsp.process(&flush).expect("flush failed"));
    // Trim to input length
    output.truncate(input.len());
    // Align: skip STFT warm-up (frame_size = 1024 by default) and shift
    // by re-aligning against input length after flush
    output
}

fn build_config(transform_kind: TransformKind, sample_rate: u32, chains: FilterChains) -> DspConfig {
    DspConfig {
        sample_rate,
        epsilon: 1e-9,
        filter_chains: chains,
        transform: transform_kind,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: chorus_split <input.wav> <output_dir> [--transform stft|wavelet|frft]");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);

    let transform_name = args.windows(2)
        .find(|w| w[0] == "--transform")
        .map(|w| w[1].as_str())
        .unwrap_or("stft");

    let transform_kind: TransformKind = match transform_name {
        "wavelet" => TransformKind::Wavelet {
            wavelet: chorus_dsp::transforms::WaveletKind::Db4,
            level: 3,
            frame_size: 512,
            smoothing_alpha: 0.0,
        },
        "frft" => TransformKind::Frft {
            order: 0.5,
            frame_size: 1024,
            smoothing_alpha: 0.0,
        },
        _ => TransformKind::Stft {
            frame_size: 1024,
            hop_size: 512,
            smoothing_alpha: 0.0,
        },
    };

    std::fs::create_dir_all(&output_dir)
        .unwrap_or_else(|e| panic!("cannot create output dir: {e}"));

    println!("Loading {}...", input_path.display());
    let (sample_rate, input) = load_wav_stereo(&input_path);
    println!("  {} samples @ {} Hz ({:.2}s)", input.len(), sample_rate, input.len() as f64 / sample_rate as f64);

    let stems: &[(&str, &[&str])] = &[
        ("center", &["Lc", "Rc"]),
        ("only",   &["Lo", "Ro"]),
        ("surround", &["Ls", "Rs"]),
    ];

    for (stem_name, keep) in stems {
        println!("Processing {} stem ({})...", stem_name, keep.join("+"));
        let chains = chains_for_stem(keep);
        let config = build_config(transform_kind.clone(), sample_rate, chains);
        let output = run_pass(&input, config);

        let out_path = output_dir.join(format!("{stem_name}.wav"));
        write_wav_stereo(&out_path, sample_rate, &output);
        println!("  Wrote {}", out_path.display());
    }

    println!("Done.");
}
