/// chorus_split: process a stereo WAV through ChorusDsp and write center/only/surround stems.
///
/// Usage: chorus_split <input.wav> <output_dir> [--transform stft|wavelet|frft]
///
/// Runs 3 passes with contribution muting to isolate each stem. The estimators
/// are unaffected by output muting (they depend only on input + prototypes), so
/// all 3 passes produce identical per-contribution estimates — just different subsets summed.
use std::path::{Path, PathBuf};

use chorus_dsp::{load_wav_stereo, ChorusDsp, DspConfig};
use chorus_dsp::filters::{FilterSpec, FilterChains, unity_chains};
use chorus_dsp::transforms::TransformKind;

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
    output.truncate(input.len());
    output
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
    let (sample_rate, input) = load_wav_stereo(&input_path)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", input_path.display()));
    println!("  {} samples @ {} Hz ({:.2}s)", input.len(), sample_rate, input.len() as f64 / sample_rate as f64);

    let stems: &[(&str, &[&str])] = &[
        ("center", &["Lc", "Rc"]),
        ("only",   &["Lo", "Ro"]),
        ("surround", &["Ls", "Rs"]),
    ];

    for (stem_name, keep) in stems {
        println!("Processing {} stem ({})...", stem_name, keep.join("+"));
        let chains = chains_for_stem(keep);
        let config = DspConfig {
            sample_rate,
            epsilon: 1e-9,
            filter_chains: chains,
            transform: transform_kind.clone(),
        };
        let output = run_pass(&input, config);

        let out_path = output_dir.join(format!("{stem_name}.wav"));
        write_wav_stereo(&out_path, sample_rate, &output);
        println!("  Wrote {}", out_path.display());
    }

    println!("Done.");
}
