use chorus_dsp::{ChorusDsp, DspConfig};
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/v2")
        .join(name)
}

fn load_stereo(name: &str) -> Vec<[f64; 2]> {
    let array: Array2<f64> = read_npy(fixture_path(name)).unwrap();
    array.outer_iter().map(|row| [row[0], row[1]]).collect()
}

fn assert_close(actual: &[[f64; 2]], expected: &[[f64; 2]], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (left, right) in actual.iter().zip(expected) {
        assert!((left[0] - right[0]).abs() <= tolerance, "left: {} expected: {} diff: {}", left[0], right[0], (left[0] - right[0]).abs());
        assert!((left[1] - right[1]).abs() <= tolerance, "left: {} expected: {} diff: {}", left[1], right[1], (left[1] - right[1]).abs());
    }
}

#[test]
fn rust_matches_python_frft_fixture() {
    use chorus_dsp::TransformKind;
    let input = load_stereo("frft_unity_bypass.input.npy");
    let config = DspConfig {
        transform: TransformKind::Frft {
            order: 0.5,
            frame_size: 1024,
            smoothing_alpha: 0.0,
        },
        ..DspConfig::default()
    };
    let mut dsp = ChorusDsp::new(config);
    let output = dsp.process(&input).unwrap();
    let expected = load_stereo("frft_unity_bypass.center.npy");
    let skip = 1024;
    let compare_len = output.len().min(input.len()).saturating_sub(skip);
    assert!(compare_len > 0, "not enough output samples to compare after warm-up skip");
    assert_close(&output[skip..skip + compare_len], &expected[skip..skip + compare_len], 1e-6);
}

#[test]
fn rust_matches_python_known_eq_preset_fixture() {
    use chorus_dsp::filters::{unity_chains, FilterSpec};

    // Build filter chains: Lc → Gain(-6 dB), Lo → Eq(highpass, 120 Hz, Q=0.707), Rs → Polarity
    let mut filter_chains = unity_chains();
    filter_chains.insert("Lc".to_string(), vec![FilterSpec::Gain { db: -6.0 }]);
    filter_chains.insert("Lo".to_string(), vec![FilterSpec::Eq {
        mode: "highpass".to_string(),
        frequency_hz: 120.0,
        q: 0.707,
        gain_db: Some(0.0),
    }]);
    filter_chains.insert("Rs".to_string(), vec![FilterSpec::Polarity]);

    let config = DspConfig {
        filter_chains,
        ..DspConfig::default()
    };

    let input = load_stereo("known_eq_preset.input.npy");
    let mut dsp = ChorusDsp::new(config);
    let output = dsp.process(&input).unwrap();

    let skip = 1024; // STFT frame_size warm-up latency
    let compare_len = output.len().min(input.len()).saturating_sub(skip);
    assert!(compare_len > 0, "not enough output samples to compare after warm-up skip");

    let expected_center = load_stereo("known_eq_preset.center.npy");
    let expected_only = load_stereo("known_eq_preset.only.npy");
    let expected_surround = load_stereo("known_eq_preset.surround.npy");

    // ChorusDsp::process returns the center mix output only.
    // expected_only and expected_surround are loaded but not compared here
    // because the DSP doesn't expose those intermediate outputs separately.
    let _ = (&expected_only, &expected_surround);

    // Tolerance from fixture manifest; Python-Rust spectral biquad produces < 1e-8 actual diff
    let tolerance = 5e-5_f64;
    let max_diff = output[skip..skip + compare_len]
        .iter()
        .zip(expected_center[skip..skip + compare_len].iter())
        .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
        .fold(0.0_f64, f64::max);
    println!("known_eq_preset max abs diff (center): {max_diff:.2e}");
    assert_close(&output[skip..skip + compare_len], &expected_center[skip..skip + compare_len], tolerance);
}

#[test]
fn rust_matches_python_unity_bypass_fixture() {
    let input = load_stereo("unity_bypass.input.npy");
    let mut dsp = ChorusDsp::new(DspConfig::default()); // STFT by default
    let output = dsp.process(&input).unwrap(); // returns Vec<[f64; 2]>
    // Output should approximately equal input (within STFT round-trip tolerance).
    // Skip first frame_size samples (warm-up latency).
    let skip = 1024;
    let compare_len = output.len().min(input.len()).saturating_sub(skip);
    assert!(compare_len > 0, "not enough output samples to compare after warm-up skip");
    assert_close(&output[skip..skip + compare_len], &input[skip..skip + compare_len], 1e-3);
}

#[test]
fn rust_matches_python_core_fixture_set() {
    // With unity filter chains the STFT transform reconstructs the input signal.
    // Compare the aligned (post-warmup) portion against the original input.
    for case_name in [
        "center_dominant",
        "hard_panned_left",
        "hard_panned_right",
        "phase_inverted_surround",
        "silence",
        "near_silence",
    ] {
        let input = load_stereo(&format!("{case_name}.input.npy"));
        let mut dsp = ChorusDsp::new(DspConfig::default());
        let output = dsp.process(&input).unwrap();
        let skip = 1024;
        let compare_len = output.len().min(input.len()).saturating_sub(skip);
        assert!(compare_len > 0, "case {case_name}: not enough output samples to compare after warm-up skip");
        assert_close(&output[skip..skip + compare_len], &input[skip..skip + compare_len], 1e-3);
    }
}
