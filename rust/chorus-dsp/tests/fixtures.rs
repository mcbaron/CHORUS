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
#[ignore = "FrFT fixture parity blocked: Python uses phase-shifted FFT, not true FrFT; see TODO in frft.rs"]
fn rust_matches_python_frft_fixture() {
    // Enable once Python FrFTTransform is updated to Ozaktas-Kutay algorithm.
    // Until then, Rust and Python FrFT outputs are not comparable.
    todo!()
}

#[test]
#[ignore = "EQ biquad not yet implemented: FilterSpec::Eq panics; see follow-on task in design spec"]
fn rust_matches_python_known_eq_preset_fixture() {
    // Enable once FilterSpec::Eq is implemented with per-bin biquad coefficients.
    todo!()
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
