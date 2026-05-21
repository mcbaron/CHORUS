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
fn rust_matches_python_unity_bypass_fixture() {
    let input = load_stereo("unity_bypass.input.npy");
    let expected_center = load_stereo("unity_bypass.center.npy");
    let expected_only = load_stereo("unity_bypass.only.npy");
    let expected_surround = load_stereo("unity_bypass.surround.npy");
    let mut dsp = ChorusDsp::new(DspConfig::default());

    let output = dsp.process(&input).unwrap();

    assert_close(&output.center, &expected_center, 1e-6);
    assert_close(&output.only, &expected_only, 1e-6);
    assert_close(&output.surround, &expected_surround, 1e-6);
}

#[test]
fn rust_matches_python_core_fixture_set() {
    for case_name in [
        "center_dominant",
        "hard_panned_left",
        "hard_panned_right",
        "phase_inverted_surround",
        "silence",
        "near_silence",
    ] {
        let input = load_stereo(&format!("{case_name}.input.npy"));
        let expected_center = load_stereo(&format!("{case_name}.center.npy"));
        let expected_only = load_stereo(&format!("{case_name}.only.npy"));
        let expected_surround = load_stereo(&format!("{case_name}.surround.npy"));
        let mut dsp = ChorusDsp::new(DspConfig::default());
        let output = dsp.process(&input).unwrap();
        assert_close(&output.center, &expected_center, 1e-6);
        assert_close(&output.only, &expected_only, 1e-6);
        assert_close(&output.surround, &expected_surround, 1e-6);
    }
}
