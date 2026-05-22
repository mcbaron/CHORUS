use chorus_dsp::{ChorusDsp, DspConfig};
use chorus_dsp::transforms::TransformKind;
use chorus_dsp::filters::unity_chains;

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

fn main() {
    println!("=== Sanity Checks ===");
    for &(fs, hs) in &[(256usize, 128usize), (512, 256), (1024, 512)] {
        sanity_check(fs, hs);
    }
    println!("All sanity checks passed.\n");
}
