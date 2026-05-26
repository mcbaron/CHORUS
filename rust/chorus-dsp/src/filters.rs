use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::PI;

const CONTRIBUTIONS: [&str; 6] = ["Lc", "Rc", "Lo", "Ro", "Ls", "Rs"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum FilterSpec {
    #[serde(rename = "unity")]
    Unity,
    #[serde(rename = "gain")]
    Gain { db: f64 },
    #[serde(rename = "mute")]
    Mute,
    #[serde(rename = "solo")]
    Solo,
    #[serde(rename = "polarity")]
    Polarity,
    #[serde(rename = "eq")]
    Eq { mode: String, frequency_hz: f64, q: f64, gain_db: Option<f64> },
}

pub type FilterChains = BTreeMap<String, Vec<FilterSpec>>;

pub fn unity_chains() -> FilterChains {
    CONTRIBUTIONS
        .iter()
        .map(|name| (name.to_string(), vec![FilterSpec::Unity]))
        .collect()
}

pub fn apply_chains(contributions: &BTreeMap<String, Vec<f64>>, chains: &FilterChains) -> BTreeMap<String, Vec<f64>> {
    let soloed: Vec<String> = chains
        .iter()
        .filter(|(_, chain)| chain.iter().any(|spec| matches!(spec, FilterSpec::Solo)))
        .map(|(name, _)| name.clone())
        .collect();
    let mut output = BTreeMap::new();
    for name in CONTRIBUTIONS {
        let mut audio = contributions.get(name).cloned().unwrap_or_default();
        let default_chain = vec![FilterSpec::Unity];
        let chain = chains.get(name).unwrap_or(&default_chain);
        for spec in chain {
            match spec {
                FilterSpec::Unity | FilterSpec::Solo => {}
                FilterSpec::Gain { db } => {
                    let scale = 10.0_f64.powf(db / 20.0);
                    for sample in &mut audio {
                        *sample *= scale;
                    }
                }
                FilterSpec::Mute => audio.fill(0.0),
                FilterSpec::Polarity => {
                    for sample in &mut audio {
                        *sample = -*sample;
                    }
                }
                FilterSpec::Eq { .. } => {
                    // Implement this to match Python v2 EQ fixtures before running parity tests.
                    panic!("Rust EQ filter must be implemented against Python v2 fixtures")
                }
            }
        }
        if !soloed.is_empty() && !soloed.iter().any(|solo| solo == name) {
            audio.fill(0.0);
        }
        output.insert(name.to_string(), audio);
    }
    output
}

/// Returns the effective scalar multiplier for a single contribution after applying its filter chain.
/// `soloed` should be the pre-computed list of soloed contribution names (empty = none soloed).
pub fn chain_scalar(chains: &FilterChains, name: &str, soloed: &[String]) -> f64 {
    // If any solo exists and this contribution is not soloed, mute it
    if !soloed.is_empty() && !soloed.iter().any(|s| s == name) {
        return 0.0;
    }

    let default_chain = vec![FilterSpec::Unity];
    let chain = chains.get(name).unwrap_or(&default_chain);

    let mut scalar = 1.0_f64;
    for spec in chain {
        match spec {
            FilterSpec::Unity | FilterSpec::Solo => {}
            FilterSpec::Gain { db } => {
                scalar *= 10.0_f64.powf(db / 20.0);
            }
            FilterSpec::Mute => {
                scalar = 0.0;
            }
            FilterSpec::Polarity => {
                scalar = -scalar;
            }
            FilterSpec::Eq { .. } => {
                unimplemented!("EQ not yet supported in chain_scalar");
            }
        }
    }
    scalar
}

/// Apply a filter chain to a mutable slice of complex spectral bins in-place.
/// Returns false if the contribution should be zeroed (muted or not soloed).
pub fn apply_chain_to_bins(
    bins: &mut [Complex<f64>],
    chain: &[FilterSpec],
    sample_rate: u32,
    soloed: &[String],
    name: &str,
) -> bool {
    if !soloed.is_empty() && !soloed.iter().any(|s| s == name) {
        return false;
    }

    for spec in chain {
        match spec {
            FilterSpec::Unity | FilterSpec::Solo => {}
            FilterSpec::Gain { db } => {
                let scale = 10.0_f64.powf(db / 20.0);
                for bin in bins.iter_mut() {
                    *bin *= scale;
                }
            }
            FilterSpec::Mute => {
                return false;
            }
            FilterSpec::Polarity => {
                for bin in bins.iter_mut() {
                    *bin = -*bin;
                }
            }
            FilterSpec::Eq { mode, frequency_hz, q, gain_db } => {
                let gdb = gain_db.unwrap_or(0.0);
                let h_bins = eq_frequency_response(mode, *frequency_hz, *q, gdb, sample_rate, bins.len());
                for (bin, h) in bins.iter_mut().zip(h_bins.iter()) {
                    *bin *= h;
                }
            }
        }
    }
    true
}

/// Compute per-bin complex frequency response H(e^{jω}) for a biquad EQ filter.
///
/// `bins_count` = frame_size / 2 + 1 (number of one-sided RFFT bins).
/// Bin k corresponds to ω_k = 2π·k / (2·(bins_count − 1)).
///
/// Modes: "highpass" and "lowpass" use 2nd-order Butterworth via bilinear transform.
/// "peaking" (default) uses iirpeak with H_total = 1 + H_peak·(A − 1).
pub fn eq_frequency_response(
    mode: &str,
    frequency_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: u32,
    bins_count: usize,
) -> Vec<Complex<f64>> {
    let fs = sample_rate as f64;
    let n = 2 * (bins_count.saturating_sub(1)).max(1);
    let mut h = Vec::with_capacity(bins_count);

    // Compute coefficients based on mode, then evaluate per-bin
    match mode {
        "highpass" | "lowpass" => {
            let kw = (PI * frequency_hz / fs).tan();
            let denom = 1.0 + 2.0_f64.sqrt() * kw + kw * kw;
            let (b0, b1, b2, a1, a2) = if mode == "highpass" {
                let b0 = 1.0 / denom;
                (b0, -2.0 * b0, b0, 2.0 * (kw * kw - 1.0) / denom, (1.0 - 2.0_f64.sqrt() * kw + kw * kw) / denom)
            } else {
                let b0 = kw * kw / denom;
                (b0, 2.0 * b0, b0, 2.0 * (kw * kw - 1.0) / denom, (1.0 - 2.0_f64.sqrt() * kw + kw * kw) / denom)
            };
            for k in 0..bins_count {
                let omega = 2.0 * PI * k as f64 / n as f64;
                let z_inv = Complex::from_polar(1.0, -omega);
                let z_inv2 = Complex::from_polar(1.0, -2.0 * omega);
                let num = b0 + b1 * z_inv + b2 * z_inv2;
                let den = Complex::new(1.0, 0.0) + a1 * z_inv + a2 * z_inv2;
                h.push(num / den);
            }
        }
        _ => {
            // peaking: H_total = 1 + H_peak * (A - 1), where A is linear gain
            let w0 = 2.0 * PI * frequency_hz / fs;
            let t_bw2 = (w0 / q / 2.0).tan();
            let a0 = 1.0 + t_bw2;
            let b0_peak = t_bw2 / a0;
            let b2_peak = -t_bw2 / a0;
            let a1_peak = -2.0 * w0.cos() / a0;
            let a2_peak = (1.0 - t_bw2) / a0;
            let a_lin = 10.0_f64.powf(gain_db / 20.0);
            for k in 0..bins_count {
                let omega = 2.0 * PI * k as f64 / n as f64;
                let z_inv = Complex::from_polar(1.0, -omega);
                let z_inv2 = Complex::from_polar(1.0, -2.0 * omega);
                let num_peak = b0_peak + b2_peak * z_inv2;
                let den_peak = Complex::new(1.0, 0.0) + a1_peak * z_inv + a2_peak * z_inv2;
                h.push(Complex::new(1.0, 0.0) + (num_peak / den_peak) * (a_lin - 1.0));
            }
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::{apply_chains, unity_chains, FilterSpec};

    #[test]
    fn applies_gain_and_polarity() {
        let contributions = ["Lc", "Rc", "Lo", "Ro", "Ls", "Rs"]
            .into_iter()
            .map(|name| (name.to_string(), vec![1.0, 1.0]))
            .collect();
        let mut chains = unity_chains();
        chains.insert("Lc".to_string(), vec![FilterSpec::Gain { db: 6.0 }]);
        chains.insert("Rs".to_string(), vec![FilterSpec::Polarity]);
        let output = apply_chains(&contributions, &chains);
        assert!(output["Lc"][0] > 1.99);
        assert_eq!(output["Rs"], vec![-1.0, -1.0]);
        assert_eq!(output["Rc"], vec![1.0, 1.0]);
    }

    #[test]
    fn eq_highpass_attenuates_below_cutoff() {
        let sample_rate = 48_000u32;
        let bins_count = 1024usize;
        let h = super::eq_frequency_response("highpass", 1000.0, 0.707, 0.0, sample_rate, bins_count);
        // DC bin is exactly 0 for any highpass
        assert!(
            h[0].norm() < 0.1,
            "DC bin |H|={} should be < 0.1 for highpass at 1000 Hz",
            h[0].norm()
        );
        // Bin 5 ≈ 117 Hz — well below the 1000 Hz cutoff, so a 2nd-order Butterworth
        // attenuates this deeply (|H| << 0.1)
        assert!(
            h[5].norm() < 0.1,
            "Bin 5 |H|={} should be < 0.1 for highpass at 1000 Hz",
            h[5].norm()
        );
        // Passband: near-Nyquist bin should be close to 1.0
        let nyquist_bin = bins_count - 1;
        assert!(
            h[nyquist_bin].norm() > 0.99,
            "Nyquist bin |H|={} should be ≈ 1.0 for highpass",
            h[nyquist_bin].norm()
        );
    }

    #[test]
    fn eq_lowpass_attenuates_above_cutoff() {
        let sample_rate = 48_000u32;
        let bins_count = 1024usize;
        let h = super::eq_frequency_response("lowpass", 1000.0, 0.707, 0.0, sample_rate, bins_count);
        // DC should pass through (near 1.0)
        assert!(
            (h[0].norm() - 1.0).abs() < 0.01,
            "DC bin |H|={} should be ≈ 1.0 for lowpass",
            h[0].norm()
        );
        // Near-Nyquist bin should be strongly attenuated
        let nyquist_bin = bins_count - 1;
        assert!(
            h[nyquist_bin].norm() < 0.1,
            "Nyquist bin |H|={} should be < 0.1 for lowpass at 1000 Hz",
            h[nyquist_bin].norm()
        );
    }

    #[test]
    fn eq_peaking_boosts_center_bin() {
        let sample_rate = 48_000u32;
        let bins_count = 4096usize;
        let h = super::eq_frequency_response("peaking", 1000.0, 10.0, 6.0, sample_rate, bins_count);
        let peak_norm = h.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        let expected_a = 10.0_f64.powf(6.0 / 20.0);
        assert!(
            (peak_norm - expected_a).abs() < 0.1,
            "Peak |H|={} should be ≈ {} (6 dB linear)",
            peak_norm, expected_a
        );
    }

    #[test]
    fn eq_chain_compose_multiplies_responses() {
        let sample_rate = 48_000u32;
        let bins_count = 512usize;
        let h1 = super::eq_frequency_response("highpass", 500.0, 0.707, 0.0, sample_rate, bins_count);
        let h2 = super::eq_frequency_response("highpass", 1000.0, 0.707, 0.0, sample_rate, bins_count);
        // Use bin 5 (≈235 Hz with these params) — nonzero but attenuated by both filters.
        // DC (bin 0) is exactly 0 for highpass, making 0 < 0 comparisons vacuously false.
        let combined = h1[5] * h2[5];
        assert!(
            combined.norm() < h1[5].norm(),
            "Combined bin 5 |H|={} should be less than h1 alone |H|={}",
            combined.norm(), h1[5].norm()
        );
        assert!(
            combined.norm() < h2[5].norm(),
            "Combined bin 5 |H|={} should be less than h2 alone |H|={}",
            combined.norm(), h2[5].norm()
        );
    }
}
