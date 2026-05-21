use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
}
