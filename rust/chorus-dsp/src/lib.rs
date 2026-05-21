pub mod estimation;
pub mod filters;
pub mod processor;
pub mod prototypes;
pub mod transforms;

pub use processor::{ChorusDsp, ChorusError, ChorusOutput, DspConfig};
