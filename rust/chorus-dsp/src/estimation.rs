#[derive(Debug, Clone)]
pub struct SmoothedScalarEstimator {
    alpha: f64,
    epsilon: f64,
    previous: f64,
}

impl SmoothedScalarEstimator {
    pub fn new(alpha: f64, epsilon: f64) -> Self {
        Self {
            alpha,
            epsilon,
            previous: 0.0,
        }
    }

    pub fn estimate(&mut self, prototype: &[f64], source: &[f64]) -> Vec<f64> {
        let numerator: f64 = prototype.iter().zip(source).map(|(p, s)| p * s).sum();
        let denominator: f64 = prototype.iter().map(|p| p * p).sum::<f64>() + self.epsilon;
        let instant = numerator / denominator;
        let weight = self.alpha * self.previous + (1.0 - self.alpha) * instant;
        self.previous = weight;
        prototype.iter().map(|p| p * weight).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::SmoothedScalarEstimator;

    #[test]
    fn estimates_scaled_source() {
        let mut estimator = SmoothedScalarEstimator::new(0.0, 1e-12);
        let estimated = estimator.estimate(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        for (actual, expected) in estimated.iter().zip(&[2.0_f64, 4.0, 6.0]) {
            approx::assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
        }
    }
}
