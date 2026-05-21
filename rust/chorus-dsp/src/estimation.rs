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
        let expected = vec![2.0, 4.0, 6.0];
        for (e, a) in estimated.iter().zip(expected.iter()) {
            assert!((e - a).abs() < 1e-9, "expected {}, got {}", a, e);
        }
    }
}
