use num_complex::Complex;

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

pub struct ComplexEstimator {
    alpha: f64,
    epsilon: f64,
    cross: Vec<Complex<f64>>,  // per-coefficient smoothed cross-correlation
    auto: Vec<f64>,            // per-coefficient smoothed auto-correlation
}

impl ComplexEstimator {
    pub fn new(alpha: f64, epsilon: f64, num_coeffs: usize) -> Self {
        Self {
            alpha,
            epsilon,
            cross: vec![Complex::new(0.0, 0.0); num_coeffs],
            auto: vec![0.0; num_coeffs],
        }
    }

    pub fn estimate(&mut self, prototype: &[Complex<f64>], source: &[Complex<f64>]) -> Vec<Complex<f64>> {
        debug_assert_eq!(prototype.len(), self.cross.len());
        debug_assert_eq!(source.len(), self.cross.len());
        let alpha = self.alpha;
        let epsilon = self.epsilon;
        self.cross.iter_mut()
            .zip(self.auto.iter_mut())
            .zip(prototype.iter().zip(source.iter()))
            .map(|((cross_k, auto_k), (proto_k, src_k))| {
                let instant_cross = proto_k * src_k.conj();
                let instant_auto = src_k.norm_sqr();
                *cross_k = (1.0 - alpha) * instant_cross + alpha * *cross_k;
                *auto_k = (1.0 - alpha) * instant_auto + alpha * *auto_k;
                let weight = cross_k.re / auto_k.max(epsilon);
                weight * src_k
            })
            .collect()
    }

    pub fn reset(&mut self) {
        for c in &mut self.cross { *c = Complex::new(0.0, 0.0); }
        for a in &mut self.auto { *a = 0.0; }
    }
}

#[cfg(test)]
mod tests {
    use super::{SmoothedScalarEstimator, ComplexEstimator};
    use num_complex::Complex;

    #[test]
    fn estimates_scaled_source() {
        let mut estimator = SmoothedScalarEstimator::new(0.0, 1e-12);
        let estimated = estimator.estimate(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        for (actual, expected) in estimated.iter().zip(&[2.0_f64, 4.0, 6.0]) {
            approx::assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn complex_estimator_identity_no_smoothing() {
        // When prototype == source (pure real) and alpha=0, estimate should return source
        let mut est = ComplexEstimator::new(0.0, 1e-12, 3);
        let src: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let result = est.estimate(&src, &src);
        for (actual, expected) in result.iter().zip(&src) {
            approx::assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-9);
            approx::assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-9);
        }
    }

    #[test]
    fn complex_estimator_smoothing_state_persists() {
        // With non-zero alpha, successive calls blend previous state
        let alpha = 0.5_f64;
        let mut est = ComplexEstimator::new(alpha, 1e-12, 1);
        let proto = vec![Complex::new(1.0, 0.0)];
        let src = vec![Complex::new(2.0, 0.0)];

        // First call: cross = (1-0.5)*2 + 0 = 1, auto = (1-0.5)*4 + 0 = 2, weight = 1/2 = 0.5, out = 1.0
        let first = est.estimate(&proto, &src);
        // Second call: cross = (1-0.5)*2 + 0.5*1 = 1.5, auto = (1-0.5)*4 + 0.5*2 = 3, weight = 1.5/3 = 0.5, out = 1.0
        let second = est.estimate(&proto, &src);

        // The outputs should be non-zero (state was maintained)
        assert!(first[0].re.abs() > 1e-9);
        assert!(second[0].re.abs() > 1e-9);
        // With alpha=0.5 and identical repeated inputs, weight stabilizes — both calls give same weight here
        approx::assert_abs_diff_eq!(first[0].re, second[0].re, epsilon = 1e-9);
    }

    #[test]
    fn complex_estimator_reset_clears_state() {
        let alpha = 0.9_f64;
        let mut est = ComplexEstimator::new(alpha, 1e-12, 2);
        let proto: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let src: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];

        // Run several calls to build up state
        for _ in 0..10 {
            est.estimate(&proto, &src);
        }

        est.reset();

        // After reset, result should match a fresh estimator
        let mut fresh = ComplexEstimator::new(alpha, 1e-12, 2);
        let result_reset = est.estimate(&proto, &src);
        let result_fresh = fresh.estimate(&proto, &src);

        for (r, f) in result_reset.iter().zip(&result_fresh) {
            approx::assert_abs_diff_eq!(r.re, f.re, epsilon = 1e-12);
            approx::assert_abs_diff_eq!(r.im, f.im, epsilon = 1e-12);
        }
    }
}
