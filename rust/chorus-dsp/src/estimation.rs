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

#[derive(Debug, Clone)]
pub struct ComplexEstimator {
    alpha: f64,
    epsilon: f64,
    cross: Vec<Complex<f64>>,  // per-coefficient smoothed cross-correlation
    auto: Vec<f64>,            // per-coefficient smoothed auto-correlation
    buf: Vec<Complex<f64>>,    // pre-allocated output buffer
}

impl ComplexEstimator {
    pub fn new(alpha: f64, epsilon: f64, num_coeffs: usize) -> Self {
        Self {
            alpha,
            epsilon,
            cross: vec![Complex::new(0.0, 0.0); num_coeffs],
            auto: vec![0.0; num_coeffs],
            buf: vec![Complex::new(0.0, 0.0); num_coeffs],
        }
    }

    pub fn estimate(&mut self, prototype: &[Complex<f64>], source: &[Complex<f64>]) -> &[Complex<f64>] {
        debug_assert_eq!(prototype.len(), self.cross.len());
        debug_assert_eq!(source.len(), self.cross.len());
        let alpha = self.alpha;
        let epsilon = self.epsilon;
        for (((cross_k, auto_k), buf_k), (proto_k, src_k)) in self.cross.iter_mut()
            .zip(self.auto.iter_mut())
            .zip(self.buf.iter_mut())
            .zip(prototype.iter().zip(source.iter()))
        {
            let instant_cross = proto_k * src_k.conj();
            let instant_auto = src_k.norm_sqr();
            *cross_k = (1.0 - alpha) * instant_cross + alpha * *cross_k;
            *auto_k = (1.0 - alpha) * instant_auto + alpha * *auto_k;
            let weight = cross_k.re / auto_k.max(epsilon);
            *buf_k = weight * src_k;
        }
        &self.buf
    }

    pub fn reset(&mut self) {
        for c in &mut self.cross { *c = Complex::new(0.0, 0.0); }
        for a in &mut self.auto { *a = 0.0; }
        for b in &mut self.buf  { *b = Complex::new(0.0, 0.0); }
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
        let result = est.estimate(&src, &src).to_vec();
        for (actual, expected) in result.iter().zip(&src) {
            approx::assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-9);
            approx::assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-9);
        }
    }

    #[test]
    fn complex_estimator_smoothing_state_persists() {
        let alpha = 0.9_f64;
        let mut est = ComplexEstimator::new(alpha, 1e-12, 1);
        let proto1 = vec![Complex::new(1.0, 0.0)];
        let src1 = vec![Complex::new(2.0, 0.0)];
        let proto2 = vec![Complex::new(1.0, 0.0)];
        let src2 = vec![Complex::new(1.0, 0.0)];

        // First call with proto=1, src=2
        let _first: Vec<_> = est.estimate(&proto1, &src1).to_vec();

        // Second call with proto=1, src=1 (different input, state carries over)
        let second: Vec<_> = est.estimate(&proto2, &src2).to_vec();

        // Fresh estimator (no prior state) on same second input
        let mut fresh = ComplexEstimator::new(alpha, 1e-12, 1);
        let fresh_first: Vec<_> = fresh.estimate(&proto2, &src2).to_vec();

        // second != fresh_first (state was carried over from first call, affecting second)
        assert!((second[0].re - fresh_first[0].re).abs() > 1e-9,
            "second call with state should differ from fresh estimator with same input: {} != {}", second[0].re, fresh_first[0].re);
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
        let result_reset = est.estimate(&proto, &src).to_vec();
        let result_fresh = fresh.estimate(&proto, &src).to_vec();

        for (r, f) in result_reset.iter().zip(&result_fresh) {
            approx::assert_abs_diff_eq!(r.re, f.re, epsilon = 1e-12);
            approx::assert_abs_diff_eq!(r.im, f.im, epsilon = 1e-12);
        }
    }
}
