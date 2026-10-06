use num_complex::Complex;

use crate::filters::{apply_chain_to_bins, apply_chain_to_real_signal, FilterChains, FilterSpec};
use crate::prototypes::{center_prototype, surround_prototype};

#[derive(Debug, Clone)]
pub struct ComplexEstimator {
    alpha: f64,
    epsilon: f64,
    cross: Vec<Complex<f64>>,  // per-coefficient smoothed cross-correlation
    auto: Vec<f64>,            // per-coefficient smoothed auto-correlation
    buf: Vec<Complex<f64>>,    // pre-allocated output buffer
    seeded: bool,              // false until the first frame sets the state
}

impl ComplexEstimator {
    pub fn new(alpha: f64, epsilon: f64, num_coeffs: usize) -> Self {
        Self {
            alpha,
            epsilon,
            cross: vec![Complex::new(0.0, 0.0); num_coeffs],
            auto: vec![0.0; num_coeffs],
            buf: vec![Complex::new(0.0, 0.0); num_coeffs],
            seeded: false,
        }
    }

    pub fn estimate(&mut self, prototype: &[Complex<f64>], source: &[Complex<f64>]) -> &[Complex<f64>] {
        debug_assert_eq!(prototype.len(), self.cross.len());
        debug_assert_eq!(source.len(), self.cross.len());
        // The first frame sets the state directly, as in the Python reference.
        let alpha = if self.seeded { self.alpha } else { 0.0 };
        self.seeded = true;
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
        self.seeded = false;
    }
}

const UNITY: &[FilterSpec] = &[FilterSpec::Unity];

/// The prototype, estimation, and filter steps that every transform shares.
pub struct Splitter {
    estimators: [ComplexEstimator; 4], // Lc, Rc, Ls, Rs
    filter_chains: FilterChains,
    soloed: Vec<String>,
    sample_rate: u32,
}

impl Splitter {
    pub fn new(smoothing_alpha: f64, epsilon: f64, num_coeffs: usize, sample_rate: u32, filter_chains: FilterChains) -> Self {
        let soloed = filter_chains
            .iter()
            .filter(|(_, chain)| chain.iter().any(|spec| matches!(spec, FilterSpec::Solo)))
            .map(|(name, _)| name.clone())
            .collect();
        Self {
            estimators: std::array::from_fn(|_| ComplexEstimator::new(smoothing_alpha, epsilon, num_coeffs)),
            filter_chains,
            soloed,
            sample_rate,
        }
    }

    /// Returns the unfiltered contributions in `filters::CONTRIBUTIONS` order.
    pub fn split(&mut self, left: &[Complex<f64>], right: &[Complex<f64>]) -> [Vec<Complex<f64>>; 6] {
        let center = center_prototype(left, right);
        let surround = surround_prototype(left, right);
        let [lc_est, rc_est, ls_est, rs_est] = &mut self.estimators;
        let lc = lc_est.estimate(&center, left).to_vec();
        let rc = rc_est.estimate(&center, right).to_vec();
        let ls = ls_est.estimate(&surround, left).to_vec();
        // S is in phase with L, so its estimate from R has the opposite sign of R. Negate it.
        let rs: Vec<Complex<f64>> = rs_est.estimate(&surround, right).iter().map(|c| -c).collect();
        let lo = (0..left.len()).map(|k| left[k] - lc[k] - ls[k]).collect();
        let ro = (0..right.len()).map(|k| right[k] - rc[k] - rs[k]).collect();
        [lc, rc, lo, ro, ls, rs]
    }

    fn chain(&self, name: &str) -> &[FilterSpec] {
        self.filter_chains.get(name).map_or(UNITY, Vec::as_slice)
    }

    pub fn filter_bins(&self, name: &str, mut bins: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        if apply_chain_to_bins(&mut bins, self.chain(name), self.sample_rate, &self.soloed, name) {
            bins
        } else {
            vec![Complex::new(0.0, 0.0); bins.len()]
        }
    }

    pub fn filter_real(&self, name: &str, signal: &[f64]) -> Vec<f64> {
        apply_chain_to_real_signal(signal, self.chain(name), self.sample_rate, &self.soloed, name)
    }

    pub fn reset(&mut self) {
        self.estimators.iter_mut().for_each(ComplexEstimator::reset);
    }
}

#[cfg(test)]
mod tests {
    use super::ComplexEstimator;
    use num_complex::Complex;

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
    fn opposite_polarity_goes_to_surround_with_input_sign() {
        // L = x, R = -x: all of it is surround, so Ls = L, Rs = R, and Lo = Ro = 0.
        let left: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.5), Complex::new(-2.0, 0.25)];
        let right: Vec<Complex<f64>> = left.iter().map(|c| -c).collect();
        let mut splitter = super::Splitter::new(0.0, 1e-12, 2, 48_000, crate::filters::unity_chains());
        let [lc, rc, lo, ro, ls, rs] = splitter.split(&left, &right);
        for k in 0..2 {
            for (got, want) in [(lc[k], 0.0 * left[k]), (rc[k], 0.0 * right[k]), (lo[k], 0.0 * left[k]),
                                (ro[k], 0.0 * right[k]), (ls[k], left[k]), (rs[k], right[k])] {
                approx::assert_abs_diff_eq!((got - want).norm(), 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn complex_estimator_seeds_state_with_first_frame() {
        // Python seeds the smoothed state with the first frame, then smooths.
        // Frame 2 with alpha=0.5: cross = 0.5*2 + 0.5*1 = 1.5, auto = 0.5*4 + 0.5*1 = 2.5, w = 0.6.
        let mut est = ComplexEstimator::new(0.5, 1e-12, 1);
        let one = [Complex::new(1.0, 0.0)];
        est.estimate(&one, &[Complex::new(2.0, 0.0)]);
        let second = est.estimate(&one, &one)[0];
        approx::assert_abs_diff_eq!(second.re, 0.6, epsilon = 1e-12);
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
