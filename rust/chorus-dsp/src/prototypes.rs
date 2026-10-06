use num_complex::Complex;

pub fn center_prototype(left: &[Complex<f64>], right: &[Complex<f64>]) -> Vec<Complex<f64>> {
    left.iter().zip(right).map(|(l, r)| {
        let shared = l.norm().min(r.norm());
        (safe_unit_phase(*l) * shared + safe_unit_phase(*r) * shared) * 0.5
    }).collect()
}

pub fn surround_prototype(left: &[Complex<f64>], right: &[Complex<f64>]) -> Vec<Complex<f64>> {
    left.iter().zip(right).map(|(l, r)| {
        let shared = l.norm().min(r.norm());
        (safe_unit_phase(*l) * shared - safe_unit_phase(*r) * shared) * 0.5
    }).collect()
}

fn safe_unit_phase(v: Complex<f64>) -> Complex<f64> {
    let mag = v.norm();
    if mag > 0.0 { v / mag } else { Complex::new(0.0, 0.0) }
}

#[cfg(test)]
mod tests {
    use super::{center_prototype, surround_prototype};
    use num_complex::Complex;

    #[test]
    fn complex_prototypes_pure_real_inputs() {
        // Pure real, in-phase signals: both channels identical
        let left: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let right: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let center = center_prototype(&left, &right);
        let surround = surround_prototype(&left, &right);
        // Center should equal the common value (shared=min(mag,mag)=mag, both phases same)
        for (c, l) in center.iter().zip(&left) {
            approx::assert_abs_diff_eq!(c.re, l.re, epsilon = 1e-12);
            approx::assert_abs_diff_eq!(c.im, 0.0, epsilon = 1e-12);
        }
        // Surround should be zero (phases cancel)
        for s in &surround {
            approx::assert_abs_diff_eq!(s.norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn complex_prototypes_zero_input_no_panic_no_nan() {
        let left: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let right: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let center = center_prototype(&left, &right);
        let surround = surround_prototype(&left, &right);
        for c in &center {
            assert!(!c.re.is_nan() && !c.im.is_nan());
            approx::assert_abs_diff_eq!(c.norm(), 0.0, epsilon = 1e-12);
        }
        for s in &surround {
            assert!(!s.re.is_nan() && !s.im.is_nan());
            approx::assert_abs_diff_eq!(s.norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn complex_surround_captures_anti_phase_difference() {
        // Anti-phase: left = +A, right = -A (pure real)
        let a = 2.0_f64;
        let left: Vec<Complex<f64>> = vec![Complex::new(a, 0.0)];
        let right: Vec<Complex<f64>> = vec![Complex::new(-a, 0.0)];
        let surround = surround_prototype(&left, &right);
        // shared = min(|a|, |-a|) = a
        // safe_unit_phase(l)=1, safe_unit_phase(r)=-1
        // surround = (1*a - (-1)*a)*0.5 = a
        approx::assert_abs_diff_eq!(surround[0].re, a, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(surround[0].im, 0.0, epsilon = 1e-12);

        // Center should be zero for anti-phase
        let center = center_prototype(&left, &right);
        approx::assert_abs_diff_eq!(center[0].norm(), 0.0, epsilon = 1e-12);
    }
}
