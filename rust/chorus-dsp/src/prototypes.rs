pub fn center_prototype(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(l, r)| 0.5 * (l + r)).collect()
}

pub fn surround_prototype(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(l, r)| 0.5 * (l - r)).collect()
}

#[cfg(test)]
mod tests {
    use super::{center_prototype, surround_prototype};

    #[test]
    fn builds_center_and_surround_prototypes() {
        assert_eq!(center_prototype(&[1.0, 1.0], &[1.0, -1.0]), vec![1.0, 0.0]);
        assert_eq!(surround_prototype(&[1.0, 1.0], &[1.0, -1.0]), vec![0.0, 1.0]);
    }
}
