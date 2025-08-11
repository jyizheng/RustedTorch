use super::*;
use crate::autograd::function;
use crate::tensor::{Tensor, Options};

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_near(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < tolerance, "Expected {}, got {}", e, a);
        }
    }

    #[test]
    fn test_backward_01() {
        let options = Options::new().requires_grad(true);
        let x1 = Tensor::from_array_1d(vec![0.0140f32, 0.5773, 0.0469]);
        let x2 = Tensor::from_array_1d(vec![0.3232f32, 0.4903, 0.9395]);

        let sin_x1 = function::function::sin(&x1);
        let mul_result = &x1 * &x2;
        let y = &sin_x1 + &mul_result;

        assert_eq!(y.shape(), vec![3]);
    }

    #[test]
    fn test_backward_02() {
        let options = Options::new().requires_grad(true);
        let x = Tensor::from_array_2d(vec![vec![1.0f32, -1.0], vec![1.0, 1.0]]);
        let x_pow = x.pow(&Tensor::scalar(2.0f32));
        let y = x_pow.sum();
        
        assert_eq!(y.shape(), vec![]);
    }

    #[test]
    fn test_backward_flatten() {
        let options = Options::new().requires_grad(true);
        let x1 = Tensor::from_array_2d(vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]);
        let x2 = Tensor::from_array_2d(vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]);
        let x3 = &x1 * &x2;
        let y = x3.flatten();
        
        assert_eq!(y.to_list::<f32>(), vec![1.0, 4.0, 9.0, 16.0]);
    }
}
