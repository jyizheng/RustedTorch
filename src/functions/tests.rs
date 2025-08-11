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
    fn test_func_add() {
        let options = Options::new().requires_grad(true);
        let a = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        let b = Tensor::from_array_1d(vec![4.0f32, 5.0, 6.0]);
        let y = function::function::add(&a, &b, 0.5);
        assert_eq!(y.to_list::<f32>(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_func_sub() {
        let options = Options::new().requires_grad(true);
        let a = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        let b = Tensor::from_array_1d(vec![4.0f32, 5.0, 6.0]);
        let y = function::function::sub(&a, &b);
        assert_eq!(y.to_list::<f32>(), vec![-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_func_mul() {
        let options = Options::new().requires_grad(true);
        let a = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        let b = Tensor::from_array_1d(vec![4.0f32, 5.0, 6.0]);
        let y = function::function::mul(&a, &b);
        assert_eq!(y.to_list::<f32>(), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_func_div() {
        let options = Options::new().requires_grad(true);
        let a = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        let b = Tensor::from_array_1d(vec![4.0f32, 5.0, 6.0]);
        let y = function::function::div(&a, &b);
        assert_vec_near(&y.to_list::<f32>(), &[0.25, 0.4, 0.5], 1e-6);
    }

    #[test]
    fn test_func_sin() {
        let options = Options::new().requires_grad(true);
        let x = Tensor::from_array_1d(vec![0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
        let y = function::function::sin(&x);
        assert_vec_near(&y.to_list::<f32>(), &[0.0, 1.0, 0.0], 1e-6);
    }

    #[test]
    fn test_func_cos() {
        let options = Options::new().requires_grad(true);
        let x = Tensor::from_array_1d(vec![0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
        let y = function::function::cos(&x);
        assert_vec_near(&y.to_list::<f32>(), &[1.0, 0.0, -1.0], 1e-6);
    }

    #[test]
    fn test_func_pow() {
        let options = Options::new().requires_grad(true);
        let x1 = Tensor::from_array_1d(vec![2.0f32, 3.0, 4.0]);
        let x2 = Tensor::from_array_1d(vec![3.0f32, 3.0, 3.0]);
        let y = function::function::pow(&x1, &x2);
        assert_eq!(y.to_list::<f32>(), vec![8.0, 27.0, 64.0]);

        let y_scalar = function::function::pow(&x1, &Tensor::scalar(3.0f32));
        assert_eq!(y_scalar.to_list::<f32>(), vec![8.0, 27.0, 64.0]);
    }

    #[test]
    fn test_func_sum() {
        let options = Options::new().requires_grad(true);
        let x = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        let y = function::function::sum(&x);
        assert_eq!(y.to_list::<f32>(), vec![6.0]);
    }

    #[test]
    fn test_func_relu() {
        let options = Options::new().requires_grad(true);
        let x = Tensor::from_array_2d(vec![vec![-1.0f32, 2.0], vec![3.0, -4.0]]);
        let y = function::function::relu(&x);
        assert_eq!(y.to_list::<f32>(), vec![0.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn test_func_gelu() {
        let x = Tensor::from_array_1d(vec![-1.0f32, -0.5, 0.5, 1.0]);
        let y = function::function::gelu(&x);
        assert_vec_near(&y.to_list::<f32>(), &[-0.1587, -0.1543, 0.3457, 0.8413], 1e-3);
    }

    #[test]
    fn test_func_silu() {
        let x = Tensor::from_array_1d(vec![-1.0f32, -0.5, 0.5, 1.0]);
        let y = function::function::silu(&x);
        assert_vec_near(&y.to_list::<f32>(), &[-0.2689, -0.1888, 0.3112, 0.7311], 1e-3);
    }

    #[test]
    fn test_func_softmax() {
        let options = Options::new().requires_grad(true);
        let input = Tensor::from_array_1d(vec![1.1f32, 1.2, 1.3, 1.6]);
        let output = function::function::softmax(&input, 0);
        assert_vec_near(&output.to_list::<f32>(), &[0.2010, 0.2221, 0.2455, 0.3314], 1e-3);
    }

    #[test]
    fn test_func_log_softmax() {
        let options = Options::new().requires_grad(true);
        let input = Tensor::from_array_1d(vec![1.1f32, 1.2, 1.3, 1.6]);
        let output = function::function::log_softmax(&input, 0);
        assert_vec_near(&output.to_list::<f32>(), &[-1.6045, -1.5045, -1.4045, -1.1045], 1e-3);
    }

    #[test]
    fn test_func_mse_loss_none() {
        let options = Options::new().requires_grad(true);
        let x = Tensor::from_array_2d(vec![
            vec![-0.3089f32, 0.5301, -0.0245],
            vec![1.5852, 0.8954, 0.7485]
        ]);
        let y = Tensor::from_array_2d(vec![
            vec![0.8397f32, 1.7990, -0.2738],
            vec![-0.8910, -0.6746, 0.3419]
        ]);
        let loss = mse_loss(&x, &y, LossReduction::None);
        assert_vec_near(&loss.to_list::<f32>(), 
                       &[1.31928194, 1.6101073, 0.0621504858, 6.13156557, 2.46489978, 0.165323555], 
                       1e-3);
    }

    #[test]
    fn test_func_mse_loss_mean() {
        let options = Options::new().requires_grad(true);
        let x = Tensor::from_array_2d(vec![
            vec![-0.3089f32, 0.5301, -0.0245],
            vec![1.5852, 0.8954, 0.7485]
        ]);
        let y = Tensor::from_array_2d(vec![
            vec![0.8397f32, 1.7990, -0.2738],
            vec![-0.8910, -0.6746, 0.3419]
        ]);
        let loss = mse_loss(&x, &y, LossReduction::Mean);
        assert!((loss.item::<f32>() - 1.95888805).abs() < 1e-3);
    }

    #[test]
    fn test_func_nll_loss() {
        let options = Options::new().requires_grad(true);
        let input = Tensor::from_array_2d(vec![vec![0.1f32, 0.2, 0.7], vec![0.3, 0.4, 0.3]]);
        let target = Tensor::from_array_1d(vec![2i64, 1]);
        let loss = nll_loss(&input, &target, LossReduction::None);
        assert_vec_near(&loss.to_list::<f32>(), &[-0.7, -0.4], 1e-6);
    }

    #[test]
    fn test_func_dropout() {
        let options = Options::new().requires_grad(true);
        let input = Tensor::ones(&[100, 10]);
        let p = 0.3f32;
        let output = dropout(&input, p, true);
        assert_eq!(output.shape(), input.shape());

        let output_no_training = dropout(&input, p, false);
        assert_eq!(output_no_training.to_list::<f32>(), input.to_list::<f32>());
    }

    #[test]
    fn test_func_tanh() {
        let x = Tensor::from_array_1d(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let y = function::function::tanh(&x);
        assert_vec_near(&y.to_list::<f32>(), &[-0.9640, -0.7616, 0.0, 0.7616, 0.9640], 1e-3);
    }

    #[test]
    fn test_func_sigmoid() {
        let x = Tensor::from_array_1d(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let y = function::function::sigmoid(&x);
        assert_vec_near(&y.to_list::<f32>(), &[0.1192, 0.2689, 0.5, 0.7311, 0.8808], 1e-3);
    }

    #[test]
    fn test_func_leaky_relu() {
        let x = Tensor::from_array_1d(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let y = function::function::leaky_relu(&x, 0.01);
        assert_vec_near(&y.to_list::<f32>(), &[-0.02, -0.01, 0.0, 1.0, 2.0], 1e-6);
    }

    #[test]
    fn test_func_swish() {
        let x = Tensor::from_array_1d(vec![-1.0f32, -0.5, 0.0, 0.5, 1.0]);
        let y = function::function::swish(&x);
        assert_vec_near(&y.to_list::<f32>(), &[-0.2689, -0.1888, 0.0, 0.3112, 0.7311], 1e-3);
    }
}
