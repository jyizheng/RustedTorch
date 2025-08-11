use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructor_default() {
        let x = Tensor::new();
        assert!(!x.defined());
    }

    #[test]
    fn test_constructor_shape() {
        let x = Tensor::empty(&[2, 3]);
        assert!(x.defined());
        assert_eq!(x.dim(), 2);
        assert_eq!(x.numel(), 6);
        assert_eq!(x.shape(), vec![2, 3]);
        assert_eq!(x.strides(), vec![3, 1]);
    }

    #[test]
    fn test_constructor_scalar() {
        let x = Tensor::scalar(2.0f32);
        assert!(x.defined());
        assert_eq!(x.dim(), 0);
        assert_eq!(x.numel(), 1);
        assert_eq!(x.to_list::<f32>(), vec![2.0]);
    }

    #[test]
    fn test_constructor_ones() {
        let x = Tensor::ones(&[2, 3]);
        assert_eq!(x.dim(), 2);
        assert_eq!(x.numel(), 6);
        assert_eq!(x.shape(), vec![2, 3]);
        assert_eq!(x.strides(), vec![3, 1]);
        assert_eq!(x.to_list::<f32>(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_constructor_zeros() {
        let x = Tensor::zeros(&[2, 3]);
        assert_eq!(x.dim(), 2);
        assert_eq!(x.numel(), 6);
        assert_eq!(x.shape(), vec![2, 3]);
        assert_eq!(x.strides(), vec![3, 1]);
        assert_eq!(x.to_list::<f32>(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_constructor_rand() {
        let x = Tensor::rand(&[2, 3]);
        assert_eq!(x.dim(), 2);
        assert_eq!(x.numel(), 6);
        assert_eq!(x.shape(), vec![2, 3]);
        assert_eq!(x.strides(), vec![3, 1]);
    }

    #[test]
    fn test_constructor_randn() {
        let x = Tensor::randn(&[2, 3]);
        assert_eq!(x.dim(), 2);
        assert_eq!(x.numel(), 6);
        assert_eq!(x.shape(), vec![2, 3]);
        assert_eq!(x.strides(), vec![3, 1]);
    }

    #[test]
    fn test_constructor_bernoulli() {
        let x = Tensor::bernoulli(&[2, 3], 0.5);
        assert_eq!(x.dim(), 2);
        assert_eq!(x.numel(), 6);
        assert_eq!(x.shape(), vec![2, 3]);
        assert_eq!(x.strides(), vec![3, 1]);
    }

    #[test]
    fn test_constructor_1d() {
        let x = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(x.dim(), 1);
        assert_eq!(x.numel(), 3);
        assert_eq!(x.shape(), vec![3]);
        assert_eq!(x.strides(), vec![1]);
        assert_eq!(x.to_list::<f32>(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_constructor_2d() {
        let x = Tensor::from_array_2d(vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(x.dim(), 2);
        assert_eq!(x.numel(), 6);
        assert_eq!(x.shape(), vec![3, 2]);
        assert_eq!(x.strides(), vec![2, 1]);
        assert_eq!(x.to_list::<f32>(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_constructor_3d() {
        let x = Tensor::from_array_3d(vec![
            vec![vec![4.0f32, 2.0, 3.0], vec![1.0, 0.0, 3.0]],
            vec![vec![4.0, 2.0, 3.0], vec![1.0, 0.0, 3.0]]
        ]);
        assert_eq!(x.dim(), 3);
        assert_eq!(x.numel(), 12);
        assert_eq!(x.shape(), vec![2, 2, 3]);
        assert_eq!(x.strides(), vec![6, 3, 1]);
        assert_eq!(x.to_list::<f32>(), vec![4.0, 2.0, 3.0, 1.0, 0.0, 3.0, 4.0, 2.0, 3.0, 1.0, 0.0, 3.0]);
    }
}
