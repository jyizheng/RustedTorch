use super::*;
use crate::tensor::{Tensor, Options};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_creation() {
        let params = vec![
            Tensor::ones(&[2, 3]),
            Tensor::zeros(&[1, 5]),
        ];
        
        let config = SGDConfig::new(0.01);
        let optimizer = SGD::new(params, config);
        
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.param_groups()[0].len(), 2);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let params = vec![Tensor::ones(&[2, 2])];
        
        let config = SGDConfig::new(0.1)
            .momentum(0.9)
            .weight_decay(0.01);
        
        let optimizer = SGD::new(params, config);
        
        assert_eq!(optimizer.config().lr, 0.1);
        assert_eq!(optimizer.config().momentum, 0.9);
        assert_eq!(optimizer.config().weight_decay, 0.01);
    }

    #[test]
    fn test_adam_creation() {
        let params = vec![
            Tensor::randn(&[3, 3]),
            Tensor::ones(&[1]),
        ];
        
        let config = AdamConfig::new(0.001);
        let optimizer = Adam::new(params, config);
        
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.param_groups()[0].len(), 2);
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_adam_with_custom_config() {
        let params = vec![Tensor::zeros(&[2, 2])];
        
        let config = AdamConfig::new(0.002)
            .betas(0.8, 0.99)
            .eps(1e-6)
            .weight_decay(0.1);
        
        let optimizer = Adam::new(params, config);
        
        assert_eq!(optimizer.config().lr, 0.002);
        assert_eq!(optimizer.config().beta1, 0.8);
        assert_eq!(optimizer.config().beta2, 0.99);
        assert_eq!(optimizer.config().eps, 1e-6);
        assert_eq!(optimizer.config().weight_decay, 0.1);
    }

    #[test]
    fn test_optimizer_trait_sgd() {
        let params = vec![Tensor::ones(&[2, 2])];
        let mut optimizer = SGD::with_lr(params, 0.01);
        
        optimizer.zero_grad();
        
        optimizer.step();
        
        let new_params = vec![Tensor::zeros(&[1, 3])];
        optimizer.add_param_group(new_params);
        assert_eq!(optimizer.param_groups().len(), 2);
    }

    #[test]
    fn test_optimizer_trait_adam() {
        let params = vec![Tensor::randn(&[3, 2])];
        let mut optimizer = Adam::with_lr(params, 0.001);
        
        optimizer.zero_grad();
        
        optimizer.step();
        assert_eq!(optimizer.step_count(), 1);
        
        let new_params = vec![Tensor::ones(&[2, 1])];
        optimizer.add_param_group(new_params);
        assert_eq!(optimizer.param_groups().len(), 2);
    }

    #[test]
    fn test_sgd_parameter_update_simulation() {
        let mut param = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        
        let params = vec![param.clone()];
        let mut optimizer = SGD::with_lr(params, 0.1);
        
        optimizer.step();
        optimizer.zero_grad();
    }

    #[test]
    fn test_adam_parameter_update_simulation() {
        let param = Tensor::from_array_2d(vec![
            vec![0.5f32, -0.2],
            vec![1.0, 0.8]
        ]);
        
        let params = vec![param];
        let mut optimizer = Adam::with_lr(params, 0.001);
        
        optimizer.step();
        assert_eq!(optimizer.step_count(), 1);
        
        optimizer.step();
        assert_eq!(optimizer.step_count(), 2);
        
        optimizer.zero_grad();
    }

    #[test]
    fn test_adamw_creation() {
        let params = vec![
            Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]),
            Tensor::from_array_2d(vec![vec![4.0f32, 5.0], vec![6.0, 7.0]]),
        ];
        
        let config = AdamWConfig::new(0.001)
            .betas(0.9, 0.999)
            .eps(1e-8)
            .weight_decay(0.01);
        
        let optimizer = AdamW::new(params, config);
        
        assert_eq!(optimizer.config().lr, 0.001);
        assert_eq!(optimizer.config().beta1, 0.9);
        assert_eq!(optimizer.config().beta2, 0.999);
        assert_eq!(optimizer.config().eps, 1e-8);
        assert_eq!(optimizer.config().weight_decay, 0.01);
        assert_eq!(optimizer.step_count(), 0);
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.param_groups()[0].len(), 2);
    }

    #[test]
    fn test_adamw_with_lr() {
        let params = vec![Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0])];
        let optimizer = AdamW::with_lr(params, 0.01);
        
        assert_eq!(optimizer.config().lr, 0.01);
        assert_eq!(optimizer.config().beta1, 0.9);
        assert_eq!(optimizer.config().beta2, 0.999);
        assert_eq!(optimizer.config().eps, 1e-8);
        assert_eq!(optimizer.config().weight_decay, 0.01);
    }

    #[test]
    fn test_adamw_parameter_update_simulation() {
        let param = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        let mut optimizer = AdamW::with_lr(vec![param], 0.01);
        
        optimizer.step();
        
        assert_eq!(optimizer.step_count(), 1);
    }

    #[test]
    fn test_adamw_with_custom_config() {
        let params = vec![Tensor::from_array_1d(vec![1.0f32, 2.0])];
        
        let config = AdamWConfig::new(0.002)
            .betas(0.95, 0.9999)
            .eps(1e-7)
            .weight_decay(0.05);
        
        let optimizer = AdamW::new(params, config);
        
        assert_eq!(optimizer.config().lr, 0.002);
        assert_eq!(optimizer.config().beta1, 0.95);
        assert_eq!(optimizer.config().beta2, 0.9999);
        assert_eq!(optimizer.config().eps, 1e-7);
        assert_eq!(optimizer.config().weight_decay, 0.05);
    }

    #[test]
    fn test_optimizer_trait_adamw() {
        let param = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        let mut optimizer: Box<dyn Optimizer> = Box::new(AdamW::with_lr(vec![param], 0.01));
        
        optimizer.step();
        optimizer.zero_grad();
        optimizer.add_param_group(vec![Tensor::from_array_1d(vec![4.0f32, 5.0])]);
    }
}
