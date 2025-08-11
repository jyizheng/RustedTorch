use super::*;
use crate::tensor::Tensor;
use std::fs;

#[cfg(test)]
mod tests {
    use super::*;

    fn cleanup_test_file(path: &str) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_model_state_creation() {
        let mut state = ModelState::new();
        assert_eq!(state.num_parameters(), 0);
        
        let tensor = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        state.add_parameter("weight".to_string(), tensor);
        
        assert_eq!(state.num_parameters(), 1);
        assert!(state.get_parameter("weight").is_some());
        assert!(state.get_parameter("bias").is_none());
    }
    
    #[test]
    fn test_model_state_metadata() {
        let mut state = ModelState::new();
        
        state.add_metadata("model_type".to_string(), "linear".to_string());
        state.add_metadata("version".to_string(), "1.0".to_string());
        
        assert_eq!(state.get_metadata("model_type"), Some(&"linear".to_string()));
        assert_eq!(state.get_metadata("version"), Some(&"1.0".to_string()));
        assert_eq!(state.get_metadata("nonexistent"), None);
    }
    
    #[test]
    fn test_model_state_parameter_operations() {
        let mut state = ModelState::new();
        
        let weight = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0],
            vec![3.0, 4.0],
        ]);
        let bias = Tensor::from_array_1d(vec![0.1f32, 0.2]);
        
        state.add_parameter("weight".to_string(), weight);
        state.add_parameter("bias".to_string(), bias);
        
        let names = state.parameter_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&&"weight".to_string()));
        assert!(names.contains(&&"bias".to_string()));
        
        let removed = state.remove_parameter("weight");
        assert!(removed.is_some());
        assert_eq!(state.num_parameters(), 1);
    }
    
    #[test]
    fn test_model_state_save_load() {
        let test_file = "test_model_state.bin";
        cleanup_test_file(test_file);
        
        let mut original_state = ModelState::new();
        original_state.add_metadata("model_type".to_string(), "test".to_string());
        
        let weight = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let bias = Tensor::from_array_1d(vec![0.1f32, 0.2]);
        
        original_state.add_parameter("weight".to_string(), weight);
        original_state.add_parameter("bias".to_string(), bias);
        
        let save_result = original_state.save_to_file(test_file);
        assert!(save_result.is_ok());
        
        let loaded_state = ModelState::load_from_file(test_file);
        assert!(loaded_state.is_ok());
        
        let loaded_state = loaded_state.unwrap();
        assert_eq!(loaded_state.num_parameters(), 2);
        assert_eq!(loaded_state.get_metadata("model_type"), Some(&"test".to_string()));
        
        let loaded_weight = loaded_state.get_parameter("weight").unwrap();
        assert_eq!(loaded_weight.shape(), vec![2, 3]);
        assert_eq!(loaded_weight.to_list::<f32>(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let loaded_bias = loaded_state.get_parameter("bias").unwrap();
        assert_eq!(loaded_bias.shape(), vec![2]);
        assert_eq!(loaded_bias.to_list::<f32>(), vec![0.1, 0.2]);
        
        cleanup_test_file(test_file);
    }
    
    #[test]
    fn test_checkpoint_creation() {
        let mut model_state = ModelState::new();
        let weight = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        model_state.add_parameter("weight".to_string(), weight);
        
        let mut checkpoint = Checkpoint::new(model_state)
            .with_epoch(10)
            .with_loss(0.5);
        
        checkpoint.add_metric("accuracy".to_string(), 0.95);
        checkpoint.add_metric("f1_score".to_string(), 0.92);
        
        assert_eq!(checkpoint.epoch, 10);
        assert_eq!(checkpoint.loss, 0.5);
        assert_eq!(checkpoint.get_metric("accuracy"), Some(0.95));
        assert_eq!(checkpoint.get_metric("f1_score"), Some(0.92));
        assert_eq!(checkpoint.get_metric("nonexistent"), None);
    }
    
    #[test]
    fn test_checkpoint_save_load() {
        let test_file = "test_checkpoint.bin";
        cleanup_test_file(test_file);
        
        let mut model_state = ModelState::new();
        model_state.add_metadata("model_type".to_string(), "classifier".to_string());
        
        let weight = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0],
            vec![3.0, 4.0],
        ]);
        model_state.add_parameter("weight".to_string(), weight);
        
        let mut original_checkpoint = Checkpoint::new(model_state)
            .with_epoch(5)
            .with_loss(0.25);
        
        original_checkpoint.add_metric("accuracy".to_string(), 0.88);
        original_checkpoint.add_metric("precision".to_string(), 0.90);
        
        let save_result = original_checkpoint.save_to_file(test_file);
        assert!(save_result.is_ok());
        
        let loaded_checkpoint = Checkpoint::load_from_file(test_file);
        assert!(loaded_checkpoint.is_ok());
        
        let loaded_checkpoint = loaded_checkpoint.unwrap();
        assert_eq!(loaded_checkpoint.epoch, 5);
        assert_eq!(loaded_checkpoint.loss, 0.25);
        assert_eq!(loaded_checkpoint.get_metric("accuracy"), Some(0.88));
        assert_eq!(loaded_checkpoint.get_metric("precision"), Some(0.90));
        
        assert_eq!(loaded_checkpoint.model_state.num_parameters(), 1);
        let loaded_weight = loaded_checkpoint.model_state.get_parameter("weight").unwrap();
        assert_eq!(loaded_weight.shape(), vec![2, 2]);
        assert_eq!(loaded_weight.to_list::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
        
        cleanup_test_file(test_file);
    }
    
    #[test]
    fn test_optimizer_state_creation() {
        let parameter_names = vec!["weight".to_string(), "bias".to_string()];
        
        let sgd_state = OptimizerState::from_sgd_config(0.01, 0.9, 1e-4, &parameter_names);
        
        match sgd_state {
            OptimizerState::SGD { lr, momentum, weight_decay, momentum_buffers } => {
                assert_eq!(lr, 0.01);
                assert_eq!(momentum, 0.9);
                assert_eq!(weight_decay, 1e-4);
                assert_eq!(momentum_buffers.len(), 2);
            },
            _ => panic!("Expected SGD optimizer state"),
        }
        
        let adam_state = OptimizerState::from_adam_config(0.001, 0.9, 0.999, 1e-8, 0.0, 0, &parameter_names);
        
        match adam_state {
            OptimizerState::Adam { lr, beta1, beta2, eps, exp_avg, exp_avg_sq, .. } => {
                assert_eq!(lr, 0.001);
                assert_eq!(beta1, 0.9);
                assert_eq!(beta2, 0.999);
                assert_eq!(eps, 1e-8);
                assert_eq!(exp_avg.len(), 2);
                assert_eq!(exp_avg_sq.len(), 2);
            },
            _ => panic!("Expected Adam optimizer state"),
        }
    }
    
    #[test]
    fn test_invalid_file_format() {
        let test_file = "test_invalid.bin";
        
        fs::write(test_file, b"INVALID_HEADER").unwrap();
        
        let result = ModelState::load_from_file(test_file);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid file format"));
        
        cleanup_test_file(test_file);
    }
    
    #[test]
    fn test_empty_model_state_save_load() {
        let test_file = "test_empty_model.bin";
        cleanup_test_file(test_file);
        
        let original_state = ModelState::new();
        let save_result = original_state.save_to_file(test_file);
        assert!(save_result.is_ok());
        
        let loaded_state = ModelState::load_from_file(test_file);
        assert!(loaded_state.is_ok());
        
        let loaded_state = loaded_state.unwrap();
        assert_eq!(loaded_state.num_parameters(), 0);
        
        cleanup_test_file(test_file);
    }
}
