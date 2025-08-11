use super::*;
use crate::tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_dataset_creation() {
        let features = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        let targets = Tensor::from_array_1d(vec![0.0f32, 1.0, 2.0]);
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
    }
    
    #[test]
    fn test_tensor_dataset_get_item() {
        let features = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let targets = Tensor::from_array_1d(vec![0.0f32, 1.0]);
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        
        let (feature, target) = dataset.get_item(0).unwrap();
        assert_eq!(feature.shape(), vec![3]);
        assert_eq!(feature.to_list::<f32>(), vec![1.0, 2.0, 3.0]);
        assert_eq!(target.to_list::<f32>(), vec![0.0]);
        
        let (feature, target) = dataset.get_item(1).unwrap();
        assert_eq!(feature.to_list::<f32>(), vec![4.0, 5.0, 6.0]);
        assert_eq!(target.to_list::<f32>(), vec![1.0]);
        
        assert!(dataset.get_item(2).is_none());
    }
    
    #[test]
    fn test_tensor_dataset_mismatched_lengths() {
        let features = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let targets = Tensor::from_array_1d(vec![0.0f32]);
        
        let result = TensorDataset::new(features, targets);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_in_memory_dataset() {
        let mut dataset = InMemoryDataset::new();
        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());
        
        let feature1 = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
        let target1 = Tensor::scalar(0.0f32);
        dataset.add_sample(feature1, target1);
        
        let feature2 = Tensor::from_array_1d(vec![4.0f32, 5.0, 6.0]);
        let target2 = Tensor::scalar(1.0f32);
        dataset.add_sample(feature2, target2);
        
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
        
        let (feature, target) = dataset.get_item(0).unwrap();
        assert_eq!(feature.to_list::<f32>(), vec![1.0, 2.0, 3.0]);
        assert_eq!(target.to_list::<f32>(), vec![0.0]);
    }
    
    #[test]
    fn test_dataloader_basic() {
        let features = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);
        let targets = Tensor::from_array_1d(vec![0.0f32, 1.0, 0.0, 1.0]);
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        let mut dataloader = DataLoader::new(dataset, 2);
        
        assert_eq!(dataloader.len(), 2);
        assert!(!dataloader.is_empty());
        
        let (batch_features, batch_targets) = dataloader.next_batch().unwrap();
        assert_eq!(batch_features.shape(), vec![2, 2]);
        assert_eq!(batch_targets.shape(), vec![2]);
        
        let (batch_features, batch_targets) = dataloader.next_batch().unwrap();
        assert_eq!(batch_features.shape(), vec![2, 2]);
        assert_eq!(batch_targets.shape(), vec![2]);
        
        assert!(dataloader.next_batch().is_none());
    }
    
    #[test]
    fn test_dataloader_drop_last() {
        let features = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]);
        let targets = Tensor::from_array_1d(vec![0.0f32, 1.0, 0.0]);
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        let mut dataloader = DataLoader::new(dataset, 2).drop_last(true);
        
        assert_eq!(dataloader.len(), 1);
        
        let (batch_features, batch_targets) = dataloader.next_batch().unwrap();
        assert_eq!(batch_features.shape(), vec![2, 2]);
        assert_eq!(batch_targets.shape(), vec![2]);
        
        assert!(dataloader.next_batch().is_none());
    }
    
    #[test]
    fn test_dataloader_iterator() {
        let features = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0],
            vec![3.0, 4.0],
        ]);
        let targets = Tensor::from_array_1d(vec![0.0f32, 1.0]);
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        let dataloader = DataLoader::new(dataset, 1);
        
        let batches: Vec<_> = dataloader.collect();
        assert_eq!(batches.len(), 2);
        
        let (features, targets) = &batches[0];
        assert_eq!(features.shape(), vec![1, 2]);
        assert_eq!(targets.shape(), vec![1]);
    }
    
    #[test]
    fn test_dataloader_reset() {
        let features = Tensor::from_array_2d(vec![
            vec![1.0f32, 2.0],
            vec![3.0, 4.0],
        ]);
        let targets = Tensor::from_array_1d(vec![0.0f32, 1.0]);
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        let mut dataloader = DataLoader::new(dataset, 1);
        
        let _batch1 = dataloader.next_batch().unwrap();
        let _batch2 = dataloader.next_batch().unwrap();
        assert!(dataloader.next_batch().is_none());
        
        dataloader.reset();
        let _batch1 = dataloader.next_batch().unwrap();
        let _batch2 = dataloader.next_batch().unwrap();
        assert!(dataloader.next_batch().is_none());
    }
}
