use crate::data::Dataset;
use crate::tensor::{Tensor, TensorImpl, Options, DType};
use std::rc::Rc;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    indices: Vec<usize>,
    current_batch: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).collect();
        
        DataLoader {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            indices,
            current_batch: 0,
        }
    }
    
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        if shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
        self
    }
    
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }
    
    pub fn len(&self) -> usize {
        if self.drop_last {
            self.dataset.len() / self.batch_size
        } else {
            self.dataset.len().div_ceil(self.batch_size)
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    pub fn reset(&mut self) {
        self.current_batch = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }
    
    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.current_batch >= self.len() {
            return None;
        }
        
        let start_idx = self.current_batch * self.batch_size;
        let end_idx = std::cmp::min(start_idx + self.batch_size, self.dataset.len());
        
        if self.drop_last && (end_idx - start_idx) < self.batch_size {
            return None;
        }
        
        let mut batch_features = Vec::new();
        let mut batch_targets = Vec::new();
        let mut feature_shape: Option<Vec<i64>> = None;
        let mut target_shape: Option<Vec<i64>> = None;
        
        for i in start_idx..end_idx {
            let dataset_idx = self.indices[i];
            if let Some((features, targets)) = self.dataset.get_item(dataset_idx) {
                let f_shape = features.shape();
                let t_shape = targets.shape();
                
                if feature_shape.is_none() {
                    feature_shape = Some(f_shape.clone());
                }
                if target_shape.is_none() {
                    target_shape = Some(t_shape.clone());
                }
                
                batch_features.extend(features.to_list::<f32>());
                batch_targets.extend(targets.to_list::<f32>());
            }
        }
        
        if batch_features.is_empty() || batch_targets.is_empty() {
            return None;
        }
        
        let actual_batch_size = end_idx - start_idx;
        
        let mut batched_feature_shape = vec![actual_batch_size as i64];
        if let Some(ref f_shape) = feature_shape {
            batched_feature_shape.extend(f_shape);
        }
        
        let mut batched_target_shape = vec![actual_batch_size as i64];
        if let Some(ref t_shape) = target_shape {
            if !t_shape.is_empty() {
                batched_target_shape.extend(t_shape);
            }
        }
        
        let options = Options::default().dtype(DType::Float32);
        
        let batched_features = match TensorImpl::new_from_data(&batch_features, &batched_feature_shape, options.clone()) {
            Ok(impl_) => Tensor {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => return None,
        };
        
        let batched_targets = match TensorImpl::new_from_data(&batch_targets, &batched_target_shape, options) {
            Ok(impl_) => Tensor { impl_: Some(Rc::new(impl_)) },
            Err(_) => return None,
        };
        
        self.current_batch += 1;
        Some((batched_features, batched_targets))
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = (Tensor, Tensor);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}
