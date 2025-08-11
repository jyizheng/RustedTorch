use crate::tensor::Tensor;

pub trait Dataset {
    fn len(&self) -> usize;
    fn get_item(&self, index: usize) -> Option<(Tensor, Tensor)>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct TensorDataset {
    features: Tensor,
    targets: Tensor,
    length: usize,
}

impl TensorDataset {
    pub fn new(features: Tensor, targets: Tensor) -> Result<Self, String> {
        if !features.defined() || !targets.defined() {
            return Err("Features and targets must be defined tensors".to_string());
        }
        
        let features_shape = features.shape();
        let targets_shape = targets.shape();
        
        if features_shape.is_empty() || targets_shape.is_empty() {
            return Err("Features and targets must have at least one dimension".to_string());
        }
        
        let features_len = features_shape[0] as usize;
        let targets_len = targets_shape[0] as usize;
        
        if features_len != targets_len {
            return Err(format!(
                "Features and targets must have same length: {} vs {}",
                features_len, targets_len
            ));
        }
        
        Ok(TensorDataset {
            features,
            targets,
            length: features_len,
        })
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.length
    }
    
    fn get_item(&self, index: usize) -> Option<(Tensor, Tensor)> {
        if index >= self.length {
            return None;
        }
        
        let features_shape = self.features.shape();
        let targets_shape = self.targets.shape();
        
        if features_shape.len() < 2 || targets_shape.len() < 1 {
            return None;
        }
        
        let features_data = self.features.to_list::<f32>();
        let targets_data = self.targets.to_list::<f32>();
        
        let feature_size = features_shape[1..].iter().product::<i64>() as usize;
        let target_size = if targets_shape.len() > 1 {
            targets_shape[1..].iter().product::<i64>() as usize
        } else {
            1
        };
        
        let feature_start = index * feature_size;
        let feature_end = feature_start + feature_size;
        let target_start = index * target_size;
        let target_end = target_start + target_size;
        
        if feature_end > features_data.len() || target_end > targets_data.len() {
            return None;
        }
        
        let feature_slice = &features_data[feature_start..feature_end];
        let target_slice = &targets_data[target_start..target_end];
        
        let feature_shape = &features_shape[1..];
        let target_shape = if targets_shape.len() > 1 {
            &targets_shape[1..]
        } else {
            &[]
        };
        
        let feature_tensor = match crate::tensor::TensorImpl::new_from_data(
            feature_slice,
            feature_shape,
            crate::tensor::Options::default().dtype(crate::tensor::DType::Float32),
        ) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => return None,
        };
        
        let target_tensor = if target_shape.is_empty() {
            Tensor::scalar(target_slice[0])
        } else {
            match crate::tensor::TensorImpl::new_from_data(
                target_slice,
                target_shape,
                crate::tensor::Options::default().dtype(crate::tensor::DType::Float32),
            ) {
                Ok(impl_) => Tensor {
                    impl_: Some(std::rc::Rc::new(impl_)),
                },
                Err(_) => return None,
            }
        };
        
        Some((feature_tensor, target_tensor))
    }
}

pub struct InMemoryDataset {
    data: Vec<(Tensor, Tensor)>,
}

impl InMemoryDataset {
    pub fn new() -> Self {
        InMemoryDataset {
            data: Vec::new(),
        }
    }
    
    pub fn add_sample(&mut self, features: Tensor, target: Tensor) {
        self.data.push((features, target));
    }
    
    pub fn from_vec(data: Vec<(Tensor, Tensor)>) -> Self {
        InMemoryDataset { data }
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn get_item(&self, index: usize) -> Option<(Tensor, Tensor)> {
        self.data.get(index).cloned()
    }
}

impl Default for InMemoryDataset {
    fn default() -> Self {
        Self::new()
    }
}
