use crate::tensor::Tensor;
use crate::serialization::ModelState;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub model_state: ModelState,
    pub optimizer_state: Option<OptimizerState>,
    pub epoch: u32,
    pub loss: f32,
    pub metrics: HashMap<String, f32>,
}

#[derive(Clone)]
pub enum OptimizerState {
    SGD {
        momentum_buffers: HashMap<String, Tensor>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    },
    Adam {
        exp_avg: HashMap<String, Tensor>,
        exp_avg_sq: HashMap<String, Tensor>,
        step: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    },
}

impl std::fmt::Debug for OptimizerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerState::SGD { lr, momentum, weight_decay, .. } => {
                f.debug_struct("SGD")
                    .field("lr", lr)
                    .field("momentum", momentum)
                    .field("weight_decay", weight_decay)
                    .field("momentum_buffers", &"<tensors>")
                    .finish()
            },
            OptimizerState::Adam { lr, beta1, beta2, eps, weight_decay, step, .. } => {
                f.debug_struct("Adam")
                    .field("lr", lr)
                    .field("beta1", beta1)
                    .field("beta2", beta2)
                    .field("eps", eps)
                    .field("weight_decay", weight_decay)
                    .field("step", step)
                    .field("exp_avg", &"<tensors>")
                    .field("exp_avg_sq", &"<tensors>")
                    .finish()
            },
        }
    }
}

impl Checkpoint {
    pub fn new(model_state: ModelState) -> Self {
        Checkpoint {
            model_state,
            optimizer_state: None,
            epoch: 0,
            loss: 0.0,
            metrics: HashMap::new(),
        }
    }
    
    pub fn with_optimizer_state(mut self, optimizer_state: OptimizerState) -> Self {
        self.optimizer_state = Some(optimizer_state);
        self
    }
    
    pub fn with_epoch(mut self, epoch: u32) -> Self {
        self.epoch = epoch;
        self
    }
    
    pub fn with_loss(mut self, loss: f32) -> Self {
        self.loss = loss;
        self
    }
    
    pub fn add_metric(&mut self, name: String, value: f32) {
        self.metrics.insert(name, value);
    }
    
    pub fn get_metric(&self, name: &str) -> Option<f32> {
        self.metrics.get(name).copied()
    }
    
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let mut extended_model_state = self.model_state.clone();
        
        extended_model_state.add_metadata("epoch".to_string(), self.epoch.to_string());
        extended_model_state.add_metadata("loss".to_string(), self.loss.to_string());
        
        for (key, value) in &self.metrics {
            extended_model_state.add_metadata(format!("metric_{}", key), value.to_string());
        }
        
        if let Some(ref opt_state) = self.optimizer_state {
            match opt_state {
                OptimizerState::SGD { lr, momentum, weight_decay, .. } => {
                    extended_model_state.add_metadata("optimizer_type".to_string(), "SGD".to_string());
                    extended_model_state.add_metadata("optimizer_lr".to_string(), lr.to_string());
                    extended_model_state.add_metadata("optimizer_momentum".to_string(), momentum.to_string());
                    extended_model_state.add_metadata("optimizer_weight_decay".to_string(), weight_decay.to_string());
                },
                OptimizerState::Adam { lr, beta1, beta2, eps, weight_decay, step, .. } => {
                    extended_model_state.add_metadata("optimizer_type".to_string(), "Adam".to_string());
                    extended_model_state.add_metadata("optimizer_lr".to_string(), lr.to_string());
                    extended_model_state.add_metadata("optimizer_beta1".to_string(), beta1.to_string());
                    extended_model_state.add_metadata("optimizer_beta2".to_string(), beta2.to_string());
                    extended_model_state.add_metadata("optimizer_eps".to_string(), eps.to_string());
                    extended_model_state.add_metadata("optimizer_weight_decay".to_string(), weight_decay.to_string());
                    extended_model_state.add_metadata("optimizer_step".to_string(), step.to_string());
                },
            }
        }
        
        extended_model_state.save_to_file(path)
    }
    
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let model_state = ModelState::load_from_file(path)?;
        
        let epoch = model_state.get_metadata("epoch")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
            
        let loss = model_state.get_metadata("loss")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        
        let mut metrics = HashMap::new();
        for (key, value) in model_state.metadata.iter() {
            if key.starts_with("metric_") {
                let metric_name = key.strip_prefix("metric_").unwrap().to_string();
                if let Ok(metric_value) = value.parse::<f32>() {
                    metrics.insert(metric_name, metric_value);
                }
            }
        }
        
        let optimizer_state = if let Some(opt_type) = model_state.get_metadata("optimizer_type") {
            match opt_type.as_str() {
                "SGD" => {
                    let lr = model_state.get_metadata("optimizer_lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.01);
                    let momentum = model_state.get_metadata("optimizer_momentum")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    let weight_decay = model_state.get_metadata("optimizer_weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    
                    Some(OptimizerState::SGD {
                        momentum_buffers: HashMap::new(), // Would need to be serialized separately
                        lr,
                        momentum,
                        weight_decay,
                    })
                },
                "Adam" => {
                    let lr = model_state.get_metadata("optimizer_lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.001);
                    let beta1 = model_state.get_metadata("optimizer_beta1")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.9);
                    let beta2 = model_state.get_metadata("optimizer_beta2")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.999);
                    let eps = model_state.get_metadata("optimizer_eps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-8);
                    let weight_decay = model_state.get_metadata("optimizer_weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    let step = model_state.get_metadata("optimizer_step")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    
                    Some(OptimizerState::Adam {
                        exp_avg: HashMap::new(), // Would need to be serialized separately
                        exp_avg_sq: HashMap::new(), // Would need to be serialized separately
                        step,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                    })
                },
                _ => None,
            }
        } else {
            None
        };
        
        Ok(Checkpoint {
            model_state,
            optimizer_state,
            epoch,
            loss,
            metrics,
        })
    }
}

impl OptimizerState {
    pub fn from_sgd_config(lr: f32, momentum: f32, weight_decay: f32, parameter_names: &[String]) -> Self {
        let momentum_buffers = parameter_names.iter()
            .map(|name| (name.clone(), Tensor::new()))
            .collect();
            
        OptimizerState::SGD {
            momentum_buffers,
            lr,
            momentum,
            weight_decay,
        }
    }
    
    pub fn from_adam_config(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, step: u32, parameter_names: &[String]) -> Self {
        let exp_avg = parameter_names.iter()
            .map(|name| (name.clone(), Tensor::new()))
            .collect();
        let exp_avg_sq = parameter_names.iter()
            .map(|name| (name.clone(), Tensor::new()))
            .collect();
            
        OptimizerState::Adam {
            exp_avg,
            exp_avg_sq,
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        }
    }
}
