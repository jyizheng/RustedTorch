use crate::tensor::Tensor;
use crate::optimizers::Optimizer;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SGDConfig {
    pub lr: f32,
    pub momentum: f32,
    pub dampening: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

impl SGDConfig {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }
    
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
    
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

pub struct SGD {
    config: SGDConfig,
    param_groups: Vec<Vec<Tensor>>,
    momentum_buffers: HashMap<usize, Tensor>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, config: SGDConfig) -> Self {
        Self {
            config,
            param_groups: vec![params],
            momentum_buffers: HashMap::new(),
        }
    }
    
    pub fn with_lr(params: Vec<Tensor>, lr: f32) -> Self {
        Self::new(params, SGDConfig::new(lr))
    }
    
    pub fn param_groups(&self) -> &Vec<Vec<Tensor>> {
        &self.param_groups
    }
    
    pub fn config(&self) -> &SGDConfig {
        &self.config
    }
    
    fn compute_update_from_grad(&mut self, param: &Tensor, grad: &Tensor, param_id: usize) -> Option<Tensor> {
        let mut d_p = grad.clone();
        
        if self.config.weight_decay != 0.0 {
            d_p = &d_p + &(&*param * &Tensor::scalar(self.config.weight_decay));
        }
        
        if self.config.momentum != 0.0 {
            if let Some(buf) = self.momentum_buffers.get(&param_id) {
                let mut momentum_buf = buf * &Tensor::scalar(self.config.momentum);
                momentum_buf = &momentum_buf + &(&d_p * &Tensor::scalar(1.0 - self.config.dampening));
                
                if self.config.nesterov {
                    d_p = &d_p + &(&momentum_buf * &Tensor::scalar(self.config.momentum));
                } else {
                    d_p = momentum_buf.clone();
                }
                
                self.momentum_buffers.insert(param_id, momentum_buf);
            } else {
                let momentum_buf = d_p.clone();
                self.momentum_buffers.insert(param_id, momentum_buf.clone());
                d_p = momentum_buf;
            }
        }
        
        let update = &d_p * &Tensor::scalar(-self.config.lr);
        Some(update)
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        let mut param_data = Vec::new();
        
        for (group_idx, param_group) in self.param_groups.iter().enumerate() {
            for (param_idx, param) in param_group.iter().enumerate() {
                let param_id = group_idx * 1000 + param_idx;
                if param.defined() {
                    let grad = param.grad();
                    if grad.defined() {
                        param_data.push((group_idx, param_idx, param_id, param.clone(), grad));
                    }
                }
            }
        }
        
        let mut updates = Vec::new();
        for (group_idx, param_idx, param_id, param, grad) in param_data {
            if let Some(update) = self.compute_update_from_grad(&param, &grad, param_id) {
                updates.push((group_idx, param_idx, update));
            }
        }
        
        for (group_idx, param_idx, update) in updates {
            if let Some(param_group) = self.param_groups.get_mut(group_idx) {
                if let Some(param) = param_group.get_mut(param_idx) {
                    *param = &*param + &update;
                }
            }
        }
    }
    
    fn zero_grad(&mut self) {
        for param_group in &mut self.param_groups {
            for param in param_group {
                param.zero_grad();
            }
        }
    }
    
    fn add_param_group(&mut self, params: Vec<Tensor>) {
        self.param_groups.push(params);
    }
}
