use crate::tensor::Tensor;
use crate::optimizers::Optimizer;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub amsgrad: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        }
    }
}

impl AdamConfig {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }
    
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
    
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
    
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

pub struct Adam {
    config: AdamConfig,
    param_groups: Vec<Vec<Tensor>>,
    exp_avg: HashMap<usize, Tensor>,      // First moment estimates
    exp_avg_sq: HashMap<usize, Tensor>,   // Second moment estimates
    max_exp_avg_sq: HashMap<usize, Tensor>, // For AMSGrad
    step_count: usize,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, config: AdamConfig) -> Self {
        Self {
            config,
            param_groups: vec![params],
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            max_exp_avg_sq: HashMap::new(),
            step_count: 0,
        }
    }
    
    pub fn with_lr(params: Vec<Tensor>, lr: f32) -> Self {
        Self::new(params, AdamConfig::new(lr))
    }
    
    pub fn param_groups(&self) -> &Vec<Vec<Tensor>> {
        &self.param_groups
    }
    
    pub fn config(&self) -> &AdamConfig {
        &self.config
    }
    
    pub fn step_count(&self) -> usize {
        self.step_count
    }
    
    fn compute_update_from_grad(&mut self, param: &Tensor, grad: &Tensor, param_id: usize) -> Option<Tensor> {
        let mut d_p = grad.clone();
        
        if self.config.weight_decay != 0.0 {
            d_p = &d_p + &(param * &Tensor::scalar(self.config.weight_decay));
        }
        
        if let std::collections::hash_map::Entry::Vacant(e) = self.exp_avg.entry(param_id) {
            e.insert(Tensor::zeros_like(param));
            self.exp_avg_sq.insert(param_id, Tensor::zeros_like(param));
            if self.config.amsgrad {
                self.max_exp_avg_sq.insert(param_id, Tensor::zeros_like(param));
            }
        }
        
        let exp_avg = self.exp_avg.get_mut(&param_id).unwrap();
        let exp_avg_sq = self.exp_avg_sq.get_mut(&param_id).unwrap();
        
        *exp_avg = exp_avg.clone() * &Tensor::scalar(self.config.beta1) + 
                   &(&d_p * &Tensor::scalar(1.0 - self.config.beta1));
        
        let grad_squared = &d_p * &d_p;
        *exp_avg_sq = exp_avg_sq.clone() * &Tensor::scalar(self.config.beta2) + 
                      &(&grad_squared * &Tensor::scalar(1.0 - self.config.beta2));
        
        let denom = if self.config.amsgrad {
            let max_exp_avg_sq = self.max_exp_avg_sq.get_mut(&param_id).unwrap();
            *max_exp_avg_sq = element_wise_max(max_exp_avg_sq, exp_avg_sq);
            sqrt_tensor(&(max_exp_avg_sq.clone() + &Tensor::scalar(self.config.eps)))
        } else {
            sqrt_tensor(&(exp_avg_sq.clone() + &Tensor::scalar(self.config.eps)))
        };
        
        let bias_correction1 = 1.0 - self.config.beta1.powi(self.step_count as i32 + 1);
        let bias_correction2 = 1.0 - self.config.beta2.powi(self.step_count as i32 + 1);
        let step_size = self.config.lr * (bias_correction2.sqrt() / bias_correction1);
        
        let update = &(exp_avg.clone() / &denom) * &Tensor::scalar(-step_size);
        Some(update)
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.step_count += 1;
        
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

fn element_wise_max(a: &Tensor, b: &Tensor) -> Tensor {
    a.max_elementwise(b)
}

fn sqrt_tensor(tensor: &Tensor) -> Tensor {
    tensor.sqrt()
}
