use crate::tensor::{Tensor, DType, Options, TensorImpl};
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
pub enum LossReduction {
    None,
    Mean,
    Sum,
}

pub fn mse_loss(input: &Tensor, target: &Tensor, reduction: LossReduction) -> Tensor {
    if !input.defined() || !target.defined() {
        return Tensor::new();
    }

    let input_data = input.to_list::<f32>();
    let target_data = target.to_list::<f32>();
    
    let diff_squared: Vec<f32> = input_data.iter().zip(target_data.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .collect();

    match reduction {
        LossReduction::None => {
            let shape = input.shape();
            let options = Options::default().dtype(DType::Float32);
            match TensorImpl::new_from_data(&diff_squared, &shape, options) {
                Ok(impl_) => Tensor {
                    impl_: Some(Rc::new(impl_)),
                },
                Err(_) => Tensor::new(),
            }
        }
        LossReduction::Mean => {
            let mean_val = diff_squared.iter().sum::<f32>() / diff_squared.len() as f32;
            Tensor::scalar(mean_val)
        }
        LossReduction::Sum => {
            let sum_val = diff_squared.iter().sum::<f32>();
            Tensor::scalar(sum_val)
        }
    }
}

pub fn nll_loss(input: &Tensor, target: &Tensor, reduction: LossReduction) -> Tensor {
    if !input.defined() || !target.defined() {
        return Tensor::new();
    }

    let input_data = input.to_list::<f32>();
    let target_data = target.to_list::<i64>();
    let input_shape = input.shape();
    
    if input_shape.len() != 2 {
        return Tensor::new();
    }

    let batch_size = input_shape[0] as usize;
    let num_classes = input_shape[1] as usize;
    
    let mut losses = Vec::new();
    for i in 0..batch_size {
        let target_class = target_data[i] as usize;
        if target_class < num_classes {
            let loss = -input_data[i * num_classes + target_class];
            losses.push(loss);
        }
    }

    match reduction {
        LossReduction::None => {
            let shape = [batch_size as i64];
            let options = Options::default().dtype(DType::Float32);
            match TensorImpl::new_from_data(&losses, &shape, options) {
                Ok(impl_) => Tensor {
                    impl_: Some(Rc::new(impl_)),
                },
                Err(_) => Tensor::new(),
            }
        }
        LossReduction::Mean => {
            let mean_val = losses.iter().sum::<f32>() / losses.len() as f32;
            Tensor::scalar(mean_val)
        }
        LossReduction::Sum => {
            let sum_val = losses.iter().sum::<f32>();
            Tensor::scalar(sum_val)
        }
    }
}

pub fn cross_entropy_loss(input: &Tensor, target: &Tensor, reduction: LossReduction) -> Tensor {
    if !input.defined() || !target.defined() {
        return Tensor::new();
    }

    let log_softmax_input = crate::autograd::function::function::log_softmax(input, 1);
    nll_loss(&log_softmax_input, target, reduction)
}

pub fn bce_loss(input: &Tensor, target: &Tensor, reduction: LossReduction) -> Tensor {
    if !input.defined() || !target.defined() {
        return Tensor::new();
    }

    let input_data = input.to_list::<f32>();
    let target_data = target.to_list::<f32>();
    
    if input_data.len() != target_data.len() {
        return Tensor::new();
    }

    let losses: Vec<f32> = input_data.iter().zip(target_data.iter())
        .map(|(&pred, &target)| {
            let pred_clamped = pred.max(1e-7).min(1.0 - 1e-7);
            -(target * pred_clamped.ln() + (1.0 - target) * (1.0 - pred_clamped).ln())
        })
        .collect();

    match reduction {
        LossReduction::None => {
            let shape = input.shape();
            let options = Options::default().dtype(DType::Float32);
            match TensorImpl::new_from_data(&losses, &shape, options) {
                Ok(impl_) => Tensor {
                    impl_: Some(Rc::new(impl_)),
                },
                Err(_) => Tensor::new(),
            }
        }
        LossReduction::Mean => {
            let mean_val = losses.iter().sum::<f32>() / losses.len() as f32;
            Tensor::scalar(mean_val)
        }
        LossReduction::Sum => {
            let sum_val = losses.iter().sum::<f32>();
            Tensor::scalar(sum_val)
        }
    }
}

pub fn l1_loss(input: &Tensor, target: &Tensor, reduction: LossReduction) -> Tensor {
    if !input.defined() || !target.defined() {
        return Tensor::new();
    }

    let input_data = input.to_list::<f32>();
    let target_data = target.to_list::<f32>();
    
    if input_data.len() != target_data.len() {
        return Tensor::new();
    }

    let losses: Vec<f32> = input_data.iter().zip(target_data.iter())
        .map(|(&x, &y)| (x - y).abs())
        .collect();

    match reduction {
        LossReduction::None => {
            let shape = input.shape();
            let options = Options::default().dtype(DType::Float32);
            match TensorImpl::new_from_data(&losses, &shape, options) {
                Ok(impl_) => Tensor {
                    impl_: Some(Rc::new(impl_)),
                },
                Err(_) => Tensor::new(),
            }
        }
        LossReduction::Mean => {
            let mean_val = losses.iter().sum::<f32>() / losses.len() as f32;
            Tensor::scalar(mean_val)
        }
        LossReduction::Sum => {
            let sum_val = losses.iter().sum::<f32>();
            Tensor::scalar(sum_val)
        }
    }
}
