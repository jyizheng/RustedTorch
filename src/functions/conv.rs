use crate::tensor::{Tensor, DType, Options, TensorImpl};
use std::rc::Rc;

pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (i64, i64),
    padding: (i64, i64),
    dilation: (i64, i64),
) -> Tensor {
    if !input.defined() || !weight.defined() {
        return Tensor::new();
    }

    let input_shape = input.shape();
    let weight_shape = weight.shape();
    
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return Tensor::new();
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];
    
    let out_channels = weight_shape[0];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];
    
    if weight_shape[1] != in_channels {
        return Tensor::new();
    }

    let output_height = (input_height + 2 * padding.0 - dilation.0 * (kernel_height - 1) - 1) / stride.0 + 1;
    let output_width = (input_width + 2 * padding.1 - dilation.1 * (kernel_width - 1) - 1) / stride.1 + 1;
    
    let output_shape = vec![batch_size, out_channels, output_height, output_width];
    let output_size = output_shape.iter().product::<i64>() as usize;
    
    let input_data = input.to_list::<f32>();
    let weight_data = weight.to_list::<f32>();
    let bias_data = bias.map(|b| b.to_list::<f32>());
    
    let mut output_data = vec![0.0f32; output_size];
    
    for batch in 0..batch_size as usize {
        for out_ch in 0..out_channels as usize {
            for out_h in 0..output_height as usize {
                for out_w in 0..output_width as usize {
                    let mut sum = 0.0f32;
                    
                    for in_ch in 0..in_channels as usize {
                        for kh in 0..kernel_height as usize {
                            for kw in 0..kernel_width as usize {
                                let in_h = out_h as i64 * stride.0 - padding.0 + kh as i64 * dilation.0;
                                let in_w = out_w as i64 * stride.1 - padding.1 + kw as i64 * dilation.1;
                                
                                if in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width {
                                    let input_idx = batch * (in_channels as usize) * (input_height as usize) * (input_width as usize) +
                                                  in_ch * (input_height as usize) * (input_width as usize) +
                                                  (in_h as usize) * (input_width as usize) +
                                                  (in_w as usize);
                                    
                                    let weight_idx = out_ch * (in_channels as usize) * (kernel_height as usize) * (kernel_width as usize) +
                                                   in_ch * (kernel_height as usize) * (kernel_width as usize) +
                                                   kh * (kernel_width as usize) +
                                                   kw;
                                    
                                    sum += input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                    
                    if let Some(ref bias_vec) = bias_data {
                        sum += bias_vec[out_ch];
                    }
                    
                    let output_idx = batch * (out_channels as usize) * (output_height as usize) * (output_width as usize) +
                                   out_ch * (output_height as usize) * (output_width as usize) +
                                   out_h * (output_width as usize) +
                                   out_w;
                    
                    output_data[output_idx] = sum;
                }
            }
        }
    }
    
    let options = Options::default().dtype(DType::Float32);
    match TensorImpl::new_from_data(&output_data, &output_shape, options) {
        Ok(impl_) => Tensor {
            impl_: Some(Rc::new(impl_)),
        },
        Err(_) => Tensor::new(),
    }
}

pub fn max_pool2d(
    input: &Tensor,
    kernel_size: (i64, i64),
    stride: Option<(i64, i64)>,
    padding: (i64, i64),
) -> Tensor {
    if !input.defined() {
        return Tensor::new();
    }

    let input_shape = input.shape();
    if input_shape.len() != 4 {
        return Tensor::new();
    }

    let stride = stride.unwrap_or(kernel_size);
    
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];
    
    let output_height = (input_height + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let output_width = (input_width + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
    
    let output_shape = vec![batch_size, channels, output_height, output_width];
    let output_size = output_shape.iter().product::<i64>() as usize;
    
    let input_data = input.to_list::<f32>();
    let mut output_data = vec![f32::NEG_INFINITY; output_size];
    
    for batch in 0..batch_size as usize {
        for ch in 0..channels as usize {
            for out_h in 0..output_height as usize {
                for out_w in 0..output_width as usize {
                    let mut max_val = f32::NEG_INFINITY;
                    
                    for kh in 0..kernel_size.0 as usize {
                        for kw in 0..kernel_size.1 as usize {
                            let in_h = out_h as i64 * stride.0 - padding.0 + kh as i64;
                            let in_w = out_w as i64 * stride.1 - padding.1 + kw as i64;
                            
                            if in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width {
                                let input_idx = batch * (channels as usize) * (input_height as usize) * (input_width as usize) +
                                              ch * (input_height as usize) * (input_width as usize) +
                                              (in_h as usize) * (input_width as usize) +
                                              (in_w as usize);
                                
                                max_val = max_val.max(input_data[input_idx]);
                            }
                        }
                    }
                    
                    let output_idx = batch * (channels as usize) * (output_height as usize) * (output_width as usize) +
                                   ch * (output_height as usize) * (output_width as usize) +
                                   out_h * (output_width as usize) +
                                   out_w;
                    
                    output_data[output_idx] = max_val;
                }
            }
        }
    }
    
    let options = Options::default().dtype(DType::Float32);
    match TensorImpl::new_from_data(&output_data, &output_shape, options) {
        Ok(impl_) => Tensor {
            impl_: Some(Rc::new(impl_)),
        },
        Err(_) => Tensor::new(),
    }
}

pub fn batch_norm2d(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    training: bool,
    _momentum: f32,
    eps: f32,
) -> Tensor {
    if !input.defined() {
        return Tensor::new();
    }

    let input_shape = input.shape();
    if input_shape.len() != 4 {
        return Tensor::new();
    }

    let batch_size = input_shape[0] as usize;
    let channels = input_shape[1] as usize;
    let height = input_shape[2] as usize;
    let width = input_shape[3] as usize;
    
    let input_data = input.to_list::<f32>();
    let mut output_data = vec![0.0f32; input_data.len()];
    
    let weight_data = weight.map(|w| w.to_list::<f32>());
    let bias_data = bias.map(|b| b.to_list::<f32>());
    
    for ch in 0..channels {
        let mut mean = 0.0f32;
        let mut var = 0.0f32;
        
        if training {
            let mut sum = 0.0f32;
            let count = (batch_size * height * width) as f32;
            
            for batch in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let idx = batch * channels * height * width + ch * height * width + h * width + w;
                        sum += input_data[idx];
                    }
                }
            }
            mean = sum / count;
            
            let mut sum_sq_diff = 0.0f32;
            for batch in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let idx = batch * channels * height * width + ch * height * width + h * width + w;
                        let diff = input_data[idx] - mean;
                        sum_sq_diff += diff * diff;
                    }
                }
            }
            var = sum_sq_diff / count;
        } else {
            if let Some(rm) = running_mean {
                let rm_data = rm.to_list::<f32>();
                mean = rm_data[ch];
            }
            if let Some(rv) = running_var {
                let rv_data = rv.to_list::<f32>();
                var = rv_data[ch];
            }
        }
        
        let std_dev = (var + eps).sqrt();
        let gamma = weight_data.as_ref().map(|w| w[ch]).unwrap_or(1.0);
        let beta = bias_data.as_ref().map(|b| b[ch]).unwrap_or(0.0);
        
        for batch in 0..batch_size {
            for h in 0..height {
                for w in 0..width {
                    let idx = batch * channels * height * width + ch * height * width + h * width + w;
                    let normalized = (input_data[idx] - mean) / std_dev;
                    output_data[idx] = gamma * normalized + beta;
                }
            }
        }
    }
    
    let options = Options::default().dtype(DType::Float32);
    match TensorImpl::new_from_data(&output_data, &input_shape, options) {
        Ok(impl_) => Tensor {
            impl_: Some(Rc::new(impl_)),
        },
        Err(_) => Tensor::new(),
    }
}
