use crate::tensor::Tensor;

pub trait Function {
    fn forward(&self, inputs: &[Tensor]) -> Tensor;
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}

pub struct AddFunction;

impl Function for AddFunction {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        if inputs.len() != 2 {
            return Tensor::new();
        }
        &inputs[0] + &inputs[1]
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.clone(), grad_output.clone()]
    }
}

pub struct MulFunction;

impl Function for MulFunction {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        if inputs.len() != 2 {
            return Tensor::new();
        }
        &inputs[0] * &inputs[1]
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if let Some(inputs) = self.get_saved_inputs() {
            vec![
                &*grad_output * &inputs[1],
                &*grad_output * &inputs[0],
            ]
        } else {
            vec![grad_output.clone(), grad_output.clone()]
        }
    }
}

impl MulFunction {
    fn get_saved_inputs(&self) -> Option<Vec<Tensor>> {
        None
    }
}

pub mod function {
    use super::*;

    pub fn add(a: &Tensor, b: &Tensor, _alpha: f32) -> Tensor {
        a + b
    }

    pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
        if !a.defined() || !b.defined() {
            return Tensor::new();
        }

        let a_data = a.to_list::<f32>();
        let b_data = b.to_list::<f32>();
        
        let result_data: Vec<f32> = a_data.iter().zip(b_data.iter())
            .map(|(&x, &y)| x - y)
            .collect();

        let shape = a.shape();
        let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
        match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }

    pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
        a * b
    }

    pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
        if !a.defined() || !b.defined() {
            return Tensor::new();
        }

        let a_data = a.to_list::<f32>();
        let b_data = b.to_list::<f32>();
        
        let result_data: Vec<f32> = a_data.iter().zip(b_data.iter())
            .map(|(&x, &y)| x / y)
            .collect();

        let shape = a.shape();
        let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
        match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }

    pub fn sin(x: &Tensor) -> Tensor {
        if !x.defined() {
            return Tensor::new();
        }

        let data = x.to_list::<f32>();
        let result_data: Vec<f32> = data.iter().map(|&val| val.sin()).collect();

        let shape = x.shape();
        let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
        match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }

    pub fn cos(x: &Tensor) -> Tensor {
        if !x.defined() {
            return Tensor::new();
        }

        let data = x.to_list::<f32>();
        let result_data: Vec<f32> = data.iter().map(|&val| val.cos()).collect();

        let shape = x.shape();
        let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
        match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }

    pub fn pow(base: &Tensor, exponent: &Tensor) -> Tensor {
        base.pow(exponent)
    }

    pub fn sum(x: &Tensor) -> Tensor {
        x.sum()
    }

    pub fn relu(x: &Tensor) -> Tensor {
        if !x.defined() {
            return Tensor::new();
        }

        let data = x.to_list::<f32>();
        let result_data: Vec<f32> = data.iter().map(|&val| val.max(0.0)).collect();

        let shape = x.shape();
        let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
        match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }

    pub fn gelu(x: &Tensor) -> Tensor {
        if !x.defined() {
            return Tensor::new();
        }

        let data = x.to_list::<f32>();
        let result_data: Vec<f32> = data.iter().map(|&val| {
            0.5 * val * (1.0 + (val * 0.7978845608 * (1.0 + 0.044715 * val * val)).tanh())
        }).collect();

        let shape = x.shape();
        let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
        match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }

    pub fn silu(x: &Tensor) -> Tensor {
        if !x.defined() {
            return Tensor::new();
        }

        let data = x.to_list::<f32>();
        let result_data: Vec<f32> = data.iter().map(|&val| {
            val / (1.0 + (-val).exp())
        }).collect();

        let shape = x.shape();
        let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
        match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }

    pub fn softmax(x: &Tensor, _dim: i64) -> Tensor {
        if !x.defined() {
            return Tensor::new();
        }

        let data = x.to_list::<f32>();
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let exp_data: Vec<f32> = data.iter().map(|&val| (val - max_val).exp()).collect();
        let sum_exp: f32 = exp_data.iter().sum();
        
        let result_data: Vec<f32> = exp_data.iter().map(|&val| val / sum_exp).collect();

        let shape = x.shape();
        let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
        match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }

    pub fn log_softmax(x: &Tensor, _dim: i64) -> Tensor {
        if !x.defined() {
            return Tensor::new();
        }

        let data = x.to_list::<f32>();
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let exp_data: Vec<f32> = data.iter().map(|&val| (val - max_val).exp()).collect();
        let sum_exp: f32 = exp_data.iter().sum();
        let log_sum_exp = sum_exp.ln();
        
        let result_data: Vec<f32> = data.iter().map(|&val| val - max_val - log_sum_exp).collect();

        let shape = x.shape();
        let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
        match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(std::rc::Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }
}
