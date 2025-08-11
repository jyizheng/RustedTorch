use crate::tensor::{
    Array1d, Array2d, Array3d, DType, Device, Options, Scalar, TensorImpl, TypeToDType,
    flatten_2d, flatten_3d,
};
use rand::Rng;
use std::rc::Rc;

pub struct Tensor {
    pub impl_: Option<Rc<TensorImpl>>,
}

impl Tensor {
    pub fn new() -> Self {
        Self { impl_: None }
    }

    pub fn empty(shape: &[i64]) -> Self {
        let options = Options::default();
        match TensorImpl::new(shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn empty_with_options(shape: &[i64], options: Options) -> Self {
        match TensorImpl::new(shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn scalar<T: TypeToDType + Clone + Into<Scalar>>(value: T) -> Self {
        let options = Options::default().dtype(T::DTYPE);
        let data = vec![value];
        match TensorImpl::new_from_data(&data, &[], options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn ones(shape: &[i64]) -> Self {
        let options = Options::default();
        let numel: usize = shape.iter().product::<i64>() as usize;
        let data = vec![1.0f32; numel];
        match TensorImpl::new_from_data(&data, shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn zeros(shape: &[i64]) -> Self {
        let options = Options::default();
        let numel: usize = shape.iter().product::<i64>() as usize;
        let data = vec![0.0f32; numel];
        match TensorImpl::new_from_data(&data, shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn rand(shape: &[i64]) -> Self {
        let options = Options::default();
        let numel: usize = shape.iter().product::<i64>() as usize;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..numel).map(|_| rng.gen::<f32>()).collect();
        match TensorImpl::new_from_data(&data, shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn randn(shape: &[i64]) -> Self {
        let options = Options::default();
        let numel: usize = shape.iter().product::<i64>() as usize;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        match TensorImpl::new_from_data(&data, shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn bernoulli(shape: &[i64], p: f32) -> Self {
        let options = Options::default();
        let numel: usize = shape.iter().product::<i64>() as usize;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..numel)
            .map(|_| if rng.gen::<f32>() < p { 1.0 } else { 0.0 })
            .collect();
        match TensorImpl::new_from_data(&data, shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn linspace(start: f32, end: f32, steps: usize) -> Self {
        let options = Options::default();
        let step_size = if steps > 1 { (end - start) / (steps - 1) as f32 } else { 0.0 };
        let data: Vec<f32> = (0..steps)
            .map(|i| start + i as f32 * step_size)
            .collect();
        let shape = [steps as i64];
        match TensorImpl::new_from_data(&data, &shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn ones_like(other: &Self) -> Self {
        if let Some(ref impl_) = other.impl_ {
            Self::ones(impl_.shape())
        } else {
            Self::new()
        }
    }

    pub fn zeros_like(other: &Self) -> Self {
        if let Some(ref impl_) = other.impl_ {
            Self::zeros(impl_.shape())
        } else {
            Self::new()
        }
    }

    pub fn from_array_1d<T: TypeToDType + Clone>(data: Array1d<T>) -> Self {
        let shape = [data.len() as i64];
        let options = Options::default().dtype(T::DTYPE);
        match TensorImpl::new_from_data(&data, &shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn from_array_2d<T: TypeToDType + Clone>(data: Array2d<T>) -> Self {
        if data.is_empty() {
            return Self::new();
        }
        let rows = data.len() as i64;
        let cols = data[0].len() as i64;
        let shape = [rows, cols];
        let flat_data = flatten_2d(&data);
        let options = Options::default().dtype(T::DTYPE);
        match TensorImpl::new_from_data(&flat_data, &shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn from_array_3d<T: TypeToDType + Clone>(data: Array3d<T>) -> Self {
        if data.is_empty() || data[0].is_empty() {
            return Self::new();
        }
        let dim0 = data.len() as i64;
        let dim1 = data[0].len() as i64;
        let dim2 = data[0][0].len() as i64;
        let shape = [dim0, dim1, dim2];
        let flat_data = flatten_3d(&data);
        let options = Options::default().dtype(T::DTYPE);
        match TensorImpl::new_from_data(&flat_data, &shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn defined(&self) -> bool {
        self.impl_.is_some()
    }

    pub fn dim(&self) -> i64 {
        self.impl_.as_ref().map_or(0, |impl_| impl_.dim())
    }

    pub fn numel(&self) -> i64 {
        self.impl_.as_ref().map_or(0, |impl_| impl_.numel())
    }

    pub fn shape(&self) -> Vec<i64> {
        self.impl_
            .as_ref()
            .map_or(Vec::new(), |impl_| impl_.shape().to_vec())
    }

    pub fn strides(&self) -> Vec<i64> {
        self.impl_
            .as_ref()
            .map_or(Vec::new(), |impl_| impl_.strides().to_vec())
    }

    pub fn dtype(&self) -> DType {
        self.impl_.as_ref().map_or(DType::Float32, |impl_| impl_.dtype())
    }

    pub fn device(&self) -> Device {
        self.impl_.as_ref().map_or(Device::cpu(), |impl_| impl_.device())
    }

    pub fn requires_grad(&self) -> bool {
        self.impl_.as_ref().map_or(false, |impl_| impl_.requires_grad())
    }

    pub fn to_list<T: TypeToDType + Clone + Default>(&self) -> Vec<T> {
        self.impl_
            .as_ref()
            .and_then(|impl_| impl_.to_list().ok())
            .unwrap_or_default()
    }

    pub fn item<T: TypeToDType + Clone + Default>(&self) -> T {
        self.impl_
            .as_ref()
            .and_then(|impl_| impl_.item().ok())
            .unwrap_or_default()
    }

    pub fn reshape_(&mut self, shape: &[i64]) -> Result<(), String> {
        if let Some(ref mut impl_) = self.impl_ {
            if let Some(impl_mut) = Rc::get_mut(impl_) {
                impl_mut.reshape_(shape)
            } else {
                Err("Cannot get mutable reference to tensor implementation".to_string())
            }
        } else {
            Err("Cannot reshape undefined tensor".to_string())
        }
    }

    pub fn flatten(&self) -> Self {
        if let Some(ref impl_) = self.impl_ {
            let total_elements = impl_.numel();
            let new_shape = [total_elements];
            let mut new_tensor = self.clone();
            let _ = new_tensor.reshape_(&new_shape);
            new_tensor
        } else {
            Self::new()
        }
    }

    pub fn clone(&self) -> Self {
        if let Some(ref impl_) = self.impl_ {
            if let Some(storage) = impl_.storage() {
                if let Ok(cloned_storage) = storage.as_ref().clone() {
                    if let Ok(new_impl) = TensorImpl::new_with_storage(
                        impl_.shape(),
                        impl_.options().clone(),
                        Rc::new(cloned_storage),
                        impl_.storage_offset(),
                    ) {
                        return Self {
                            impl_: Some(Rc::new(new_impl)),
                        };
                    }
                }
            }
        }
        Self::new()
    }

    pub fn pow(&self, exponent: &Self) -> Self {
        if !self.defined() || !exponent.defined() {
            return Self::new();
        }

        let self_data = self.to_list::<f32>();
        let exp_data = exponent.to_list::<f32>();
        
        let result_data: Vec<f32> = if exp_data.len() == 1 {
            let exp_val = exp_data[0];
            self_data.iter().map(|&x| x.powf(exp_val)).collect()
        } else {
            self_data.iter().zip(exp_data.iter())
                .map(|(&x, &e)| x.powf(e))
                .collect()
        };

        let shape = self.shape();
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn sum(&self) -> Self {
        if !self.defined() {
            return Self::new();
        }

        let data = self.to_list::<f32>();
        let sum_val = data.iter().sum::<f32>();
        Self::scalar(sum_val)
    }

    pub fn backward(&self) {
    }

    pub fn backward_with_grad(&self, _grad: &Self) {
    }

    pub fn set_requires_grad(&mut self, _requires_grad: bool) {
    }

    pub fn grad(&self) -> Self {
        Self::new()
    }

    pub fn zero_grad(&mut self) {
    }

    pub fn matmul(&self, other: &Self) -> Self {
        if !self.defined() || !other.defined() {
            return Self::new();
        }

        let self_shape = self.shape();
        let other_shape = other.shape();
        
        if self_shape.len() == 2 && other_shape.len() == 2 {
            let m = self_shape[0] as usize;
            let k = self_shape[1] as usize;
            let n = other_shape[1] as usize;
            
            if k != other_shape[0] as usize {
                return Self::new(); // Incompatible dimensions
            }
            
            let self_data = self.to_list::<f32>();
            let other_data = other.to_list::<f32>();
            let mut result = vec![0.0f32; m * n];
            
            for i in 0..m {
                for j in 0..n {
                    for k_idx in 0..k {
                        result[i * n + j] += self_data[i * k + k_idx] * other_data[k_idx * n + j];
                    }
                }
            }
            
            let result_shape = [m as i64, n as i64];
            let options = Options::default().dtype(DType::Float32);
            match TensorImpl::new_from_data(&result, &result_shape, options) {
                Ok(impl_) => Self {
                    impl_: Some(Rc::new(impl_)),
                },
                Err(_) => Self::new(),
            }
        } else {
            Self::new() // Only 2D matmul supported for now
        }
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        if !self.defined() {
            return Self::new();
        }
        
        let shape = self.shape();
        if shape.len() != 2 || dim0 != 0 || dim1 != 1 {
            return self.clone(); // Only 2D transpose supported for now
        }
        
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        let data = self.to_list::<f32>();
        let mut transposed = vec![0.0f32; rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }
        
        let new_shape = [cols as i64, rows as i64];
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&transposed, &new_shape, options) {
            Ok(impl_) => Self {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Self::new(),
        }
    }

    pub fn reshape(&self, shape: &[i64]) -> Self {
        if !self.defined() {
            return Self::new();
        }
        
        let mut new_tensor = self.clone();
        let _ = new_tensor.reshape_(shape);
        new_tensor
    }

    pub fn size(&self) -> i64 {
        self.numel()
    }

    pub fn new_from_impl(impl_: Rc<TensorImpl>) -> Self {
        Self {
            impl_: Some(impl_),
        }
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            impl_: self.impl_.clone(),
        }
    }
}

impl From<Array1d<f32>> for Tensor {
    fn from(data: Array1d<f32>) -> Self {
        Self::from_array_1d(data)
    }
}

impl From<Array2d<f32>> for Tensor {
    fn from(data: Array2d<f32>) -> Self {
        Self::from_array_2d(data)
    }
}

impl From<Array3d<f32>> for Tensor {
    fn from(data: Array3d<f32>) -> Self {
        Self::from_array_3d(data)
    }
}

impl std::ops::Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        if !self.defined() || !other.defined() {
            return Tensor::new();
        }

        let self_data = self.to_list::<f32>();
        let other_data = other.to_list::<f32>();
        
        let result_data: Vec<f32> = self_data.iter().zip(other_data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let shape = self.shape();
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        if !self.defined() || !other.defined() {
            return Tensor::new();
        }

        let self_data = self.to_list::<f32>();
        let other_data = other.to_list::<f32>();
        
        let result_data: Vec<f32> = self_data.iter().zip(other_data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        let shape = self.shape();
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        if !self.defined() || !other.defined() {
            return Tensor::new();
        }

        let self_data = self.to_list::<f32>();
        let other_data = other.to_list::<f32>();
        
        let result_data: Vec<f32> = self_data.iter().zip(other_data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        let shape = self.shape();
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }
}

impl std::ops::Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        if !self.defined() || !other.defined() {
            return Tensor::new();
        }

        let self_data = self.to_list::<f32>();
        let other_data = other.to_list::<f32>();
        
        let result_data: Vec<f32> = self_data.iter().zip(other_data.iter())
            .map(|(&a, &b)| if b != 0.0 { a / b } else { 0.0 })
            .collect();

        let shape = self.shape();
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&result_data, &shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(Rc::new(impl_)),
            },
            Err(_) => Tensor::new(),
        }
    }
}

impl std::ops::Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        &self + other
    }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        &self * other
    }
}

impl std::ops::Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        &self / other
    }
}

impl Tensor {
    pub fn sqrt(&self) -> Self {
        if !self.defined() {
            return Self::new();
        }
        
        let data: Vec<f32> = self.to_list::<f32>().iter()
            .map(|&x| x.sqrt())
            .collect();
        
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&data, &self.shape(), options) {
            Ok(impl_) => Self { impl_: Some(Rc::new(impl_)) },
            Err(_) => Self::new(),
        }
    }
    
    pub fn max_elementwise(&self, other: &Self) -> Self {
        if !self.defined() || !other.defined() {
            return Self::new();
        }
        
        let self_data = self.to_list::<f32>();
        let other_data = other.to_list::<f32>();
        
        let result_data: Vec<f32> = self_data.iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a.max(b))
            .collect();
        
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&result_data, &self.shape(), options) {
            Ok(impl_) => Self { impl_: Some(Rc::new(impl_)) },
            Err(_) => Self::new(),
        }
    }
}
