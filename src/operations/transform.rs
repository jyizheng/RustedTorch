use crate::tensor::Tensor;

pub fn reshape(x: &Tensor, shape: &[i64]) -> Tensor {
    let mut result = x.clone();
    let _ = result.reshape_(shape);
    result
}

pub fn flatten(x: &Tensor) -> Tensor {
    x.flatten()
}
