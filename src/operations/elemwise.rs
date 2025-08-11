use crate::tensor::Tensor;

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    a + b
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    a * b
}
