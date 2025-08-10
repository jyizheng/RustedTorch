use crate::tensor::Tensor;

pub fn sum(x: &Tensor) -> Tensor {
    x.sum()
}
