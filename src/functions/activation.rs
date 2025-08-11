use crate::autograd::function;
use crate::tensor::Tensor;

pub fn relu(x: &Tensor) -> Tensor {
    function::function::relu(x)
}

pub fn gelu(x: &Tensor) -> Tensor {
    function::function::gelu(x)
}

pub fn silu(x: &Tensor) -> Tensor {
    function::function::silu(x)
}

pub fn softmax(x: &Tensor, dim: i64) -> Tensor {
    function::function::softmax(x, dim)
}

pub fn log_softmax(x: &Tensor, dim: i64) -> Tensor {
    function::function::log_softmax(x, dim)
}
