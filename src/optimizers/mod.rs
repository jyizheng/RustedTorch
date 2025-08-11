pub mod sgd;
pub mod adam;

pub use sgd::*;
pub use adam::*;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn add_param_group(&mut self, params: Vec<crate::tensor::Tensor>);
}

#[cfg(test)]
mod tests;
