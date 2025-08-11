use crate::tensor::Tensor;

pub struct AutogradMeta {
    pub grad: Option<Tensor>,
    requires_grad: bool,
}

impl AutogradMeta {
    pub fn new() -> Self {
        Self {
            grad: None,
            requires_grad: false,
        }
    }

    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(grad);
    }

    pub fn add_grad(&mut self, grad: Tensor) {
        match &mut self.grad {
            Some(existing_grad) => {
                *existing_grad = &*existing_grad + &grad;
            }
            None => {
                self.grad = Some(grad);
            }
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    pub fn is_leaf(&self) -> bool {
        true
    }

    pub fn backward(&mut self, grad: &Tensor) {
        if !self.requires_grad {
            return;
        }
        
        self.add_grad(grad.clone());
    }
}

impl Default for AutogradMeta {
    fn default() -> Self {
        Self::new()
    }
}
