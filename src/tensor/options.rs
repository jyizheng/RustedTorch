use crate::tensor::{Device, DeviceType, DType};

#[derive(Debug, Clone)]
pub struct Options {
    pub device: Device,
    pub dtype: DType,
    pub requires_grad: bool,
    pub pinned_memory: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            device: Device::cpu(),
            dtype: DType::Float32,
            requires_grad: false,
            pinned_memory: false,
        }
    }
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    pub fn requires_grad_value(&self) -> bool {
        self.requires_grad
    }

    pub fn pinned_memory(mut self, pinned_memory: bool) -> Self {
        self.pinned_memory = pinned_memory;
        self
    }

    pub fn no_grad(&self) -> Self {
        let mut opts = self.clone();
        opts.requires_grad = false;
        opts
    }

    pub fn indices(&self) -> Self {
        let mut opts = self.clone();
        opts.dtype = DType::Int64;
        opts
    }
}

pub mod options {
    use super::*;

    pub fn device(device_type: DeviceType, index: i8) -> Options {
        Options::new().device(Device::new(device_type, index))
    }

    pub fn device_from(device: Device) -> Options {
        Options::new().device(device)
    }

    pub fn dtype(dtype: DType) -> Options {
        Options::new().dtype(dtype)
    }

    pub fn requires_grad(requires_grad: bool) -> Options {
        Options::new().requires_grad(requires_grad)
    }

    pub fn pinned_memory(pinned_memory: bool) -> Options {
        Options::new().pinned_memory(pinned_memory)
    }
}
