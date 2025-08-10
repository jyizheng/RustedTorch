use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    CPU = 0,
    CUDA = 1,
}

pub type DeviceIndex = i8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    pub device_type: DeviceType,
    pub index: DeviceIndex,
}

impl Device {
    pub fn new(device_type: DeviceType, index: DeviceIndex) -> Self {
        Self { device_type, index }
    }

    pub fn cpu() -> Self {
        Self::new(DeviceType::CPU, 0)
    }

    pub fn cuda(index: DeviceIndex) -> Self {
        Self::new(DeviceType::CUDA, index)
    }

    pub fn is_cpu(&self) -> bool {
        self.device_type == DeviceType::CPU
    }

    pub fn is_cuda(&self) -> bool {
        self.device_type == DeviceType::CUDA
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.device_type {
            DeviceType::CPU => write!(f, "CPU"),
            DeviceType::CUDA => write!(f, "CUDA:{}", self.index),
        }
    }
}
