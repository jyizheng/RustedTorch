use crate::tensor::Device;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

#[derive(Debug)]
pub struct Storage {
    data: NonNull<u8>,
    size: usize,
    device: Device,
    layout: Layout,
}

impl Storage {
    pub fn new(size: usize, device: Device) -> Result<Self, String> {
        if size == 0 {
            return Err("Storage size cannot be zero".to_string());
        }

        let layout = Layout::from_size_align(size, 8)
            .map_err(|e| format!("Invalid layout: {}", e))?;

        let data = if device.is_cpu() {
            unsafe {
                let ptr = alloc(layout);
                if ptr.is_null() {
                    return Err("Failed to allocate memory".to_string());
                }
                NonNull::new_unchecked(ptr)
            }
        } else {
            return Err("CUDA storage not yet implemented".to_string());
        };

        Ok(Self {
            data,
            size,
            device,
            layout,
        })
    }

    pub fn data_ptr<T>(&self) -> *mut T {
        self.data.as_ptr() as *mut T
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn copy_from_slice<T>(&mut self, src: &[T]) -> Result<(), String> {
        let src_size = std::mem::size_of_val(src);
        if src_size > self.size {
            return Err("Source data too large for storage".to_string());
        }

        if self.device.is_cpu() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr() as *const u8,
                    self.data.as_ptr(),
                    src_size,
                );
            }
            Ok(())
        } else {
            Err("CUDA copy not yet implemented".to_string())
        }
    }

    pub fn copy_to_slice<T>(&self, dst: &mut [T]) -> Result<(), String> {
        let dst_size = std::mem::size_of_val(dst);
        if dst_size > self.size {
            return Err("Destination slice too large".to_string());
        }

        if self.device.is_cpu() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data.as_ptr(),
                    dst.as_mut_ptr() as *mut u8,
                    dst_size,
                );
            }
            Ok(())
        } else {
            Err("CUDA copy not yet implemented".to_string())
        }
    }

    pub fn clone(&self) -> Result<Self, String> {
        let new_storage = Self::new(self.size, self.device)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.data.as_ptr(),
                new_storage.data.as_ptr(),
                self.size,
            );
        }
        Ok(new_storage)
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if self.device.is_cpu() {
            unsafe {
                dealloc(self.data.as_ptr(), self.layout);
            }
        }
    }
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}
