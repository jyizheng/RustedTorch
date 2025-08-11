use crate::tensor::{check_dtype_match, Device, DType, Options, Storage, TypeToDType};
use crate::autograd::AutogradMeta;
use std::rc::Rc;
use std::cell::RefCell;

pub type IntArrayView = [i64];
pub type SizeVector = Vec<i64>;

#[derive(Debug)]
pub struct TensorImpl {
    shape: SizeVector,
    strides: SizeVector,
    numel: i64,
    storage_offset: i64,
    options: Options,
    storage: Option<Rc<Storage>>,
    pub autograd_meta: Option<Rc<RefCell<AutogradMeta>>>,
}

impl TensorImpl {
    pub fn new(shape: &IntArrayView, options: Options) -> Result<Self, String> {
        let autograd_meta = if options.requires_grad_value() {
            Some(Rc::new(RefCell::new(AutogradMeta::new())))
        } else {
            None
        };

        let mut impl_ = Self {
            shape: shape.to_vec(),
            strides: Vec::new(),
            numel: 0,
            storage_offset: 0,
            options,
            storage: None,
            autograd_meta,
        };

        Self::compute_strides(&mut impl_.strides, &impl_.shape);
        Self::compute_numel(&mut impl_.numel, &impl_.shape);
        impl_.ensure_storage()?;

        Ok(impl_)
    }

    pub fn new_with_storage(
        shape: &IntArrayView,
        options: Options,
        storage: Rc<Storage>,
        offset: i64,
    ) -> Result<Self, String> {
        let autograd_meta = if options.requires_grad_value() {
            Some(Rc::new(RefCell::new(AutogradMeta::new())))
        } else {
            None
        };

        let mut impl_ = Self {
            shape: shape.to_vec(),
            strides: Vec::new(),
            numel: 0,
            storage_offset: offset,
            options,
            storage: Some(storage),
            autograd_meta,
        };

        Self::compute_strides(&mut impl_.strides, &impl_.shape);
        Self::compute_numel(&mut impl_.numel, &impl_.shape);

        Ok(impl_)
    }

    pub fn new_from_data<T: TypeToDType + Clone>(
        data: &[T],
        shape: &IntArrayView,
        options: Options,
    ) -> Result<Self, String> {
        check_dtype_match::<T>(options.dtype)?;

        let mut impl_ = Self::new(shape, options)?;
        
        if data.len() != impl_.numel as usize {
            return Err(format!(
                "Data length {} doesn't match tensor numel {}",
                data.len(),
                impl_.numel
            ));
        }

        if let Some(ref mut storage) = impl_.storage {
            let storage_mut = Rc::get_mut(storage)
                .ok_or("Cannot get mutable reference to storage")?;
            storage_mut.copy_from_slice(data)?;
        }

        Ok(impl_)
    }

    pub fn dtype(&self) -> DType {
        self.options.dtype
    }

    pub fn device(&self) -> Device {
        self.options.device
    }

    pub fn pinned_memory(&self) -> bool {
        self.options.pinned_memory
    }

    pub fn requires_grad(&self) -> bool {
        self.options.requires_grad
    }

    pub fn dim(&self) -> i64 {
        self.shape.len() as i64
    }

    pub fn numel(&self) -> i64 {
        self.numel
    }

    pub fn storage_offset(&self) -> i64 {
        self.storage_offset
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn data_ptr<T>(&self) -> *mut T {
        if let Some(ref storage) = self.storage {
            storage.data_ptr::<T>()
        } else {
            std::ptr::null_mut()
        }
    }

    pub fn options(&self) -> &Options {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut Options {
        &mut self.options
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    pub fn shape_at(&self, dim: i64) -> i64 {
        let idx = if dim < 0 { dim + self.dim() } else { dim };
        self.shape[idx as usize]
    }

    pub fn stride_at(&self, dim: i64) -> i64 {
        let idx = if dim < 0 { dim + self.dim() } else { dim };
        self.strides[idx as usize]
    }

    pub fn storage(&self) -> Option<&Rc<Storage>> {
        self.storage.as_ref()
    }

    pub fn set_storage(&mut self, storage: Rc<Storage>, offset: i64) {
        self.storage = Some(storage);
        self.storage_offset = offset;
    }

    pub fn reshape_(&mut self, shape: &IntArrayView) -> Result<(), String> {
        let new_numel = shape.iter().product::<i64>();
        if new_numel != self.numel {
            return Err(format!(
                "Cannot reshape tensor of size {} to size {}",
                self.numel, new_numel
            ));
        }

        self.shape = shape.to_vec();
        Self::compute_strides(&mut self.strides, &self.shape);
        Ok(())
    }

    pub fn flatten_(&mut self, start_dim: i64, end_dim: i64) -> Result<(), String> {
        let start = if start_dim < 0 { start_dim + self.dim() } else { start_dim };
        let end = if end_dim < 0 { end_dim + self.dim() } else { end_dim };

        if start < 0 || end >= self.dim() || start > end {
            return Err("Invalid flatten dimensions".to_string());
        }

        let mut new_shape = Vec::new();
        
        for i in 0..start {
            new_shape.push(self.shape[i as usize]);
        }

        let flattened_size: i64 = (start..=end)
            .map(|i| self.shape[i as usize])
            .product();
        new_shape.push(flattened_size);

        for i in (end + 1)..self.dim() {
            new_shape.push(self.shape[i as usize]);
        }

        self.shape = new_shape;
        Self::compute_strides(&mut self.strides, &self.shape);
        Ok(())
    }

    pub fn squeeze_(&mut self, dim: Option<i64>) -> Result<(), String> {
        match dim {
            Some(d) => {
                let idx = if d < 0 { d + self.dim() } else { d };
                if idx < 0 || idx >= self.dim() {
                    return Err("Dimension out of range".to_string());
                }
                if self.shape[idx as usize] == 1 {
                    self.shape.remove(idx as usize);
                    self.strides.remove(idx as usize);
                }
            }
            None => {
                let mut i = 0;
                while i < self.shape.len() {
                    if self.shape[i] == 1 {
                        self.shape.remove(i);
                        self.strides.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn unsqueeze_(&mut self, dim: i64) -> Result<(), String> {
        let idx = if dim < 0 { dim + self.dim() + 1 } else { dim };
        if idx < 0 || idx > self.dim() {
            return Err("Dimension out of range".to_string());
        }

        self.shape.insert(idx as usize, 1);
        let stride = if idx as usize == self.strides.len() {
            1
        } else {
            self.strides[idx as usize]
        };
        self.strides.insert(idx as usize, stride);
        Ok(())
    }

    pub fn to_list<T: TypeToDType + Clone + Default>(&self) -> Result<Vec<T>, String> {
        check_dtype_match::<T>(self.dtype())?;

        if self.device().is_cpu() {
            let ptr = self.data_ptr::<T>();
            if ptr.is_null() {
                return Err("Null data pointer".to_string());
            }
            
            let mut result = vec![T::default(); self.numel as usize];
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), self.numel as usize);
            }
            Ok(result)
        } else {
            Err("CUDA tensor to_list not yet implemented".to_string())
        }
    }

    pub fn item<T: TypeToDType + Clone + Default>(&self) -> Result<T, String> {
        if self.numel != 1 {
            return Err("item() can only be called on tensors with exactly one element".to_string());
        }

        check_dtype_match::<T>(self.dtype())?;

        if self.device().is_cpu() {
            let ptr = self.data_ptr::<T>();
            if ptr.is_null() {
                return Err("Null data pointer".to_string());
            }
            
            unsafe { Ok(ptr.read()) }
        } else {
            Err("CUDA tensor item not yet implemented".to_string())
        }
    }

    fn ensure_storage(&mut self) -> Result<(), String> {
        if self.storage.is_none() {
            let size = (self.numel as usize) * self.options.dtype.size();
            let storage = Storage::new(size, self.options.device)?;
            self.storage = Some(Rc::new(storage));
        }
        Ok(())
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if requires_grad && self.autograd_meta.is_none() {
            self.autograd_meta = Some(Rc::new(RefCell::new(AutogradMeta::new())));
        }
        if let Some(ref autograd_meta) = self.autograd_meta {
            autograd_meta.borrow_mut().set_requires_grad(requires_grad);
        }
    }

    fn compute_strides(strides: &mut SizeVector, shape: &[i64]) {
        strides.clear();
        strides.resize(shape.len(), 1);

        if !shape.is_empty() {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
    }

    fn compute_numel(numel: &mut i64, shape: &[i64]) {
        *numel = shape.iter().product();
    }
}

impl Default for TensorImpl {
    fn default() -> Self {
        Self {
            shape: Vec::new(),
            strides: Vec::new(),
            numel: 0,
            storage_offset: 0,
            options: Options::default(),
            storage: None,
            autograd_meta: None,
        }
    }
}
