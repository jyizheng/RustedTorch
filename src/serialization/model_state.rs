use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ModelState {
    parameters: HashMap<String, Tensor>,
    pub(crate) metadata: HashMap<String, String>,
}

impl ModelState {
    pub fn new() -> Self {
        ModelState {
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn add_parameter(&mut self, name: String, tensor: Tensor) {
        self.parameters.insert(name, tensor);
    }
    
    pub fn get_parameter(&self, name: &str) -> Option<&Tensor> {
        self.parameters.get(name)
    }
    
    pub fn get_parameter_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.parameters.get_mut(name)
    }
    
    pub fn remove_parameter(&mut self, name: &str) -> Option<Tensor> {
        self.parameters.remove(name)
    }
    
    pub fn parameter_names(&self) -> Vec<&String> {
        self.parameters.keys().collect()
    }
    
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }
    
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);
        
        writer.write_all(b"RTORCH01").map_err(|e| format!("Failed to write header: {}", e))?;
        
        let metadata_count = self.metadata.len() as u32;
        writer.write_all(&metadata_count.to_le_bytes()).map_err(|e| format!("Failed to write metadata count: {}", e))?;
        
        for (key, value) in &self.metadata {
            let key_bytes = key.as_bytes();
            let value_bytes = value.as_bytes();
            
            writer.write_all(&(key_bytes.len() as u32).to_le_bytes()).map_err(|e| format!("Failed to write key length: {}", e))?;
            writer.write_all(key_bytes).map_err(|e| format!("Failed to write key: {}", e))?;
            
            writer.write_all(&(value_bytes.len() as u32).to_le_bytes()).map_err(|e| format!("Failed to write value length: {}", e))?;
            writer.write_all(value_bytes).map_err(|e| format!("Failed to write value: {}", e))?;
        }
        
        let param_count = self.parameters.len() as u32;
        writer.write_all(&param_count.to_le_bytes()).map_err(|e| format!("Failed to write parameter count: {}", e))?;
        
        for (name, tensor) in &self.parameters {
            if !tensor.defined() {
                return Err(format!("Parameter '{}' is not defined", name));
            }
            
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(|e| format!("Failed to write parameter name length: {}", e))?;
            writer.write_all(name_bytes).map_err(|e| format!("Failed to write parameter name: {}", e))?;
            
            let shape = tensor.shape();
            writer.write_all(&(shape.len() as u32).to_le_bytes()).map_err(|e| format!("Failed to write shape length: {}", e))?;
            for dim in &shape {
                writer.write_all(&dim.to_le_bytes()).map_err(|e| format!("Failed to write shape dimension: {}", e))?;
            }
            
            let data = tensor.to_list::<f32>();
            writer.write_all(&(data.len() as u32).to_le_bytes()).map_err(|e| format!("Failed to write data length: {}", e))?;
            for value in &data {
                writer.write_all(&value.to_le_bytes()).map_err(|e| format!("Failed to write data value: {}", e))?;
            }
        }
        
        writer.flush().map_err(|e| format!("Failed to flush writer: {}", e))?;
        Ok(())
    }
    
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);
        
        let mut header = [0u8; 8];
        reader.read_exact(&mut header).map_err(|e| format!("Failed to read header: {}", e))?;
        if &header != b"RTORCH01" {
            return Err("Invalid file format: wrong magic header".to_string());
        }
        
        let mut state = ModelState::new();
        
        let mut metadata_count_bytes = [0u8; 4];
        reader.read_exact(&mut metadata_count_bytes).map_err(|e| format!("Failed to read metadata count: {}", e))?;
        let metadata_count = u32::from_le_bytes(metadata_count_bytes);
        
        for _ in 0..metadata_count {
            let mut key_len_bytes = [0u8; 4];
            reader.read_exact(&mut key_len_bytes).map_err(|e| format!("Failed to read key length: {}", e))?;
            let key_len = u32::from_le_bytes(key_len_bytes) as usize;
            
            let mut key_bytes = vec![0u8; key_len];
            reader.read_exact(&mut key_bytes).map_err(|e| format!("Failed to read key: {}", e))?;
            let key = String::from_utf8(key_bytes).map_err(|e| format!("Invalid UTF-8 in key: {}", e))?;
            
            let mut value_len_bytes = [0u8; 4];
            reader.read_exact(&mut value_len_bytes).map_err(|e| format!("Failed to read value length: {}", e))?;
            let value_len = u32::from_le_bytes(value_len_bytes) as usize;
            
            let mut value_bytes = vec![0u8; value_len];
            reader.read_exact(&mut value_bytes).map_err(|e| format!("Failed to read value: {}", e))?;
            let value = String::from_utf8(value_bytes).map_err(|e| format!("Invalid UTF-8 in value: {}", e))?;
            
            state.add_metadata(key, value);
        }
        
        let mut param_count_bytes = [0u8; 4];
        reader.read_exact(&mut param_count_bytes).map_err(|e| format!("Failed to read parameter count: {}", e))?;
        let param_count = u32::from_le_bytes(param_count_bytes);
        
        for _ in 0..param_count {
            let mut name_len_bytes = [0u8; 4];
            reader.read_exact(&mut name_len_bytes).map_err(|e| format!("Failed to read parameter name length: {}", e))?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;
            
            let mut name_bytes = vec![0u8; name_len];
            reader.read_exact(&mut name_bytes).map_err(|e| format!("Failed to read parameter name: {}", e))?;
            let name = String::from_utf8(name_bytes).map_err(|e| format!("Invalid UTF-8 in parameter name: {}", e))?;
            
            let mut shape_len_bytes = [0u8; 4];
            reader.read_exact(&mut shape_len_bytes).map_err(|e| format!("Failed to read shape length: {}", e))?;
            let shape_len = u32::from_le_bytes(shape_len_bytes) as usize;
            
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let mut dim_bytes = [0u8; 8];
                reader.read_exact(&mut dim_bytes).map_err(|e| format!("Failed to read shape dimension: {}", e))?;
                shape.push(i64::from_le_bytes(dim_bytes));
            }
            
            let mut data_len_bytes = [0u8; 4];
            reader.read_exact(&mut data_len_bytes).map_err(|e| format!("Failed to read data length: {}", e))?;
            let data_len = u32::from_le_bytes(data_len_bytes) as usize;
            
            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                let mut value_bytes = [0u8; 4];
                reader.read_exact(&mut value_bytes).map_err(|e| format!("Failed to read data value: {}", e))?;
                data.push(f32::from_le_bytes(value_bytes));
            }
            
            let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
            let tensor = match crate::tensor::TensorImpl::new_from_data(&data, &shape, options) {
                Ok(impl_) => Tensor {
                    impl_: Some(std::rc::Rc::new(impl_)),
                },
                Err(e) => return Err(format!("Failed to create tensor '{}': {}", name, e)),
            };
            
            state.add_parameter(name, tensor);
        }
        
        Ok(state)
    }
}

impl Default for ModelState {
    fn default() -> Self {
        Self::new()
    }
}
