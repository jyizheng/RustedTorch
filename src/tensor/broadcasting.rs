use crate::tensor::{Tensor, TensorImpl, Options, DType};
use std::rc::Rc;

pub fn is_broadcastable(shape1: &[i64], shape2: &[i64]) -> bool {
    let max_dims = shape1.len().max(shape2.len());
    
    for i in 0..max_dims {
        let dim1 = if i < shape1.len() { shape1[shape1.len() - 1 - i] } else { 1 };
        let dim2 = if i < shape2.len() { shape2[shape2.len() - 1 - i] } else { 1 };
        
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }
    true
}

pub fn broadcast_shapes(shape1: &[i64], shape2: &[i64]) -> Result<Vec<i64>, String> {
    if !is_broadcastable(shape1, shape2) {
        return Err(format!("Shapes {:?} and {:?} are not broadcastable", shape1, shape2));
    }
    
    let max_dims = shape1.len().max(shape2.len());
    let mut result_shape = Vec::with_capacity(max_dims);
    
    for i in 0..max_dims {
        let dim1 = if i < shape1.len() { shape1[shape1.len() - 1 - i] } else { 1 };
        let dim2 = if i < shape2.len() { shape2[shape2.len() - 1 - i] } else { 1 };
        result_shape.push(dim1.max(dim2));
    }
    
    result_shape.reverse();
    Ok(result_shape)
}

pub fn broadcast_tensor_data(data: &[f32], from_shape: &[i64], to_shape: &[i64]) -> Result<Vec<f32>, String> {
    if from_shape == to_shape {
        return Ok(data.to_vec());
    }
    
    let total_elements = to_shape.iter().product::<i64>() as usize;
    let mut result = Vec::with_capacity(total_elements);
    
    let from_strides = compute_strides(from_shape);
    let to_strides = compute_strides(to_shape);
    
    for i in 0..total_elements {
        let mut from_idx = 0;
        let mut temp_i = i;
        
        for (dim_idx, &to_stride) in to_strides.iter().enumerate() {
            let coord = temp_i / to_stride as usize;
            temp_i %= to_stride as usize;
            
            let from_dim_offset = to_shape.len() - from_shape.len();
            if dim_idx >= from_dim_offset {
                let from_dim_idx = dim_idx - from_dim_offset;
                let from_coord = if from_shape[from_dim_idx] == 1 { 0 } else { coord };
                from_idx += from_coord * from_strides[from_dim_idx] as usize;
            }
        }
        
        if from_idx < data.len() {
            result.push(data[from_idx]);
        } else {
            return Err("Index out of bounds during broadcasting".to_string());
        }
    }
    
    Ok(result)
}

fn compute_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1;
    
    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    
    strides.reverse();
    strides
}

pub fn broadcast_tensors(tensor1: &Tensor, tensor2: &Tensor) -> Result<(Tensor, Tensor), String> {
    if !tensor1.defined() || !tensor2.defined() {
        return Err("Cannot broadcast undefined tensors".to_string());
    }
    
    let shape1 = tensor1.shape();
    let shape2 = tensor2.shape();
    
    let result_shape = broadcast_shapes(&shape1, &shape2)?;
    
    let broadcasted1 = if shape1 == result_shape {
        tensor1.clone()
    } else {
        let data1 = tensor1.to_list::<f32>();
        let broadcasted_data1 = broadcast_tensor_data(&data1, &shape1, &result_shape)?;
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&broadcasted_data1, &result_shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(Rc::new(impl_)),
            },
            Err(e) => return Err(format!("Failed to create broadcasted tensor: {}", e)),
        }
    };
    
    let broadcasted2 = if shape2 == result_shape {
        tensor2.clone()
    } else {
        let data2 = tensor2.to_list::<f32>();
        let broadcasted_data2 = broadcast_tensor_data(&data2, &shape2, &result_shape)?;
        let options = Options::default().dtype(DType::Float32);
        match TensorImpl::new_from_data(&broadcasted_data2, &result_shape, options) {
            Ok(impl_) => Tensor {
                impl_: Some(Rc::new(impl_)),
            },
            Err(e) => return Err(format!("Failed to create broadcasted tensor: {}", e)),
        }
    };
    
    Ok((broadcasted1, broadcasted2))
}
