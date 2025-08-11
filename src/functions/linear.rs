use crate::tensor::Tensor;

pub fn linear(_input: &Tensor, _weight: &Tensor, _bias: Option<&Tensor>) -> Tensor {
    Tensor::new()
}

pub fn dropout(input: &Tensor, p: f32, training: bool) -> Tensor {
    if !training {
        return input.clone();
    }

    if !input.defined() {
        return Tensor::new();
    }

    let data = input.to_list::<f32>();
    let mut rng = rand::thread_rng();
    use rand::Rng;
    
    let scale = 1.0 / (1.0 - p);
    let result_data: Vec<f32> = data.iter().map(|&val| {
        if rng.gen::<f32>() < p {
            0.0
        } else {
            val * scale
        }
    }).collect();

    let shape = input.shape();
    let options = crate::tensor::Options::default().dtype(crate::tensor::DType::Float32);
    match crate::tensor::TensorImpl::new_from_data(&result_data, &shape, options) {
        Ok(impl_) => Tensor {
            impl_: Some(std::rc::Rc::new(impl_)),
        },
        Err(_) => Tensor::new(),
    }
}
