use rusted_torch::*;

fn main() {
    println!("Running RustedTorch tests...");
    
    println!("Testing tensor creation...");
    let x = Tensor::ones(&[2, 3]);
    println!("Created tensor with shape: {:?}", x.shape());
    println!("Tensor data: {:?}", x.to_list::<f32>());
    
    println!("Testing tensor operations...");
    let a = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
    let b = Tensor::from_array_1d(vec![4.0f32, 5.0, 6.0]);
    let c = &a + &b;
    println!("Addition result: {:?}", c.to_list::<f32>());
    
    println!("Testing mathematical functions...");
    let x = Tensor::from_array_1d(vec![0.0f32, std::f32::consts::PI / 2.0]);
    let sin_x = autograd::function::function::sin(&x);
    println!("Sin result: {:?}", sin_x.to_list::<f32>());
    
    println!("All basic tests completed successfully!");
}
