
use rusted_torch::*;

fn main() {
    println!("=== Basic Tensor Operations Example ===\n");

    println!("1. Creating tensors:");
    
    let zeros = Tensor::zeros(&[2, 3]);
    println!("Zeros tensor (2x3): {:?}", zeros.to_list::<f32>());
    
    let ones = Tensor::ones(&[2, 3]);
    println!("Ones tensor (2x3): {:?}", ones.to_list::<f32>());
    
    let random = Tensor::rand(&[2, 3]);
    println!("Random tensor (2x3): {:?}", random.to_list::<f32>());
    
    let from_1d = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0, 4.0]);
    println!("From 1D array: {:?}", from_1d.to_list::<f32>());
    
    let from_2d = Tensor::from_array_2d(vec![
        vec![1.0f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0]
    ]);
    println!("From 2D array: {:?}", from_2d.to_list::<f32>());

    println!("\n2. Basic arithmetic operations:");
    
    let a = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
    let b = Tensor::from_array_1d(vec![4.0f32, 5.0, 6.0]);
    
    let add_result = &a + &b;
    println!("a + b = {:?}", add_result.to_list::<f32>());
    
    let sub_result = &a - &b;
    println!("a - b = {:?}", sub_result.to_list::<f32>());
    
    let mul_result = &a * &b;
    println!("a * b = {:?}", mul_result.to_list::<f32>());
    
    let div_result = &a / &b;
    println!("a / b = {:?}", div_result.to_list::<f32>());

    println!("\n3. Tensor properties and manipulation:");
    
    let matrix = Tensor::from_array_2d(vec![
        vec![1.0f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0]
    ]);
    
    println!("Matrix shape: {:?}", matrix.shape());
    println!("Matrix size: {}", matrix.size());
    println!("Matrix data: {:?}", matrix.to_list::<f32>());
    
    let reshaped = matrix.reshape(&[3, 2]);
    println!("Reshaped to (3x2): {:?}", reshaped.to_list::<f32>());
    
    let flattened = matrix.flatten();
    println!("Flattened: {:?}", flattened.to_list::<f32>());
    
    let transposed = matrix.transpose(0, 1);
    println!("Transposed: {:?}", transposed.to_list::<f32>());

    println!("\n4. Scalar operations:");
    
    let tensor = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0]);
    let scalar = Tensor::scalar(2.0f32);
    
    let scaled = &tensor * &scalar;
    println!("Tensor * 2 = {:?}", scaled.to_list::<f32>());
    
    let powered = tensor.pow(&scalar);
    println!("Tensor^2 = {:?}", powered.to_list::<f32>());

    println!("\n=== Example completed successfully! ===");
}
