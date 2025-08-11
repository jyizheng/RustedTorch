
use rusted_torch::*;
use rusted_torch::autograd::function;

fn main() {
    println!("=== Mathematical Functions Example ===\n");

    println!("1. Trigonometric functions:");
    
    let angles = Tensor::from_array_1d(vec![
        0.0f32, 
        std::f32::consts::PI / 4.0,  // 45 degrees
        std::f32::consts::PI / 2.0,  // 90 degrees
        std::f32::consts::PI,        // 180 degrees
    ]);
    
    let sin_result = function::function::sin(&angles);
    let cos_result = function::function::cos(&angles);
    
    println!("Angles (radians): {:?}", angles.to_list::<f32>());
    println!("Sin values: {:?}", sin_result.to_list::<f32>());
    println!("Cos values: {:?}", cos_result.to_list::<f32>());

    println!("\n2. Activation functions:");
    
    let input = Tensor::from_array_1d(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    
    let relu_output = function::function::relu(&input);
    println!("Input: {:?}", input.to_list::<f32>());
    println!("ReLU: {:?}", relu_output.to_list::<f32>());
    
    let gelu_output = function::function::gelu(&input);
    println!("GELU: {:?}", gelu_output.to_list::<f32>());
    
    let silu_output = function::function::silu(&input);
    println!("SiLU: {:?}", silu_output.to_list::<f32>());

    println!("\n3. Softmax functions:");
    
    let logits = Tensor::from_array_1d(vec![1.0f32, 2.0, 3.0, 4.0]);
    
    let softmax_output = function::function::softmax(&logits, 0);
    let log_softmax_output = function::function::log_softmax(&logits, 0);
    
    println!("Logits: {:?}", logits.to_list::<f32>());
    println!("Softmax: {:?}", softmax_output.to_list::<f32>());
    println!("Log Softmax: {:?}", log_softmax_output.to_list::<f32>());
    
    let softmax_sum = function::function::sum(&softmax_output);
    println!("Softmax sum (should be ~1.0): {:?}", softmax_sum.to_list::<f32>());

    println!("\n4. Power and exponential functions:");
    
    let base = Tensor::from_array_1d(vec![2.0f32, 3.0, 4.0]);
    let exponent = Tensor::from_array_1d(vec![2.0f32, 3.0, 0.5]);
    
    let power_result = function::function::pow(&base, &exponent);
    println!("Base: {:?}", base.to_list::<f32>());
    println!("Exponent: {:?}", exponent.to_list::<f32>());
    println!("Power result: {:?}", power_result.to_list::<f32>());

    println!("\n5. Reduction operations:");
    
    let matrix = Tensor::from_array_2d(vec![
        vec![1.0f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0]
    ]);
    
    let sum_all = function::function::sum(&matrix);
    println!("Matrix: {:?}", matrix.to_list::<f32>());
    println!("Sum of all elements: {:?}", sum_all.to_list::<f32>());

    println!("\n6. Element-wise operations:");
    
    let tensor_a = Tensor::from_array_2d(vec![
        vec![1.0f32, 2.0],
        vec![3.0, 4.0]
    ]);
    let tensor_b = Tensor::from_array_2d(vec![
        vec![5.0f32, 6.0],
        vec![7.0, 8.0]
    ]);
    
    let add_result = function::function::add(&tensor_a, &tensor_b, 1.0);
    let mul_result = function::function::mul(&tensor_a, &tensor_b);
    let div_result = function::function::div(&tensor_a, &tensor_b);
    
    println!("Tensor A: {:?}", tensor_a.to_list::<f32>());
    println!("Tensor B: {:?}", tensor_b.to_list::<f32>());
    println!("A + B: {:?}", add_result.to_list::<f32>());
    println!("A * B: {:?}", mul_result.to_list::<f32>());
    println!("A / B: {:?}", div_result.to_list::<f32>());

    println!("\n=== Example completed successfully! ===");
}
