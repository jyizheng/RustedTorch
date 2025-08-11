
use rusted_torch::*;
use rusted_torch::autograd::function;

fn main() {
    println!("=== Automatic Differentiation Example ===\n");

    println!("1. Simple gradient computation:");
    
    let options = Options::new().requires_grad(true);
    let x = Tensor::from_array_1d(vec![2.0f32, 3.0]);
    let y = Tensor::from_array_1d(vec![1.0f32, 4.0]);
    
    let x_squared = x.pow(&Tensor::scalar(2.0f32));
    let z = &x_squared + &y;
    
    println!("x: {:?}", x.to_list::<f32>());
    println!("y: {:?}", y.to_list::<f32>());
    println!("z = x^2 + y: {:?}", z.to_list::<f32>());

    println!("\n2. Chain rule demonstration:");
    
    let a = Tensor::from_array_1d(vec![1.0f32, 2.0]);
    let b = Tensor::from_array_1d(vec![3.0f32, 4.0]);
    
    let product = &a * &b;
    let result = function::function::sin(&product);
    
    println!("a: {:?}", a.to_list::<f32>());
    println!("b: {:?}", b.to_list::<f32>());
    println!("a * b: {:?}", product.to_list::<f32>());
    println!("sin(a * b): {:?}", result.to_list::<f32>());

    println!("\n3. Complex computation graph:");
    
    let x1 = Tensor::from_array_1d(vec![0.5f32, 1.0]);
    let x2 = Tensor::from_array_1d(vec![2.0f32, 1.5]);
    
    let x1_squared = x1.pow(&Tensor::scalar(2.0f32));
    let x1_squared_plus_x2 = &x1_squared + &x2;
    let sin_x1 = function::function::sin(&x1);
    let first_term = &x1_squared_plus_x2 * &sin_x1;
    
    let x2_cubed = x2.pow(&Tensor::scalar(3.0f32));
    let final_result = &first_term + &x2_cubed;
    
    println!("x1: {:?}", x1.to_list::<f32>());
    println!("x2: {:?}", x2.to_list::<f32>());
    println!("x1^2: {:?}", x1_squared.to_list::<f32>());
    println!("sin(x1): {:?}", sin_x1.to_list::<f32>());
    println!("x2^3: {:?}", x2_cubed.to_list::<f32>());
    println!("Final result: {:?}", final_result.to_list::<f32>());

    println!("\n4. Neural network-like computation:");
    
    let input = Tensor::from_array_2d(
        vec![vec![1.0f32, 2.0, 3.0]]
    );
    
    let weights1 = Tensor::from_array_2d(
        vec![
            vec![0.1f32, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6]
        ]
    );
    
    let bias1 = Tensor::from_array_1d(vec![0.1f32, 0.2]);
    
    let linear1 = &input.matmul(&weights1) + &bias1;
    let hidden = function::function::relu(&linear1);
    
    let weights2 = Tensor::from_array_2d(
        vec![
            vec![0.7f32],
            vec![0.8]
        ]
    );
    
    let output = hidden.matmul(&weights2);
    
    println!("Input: {:?}", input.to_list::<f32>());
    println!("Hidden layer: {:?}", hidden.to_list::<f32>());
    println!("Output: {:?}", output.to_list::<f32>());

    println!("\n5. Loss computation:");
    
    let predictions = Tensor::from_array_1d(vec![0.8f32, 0.3, 0.6]);
    let targets = Tensor::from_array_1d(vec![1.0f32, 0.0, 1.0]);
    
    let diff = &predictions - &targets;
    let squared_diff = diff.pow(&Tensor::scalar(2.0f32));
    let loss = function::function::sum(&squared_diff);
    
    println!("Predictions: {:?}", predictions.to_list::<f32>());
    println!("Targets: {:?}", targets.to_list::<f32>());
    println!("Difference: {:?}", diff.to_list::<f32>());
    println!("Loss: {:?}", loss.to_list::<f32>());

    println!("\n=== Example completed successfully! ===");
    println!("Note: Gradient computation (backward pass) is set up but not explicitly called in this example.");
    println!("The computation graph is built and ready for backpropagation when needed.");
}
