
use rusted_torch::*;
use rusted_torch::autograd::function;
use rusted_torch::functions::{mse_loss, nll_loss, dropout, LossReduction};

fn main() {
    println!("=== Neural Network Basics Example ===\n");

    println!("1. Linear layer simulation:");
    
    let input = Tensor::from_array_2d(vec![
        vec![1.0f32, 2.0, 3.0],  // Sample 1
        vec![4.0, 5.0, 6.0],     // Sample 2
    ]);
    
    let weights = Tensor::from_array_2d(vec![
        vec![0.1f32, 0.2],       // Input feature 1 -> Output neurons
        vec![0.3, 0.4],          // Input feature 2 -> Output neurons  
        vec![0.5, 0.6],          // Input feature 3 -> Output neurons
    ]);
    
    let bias = Tensor::from_array_1d(vec![0.1f32, 0.2]);
    
    let linear_output = input.matmul(&weights);
    let output_with_bias = &linear_output + &bias;
    
    println!("Input shape: {:?}", input.shape());
    println!("Weights shape: {:?}", weights.shape());
    println!("Linear output: {:?}", linear_output.to_list::<f32>());
    println!("Output with bias: {:?}", output_with_bias.to_list::<f32>());

    println!("\n2. Activation functions:");
    
    let hidden_layer = Tensor::from_array_2d(vec![
        vec![-1.0f32, 0.5, 2.0],
        vec![1.5, -0.5, 0.0]
    ]);
    
    let relu_activated = function::function::relu(&hidden_layer);
    let gelu_activated = function::function::gelu(&hidden_layer);
    
    println!("Hidden layer: {:?}", hidden_layer.to_list::<f32>());
    println!("ReLU activated: {:?}", relu_activated.to_list::<f32>());
    println!("GELU activated: {:?}", gelu_activated.to_list::<f32>());

    println!("\n3. Loss functions:");
    
    let predictions = Tensor::from_array_2d(vec![
        vec![0.8f32, 0.2, 0.1],
        vec![0.1, 0.7, 0.3]
    ]);
    let targets = Tensor::from_array_2d(vec![
        vec![1.0f32, 0.0, 0.0],
        vec![0.0, 1.0, 0.0]
    ]);
    
    let mse_loss_value = mse_loss(&predictions, &targets, LossReduction::Mean);
    println!("Predictions: {:?}", predictions.to_list::<f32>());
    println!("Targets: {:?}", targets.to_list::<f32>());
    println!("MSE Loss: {:.6}", mse_loss_value.item::<f32>());
    
    let log_probs = Tensor::from_array_2d(vec![
        vec![-0.2f32, -1.6, -2.3],  // Log probabilities
        vec![-2.3, -0.4, -1.2]
    ]);
    let class_targets = Tensor::from_array_1d(vec![0i64, 1i64]);  // Class indices
    
    let nll_loss_value = nll_loss(&log_probs, &class_targets, LossReduction::Mean);
    println!("Log probabilities: {:?}", log_probs.to_list::<f32>());
    println!("Class targets: {:?}", class_targets.to_list::<i64>());
    println!("NLL Loss: {:.6}", nll_loss_value.item::<f32>());

    println!("\n4. Dropout regularization:");
    
    let layer_output = Tensor::ones(&[4, 5]);
    let dropout_prob = 0.3f32;
    
    let training_output = dropout(&layer_output, dropout_prob, true);
    println!("Original output: {:?}", layer_output.to_list::<f32>());
    println!("With dropout (training): {:?}", training_output.to_list::<f32>());
    
    let inference_output = dropout(&layer_output, dropout_prob, false);
    println!("Without dropout (inference): {:?}", inference_output.to_list::<f32>());

    println!("\n5. Simple neural network forward pass:");
    
    let input_data = Tensor::from_array_2d(vec![
        vec![0.5f32, -0.2, 0.8]
    ]);
    
    let w1 = Tensor::from_array_2d(vec![
        vec![0.1f32, 0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7, 0.8],
        vec![0.9, 1.0, 1.1, 1.2]
    ]);
    let b1 = Tensor::from_array_1d(vec![0.1f32, 0.1, 0.1, 0.1]);
    
    let hidden = function::function::relu(&(&input_data.matmul(&w1) + &b1));
    
    let w2 = Tensor::from_array_2d(vec![
        vec![0.1f32, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
        vec![1.0, 1.1, 1.2]
    ]);
    let b2 = Tensor::from_array_1d(vec![0.1f32, 0.1, 0.1]);
    
    let output = &hidden.matmul(&w2) + &b2;
    let probabilities = function::function::softmax(&output, 1);
    
    println!("Input: {:?}", input_data.to_list::<f32>());
    println!("Hidden layer: {:?}", hidden.to_list::<f32>());
    println!("Raw output: {:?}", output.to_list::<f32>());
    println!("Probabilities: {:?}", probabilities.to_list::<f32>());

    println!("\n=== Example completed successfully! ===");
}
