# RustedTorch Examples

This directory contains comprehensive examples demonstrating how to use the RustedTorch library for deep learning tasks. Each example focuses on different aspects of the library and can be run independently.

## Available Examples

### 1. Basic Tensor Operations (`basic_tensor_operations.rs`)
Demonstrates fundamental tensor operations including:
- Creating tensors (zeros, ones, random, from arrays)
- Basic arithmetic operations (+, -, *, /)
- Tensor properties and manipulation (shape, size, reshape, transpose)
- Scalar operations

**Run with:**
```bash
cargo run --example basic_tensor_operations
```

### 2. Mathematical Functions (`mathematical_functions.rs`)
Shows various mathematical functions available in RustedTorch:
- Trigonometric functions (sin, cos)
- Activation functions (ReLU, GELU, SiLU)
- Softmax and log_softmax
- Power and exponential functions
- Reduction operations
- Element-wise operations

**Run with:**
```bash
cargo run --example mathematical_functions
```

### 3. Neural Network Basics (`neural_network_basics.rs`)
Demonstrates building neural network components:
- Linear layer simulation
- Activation functions in neural networks
- Loss functions (MSE, NLL)
- Dropout for regularization
- Simple forward pass example

**Run with:**
```bash
cargo run --example neural_network_basics
```

### 4. Automatic Differentiation (`autograd_example.rs`)
Shows the automatic differentiation capabilities:
- Simple gradient computation setup
- Chain rule demonstration
- Complex computation graphs
- Neural network-like computations with gradients
- Loss computation for training

**Run with:**
```bash
cargo run --example autograd_example
```

## Running All Examples

To run all examples at once:

```bash
# Run each example individually
cargo run --example basic_tensor_operations
cargo run --example mathematical_functions
cargo run --example neural_network_basics
cargo run --example autograd_example
```

## Key Features Demonstrated

### Tensor Creation and Manipulation
- Multiple ways to create tensors
- Reshaping and transformation operations
- Data type handling

### Mathematical Operations
- Element-wise operations
- Matrix operations (matmul, transpose)
- Reduction operations (sum, mean)
- Broadcasting support

### Neural Network Components
- Linear layers
- Activation functions
- Loss functions
- Regularization techniques

### Automatic Differentiation
- Computation graph construction
- Gradient-enabled tensors
- Complex mathematical expressions
- Neural network forward passes

## Code Style and Patterns

The examples follow Rust best practices and demonstrate:
- Proper error handling with Result types
- Memory-safe tensor operations
- Idiomatic Rust patterns
- Clear documentation and comments

## Integration with Main Library

These examples use the public API of RustedTorch and demonstrate real-world usage patterns. They serve as both documentation and testing for the library's functionality.

## Next Steps

After running these examples, you can:
1. Modify the examples to experiment with different parameters
2. Combine concepts from multiple examples
3. Build more complex neural networks
4. Implement custom training loops
5. Explore advanced features like custom loss functions

For more detailed API documentation, see the main library documentation.
