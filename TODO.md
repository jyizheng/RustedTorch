# RustedTorch Development Roadmap

This document outlines the prioritized features to implement in RustedTorch to make it a practical deep learning framework.

## 🚀 **HIGH PRIORITY** (Immediate Impact)

### ✅ 1. **Optimizers** (Essential for training) - COMPLETED
- [x] SGD with momentum support
- [x] Adam optimizer (most commonly used)
- [x] AdamW (improved Adam variant) - PR #5

**Why first**: Without optimizers, you can't actually train neural networks. This is the biggest gap preventing practical use.

**Implementation complexity**: Medium - requires parameter updates, momentum tracking, and learning rate scheduling.

### ✅ 2. **Proper Backward Pass Implementation** - COMPLETED
- [x] Implemented `backward()`, `grad()`, and `zero_grad()` methods - PR #3
- [x] Gradient computation and accumulation working
- [x] Autograd system functional with gradient tracking

**Why second**: The autograd system exists but doesn't work - this is critical for any learning.

### ✅ 3. **Broadcasting Support** - COMPLETED
- [x] Tensor operations with different shapes (e.g., `[3,1] + [1,4] = [3,4]`) - PR #3
- [x] Full PyTorch-compatible broadcasting semantics implemented

**Why third**: Many tensor operations fail without proper broadcasting, limiting practical use.

## 🎯 **MEDIUM PRIORITY** (Core Deep Learning)

### ✅ 4. **Convolutional Layers** - COMPLETED
- [x] `Conv2d` with basic implementation - PR #4
- [x] `MaxPool2d` implemented - PR #4
- [x] `BatchNorm2d` for training stability - PR #4

**Why important**: Essential for computer vision applications, which are very common.

### ✅ 5. **More Activation Functions** - COMPLETED
- [x] `Tanh`, `Sigmoid`, `LeakyReLU`, `Swish` - PR #4
- [x] Currently has ReLU, GELU, SiLU, Tanh, Sigmoid, LeakyReLU, Swish

### ✅ 6. **Additional Loss Functions** - COMPLETED
- [x] `CrossEntropyLoss` (combines softmax + NLL) - PR #4
- [x] `BCELoss` for binary classification - PR #4
- [x] `L1Loss` (Mean Absolute Error) - PR #4

## 🔧 **MEDIUM-LOW PRIORITY** (Quality of Life)

### ✅ 7. **Data Loading Utilities** - COMPLETED
- [x] `Dataset` trait for custom datasets - PR #4
- [x] `DataLoader` for batching and shuffling - PR #4
- [ ] Basic image/text preprocessing utilities

### ✅ 8. **Model Serialization** - COMPLETED
- [x] Save/load model weights with custom binary format - PR #4
- [x] Checkpoint saving during training with optimizer state - PR #4

### 9. **Recurrent Layers**
- [ ] `LSTM` and `GRU` for sequence modeling
- [ ] Bidirectional variants

## 🚀 **LONG-TERM** (Advanced Features)

### 10. **GPU/CUDA Support**
- [ ] CUDA tensor operations
- [ ] Device management (`tensor.to(device)`)
- [ ] Memory management for GPU

### 11. **Advanced Operations**
- [ ] Proper `einsum` implementation
- [ ] Advanced indexing and slicing
- [ ] Tensor concatenation and stacking

### 12. **Training Utilities**
- [ ] Learning rate schedulers
- [ ] Gradient clipping
- [ ] Mixed precision training

## 🔍 **Current Implementation Status**

### ✅ **Completed Features**
- [x] Basic tensor operations (creation, arithmetic, reshaping)
- [x] Core autograd infrastructure with functional backward pass
- [x] Broadcasting support for tensor operations
- [x] Comprehensive optimizers (SGD, Adam, AdamW)
- [x] Extended activation functions (ReLU, GELU, SiLU, Tanh, Sigmoid, LeakyReLU, Swish)
- [x] Extended loss functions (MSE, NLL, CrossEntropy, BCE, L1)
- [x] Convolutional layers (Conv2d, MaxPool2d, BatchNorm2d)
- [x] Data loading utilities (Dataset, DataLoader)
- [x] Model serialization and checkpointing
- [x] Comprehensive test suite (66 tests passing)
- [x] Usage examples and documentation

### 🚧 **Known Issues**
- GPU support absent (CPU-only implementation)
- Limited advanced indexing and slicing
- No learning rate schedulers
- No recurrent layers (LSTM, GRU)
- Custom binary format instead of PyTorch .pth compatibility

## 🎯 **Next Steps**

1. **Merge open PRs** - PR #3 (broadcasting/autograd), PR #4 (comprehensive features), PR #5 (AdamW)
2. **Implement recurrent layers** (LSTM, GRU) - enables sequence modeling
3. **Add learning rate schedulers** - improves training flexibility
4. **Implement advanced tensor operations** - einsum, advanced indexing
5. **Add GPU/CUDA support** - enables accelerated computation

---

*Last updated: August 11, 2025*
*Repository: https://github.com/jyizheng/RustedTorch*
