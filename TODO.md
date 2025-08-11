# RustedTorch Development Roadmap

This document outlines the prioritized features to implement in RustedTorch to make it a practical deep learning framework.

## 🚀 **HIGH PRIORITY** (Immediate Impact)

### ✅ 1. **Optimizers** (Essential for training) - IN PROGRESS
- [x] SGD with momentum support
- [x] Adam optimizer (most commonly used)
- [ ] AdamW (improved Adam variant)

**Why first**: Without optimizers, you can't actually train neural networks. This is the biggest gap preventing practical use.

**Implementation complexity**: Medium - requires parameter updates, momentum tracking, and learning rate scheduling.

### 2. **Proper Backward Pass Implementation** 
- [ ] Currently `backward()`, `grad()`, and `zero_grad()` are empty stubs
- [ ] Implement gradient computation and accumulation
- [ ] Fix the autograd system to actually compute gradients

**Why second**: The autograd system exists but doesn't work - this is critical for any learning.

### 3. **Broadcasting Support**
- [ ] Tensor operations with different shapes (e.g., `[3,1] + [1,4] = [3,4]`)
- [ ] Currently operations assume same shapes

**Why third**: Many tensor operations fail without proper broadcasting, limiting practical use.

## 🎯 **MEDIUM PRIORITY** (Core Deep Learning)

### 4. **Convolutional Layers**
- [ ] `Conv2d` with padding, stride, dilation options
- [ ] `MaxPool2d` and `AvgPool2d`
- [ ] `BatchNorm2d` for training stability

**Why important**: Essential for computer vision applications, which are very common.

### 5. **More Activation Functions**
- [ ] `Tanh`, `Sigmoid`, `LeakyReLU`, `Swish`
- [x] Currently has ReLU, GELU, SiLU

### 6. **Additional Loss Functions**
- [ ] `CrossEntropyLoss` (combines softmax + NLL)
- [ ] `BCELoss` for binary classification
- [ ] `L1Loss` (Mean Absolute Error)

## 🔧 **MEDIUM-LOW PRIORITY** (Quality of Life)

### 7. **Data Loading Utilities**
- [ ] `Dataset` trait for custom datasets
- [ ] `DataLoader` for batching and shuffling
- [ ] Basic image/text preprocessing utilities

### 8. **Model Serialization**
- [ ] Save/load model weights (`.pth` format compatibility)
- [ ] Checkpoint saving during training

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
- [x] Core autograd infrastructure (though backward pass needs work)
- [x] Basic activation functions (ReLU, GELU, SiLU, softmax)
- [x] Basic loss functions (MSE, NLL)
- [x] Comprehensive test suite (31 tests passing)
- [x] Usage examples and documentation

### 🚧 **Known Issues**
- Autograd functions exist but `backward()` methods are empty
- Many operations return empty tensors (like scalar multiplication)
- No optimizers at all
- Limited broadcasting support
- Convolutional layers completely missing
- GPU support absent

## 🎯 **Next Steps**

1. **Implement optimizers** (SGD, Adam) - enables actual training
2. **Fix backward pass** - makes autograd functional
3. **Add broadcasting** - fixes many tensor operation bugs
4. **Add convolutional layers** - enables computer vision applications

---

*Last updated: August 11, 2025*
*Repository: https://github.com/jyizheng/RustedTorch*
