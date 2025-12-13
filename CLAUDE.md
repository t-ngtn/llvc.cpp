# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llvc.cpp is a C++ reimplementation of LLVC (Low-Latency Low-Resource Voice Conversion), a real-time voice conversion system designed for CPU inference with ~15ms algorithmic latency. The goal is to create a hardware-friendly implementation suitable for embedded systems and native applications.

The reference Python implementation is available at `/Users/tomoya/labo/LLVC` and documented in its CLAUDE.md. The paper is at https://koe.ai/papers/llvc.pdf

## Build Commands

```bash
make              # Build release version (default)
make debug        # Build debug version with -g -O0
make clean        # Remove build directory
make rebuild      # Clean and rebuild
make infer        # Run inference on test_wavs/ -> converted_out/
make infer-streaming  # Run streaming inference on test_wavs/
```

### Running Inference

```bash
# Non-streaming (batch) inference
./build/llvc_infer -w models/llvc_weights.bin -i input.wav -o output.wav

# Streaming inference
./build/llvc_infer -w models/llvc_weights.bin -i input.wav -o output.wav -s

# Process entire directory (default behavior when run without args)
./build/llvc_infer   # processes test_wavs/ -> converted_out/
```

### Model Setup

Download and convert the pretrained model:
```bash
pip install torch huggingface_hub numpy
python scripts/download_model.py
```

## Code Architecture

### Directory Structure
- `include/llvc/` - Header files
- `src/` - Implementation files
- `scripts/` - Python utilities for model conversion

### Key Components (in dependency order)

1. **tensor.hpp/cpp** - Custom tensor class with 1D/2D/3D support, no external dependencies
2. **activations.hpp/cpp** - Activation functions (GELU, ReLU, Sigmoid, Tanh)
3. **layers.hpp/cpp** - Core layers: Conv1d, ConvTranspose1d, Linear, LayerNorm
4. **attention.hpp/cpp** - Multi-head attention with causal masking
5. **blocks.hpp/cpp** - Composite blocks: DilatedCausalConvEncoder, CausalTransformerDecoder, CachedConvNet
6. **model.hpp/cpp** - Main `Net` class with `forward()` (batch) and `forward_stream()` (streaming)
7. **weights.hpp/cpp** - Binary weight file loading
8. **audio.hpp/cpp** - WAV file I/O and resampling

### Main Model Class (`llvc::Net`)

```cpp
// Initialize
llvc::Net::Config config;
llvc::Net model(config);
model.load_weights(weights);

// Batch inference: input [1, 1, T] -> output [1, 1, T]
Tensor output = model.forward(input);

// Streaming inference
auto bufs = model.init_buffers(1);
auto [out, new_bufs] = model.forward_stream(chunk, bufs);
```

### Buffer State for Streaming
Four buffer types maintained in `Net::Buffers`:
- `enc_buf`: Encoder context (dilated conv history)
- `dec_buf`: Decoder context (transformer state)
- `out_buf`: Output overlap buffer
- `convnet_ctx`: Optional preprocessing context

## Reference Architecture

The model consists of:
1. **CachedConvNet** (optional): Causal convolutions with context buffering
2. **Input Conv**: Projects audio 1 -> 512 channels
3. **DilatedCausalConvEncoder**: 8 layers with dilation 2^i
4. **CausalTransformerDecoder**: Cross-attention decoder
5. **Output Conv**: Projects 512 -> 1 channel

### Key Parameters
- Sample rate: 16kHz (fixed)
- L (hop length): 16 samples
- dec_chunk_size: 13 frames
- Encoder: 8 layers, 512 dimensions
- Decoder: 1 layer, 256 dimensions
- Chunk length: `dec_chunk_size * L * chunk_factor` = 208 samples @ chunk_factor=1 (~13ms)

## Implementation Notes

- All operations are causal (no future samples needed)
- Buffer sizes are deterministic and can be statically allocated
- Transformer attention is limited to chunk_size + ctx_len tokens
- WAV I/O implemented without external libraries
