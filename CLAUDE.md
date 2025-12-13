# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llvc.cpp is a C++ reimplementation of LLVC (Low-Latency Low-Resource Voice Conversion), a real-time voice conversion system designed for CPU inference with ~15ms algorithmic latency. The goal is to create a hardware-friendly implementation suitable for embedded systems and native applications.

The reference Python implementation is available at `/Users/tomoya/labo/LLVC` and documented in its CLAUDE.md. The paper is at https://koe.ai/papers/llvc.pdf

## Reference Architecture (from Python LLVC)

The model architecture consists of:

1. **CachedConvNet** (optional preprocessing): Causal convolutions with context buffering
2. **DilatedCausalConvEncoder**: Dilated causal convolutions with exponentially increasing dilation (2^i for i layers)
3. **CausalTransformerDecoder**: Processes features in chunks with positional encoding
4. **MaskNet**: Generates time-domain masks combining encoder output and label embeddings

### Key Parameters (from default config)
- Sample rate: 16kHz (fixed)
- L (hop length): 16 samples
- dec_chunk_size: 13 frames
- Encoder: 8 layers, 512 dimensions
- Decoder: 1 layer, 256 dimensions
- Chunk length: `dec_chunk_size * L * chunk_factor` = 208 samples @ chunk_factor=1 (~13ms)

### Buffer State for Streaming
Four buffer types must be maintained:
- `enc_buf`: Encoder context (dilated conv history)
- `dec_buf`: Decoder context (transformer state)
- `out_buf`: Output overlap buffer
- `convnet_pre_ctx`: Optional preprocessing context

### Streaming Inference Flow
1. Split audio into chunks of `dec_chunk_size * L * chunk_factor` samples
2. Prepend each chunk with 2*L lookahead samples from previous chunk
3. Initialize buffers with `model.init_buffers()`
4. Process chunks sequentially, passing buffer state between calls
5. Concatenate outputs

## Implementation Considerations for C++

- All operations are causal (no future samples needed)
- Use SIMD for convolutions (SSE/AVX on x86, NEON on ARM)
- Consider fixed-point arithmetic for embedded targets
- Buffer sizes are deterministic and can be statically allocated
- Transformer attention is limited to chunk_size + ctx_len tokens
