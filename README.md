# llvc.cpp

C++ reimplementation of LLVC (Low-Latency Low-Resource Voice Conversion) for CPU inference.

This repository provides an independent C++ reimplementation of LLVC
for research and hardware acceleration purposes. The implementation was
written from scratch in C++, while referring to the original PyTorch
implementation released by Koe AI (https://github.com/KoeAI/LLVC).

## Features

- Pure C++ implementation with minimal dependencies
- Standard C++ project structure (header/source separation)
- Supports both streaming and non-streaming inference
- FPGA-friendly implementation (simple loops, explicit memory access)
- WAV file I/O without external libraries

## Building

### Using Make (Recommended)

```bash
make            # Build release version
make debug      # Build debug version
make clean      # Clean build directory
make help       # Show all available commands
```

### Using CMake directly

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### 1. Download and Convert Model

First, download the pretrained models and convert to C++ format:

```bash
python scripts/download_model.py
```

This downloads both model variants (llvc and llvc_nc) from HuggingFace Hub and converts them to C++ format.
Requires: `pip install torch huggingface_hub numpy`

### 2. Run Inference

```bash
# Process test_wavs/ directory with default settings (llvc model)
# Output goes to converted_out/
./build/llvc_infer

# Single file inference
./build/llvc_infer -i input.wav -o output.wav

# Use llvc_nc model (no CachedConvNet)
./build/llvc_infer -m llvc_nc -i input.wav -o output.wav

# Streaming inference
./build/llvc_infer -i input.wav -o output.wav -s

# Streaming with larger chunks (higher latency, better performance)
./build/llvc_infer -i input.wav -o output.wav -s -n 2
```

### Command Line Options

```
Options:
  -m, --model <type>     Model type: llvc (default), llvc_nc
  -w, --weights <path>   Path to weights file (auto-selected by model type if not specified)
  -i, --input <path>     Path to input WAV file or directory
  -o, --output <path>    Path to output WAV file or directory
  -s, --streaming        Use streaming inference
  -n, --chunk-factor <n> Chunk factor for streaming (default: 1)
  -h, --help             Show this help
```

### Model Variants

This implementation supports two model variants:

- **llvc** (default): Uses CachedConvNet preprocessing for enhanced audio quality
- **llvc_nc**: No CachedConvNet - lighter model with slightly different characteristics

Both variants share the same core architecture (encoder, decoder, transformer) but differ in preprocessing.


## Model Architecture

The model consists of:

1. **CachedConvNet**: Optional preprocessing with causal convolutions
2. **Input Conv**: Projects audio to latent space (1 -> 512 channels)
3. **DilatedCausalConvEncoder**: 8 layers of dilated causal convolutions
4. **CausalTransformerDecoder**: Cross-attention based decoder
5. **Output Conv**: Projects back to audio (512 -> 1 channel)

## License

MIT License

## Test Audio Samples

The test audio samples under `test_wavs/` are taken from the original
LLVC repository by Koe AI and are distributed under the MIT License,
same as the original project.