#!/usr/bin/env python3
"""
Download LLVC models from HuggingFace Hub and convert to C++ format.
"""

import argparse
import sys
from pathlib import Path


def convert_checkpoint(checkpoint_path, output_path):
    """Convert a PyTorch checkpoint to C++ binary format."""
    import torch
    import struct

    print(f"Converting {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # Write binary format
    with open(output_path, "wb") as f:
        # Header
        f.write(b"LLVC")  # magic
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", len(state_dict)))  # num_tensors

        for name, tensor in state_dict.items():
            data = tensor.detach().float().contiguous().numpy()

            # Name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # Shape
            f.write(struct.pack("<I", len(data.shape)))
            for dim in data.shape:
                f.write(struct.pack("<I", dim))

            # Data
            f.write(data.tobytes())

    print(f"Saved converted weights to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Download LLVC models and convert to C++ format"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="models",
        help="Output directory for converted weights"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if checkpoints already exist"
    )

    args = parser.parse_args()

    # Try to import huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    # Both models to download
    models = ["llvc", "llvc_nc"]

    # Download models
    cache_dir = Path("llvc_cache")

    # Check if we need to download
    need_download = False
    for model_name in models:
        checkpoint_path = cache_dir / "models" / "checkpoints" / model_name / "G_500000.pth"
        if not checkpoint_path.exists():
            need_download = True
            break

    if not args.skip_download or need_download:
        print("Downloading LLVC models from HuggingFace Hub...")
        snapshot_download(
            repo_id="KoeAI/llvc",
            local_dir=str(cache_dir),
            allow_patterns=["models/checkpoints/llvc/*", "models/checkpoints/llvc_nc/*"]
        )
        print(f"Downloaded to {cache_dir}")
    else:
        print("Using cached checkpoints")

    # Check torch is available before conversion
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed.")
        print("Install with: pip install torch")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each model
    for model_name in models:
        checkpoint_path = cache_dir / "models" / "checkpoints" / model_name / "G_500000.pth"

        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)

        # Output filename: llvc_weights.bin or llvc_nc_weights.bin
        output_filename = f"{model_name}_weights.bin"
        output_path = output_dir / output_filename

        convert_checkpoint(checkpoint_path, output_path)

    print("\nConversion complete!")
    print(f"  - models/llvc_weights.bin (with CachedConvNet)")
    print(f"  - models/llvc_nc_weights.bin (no CachedConvNet)")

    # Cleanup cache (optional)
    print("\nTo save disk space, you can delete the cache directory:")
    print(f"  rm -rf {cache_dir}")


if __name__ == "__main__":
    main()
