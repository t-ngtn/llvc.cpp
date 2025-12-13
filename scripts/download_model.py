#!/usr/bin/env python3
"""
Download LLVC model from HuggingFace Hub and convert to C++ format.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download LLVC model and convert to C++ format"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="models",
        help="Output directory for converted weights"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if checkpoint already exists"
    )

    args = parser.parse_args()

    # Try to import huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    # Download model
    cache_dir = Path("llvc_cache")
    checkpoint_path = cache_dir / "models" / "checkpoints" / "llvc" / "G_500000.pth"

    if not args.skip_download or not checkpoint_path.exists():
        print("Downloading LLVC model from HuggingFace Hub...")
        snapshot_download(
            repo_id="KoeAI/llvc",
            local_dir=str(cache_dir),
            allow_patterns=["models/checkpoints/llvc/*"]
        )
        print(f"Downloaded to {cache_dir}")
    else:
        print(f"Using cached checkpoint at {checkpoint_path}")

    # Convert weights
    print("Converting weights to C++ format...")

    # Import torch only when needed
    try:
        import torch
        import struct
    except ImportError:
        print("Error: PyTorch not installed.")
        print("Install with: pip install torch")
        sys.exit(1)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "llvc_weights.bin"

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

    # Cleanup cache (optional)
    print("\nTo save disk space, you can delete the cache directory:")
    print(f"  rm -rf {cache_dir}")


if __name__ == "__main__":
    main()
