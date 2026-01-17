#!/bin/bash
# Build llama-server with CUDA support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building llama.cpp from: $LLAMA_DIR"

cd "$LLAMA_DIR"

# Clean previous build
rm -rf build

# Configure with CUDA
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release

# Build llama-server
cmake --build build --config Release -t llama-server -j$(nproc)

echo ""
echo "Build complete!"
echo "Binary location: $LLAMA_DIR/build/bin/llama-server"
