#!/bin/bash
# Start prefill and decode servers for disaggregated inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(dirname "$SCRIPT_DIR")"

# Try to find llama-server in common locations
SERVER_BIN=""
for path in \
    "$LLAMA_DIR/build/bin/llama-server" \
    "$LLAMA_DIR/build/llama-server" \
    "$(which llama-server 2>/dev/null)" \
    "/usr/local/bin/llama-server" \
    "$HOME/.local/bin/llama-server"
do
    if [ -x "$path" ]; then
        SERVER_BIN="$path"
        break
    fi
done

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <model_path> [context_size] [mode]"
    echo ""
    echo "Arguments:"
    echo "  model_path    Path to GGUF model file"
    echo "  context_size  Context size (default: 8192)"
    echo "  mode          'dual' for dual GPU, 'single' for single GPU (default: auto-detect)"
    echo ""
    echo "Environment variables:"
    echo "  N_SLOTS       Number of slots per server (default: 2 for dual, 1 for single)"
    echo "                IMPORTANT: Both servers MUST have same ctx and slots for KV cache compatibility"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/model.gguf 2048 dual           # Dual GPU, 2 slots each"
    echo "  N_SLOTS=4 $0 /path/to/model.gguf 4096 dual # Dual GPU, 4 slots each"
    echo "  $0 /path/to/model.gguf 2048 single         # Single GPU"
    exit 1
fi

MODEL_PATH="$1"
CTX_SIZE="${2:-8192}"
MODE="${3:-auto}"

# Create shared KV cache directory
KV_CACHE_DIR="/tmp/llama_kv_cache"
mkdir -p "$KV_CACHE_DIR"

# Check if server binary exists
if [ -z "$SERVER_BIN" ]; then
    echo "Error: llama-server not found!"
    echo "Searched in:"
    echo "  - $LLAMA_DIR/build/bin/llama-server"
    echo "  - $LLAMA_DIR/build/llama-server"
    echo "  - PATH"
    echo ""
    echo "Please either:"
    echo "  1. Run ./disagg/build.sh to build"
    echo "  2. Set SERVER_BIN environment variable"
    echo "  3. Add llama-server to your PATH"
    exit 1
fi

echo "Using llama-server: $SERVER_BIN"

# Auto-detect GPU mode
if [ "$MODE" = "auto" ]; then
    # Check for NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
        if [ "$GPU_COUNT" -ge 2 ]; then
            MODE="dual"
            echo "Auto-detected: $GPU_COUNT NVIDIA GPUs -> dual mode"
        else
            MODE="single"
            echo "Auto-detected: $GPU_COUNT NVIDIA GPU -> single mode"
        fi
    # Check for Metal (macOS)
    elif system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
        MODE="single"
        echo "Auto-detected: Metal GPU -> single mode"
    else
        MODE="single"
        echo "Auto-detected: No multi-GPU found -> single mode"
    fi
fi

# Kill any existing servers
pkill -f "llama-server.*8080" 2>/dev/null || true
pkill -f "llama-server.*8081" 2>/dev/null || true
sleep 1

echo ""
echo "========================================"
echo "Starting Disaggregated Prefill Servers"
echo "========================================"
echo "Mode: $MODE"
echo "Model: $MODEL_PATH"
echo "Context size: $CTX_SIZE"
echo "KV cache dir: $KV_CACHE_DIR"
echo ""

if [ "$MODE" = "dual" ]; then
    # DUAL GPU MODE: Each server on separate GPU
    # IMPORTANT: Both servers MUST have same -np for KV cache compatibility
    
    N_SLOTS="${N_SLOTS:-2}"  # Default 2, can override with N_SLOTS=4 ./start_servers.sh ...
    
    echo "Starting Prefill Server on port 8080 (GPU 0, $N_SLOTS slots, ctx=$CTX_SIZE)..."
    CUDA_VISIBLE_DEVICES=0 $SERVER_BIN \
        -m "$MODEL_PATH" \
        --port 8080 \
        --host 0.0.0.0 \
        -c "$CTX_SIZE" \
        -ngl 99 \
        --slot-save-path "$KV_CACHE_DIR/" \
        -np $N_SLOTS \
        -cb \
        -fa \
        --metrics \
        2>&1 | sed 's/^/[PREFILL] /' &
    PREFILL_PID=$!

    echo "Starting Decode Server on port 8081 (GPU 1, $N_SLOTS slots, ctx=$CTX_SIZE)..."
    CUDA_VISIBLE_DEVICES=1 $SERVER_BIN \
        -m "$MODEL_PATH" \
        --port 8081 \
        --host 0.0.0.0 \
        -c "$CTX_SIZE" \
        -ngl 99 \
        --slot-save-path "$KV_CACHE_DIR/" \
        -np $N_SLOTS \
        -cb \
        -fa \
        --metrics \
        2>&1 | sed 's/^/[DECODE]  /' &
    DECODE_PID=$!
    
else
    # SINGLE GPU MODE: Both servers on same GPU (different ports)
    # This still tests the architecture, just without GPU parallelism
    # IMPORTANT: Both servers MUST have same ctx and slots for KV cache compatibility
    
    N_SLOTS="${N_SLOTS:-1}"  # Default 1 for single GPU (lower VRAM)
    
    echo "Starting Prefill Server on port 8080 ($N_SLOTS slots, ctx=$CTX_SIZE)..."
    $SERVER_BIN \
        -m "$MODEL_PATH" \
        --port 8080 \
        --host 0.0.0.0 \
        -c "$CTX_SIZE" \
        -ngl 99 \
        --slot-save-path "$KV_CACHE_DIR/" \
        -np $N_SLOTS \
        -cb \
        -fa \
        --metrics \
        2>&1 | sed 's/^/[PREFILL] /' &
    PREFILL_PID=$!

    # Wait for first server to load model before starting second
    echo "Waiting for prefill server to load model..."
    sleep 15

    echo "Starting Decode Server on port 8081 ($N_SLOTS slots, ctx=$CTX_SIZE)..."
    $SERVER_BIN \
        -m "$MODEL_PATH" \
        --port 8081 \
        --host 0.0.0.0 \
        -c "$CTX_SIZE" \
        -ngl 99 \
        --slot-save-path "$KV_CACHE_DIR/" \
        -np $N_SLOTS \
        -cb \
        -fa \
        --metrics \
        2>&1 | sed 's/^/[DECODE]  /' &
    DECODE_PID=$!
fi

echo ""
echo "Waiting for servers to start..."
sleep 10

# Health check
check_health() {
    local port=$1
    local name=$2
    if curl -s "http://localhost:$port/health" | grep -q "ok"; then
        echo "✓ $name server healthy (port $port)"
        return 0
    else
        echo "✗ $name server not responding (port $port)"
        return 1
    fi
}

check_health 8080 "Prefill" || exit 1
check_health 8081 "Decode" || exit 1

# Verify both servers have matching configurations (critical for KV cache compatibility)
echo ""
echo "Verifying server configurations..."
PREFILL_CTX=$(curl -s http://localhost:8080/props | grep -o '"n_ctx":[0-9]*' | cut -d: -f2)
DECODE_CTX=$(curl -s http://localhost:8081/props | grep -o '"n_ctx":[0-9]*' | cut -d: -f2)
PREFILL_SLOTS=$(curl -s http://localhost:8080/props | grep -o '"total_slots":[0-9]*' | cut -d: -f2)
DECODE_SLOTS=$(curl -s http://localhost:8081/props | grep -o '"total_slots":[0-9]*' | cut -d: -f2)

echo "  Prefill: n_ctx=$PREFILL_CTX, slots=$PREFILL_SLOTS"
echo "  Decode:  n_ctx=$DECODE_CTX, slots=$DECODE_SLOTS"

if [ "$PREFILL_CTX" != "$DECODE_CTX" ] || [ "$PREFILL_SLOTS" != "$DECODE_SLOTS" ]; then
    echo ""
    echo "⚠️  WARNING: Server configurations don't match!"
    echo "   KV cache transfer will FAIL with mismatched ctx or slots."
    echo "   Please restart both servers with identical -c and -np settings."
    exit 1
fi
echo "✓ Configurations match - KV cache transfer should work"

echo ""
echo "========================================"
echo "Both servers running!"
echo "========================================"
echo "Prefill Server: http://localhost:8080 (PID: $PREFILL_PID)"
echo "Decode Server:  http://localhost:8081 (PID: $DECODE_PID)"
echo "KV Cache Dir:   $KV_CACHE_DIR"
echo ""
echo "To stop: pkill -f llama-server"
echo ""

# Keep script running
wait
