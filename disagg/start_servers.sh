#!/bin/bash
# Start prefill and decode servers for disaggregated inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <model_path> [context_size]"
    echo "Example: $0 /path/to/model.gguf 8192"
    exit 1
fi

MODEL_PATH="$1"
CTX_SIZE="${2:-8192}"

# Create shared KV cache directory (using tmpfs for speed if available)
KV_CACHE_DIR="/tmp/llama_kv_cache"
mkdir -p "$KV_CACHE_DIR"

# Check if server binary exists
if [ ! -f "$SERVER_BIN" ]; then
    echo "Error: llama-server not found at $SERVER_BIN"
    echo "Run ./disagg/build.sh first"
    exit 1
fi

# Kill any existing servers
pkill -f "llama-server.*8080" 2>/dev/null || true
pkill -f "llama-server.*8081" 2>/dev/null || true
sleep 1

echo "========================================"
echo "Starting Disaggregated Prefill Servers"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Context size: $CTX_SIZE"
echo "KV cache dir: $KV_CACHE_DIR"
echo ""

# Start Prefill Server (GPU 0, port 8080)
echo "Starting Prefill Server on port 8080 (GPU 0)..."
$SERVER_BIN \
    -m "$MODEL_PATH" \
    --port 8080 \
    --host 0.0.0.0 \
    -c "$CTX_SIZE" \
    --slot-save-path "$KV_CACHE_DIR/" \
    -np 1 \
    --metrics \
    -dev cuda0 \
    2>&1 | sed 's/^/[PREFILL] /' &
PREFILL_PID=$!

# Start Decode Server (GPU 1, port 8081)
echo "Starting Decode Server on port 8081 (GPU 1)..."
$SERVER_BIN \
    -m "$MODEL_PATH" \
    --port 8081 \
    --host 0.0.0.0 \
    -c "$CTX_SIZE" \
    --slot-save-path "$KV_CACHE_DIR/" \
    -np 4 \
    --metrics \
    -dev cuda1 \
    2>&1 | sed 's/^/[DECODE]  /' &
DECODE_PID=$!

echo ""
echo "Waiting for servers to start..."
sleep 5

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
