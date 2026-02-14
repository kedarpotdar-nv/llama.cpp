#!/bin/bash
# Start prefill server on Spark 1 and decode server on Spark 2
# Servers listen on 0.0.0.0 so they're reachable over the 200Gbps interconnect
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Allow overriding model paths
PREFILL_MODEL="${PREFILL_MODEL_OVERRIDE:-$PREFILL_MODEL}"
DECODE_MODEL="${DECODE_MODEL_OVERRIDE:-$DECODE_MODEL}"

echo "========================================"
echo "Starting Distributed Disagg Servers"
echo "========================================"
echo "Prefill: ${SSH_USER}@${PREFILL_MGMT_IP} (highspeed: ${PREFILL_IP}:${PREFILL_PORT})"
echo "Decode:  ${SSH_USER}@${DECODE_MGMT_IP} (highspeed: ${DECODE_IP}:${DECODE_PORT})"
echo "Model:   Qwen3-4B-Q8_0.gguf"
echo "Context: ${CTX_SIZE}, Slots: ${N_SLOTS}"
echo ""

# Stop any existing servers
echo "Stopping any existing servers..."
ssh "${SSH_USER}@${PREFILL_MGMT_IP}" "pkill -f 'llama-server.*${PREFILL_PORT}' 2>/dev/null; true"
ssh "${SSH_USER}@${DECODE_MGMT_IP}" "pkill -f 'llama-server.*${DECODE_PORT}' 2>/dev/null; true"
sleep 2

# Start prefill server on Spark 1
echo "Starting Prefill Server on Spark 1 (${PREFILL_IP}:${PREFILL_PORT})..."
ssh "${SSH_USER}@${PREFILL_MGMT_IP}" bash -c "'
    mkdir -p ${KV_CACHE_DIR}
    nohup ${SERVER_BIN} \
        -m ${PREFILL_MODEL} \
        --port ${PREFILL_PORT} \
        --host 0.0.0.0 \
        -c ${CTX_SIZE} \
        -ngl 99 \
        --slot-save-path ${KV_CACHE_DIR}/ \
        -np ${N_SLOTS} \
        -cb \
        -fa on \
        --metrics \
        > /tmp/prefill_server.log 2>&1 &
    echo \$!
'" &
PREFILL_SSH_PID=$!

# Start decode server on Spark 2
echo "Starting Decode Server on Spark 2 (${DECODE_IP}:${DECODE_PORT})..."
ssh "${SSH_USER}@${DECODE_MGMT_IP}" bash -c "'
    mkdir -p ${KV_CACHE_DIR}
    nohup ${SERVER_BIN} \
        -m ${DECODE_MODEL} \
        --port ${DECODE_PORT} \
        --host 0.0.0.0 \
        -c ${CTX_SIZE} \
        -ngl 99 \
        --slot-save-path ${KV_CACHE_DIR}/ \
        -np ${N_SLOTS} \
        -cb \
        -fa on \
        --metrics \
        > /tmp/decode_server.log 2>&1 &
    echo \$!
'" &
DECODE_SSH_PID=$!

wait $PREFILL_SSH_PID $DECODE_SSH_PID

echo ""
echo "Waiting for servers to start (loading model)..."

# Health check with retries
check_health() {
    local url="$1"
    local name="$2"
    local max_retries=30
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if curl -s --connect-timeout 2 "${url}/health" 2>/dev/null | grep -q "ok"; then
            echo "  OK: $name server healthy at $url"
            return 0
        fi
        retry=$((retry + 1))
        sleep 2
    done
    echo "  FAIL: $name server not responding at $url after ${max_retries} retries"
    return 1
}

check_health "$PREFILL_URL" "Prefill" || exit 1
check_health "$DECODE_URL" "Decode" || exit 1

# Verify matching configurations
echo ""
echo "Verifying configurations match..."
PREFILL_CTX=$(curl -s "${PREFILL_URL}/props" | python3 -c "import sys,json; print(json.load(sys.stdin).get('default_generation_settings',{}).get('n_ctx','?'))" 2>/dev/null)
DECODE_CTX=$(curl -s "${DECODE_URL}/props" | python3 -c "import sys,json; print(json.load(sys.stdin).get('default_generation_settings',{}).get('n_ctx','?'))" 2>/dev/null)

echo "  Prefill n_ctx: ${PREFILL_CTX}"
echo "  Decode  n_ctx: ${DECODE_CTX}"

if [ "$PREFILL_CTX" != "$DECODE_CTX" ]; then
    echo "  WARNING: Context sizes don't match! KV cache transfer may fail."
fi

echo ""
echo "========================================"
echo "Both servers running!"
echo "========================================"
echo "Prefill: ${PREFILL_URL}"
echo "Decode:  ${DECODE_URL}"
echo "KV Cache Dir: ${KV_CACHE_DIR} (on each machine)"
echo ""
echo "Logs:"
echo "  ssh ${SSH_USER}@${PREFILL_MGMT_IP} tail -f /tmp/prefill_server.log"
echo "  ssh ${SSH_USER}@${DECODE_MGMT_IP} tail -f /tmp/decode_server.log"
echo ""
echo "Next: python3 disagg/spark/benchmark.py"
