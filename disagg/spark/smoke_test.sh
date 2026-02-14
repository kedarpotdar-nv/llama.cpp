#!/bin/bash
# Quick smoke test for distributed disaggregated inference
# Tests the full pipeline: prefill -> save -> SCP transfer -> restore -> decode
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "========================================"
echo "Smoke Test: Distributed Disagg Inference"
echo "========================================"
echo ""

# 1. Health check
echo "[1/6] Health check..."
PREFILL_HEALTH=$(curl -s --connect-timeout 5 "${PREFILL_URL}/health" 2>/dev/null || echo "FAIL")
DECODE_HEALTH=$(curl -s --connect-timeout 5 "${DECODE_URL}/health" 2>/dev/null || echo "FAIL")

echo "  Prefill (${PREFILL_URL}): ${PREFILL_HEALTH}"
echo "  Decode  (${DECODE_URL}): ${DECODE_HEALTH}"

if echo "$PREFILL_HEALTH" | grep -q "ok" && echo "$DECODE_HEALTH" | grep -q "ok"; then
    echo "  OK"
else
    echo "  FAIL - servers not healthy. Run: ./disagg/spark/start_servers.sh"
    exit 1
fi

# 2. Prefill request
echo ""
echo "[2/6] Prefill (n_predict=0 on Spark 1)..."
PREFILL_RESULT=$(curl -s --connect-timeout 30 "${PREFILL_URL}/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The capital of France is", "n_predict": 0, "cache_prompt": true, "id_slot": 0}')

PROMPT_N=$(echo "$PREFILL_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('timings',{}).get('prompt_n',0))" 2>/dev/null)
echo "  Prefilled ${PROMPT_N} tokens"

# 3. Save KV cache
echo ""
echo "[3/6] Save KV cache on Spark 1..."
SAVE_RESULT=$(curl -s --connect-timeout 30 "${PREFILL_URL}/slots/0?action=save" \
    -H "Content-Type: application/json" \
    -d '{"filename": "smoke_test.bin"}')

N_SAVED=$(echo "$SAVE_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('n_saved',0))" 2>/dev/null)
echo "  Saved ${N_SAVED} tokens"

# 4. Transfer KV cache over 200Gbps interconnect
echo ""
echo "[4/6] Transfer KV cache (Spark 1 -> Spark 2 via ${PREFILL_IP} -> ${DECODE_IP})..."

# Determine if we're running on the prefill machine
HOSTNAME=$(hostname)
if ssh -o ConnectTimeout=2 ${SSH_USER}@${PREFILL_IP} "hostname" 2>/dev/null | grep -q "$HOSTNAME"; then
    # We're on the prefill machine, local source
    SRC="${KV_CACHE_DIR}/smoke_test.bin"
else
    # We're on a different machine, SSH source
    SRC="${SSH_USER}@${PREFILL_MGMT_IP}:${KV_CACHE_DIR}/smoke_test.bin"
fi
DST="${SSH_USER}@${DECODE_IP}:${KV_CACHE_DIR}/smoke_test.bin"

# Ensure target dir exists
ssh -o StrictHostKeyChecking=no "${SSH_USER}@${DECODE_IP}" "mkdir -p ${KV_CACHE_DIR}" 2>/dev/null || \
ssh -o StrictHostKeyChecking=no "${SSH_USER}@${DECODE_MGMT_IP}" "mkdir -p ${KV_CACHE_DIR}" 2>/dev/null

XFER_START=$(date +%s%N)
scp -o StrictHostKeyChecking=no -o Compression=no "$SRC" "$DST" 2>/dev/null || \
scp -o StrictHostKeyChecking=no -o Compression=no \
    "${SSH_USER}@${PREFILL_MGMT_IP}:${KV_CACHE_DIR}/smoke_test.bin" \
    "${SSH_USER}@${DECODE_MGMT_IP}:${KV_CACHE_DIR}/smoke_test.bin"
XFER_END=$(date +%s%N)

XFER_MS=$(( (XFER_END - XFER_START) / 1000000 ))
echo "  Transferred in ${XFER_MS}ms"

# 5. Restore KV cache on Spark 2
echo ""
echo "[5/6] Restore KV cache on Spark 2..."
RESTORE_RESULT=$(curl -s --connect-timeout 30 "${DECODE_URL}/slots/0?action=restore" \
    -H "Content-Type: application/json" \
    -d '{"filename": "smoke_test.bin"}')

echo "  Restore result: ${RESTORE_RESULT}"

# 6. Decode
echo ""
echo "[6/6] Decode on Spark 2..."
DECODE_RESULT=$(curl -s --connect-timeout 60 "${DECODE_URL}/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The capital of France is", "n_predict": 20, "cache_prompt": true, "id_slot": 0}')

CONTENT=$(echo "$DECODE_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('content','ERROR'))" 2>/dev/null)
PRED_N=$(echo "$DECODE_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('timings',{}).get('predicted_n',0))" 2>/dev/null)
PROMPT_MS=$(echo "$DECODE_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('timings',{}).get('prompt_ms',0))" 2>/dev/null)

echo "  Generated ${PRED_N} tokens"
echo "  Prompt eval: ${PROMPT_MS}ms (should be ~0 if cache hit)"
echo "  Content: ${CONTENT}"

# Cleanup
ssh -o StrictHostKeyChecking=no "${SSH_USER}@${PREFILL_MGMT_IP}" "rm -f ${KV_CACHE_DIR}/smoke_test.bin" 2>/dev/null || true
ssh -o StrictHostKeyChecking=no "${SSH_USER}@${DECODE_MGMT_IP}" "rm -f ${KV_CACHE_DIR}/smoke_test.bin" 2>/dev/null || true

echo ""
echo "========================================"
if [ "$PRED_N" -gt 0 ] 2>/dev/null; then
    echo "SMOKE TEST PASSED"
else
    echo "SMOKE TEST FAILED"
    exit 1
fi
echo "========================================"
