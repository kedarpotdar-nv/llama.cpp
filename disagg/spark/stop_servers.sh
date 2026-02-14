#!/bin/bash
# Stop all llama-server processes on both DGX Sparks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "Stopping servers..."
ssh "${SSH_USER}@${PREFILL_MGMT_IP}" "pkill -f llama-server 2>/dev/null && echo 'Stopped on Spark 1' || echo 'Nothing running on Spark 1'"
ssh "${SSH_USER}@${DECODE_MGMT_IP}" "pkill -f llama-server 2>/dev/null && echo 'Stopped on Spark 2' || echo 'Nothing running on Spark 2'"
echo "Done."
