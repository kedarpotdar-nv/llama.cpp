#!/bin/bash
# Deploy and build llama-disagg on both DGX Spark machines
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

REPO_URL="https://github.com/kedarpotdar-nv/llama.cpp"
BRANCH="${1:-spark}"

echo "========================================"
echo "Deploying llama-disagg to DGX Sparks"
echo "========================================"
echo "Repo:   $REPO_URL"
echo "Branch: $BRANCH"
echo ""

deploy_one() {
    local host_ip="$1"
    local host_name="$2"

    echo "--- Deploying to $host_name ($host_ip) ---"

    ssh "${SSH_USER}@${host_ip}" bash -s "$REPO_URL" "$BRANCH" <<'REMOTE_SCRIPT'
        REPO_URL="$1"
        BRANCH="$2"
        set -e

        cd ~
        if [ -d llama-disagg ]; then
            echo "Updating existing repo..."
            cd llama-disagg
            git fetch origin
            git checkout "$BRANCH"
            git pull origin "$BRANCH"
        else
            echo "Cloning repo..."
            git clone "$REPO_URL" llama-disagg
            cd llama-disagg
            git checkout "$BRANCH"
        fi

        echo "Building llama-server with CUDA..."
        export PATH=/usr/local/cuda/bin:$PATH
        rm -rf build
        cmake -B build \
            -DGGML_CUDA=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
            2>&1 | tail -5

        cmake --build build --config Release -t llama-server -j$(nproc) \
            2>&1 | tail -10

        if [ -x build/bin/llama-server ]; then
            echo "BUILD SUCCESS: $(ls -lh build/bin/llama-server | awk '{print $5}')"
        else
            echo "BUILD FAILED"
            exit 1
        fi

        # Create KV cache directory
        mkdir -p /tmp/llama_kv_cache

        echo "DONE"
REMOTE_SCRIPT

    echo ""
}

# Deploy to both machines
deploy_one "$PREFILL_MGMT_IP" "Spark 1 (Prefill)"
deploy_one "$DECODE_MGMT_IP"  "Spark 2 (Decode)"

echo "========================================"
echo "Deployment complete on both machines!"
echo "========================================"
echo ""
echo "Next: ./disagg/spark/start_servers.sh"
