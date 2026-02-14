#!/bin/bash
# Configuration for dual DGX Spark disaggregated inference setup
#
# Spark 1 (spark-afe0): Prefill server
# Spark 2 (spark-ae11): Decode server
# Connected via 200Gbps RoCE interconnect on 192.168.200.0/24

# === Network Configuration ===
# High-speed interconnect IPs (200Gbps RoCE)
PREFILL_IP="192.168.200.13"    # Spark 1
DECODE_IP="192.168.200.12"     # Spark 2

# Management IPs (for SSH from external machines)
PREFILL_MGMT_IP="10.110.22.129"
DECODE_MGMT_IP="10.110.22.138"

# SSH user
SSH_USER="nvidia"

# === Server Configuration ===
PREFILL_PORT=8080
DECODE_PORT=8081

# Model paths (different locations on each machine)
PREFILL_MODEL="$HOME/models/Qwen3-4B-Q8_0.gguf"
DECODE_MODEL="$HOME/models/gguf_models/Qwen3-4B-Q8_0.gguf"

# llama-server binary (on each machine)
SERVER_BIN="$HOME/llama-disagg/build/bin/llama-server"

# Context size and slots - MUST match on both servers
CTX_SIZE="${CTX_SIZE:-8192}"
N_SLOTS="${N_SLOTS:-2}"

# KV cache directory (on each machine)
KV_CACHE_DIR="/tmp/llama_kv_cache"

# === SSH ControlMaster ===
# For fast KV cache transfer, set up persistent SSH on Spark 1:
#   ssh -fN spark2
# This uses the alias defined in ~/.ssh/config with ControlMaster auto
DECODE_SSH_ALIAS="spark2"

# === Derived URLs ===
PREFILL_URL="http://${PREFILL_IP}:${PREFILL_PORT}"
DECODE_URL="http://${DECODE_IP}:${DECODE_PORT}"
