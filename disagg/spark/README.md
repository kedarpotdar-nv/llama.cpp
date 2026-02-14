# Disaggregated Inference on Dual DGX Spark

Cross-machine disaggregated prefill/decode across two DGX Spark systems connected via 200Gbps RoCE interconnect.

## Architecture

```
                    200Gbps RoCE (192.168.200.0/24)
                    ================================

Spark 1 (spark-afe0)              Spark 2 (spark-ae11)
192.168.200.13                    192.168.200.12
┌──────────────────┐              ┌──────────────────┐
│  PREFILL SERVER  │              │  DECODE SERVER   │
│  port 8080       │    SCP/SSH   │  port 8081       │
│                  │ ──────────→  │                  │
│  1. Prefill      │  KV cache    │  4. Restore      │
│  2. Save KV      │  transfer    │  5. Decode       │
│  3. (Transfer)   │  (~8 Gbps)   │                  │
│                  │              │                  │
│  GB10 GPU        │              │  GB10 GPU        │
│  128GB unified   │              │  128GB unified   │
└──────────────────┘              └──────────────────┘
```

## Quick Start

```bash
# 1. Deploy and build on both machines (from any machine with SSH access)
./disagg/spark/deploy.sh spark

# 2. Start servers
./disagg/spark/start_servers.sh

# 3. Run smoke test
./disagg/spark/smoke_test.sh

# 4. Run benchmark
ssh nvidia@10.110.22.129   # Run from Spark 1 for best performance
cd ~/llama-disagg
python3 disagg/spark/benchmark.py --prompt-tokens 1000 --output-tokens 50

# 5. Stop servers
./disagg/spark/stop_servers.sh
```

## SSH ControlMaster (Important for Performance)

The KV cache transfer uses SCP. Without a persistent SSH connection, each transfer incurs ~500ms SSH handshake overhead. With ControlMaster, this drops to ~17ms.

On Spark 1, the SSH config is set up at `~/.ssh/config` with alias `spark2`. Initialize the persistent connection:

```bash
ssh -fN spark2
```

This keeps a persistent SSH connection to Spark 2 (192.168.200.12) over the high-speed link.

## Benchmark Results (Qwen3-4B-Q8_0)

### Single Request (2000 tokens prompt, 100 output tokens)

| Metric | Baseline | Disagg |
|--------|----------|--------|
| Total latency | ~2200ms | ~2900ms |
| Prefill | ~400ms | ~400ms (on Spark 1) |
| Save | - | ~150ms |
| Transfer (262MB) | - | ~220ms (~8 Gbps) |
| Restore | - | ~65ms |
| Decode | ~1960ms | ~1960ms (on Spark 2) |
| Overhead | - | ~435ms |

### Pipeline Analysis

With proper pipelining (prefill N+1 while decoding N):

| Metric | Sequential | Pipelined |
|--------|-----------|-----------|
| Throughput | 0.37 req/s | 0.49 req/s |
| Improvement | - | **+33%** |

The pipeline benefit increases with:
- Longer decode times (more overlap window)
- Multiple slots per server (`-np 4`)
- Higher concurrency

### Concurrency Sweep (ISL=4000, OSL=128, -np 16, -c 131072)

| Concurrency | Baseline Tok/s | Disagg Tok/s | Speedup |
|:-----------:|:--------------:|:------------:|:-------:|
| 1 | 37.2 | 27.1 | 0.73x |
| 2 | 52.3 | 44.3 | 0.85x |
| 4 | 76.4 | 61.2 | 0.80x |
| **8** | **78.1** | **90.6** | **1.16x** |
| **12** | **92.6** | **109.1** | **1.18x** |
| 16 | 117.2 | 121.7 | 1.04x |

**Crossover at concurrency 8**: disagg overtakes baseline when the single GPU
becomes saturated handling both prefill and decode. Peak advantage is 1.18x
at concurrency 12.

Trade-off: disagg has higher TTFT (time to first token) due to KV cache
transfer overhead (~1.3s avg at high concurrency).

## File Structure

| File | Purpose |
|------|---------|
| `config.sh` | IP addresses, ports, model paths |
| `deploy.sh` | Clone repo and build on both machines |
| `start_servers.sh` | Launch prefill/decode servers remotely |
| `stop_servers.sh` | Stop all servers |
| `smoke_test.sh` | End-to-end validation |
| `orchestrator.py` | HTTP orchestrator with SCP transfer |
| `benchmark.py` | Baseline vs disagg latency comparison |
| `pipeline_benchmark.py` | Pipeline throughput analysis |
| `concurrency_sweep.py` | Concurrency sweep: find crossover point |

## Configuration

Edit `config.sh` to change:
- IP addresses (if different network setup)
- Model paths
- Context size (`CTX_SIZE`, default 8192)
- Slots per server (`N_SLOTS`, default 2)

## KV Cache Transfer Details

- Transfer method: SCP over SSH ControlMaster
- Effective bandwidth: ~7-9 Gbps (of 200 Gbps link)
- Limited by SSH encryption overhead, not link capacity
- KV cache size: ~147KB per token (Qwen3-4B-Q8_0)
- 1000 tokens → ~131MB, 2000 tokens → ~262MB
