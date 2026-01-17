# Disaggregated Prefill Experiment

This directory contains scripts for experimenting with disaggregated prefill/decode
using two separate llama-server instances.

## Architecture

```
                    ┌─────────────────────┐
                    │   Orchestrator      │
                    │   (Python script)   │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┴───────────────────┐
           ▼                                       ▼
┌─────────────────────┐              ┌─────────────────────┐
│   Prefill Server    │              │   Decode Server     │
│   localhost:8080    │              │   localhost:8081    │
│      (GPU 0)        │    Shared    │      (GPU 1)        │
│                     │◄──Storage───►│                     │
│  --slot-save-path   │   /tmp/kv/   │  --slot-save-path   │
└─────────────────────┘              └─────────────────────┘
```

## Quick Start

1. Build llama.cpp:
   ```bash
   ./disagg/build.sh
   ```

2. Start both servers:
   ```bash
   ./disagg/start_servers.sh /path/to/model.gguf
   ```

3. Run the benchmark:
   ```bash
   python3 disagg/benchmark.py
   ```

## Files

- `build.sh` - Build script for llama-server
- `start_servers.sh` - Launch prefill and decode servers
- `benchmark.py` - Benchmark comparing disagg vs baseline
- `orchestrator.py` - Simple request orchestrator
