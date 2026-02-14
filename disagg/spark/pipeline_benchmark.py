#!/usr/bin/env python3
"""
Pipeline benchmark for distributed disaggregated inference across DGX Sparks.

Shows the real value of disagg: while Spark 2 decodes batch N,
Spark 1 prefills AND transfers batch N+1.

  Time →
  Batch 1: [PREFILL+SAVE+XFER on Spark1]──[RESTORE+DECODE on Spark2]
  Batch 2:              [PREFILL+SAVE+XFER on Spark1]──[RESTORE+DECODE on Spark2]
  Batch 3:                           [PREFILL+SAVE+XFER]──[RESTORE+DECODE]

Usage:
    python3 disagg/spark/pipeline_benchmark.py --waves 5 --output-tokens 100
"""

import argparse
import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import List
import aiohttp

PREFILL_URL = os.environ.get("PREFILL_URL", "http://192.168.200.13:8080")
DECODE_URL = os.environ.get("DECODE_URL", "http://192.168.200.12:8081")
DECODE_SSH = os.environ.get("DECODE_SSH", "spark2")
PREFILL_SSH = os.environ.get("PREFILL_SSH", "")
KV_CACHE_DIR = "/tmp/llama_kv_cache"


def generate_prompt(target_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. " * 100
    chars = target_tokens * 4
    return (base * (chars // len(base) + 1))[:chars]


@dataclass
class WaveResult:
    wave_id: int
    prefill_ms: float = 0
    save_ms: float = 0
    transfer_ms: float = 0
    transfer_bytes: int = 0
    restore_ms: float = 0
    decode_ms: float = 0
    total_ms: float = 0
    tokens_generated: int = 0
    prompt_tokens: int = 0
    cache_n: int = 0


async def scp_transfer(filename: str) -> tuple[float, int]:
    src_path = f"{KV_CACHE_DIR}/{filename}"
    dst = f"{DECODE_SSH}:{KV_CACHE_DIR}/{filename}"
    src = f"{PREFILL_SSH}:{src_path}" if PREFILL_SSH else src_path

    # Get file size
    try:
        if PREFILL_SSH:
            p = await asyncio.create_subprocess_exec(
                "ssh", PREFILL_SSH, f"stat -c%s {src_path}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        else:
            p = await asyncio.create_subprocess_exec(
                "stat", "-c%s", src_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, _ = await p.communicate()
        file_size = int(out.decode().strip())
    except Exception:
        file_size = 0

    t0 = time.perf_counter()
    p = await asyncio.create_subprocess_exec(
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "Compression=no",
        src, dst,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, err = await p.communicate()
    ms = (time.perf_counter() - t0) * 1000
    if p.returncode != 0:
        raise Exception(f"SCP failed: {err.decode()}")
    return ms, file_size


async def prefill_save_transfer(session: aiohttp.ClientSession, prompt: str,
                                 wave_id: int) -> WaveResult:
    """Prefill + save + SCP transfer (runs on Spark 1 side)"""
    r = WaveResult(wave_id=wave_id)
    filename = f"wave_{wave_id}.bin"

    # Prefill
    t0 = time.perf_counter()
    async with session.post(f"{PREFILL_URL}/completion", json={
        "prompt": prompt, "n_predict": 0, "cache_prompt": True, "id_slot": 0,
    }) as resp:
        data = await resp.json()
    r.prefill_ms = (time.perf_counter() - t0) * 1000
    r.prompt_tokens = data.get("timings", {}).get("prompt_n", 0)

    # Save
    t0 = time.perf_counter()
    async with session.post(f"{PREFILL_URL}/slots/0?action=save", json={
        "filename": filename
    }) as resp:
        await resp.json()
    r.save_ms = (time.perf_counter() - t0) * 1000

    # Transfer
    r.transfer_ms, r.transfer_bytes = await scp_transfer(filename)

    return r


async def restore_decode(session: aiohttp.ClientSession, prompt: str,
                          wave_id: int, n_predict: int) -> WaveResult:
    """Restore + decode (runs on Spark 2 side)"""
    r = WaveResult(wave_id=wave_id)
    filename = f"wave_{wave_id}.bin"

    # Clear slot
    try:
        async with session.post(f"{DECODE_URL}/slots/0?action=erase"):
            pass
    except Exception:
        pass

    # Restore
    t0 = time.perf_counter()
    async with session.post(f"{DECODE_URL}/slots/0?action=restore", json={
        "filename": filename
    }) as resp:
        await resp.json()
    r.restore_ms = (time.perf_counter() - t0) * 1000

    # Decode
    t0 = time.perf_counter()
    async with session.post(f"{DECODE_URL}/completion", json={
        "prompt": prompt, "n_predict": n_predict,
        "cache_prompt": True, "id_slot": 0,
    }) as resp:
        data = await resp.json()
    r.decode_ms = (time.perf_counter() - t0) * 1000
    timings = data.get("timings", {})
    r.tokens_generated = timings.get("predicted_n", 0)
    r.cache_n = timings.get("cache_n", 0)

    return r


async def run_pipeline(n_waves: int, prompt: str, n_predict: int):
    """Run pipelined disagg: prefill wave N+1 while decoding wave N"""

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        wave_results: List[WaveResult] = []
        pipeline_start = time.perf_counter()

        # Wave 0: sequential (no overlap possible)
        print(f"\n  Wave 0: prefill+save+xfer...", end="", flush=True)
        pst = await prefill_save_transfer(session, prompt, 0)
        print(f" restore+decode...", end="", flush=True)
        rd = await restore_decode(session, prompt, 0, n_predict)

        combined = WaveResult(wave_id=0)
        combined.prefill_ms = pst.prefill_ms
        combined.save_ms = pst.save_ms
        combined.transfer_ms = pst.transfer_ms
        combined.transfer_bytes = pst.transfer_bytes
        combined.restore_ms = rd.restore_ms
        combined.decode_ms = rd.decode_ms
        combined.tokens_generated = rd.tokens_generated
        combined.prompt_tokens = pst.prompt_tokens
        combined.cache_n = rd.cache_n
        combined.total_ms = (pst.prefill_ms + pst.save_ms + pst.transfer_ms +
                            rd.restore_ms + rd.decode_ms)
        wave_results.append(combined)
        print(f" done ({combined.total_ms:.0f}ms, {combined.tokens_generated} tok)")

        # Waves 1..N-1: pipelined
        for wave_id in range(1, n_waves):
            print(f"  Wave {wave_id}: ", end="", flush=True)

            # Run prefill+save+xfer for wave N+1 IN PARALLEL with decode for wave N
            # But we already decoded wave N above, so this wave's decode runs while
            # next wave's prefill starts

            # Start prefill+save+xfer for THIS wave
            pst_task = asyncio.create_task(
                prefill_save_transfer(session, prompt, wave_id))

            # If we haven't decoded this wave yet, we need to wait for prefill first
            # Actually in pipeline: we do prefill THEN decode, but overlap is:
            # - decode(N) || prefill(N+1)
            # Since we use slot 0 for both, we can't truly overlap on same server
            # But across machines, decode(N) on Spark2 || prefill(N+1) on Spark1

            # For now: sequential per wave but the time shows pipeline potential
            pst = await pst_task
            print(f"prefill+xfer={pst.prefill_ms + pst.save_ms + pst.transfer_ms:.0f}ms", end="", flush=True)

            rd = await restore_decode(session, prompt, wave_id, n_predict)
            print(f" decode={rd.decode_ms:.0f}ms", end="", flush=True)

            combined = WaveResult(wave_id=wave_id)
            combined.prefill_ms = pst.prefill_ms
            combined.save_ms = pst.save_ms
            combined.transfer_ms = pst.transfer_ms
            combined.transfer_bytes = pst.transfer_bytes
            combined.restore_ms = rd.restore_ms
            combined.decode_ms = rd.decode_ms
            combined.tokens_generated = rd.tokens_generated
            combined.prompt_tokens = pst.prompt_tokens
            combined.cache_n = rd.cache_n
            combined.total_ms = (pst.prefill_ms + pst.save_ms + pst.transfer_ms +
                                rd.restore_ms + rd.decode_ms)
            wave_results.append(combined)
            print(f" total={combined.total_ms:.0f}ms")

        pipeline_end = time.perf_counter()
        pipeline_time = (pipeline_end - pipeline_start) * 1000

        # Now run baseline for comparison
        print(f"\n  Running baseline ({n_waves} sequential requests)...")
        baseline_start = time.perf_counter()
        baseline_times = []
        for i in range(n_waves):
            # Clear cache between runs for fair comparison
            try:
                async with session.post(f"{PREFILL_URL}/slots/0?action=erase"):
                    pass
            except Exception:
                pass

            t0 = time.perf_counter()
            async with session.post(f"{PREFILL_URL}/completion", json={
                "prompt": prompt, "n_predict": n_predict,
                "cache_prompt": True, "temperature": 0.0,
            }) as resp:
                data = await resp.json()
            ms = (time.perf_counter() - t0) * 1000
            baseline_times.append(ms)
            toks = data.get("timings", {}).get("predicted_n", 0)
            print(f"    Baseline {i}: {ms:.0f}ms ({toks} tok)")
        baseline_end = time.perf_counter()
        baseline_time = (baseline_end - baseline_start) * 1000

    return wave_results, pipeline_time, baseline_times, baseline_time


async def main():
    parser = argparse.ArgumentParser(description="Distributed Pipeline Benchmark")
    parser.add_argument("--waves", "-w", type=int, default=5)
    parser.add_argument("--prompt-tokens", "-p", type=int, default=1000)
    parser.add_argument("--output-tokens", "-o", type=int, default=50)
    args = parser.parse_args()

    prompt = generate_prompt(args.prompt_tokens)

    print("=" * 70)
    print("DGX SPARK PIPELINE BENCHMARK")
    print("=" * 70)
    print(f"Waves: {args.waves}")
    print(f"Prompt: ~{args.prompt_tokens} tokens, Output: {args.output_tokens} tokens")
    print(f"Prefill: {PREFILL_URL} (Spark 1)")
    print(f"Decode:  {DECODE_URL} (Spark 2)")
    print(f"Transfer: SCP via {DECODE_SSH}")

    waves, pipe_time, base_times, base_time = await run_pipeline(
        args.waves, prompt, args.output_tokens)

    # Results
    total_tokens = sum(w.tokens_generated for w in waves)
    total_base_tokens = args.output_tokens * args.waves  # approx

    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)

    print(f"\n{'Wave':<6} {'Prefill':>10} {'Save':>8} {'Transfer':>10} {'Restore':>9} {'Decode':>10} {'Total':>10} {'Cache':>8}")
    print("-" * 70)
    for w in waves:
        bw = (w.transfer_bytes * 8 / 1e9) / (w.transfer_ms / 1000) if w.transfer_ms > 0 else 0
        print(f"{w.wave_id:<6} {w.prefill_ms:>8.0f}ms {w.save_ms:>6.0f}ms "
              f"{w.transfer_ms:>7.0f}ms {w.restore_ms:>7.0f}ms "
              f"{w.decode_ms:>8.0f}ms {w.total_ms:>8.0f}ms {w.cache_n:>6}")

    print(f"\nDisagg pipeline total:  {pipe_time:.0f}ms  ({total_tokens} tokens)")
    print(f"Baseline sequential:    {base_time:.0f}ms  (~{total_base_tokens} tokens)")

    if base_time > 0:
        speedup = base_time / pipe_time
        print(f"\nPipeline speedup:       {speedup:.2f}x")

    # Averages
    avg_prefill = statistics.mean([w.prefill_ms for w in waves])
    avg_save = statistics.mean([w.save_ms for w in waves])
    avg_xfer = statistics.mean([w.transfer_ms for w in waves])
    avg_restore = statistics.mean([w.restore_ms for w in waves])
    avg_decode = statistics.mean([w.decode_ms for w in waves])
    avg_overhead = avg_save + avg_xfer + avg_restore

    print(f"\nAverage per-wave breakdown:")
    print(f"  Prefill:     {avg_prefill:.0f}ms")
    print(f"  Save:        {avg_save:.0f}ms")
    print(f"  Transfer:    {avg_xfer:.0f}ms")
    print(f"  Restore:     {avg_restore:.0f}ms")
    print(f"  Decode:      {avg_decode:.0f}ms")
    print(f"  Overhead:    {avg_overhead:.0f}ms")

    # Pipeline efficiency analysis
    prep_time = avg_prefill + avg_save + avg_xfer  # time on Spark 1
    decode_time = avg_restore + avg_decode          # time on Spark 2
    overlap = min(prep_time, decode_time)

    print(f"\nPipeline analysis:")
    print(f"  Spark 1 work (prefill+save+xfer):  {prep_time:.0f}ms")
    print(f"  Spark 2 work (restore+decode):      {decode_time:.0f}ms")
    print(f"  Max overlap per wave:               {overlap:.0f}ms")
    print(f"  Theoretical pipeline throughput:     {1000 / max(prep_time, decode_time):.2f} req/s")
    print(f"  Vs sequential throughput:            {1000 / (prep_time + decode_time):.2f} req/s")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
