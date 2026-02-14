#!/usr/bin/env python3
"""
Benchmark for disaggregated inference across two DGX Spark machines.

Tests:
1. Baseline: Single server on one Spark doing prefill + decode
2. Disagg cross-machine: Prefill on Spark 1, KV transfer over 200Gbps, decode on Spark 2

Usage:
    # Run from any machine that can reach both Sparks
    python3 disagg/spark/benchmark.py

    # Run from Spark 1 (prefill machine) - default config
    python3 benchmark.py

    # Custom settings
    python3 benchmark.py --prompt-tokens 1000 --output-tokens 100 --runs 5
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import aiohttp


# Default: high-speed interconnect IPs
PREFILL_URL = os.environ.get("PREFILL_URL", "http://192.168.200.13:8080")
DECODE_URL = os.environ.get("DECODE_URL", "http://192.168.200.12:8081")
DECODE_SSH = os.environ.get("DECODE_SSH", "spark2")
PREFILL_SSH = os.environ.get("PREFILL_SSH", "")  # Empty = local
KV_CACHE_DIR = "/tmp/llama_kv_cache"


@dataclass
class TimingResult:
    total_ms: float
    prefill_ms: float
    decode_ms: float
    save_ms: float = 0.0
    transfer_ms: float = 0.0
    restore_ms: float = 0.0
    tokens_generated: int = 0
    prompt_tokens: int = 0
    transfer_bytes: int = 0
    cache_hit: bool = False


def generate_prompt(length: int = 2000) -> str:
    base = "The quick brown fox jumps over the lazy dog. " * 100
    chars_needed = length * 4
    return (base * (chars_needed // len(base) + 1))[:chars_needed]


async def scp_transfer(filename: str) -> tuple[float, int]:
    """Transfer KV cache file from prefill machine to decode machine."""
    src_path = f"{KV_CACHE_DIR}/{filename}"
    dst = f"{DECODE_SSH}:{KV_CACHE_DIR}/{filename}"

    if PREFILL_SSH:
        src = f"{PREFILL_SSH}:{src_path}"
    else:
        src = src_path

    # Get file size first (before transfer, for accurate bandwidth calc)
    try:
        if PREFILL_SSH:
            proc = await asyncio.create_subprocess_exec(
                "ssh", PREFILL_SSH, f"stat -c%s {src_path}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                "stat", "-c%s", src_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        stdout, _ = await proc.communicate()
        file_size = int(stdout.decode().strip())
    except Exception:
        file_size = 0

    t_start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        "scp", "-o", "StrictHostKeyChecking=no",
        "-o", "Compression=no",
        src, dst,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    transfer_ms = (time.perf_counter() - t_start) * 1000

    if proc.returncode != 0:
        raise Exception(f"SCP failed: {stderr.decode()}")

    return transfer_ms, file_size


async def baseline_completion(
    session: aiohttp.ClientSession,
    prompt: str,
    n_predict: int = 50,
) -> TimingResult:
    """Single server: prefill + decode on one Spark."""
    start = time.perf_counter()

    async with session.post(
        f"{PREFILL_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": True,
            "temperature": 0.0,
        }
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Baseline failed: {await resp.text()}")
        data = await resp.json()

    total_ms = (time.perf_counter() - start) * 1000
    timings = data.get("timings", {})

    return TimingResult(
        total_ms=total_ms,
        prefill_ms=timings.get("prompt_ms", 0),
        decode_ms=timings.get("predicted_ms", 0),
        tokens_generated=timings.get("predicted_n", 0),
        prompt_tokens=timings.get("prompt_n", 0),
    )


async def disagg_completion(
    session: aiohttp.ClientSession,
    prompt: str,
    n_predict: int = 50,
    session_id: str = "test",
) -> TimingResult:
    """Disaggregated: prefill on Spark 1, transfer, decode on Spark 2."""
    filename = f"session_{session_id}.bin"
    total_start = time.perf_counter()

    # Clear decode slot
    try:
        async with session.post(f"{DECODE_URL}/slots/0?action=erase") as resp:
            pass
    except Exception:
        pass

    # Step 1: Prefill
    t0 = time.perf_counter()
    async with session.post(
        f"{PREFILL_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": 0,
            "cache_prompt": True,
            "id_slot": 0,
        }
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Prefill failed: {await resp.text()}")
        prefill_data = await resp.json()
    prefill_ms = (time.perf_counter() - t0) * 1000
    prompt_tokens = prefill_data.get("timings", {}).get("prompt_n", 0)

    # Step 2: Save KV cache
    t0 = time.perf_counter()
    async with session.post(
        f"{PREFILL_URL}/slots/0?action=save",
        json={"filename": filename}
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Save failed: {await resp.text()}")
        save_data = await resp.json()
    save_ms = (time.perf_counter() - t0) * 1000
    print(f"    Saved {save_data.get('n_saved', 0)} tokens")

    # Step 3: Transfer KV cache over 200Gbps interconnect
    transfer_ms, transfer_bytes = await scp_transfer(filename)
    bw = (transfer_bytes * 8 / 1e9) / (transfer_ms / 1000) if transfer_ms > 0 else 0
    print(f"    Transferred {transfer_bytes/1e6:.1f}MB in {transfer_ms:.1f}ms ({bw:.1f} Gbps)")

    # Step 4: Restore on decode server
    t0 = time.perf_counter()
    async with session.post(
        f"{DECODE_URL}/slots/0?action=restore",
        json={"filename": filename}
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Restore failed: {await resp.text()}")
    restore_ms = (time.perf_counter() - t0) * 1000

    # Step 5: Decode
    t0 = time.perf_counter()
    async with session.post(
        f"{DECODE_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": True,
            "id_slot": 0,
        }
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Decode failed: {await resp.text()}")
        decode_data = await resp.json()
    decode_ms = (time.perf_counter() - t0) * 1000
    decode_timings = decode_data.get("timings", {})
    cache_hit = decode_timings.get("prompt_n", 0) == 0 or decode_timings.get("prompt_ms", 0) < 10

    total_ms = (time.perf_counter() - total_start) * 1000

    return TimingResult(
        total_ms=total_ms,
        prefill_ms=prefill_ms,
        save_ms=save_ms,
        transfer_ms=transfer_ms,
        restore_ms=restore_ms,
        decode_ms=decode_ms,
        tokens_generated=decode_timings.get("predicted_n", 0),
        prompt_tokens=prompt_tokens,
        transfer_bytes=transfer_bytes,
        cache_hit=cache_hit,
    )


async def disagg_completion_buffer(
    session: aiohttp.ClientSession,
    prompt: str,
    n_predict: int = 50,
    session_id: str = "test",
) -> TimingResult:
    """Disaggregated using HTTP buffer transfer (no files, no SCP)."""
    total_start = time.perf_counter()

    # Clear decode slot
    try:
        async with session.post(f"{DECODE_URL}/slots/0?action=erase") as resp:
            pass
    except Exception:
        pass

    # Step 1: Prefill
    t0 = time.perf_counter()
    async with session.post(
        f"{PREFILL_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": 0,
            "cache_prompt": True,
            "id_slot": 0,
        }
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Prefill failed: {await resp.text()}")
        prefill_data = await resp.json()
    prefill_ms = (time.perf_counter() - t0) * 1000
    prompt_tokens = prefill_data.get("timings", {}).get("prompt_n", 0)

    # Step 2: Export KV cache as binary from prefill server
    t0 = time.perf_counter()
    async with session.get(f"{PREFILL_URL}/slots/0/kv-data") as resp:
        if resp.status != 200:
            raise Exception(f"Export failed: {await resp.text()}")
        kv_data = await resp.read()
    export_ms = (time.perf_counter() - t0) * 1000
    transfer_bytes = len(kv_data)

    # Step 3: Import KV cache binary to decode server
    t0 = time.perf_counter()
    async with session.post(
        f"{DECODE_URL}/slots/0/kv-data",
        data=kv_data,
        headers={"Content-Type": "application/octet-stream"},
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Import failed: {await resp.text()}")
        import_data = await resp.json()
    import_ms = (time.perf_counter() - t0) * 1000
    print(f"    Buffer: {transfer_bytes/1e6:.1f}MB, export={export_ms:.0f}ms, import={import_ms:.0f}ms")

    # Step 4: Decode
    t0 = time.perf_counter()
    async with session.post(
        f"{DECODE_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": True,
            "id_slot": 0,
        }
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Decode failed: {await resp.text()}")
        decode_data = await resp.json()
    decode_ms = (time.perf_counter() - t0) * 1000
    decode_timings = decode_data.get("timings", {})
    cache_hit = decode_timings.get("prompt_n", 0) == 0 or decode_timings.get("prompt_ms", 0) < 10

    total_ms = (time.perf_counter() - total_start) * 1000

    bw = (transfer_bytes * 8 / 1e9) / (import_ms / 1000) if import_ms > 0 else 0
    print(f"    Transfer BW: {bw:.1f} Gbps")

    return TimingResult(
        total_ms=total_ms,
        prefill_ms=prefill_ms,
        save_ms=export_ms,       # export (serialize to buffer)
        transfer_ms=import_ms,   # import (network transfer + deserialize)
        restore_ms=0,            # no separate restore step
        decode_ms=decode_ms,
        tokens_generated=decode_timings.get("predicted_n", 0),
        prompt_tokens=prompt_tokens,
        transfer_bytes=transfer_bytes,
        cache_hit=cache_hit,
    )


async def disagg_completion_push(
    session: aiohttp.ClientSession,
    prompt: str,
    n_predict: int = 50,
    session_id: str = "test",
) -> TimingResult:
    """Disaggregated using direct server-to-server push (single hop)."""
    total_start = time.perf_counter()

    # Clear decode slot
    try:
        async with session.post(f"{DECODE_URL}/slots/0?action=erase") as resp:
            pass
    except Exception:
        pass

    # Step 1: Prefill
    t0 = time.perf_counter()
    async with session.post(
        f"{PREFILL_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": 0,
            "cache_prompt": True,
            "id_slot": 0,
        }
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Prefill failed: {await resp.text()}")
        prefill_data = await resp.json()
    prefill_ms = (time.perf_counter() - t0) * 1000
    prompt_tokens = prefill_data.get("timings", {}).get("prompt_n", 0)

    # Step 2: Push KV cache directly from prefill server to decode server
    t0 = time.perf_counter()
    async with session.post(
        f"{PREFILL_URL}/slots/0/kv-data/push",
        json={
            "target_url": DECODE_URL,
            "target_slot": 0,
        }
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Push failed: {await resp.text()}")
        push_data = await resp.json()
    push_ms = (time.perf_counter() - t0) * 1000
    push_timings = push_data.get("timings", {})
    transfer_bytes = push_data.get("buffer_size", 0)
    export_ms = push_timings.get("export_ms", 0)
    transfer_ms = push_timings.get("transfer_ms", 0)
    import_ms = push_timings.get("import_ms", 0)
    push_total_ms = push_timings.get("total_ms", 0)
    gbps = push_data.get("throughput_gbps", 0)
    print(f"    Push: {transfer_bytes/1e6:.1f}MB, export={export_ms:.0f}ms, "
          f"transfer={transfer_ms:.0f}ms, import={import_ms:.0f}ms, "
          f"total={push_total_ms:.0f}ms ({gbps:.1f} Gbps)")

    # Step 3: Decode
    t0 = time.perf_counter()
    async with session.post(
        f"{DECODE_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": True,
            "id_slot": 0,
        }
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Decode failed: {await resp.text()}")
        decode_data = await resp.json()
    decode_ms = (time.perf_counter() - t0) * 1000
    decode_timings = decode_data.get("timings", {})
    cache_hit = decode_timings.get("prompt_n", 0) == 0 or decode_timings.get("prompt_ms", 0) < 10

    total_ms = (time.perf_counter() - total_start) * 1000

    return TimingResult(
        total_ms=total_ms,
        prefill_ms=prefill_ms,
        save_ms=export_ms,        # export (serialize)
        transfer_ms=transfer_ms,  # network transfer
        restore_ms=import_ms,     # import (deserialize on target)
        decode_ms=decode_ms,
        tokens_generated=decode_timings.get("predicted_n", 0),
        prompt_tokens=prompt_tokens,
        transfer_bytes=transfer_bytes,
        cache_hit=cache_hit,
    )


def print_result(name: str, r: TimingResult):
    print(f"\n{name}:")
    print(f"  Total:        {r.total_ms:8.1f} ms")
    print(f"  Prefill:      {r.prefill_ms:8.1f} ms")
    if r.save_ms > 0:
        print(f"  Save:         {r.save_ms:8.1f} ms")
    if r.transfer_ms > 0:
        bw = (r.transfer_bytes * 8 / 1e9) / (r.transfer_ms / 1000) if r.transfer_ms > 0 else 0
        print(f"  Transfer:     {r.transfer_ms:8.1f} ms  ({r.transfer_bytes/1e6:.1f}MB, {bw:.1f} Gbps)")
    if r.restore_ms > 0:
        print(f"  Restore:      {r.restore_ms:8.1f} ms")
    print(f"  Decode:       {r.decode_ms:8.1f} ms")
    print(f"  Prompt tok:   {r.prompt_tokens}")
    print(f"  Output tok:   {r.tokens_generated}")
    if r.tokens_generated > 0 and r.decode_ms > 0:
        print(f"  Decode tok/s: {r.tokens_generated / (r.decode_ms / 1000):.1f}")
    if r.cache_hit:
        print(f"  Cache hit:    YES")


async def run_benchmark(prompt: str, n_predict: int = 50, num_runs: int = 3, mode: str = "buffer"):
    print("=" * 70)
    print("DGX SPARK DISAGGREGATED PREFILL BENCHMARK")
    print("=" * 70)
    print(f"Prefill server: {PREFILL_URL}")
    print(f"Decode server:  {DECODE_URL}")
    print(f"Transfer mode:  {mode}")
    print(f"Prompt:         ~{len(prompt)} chars")
    print(f"Output tokens:  {n_predict}")
    print(f"Runs:           {num_runs}")

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warmup
        print("\nWarmup...")
        try:
            await baseline_completion(session, prompt[:500], n_predict=5)
        except Exception as e:
            print(f"Warmup failed: {e}")

        # Clear slots
        for url in [PREFILL_URL, DECODE_URL]:
            try:
                async with session.post(f"{url}/slots/0?action=erase"):
                    pass
            except Exception:
                pass

        # Baseline
        print("\n" + "-" * 70)
        print("BASELINE (single Spark, prefill + decode on same GPU)")
        print("-" * 70)
        baseline_results = []
        for i in range(num_runs):
            try:
                async with session.post(f"{PREFILL_URL}/slots/0?action=erase"):
                    pass
            except Exception:
                pass
            r = await baseline_completion(session, prompt, n_predict)
            baseline_results.append(r)
            print(f"  Run {i+1}: {r.total_ms:.1f}ms total, {r.prefill_ms:.1f}ms prefill, "
                  f"{r.decode_ms:.1f}ms decode")

        mode_map = {
            "scp": ["scp"],
            "buffer": ["buffer"],
            "push": ["push"],
            "both": ["scp", "buffer"],
            "all": ["scp", "buffer", "push"],
        }
        modes_to_run = mode_map.get(mode, [mode])
        all_disagg = {}

        for m in modes_to_run:
            labels = {"scp": "SCP (file-based)", "buffer": "HTTP BUFFER (double-hop)", "push": "HTTP PUSH (direct)"}
            label = labels.get(m, m)
            print("\n" + "-" * 70)
            print(f"DISAGGREGATED - {label}")
            print("-" * 70)
            disagg_results = []
            for i in range(num_runs):
                for url in [PREFILL_URL, DECODE_URL]:
                    try:
                        async with session.post(f"{url}/slots/0?action=erase"):
                            pass
                    except Exception:
                        pass
                if m == "scp":
                    r = await disagg_completion(session, prompt, n_predict, session_id=f"run_{i}")
                elif m == "push":
                    r = await disagg_completion_push(session, prompt, n_predict, session_id=f"run_{i}")
                else:
                    r = await disagg_completion_buffer(session, prompt, n_predict, session_id=f"run_{i}")
                disagg_results.append(r)
                print(f"  Run {i+1}: {r.total_ms:.1f}ms total "
                      f"(prefill:{r.prefill_ms:.0f} save/export:{r.save_ms:.0f} "
                      f"xfer/import:{r.transfer_ms:.0f} restore:{r.restore_ms:.0f} "
                      f"decode:{r.decode_ms:.0f})")
            all_disagg[m] = disagg_results

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    b_avg = statistics.mean([r.total_ms for r in baseline_results])
    print(f"\nBaseline avg:        {b_avg:8.1f} ms")

    for m, disagg_results in all_disagg.items():
        labels = {"scp": "SCP", "buffer": "Buffer", "push": "Push"}
        label = labels.get(m, m)
        d_avg = statistics.mean([r.total_ms for r in disagg_results])

        print(f"\nDisagg ({label}) avg:  {d_avg:8.1f} ms")
        print(f"Difference:          {d_avg - b_avg:+8.1f} ms")
        if d_avg > 0:
            print(f"Speedup:             {b_avg / d_avg:.2f}x")

        print(f"\nDisagg ({label}) breakdown (avg):")
        print(f"  Prefill:           {statistics.mean([r.prefill_ms for r in disagg_results]):8.1f} ms")
        print(f"  Save/Export:       {statistics.mean([r.save_ms for r in disagg_results]):8.1f} ms")
        print(f"  Transfer:          {statistics.mean([r.transfer_ms for r in disagg_results]):8.1f} ms")
        if any(r.restore_ms > 0 for r in disagg_results):
            print(f"  Restore/Import:    {statistics.mean([r.restore_ms for r in disagg_results]):8.1f} ms")
        print(f"  Decode:            {statistics.mean([r.decode_ms for r in disagg_results]):8.1f} ms")

        overhead = statistics.mean([r.save_ms + r.transfer_ms + r.restore_ms for r in disagg_results])
        print(f"  Total overhead:    {overhead:8.1f} ms")

        if disagg_results[0].transfer_bytes > 0:
            avg_bytes = statistics.mean([r.transfer_bytes for r in disagg_results])
            avg_xfer_ms = statistics.mean([r.transfer_ms for r in disagg_results])
            if avg_xfer_ms > 0:
                bw = (avg_bytes * 8 / 1e9) / (avg_xfer_ms / 1000)
                print(f"  Avg transfer:      {avg_bytes/1e6:.1f} MB at {bw:.1f} Gbps")

    print("\n" + "-" * 70)
    print("Best results:")
    print_result("Baseline (best)", min(baseline_results, key=lambda r: r.total_ms))
    for m, disagg_results in all_disagg.items():
        labels = {"scp": "SCP", "buffer": "Buffer", "push": "Push"}
        label = labels.get(m, m)
        print_result(f"Disagg {label} (best)", min(disagg_results, key=lambda r: r.total_ms))
    print("=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="DGX Spark Disagg Benchmark")
    parser.add_argument("--prompt-tokens", type=int, default=2000)
    parser.add_argument("--output-tokens", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mode", choices=["scp", "buffer", "push", "both", "all"], default="push",
                        help="Transfer mode: scp (file+SSH), buffer (HTTP double-hop), push (HTTP direct), both (scp+buffer), all")
    parser.add_argument("--prefill-url", type=str, default=None)
    parser.add_argument("--decode-url", type=str, default=None)
    parser.add_argument("--decode-ssh", type=str, default=None)

    args = parser.parse_args()

    global PREFILL_URL, DECODE_URL, DECODE_SSH
    if args.prefill_url:
        PREFILL_URL = args.prefill_url
    if args.decode_url:
        DECODE_URL = args.decode_url
    if args.decode_ssh:
        DECODE_SSH = args.decode_ssh

    prompt = generate_prompt(args.prompt_tokens)

    # Check servers
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for name, url in [("Prefill", PREFILL_URL), ("Decode", DECODE_URL)]:
            try:
                async with session.get(f"{url}/health") as resp:
                    if "ok" not in (await resp.text()).lower():
                        raise Exception("Not healthy")
                print(f"  OK: {name} at {url}")
            except Exception as e:
                print(f"  FAIL: {name} at {url} - {e}")
                print("Start servers: ./disagg/spark/start_servers.sh")
                return

    await run_benchmark(prompt, args.output_tokens, args.runs, mode=args.mode)


if __name__ == "__main__":
    asyncio.run(main())
