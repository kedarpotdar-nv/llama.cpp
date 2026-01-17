#!/usr/bin/env python3
"""
Benchmark script for disaggregated prefill vs baseline.

This script measures:
1. Baseline: Single server doing prefill + decode
2. Disaggregated: Prefill server -> save KV -> Decode server -> generate

Usage:
    python3 benchmark.py [--prompt-file FILE] [--output-tokens N]
"""

import argparse
import json
import os
import time
import requests
from dataclasses import dataclass
from typing import Optional
import statistics


@dataclass
class TimingResult:
    """Timing results for a single request"""
    total_ms: float
    prefill_ms: float
    decode_ms: float
    save_ms: float = 0.0
    restore_ms: float = 0.0
    tokens_generated: int = 0
    prompt_tokens: int = 0


PREFILL_SERVER = "http://localhost:8080"
DECODE_SERVER = "http://localhost:8081"
BASELINE_SERVER = "http://localhost:8080"  # Use prefill server as baseline too


def generate_prompt(length: int = 2000) -> str:
    """Generate a test prompt of approximately `length` tokens"""
    # Simple repeated text - roughly 1 token per 4 chars
    base = "The quick brown fox jumps over the lazy dog. " * 100
    # Adjust to approximate token count
    chars_needed = length * 4
    prompt = (base * (chars_needed // len(base) + 1))[:chars_needed]
    return prompt


def baseline_completion(prompt: str, n_predict: int = 50) -> TimingResult:
    """Single server: prefill + decode together"""
    start = time.perf_counter()
    
    response = requests.post(
        f"{BASELINE_SERVER}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": True,
            "temperature": 0.0,
        },
        timeout=300
    )
    
    total_ms = (time.perf_counter() - start) * 1000
    
    if response.status_code != 200:
        raise Exception(f"Baseline request failed: {response.text}")
    
    data = response.json()
    timings = data.get("timings", {})
    
    return TimingResult(
        total_ms=total_ms,
        prefill_ms=timings.get("prompt_ms", 0),
        decode_ms=timings.get("predicted_ms", 0),
        tokens_generated=timings.get("predicted_n", 0),
        prompt_tokens=timings.get("prompt_n", 0),
    )


def disagg_completion(prompt: str, n_predict: int = 50, session_id: str = "test") -> TimingResult:
    """
    Disaggregated: 
    1. Prefill on server A (n_predict=0)
    2. Save KV cache
    3. Restore on server B
    4. Generate tokens
    """
    filename = f"session_{session_id}.bin"
    
    total_start = time.perf_counter()
    
    # Step 0: Clear decode server slot first to make room
    try:
        requests.post(f"{DECODE_SERVER}/slots/0?action=erase", timeout=10)
    except:
        pass
    
    # Step 1: Prefill only (n_predict=0)
    prefill_start = time.perf_counter()
    response = requests.post(
        f"{PREFILL_SERVER}/completion",
        json={
            "prompt": prompt,
            "n_predict": 0,  # Prefill only!
            "cache_prompt": True,
            "id_slot": 0,
        },
        timeout=300
    )
    prefill_ms = (time.perf_counter() - prefill_start) * 1000
    
    if response.status_code != 200:
        raise Exception(f"Prefill request failed: {response.text}")
    
    prefill_data = response.json()
    prompt_tokens = prefill_data.get("timings", {}).get("prompt_n", 0)
    
    # Step 2: Save KV cache
    save_start = time.perf_counter()
    response = requests.post(
        f"{PREFILL_SERVER}/slots/0?action=save",
        json={"filename": filename},
        timeout=60
    )
    save_ms = (time.perf_counter() - save_start) * 1000
    
    if response.status_code != 200:
        raise Exception(f"Save request failed: {response.text}")
    
    save_data = response.json()
    n_saved = save_data.get("n_saved", 0)
    print(f"    [DEBUG] Saved {n_saved} tokens to {filename}")
    
    # Step 3: Restore on decode server
    restore_start = time.perf_counter()
    response = requests.post(
        f"{DECODE_SERVER}/slots/0?action=restore",
        json={"filename": filename},
        timeout=60
    )
    restore_ms = (time.perf_counter() - restore_start) * 1000
    
    if response.status_code != 200:
        # Debug: check if file exists and server config
        print(f"    [DEBUG] Restore failed. Checking configs...")
        try:
            prefill_props = requests.get(f"{PREFILL_SERVER}/props", timeout=5).json()
            decode_props = requests.get(f"{DECODE_SERVER}/props", timeout=5).json()
            print(f"    [DEBUG] Prefill n_ctx: {prefill_props.get('default_generation_settings', {}).get('n_ctx', 'unknown')}")
            print(f"    [DEBUG] Decode n_ctx: {decode_props.get('default_generation_settings', {}).get('n_ctx', 'unknown')}")
        except Exception as e:
            print(f"    [DEBUG] Could not get props: {e}")
        raise Exception(f"Restore request failed: {response.text}")
    
    # Step 4: Generate tokens (should skip prefill due to restored cache)
    decode_start = time.perf_counter()
    response = requests.post(
        f"{DECODE_SERVER}/completion",
        json={
            "prompt": prompt,  # Same prompt - should hit cache
            "n_predict": n_predict,
            "cache_prompt": True,
            "id_slot": 0,
        },
        timeout=300
    )
    decode_ms = (time.perf_counter() - decode_start) * 1000
    
    if response.status_code != 200:
        raise Exception(f"Decode request failed: {response.text}")
    
    decode_data = response.json()
    decode_timings = decode_data.get("timings", {})
    
    total_ms = (time.perf_counter() - total_start) * 1000
    
    return TimingResult(
        total_ms=total_ms,
        prefill_ms=prefill_ms,
        save_ms=save_ms,
        restore_ms=restore_ms,
        decode_ms=decode_ms,
        tokens_generated=decode_timings.get("predicted_n", 0),
        prompt_tokens=prompt_tokens,
    )


def print_result(name: str, result: TimingResult):
    """Pretty print a timing result"""
    print(f"\n{name}:")
    print(f"  Total time:      {result.total_ms:8.1f} ms")
    print(f"  Prefill time:    {result.prefill_ms:8.1f} ms")
    if result.save_ms > 0:
        print(f"  Save time:       {result.save_ms:8.1f} ms")
    if result.restore_ms > 0:
        print(f"  Restore time:    {result.restore_ms:8.1f} ms")
    print(f"  Decode time:     {result.decode_ms:8.1f} ms")
    print(f"  Prompt tokens:   {result.prompt_tokens}")
    print(f"  Output tokens:   {result.tokens_generated}")
    if result.tokens_generated > 0:
        print(f"  Tokens/sec:      {result.tokens_generated / (result.decode_ms / 1000):.1f}")


def run_benchmark(
    prompt: str,
    n_predict: int = 50,
    num_runs: int = 3,
    warmup: bool = True
):
    """Run the full benchmark comparison"""
    
    print("=" * 60)
    print("DISAGGREGATED PREFILL BENCHMARK")
    print("=" * 60)
    print(f"Prompt length: ~{len(prompt)} chars")
    print(f"Output tokens: {n_predict}")
    print(f"Runs: {num_runs}")
    
    # Warmup run
    if warmup:
        print("\nWarmup run...")
        try:
            baseline_completion(prompt[:500], n_predict=5)
        except Exception as e:
            print(f"Warmup failed: {e}")
    
    # Clear caches before benchmark
    try:
        requests.post(f"{PREFILL_SERVER}/slots/0?action=erase", timeout=10)
        requests.post(f"{DECODE_SERVER}/slots/0?action=erase", timeout=10)
    except:
        pass
    
    baseline_results = []
    disagg_results = []
    
    print("\n" + "-" * 60)
    print("Running Baseline (single server)...")
    print("-" * 60)
    
    for i in range(num_runs):
        # Clear cache between runs for fair comparison
        try:
            requests.post(f"{BASELINE_SERVER}/slots/0?action=erase", timeout=10)
        except:
            pass
        
        result = baseline_completion(prompt, n_predict)
        baseline_results.append(result)
        print(f"  Run {i+1}: {result.total_ms:.1f} ms total, {result.prefill_ms:.1f} ms prefill")
    
    print("\n" + "-" * 60)
    print("Running Disaggregated (prefill + transfer + decode)...")
    print("-" * 60)
    
    for i in range(num_runs):
        # Clear caches between runs
        try:
            requests.post(f"{PREFILL_SERVER}/slots/0?action=erase", timeout=10)
            requests.post(f"{DECODE_SERVER}/slots/0?action=erase", timeout=10)
        except:
            pass
        
        result = disagg_completion(prompt, n_predict, session_id=f"run_{i}")
        disagg_results.append(result)
        print(f"  Run {i+1}: {result.total_ms:.1f} ms total "
              f"(prefill: {result.prefill_ms:.1f}, save: {result.save_ms:.1f}, "
              f"restore: {result.restore_ms:.1f}, decode: {result.decode_ms:.1f})")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    baseline_avg = statistics.mean([r.total_ms for r in baseline_results])
    disagg_avg = statistics.mean([r.total_ms for r in disagg_results])
    
    print(f"\nBaseline avg:      {baseline_avg:8.1f} ms")
    print(f"Disagg avg:        {disagg_avg:8.1f} ms")
    print(f"Difference:        {disagg_avg - baseline_avg:+8.1f} ms")
    print(f"Speedup:           {baseline_avg / disagg_avg:.2f}x")
    
    # Breakdown
    disagg_overhead = statistics.mean([r.save_ms + r.restore_ms for r in disagg_results])
    print(f"\nDisagg overhead (save+restore): {disagg_overhead:.1f} ms")
    
    print("\n" + "-" * 60)
    print("Best results:")
    print_result("Baseline (best)", min(baseline_results, key=lambda r: r.total_ms))
    print_result("Disaggregated (best)", min(disagg_results, key=lambda r: r.total_ms))
    
    return baseline_results, disagg_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark disaggregated prefill")
    parser.add_argument("--prompt-file", type=str, help="File containing prompt text")
    parser.add_argument("--prompt-tokens", type=int, default=2000, 
                        help="Approximate prompt length in tokens (if no file)")
    parser.add_argument("--output-tokens", type=int, default=50,
                        help="Number of tokens to generate")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    
    args = parser.parse_args()
    
    # Get prompt
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r") as f:
            prompt = f.read()
        print(f"Loaded prompt from {args.prompt_file}")
    else:
        prompt = generate_prompt(args.prompt_tokens)
        print(f"Generated synthetic prompt (~{args.prompt_tokens} tokens)")
    
    # Check servers are running
    for name, url in [("Prefill", PREFILL_SERVER), ("Decode", DECODE_SERVER)]:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if "ok" not in r.text.lower():
                raise Exception("Not healthy")
        except Exception as e:
            print(f"Error: {name} server not responding at {url}")
            print("Start servers with: ./disagg/start_servers.sh <model_path>")
            return 1
    
    # Run benchmark
    run_benchmark(
        prompt=prompt,
        n_predict=args.output_tokens,
        num_runs=args.runs,
        warmup=not args.no_warmup,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
