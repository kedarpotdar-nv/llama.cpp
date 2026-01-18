#!/usr/bin/env python3
"""
Concurrent Benchmark for Disaggregated Prefill

Tests the scenario where disaggregated prefill should help:
- High Input Sequence Length (ISL): Long prompts
- Small Output Sequence Length (OSL): Short generations
- High Concurrency: Many simultaneous requests

Usage:
    python3 disagg/concurrent_benchmark.py --concurrency 10 --prompt-tokens 1000 --output-tokens 20
"""

import argparse
import asyncio
import time
import statistics
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import aiohttp

# Configuration
ORCHESTRATOR_URL = "http://localhost:9000"
BASELINE_URL = "http://localhost:8080"  # Direct to prefill server for baseline
PREFILL_URL = "http://localhost:8080"
DECODE_URL = "http://localhost:8081"


@dataclass
class RequestResult:
    """Result of a single request"""
    request_id: int
    success: bool
    total_time_ms: float
    prefill_time_ms: float = 0
    transfer_time_ms: float = 0
    decode_time_ms: float = 0
    cache_n: int = 0
    prompt_n: int = 0
    tokens_generated: int = 0
    tokens_evaluated: int = 0  # Prompt tokens processed
    prompt_tps: float = 0      # Prompt processing tokens/sec
    decode_tps: float = 0      # Decode tokens/sec
    error: str = ""


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results"""
    mode: str
    concurrency: int
    n_requests: int
    prompt_tokens: int
    output_tokens: int
    
    # Timing stats
    total_time_sec: float = 0
    avg_latency_ms: float = 0
    p50_latency_ms: float = 0
    p90_latency_ms: float = 0
    p99_latency_ms: float = 0
    min_latency_ms: float = 0
    max_latency_ms: float = 0
    
    # Throughput
    requests_per_sec: float = 0
    tokens_per_sec: float = 0
    
    # TPS metrics (server-reported)
    avg_prompt_tps: float = 0   # Avg prompt processing TPS
    avg_decode_tps: float = 0   # Avg decode TPS
    
    # Phase timing breakdown (for disagg)
    avg_prefill_ms: float = 0
    avg_transfer_ms: float = 0
    avg_decode_ms: float = 0
    
    # Cache stats (for disagg)
    avg_cache_n: float = 0
    cache_hit_rate: float = 0
    
    # Success rate
    success_count: int = 0
    failure_count: int = 0
    
    results: List[RequestResult] = field(default_factory=list)


def generate_long_prompt(target_tokens: int) -> str:
    """Generate a prompt that's approximately target_tokens long"""
    # Each word is roughly 1-2 tokens, so use ~0.75 words per token
    base_text = """
    Please analyze the following complex scenario in detail. Consider all aspects carefully.
    
    In a large distributed computing environment, we need to optimize the allocation of 
    computational resources across multiple data centers. Each data center has different 
    capabilities, costs, and latency characteristics. The workload consists of various 
    types of tasks including real-time processing, batch analytics, machine learning 
    inference, and data storage operations.
    
    The system must handle variable load patterns throughout the day, with peak usage 
    during business hours and lower utilization during nights and weekends. Cost 
    optimization is important, but we must also maintain strict SLAs for response times 
    and availability. The infrastructure includes a mix of on-premises servers, cloud 
    instances, and edge computing nodes.
    
    Key considerations include network bandwidth between locations, data locality 
    requirements, failover capabilities, and the need for geographic redundancy. 
    Some workloads are stateless and can be easily migrated, while others require 
    persistent state and careful coordination during any migration.
    
    Additionally, we must consider security requirements, compliance regulations, 
    and the environmental impact of our computing choices. Energy efficiency and 
    carbon footprint are increasingly important factors in infrastructure decisions.
    """
    
    # Repeat to reach target length
    words_needed = int(target_tokens * 0.75)
    base_words = base_text.split()
    
    result_words = []
    while len(result_words) < words_needed:
        result_words.extend(base_words)
    
    return " ".join(result_words[:words_needed])


async def make_baseline_request(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
    n_predict: int
) -> RequestResult:
    """Make a direct request to the baseline server"""
    t_start = time.perf_counter()
    
    try:
        async with session.post(
            f"{BASELINE_URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": 0.7,
            },
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                return RequestResult(
                    request_id=request_id,
                    success=False,
                    total_time_ms=(time.perf_counter() - t_start) * 1000,
                    error=f"HTTP {resp.status}"
                )
            
            result = await resp.json()
            t_end = time.perf_counter()
            
            timings = result.get("timings", {})
            
            return RequestResult(
                request_id=request_id,
                success=True,
                total_time_ms=(t_end - t_start) * 1000,
                prefill_time_ms=timings.get("prompt_ms", 0),
                decode_time_ms=timings.get("predicted_ms", 0),
                tokens_generated=timings.get("predicted_n", 0),
                tokens_evaluated=timings.get("prompt_n", 0),
                prompt_n=timings.get("prompt_n", 0),
                cache_n=timings.get("cache_n", 0),
                prompt_tps=timings.get("prompt_per_second", 0),
                decode_tps=timings.get("predicted_per_second", 0),
            )
    
    except Exception as e:
        return RequestResult(
            request_id=request_id,
            success=False,
            total_time_ms=(time.perf_counter() - t_start) * 1000,
            error=str(e)
        )


async def make_disagg_request(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
    n_predict: int
) -> RequestResult:
    """Make a request through the disaggregated orchestrator"""
    t_start = time.perf_counter()
    
    try:
        async with session.post(
            f"{ORCHESTRATOR_URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": 0.7,
                "disagg": True,
                "session_id": f"bench_{request_id}_{int(time.time()*1000)}",
            },
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return RequestResult(
                    request_id=request_id,
                    success=False,
                    total_time_ms=(time.perf_counter() - t_start) * 1000,
                    error=f"HTTP {resp.status}: {error_text[:100]}"
                )
            
            result = await resp.json()
            t_end = time.perf_counter()
            
            metrics = result.get("disagg_metrics", {})
            timings = result.get("timings", {})
            
            # For disagg: use PREFILL server's prompt TPS (where real work happens)
            # The decode server's prompt TPS is meaningless with cache hit (~1 token)
            return RequestResult(
                request_id=request_id,
                success=True,
                total_time_ms=(t_end - t_start) * 1000,
                prefill_time_ms=metrics.get("prefill_time_ms", 0),
                transfer_time_ms=metrics.get("transfer_time_ms", 0),
                decode_time_ms=metrics.get("decode_time_ms", 0),
                cache_n=metrics.get("cache_n", 0),
                prompt_n=metrics.get("prompt_n", 0),
                tokens_evaluated=metrics.get("prefill_tokens_evaluated", metrics.get("tokens_evaluated", 0)),
                tokens_generated=metrics.get("n_generated_tokens", timings.get("predicted_n", 0)),
                prompt_tps=metrics.get("prefill_prompt_tps", 0),  # From prefill server!
                decode_tps=metrics.get("decode_tps", timings.get("predicted_per_second", 0)),
            )
    
    except Exception as e:
        return RequestResult(
            request_id=request_id,
            success=False,
            total_time_ms=(time.perf_counter() - t_start) * 1000,
            error=str(e)
        )


async def run_benchmark(
    mode: str,  # "baseline" or "disagg"
    concurrency: int,
    n_requests: int,
    prompt: str,
    output_tokens: int,
    prompt_tokens: int,
) -> BenchmarkResult:
    """Run benchmark with specified concurrency"""
    
    result = BenchmarkResult(
        mode=mode,
        concurrency=concurrency,
        n_requests=n_requests,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )
    
    # Clear orchestrator sessions before disagg test
    if mode == "disagg":
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(f"{ORCHESTRATOR_URL}/clear")
            except:
                pass
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(session: aiohttp.ClientSession, request_id: int):
        async with semaphore:
            if mode == "baseline":
                return await make_baseline_request(session, request_id, prompt, output_tokens)
            else:
                return await make_disagg_request(session, request_id, prompt, output_tokens)
    
    # Run all requests
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        t_start = time.perf_counter()
        
        tasks = [bounded_request(session, i) for i in range(n_requests)]
        results = await asyncio.gather(*tasks)
        
        t_end = time.perf_counter()
        result.total_time_sec = t_end - t_start
    
    # Analyze results
    result.results = results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    result.success_count = len(successful)
    result.failure_count = len(failed)
    
    if successful:
        latencies = [r.total_time_ms for r in successful]
        latencies.sort()
        
        result.avg_latency_ms = statistics.mean(latencies)
        result.min_latency_ms = min(latencies)
        result.max_latency_ms = max(latencies)
        result.p50_latency_ms = latencies[len(latencies) // 2]
        result.p90_latency_ms = latencies[int(len(latencies) * 0.9)]
        result.p99_latency_ms = latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1]
        
        # Throughput
        total_tokens = sum(r.tokens_generated for r in successful)
        result.requests_per_sec = len(successful) / result.total_time_sec
        result.tokens_per_sec = total_tokens / result.total_time_sec
        
        # TPS metrics (server-reported averages)
        prompt_tps_values = [r.prompt_tps for r in successful if r.prompt_tps > 0]
        decode_tps_values = [r.decode_tps for r in successful if r.decode_tps > 0]
        result.avg_prompt_tps = statistics.mean(prompt_tps_values) if prompt_tps_values else 0
        result.avg_decode_tps = statistics.mean(decode_tps_values) if decode_tps_values else 0
        
        # Phase timing breakdown
        result.avg_prefill_ms = statistics.mean([r.prefill_time_ms for r in successful])
        result.avg_decode_ms = statistics.mean([r.decode_time_ms for r in successful])
        if mode == "disagg":
            result.avg_transfer_ms = statistics.mean([r.transfer_time_ms for r in successful])
        
        # Cache stats (disagg only)
        if mode == "disagg":
            cache_ns = [r.cache_n for r in successful]
            result.avg_cache_n = statistics.mean(cache_ns) if cache_ns else 0
            result.cache_hit_rate = sum(1 for c in cache_ns if c > 0) / len(cache_ns) if cache_ns else 0
    
    return result


def print_results(baseline: BenchmarkResult, disagg: BenchmarkResult):
    """Print comparison of results"""
    
    print("\n" + "="*80)
    print("CONCURRENT BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Concurrency:    {baseline.concurrency}")
    print(f"  Requests:       {baseline.n_requests}")
    print(f"  Prompt tokens:  ~{baseline.prompt_tokens}")
    print(f"  Output tokens:  {baseline.output_tokens}")
    
    print("\n" + "-"*80)
    print(f"{'Metric':<35} {'Baseline':>18} {'Disaggregated':>18}")
    print("-"*80)
    
    def fmt(val, unit=""):
        if isinstance(val, float):
            return f"{val:.1f}{unit}"
        return f"{val}{unit}"
    
    print(f"{'Total Time':<35} {fmt(baseline.total_time_sec, 's'):>18} {fmt(disagg.total_time_sec, 's'):>18}")
    print(f"{'Success/Fail':<35} {f'{baseline.success_count}/{baseline.failure_count}':>18} {f'{disagg.success_count}/{disagg.failure_count}':>18}")
    print()
    
    # Latency section
    print("LATENCY:")
    print(f"{'  Avg Latency':<35} {fmt(baseline.avg_latency_ms, 'ms'):>18} {fmt(disagg.avg_latency_ms, 'ms'):>18}")
    print(f"{'  P50 Latency':<35} {fmt(baseline.p50_latency_ms, 'ms'):>18} {fmt(disagg.p50_latency_ms, 'ms'):>18}")
    print(f"{'  P90 Latency':<35} {fmt(baseline.p90_latency_ms, 'ms'):>18} {fmt(disagg.p90_latency_ms, 'ms'):>18}")
    print(f"{'  P99 Latency':<35} {fmt(baseline.p99_latency_ms, 'ms'):>18} {fmt(disagg.p99_latency_ms, 'ms'):>18}")
    print()
    
    # Phase breakdown
    print("PHASE BREAKDOWN (avg):")
    print(f"{'  Prefill time':<35} {fmt(baseline.avg_prefill_ms, 'ms'):>18} {fmt(disagg.avg_prefill_ms, 'ms'):>18}")
    if disagg.avg_transfer_ms > 0:
        print(f"{'  Transfer time (save+restore)':<35} {'-':>18} {fmt(disagg.avg_transfer_ms, 'ms'):>18}")
    print(f"{'  Decode time':<35} {fmt(baseline.avg_decode_ms, 'ms'):>18} {fmt(disagg.avg_decode_ms, 'ms'):>18}")
    print()
    
    # TPS metrics
    print("TOKENS/SEC (server-reported):")
    print(f"{'  Prompt processing TPS':<35} {fmt(baseline.avg_prompt_tps, ' t/s'):>18} {fmt(disagg.avg_prompt_tps, ' t/s'):>18}")
    print(f"{'    (baseline: single server, disagg: prefill server)':<70}")
    print(f"{'  Decode TPS':<35} {fmt(baseline.avg_decode_tps, ' t/s'):>18} {fmt(disagg.avg_decode_tps, ' t/s'):>18}")
    print()
    
    # Throughput
    print("THROUGHPUT:")
    print(f"{'  Requests/sec':<35} {fmt(baseline.requests_per_sec):>18} {fmt(disagg.requests_per_sec):>18}")
    print(f"{'  Output tokens/sec':<35} {fmt(baseline.tokens_per_sec):>18} {fmt(disagg.tokens_per_sec):>18}")
    print()
    
    # Cache stats
    print("CACHE (disagg only):")
    print(f"{'  Avg Cache Reuse':<35} {'-':>18} {fmt(disagg.avg_cache_n, ' tokens'):>18}")
    print(f"{'  Cache Hit Rate':<35} {'-':>18} {fmt(disagg.cache_hit_rate * 100, '%'):>18}")
    
    print("\n" + "-"*80)
    
    # Summary
    if baseline.avg_latency_ms > 0 and disagg.avg_latency_ms > 0:
        latency_ratio = baseline.avg_latency_ms / disagg.avg_latency_ms
        throughput_ratio = disagg.requests_per_sec / baseline.requests_per_sec if baseline.requests_per_sec > 0 else 0
        
        print("\nSUMMARY:")
        if latency_ratio > 1:
            print(f"  ✓ Disaggregated is {latency_ratio:.2f}x FASTER (avg latency)")
        else:
            print(f"  ✗ Disaggregated is {1/latency_ratio:.2f}x SLOWER (avg latency)")
        
        if throughput_ratio > 1:
            print(f"  ✓ Disaggregated has {throughput_ratio:.2f}x HIGHER throughput")
        else:
            print(f"  ✗ Disaggregated has {1/throughput_ratio:.2f}x LOWER throughput")
        
        if disagg.cache_hit_rate > 0.9:
            print(f"  ✓ Cache hit rate: {disagg.cache_hit_rate*100:.0f}% - KV cache reuse working!")
        elif disagg.cache_hit_rate > 0:
            print(f"  ~ Cache hit rate: {disagg.cache_hit_rate*100:.0f}% - Partial cache reuse")
        else:
            print(f"  ✗ Cache hit rate: 0% - No cache reuse")
    
    print("="*80)


async def clear_all_slots():
    """Clear all slots on both servers and orchestrator sessions"""
    async with aiohttp.ClientSession() as session:
        # Clear orchestrator sessions
        try:
            async with session.post(f"{ORCHESTRATOR_URL}/clear") as resp:
                if resp.status == 200:
                    print("  ✓ Cleared orchestrator sessions")
        except:
            pass
        
        # Erase slots on prefill server
        for slot_id in range(8):  # Try up to 8 slots
            try:
                async with session.post(f"{PREFILL_URL}/slots/{slot_id}?action=erase") as resp:
                    pass
            except:
                break
        print("  ✓ Cleared prefill server slots")
        
        # Erase slots on decode server
        for slot_id in range(8):
            try:
                async with session.post(f"{DECODE_URL}/slots/{slot_id}?action=erase") as resp:
                    pass
            except:
                break
        print("  ✓ Cleared decode server slots")


async def main():
    parser = argparse.ArgumentParser(description="Concurrent Benchmark for Disaggregated Prefill")
    parser.add_argument("--concurrency", "-c", type=int, default=5,
                        help="Number of concurrent requests (default: 5)")
    parser.add_argument("--requests", "-n", type=int, default=20,
                        help="Total number of requests (default: 20)")
    parser.add_argument("--prompt-tokens", "-p", type=int, default=500,
                        help="Approximate prompt length in tokens (default: 500)")
    parser.add_argument("--output-tokens", "-o", type=int, default=20,
                        help="Number of tokens to generate (default: 20)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline test")
    parser.add_argument("--disagg-only", action="store_true",
                        help="Only run disaggregated test")
    parser.add_argument("--no-clear", action="store_true",
                        help="Skip clearing slots before benchmark")
    parser.add_argument("--output", "-O", type=str, default=None,
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           CONCURRENT BENCHMARK: Disaggregated vs Baseline                    ║
║                                                                              ║
║  Testing scenario: High ISL (long prompts) + Low OSL (short outputs)         ║
║  This is where disaggregated prefill should provide benefits.                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Clear slots before starting
    if not args.no_clear:
        print("Clearing slots and sessions...")
        await clear_all_slots()
        print()
    
    # Generate test prompt
    print(f"Generating prompt with ~{args.prompt_tokens} tokens...")
    prompt = generate_long_prompt(args.prompt_tokens)
    print(f"  Prompt length: {len(prompt)} chars (~{len(prompt.split())} words)")
    
    # Check servers are up
    print("\nChecking servers...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASELINE_URL}/health") as resp:
                if resp.status == 200:
                    print(f"  ✓ Baseline server (8080) is up")
                else:
                    print(f"  ✗ Baseline server returned {resp.status}")
                    return
        except Exception as e:
            print(f"  ✗ Cannot connect to baseline server: {e}")
            return
        
        if not args.baseline_only:
            try:
                async with session.get(f"{ORCHESTRATOR_URL}/health") as resp:
                    if resp.status == 200:
                        print(f"  ✓ Orchestrator (9000) is up")
                    else:
                        print(f"  ✗ Orchestrator returned {resp.status}")
                        return
            except Exception as e:
                print(f"  ✗ Cannot connect to orchestrator: {e}")
                return
    
    baseline_result = None
    disagg_result = None
    
    # Run baseline
    if not args.disagg_only:
        print(f"\n{'='*60}")
        print("Running BASELINE benchmark...")
        print(f"{'='*60}")
        baseline_result = await run_benchmark(
            mode="baseline",
            concurrency=args.concurrency,
            n_requests=args.requests,
            prompt=prompt,
            output_tokens=args.output_tokens,
            prompt_tokens=args.prompt_tokens,
        )
        print(f"  Completed: {baseline_result.success_count}/{args.requests} successful")
        print(f"  Total time: {baseline_result.total_time_sec:.1f}s")
        print(f"  Avg latency: {baseline_result.avg_latency_ms:.0f}ms")
    
    # Run disaggregated
    if not args.baseline_only:
        print(f"\n{'='*60}")
        print("Running DISAGGREGATED benchmark...")
        print(f"{'='*60}")
        disagg_result = await run_benchmark(
            mode="disagg",
            concurrency=args.concurrency,
            n_requests=args.requests,
            prompt=prompt,
            output_tokens=args.output_tokens,
            prompt_tokens=args.prompt_tokens,
        )
        print(f"  Completed: {disagg_result.success_count}/{args.requests} successful")
        print(f"  Total time: {disagg_result.total_time_sec:.1f}s")
        print(f"  Avg latency: {disagg_result.avg_latency_ms:.0f}ms")
        print(f"  Cache hit rate: {disagg_result.cache_hit_rate*100:.0f}%")
    
    # Print comparison
    if baseline_result and disagg_result:
        print_results(baseline_result, disagg_result)
    elif baseline_result:
        print(f"\n\nBaseline only results:")
        print(f"  Avg latency: {baseline_result.avg_latency_ms:.0f}ms")
        print(f"  Throughput: {baseline_result.requests_per_sec:.1f} req/s")
    elif disagg_result:
        print(f"\n\nDisaggregated only results:")
        print(f"  Avg latency: {disagg_result.avg_latency_ms:.0f}ms")
        print(f"  Cache hit rate: {disagg_result.cache_hit_rate*100:.0f}%")
        print(f"  Throughput: {disagg_result.requests_per_sec:.1f} req/s")
    
    # Save JSON results
    if args.output:
        from dataclasses import asdict
        from datetime import datetime
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "concurrency": args.concurrency,
                "requests": args.requests,
                "prompt_tokens": args.prompt_tokens,
                "output_tokens": args.output_tokens,
            },
            "baseline": asdict(baseline_result) if baseline_result else None,
            "disagg": asdict(disagg_result) if disagg_result else None,
        }
        
        # Add summary if both present
        if baseline_result and disagg_result:
            speedup = baseline_result.avg_latency_ms / disagg_result.avg_latency_ms if disagg_result.avg_latency_ms > 0 else 0
            throughput_gain = disagg_result.requests_per_sec / baseline_result.requests_per_sec if baseline_result.requests_per_sec > 0 else 0
            results["summary"] = {
                "latency_speedup": speedup,
                "throughput_gain": throughput_gain,
                "disagg_wins": speedup > 1,
            }
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
