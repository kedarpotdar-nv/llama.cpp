#!/usr/bin/env python3
"""
Concurrent load test for disaggregated vs baseline.

This tests the REAL hypothesis: Does disaggregated prefill help under
high concurrency with long prompts and short outputs?

Usage:
    python3 concurrent_test.py --concurrency 10 --requests 50
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional
import aiohttp


@dataclass 
class RequestResult:
    success: bool
    total_ms: float
    prefill_ms: float = 0
    decode_ms: float = 0
    ttft_ms: float = 0  # Time to first token
    tokens_generated: int = 0
    error: Optional[str] = None


def generate_prompt(tokens: int = 2000) -> str:
    """Generate a prompt of approximately `tokens` length"""
    base = "The quick brown fox jumps over the lazy dog. " * 100
    chars = tokens * 4
    return (base * (chars // len(base) + 1))[:chars]


async def make_disagg_request(
    session: aiohttp.ClientSession,
    orchestrator_url: str,
    prompt: str,
    n_predict: int
) -> RequestResult:
    """Make a request through the disaggregated orchestrator"""
    start = time.perf_counter()
    
    try:
        async with session.post(
            f"{orchestrator_url}/completion",
            json={
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": 0.0,
                "disagg": True,
            },
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                return RequestResult(
                    success=False,
                    total_ms=(time.perf_counter() - start) * 1000,
                    error=error
                )
            
            data = await resp.json()
            total_ms = (time.perf_counter() - start) * 1000
            
            metrics = data.get("disagg_metrics", {})
            timings = data.get("timings", {})
            
            return RequestResult(
                success=True,
                total_ms=total_ms,
                prefill_ms=metrics.get("prefill_ms", 0),
                decode_ms=metrics.get("decode_ms", 0),
                ttft_ms=metrics.get("prefill_ms", 0) + metrics.get("save_ms", 0) + metrics.get("restore_ms", 0),
                tokens_generated=timings.get("predicted_n", 0),
            )
    except Exception as e:
        return RequestResult(
            success=False,
            total_ms=(time.perf_counter() - start) * 1000,
            error=str(e)
        )


async def make_baseline_request(
    session: aiohttp.ClientSession,
    baseline_url: str,
    prompt: str,
    n_predict: int
) -> RequestResult:
    """Make a request to single baseline server"""
    start = time.perf_counter()
    
    try:
        async with session.post(
            f"{baseline_url}/completion",
            json={
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": 0.0,
                "cache_prompt": False,  # Don't cache to simulate fresh requests
            },
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                return RequestResult(
                    success=False,
                    total_ms=(time.perf_counter() - start) * 1000,
                    error=error
                )
            
            data = await resp.json()
            total_ms = (time.perf_counter() - start) * 1000
            timings = data.get("timings", {})
            
            return RequestResult(
                success=True,
                total_ms=total_ms,
                prefill_ms=timings.get("prompt_ms", 0),
                decode_ms=timings.get("predicted_ms", 0),
                ttft_ms=timings.get("prompt_ms", 0),
                tokens_generated=timings.get("predicted_n", 0),
            )
    except Exception as e:
        return RequestResult(
            success=False,
            total_ms=(time.perf_counter() - start) * 1000,
            error=str(e)
        )


async def run_concurrent_test(
    make_request_fn,
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    n_predict: int,
    num_requests: int,
    concurrency: int
) -> List[RequestResult]:
    """Run concurrent requests with controlled concurrency"""
    
    semaphore = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []
    
    async def bounded_request(idx: int):
        async with semaphore:
            result = await make_request_fn(session, url, prompt, n_predict)
            return result
    
    tasks = [bounded_request(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    return list(results)


def print_stats(name: str, results: List[RequestResult]):
    """Print statistics for a set of results"""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    print(f"Requests: {len(results)} total, {len(successful)} success, {len(failed)} failed")
    
    if not successful:
        print("No successful requests!")
        return
    
    total_times = [r.total_ms for r in successful]
    ttft_times = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    
    print(f"\nLatency (total request):")
    print(f"  Mean:   {statistics.mean(total_times):8.1f} ms")
    print(f"  Median: {statistics.median(total_times):8.1f} ms")
    print(f"  P95:    {sorted(total_times)[int(len(total_times)*0.95)]:8.1f} ms")
    print(f"  P99:    {sorted(total_times)[int(len(total_times)*0.99)]:8.1f} ms")
    print(f"  Min:    {min(total_times):8.1f} ms")
    print(f"  Max:    {max(total_times):8.1f} ms")
    
    if ttft_times:
        print(f"\nTime to First Token (TTFT):")
        print(f"  Mean:   {statistics.mean(ttft_times):8.1f} ms")
        print(f"  Median: {statistics.median(ttft_times):8.1f} ms")
        print(f"  P95:    {sorted(ttft_times)[int(len(ttft_times)*0.95)]:8.1f} ms")
    
    tokens = sum(r.tokens_generated for r in successful)
    total_decode_time = sum(r.decode_ms for r in successful) / 1000
    if total_decode_time > 0:
        print(f"\nThroughput:")
        print(f"  Total tokens:     {tokens}")
        print(f"  Decode tok/s:     {tokens / total_decode_time:.1f}")
    
    if failed:
        print(f"\nErrors ({len(failed)}):")
        for r in failed[:3]:
            print(f"  - {r.error[:100]}")


async def main():
    parser = argparse.ArgumentParser(description="Concurrent load test")
    parser.add_argument("--orchestrator-url", type=str, default="http://localhost:9000",
                        help="Orchestrator URL (for disagg)")
    parser.add_argument("--baseline-url", type=str, default="http://localhost:8080",
                        help="Baseline server URL")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent requests")
    parser.add_argument("--requests", type=int, default=30,
                        help="Total number of requests")
    parser.add_argument("--prompt-tokens", type=int, default=2000,
                        help="Approximate prompt length in tokens")
    parser.add_argument("--output-tokens", type=int, default=50,
                        help="Number of tokens to generate")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline test")
    parser.add_argument("--skip-disagg", action="store_true",
                        help="Skip disaggregated test")
    
    args = parser.parse_args()
    
    prompt = generate_prompt(args.prompt_tokens)
    
    print("="*60)
    print("CONCURRENT LOAD TEST: Disaggregated vs Baseline")
    print("="*60)
    print(f"Concurrency:    {args.concurrency}")
    print(f"Total requests: {args.requests}")
    print(f"Prompt tokens:  ~{args.prompt_tokens}")
    print(f"Output tokens:  {args.output_tokens}")
    
    async with aiohttp.ClientSession() as session:
        
        # Test baseline
        if not args.skip_baseline:
            print(f"\n{'~'*60}")
            print("Running BASELINE test (single server)...")
            print(f"{'~'*60}")
            
            start = time.perf_counter()
            baseline_results = await run_concurrent_test(
                make_baseline_request,
                session,
                args.baseline_url,
                prompt,
                args.output_tokens,
                args.requests,
                args.concurrency
            )
            baseline_duration = time.perf_counter() - start
            
            print_stats("BASELINE RESULTS", baseline_results)
            print(f"\nTotal test duration: {baseline_duration:.1f}s")
            print(f"Throughput: {len([r for r in baseline_results if r.success]) / baseline_duration:.2f} req/s")
        
        # Test disaggregated
        if not args.skip_disagg:
            print(f"\n{'~'*60}")
            print("Running DISAGGREGATED test (orchestrator)...")
            print(f"{'~'*60}")
            
            start = time.perf_counter()
            disagg_results = await run_concurrent_test(
                make_disagg_request,
                session,
                args.orchestrator_url,
                prompt,
                args.output_tokens,
                args.requests,
                args.concurrency
            )
            disagg_duration = time.perf_counter() - start
            
            print_stats("DISAGGREGATED RESULTS", disagg_results)
            print(f"\nTotal test duration: {disagg_duration:.1f}s")
            print(f"Throughput: {len([r for r in disagg_results if r.success]) / disagg_duration:.2f} req/s")
        
        # Comparison
        if not args.skip_baseline and not args.skip_disagg:
            print(f"\n{'='*60}")
            print("COMPARISON")
            print(f"{'='*60}")
            
            baseline_success = [r for r in baseline_results if r.success]
            disagg_success = [r for r in disagg_results if r.success]
            
            if baseline_success and disagg_success:
                baseline_median = statistics.median([r.total_ms for r in baseline_success])
                disagg_median = statistics.median([r.total_ms for r in disagg_success])
                
                print(f"Median latency - Baseline:      {baseline_median:8.1f} ms")
                print(f"Median latency - Disaggregated: {disagg_median:8.1f} ms")
                print(f"Difference:                     {disagg_median - baseline_median:+8.1f} ms")
                
                if disagg_median < baseline_median:
                    print(f"\n✓ Disaggregated is {baseline_median/disagg_median:.2f}x FASTER")
                else:
                    print(f"\n✗ Disaggregated is {disagg_median/baseline_median:.2f}x SLOWER")
                
                print(f"\nThroughput - Baseline:      {len(baseline_success) / baseline_duration:.2f} req/s")
                print(f"Throughput - Disaggregated: {len(disagg_success) / disagg_duration:.2f} req/s")


if __name__ == "__main__":
    asyncio.run(main())
