#!/usr/bin/env python3
"""
Sweep Benchmark - Run disaggregated vs baseline across multiple configurations
Saves JSON results for analysis and plotting.

Usage:
    python3 disagg/sweep_benchmark.py --output results.json
    python3 disagg/sweep_benchmark.py --quick  # Fast sweep with fewer configs
"""

import asyncio
import aiohttp
import argparse
import json
import time
import statistics
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

# URLs
ORCHESTRATOR_URL = "http://localhost:9000"
BASELINE_URL = "http://localhost:8080"
PREFILL_URL = "http://localhost:8080"
DECODE_URL = "http://localhost:8081"

# Sweep configurations
DEFAULT_ISL_OSL_CONFIGS = [
    # (input_seq_len, output_seq_len)
    (128, 32),
    (128, 128),
    (256, 32),
    (256, 128),
    (512, 32),
    (512, 128),
    (512, 256),
    (1024, 32),
    (1024, 128),
    (1024, 256),
    (2048, 32),
    (2048, 128),
]

QUICK_ISL_OSL_CONFIGS = [
    (128, 32),
    (512, 32),
    (512, 128),
    (1024, 32),
    (1024, 128),
]

DEFAULT_BATCH_SIZES = [1, 2, 4]
QUICK_BATCH_SIZES = [2, 4]


@dataclass
class RunResult:
    """Result of a single benchmark run"""
    mode: str  # "baseline" or "disagg"
    isl: int
    osl: int
    batch_size: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    avg_prefill_ms: float
    avg_transfer_ms: float
    avg_decode_ms: float
    prefill_tps: float
    decode_tps: float
    requests_per_sec: float
    tokens_per_sec: float
    cache_hit_rate: float = 0.0
    avg_cache_n: float = 0.0
    error: str = ""


@dataclass 
class SweepResult:
    """Complete sweep results"""
    timestamp: str
    server_info: Dict[str, Any]
    configs_tested: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def generate_prompt(target_tokens: int) -> str:
    """Generate a prompt with approximately target_tokens tokens"""
    base = "Analyze the following comprehensive dataset and provide insights. "
    words_needed = int(target_tokens * 0.75)  # ~0.75 words per token
    
    topics = [
        "market trends show increasing volatility across sectors",
        "user engagement metrics indicate strong growth patterns",
        "performance benchmarks reveal optimization opportunities", 
        "data analysis suggests emerging patterns in behavior",
        "system metrics demonstrate improved efficiency gains",
    ]
    
    content = []
    while len(content) < words_needed:
        for topic in topics:
            content.append(topic)
            if len(content) >= words_needed:
                break
    
    return base + " ".join(content[:words_needed])


async def clear_all_slots():
    """Clear slots before benchmark"""
    async with aiohttp.ClientSession() as session:
        # Clear orchestrator
        try:
            async with session.post(f"{ORCHESTRATOR_URL}/clear") as resp:
                pass
        except:
            pass
        
        # Clear server slots
        for url in [PREFILL_URL, DECODE_URL]:
            for slot_id in range(8):
                try:
                    async with session.post(f"{url}/slots/{slot_id}?action=erase") as resp:
                        pass
                except:
                    break


async def get_server_info() -> Dict[str, Any]:
    """Get server configuration info"""
    info = {}
    async with aiohttp.ClientSession() as session:
        for name, url in [("prefill", PREFILL_URL), ("decode", DECODE_URL)]:
            try:
                async with session.get(f"{url}/props") as resp:
                    if resp.status == 200:
                        props = await resp.json()
                        info[name] = {
                            "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
                            "total_slots": props.get("total_slots"),
                            "model": props.get("model_alias", "unknown"),
                        }
            except Exception as e:
                info[name] = {"error": str(e)}
        
        # Check orchestrator
        try:
            async with session.get(f"{ORCHESTRATOR_URL}/health") as resp:
                if resp.status == 200:
                    info["orchestrator"] = {"status": "healthy"}
                else:
                    info["orchestrator"] = {"error": f"HTTP {resp.status}"}
        except Exception as e:
            info["orchestrator"] = {"error": str(e)}
    
    return info


async def run_baseline_batch(
    prompt: str,
    batch_size: int,
    output_tokens: int,
    n_requests: int,
) -> List[Dict[str, Any]]:
    """Run baseline requests"""
    results = []
    
    async def make_request(req_id: int) -> Dict[str, Any]:
        t_start = time.perf_counter()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post(
                    f"{BASELINE_URL}/completion",
                    json={
                        "prompt": prompt,
                        "n_predict": output_tokens,
                        "temperature": 0.7,
                        "cache_prompt": True,
                    }
                ) as resp:
                    if resp.status != 200:
                        return {"error": f"HTTP {resp.status}", "request_id": req_id}
                    result = await resp.json()
                    
            t_end = time.perf_counter()
            timings = result.get("timings", {})
            
            return {
                "request_id": req_id,
                "success": True,
                "total_time_ms": (t_end - t_start) * 1000,
                "prefill_ms": timings.get("prompt_ms", 0),
                "decode_ms": timings.get("predicted_ms", 0),
                "transfer_ms": 0,
                "tokens_generated": timings.get("predicted_n", 0),
                "prompt_n": timings.get("prompt_n", 0),
                "prefill_tps": timings.get("prompt_per_second", 0),
                "decode_tps": timings.get("predicted_per_second", 0),
                "cache_n": 0,
            }
        except Exception as e:
            return {"error": str(e), "request_id": req_id}
    
    # Run in batches
    for batch_start in range(0, n_requests, batch_size):
        batch_end = min(batch_start + batch_size, n_requests)
        tasks = [make_request(i) for i in range(batch_start, batch_end)]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        await clear_all_slots()  # Clear between batches
    
    return results


async def run_disagg_batch(
    prompt: str,
    batch_size: int,
    output_tokens: int,
    n_requests: int,
) -> List[Dict[str, Any]]:
    """Run disaggregated requests through orchestrator"""
    results = []
    
    async def make_request(req_id: int) -> Dict[str, Any]:
        t_start = time.perf_counter()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                # Prefill
                async with session.post(
                    f"{ORCHESTRATOR_URL}/prefill",
                    json={"prompt": prompt}
                ) as resp:
                    if resp.status != 200:
                        return {"error": f"Prefill failed: {resp.status}", "request_id": req_id}
                    prefill_result = await resp.json()
                
                session_id = prefill_result.get("session_id")
                if not session_id:
                    return {"error": "No session_id", "request_id": req_id}
                
                # Generate
                async with session.post(
                    f"{ORCHESTRATOR_URL}/generate",
                    json={
                        "session_id": session_id,
                        "n_predict": output_tokens,
                        "temperature": 0.7,
                    }
                ) as resp:
                    if resp.status != 200:
                        return {"error": f"Generate failed: {resp.status}", "request_id": req_id}
                    gen_result = await resp.json()
                
            t_end = time.perf_counter()
            metrics = gen_result.get("disagg_metrics", {})
            timings = gen_result.get("timings", {})
            
            return {
                "request_id": req_id,
                "success": True,
                "total_time_ms": (t_end - t_start) * 1000,
                "prefill_ms": metrics.get("prefill_time_ms", 0),
                "transfer_ms": metrics.get("transfer_time_ms", 0),
                "decode_ms": metrics.get("decode_time_ms", 0),
                "tokens_generated": metrics.get("n_generated_tokens", 0),
                "prompt_n": metrics.get("n_prompt_tokens", 0),
                "prefill_tps": metrics.get("prefill_tps", 0),
                "decode_tps": metrics.get("decode_tps", 0),
                "cache_n": metrics.get("cache_n", 0),
            }
        except Exception as e:
            return {"error": str(e), "request_id": req_id}
    
    # Run in batches
    for batch_start in range(0, n_requests, batch_size):
        batch_end = min(batch_start + batch_size, n_requests)
        tasks = [make_request(i) for i in range(batch_start, batch_end)]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        # Clear orchestrator sessions
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{ORCHESTRATOR_URL}/clear") as resp:
                    pass
        except:
            pass
    
    return results


def aggregate_results(
    results: List[Dict[str, Any]],
    mode: str,
    isl: int,
    osl: int,
    batch_size: int,
    total_time: float,
) -> RunResult:
    """Aggregate individual request results"""
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    # Print errors for debugging
    if failed and not successful:
        print(f"    ⚠ All {len(failed)} requests failed!")
        errors = set(r.get("error", "unknown") for r in failed[:3])
        for err in errors:
            print(f"      Error: {err[:100]}")
    
    if not successful:
        return RunResult(
            mode=mode,
            isl=isl,
            osl=osl,
            batch_size=batch_size,
            total_requests=len(results),
            successful_requests=0,
            failed_requests=len(failed),
            total_time_sec=total_time,
            avg_latency_ms=0,
            p50_latency_ms=0,
            p90_latency_ms=0,
            p99_latency_ms=0,
            avg_prefill_ms=0,
            avg_transfer_ms=0,
            avg_decode_ms=0,
            prefill_tps=0,
            decode_tps=0,
            requests_per_sec=0,
            tokens_per_sec=0,
            error=failed[0].get("error", "Unknown") if failed else "",
        )
    
    latencies = [r["total_time_ms"] for r in successful]
    latencies_sorted = sorted(latencies)
    
    total_tokens = sum(r.get("tokens_generated", 0) for r in successful)
    cache_hits = [r for r in successful if r.get("cache_n", 0) > 0]
    
    return RunResult(
        mode=mode,
        isl=isl,
        osl=osl,
        batch_size=batch_size,
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_time_sec=total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=latencies_sorted[len(latencies_sorted)//2],
        p90_latency_ms=latencies_sorted[int(len(latencies_sorted)*0.9)] if len(latencies_sorted) >= 10 else latencies_sorted[-1],
        p99_latency_ms=latencies_sorted[int(len(latencies_sorted)*0.99)] if len(latencies_sorted) >= 100 else latencies_sorted[-1],
        avg_prefill_ms=statistics.mean([r.get("prefill_ms", 0) for r in successful]),
        avg_transfer_ms=statistics.mean([r.get("transfer_ms", 0) for r in successful]),
        avg_decode_ms=statistics.mean([r.get("decode_ms", 0) for r in successful]),
        prefill_tps=statistics.mean([r.get("prefill_tps", 0) for r in successful if r.get("prefill_tps", 0) > 0]) if any(r.get("prefill_tps", 0) > 0 for r in successful) else 0,
        decode_tps=statistics.mean([r.get("decode_tps", 0) for r in successful if r.get("decode_tps", 0) > 0]) if any(r.get("decode_tps", 0) > 0 for r in successful) else 0,
        requests_per_sec=len(successful) / total_time if total_time > 0 else 0,
        tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
        cache_hit_rate=len(cache_hits) / len(successful) * 100 if successful else 0,
        avg_cache_n=statistics.mean([r.get("cache_n", 0) for r in successful]) if successful else 0,
    )


async def run_single_config(
    isl: int,
    osl: int,
    batch_size: int,
    n_requests: int,
    run_baseline: bool = True,
    run_disagg: bool = True,
) -> Dict[str, RunResult]:
    """Run benchmark for a single configuration"""
    results = {}
    prompt = generate_prompt(isl)
    
    # Baseline
    if run_baseline:
        await clear_all_slots()
        t_start = time.perf_counter()
        baseline_results = await run_baseline_batch(prompt, batch_size, osl, n_requests)
        t_end = time.perf_counter()
        results["baseline"] = aggregate_results(
            baseline_results, "baseline", isl, osl, batch_size, t_end - t_start
        )
    
    # Disaggregated
    if run_disagg:
        await clear_all_slots()
        t_start = time.perf_counter()
        disagg_results = await run_disagg_batch(prompt, batch_size, osl, n_requests)
        t_end = time.perf_counter()
        results["disagg"] = aggregate_results(
            disagg_results, "disagg", isl, osl, batch_size, t_end - t_start
        )
    
    return results


def print_progress(current: int, total: int, config: str, result: Optional[Dict[str, RunResult]]):
    """Print progress update"""
    print(f"\n[{current}/{total}] {config}")
    
    if result:
        if "baseline" in result and "disagg" in result:
            b = result["baseline"]
            d = result["disagg"]
            speedup = b.avg_latency_ms / d.avg_latency_ms if d.avg_latency_ms > 0 else 0
            throughput_gain = d.requests_per_sec / b.requests_per_sec if b.requests_per_sec > 0 else 0
            
            status = "✓ DISAGG WINS" if speedup > 1 else "✗ BASELINE WINS"
            print(f"  Baseline: {b.avg_latency_ms:.0f}ms, {b.requests_per_sec:.1f} req/s")
            print(f"  Disagg:   {d.avg_latency_ms:.0f}ms, {d.requests_per_sec:.1f} req/s, {d.cache_hit_rate:.0f}% cache")
            print(f"  → {status} ({speedup:.2f}x latency, {throughput_gain:.2f}x throughput)")
        elif "baseline" in result:
            b = result["baseline"]
            print(f"  Baseline: {b.avg_latency_ms:.0f}ms, {b.requests_per_sec:.1f} req/s")
        elif "disagg" in result:
            d = result["disagg"]
            print(f"  Disagg: {d.avg_latency_ms:.0f}ms, {d.requests_per_sec:.1f} req/s")


def compute_summary(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics"""
    paired = [(r["baseline"], r["disagg"]) for r in all_results 
              if "baseline" in r and "disagg" in r]
    
    if not paired:
        return {}
    
    speedups = []
    throughput_gains = []
    best_config = None
    best_speedup = 0
    
    for b, d in paired:
        if d["avg_latency_ms"] > 0 and b["avg_latency_ms"] > 0:
            speedup = b["avg_latency_ms"] / d["avg_latency_ms"]
            speedups.append(speedup)
            
            if speedup > best_speedup:
                best_speedup = speedup
                best_config = {
                    "isl": d["isl"],
                    "osl": d["osl"],
                    "batch_size": d["batch_size"],
                    "speedup": speedup,
                    "baseline_latency_ms": b["avg_latency_ms"],
                    "disagg_latency_ms": d["avg_latency_ms"],
                }
        
        if b["requests_per_sec"] > 0:
            throughput_gains.append(d["requests_per_sec"] / b["requests_per_sec"])
    
    disagg_wins = sum(1 for s in speedups if s > 1)
    
    return {
        "total_configs": len(paired),
        "disagg_wins": disagg_wins,
        "baseline_wins": len(paired) - disagg_wins,
        "avg_speedup": statistics.mean(speedups) if speedups else 0,
        "max_speedup": max(speedups) if speedups else 0,
        "min_speedup": min(speedups) if speedups else 0,
        "avg_throughput_gain": statistics.mean(throughput_gains) if throughput_gains else 0,
        "best_config": best_config,
    }


def print_final_summary(summary: Dict[str, Any], all_results: List[Dict[str, Any]]):
    """Print final summary"""
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    
    if not summary:
        print("No paired results to summarize")
        return
    
    print(f"\nConfigurations tested: {summary['total_configs']}")
    print(f"Disaggregated wins:    {summary['disagg_wins']}")
    print(f"Baseline wins:         {summary['baseline_wins']}")
    
    print(f"\nSpeedup (disagg vs baseline):")
    print(f"  Average: {summary['avg_speedup']:.2f}x")
    print(f"  Best:    {summary['max_speedup']:.2f}x")
    print(f"  Worst:   {summary['min_speedup']:.2f}x")
    
    print(f"\nAvg throughput gain: {summary['avg_throughput_gain']:.2f}x")
    
    if summary.get("best_config"):
        bc = summary["best_config"]
        print(f"\n🏆 BEST CONFIG FOR DISAGGREGATED:")
        print(f"   ISL={bc['isl']}, OSL={bc['osl']}, batch_size={bc['batch_size']}")
        print(f"   Speedup: {bc['speedup']:.2f}x")
        print(f"   Latency: {bc['baseline_latency_ms']:.0f}ms → {bc['disagg_latency_ms']:.0f}ms")
    
    # Print table of all results
    print("\n" + "-"*80)
    print("ALL RESULTS (sorted by speedup):")
    print("-"*80)
    print(f"{'ISL':>6} {'OSL':>6} {'BS':>4} {'Base(ms)':>10} {'Disagg(ms)':>12} {'Speedup':>8} {'Cache%':>7}")
    print("-"*80)
    
    # Sort by speedup
    paired_results = []
    for r in all_results:
        if "baseline" in r and "disagg" in r:
            b, d = r["baseline"], r["disagg"]
            speedup = b["avg_latency_ms"] / d["avg_latency_ms"] if d["avg_latency_ms"] > 0 else 0
            paired_results.append((b, d, speedup))
    
    for b, d, speedup in sorted(paired_results, key=lambda x: -x[2]):
        marker = "✓" if speedup > 1 else " "
        print(f"{marker} {b['isl']:>5} {b['osl']:>6} {b['batch_size']:>4} {b['avg_latency_ms']:>10.0f} {d['avg_latency_ms']:>12.0f} {speedup:>7.2f}x {d['cache_hit_rate']:>6.0f}%")
    
    print("="*80)


async def main():
    parser = argparse.ArgumentParser(description="Sweep Benchmark")
    parser.add_argument("--output", "-o", type=str, default="sweep_results.json",
                        help="Output JSON file (default: sweep_results.json)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick sweep with fewer configurations")
    parser.add_argument("--requests", "-n", type=int, default=8,
                        help="Requests per config (default: 8)")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--disagg-only", action="store_true")
    parser.add_argument("--isl", type=int, nargs="+",
                        help="Custom ISL values (e.g., --isl 128 512 1024)")
    parser.add_argument("--osl", type=int, nargs="+",
                        help="Custom OSL values (e.g., --osl 32 128)")
    parser.add_argument("--batch-sizes", "-b", type=int, nargs="+",
                        help="Custom batch sizes (e.g., -b 2 4)")
    
    args = parser.parse_args()
    
    # Determine configurations
    if args.isl and args.osl:
        isl_osl_configs = [(i, o) for i in args.isl for o in args.osl]
    elif args.quick:
        isl_osl_configs = QUICK_ISL_OSL_CONFIGS
    else:
        isl_osl_configs = DEFAULT_ISL_OSL_CONFIGS
    
    batch_sizes = args.batch_sizes or (QUICK_BATCH_SIZES if args.quick else DEFAULT_BATCH_SIZES)
    
    total_configs = len(isl_osl_configs) * len(batch_sizes)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SWEEP BENCHMARK: ISL/OSL/Batch Size                       ║
║                                                                              ║
║  Finding optimal configurations for disaggregated vs baseline                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Configurations: {len(isl_osl_configs)} ISL/OSL × {len(batch_sizes)} batch sizes = {total_configs} total")
    print(f"Requests per config: {args.requests}")
    print(f"Output file: {args.output}")
    
    # Get server info
    print("\nChecking servers...")
    server_info = await get_server_info()
    for name, info in server_info.items():
        if "error" in info:
            print(f"  ✗ {name}: {info['error']}")
        elif name == "orchestrator":
            print(f"  ✓ {name}: {info.get('status', 'unknown')}")
        else:
            print(f"  ✓ {name}: n_ctx={info.get('n_ctx')}, slots={info.get('total_slots')}")
    
    # Warn if orchestrator not available
    if "orchestrator" in server_info and "error" in server_info["orchestrator"]:
        print("\n  ⚠ WARNING: Orchestrator not running! Disaggregated tests will fail.")
        print("    Start it with: python3 disagg/smart_orchestrator.py --port 9000")
        if not args.baseline_only:
            response = input("\n  Continue with baseline only? [Y/n]: ").strip().lower()
            if response != 'n':
                args.baseline_only = True
                args.disagg_only = False
    
    # Run sweep
    all_results = []
    current = 0
    
    for isl, osl in isl_osl_configs:
        for bs in batch_sizes:
            current += 1
            config_str = f"ISL={isl}, OSL={osl}, batch_size={bs}"
            
            try:
                result = await run_single_config(
                    isl=isl,
                    osl=osl,
                    batch_size=bs,
                    n_requests=args.requests,
                    run_baseline=not args.disagg_only,
                    run_disagg=not args.baseline_only,
                )
                
                # Convert to dict for JSON
                result_dict = {}
                for mode, run_result in result.items():
                    result_dict[mode] = asdict(run_result)
                
                all_results.append(result_dict)
                print_progress(current, total_configs, config_str, result)
                
            except Exception as e:
                print(f"\n[{current}/{total_configs}] {config_str}")
                print(f"  ✗ ERROR: {e}")
                all_results.append({
                    "config": {"isl": isl, "osl": osl, "batch_size": bs},
                    "error": str(e)
                })
    
    # Compute summary
    summary = compute_summary(all_results)
    
    # Save results
    sweep_result = SweepResult(
        timestamp=datetime.now().isoformat(),
        server_info=server_info,
        configs_tested=total_configs,
        results=all_results,
        summary=summary,
    )
    
    with open(args.output, "w") as f:
        json.dump(asdict(sweep_result), f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    # Print summary
    print_final_summary(summary, all_results)


if __name__ == "__main__":
    asyncio.run(main())
