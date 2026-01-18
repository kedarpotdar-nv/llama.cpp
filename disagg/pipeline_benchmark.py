#!/usr/bin/env python3
"""
Pipeline Benchmark for Disaggregated Prefill

Simulates realistic workload where batches of requests arrive continuously,
allowing prefill and decode to overlap:

  Time →
  Batch 1: [PREFILL]──[TRANSFER]──[DECODE]
  Batch 2:            [PREFILL]──[TRANSFER]──[DECODE]
  Batch 3:                       [PREFILL]──[TRANSFER]──[DECODE]

This pipelining is where disaggregated serving shines - prefill workers
stay busy processing new batches while decode workers generate tokens.

Usage:
    python3 disagg/pipeline_benchmark.py --waves 5 --batch-size 4 --prompt-tokens 500
"""

import argparse
import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import aiohttp

# Configuration
ORCHESTRATOR_URL = "http://localhost:9000"
BASELINE_URL = "http://localhost:8080"
PREFILL_URL = "http://localhost:8080"
DECODE_URL = "http://localhost:8081"


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    success: bool = False
    prefill_time_ms: float = 0
    transfer_time_ms: float = 0
    decode_time_ms: float = 0
    total_time_ms: float = 0
    cache_hit: bool = False
    tokens_generated: int = 0
    error: str = ""


@dataclass
class WaveMetrics:
    """Metrics for a wave/batch of requests"""
    wave_id: int
    batch_size: int
    prefill_start: float = 0
    prefill_end: float = 0
    transfer_start: float = 0
    transfer_end: float = 0
    decode_start: float = 0
    decode_end: float = 0
    requests: List[RequestMetrics] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Results from pipeline benchmark"""
    mode: str
    n_waves: int
    batch_size: int
    prompt_tokens: int
    output_tokens: int
    
    total_time_sec: float = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    # Aggregate throughput
    requests_per_sec: float = 0
    tokens_per_sec: float = 0
    
    # Phase utilization (for disagg)
    prefill_utilization: float = 0  # % time prefill workers were busy
    decode_utilization: float = 0   # % time decode workers were busy
    
    # Per-request metrics
    avg_latency_ms: float = 0
    avg_prefill_ms: float = 0
    avg_transfer_ms: float = 0
    avg_decode_ms: float = 0
    
    cache_hit_rate: float = 0
    
    waves: List[WaveMetrics] = field(default_factory=list)


def generate_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens length"""
    base = """Analyze this distributed computing scenario with multiple data centers, 
    variable workloads, cost optimization requirements, and strict SLAs. Consider 
    network bandwidth, data locality, failover capabilities, and geographic redundancy. 
    Some workloads are stateless while others require persistent state coordination. 
    Security, compliance, and environmental impact are also key factors. """
    
    words_needed = int(target_tokens * 0.75)
    base_words = base.split()
    result = []
    while len(result) < words_needed:
        result.extend(base_words)
    return " ".join(result[:words_needed])


class DisaggPipelineRunner:
    """
    Runs disaggregated pipeline with overlapping prefill/decode.
    
    Key insight: While one batch is decoding, we can prefill the next batch.
    """
    
    def __init__(self, config: Dict[str, str]):
        self.prefill_url = config.get("prefill_url", PREFILL_URL)
        self.decode_url = config.get("decode_url", DECODE_URL)
        self.kv_cache_dir = "/tmp/llama_kv_cache"
        self._http: Optional[aiohttp.ClientSession] = None
        
    async def get_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self._http
    
    async def close(self):
        if self._http:
            await self._http.close()
    
    async def tokenize(self, prompt: str) -> List[int]:
        """Tokenize prompt on prefill server"""
        http = await self.get_http()
        async with http.post(
            f"{self.prefill_url}/tokenize",
            json={"content": prompt, "add_special": True}
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Tokenize failed: {await resp.text()}")
            result = await resp.json()
            return result.get("tokens", [])
    
    async def prefill_batch(
        self, 
        requests: List[Dict[str, Any]], 
        slot_offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Prefill a batch of requests on the prefill server.
        Returns list of {tokens, slot, n_tokens, time_ms} for each request.
        
        Important: Process in chunks matching prefill server slots to avoid conflicts.
        """
        http = await self.get_http()
        n_prefill_slots = 2  # Prefill server slots (match -np in start_servers.sh)
        all_results = []
        
        async def prefill_one(req: Dict, slot: int) -> Dict[str, Any]:
            tokens = req["tokens"]
            
            t_start = time.perf_counter()
            try:
                async with http.post(
                    f"{self.prefill_url}/completion",
                    json={
                        "prompt": tokens,
                        "n_predict": 0,
                        "id_slot": slot,
                        "return_tokens": True,
                    }
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return {"error": f"HTTP {resp.status}: {error_text[:100]}", "slot": slot, "request_id": req["request_id"]}
                    result = await resp.json()
            except Exception as e:
                return {"error": str(e), "slot": slot, "request_id": req["request_id"]}
            
            t_end = time.perf_counter()
            
            # Include any tokens generated during prefill (n_predict=0 may still produce 1)
            final_tokens = result.get("tokens", tokens)
            if not final_tokens:
                final_tokens = tokens
            
            return {
                "request_id": req["request_id"],
                "tokens": final_tokens,
                "slot": slot,
                "n_tokens": len(final_tokens),
                "time_ms": (t_end - t_start) * 1000,
            }
        
        # Process in chunks of n_prefill_slots to avoid slot conflicts
        for chunk_start in range(0, len(requests), n_prefill_slots):
            chunk = requests[chunk_start:chunk_start + n_prefill_slots]
            tasks = [prefill_one(req, i % n_prefill_slots) for i, req in enumerate(chunk)]
            chunk_results = await asyncio.gather(*tasks)
            all_results.extend(chunk_results)
        
        return all_results
    
    async def save_batch(self, prefill_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Save KV caches from prefill server - must be done sequentially per slot"""
        http = await self.get_http()
        
        async def save_one(pr: Dict) -> Dict[str, Any]:
            if "error" in pr:
                return pr
            
            filename = f"kv_{pr['request_id']}.bin"
            t_start = time.perf_counter()
            
            try:
                async with http.post(
                    f"{self.prefill_url}/slots/{pr['slot']}?action=save",
                    json={"filename": filename}
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return {**pr, "error": f"Save failed: {error_text[:100]}"}
                    result = await resp.json()
            except Exception as e:
                return {**pr, "error": f"Save exception: {str(e)}"}
            
            t_end = time.perf_counter()
            return {
                **pr,
                "filename": filename,
                "n_saved": result.get("n_saved", 0),
                "save_time_ms": (t_end - t_start) * 1000,
            }
        
        # Save sequentially to avoid conflicts (saves must complete before slot reuse)
        results = []
        for pr in prefill_results:
            results.append(await save_one(pr))
        return results
    
    async def restore_batch(
        self, 
        saved_results: List[Dict[str, Any]], 
        slot_offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Restore KV caches to decode server"""
        http = await self.get_http()
        n_decode_slots = 2  # Decode server slots (must match prefill for KV compat)
        
        async def restore_one(idx: int, sr: Dict) -> Dict[str, Any]:
            if "error" in sr:
                return sr
            
            # Use unique slot per request within batch
            decode_slot = (slot_offset + idx) % n_decode_slots
            t_start = time.perf_counter()
            
            try:
                async with http.post(
                    f"{self.decode_url}/slots/{decode_slot}?action=restore",
                    json={"filename": sr["filename"]}
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return {**sr, "error": f"Restore failed: {error_text[:100]}"}
                    await resp.json()
            except Exception as e:
                return {**sr, "error": f"Restore exception: {str(e)}"}
            
            t_end = time.perf_counter()
            return {
                **sr,
                "decode_slot": decode_slot,
                "restore_time_ms": (t_end - t_start) * 1000,
            }
        
        # Can restore all in parallel since we have 8 decode slots
        tasks = [restore_one(i, sr) for i, sr in enumerate(saved_results)]
        results = await asyncio.gather(*tasks)
        return results
    
    async def decode_batch(
        self, 
        restored_results: List[Dict[str, Any]], 
        n_predict: int
    ) -> List[Dict[str, Any]]:
        """Generate tokens on decode server"""
        http = await self.get_http()
        
        async def decode_one(rr: Dict) -> Dict[str, Any]:
            if "error" in rr:
                return rr
            
            t_start = time.perf_counter()
            try:
                async with http.post(
                    f"{self.decode_url}/completion",
                    json={
                        "prompt": rr["tokens"],
                        "n_predict": n_predict,
                        "id_slot": rr["decode_slot"],
                        "cache_prompt": True,
                        "temperature": 0.7,
                    }
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return {**rr, "error": f"Decode failed: {error_text[:100]}"}
                    result = await resp.json()
            except Exception as e:
                return {**rr, "error": f"Decode exception: {str(e)}"}
            
            t_end = time.perf_counter()
            timings = result.get("timings", {})
            
            return {
                **rr,
                "decode_time_ms": (t_end - t_start) * 1000,
                "tokens_generated": timings.get("predicted_n", 0),
                "cache_n": timings.get("cache_n", 0),
                "prompt_n": timings.get("prompt_n", 0),
                "content": result.get("content", ""),
            }
        
        # Can decode all in parallel on 8 decode slots
        tasks = [decode_one(rr) for rr in restored_results]
        results = await asyncio.gather(*tasks)
        return results
    
    async def run_pipeline(
        self,
        n_waves: int,
        batch_size: int,
        prompt: str,
        output_tokens: int,
        prompt_tokens: int,
    ) -> PipelineResult:
        """
        Run pipelined disaggregated inference.
        
        Pipeline structure:
        - Wave N prefill runs while Wave N-1 is decoding
        - This overlaps prefill and decode work
        """
        result = PipelineResult(
            mode="disagg_pipeline",
            n_waves=n_waves,
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        
        # Tokenize once (all requests use same prompt)
        tokens = await self.tokenize(prompt)
        print(f"  Tokenized prompt: {len(tokens)} tokens")
        
        # Prepare all requests upfront
        all_requests = []
        for wave_id in range(n_waves):
            for i in range(batch_size):
                all_requests.append({
                    "request_id": f"w{wave_id}_r{i}",
                    "wave_id": wave_id,
                    "tokens": tokens.copy(),
                })
        
        result.total_requests = len(all_requests)
        
        # Run pipeline
        t_pipeline_start = time.perf_counter()
        
        # Track concurrent tasks
        prefill_task = None
        decode_task = None
        current_wave_data = None
        
        waves_completed = []
        
        for wave_id in range(n_waves + 1):  # +1 to drain pipeline
            wave_metrics = WaveMetrics(wave_id=wave_id, batch_size=batch_size)
            
            # Get this wave's requests
            wave_requests = [r for r in all_requests if r["wave_id"] == wave_id] if wave_id < n_waves else []
            
            # === PARALLEL PHASE: Prefill wave N while Decode wave N-1 ===
            
            tasks_to_run = []
            
            # Start prefill for current wave (if any requests)
            if wave_requests:
                wave_metrics.prefill_start = time.perf_counter()
                
                async def do_prefill_and_transfer():
                    # Prefill
                    prefilled = await self.prefill_batch(wave_requests, slot_offset=wave_id * batch_size)
                    # Save immediately
                    saved = await self.save_batch(prefilled)
                    # Restore to decode server
                    restored = await self.restore_batch(saved, slot_offset=wave_id * batch_size)
                    return restored
                
                prefill_task = asyncio.create_task(do_prefill_and_transfer())
                tasks_to_run.append(("prefill", prefill_task))
            
            # Continue decode for previous wave (if any)
            if current_wave_data is not None:
                async def do_decode():
                    return await self.decode_batch(current_wave_data, output_tokens)
                
                decode_task = asyncio.create_task(do_decode())
                tasks_to_run.append(("decode", decode_task))
            
            # Wait for both to complete
            if tasks_to_run:
                results_map = {}
                for name, task in tasks_to_run:
                    results_map[name] = await task
                
                # Record decode results from previous wave
                if "decode" in results_map and wave_id > 0:
                    prev_wave = waves_completed[-1] if waves_completed else None
                    if prev_wave:
                        prev_wave.decode_end = time.perf_counter()
                        for dr in results_map["decode"]:
                            prev_wave.requests.append(RequestMetrics(
                                request_id=dr.get("request_id", ""),
                                success="error" not in dr,
                                prefill_time_ms=dr.get("time_ms", 0),
                                transfer_time_ms=dr.get("save_time_ms", 0) + dr.get("restore_time_ms", 0),
                                decode_time_ms=dr.get("decode_time_ms", 0),
                                cache_hit=dr.get("cache_n", 0) > 0,
                                tokens_generated=dr.get("tokens_generated", 0),
                                error=dr.get("error", ""),
                            ))
                
                # Prepare current wave for decode in next iteration
                if "prefill" in results_map:
                    current_wave_data = results_map["prefill"]
                    wave_metrics.prefill_end = time.perf_counter()
                    wave_metrics.transfer_end = time.perf_counter()
                    wave_metrics.decode_start = time.perf_counter()
                    waves_completed.append(wave_metrics)
                else:
                    current_wave_data = None
        
        t_pipeline_end = time.perf_counter()
        result.total_time_sec = t_pipeline_end - t_pipeline_start
        
        # Aggregate metrics
        result.waves = waves_completed
        all_request_metrics = []
        for wave in waves_completed:
            all_request_metrics.extend(wave.requests)
        
        successful = [r for r in all_request_metrics if r.success]
        result.successful_requests = len(successful)
        
        if successful:
            result.avg_latency_ms = statistics.mean([
                r.prefill_time_ms + r.transfer_time_ms + r.decode_time_ms 
                for r in successful
            ])
            result.avg_prefill_ms = statistics.mean([r.prefill_time_ms for r in successful])
            result.avg_transfer_ms = statistics.mean([r.transfer_time_ms for r in successful])
            result.avg_decode_ms = statistics.mean([r.decode_time_ms for r in successful])
            
            total_tokens = sum(r.tokens_generated for r in successful)
            result.tokens_per_sec = total_tokens / result.total_time_sec
            result.requests_per_sec = len(successful) / result.total_time_sec
            
            cache_hits = sum(1 for r in successful if r.cache_hit)
            result.cache_hit_rate = cache_hits / len(successful) if successful else 0
        
        return result


class BaselineRunner:
    """Runs baseline (non-disaggregated) requests"""
    
    def __init__(self):
        self._http: Optional[aiohttp.ClientSession] = None
    
    async def get_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self._http
    
    async def close(self):
        if self._http:
            await self._http.close()
    
    async def run_baseline(
        self,
        n_waves: int,
        batch_size: int,
        prompt: str,
        output_tokens: int,
        prompt_tokens: int,
    ) -> PipelineResult:
        """Run baseline with same total requests"""
        result = PipelineResult(
            mode="baseline",
            n_waves=n_waves,
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        
        http = await self.get_http()
        total_requests = n_waves * batch_size
        result.total_requests = total_requests
        
        semaphore = asyncio.Semaphore(batch_size)  # Same concurrency as batch size
        
        async def do_request(request_id: int) -> RequestMetrics:
            async with semaphore:
                t_start = time.perf_counter()
                try:
                    async with http.post(
                        f"{BASELINE_URL}/completion",
                        json={
                            "prompt": prompt,
                            "n_predict": output_tokens,
                            "temperature": 0.7,
                        }
                    ) as resp:
                        if resp.status != 200:
                            return RequestMetrics(
                                request_id=str(request_id),
                                success=False,
                                error=f"HTTP {resp.status}",
                                total_time_ms=(time.perf_counter() - t_start) * 1000,
                            )
                        result = await resp.json()
                    
                    t_end = time.perf_counter()
                    timings = result.get("timings", {})
                    
                    return RequestMetrics(
                        request_id=str(request_id),
                        success=True,
                        prefill_time_ms=timings.get("prompt_ms", 0),
                        decode_time_ms=timings.get("predicted_ms", 0),
                        total_time_ms=(t_end - t_start) * 1000,
                        tokens_generated=timings.get("predicted_n", 0),
                    )
                except Exception as e:
                    return RequestMetrics(
                        request_id=str(request_id),
                        success=False,
                        error=str(e),
                        total_time_ms=(time.perf_counter() - t_start) * 1000,
                    )
        
        t_start = time.perf_counter()
        tasks = [do_request(i) for i in range(total_requests)]
        metrics = await asyncio.gather(*tasks)
        t_end = time.perf_counter()
        
        result.total_time_sec = t_end - t_start
        
        successful = [m for m in metrics if m.success]
        result.successful_requests = len(successful)
        
        if successful:
            result.avg_latency_ms = statistics.mean([m.total_time_ms for m in successful])
            result.avg_prefill_ms = statistics.mean([m.prefill_time_ms for m in successful])
            result.avg_decode_ms = statistics.mean([m.decode_time_ms for m in successful])
            
            total_tokens = sum(m.tokens_generated for m in successful)
            result.tokens_per_sec = total_tokens / result.total_time_sec
            result.requests_per_sec = len(successful) / result.total_time_sec
        
        return result


def print_results(baseline: PipelineResult, disagg: PipelineResult):
    """Print comparison"""
    print("\n" + "="*80)
    print("PIPELINE BENCHMARK RESULTS")
    print("="*80)
    
    total_reqs = baseline.n_waves * baseline.batch_size
    print(f"\nConfiguration:")
    print(f"  Waves:          {baseline.n_waves}")
    print(f"  Batch size:     {baseline.batch_size}")
    print(f"  Total requests: {total_reqs}")
    print(f"  Prompt tokens:  ~{baseline.prompt_tokens}")
    print(f"  Output tokens:  {baseline.output_tokens}")
    
    print("\n" + "-"*80)
    print(f"{'Metric':<35} {'Baseline':>18} {'Disagg Pipeline':>18}")
    print("-"*80)
    
    def fmt(val, unit=""):
        if isinstance(val, float):
            return f"{val:.1f}{unit}"
        return f"{val}{unit}"
    
    print(f"{'Total Time':<35} {fmt(baseline.total_time_sec, 's'):>18} {fmt(disagg.total_time_sec, 's'):>18}")
    print(f"{'Successful Requests':<35} {baseline.successful_requests:>18} {disagg.successful_requests:>18}")
    print()
    
    print("THROUGHPUT:")
    print(f"{'  Requests/sec':<35} {fmt(baseline.requests_per_sec):>18} {fmt(disagg.requests_per_sec):>18}")
    print(f"{'  Tokens/sec':<35} {fmt(baseline.tokens_per_sec):>18} {fmt(disagg.tokens_per_sec):>18}")
    print()
    
    print("LATENCY (per request avg):")
    print(f"{'  Total Latency':<35} {fmt(baseline.avg_latency_ms, 'ms'):>18} {fmt(disagg.avg_latency_ms, 'ms'):>18}")
    print(f"{'  Prefill Time':<35} {fmt(baseline.avg_prefill_ms, 'ms'):>18} {fmt(disagg.avg_prefill_ms, 'ms'):>18}")
    if disagg.avg_transfer_ms > 0:
        print(f"{'  Transfer Time':<35} {'-':>18} {fmt(disagg.avg_transfer_ms, 'ms'):>18}")
    print(f"{'  Decode Time':<35} {fmt(baseline.avg_decode_ms, 'ms'):>18} {fmt(disagg.avg_decode_ms, 'ms'):>18}")
    print()
    
    if disagg.cache_hit_rate > 0:
        print(f"{'Cache Hit Rate (disagg)':<35} {'-':>18} {fmt(disagg.cache_hit_rate * 100, '%'):>18}")
    
    print("\n" + "-"*80)
    
    # Summary
    if baseline.requests_per_sec > 0 and disagg.requests_per_sec > 0:
        throughput_ratio = disagg.requests_per_sec / baseline.requests_per_sec
        
        print("\nSUMMARY:")
        if throughput_ratio > 1:
            print(f"  ✓ Disagg pipeline is {throughput_ratio:.2f}x HIGHER throughput")
        else:
            print(f"  ✗ Disagg pipeline is {1/throughput_ratio:.2f}x LOWER throughput")
        
        if disagg.cache_hit_rate > 0.9:
            print(f"  ✓ Cache hit rate: {disagg.cache_hit_rate*100:.0f}%")
        
        # Calculate pipeline efficiency
        if disagg.avg_prefill_ms > 0 and disagg.avg_decode_ms > 0:
            overlap_potential = min(disagg.avg_prefill_ms, disagg.avg_decode_ms)
            print(f"\n  Pipeline overlap potential: {overlap_potential:.0f}ms per wave")
            print(f"  (Prefill and decode can run in parallel for this duration)")
    
    print("="*80)


async def main():
    parser = argparse.ArgumentParser(description="Pipeline Benchmark for Disaggregated Prefill")
    parser.add_argument("--waves", "-w", type=int, default=5,
                        help="Number of waves/batches (default: 5)")
    parser.add_argument("--batch-size", "-b", type=int, default=4,
                        help="Requests per wave (default: 4)")
    parser.add_argument("--prompt-tokens", "-p", type=int, default=500,
                        help="Approximate prompt length (default: 500)")
    parser.add_argument("--output-tokens", "-o", type=int, default=20,
                        help="Tokens to generate (default: 20)")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--disagg-only", action="store_true")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PIPELINE BENCHMARK: Disaggregated Prefill                       ║
║                                                                              ║
║  Tests overlapping prefill/decode: While batch N decodes, batch N+1 prefills ║
║  This pipeline structure is where disaggregated serving excels.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Configuration: {args.waves} waves × {args.batch_size} requests = {args.waves * args.batch_size} total")
    print(f"Prompt: ~{args.prompt_tokens} tokens, Output: {args.output_tokens} tokens\n")
    
    # Generate prompt
    prompt = generate_prompt(args.prompt_tokens)
    print(f"Generated prompt: {len(prompt)} chars")
    
    # Check servers
    print("\nChecking servers...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASELINE_URL}/health") as resp:
                if resp.status == 200:
                    print(f"  ✓ Prefill/Baseline server (8080)")
        except:
            print(f"  ✗ Cannot reach port 8080")
            return
        
        try:
            async with session.get(f"{DECODE_URL}/health") as resp:
                if resp.status == 200:
                    print(f"  ✓ Decode server (8081)")
        except:
            print(f"  ✗ Cannot reach port 8081")
            return
    
    baseline_result = None
    disagg_result = None
    
    # Baseline
    if not args.disagg_only:
        print(f"\n{'='*60}")
        print("Running BASELINE benchmark...")
        print(f"{'='*60}")
        
        runner = BaselineRunner()
        try:
            baseline_result = await runner.run_baseline(
                n_waves=args.waves,
                batch_size=args.batch_size,
                prompt=prompt,
                output_tokens=args.output_tokens,
                prompt_tokens=args.prompt_tokens,
            )
            print(f"  Completed: {baseline_result.successful_requests}/{baseline_result.total_requests}")
            print(f"  Total time: {baseline_result.total_time_sec:.1f}s")
            print(f"  Throughput: {baseline_result.requests_per_sec:.1f} req/s")
        finally:
            await runner.close()
    
    # Disaggregated Pipeline
    if not args.baseline_only:
        print(f"\n{'='*60}")
        print("Running DISAGGREGATED PIPELINE benchmark...")
        print(f"{'='*60}")
        
        runner = DisaggPipelineRunner({
            "prefill_url": PREFILL_URL,
            "decode_url": DECODE_URL,
        })
        try:
            disagg_result = await runner.run_pipeline(
                n_waves=args.waves,
                batch_size=args.batch_size,
                prompt=prompt,
                output_tokens=args.output_tokens,
                prompt_tokens=args.prompt_tokens,
            )
            print(f"  Completed: {disagg_result.successful_requests}/{disagg_result.total_requests}")
            print(f"  Total time: {disagg_result.total_time_sec:.1f}s")
            print(f"  Throughput: {disagg_result.requests_per_sec:.1f} req/s")
            print(f"  Cache hit rate: {disagg_result.cache_hit_rate*100:.0f}%")
            
            # Report any errors
            failed = disagg_result.total_requests - disagg_result.successful_requests
            if failed > 0:
                print(f"  ⚠️  {failed} requests failed")
        finally:
            await runner.close()
    
    # Print comparison
    if baseline_result and disagg_result:
        print_results(baseline_result, disagg_result)


if __name__ == "__main__":
    asyncio.run(main())
