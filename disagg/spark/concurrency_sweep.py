#!/usr/bin/env python3
"""
Concurrency sweep: Baseline vs Disaggregated across DGX Sparks.

Tests whether disagg helps under increasing concurrent load.

Theory:
- At low concurrency: disagg adds overhead (save+transfer+restore) → slower
- At high concurrency: disagg wins because prefill and decode don't compete
  for the same GPU. Each GPU focuses on one task.

Usage:
    python3 disagg/spark/concurrency_sweep.py
    python3 disagg/spark/concurrency_sweep.py --concurrency 1,2,4,8,16 --isl 4000 --osl 128
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
import aiohttp

PREFILL_URL = os.environ.get("PREFILL_URL", "http://192.168.200.13:8080")
DECODE_URL = os.environ.get("DECODE_URL", "http://192.168.200.12:8081")
BASELINE_URL = PREFILL_URL  # Baseline runs on prefill server (single GPU)
DECODE_SSH = os.environ.get("DECODE_SSH", "spark2")
PREFILL_SSH = os.environ.get("PREFILL_SSH", "")
KV_CACHE_DIR = "/tmp/llama_kv_cache"


def generate_prompt(target_tokens: int, variant: int = 0) -> str:
    bases = [
        "The quick brown fox jumps over the lazy dog and runs across the field. ",
        "In the beginning there was nothing and then everything appeared at once. ",
        "The mathematics of distributed systems requires careful analysis of edge cases. ",
        "Every morning the scientist reviewed the experimental data from yesterday. ",
        "Deep in the ocean currents carry nutrients across vast underwater landscapes. ",
        "The history of computing begins with simple mechanical calculating machines. ",
        "Musicians practice scales and arpeggios to develop their technical ability. ",
        "The architecture of modern processors involves complex pipeline stages. ",
        "Quantum entanglement allows particles to share states across any distance. ",
        "The evolution of programming languages reflects changing computing paradigms. ",
        "Climate models simulate atmospheric dynamics using differential equations. ",
        "Neural networks learn representations through iterative gradient descent. ",
        "The philosophy of artificial intelligence raises questions about consciousness. ",
        "Distributed databases use consensus algorithms to maintain data consistency. ",
        "The art of debugging requires patience and systematic hypothesis testing. ",
        "Cryptographic protocols protect communication channels from eavesdroppers. ",
    ]
    base = bases[variant % len(bases)] * 200
    chars = target_tokens * 4
    return (base * (chars // len(base) + 1))[:chars]


@dataclass
class RequestResult:
    request_id: int
    success: bool = False
    total_ms: float = 0
    prefill_ms: float = 0
    decode_ms: float = 0
    save_ms: float = 0
    transfer_ms: float = 0
    restore_ms: float = 0
    tokens_generated: int = 0
    prompt_tokens: int = 0
    ttft_ms: float = 0  # time to first token
    error: str = ""


@dataclass
class SweepPoint:
    concurrency: int
    mode: str  # "baseline" or "disagg"
    isl: int
    osl: int
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successful(self) -> List[RequestResult]:
        return [r for r in self.results if r.success]

    @property
    def avg_total_ms(self) -> float:
        s = self.successful
        return statistics.mean([r.total_ms for r in s]) if s else 0

    @property
    def avg_ttft_ms(self) -> float:
        s = self.successful
        return statistics.mean([r.ttft_ms for r in s]) if s else 0

    @property
    def p50_total_ms(self) -> float:
        s = self.successful
        return statistics.median([r.total_ms for r in s]) if s else 0

    @property
    def p99_total_ms(self) -> float:
        s = self.successful
        if not s:
            return 0
        vals = sorted([r.total_ms for r in s])
        idx = int(len(vals) * 0.99)
        return vals[min(idx, len(vals) - 1)]

    @property
    def throughput_rps(self) -> float:
        s = self.successful
        if not s:
            return 0
        wall_time = max(r.total_ms for r in s) / 1000  # approx
        return len(s) / wall_time if wall_time > 0 else 0

    @property
    def throughput_tps(self) -> float:
        s = self.successful
        if not s:
            return 0
        total_tokens = sum(r.tokens_generated for r in s)
        wall_time = max(r.total_ms for r in s) / 1000
        return total_tokens / wall_time if wall_time > 0 else 0

    @property
    def wall_time_ms(self) -> float:
        """Actual wall clock time (max of all request end times)"""
        s = self.successful
        return max(r.total_ms for r in s) if s else 0


async def scp_transfer(filename: str) -> tuple[float, int]:
    src_path = f"{KV_CACHE_DIR}/{filename}"
    dst = f"{DECODE_SSH}:{KV_CACHE_DIR}/{filename}"
    src = f"{PREFILL_SSH}:{src_path}" if PREFILL_SSH else src_path

    # File size
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
        fsize = int(out.decode().strip())
    except Exception:
        fsize = 0

    t0 = time.perf_counter()
    p = await asyncio.create_subprocess_exec(
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "Compression=no",
        src, dst,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, err = await p.communicate()
    ms = (time.perf_counter() - t0) * 1000
    if p.returncode != 0:
        raise Exception(f"SCP failed: {err.decode()}")
    return ms, fsize


async def clear_all_slots(session: aiohttp.ClientSession):
    """Clear all slots on both servers"""
    for url in [PREFILL_URL, DECODE_URL]:
        for slot_id in range(16):
            try:
                async with session.post(f"{url}/slots/{slot_id}?action=erase"):
                    pass
            except Exception:
                break


async def run_baseline_batch(
    session: aiohttp.ClientSession,
    concurrency: int,
    prompts: List[str],
    osl: int,
) -> List[RequestResult]:
    """Run concurrent baseline requests on a single server."""

    sem = asyncio.Semaphore(concurrency)

    async def do_one(req_id: int) -> RequestResult:
        r = RequestResult(request_id=req_id)
        async with sem:
            t0 = time.perf_counter()
            try:
                async with session.post(f"{BASELINE_URL}/completion", json={
                    "prompt": prompts[req_id % len(prompts)],
                    "n_predict": osl,
                    "cache_prompt": False,
                    "temperature": 0.0,
                }) as resp:
                    if resp.status != 200:
                        r.error = f"HTTP {resp.status}"
                        r.total_ms = (time.perf_counter() - t0) * 1000
                        return r
                    data = await resp.json()
            except Exception as e:
                r.error = str(e)
                r.total_ms = (time.perf_counter() - t0) * 1000
                return r

            r.total_ms = (time.perf_counter() - t0) * 1000
            t = data.get("timings", {})
            r.prefill_ms = t.get("prompt_ms", 0)
            r.decode_ms = t.get("predicted_ms", 0)
            r.tokens_generated = t.get("predicted_n", 0)
            r.prompt_tokens = t.get("prompt_n", 0)
            r.ttft_ms = r.prefill_ms  # first token after prefill
            r.success = True
            return r

    tasks = [do_one(i) for i in range(concurrency)]
    return await asyncio.gather(*tasks)


async def run_disagg_batch(
    session: aiohttp.ClientSession,
    concurrency: int,
    prompts: List[str],
    osl: int,
) -> List[RequestResult]:
    """Run concurrent disagg requests across two machines."""

    # Semaphore not needed since each request gets its own slot
    async def do_one(req_id: int) -> RequestResult:
        r = RequestResult(request_id=req_id)
        prompt = prompts[req_id % len(prompts)]
        filename = f"sweep_{req_id}.bin"
        slot_id = req_id % 16  # distribute across slots

        t0 = time.perf_counter()
        try:
            # 1. Prefill on Spark 1
            t_pf = time.perf_counter()
            async with session.post(f"{PREFILL_URL}/completion", json={
                "prompt": prompt,
                "n_predict": 0,
                "cache_prompt": False,
                "id_slot": slot_id,
            }) as resp:
                if resp.status != 200:
                    r.error = f"Prefill HTTP {resp.status}: {(await resp.text())[:100]}"
                    r.total_ms = (time.perf_counter() - t0) * 1000
                    return r
                pf_data = await resp.json()
            r.prefill_ms = (time.perf_counter() - t_pf) * 1000
            r.prompt_tokens = pf_data.get("timings", {}).get("prompt_n", 0)

            # 2. Save KV cache
            t_sv = time.perf_counter()
            async with session.post(f"{PREFILL_URL}/slots/{slot_id}?action=save", json={
                "filename": filename,
            }) as resp:
                if resp.status != 200:
                    r.error = f"Save failed: {(await resp.text())[:100]}"
                    r.total_ms = (time.perf_counter() - t0) * 1000
                    return r
                await resp.json()
            r.save_ms = (time.perf_counter() - t_sv) * 1000

            # 3. SCP transfer over 200Gbps
            r.transfer_ms, _ = await scp_transfer(filename)
            r.ttft_ms = r.prefill_ms + r.save_ms + r.transfer_ms

            # 4. Restore on Spark 2
            t_rs = time.perf_counter()
            async with session.post(f"{DECODE_URL}/slots/{slot_id}?action=restore", json={
                "filename": filename,
            }) as resp:
                if resp.status != 200:
                    r.error = f"Restore failed: {(await resp.text())[:100]}"
                    r.total_ms = (time.perf_counter() - t0) * 1000
                    return r
                await resp.json()
            r.restore_ms = (time.perf_counter() - t_rs) * 1000

            # 5. Decode on Spark 2
            t_dc = time.perf_counter()
            async with session.post(f"{DECODE_URL}/completion", json={
                "prompt": prompt,
                "n_predict": osl,
                "cache_prompt": True,
                "id_slot": slot_id,
                "temperature": 0.0,
            }) as resp:
                if resp.status != 200:
                    r.error = f"Decode failed: {(await resp.text())[:100]}"
                    r.total_ms = (time.perf_counter() - t0) * 1000
                    return r
                dc_data = await resp.json()
            r.decode_ms = (time.perf_counter() - t_dc) * 1000
            r.tokens_generated = dc_data.get("timings", {}).get("predicted_n", 0)
            r.success = True

        except Exception as e:
            r.error = str(e)

        r.total_ms = (time.perf_counter() - t0) * 1000
        return r

    tasks = [do_one(i) for i in range(concurrency)]
    return await asyncio.gather(*tasks)


async def run_sweep(concurrency_levels: List[int], isl: int, osl: int):
    """Run the full concurrency sweep."""

    # Pre-generate unique prompts
    prompts = [generate_prompt(isl, i) for i in range(max(concurrency_levels))]

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warmup
        print("Warmup...")
        try:
            async with session.post(f"{BASELINE_URL}/completion", json={
                "prompt": prompts[0][:500], "n_predict": 5, "temperature": 0.0,
            }) as resp:
                await resp.json()
        except Exception:
            pass

        all_points: List[SweepPoint] = []

        for conc in concurrency_levels:
            print(f"\n{'='*70}")
            print(f"  Concurrency = {conc}  (ISL={isl}, OSL={osl})")
            print(f"{'='*70}")

            # Clear all slots
            await clear_all_slots(session)
            await asyncio.sleep(1)

            # --- BASELINE ---
            print(f"  Baseline (single GPU on Spark 1)...", end="", flush=True)
            t0 = time.perf_counter()
            baseline_results = await run_baseline_batch(session, conc, prompts, osl)
            baseline_wall = (time.perf_counter() - t0) * 1000

            bp = SweepPoint(concurrency=conc, mode="baseline", isl=isl, osl=osl,
                           results=list(baseline_results))
            all_points.append(bp)

            ok = len(bp.successful)
            fail = conc - ok
            print(f" {ok}/{conc} ok, wall={baseline_wall:.0f}ms, "
                  f"avg={bp.avg_total_ms:.0f}ms, ttft={bp.avg_ttft_ms:.0f}ms")
            if fail > 0:
                errors = [r.error for r in bp.results if not r.success]
                print(f"    Errors: {errors[:3]}")

            # Clear all slots
            await clear_all_slots(session)
            await asyncio.sleep(1)

            # --- DISAGG ---
            print(f"  Disagg (prefill Spark1 + decode Spark2)...", end="", flush=True)
            t0 = time.perf_counter()
            disagg_results = await run_disagg_batch(session, conc, prompts, osl)
            disagg_wall = (time.perf_counter() - t0) * 1000

            dp = SweepPoint(concurrency=conc, mode="disagg", isl=isl, osl=osl,
                           results=list(disagg_results))
            all_points.append(dp)

            ok = len(dp.successful)
            fail = conc - ok
            print(f" {ok}/{conc} ok, wall={disagg_wall:.0f}ms, "
                  f"avg={dp.avg_total_ms:.0f}ms, ttft={dp.avg_ttft_ms:.0f}ms")
            if fail > 0:
                errors = [r.error for r in dp.results if not r.success]
                print(f"    Errors: {errors[:3]}")

            # Cleanup SCP files
            try:
                if PREFILL_SSH:
                    await asyncio.create_subprocess_exec(
                        "ssh", PREFILL_SSH, f"rm -f {KV_CACHE_DIR}/sweep_*.bin",
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                else:
                    p = await asyncio.create_subprocess_exec(
                        "bash", "-c", f"rm -f {KV_CACHE_DIR}/sweep_*.bin",
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                    await p.communicate()
                await asyncio.create_subprocess_exec(
                    "ssh", DECODE_SSH, f"rm -f {KV_CACHE_DIR}/sweep_*.bin",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            except Exception:
                pass

    return all_points


def print_summary(points: List[SweepPoint]):
    """Print the sweep summary table."""

    print("\n\n" + "=" * 90)
    print("CONCURRENCY SWEEP RESULTS")
    print("=" * 90)

    # Group by concurrency
    conc_levels = sorted(set(p.concurrency for p in points))

    print(f"\n{'Conc':>5} | {'--- Baseline (1 GPU) ---':^36} | {'--- Disagg (2 GPUs) ---':^36} | {'Speedup':>8}")
    print(f"{'':>5} | {'Avg(ms)':>8} {'TTFT(ms)':>9} {'Tok/s':>8} {'Ok':>4}{'':>7} | {'Avg(ms)':>8} {'TTFT(ms)':>9} {'Tok/s':>8} {'Ok':>4}{'':>7} | {'':>8}")
    print("-" * 90)

    for conc in conc_levels:
        bp = next((p for p in points if p.concurrency == conc and p.mode == "baseline"), None)
        dp = next((p for p in points if p.concurrency == conc and p.mode == "disagg"), None)

        if not bp or not dp:
            continue

        # Calculate throughput as total_tokens / wall_time
        b_ok = len(bp.successful)
        d_ok = len(dp.successful)
        b_total_tok = sum(r.tokens_generated for r in bp.successful)
        d_total_tok = sum(r.tokens_generated for r in dp.successful)
        b_wall = bp.wall_time_ms / 1000 if bp.wall_time_ms > 0 else 1
        d_wall = dp.wall_time_ms / 1000 if dp.wall_time_ms > 0 else 1
        b_tps = b_total_tok / b_wall
        d_tps = d_total_tok / d_wall

        speedup = d_tps / b_tps if b_tps > 0 else 0

        print(f"{conc:>5} | {bp.avg_total_ms:>8.0f} {bp.avg_ttft_ms:>9.0f} {b_tps:>8.1f} {b_ok:>4}/{conc:<3}"
              f" | {dp.avg_total_ms:>8.0f} {dp.avg_ttft_ms:>9.0f} {d_tps:>8.1f} {d_ok:>4}/{conc:<3}"
              f" | {speedup:>7.2f}x")

    print("-" * 90)

    # Detailed breakdown for disagg
    print(f"\nDisagg breakdown (avg ms):")
    print(f"{'Conc':>5} | {'Prefill':>8} {'Save':>8} {'Transfer':>8} {'Restore':>8} {'Decode':>8} | {'Overhead':>8}")
    print("-" * 70)
    for conc in conc_levels:
        dp = next((p for p in points if p.concurrency == conc and p.mode == "disagg"), None)
        if not dp or not dp.successful:
            continue
        s = dp.successful
        avg_pf = statistics.mean([r.prefill_ms for r in s])
        avg_sv = statistics.mean([r.save_ms for r in s])
        avg_xf = statistics.mean([r.transfer_ms for r in s])
        avg_rs = statistics.mean([r.restore_ms for r in s])
        avg_dc = statistics.mean([r.decode_ms for r in s])
        overhead = avg_sv + avg_xf + avg_rs
        print(f"{conc:>5} | {avg_pf:>8.0f} {avg_sv:>8.0f} {avg_xf:>8.0f} {avg_rs:>8.0f} {avg_dc:>8.0f} | {overhead:>8.0f}")

    print("=" * 90)


async def main():
    parser = argparse.ArgumentParser(description="Concurrency Sweep Benchmark")
    parser.add_argument("--concurrency", type=str, default="1,2,4,8,12,16",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--isl", type=int, default=4000, help="Input sequence length (tokens)")
    parser.add_argument("--osl", type=int, default=128, help="Output sequence length (tokens)")
    parser.add_argument("--prefill-url", type=str, default=None)
    parser.add_argument("--decode-url", type=str, default=None)

    args = parser.parse_args()
    conc_levels = [int(c.strip()) for c in args.concurrency.split(",")]

    global PREFILL_URL, DECODE_URL, BASELINE_URL
    if args.prefill_url:
        PREFILL_URL = args.prefill_url
        BASELINE_URL = args.prefill_url
    if args.decode_url:
        DECODE_URL = args.decode_url

    print("=" * 70)
    print("CONCURRENCY SWEEP: Baseline vs Disaggregated")
    print("=" * 70)
    print(f"ISL: {args.isl} tokens, OSL: {args.osl} tokens")
    print(f"Concurrency levels: {conc_levels}")
    print(f"Baseline: {BASELINE_URL} (single GPU)")
    print(f"Disagg:   {PREFILL_URL} (prefill) + {DECODE_URL} (decode)")
    print(f"Transfer: SCP via {DECODE_SSH}")

    # Check servers
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for name, url in [("Prefill/Baseline", PREFILL_URL), ("Decode", DECODE_URL)]:
            try:
                async with session.get(f"{url}/health") as resp:
                    if "ok" not in (await resp.text()).lower():
                        raise Exception("Not healthy")
                print(f"  OK: {name} at {url}")
            except Exception as e:
                print(f"  FAIL: {name} at {url} - {e}")
                return

    points = await run_sweep(conc_levels, args.isl, args.osl)
    print_summary(points)

    # Save raw results
    raw = []
    for p in points:
        raw.append({
            "concurrency": p.concurrency,
            "mode": p.mode,
            "isl": p.isl,
            "osl": p.osl,
            "n_ok": len(p.successful),
            "n_total": len(p.results),
            "avg_total_ms": p.avg_total_ms,
            "avg_ttft_ms": p.avg_ttft_ms,
            "p50_total_ms": p.p50_total_ms,
            "wall_time_ms": p.wall_time_ms,
        })
    print(f"\nRaw JSON:")
    print(json.dumps(raw, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
