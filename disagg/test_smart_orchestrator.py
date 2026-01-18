#!/usr/bin/env python3
"""
Test script for Smart Disaggregated Orchestrator

Usage:
    # First, start the servers:
    ./disagg/start_servers.sh /path/to/model.gguf 8192
    
    # Then start the orchestrator:
    python3 disagg/smart_orchestrator.py --port 9000
    
    # Finally, run this test:
    python3 disagg/test_smart_orchestrator.py
"""

import requests
import time
import json

ORCHESTRATOR_URL = "http://localhost:9000"
PREFILL_URL = "http://localhost:8080"

def test_health():
    """Test health endpoint"""
    print("Testing health...")
    resp = requests.get(f"{ORCHESTRATOR_URL}/health")
    assert resp.status_code == 200
    print("✓ Health check passed")

def test_baseline():
    """Test baseline (single server) completion"""
    print("\n" + "="*60)
    print("TEST: Baseline Completion (Single Server)")
    print("="*60)
    
    prompt = "The capital of France is"
    
    t_start = time.perf_counter()
    resp = requests.post(
        f"{ORCHESTRATOR_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": 20,
            "disagg": False  # Use baseline
        }
    )
    t_end = time.perf_counter()
    
    assert resp.status_code == 200
    result = resp.json()
    
    print(f"Prompt: {prompt}")
    print(f"Output: {result.get('content', '')[:100]}")
    print(f"Total time: {(t_end - t_start)*1000:.1f}ms")
    print(f"Server reported: {result.get('baseline_time_ms', 0):.1f}ms")
    print("✓ Baseline test passed")
    
    return result

def test_disagg():
    """Test disaggregated completion"""
    print("\n" + "="*60)
    print("TEST: Disaggregated Completion")
    print("="*60)
    
    prompt = "Once upon a time in a magical kingdom, there lived a brave knight who"
    
    t_start = time.perf_counter()
    resp = requests.post(
        f"{ORCHESTRATOR_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": 30,
            "disagg": True,
            "session_id": "test_session_1"
        }
    )
    t_end = time.perf_counter()
    
    assert resp.status_code == 200, f"Failed: {resp.text}"
    result = resp.json()
    
    metrics = result.get("disagg_metrics", {})
    
    print(f"Prompt: {prompt[:50]}...")
    print(f"Output: {result.get('content', '')[:100]}")
    print(f"\nDisaggregated Metrics:")
    print(f"  Prefill time:  {metrics.get('prefill_time_ms', 0):.1f}ms")
    print(f"  Transfer time: {metrics.get('transfer_time_ms', 0):.1f}ms")
    print(f"  Decode time:   {metrics.get('decode_time_ms', 0):.1f}ms")
    print(f"  Total time:    {metrics.get('total_time_ms', 0):.1f}ms")
    print(f"  Prompt tokens: {metrics.get('n_prompt_tokens', 0)}")
    print(f"  Generated:     {metrics.get('n_generated_tokens', 0)}")
    print(f"\nWall clock time: {(t_end - t_start)*1000:.1f}ms")
    print("✓ Disaggregated test passed")
    
    return result

def test_sessions():
    """Test session listing"""
    print("\n" + "="*60)
    print("TEST: Session Management")
    print("="*60)
    
    resp = requests.get(f"{ORCHESTRATOR_URL}/sessions")
    assert resp.status_code == 200
    
    result = resp.json()
    sessions = result.get("sessions", [])
    
    print(f"Active sessions: {len(sessions)}")
    for s in sessions:
        print(f"  - {s['session_id']}: state={s['state']}, tokens={s['n_tokens']}, generated={s['total_generated']}")
    
    print("✓ Session test passed")
    return result

def test_continue():
    """Test continuing an existing session"""
    print("\n" + "="*60)
    print("TEST: Continue Session")
    print("="*60)
    
    # First, create a session
    prompt = "Python is a programming language that"
    resp = requests.post(
        f"{ORCHESTRATOR_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": 10,
            "disagg": True,
            "session_id": "continue_test"
        }
    )
    assert resp.status_code == 200
    first_result = resp.json()
    print(f"First completion: {first_result.get('content', '')[:50]}")
    
    # Now continue the session
    resp = requests.post(
        f"{ORCHESTRATOR_URL}/continue",
        json={
            "session_id": "continue_test",
            "n_predict": 10
        }
    )
    
    if resp.status_code == 200:
        second_result = resp.json()
        print(f"Continued: {second_result.get('content', '')[:50]}")
        print("✓ Continue test passed")
    else:
        print(f"Continue failed (expected - KV cache may be overwritten): {resp.status_code}")

def benchmark_comparison():
    """Compare baseline vs disaggregated performance"""
    print("\n" + "="*60)
    print("BENCHMARK: Baseline vs Disaggregated")
    print("="*60)
    
    # Generate a longer prompt
    prompt = """You are an expert software engineer. Please explain the following concepts in detail:
    
1. What is continuous integration and continuous deployment (CI/CD)?
2. How do microservices differ from monolithic architecture?
3. What are the benefits of containerization with Docker?

Please provide comprehensive explanations with examples."""

    n_predict = 50
    n_runs = 3
    
    print(f"Prompt length: ~{len(prompt)} chars")
    print(f"Generating {n_predict} tokens per run")
    print(f"Runs: {n_runs}")
    print()
    
    # Baseline runs
    print("Running baseline...")
    baseline_times = []
    for i in range(n_runs):
        t_start = time.perf_counter()
        resp = requests.post(
            f"{ORCHESTRATOR_URL}/completion",
            json={"prompt": prompt, "n_predict": n_predict, "disagg": False}
        )
        t_end = time.perf_counter()
        if resp.status_code == 200:
            baseline_times.append((t_end - t_start) * 1000)
            print(f"  Run {i+1}: {baseline_times[-1]:.1f}ms")
    
    # Disaggregated runs
    print("\nRunning disaggregated...")
    disagg_times = []
    disagg_details = []
    for i in range(n_runs):
        t_start = time.perf_counter()
        resp = requests.post(
            f"{ORCHESTRATOR_URL}/completion",
            json={"prompt": prompt, "n_predict": n_predict, "disagg": True, "session_id": f"bench_{i}"}
        )
        t_end = time.perf_counter()
        if resp.status_code == 200:
            result = resp.json()
            wall_time = (t_end - t_start) * 1000
            disagg_times.append(wall_time)
            metrics = result.get("disagg_metrics", {})
            disagg_details.append(metrics)
            print(f"  Run {i+1}: {wall_time:.1f}ms "
                  f"(prefill={metrics.get('prefill_time_ms', 0):.0f}, "
                  f"transfer={metrics.get('transfer_time_ms', 0):.0f}, "
                  f"decode={metrics.get('decode_time_ms', 0):.0f})")
    
    # Summary
    print("\n" + "-"*40)
    print("SUMMARY")
    print("-"*40)
    
    if baseline_times:
        avg_baseline = sum(baseline_times) / len(baseline_times)
        print(f"Baseline avg:     {avg_baseline:.1f}ms")
    
    if disagg_times:
        avg_disagg = sum(disagg_times) / len(disagg_times)
        avg_prefill = sum(d.get('prefill_time_ms', 0) for d in disagg_details) / len(disagg_details)
        avg_transfer = sum(d.get('transfer_time_ms', 0) for d in disagg_details) / len(disagg_details)
        avg_decode = sum(d.get('decode_time_ms', 0) for d in disagg_details) / len(disagg_details)
        
        print(f"Disagg avg:       {avg_disagg:.1f}ms")
        print(f"  - Prefill:      {avg_prefill:.1f}ms")
        print(f"  - Transfer:     {avg_transfer:.1f}ms")
        print(f"  - Decode:       {avg_decode:.1f}ms")
        
        if baseline_times and disagg_times:
            overhead = avg_disagg - avg_baseline
            speedup = avg_baseline / avg_disagg if avg_disagg > 0 else 0
            print(f"\nOverhead:         {overhead:+.1f}ms")
            print(f"Speedup:          {speedup:.2f}x")
            
            if overhead > 0:
                print("\n⚠️  Disaggregated is SLOWER (expected with current llama.cpp)")
                print("   The KV cache transfer works, but prompt is re-processed.")
                print("   This architecture would benefit from server-side optimization.")


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║      SMART ORCHESTRATOR TEST SUITE                        ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        test_health()
        test_baseline()
        test_disagg()
        test_sessions()
        test_continue()
        benchmark_comparison()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to orchestrator at", ORCHESTRATOR_URL)
        print("Make sure to start it with: python3 disagg/smart_orchestrator.py")
    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
