#!/usr/bin/env python3
"""
Distributed orchestrator for disaggregated inference across two DGX Spark machines.

KV cache transfer flow:
  1. Prefill on Spark 1 (192.168.200.13:8080)
  2. Save KV cache to file on Spark 1
  3. SCP file from Spark 1 to Spark 2 over 200Gbps RoCE interconnect
  4. Restore KV cache on Spark 2
  5. Decode on Spark 2 (192.168.200.12:8081)

The orchestrator can run on either Spark or on an external machine.

Usage:
    # Run from Spark 1 (prefill machine) - fastest, SCP goes direct
    python3 orchestrator.py

    # Run from external machine
    python3 orchestrator.py --prefill-ssh nvidia@10.110.22.129

    # Custom configuration
    python3 orchestrator.py \
        --prefill-url http://192.168.200.13:8080 \
        --decode-url http://192.168.200.12:8081 \
        --decode-ssh nvidia@192.168.200.12
"""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SparkConfig:
    """Configuration for dual DGX Spark setup"""
    prefill_url: str = "http://192.168.200.13:8080"
    decode_url: str = "http://192.168.200.12:8081"

    # SSH target for file transfer (from orchestrator's perspective)
    # If orchestrator runs on prefill machine, prefill_ssh can be empty (local)
    prefill_ssh: str = ""           # e.g., "nvidia@192.168.200.13"
    decode_ssh: str = "nvidia@192.168.200.12"  # Target for SCP

    kv_cache_dir: str = "/tmp/llama_kv_cache"

    # Transfer method: "scp", "rsync", or "nc" (netcat for raw speed)
    transfer_method: str = "scp"

    # Use SSH ControlMaster alias (e.g., "spark2") for fast transfers
    # Set up with: ssh -fN spark2 (after configuring ~/.ssh/config)
    decode_ssh_alias: str = ""  # e.g., "spark2" for ControlMaster


@dataclass
class RequestMetrics:
    request_id: str
    start_time: float = field(default_factory=time.perf_counter)
    prefill_ms: float = 0.0
    save_ms: float = 0.0
    transfer_ms: float = 0.0
    restore_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    transfer_bytes: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_ms": self.total_ms,
            "prefill_ms": self.prefill_ms,
            "save_ms": self.save_ms,
            "transfer_ms": self.transfer_ms,
            "restore_ms": self.restore_ms,
            "decode_ms": self.decode_ms,
            "transfer_bytes": self.transfer_bytes,
            "transfer_gbps": (self.transfer_bytes * 8 / 1e9) / (self.transfer_ms / 1000)
            if self.transfer_ms > 0 else 0,
        }


class DistributedOrchestrator:
    """
    Orchestrator for cross-machine disaggregated inference.
    Handles KV cache file transfer between prefill and decode machines.
    """

    def __init__(self, config: SparkConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self.decode_slots = 2
        self.next_decode_slot = 0
        self._slot_lock = asyncio.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()

    async def get_decode_slot(self) -> int:
        async with self._slot_lock:
            slot = self.next_decode_slot
            self.next_decode_slot = (self.next_decode_slot + 1) % self.decode_slots
            return slot

    async def transfer_kv_cache(self, filename: str) -> tuple[float, int]:
        """
        Transfer KV cache file from prefill machine to decode machine.
        Returns (transfer_time_ms, bytes_transferred).
        """
        src_path = f"{self.config.kv_cache_dir}/{filename}"
        dst_path = f"{self.config.kv_cache_dir}/{filename}"
        cfg = self.config

        t_start = time.perf_counter()

        if cfg.transfer_method == "scp":
            # Build SCP command
            if cfg.prefill_ssh:
                # Orchestrator is remote from both machines
                src = f"{cfg.prefill_ssh}:{src_path}"
            else:
                # Orchestrator is on the prefill machine (local file)
                src = src_path

            # Use ControlMaster alias if available (much faster)
            scp_target = cfg.decode_ssh_alias or cfg.decode_ssh
            dst = f"{scp_target}:{dst_path}"

            proc = await asyncio.create_subprocess_exec(
                "scp", "-o", "StrictHostKeyChecking=no",
                "-o", "Compression=no",      # No compression - raw speed
                "-c", "aes128-gcm@openssh.com",  # Fast cipher
                src, dst,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise Exception(f"SCP failed: {stderr.decode()}")

        elif cfg.transfer_method == "rsync":
            if cfg.prefill_ssh:
                src = f"{cfg.prefill_ssh}:{src_path}"
            else:
                src = src_path
            dst = f"{cfg.decode_ssh}:{dst_path}"

            proc = await asyncio.create_subprocess_exec(
                "rsync", "-e", "ssh -o StrictHostKeyChecking=no -c aes128-gcm@openssh.com",
                "--compress-choice=none", src, dst,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise Exception(f"rsync failed: {stderr.decode()}")

        else:
            raise ValueError(f"Unknown transfer method: {cfg.transfer_method}")

        t_end = time.perf_counter()
        transfer_ms = (t_end - t_start) * 1000

        # Get file size for bandwidth calculation
        try:
            if cfg.prefill_ssh:
                proc = await asyncio.create_subprocess_exec(
                    "ssh", cfg.prefill_ssh, f"stat -c%s {src_path}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    "stat", "-c%s", src_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            stdout, _ = await proc.communicate()
            file_size = int(stdout.decode().strip())
        except Exception:
            file_size = 0

        return transfer_ms, file_size

    async def disagg_completion(
        self,
        prompt: str,
        n_predict: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute disaggregated completion across two machines."""
        request_id = str(uuid.uuid4())[:8]
        metrics = RequestMetrics(request_id=request_id)
        session = await self.get_session()

        filename = f"kv_{request_id}.bin"
        decode_slot = await self.get_decode_slot()

        logger.info(f"[{request_id}] Starting distributed disagg, decode_slot={decode_slot}")

        try:
            # Step 1: Prefill on Spark 1
            t0 = time.perf_counter()
            async with session.post(
                f"{self.config.prefill_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 0,
                    "cache_prompt": True,
                    "id_slot": 0,
                }
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Prefill failed: {await resp.text()}")
                prefill_result = await resp.json()
            metrics.prefill_ms = (time.perf_counter() - t0) * 1000

            prompt_tokens = prefill_result.get("timings", {}).get("prompt_n", 0)
            logger.info(f"[{request_id}] Prefill: {prompt_tokens} tokens, "
                       f"{metrics.prefill_ms:.1f}ms")

            # Step 2: Save KV cache on Spark 1
            t0 = time.perf_counter()
            async with session.post(
                f"{self.config.prefill_url}/slots/0?action=save",
                json={"filename": filename}
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Save failed: {await resp.text()}")
                save_result = await resp.json()
            metrics.save_ms = (time.perf_counter() - t0) * 1000

            logger.info(f"[{request_id}] Save: {save_result.get('n_saved', 0)} tokens, "
                       f"{metrics.save_ms:.1f}ms")

            # Step 3: Transfer KV cache from Spark 1 to Spark 2
            t0 = time.perf_counter()
            transfer_ms, transfer_bytes = await self.transfer_kv_cache(filename)
            metrics.transfer_ms = transfer_ms
            metrics.transfer_bytes = transfer_bytes

            bw_gbps = (transfer_bytes * 8 / 1e9) / (transfer_ms / 1000) if transfer_ms > 0 else 0
            logger.info(f"[{request_id}] Transfer: {transfer_bytes/1e6:.1f}MB in "
                       f"{transfer_ms:.1f}ms ({bw_gbps:.1f} Gbps)")

            # Step 4: Restore KV cache on Spark 2
            t0 = time.perf_counter()
            async with session.post(
                f"{self.config.decode_url}/slots/{decode_slot}?action=restore",
                json={"filename": filename}
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Restore failed: {await resp.text()}")
                restore_result = await resp.json()
            metrics.restore_ms = (time.perf_counter() - t0) * 1000

            logger.info(f"[{request_id}] Restore: {metrics.restore_ms:.1f}ms")

            # Step 5: Decode on Spark 2
            t0 = time.perf_counter()
            async with session.post(
                f"{self.config.decode_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "cache_prompt": True,
                    "id_slot": decode_slot,
                    **kwargs
                }
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Decode failed: {await resp.text()}")
                decode_result = await resp.json()
            metrics.decode_ms = (time.perf_counter() - t0) * 1000

            tokens_gen = decode_result.get("timings", {}).get("predicted_n", 0)
            logger.info(f"[{request_id}] Decode: {tokens_gen} tokens, "
                       f"{metrics.decode_ms:.1f}ms")

            metrics.total_ms = (time.perf_counter() - metrics.start_time) * 1000

            # Cleanup: delete KV cache file on both machines
            asyncio.create_task(self._cleanup_file(filename))

            decode_result["disagg_metrics"] = metrics.to_dict()
            decode_result["disagg_request_id"] = request_id
            return decode_result

        except Exception as e:
            logger.error(f"[{request_id}] Error: {e}")
            raise

    async def _cleanup_file(self, filename: str):
        """Clean up KV cache files from both machines"""
        path = f"{self.config.kv_cache_dir}/{filename}"
        try:
            if self.config.prefill_ssh:
                await asyncio.create_subprocess_exec(
                    "ssh", self.config.prefill_ssh, f"rm -f {path}",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            else:
                os.remove(path) if os.path.exists(path) else None

            await asyncio.create_subprocess_exec(
                "ssh", self.config.decode_ssh, f"rm -f {path}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception:
            pass


# === HTTP Server ===

async def handle_completion(request: web.Request) -> web.Response:
    orch: DistributedOrchestrator = request.app["orchestrator"]
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    prompt = data.pop("prompt", "")
    if not prompt:
        return web.json_response({"error": "No prompt"}, status=400)

    try:
        result = await orch.disagg_completion(prompt, **data)
        return web.json_response(result)
    except Exception as e:
        logger.exception("Request failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def on_startup(app: web.Application):
    config = SparkConfig(
        prefill_url=app["prefill_url"],
        decode_url=app["decode_url"],
        prefill_ssh=app["prefill_ssh"],
        decode_ssh=app["decode_ssh"],
        transfer_method=app["transfer_method"],
    )
    app["orchestrator"] = DistributedOrchestrator(config)
    logger.info(f"Distributed orchestrator started")
    logger.info(f"  Prefill: {config.prefill_url}")
    logger.info(f"  Decode:  {config.decode_url}")
    logger.info(f"  Transfer: {config.transfer_method} via {config.decode_ssh}")


async def on_cleanup(app: web.Application):
    orch = app.get("orchestrator")
    if orch:
        await orch.close()


def main():
    parser = argparse.ArgumentParser(description="Distributed Disagg Orchestrator")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--prefill-url", type=str, default="http://192.168.200.13:8080")
    parser.add_argument("--decode-url", type=str, default="http://192.168.200.12:8081")
    parser.add_argument("--prefill-ssh", type=str, default="",
                        help="SSH target for prefill machine (empty = local)")
    parser.add_argument("--decode-ssh", type=str, default="nvidia@192.168.200.12",
                        help="SSH target for decode machine")
    parser.add_argument("--transfer-method", choices=["scp", "rsync"], default="scp")

    args = parser.parse_args()

    app = web.Application()
    app["prefill_url"] = args.prefill_url
    app["decode_url"] = args.decode_url
    app["prefill_ssh"] = args.prefill_ssh
    app["decode_ssh"] = args.decode_ssh
    app["transfer_method"] = args.transfer_method

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/health", handle_health)
    app.router.add_post("/completion", handle_completion)

    logger.info(f"Starting on {args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
