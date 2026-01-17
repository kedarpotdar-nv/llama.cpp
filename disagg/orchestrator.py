#!/usr/bin/env python3
"""
Simple orchestrator for disaggregated prefill/decode.

This handles:
1. Routing prefill requests to the prefill server
2. Transferring KV cache to decode server
3. Managing concurrent requests

Usage:
    python3 orchestrator.py --port 9000

Then send requests to http://localhost:9000/v1/chat/completions
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    prefill_url: str = "http://localhost:8080"
    decode_url: str = "http://localhost:8081"
    kv_cache_dir: str = "/tmp/llama_kv_cache"


@dataclass
class RequestMetrics:
    request_id: str
    start_time: float = field(default_factory=time.perf_counter)
    prefill_start: float = 0.0
    prefill_end: float = 0.0
    save_start: float = 0.0
    save_end: float = 0.0
    restore_start: float = 0.0
    restore_end: float = 0.0
    decode_start: float = 0.0
    decode_end: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        now = time.perf_counter()
        return {
            "total_ms": (now - self.start_time) * 1000,
            "prefill_ms": (self.prefill_end - self.prefill_start) * 1000 if self.prefill_end else 0,
            "save_ms": (self.save_end - self.save_start) * 1000 if self.save_end else 0,
            "restore_ms": (self.restore_end - self.restore_start) * 1000 if self.restore_end else 0,
            "decode_ms": (self.decode_end - self.decode_start) * 1000 if self.decode_end else 0,
        }


class DisaggOrchestrator:
    """
    Orchestrator that routes prefill to one server, decode to another.
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.active_sessions: Dict[str, Any] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Slot management for decode server (simple round-robin)
        self.decode_slots = 4  # Matches -np 4 in start_servers.sh
        self.next_decode_slot = 0
        self._slot_lock = asyncio.Lock()
    
    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
        return self._session
    
    async def close(self):
        if self._session:
            await self._session.close()
    
    async def get_decode_slot(self) -> int:
        """Get next available decode slot (simple round-robin)"""
        async with self._slot_lock:
            slot = self.next_decode_slot
            self.next_decode_slot = (self.next_decode_slot + 1) % self.decode_slots
            return slot
    
    async def disagg_completion(
        self,
        prompt: str,
        n_predict: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute disaggregated completion:
        1. Prefill on prefill server
        2. Save KV cache
        3. Restore on decode server  
        4. Generate tokens
        """
        request_id = str(uuid.uuid4())[:8]
        metrics = RequestMetrics(request_id=request_id)
        session = await self.get_session()
        
        filename = f"kv_{request_id}.bin"
        decode_slot = await self.get_decode_slot()
        
        logger.info(f"[{request_id}] Starting disagg completion, decode_slot={decode_slot}")
        
        try:
            # Step 1: Prefill only
            metrics.prefill_start = time.perf_counter()
            async with session.post(
                f"{self.config.prefill_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 0,  # Prefill only
                    "cache_prompt": True,
                    "id_slot": 0,
                }
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Prefill failed: {error}")
                prefill_result = await resp.json()
            metrics.prefill_end = time.perf_counter()
            
            prompt_tokens = prefill_result.get("timings", {}).get("prompt_n", 0)
            logger.info(f"[{request_id}] Prefill complete: {prompt_tokens} tokens in "
                       f"{(metrics.prefill_end - metrics.prefill_start)*1000:.1f}ms")
            
            # Step 2: Save KV cache
            metrics.save_start = time.perf_counter()
            async with session.post(
                f"{self.config.prefill_url}/slots/0?action=save",
                json={"filename": filename}
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Save failed: {error}")
                save_result = await resp.json()
            metrics.save_end = time.perf_counter()
            
            n_saved = save_result.get("n_saved", 0)
            logger.info(f"[{request_id}] Save complete: {n_saved} tokens in "
                       f"{(metrics.save_end - metrics.save_start)*1000:.1f}ms")
            
            # Step 3: Restore on decode server
            metrics.restore_start = time.perf_counter()
            async with session.post(
                f"{self.config.decode_url}/slots/{decode_slot}?action=restore",
                json={"filename": filename}
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Restore failed: {error}")
                restore_result = await resp.json()
            metrics.restore_end = time.perf_counter()
            
            logger.info(f"[{request_id}] Restore complete in "
                       f"{(metrics.restore_end - metrics.restore_start)*1000:.1f}ms")
            
            # Step 4: Generate on decode server
            metrics.decode_start = time.perf_counter()
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
                    error = await resp.text()
                    raise Exception(f"Decode failed: {error}")
                decode_result = await resp.json()
            metrics.decode_end = time.perf_counter()
            
            tokens_generated = decode_result.get("timings", {}).get("predicted_n", 0)
            logger.info(f"[{request_id}] Decode complete: {tokens_generated} tokens in "
                       f"{(metrics.decode_end - metrics.decode_start)*1000:.1f}ms")
            
            # Add our metrics to the response
            decode_result["disagg_metrics"] = metrics.to_dict()
            decode_result["disagg_request_id"] = request_id
            
            return decode_result
            
        except Exception as e:
            logger.error(f"[{request_id}] Error: {e}")
            raise
    
    async def baseline_completion(
        self,
        prompt: str,
        n_predict: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Standard single-server completion for comparison"""
        session = await self.get_session()
        
        async with session.post(
            f"{self.config.prefill_url}/completion",
            json={
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": temperature,
                "cache_prompt": True,
                **kwargs
            }
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Completion failed: {error}")
            return await resp.json()


# ============= HTTP Server for Orchestrator =============

async def handle_completion(request: web.Request) -> web.Response:
    """Handle /completion endpoint"""
    orchestrator: DisaggOrchestrator = request.app["orchestrator"]
    
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    
    prompt = data.get("prompt", "")
    if not prompt:
        return web.json_response({"error": "No prompt provided"}, status=400)
    
    # Check if disaggregated mode is requested
    use_disagg = data.pop("disagg", True)  # Default to disaggregated
    
    try:
        if use_disagg:
            result = await orchestrator.disagg_completion(prompt, **data)
        else:
            result = await orchestrator.baseline_completion(prompt, **data)
        
        return web.json_response(result)
    
    except Exception as e:
        logger.exception("Request failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_chat_completion(request: web.Request) -> web.Response:
    """Handle /v1/chat/completions (OpenAI-compatible)"""
    orchestrator: DisaggOrchestrator = request.app["orchestrator"]
    
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    
    messages = data.get("messages", [])
    if not messages:
        return web.json_response({"error": "No messages provided"}, status=400)
    
    # Convert messages to prompt (simple version)
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}\n")
        elif role == "user":
            prompt_parts.append(f"User: {content}\n")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}\n")
    prompt_parts.append("Assistant: ")
    prompt = "".join(prompt_parts)
    
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    use_disagg = data.get("disagg", True)
    
    try:
        if use_disagg:
            result = await orchestrator.disagg_completion(
                prompt, n_predict=max_tokens, temperature=temperature
            )
        else:
            result = await orchestrator.baseline_completion(
                prompt, n_predict=max_tokens, temperature=temperature
            )
        
        # Format as OpenAI-style response
        response = {
            "id": f"chatcmpl-{result.get('disagg_request_id', 'baseline')}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "disagg-llama",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("content", "")
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get("timings", {}).get("prompt_n", 0),
                "completion_tokens": result.get("timings", {}).get("predicted_n", 0),
                "total_tokens": (
                    result.get("timings", {}).get("prompt_n", 0) +
                    result.get("timings", {}).get("predicted_n", 0)
                )
            }
        }
        
        if "disagg_metrics" in result:
            response["disagg_metrics"] = result["disagg_metrics"]
        
        return web.json_response(response)
    
    except Exception as e:
        logger.exception("Chat completion failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_health(request: web.Request) -> web.Response:
    """Health check"""
    return web.json_response({"status": "ok"})


async def on_startup(app: web.Application):
    """Initialize orchestrator on startup"""
    config = ServerConfig(
        prefill_url=app["prefill_url"],
        decode_url=app["decode_url"],
    )
    app["orchestrator"] = DisaggOrchestrator(config)
    logger.info(f"Orchestrator started - Prefill: {config.prefill_url}, Decode: {config.decode_url}")


async def on_cleanup(app: web.Application):
    """Cleanup on shutdown"""
    orchestrator: DisaggOrchestrator = app.get("orchestrator")
    if orchestrator:
        await orchestrator.close()


def create_app(prefill_url: str, decode_url: str) -> web.Application:
    """Create the aiohttp application"""
    app = web.Application()
    app["prefill_url"] = prefill_url
    app["decode_url"] = decode_url
    
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    
    app.router.add_get("/health", handle_health)
    app.router.add_post("/completion", handle_completion)
    app.router.add_post("/v1/chat/completions", handle_chat_completion)
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Disaggregated Prefill Orchestrator")
    parser.add_argument("--port", type=int, default=9000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--prefill-url", type=str, default="http://localhost:8080",
                        help="Prefill server URL")
    parser.add_argument("--decode-url", type=str, default="http://localhost:8081",
                        help="Decode server URL")
    
    args = parser.parse_args()
    
    app = create_app(args.prefill_url, args.decode_url)
    
    logger.info(f"Starting orchestrator on {args.host}:{args.port}")
    logger.info(f"  Prefill server: {args.prefill_url}")
    logger.info(f"  Decode server:  {args.decode_url}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  POST http://{args.host}:{args.port}/completion")
    logger.info(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    logger.info("")
    logger.info('Add "disagg": false to request body to use baseline (single server)')
    
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
