#!/usr/bin/env python3
"""
Smart Disaggregated Prefill Orchestrator

This orchestrator properly manages client sessions and slot persistence
to enable efficient KV cache transfer between prefill and decode servers.

Key insight: After restore, we DON'T re-send the prompt. Instead, we track
the token state ourselves and only send NEW tokens for continuation.

Architecture:
- Prefill Server (8080): Handles initial prompt processing
- Decode Server (8081): Handles token generation
- Orchestrator: Manages sessions, slots, and KV cache transfer

Usage:
    python3 smart_orchestrator.py --port 9000
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
import aiohttp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SessionState(Enum):
    EMPTY = "empty"
    PREFILLED = "prefilled"      # KV cache on prefill server
    TRANSFERRED = "transferred"  # KV cache transferred to decode server
    GENERATING = "generating"    # Currently generating


@dataclass
class Session:
    """Tracks a client session's state"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    
    # Token tracking
    tokens: List[int] = field(default_factory=list)
    n_tokens: int = 0
    tokens_cached: int = 0  # Actual tokens in cache (may include generated)
    n_saved: int = 0        # Tokens saved to file
    
    # Slot assignments
    prefill_slot: int = -1
    decode_slot: int = -1
    
    # State
    state: SessionState = SessionState.EMPTY
    kv_filename: str = ""
    
    # Metrics
    prefill_time_ms: float = 0
    transfer_time_ms: float = 0
    total_generated: int = 0
    
    # Prefill server timings (captured from prefill response)
    prefill_prompt_ms: float = 0      # Time to process prompt on prefill server
    prefill_prompt_tps: float = 0     # Prompt tokens/sec on prefill server
    prefill_tokens_evaluated: int = 0  # Tokens actually evaluated (not cached)


@dataclass 
class ServerConfig:
    prefill_url: str = "http://localhost:8080"
    decode_url: str = "http://localhost:8081"
    kv_cache_dir: str = "/tmp/llama_kv_cache"
    
    # Slot counts - should match server -np settings
    prefill_slots: int = 2  # Prefill server slots
    decode_slots: int = 8   # Decode server slots


class SlotManager:
    """Manages slot allocation across servers"""
    
    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self.slots: Dict[int, Optional[str]] = {i: None for i in range(n_slots)}
        self._lock = asyncio.Lock()
        self._slot_available = asyncio.Event()
        self._slot_available.set()  # Initially slots are available
    
    async def allocate(self, session_id: str, timeout: float = 30.0) -> int:
        """
        Allocate a slot for a session.
        Waits up to timeout seconds if no slot is immediately available.
        Returns slot_id or -1 if timeout.
        """
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            async with self._lock:
                # First check if session already has a slot
                for slot_id, owner in self.slots.items():
                    if owner == session_id:
                        return slot_id
                
                # Find free slot
                for slot_id, owner in self.slots.items():
                    if owner is None:
                        self.slots[slot_id] = session_id
                        # Check if all slots are now taken
                        if all(v is not None for v in self.slots.values()):
                            self._slot_available.clear()
                        return slot_id
            
            # No slot available, wait a bit
            try:
                await asyncio.wait_for(self._slot_available.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass  # Continue loop and try again
        
        return -1  # Timeout
    
    async def release(self, session_id: str):
        """Release all slots owned by a session"""
        async with self._lock:
            for slot_id, owner in self.slots.items():
                if owner == session_id:
                    self.slots[slot_id] = None
            # Signal that a slot is now available
            self._slot_available.set()
    
    async def get_slot(self, session_id: str) -> int:
        """Get the slot assigned to a session, or -1 if none"""
        async with self._lock:
            for slot_id, owner in self.slots.items():
                if owner == session_id:
                    return slot_id
            return -1


class SmartOrchestrator:
    """
    Orchestrator that properly manages disaggregated prefill/decode.
    
    Key innovation: We track token state ourselves, so after KV cache transfer,
    we know exactly what's in the decode server's slot without relying on
    the broken cache_prompt mechanism.
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.sessions: Dict[str, Session] = {}
        self.prefill_slots = SlotManager(config.prefill_slots)
        self.decode_slots = SlotManager(config.decode_slots)
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def get_http_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self._session
    
    async def close(self):
        if self._session:
            await self._session.close()
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one"""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        new_id = session_id or str(uuid.uuid4())[:8]
        session = Session(session_id=new_id)
        self.sessions[new_id] = session
        return session
    
    async def tokenize(self, text: str, add_special: bool = True) -> List[int]:
        """Tokenize text using prefill server"""
        http = await self.get_http_session()
        async with http.post(
            f"{self.config.prefill_url}/tokenize",
            json={"content": text, "add_special": add_special}
        ) as resp:
            result = await resp.json()
            return result.get("tokens", [])
    
    async def prefill(self, session: Session, prompt: str) -> Dict[str, Any]:
        """
        Prefill a prompt on the prefill server.
        Returns prefill result with timing info.
        
        We allocate a specific prefill slot to avoid race conditions during save.
        """
        http = await self.get_http_session()
        
        # Allocate prefill slot (waits if none available)
        slot_id = await self.prefill_slots.allocate(session.session_id, timeout=60.0)
        if slot_id < 0:
            raise Exception("No prefill slots available (timeout)")
        session.prefill_slot = slot_id
        
        # Tokenize first to track tokens
        tokens = await self.tokenize(prompt)
        session.tokens = tokens
        session.n_tokens = len(tokens)
        
        logger.info(f"[{session.session_id}] Prefilling {len(tokens)} tokens on slot {slot_id}")
        
        # Prefill (n_predict=0 means just process prompt, but may generate 1 token)
        t_start = time.perf_counter()
        async with http.post(
            f"{self.config.prefill_url}/completion",
            json={
                "prompt": tokens,  # Send as token IDs
                "n_predict": 0,
                "id_slot": slot_id,  # Use our allocated slot
                "return_tokens": True,  # Get generated token IDs
            }
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Prefill failed: {await resp.text()}")
            result = await resp.json()
        
        t_end = time.perf_counter()
        session.prefill_time_ms = (t_end - t_start) * 1000
        session.state = SessionState.PREFILLED
        
        # Capture prefill server's timing metrics
        prefill_timings = result.get("timings", {})
        session.prefill_prompt_ms = prefill_timings.get("prompt_ms", 0)
        session.prefill_prompt_tps = prefill_timings.get("prompt_per_second", 0)
        session.prefill_tokens_evaluated = prefill_timings.get("prompt_n", 0)
        
        # Track actual tokens cached (includes any generated token)
        session.tokens_cached = result.get("tokens_cached", len(tokens))
        
        # If any tokens were generated, append them to our token list
        generated_token_ids = result.get("tokens", [])
        if generated_token_ids:
            session.tokens = tokens + generated_token_ids
            logger.info(f"[{session.session_id}] Generated {len(generated_token_ids)} extra tokens during prefill")
        
        logger.info(f"[{session.session_id}] Prefill complete: {session.prefill_time_ms:.1f}ms, "
                   f"slot={session.prefill_slot}, tokens_cached={session.tokens_cached}, "
                   f"prefill_tps={session.prefill_prompt_tps:.1f}")
        
        return result
    
    async def transfer_kv_cache(self, session: Session) -> Dict[str, Any]:
        """
        Transfer KV cache from prefill server to decode server.
        """
        http = await self.get_http_session()
        
        if session.state != SessionState.PREFILLED:
            raise Exception(f"Cannot transfer: session state is {session.state}")
        
        # Generate unique filename
        session.kv_filename = f"kv_{session.session_id}_{int(time.time())}.bin"
        
        # Allocate decode slot
        decode_slot = await self.decode_slots.allocate(session.session_id)
        if decode_slot < 0:
            raise Exception("No decode slots available")
        session.decode_slot = decode_slot
        
        t_start = time.perf_counter()
        
        # Save from prefill server
        async with http.post(
            f"{self.config.prefill_url}/slots/{session.prefill_slot}?action=save",
            json={"filename": session.kv_filename}
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Save failed: {await resp.text()}")
            save_result = await resp.json()
        
        # Track how many tokens were saved
        session.n_saved = save_result.get("n_saved", 0)
        logger.info(f"[{session.session_id}] Saved {session.n_saved} tokens")
        
        # Restore to decode server
        async with http.post(
            f"{self.config.decode_url}/slots/{decode_slot}?action=restore",
            json={"filename": session.kv_filename}
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Restore failed: {await resp.text()}")
            restore_result = await resp.json()
        
        t_end = time.perf_counter()
        session.transfer_time_ms = (t_end - t_start) * 1000
        session.state = SessionState.TRANSFERRED
        
        # Release prefill slot now that we've saved
        await self.prefill_slots.release(session.session_id)
        
        logger.info(f"[{session.session_id}] KV transfer complete: {session.transfer_time_ms:.1f}ms, "
                   f"decode_slot={decode_slot}")
        
        return {
            "save": save_result,
            "restore": restore_result,
            "transfer_time_ms": session.transfer_time_ms
        }
    
    async def generate(
        self, 
        session: Session, 
        n_predict: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate tokens from the decode server.
        
        KEY INSIGHT: We send the EXACT same tokens that were saved (including
        any generated during prefill). This ensures get_common_prefix matches
        and the server can reuse the KV cache.
        
        The tokens to send = original prompt + any tokens generated during prefill
        """
        http = await self.get_http_session()
        
        if session.state not in [SessionState.TRANSFERRED, SessionState.GENERATING]:
            raise Exception(f"Cannot generate: session state is {session.state}")
        
        session.state = SessionState.GENERATING
        
        # Use tokens that match what was saved (original + generated during prefill)
        tokens_to_send = session.tokens
        
        logger.info(f"[{session.session_id}] Generating {n_predict} tokens on slot {session.decode_slot}, "
                   f"sending {len(tokens_to_send)} tokens (saved={session.n_saved})")
        
        t_start = time.perf_counter()
        async with http.post(
            f"{self.config.decode_url}/completion",
            json={
                "prompt": tokens_to_send,  # Must match saved tokens exactly!
                "n_predict": n_predict,
                "temperature": temperature,
                "id_slot": session.decode_slot,
                "cache_prompt": True,
                **kwargs
            }
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Generate failed: {await resp.text()}")
            result = await resp.json()
        
        t_end = time.perf_counter()
        decode_time_ms = (t_end - t_start) * 1000
        
        # Update session state
        generated_tokens = result.get("tokens_predicted", 0)
        session.total_generated += generated_tokens
        
        # Check if cache was actually used
        timings = result.get("timings", {})
        cache_n = timings.get("cache_n", 0)
        prompt_n = timings.get("prompt_n", 0)
        
        # Add metrics
        # Note: For disagg, prompt TPS comes from PREFILL server (where real work happens)
        # The decode server's prompt_n should be ~0-1 if cache hit worked
        result["disagg_metrics"] = {
            "session_id": session.session_id,
            "prefill_time_ms": session.prefill_time_ms,
            "transfer_time_ms": session.transfer_time_ms,
            "decode_time_ms": decode_time_ms,
            "total_time_ms": session.prefill_time_ms + session.transfer_time_ms + decode_time_ms,
            "n_prompt_tokens": session.n_tokens,
            "tokens_evaluated": session.n_tokens,
            "n_generated_tokens": generated_tokens,
            "cache_n": cache_n,  # Tokens reused from cache on decode server
            "prompt_n": prompt_n,  # Tokens processed on decode server (should be ~0 if cache hit)
            # Prefill server metrics (where real prompt processing happens)
            "prefill_prompt_ms": session.prefill_prompt_ms,
            "prefill_prompt_tps": session.prefill_prompt_tps,
            "prefill_tokens_evaluated": session.prefill_tokens_evaluated,
            # Decode server metrics (for comparison)
            "decode_prompt_tps": timings.get("prompt_per_second", 0),
            "decode_tps": timings.get("predicted_per_second", 0),
        }
        
        cache_status = "✓ CACHE HIT" if cache_n > 0 else "✗ cache miss"
        logger.info(f"[{session.session_id}] Generated {generated_tokens} tokens in {decode_time_ms:.1f}ms "
                   f"({cache_status}: cache_n={cache_n}, prompt_n={prompt_n})")
        
        return result
    
    async def disagg_completion(
        self,
        prompt: str,
        n_predict: int = 100,
        session_id: Optional[str] = None,
        keep_session: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Full disaggregated completion pipeline:
        1. Prefill on prefill server
        2. Transfer KV cache to decode server  
        3. Generate on decode server
        
        Args:
            keep_session: If False (default), release the decode slot after completion
                         to allow other requests to use it. Set True for multi-turn.
        """
        session = self.get_or_create_session(session_id)
        
        try:
            # Step 1: Prefill
            await self.prefill(session, prompt)
            
            # Step 2: Transfer (this releases prefill slot after save)
            await self.transfer_kv_cache(session)
            
            # Step 3: Generate
            result = await self.generate(session, n_predict=n_predict, **kwargs)
            
            return result
        
        except Exception:
            # On error, make sure to release prefill slot if still held
            await self.prefill_slots.release(session.session_id)
            raise
        
        finally:
            # Release decode slot for reuse (unless keeping session for multi-turn)
            if not keep_session:
                await self.decode_slots.release(session.session_id)
                if session.session_id in self.sessions:
                    del self.sessions[session.session_id]
    
    async def baseline_completion(
        self,
        prompt: str,
        n_predict: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """Baseline single-server completion for comparison"""
        http = await self.get_http_session()
        
        t_start = time.perf_counter()
        async with http.post(
            f"{self.config.prefill_url}/completion",
            json={
                "prompt": prompt,
                "n_predict": n_predict,
                **kwargs
            }
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Completion failed: {await resp.text()}")
            result = await resp.json()
        
        t_end = time.perf_counter()
        result["baseline_time_ms"] = (t_end - t_start) * 1000
        
        return result
    
    async def continue_session(
        self,
        session_id: str,
        additional_prompt: str = "",
        n_predict: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Continue generating from an existing session.
        If additional_prompt is provided, append it to the context.
        """
        if session_id not in self.sessions:
            raise Exception(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        if additional_prompt:
            # Tokenize additional prompt (without BOS)
            new_tokens = await self.tokenize(additional_prompt, add_special=False)
            session.tokens.extend(new_tokens)
            session.n_tokens = len(session.tokens)
        
        return await self.generate(session, n_predict=n_predict, **kwargs)


# ============= HTTP Server =============

async def handle_completion(request: web.Request) -> web.Response:
    """Handle /completion endpoint"""
    orchestrator: SmartOrchestrator = request.app["orchestrator"]
    
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    
    prompt = data.get("prompt", "")
    if not prompt:
        return web.json_response({"error": "No prompt provided"}, status=400)
    
    n_predict = data.get("n_predict", 100)
    session_id = data.get("session_id")
    use_disagg = data.get("disagg", True)
    
    try:
        if use_disagg:
            result = await orchestrator.disagg_completion(
                prompt, 
                n_predict=n_predict,
                session_id=session_id
            )
        else:
            result = await orchestrator.baseline_completion(prompt, n_predict=n_predict)
        
        return web.json_response(result)
    
    except Exception as e:
        logger.exception("Request failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_continue(request: web.Request) -> web.Response:
    """Handle /continue endpoint for existing sessions"""
    orchestrator: SmartOrchestrator = request.app["orchestrator"]
    
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    
    session_id = data.get("session_id")
    if not session_id:
        return web.json_response({"error": "No session_id provided"}, status=400)
    
    try:
        result = await orchestrator.continue_session(
            session_id,
            additional_prompt=data.get("prompt", ""),
            n_predict=data.get("n_predict", 100)
        )
        return web.json_response(result)
    
    except Exception as e:
        logger.exception("Continue failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_sessions(request: web.Request) -> web.Response:
    """List active sessions"""
    orchestrator: SmartOrchestrator = request.app["orchestrator"]
    
    sessions = []
    for sid, session in orchestrator.sessions.items():
        sessions.append({
            "session_id": sid,
            "state": session.state.value,
            "n_tokens": session.n_tokens,
            "decode_slot": session.decode_slot,
            "total_generated": session.total_generated,
        })
    
    return web.json_response({"sessions": sessions})


async def handle_clear_sessions(request: web.Request) -> web.Response:
    """Clear all sessions and release slots"""
    orchestrator: SmartOrchestrator = request.app["orchestrator"]
    
    # Release all slots
    for sid in list(orchestrator.sessions.keys()):
        await orchestrator.prefill_slots.release(sid)
        await orchestrator.decode_slots.release(sid)
    
    count = len(orchestrator.sessions)
    orchestrator.sessions.clear()
    
    logger.info(f"Cleared {count} sessions")
    return web.json_response({"cleared": count})


async def handle_health(request: web.Request) -> web.Response:
    """Health check"""
    return web.json_response({"status": "ok"})


async def on_startup(app: web.Application):
    config = ServerConfig(
        prefill_url=app["prefill_url"],
        decode_url=app["decode_url"],
    )
    app["orchestrator"] = SmartOrchestrator(config)
    logger.info(f"Smart Orchestrator started")
    logger.info(f"  Prefill: {config.prefill_url}")
    logger.info(f"  Decode:  {config.decode_url}")


async def on_cleanup(app: web.Application):
    orchestrator = app.get("orchestrator")
    if orchestrator:
        await orchestrator.close()


def create_app(prefill_url: str, decode_url: str) -> web.Application:
    app = web.Application()
    app["prefill_url"] = prefill_url
    app["decode_url"] = decode_url
    
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    
    app.router.add_get("/health", handle_health)
    app.router.add_post("/completion", handle_completion)
    app.router.add_post("/continue", handle_continue)
    app.router.add_get("/sessions", handle_sessions)
    app.router.add_post("/clear", handle_clear_sessions)
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Smart Disaggregated Orchestrator")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--prefill-url", type=str, default="http://localhost:8080")
    parser.add_argument("--decode-url", type=str, default="http://localhost:8081")
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║         SMART DISAGGREGATED PREFILL ORCHESTRATOR          ║
╠═══════════════════════════════════════════════════════════╣
║  This orchestrator manages client sessions and KV cache   ║
║  transfer between prefill and decode servers.             ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    app = create_app(args.prefill_url, args.decode_url)
    
    logger.info(f"Starting on {args.host}:{args.port}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  POST /completion     - Disaggregated completion")
    logger.info(f"  POST /continue       - Continue existing session")
    logger.info(f"  GET  /sessions       - List active sessions")
    logger.info("")
    logger.info('Add "disagg": false to use baseline (single server)')
    
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
