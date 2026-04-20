"""
adapter/inference.py
====================
Call qwen3:8b via Ollama to produce BUY/SKIP decisions.

Model config:
  - qwen3:8b in no-think mode (fast, deterministic JSON)
  - temperature 0.1  — near-deterministic for structured output
  - num_predict 512  — enough for decision + 2-sentence reasoning
  - timeout 60s      — GPU inference; bump to 120s on CPU-only

No-think mode:
  Qwen3 supports /no_think as a message prefix to suppress the <think> block.
  We prepend it to the user message so the model skips chain-of-thought and
  outputs the JSON directly. This halves latency on GPU.
  Ollama also accepts {"think": false} in options — we send both for safety.

Retry policy:
  2 retries with 3s backoff on timeout/connection error.
  Returns None on persistent failure so the caller can skip the token.

Usage:
  from adapter.inference import call_primary, call_validator
  raw = await call_primary(messages)        # list of {role, content} dicts
  raw = await call_validator(messages)      # same API, uses gate model
"""
import asyncio
import logging
import os
from typing import Optional

import httpx

log = logging.getLogger("nathanai.crypto.adapter.inference")

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://host.docker.internal:11434")
PRIMARY_MODEL    = os.getenv("PRIMARY_MODEL",    "qwen3:8b")
GATE_MODEL       = os.getenv("GATE_MODEL",       "qwen3:4b")

TIMEOUT          = 60.0   # seconds — raise to 120 for CPU-only hosts
MAX_RETRIES      = 2
RETRY_BACKOFF    = 3.0    # seconds between retries

# Ollama generation options
_PRIMARY_OPTIONS = {
    "temperature":   0.1,    # near-deterministic JSON output
    "num_predict":   1024,   # decision + 2-sentence reasoning (512 occasionally truncates)
    "top_p":         0.9,
    "repeat_penalty": 1.1,
    "think":         False,  # Qwen3 no-think mode (newer Ollama versions)
}

_GATE_OPTIONS = {
    "temperature":   0.1,
    "num_predict":   256,    # validation is short
    "top_p":         0.9,
    "repeat_penalty": 1.1,
    "think":         False,
}


# ── No-think prefix injection ─────────────────────────────────────────────────

def _inject_no_think(messages: list[dict]) -> list[dict]:
    """
    Prepend /no_think to the last user message.
    Works across all Ollama versions — the Qwen3 tokenizer converts it to the
    suppress-thinking control token regardless of the `think` option support.
    """
    msgs = [m.copy() for m in messages]
    for i in reversed(range(len(msgs))):
        if msgs[i]["role"] == "user":
            msgs[i]["content"] = "/no_think\n\n" + msgs[i]["content"]
            break
    return msgs


# ── Core Ollama call ──────────────────────────────────────────────────────────

async def _call_ollama(
    model:    str,
    messages: list[dict],
    options:  dict,
) -> Optional[str]:
    """
    POST to /api/chat and return the assistant message content string.
    Returns None on error after all retries exhausted.
    """
    url     = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  options,
    }

    for attempt in range(1, MAX_RETRIES + 2):   # +2 so range covers MAX_RETRIES retries
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                if content:
                    log.debug(
                        "inference: %s responded (%d chars) eval_duration=%.1fs",
                        model,
                        len(content),
                        data.get("eval_duration", 0) / 1e9,
                    )
                    return content
                log.warning("inference: empty content from %s on attempt %d", model, attempt)

        except httpx.TimeoutException:
            log.warning("inference: timeout from %s (attempt %d/%d)", model, attempt, MAX_RETRIES + 1)
        except httpx.HTTPStatusError as e:
            log.error("inference: HTTP %d from %s: %s", e.response.status_code, model, e)
            return None   # don't retry HTTP errors
        except Exception as e:
            log.error("inference: error from %s (attempt %d): %s", model, attempt, e)

        if attempt <= MAX_RETRIES:
            await asyncio.sleep(RETRY_BACKOFF)

    log.error("inference: %s failed after %d attempts", model, MAX_RETRIES + 1)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

async def call_primary(messages: list[dict]) -> Optional[str]:
    """
    Call the primary decision model (qwen3:8b) in no-think mode.

    Parameters
    ----------
    messages : output of adapter.prompt_builder.build_messages()

    Returns
    -------
    Raw assistant response string, or None on failure.
    Pass to adapter.decision_parser.parse() to get a ParsedDecision.
    """
    msgs = _inject_no_think(messages)
    log.info("inference: calling %s for decision...", PRIMARY_MODEL)
    return await _call_ollama(PRIMARY_MODEL, msgs, _PRIMARY_OPTIONS)


async def call_validator(messages: list[dict]) -> Optional[str]:
    """
    Call the hallucination gate model (qwen3:4b) in no-think mode.

    Parameters
    ----------
    messages : output of adapter.validator.build_validation_messages()

    Returns
    -------
    Raw assistant response string, or None on failure.
    Pass to adapter.decision_parser.parse_validator_response() to get
    {"valid": bool, "reason": str}.
    """
    msgs = _inject_no_think(messages)
    log.info("inference: calling %s for validation...", GATE_MODEL)
    return await _call_ollama(GATE_MODEL, msgs, _GATE_OPTIONS)


async def check_ollama_available() -> bool:
    """
    Quick health check — returns True if Ollama is reachable and both
    required models are listed.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = {m["name"] for m in resp.json().get("models", [])}
            ok = PRIMARY_MODEL in models and GATE_MODEL in models
            if not ok:
                missing = {PRIMARY_MODEL, GATE_MODEL} - models
                log.warning("inference: models not available in Ollama: %s", missing)
            return ok
    except Exception as e:
        log.error("inference: Ollama not reachable at %s: %s", OLLAMA_BASE_URL, e)
        return False
