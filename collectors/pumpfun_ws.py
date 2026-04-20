"""
collectors/pumpfun_ws.py
========================
Subscribes to the pump.fun WebSocket feed and emits new token launch events.

On each new token event:
  1. Parse token metadata (mint, name, ticker, dev wallet)
  2. Run rugcheck HARD SKIP gate immediately
  3. If not hard-skipped → enqueue for LLM evaluation
  4. Write to Neo4j regardless (even skipped tokens are training data)

WebSocket message types:
  - "newCoin"     : new token just launched on bonding curve
  - "tradeCreated": buy or sell on bonding curve
  - "migration"   : token graduated to Raydium (bonding curve filled)
"""
import asyncio
import json
import logging
import websockets
from typing import Callable, Optional

from collectors.rugcheck import check_token, format_for_prompt

log = logging.getLogger("nathanai.crypto.pumpfun_ws")

WS_URL = "wss://frontend-api.pump.fun/ws"

# Browser-like headers — required to pass Cloudflare (HTTP 530 without these)
WS_HEADERS = {
    "Origin":                    "https://pump.fun",
    "Host":                      "frontend-api.pump.fun",
    "User-Agent":                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language":           "en-US,en;q=0.9",
    "Cache-Control":             "no-cache",
    "Pragma":                    "no-cache",
    "Sec-WebSocket-Extensions":  "permessage-deflate; client_max_window_bits",
}


async def listen(
    on_new_token:  Callable[[dict], None],
    on_trade:      Optional[Callable[[dict], None]] = None,
    on_graduation: Optional[Callable[[dict], None]] = None,
):
    """
    Connect to pump.fun WebSocket and dispatch events.

    on_new_token  — called for every new token launch (post rugcheck gate)
    on_trade      — called for every bonding curve trade (optional)
    on_graduation — called when a token graduates to Raydium (optional)

    NOTE: pump.fun uses Cloudflare with TLS fingerprinting (JA3/JA4).
    Headers alone are insufficient. Production fix: Helius Yellowstone gRPC
    (reads new token mints directly from Solana — no pump.fun API needed).
    This implementation uses exponential backoff while that is integrated.
    """
    backoff = 5
    max_backoff = 300  # cap at 5 minutes between retries

    while True:
        try:
            log.info("Connecting to pump.fun WebSocket: %s", WS_URL)
            async with websockets.connect(
                WS_URL,
                additional_headers=WS_HEADERS,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                log.info("pump.fun WebSocket connected")
                backoff = 5  # reset on successful connection
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        _dispatch(msg, on_new_token, on_trade, on_graduation)
                    except json.JSONDecodeError:
                        log.debug("non-JSON message: %s", raw[:80])

        except websockets.exceptions.InvalidStatus as e:
            log.warning(
                "pump.fun rejected connection (HTTP %s) — "
                "Cloudflare TLS fingerprint block. "
                "TODO: switch to Helius Yellowstone gRPC. "
                "Retrying in %ds.",
                e.response.status_code if hasattr(e, "response") else "?",
                backoff,
            )
        except websockets.ConnectionClosed:
            log.warning("pump.fun WebSocket disconnected — retrying in %ds", backoff)
        except Exception as e:
            log.error("WebSocket error: %s — retrying in %ds", e, backoff)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)


def _dispatch(msg, on_new_token, on_trade, on_graduation):
    msg_type = msg.get("txType") or msg.get("type", "")

    if msg_type == "create":
        _handle_new_token(msg, on_new_token)
    elif msg_type in ("buy", "sell") and on_trade:
        on_trade(msg)
    elif msg_type == "migration" and on_graduation:
        log.info("GRADUATION: %s (%s) graduated to Raydium",
                 msg.get("name", "?"), msg.get("mint", "")[:8])
        on_graduation(msg)


def _handle_new_token(msg, on_new_token):
    mint   = msg.get("mint", "")
    name   = msg.get("name", "")
    ticker = msg.get("symbol", "")
    dev    = msg.get("traderPublicKey", "")

    log.info("NEW TOKEN: %s (%s) mint=%s dev=%s", name, ticker, mint[:8], dev[:8])

    # ── rugcheck HARD SKIP gate ───────────────────────────────────────
    # Check before enqueueing for LLM — don't waste inference on traps
    rc = check_token(mint)

    token_event = {
        "mint":     mint,
        "name":     name,
        "ticker":   ticker,
        "dev":      dev,
        "uri":      msg.get("uri", ""),
        "rugcheck": rc,
        "rugcheck_prompt": format_for_prompt(rc),
        "hard_skip": rc["hard_skip"],
    }

    if rc["hard_skip"]:
        log.warning("HARD SKIP %s (%s) — rug signals detected, not evaluating", name, mint[:8])
        # Still pass to on_new_token so it can be written to Neo4j as training data
        token_event["skip_reason"] = "rugcheck_hard_skip"

    on_new_token(token_event)
