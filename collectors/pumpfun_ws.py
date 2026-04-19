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
    """
    log.info("Connecting to pump.fun WebSocket: %s", WS_URL)
    async for ws in websockets.connect(WS_URL, ping_interval=20, ping_timeout=10):
        try:
            log.info("pump.fun WebSocket connected")
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    _dispatch(msg, on_new_token, on_trade, on_graduation)
                except json.JSONDecodeError:
                    log.debug("non-JSON message: %s", raw[:80])
        except websockets.ConnectionClosed:
            log.warning("pump.fun WebSocket disconnected — reconnecting in 3s")
            await asyncio.sleep(3)
        except Exception as e:
            log.error("WebSocket error: %s — reconnecting in 5s", e)
            await asyncio.sleep(5)


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
