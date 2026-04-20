"""
collectors/pumpportal_ws.py
===========================
Subscribes to PumpPortal's free real-time WebSocket API for pump.fun events.
No API key, no credits, no Cloudflare — purpose-built for pump.fun data.

Docs: https://pumpportal.fun/data-api/real-time

Events we subscribe to:
  - subscribeNewToken      : every new token launch (name, ticker, mint, dev)
  - subscribeMigration     : token graduated to PumpSwap/Raydium

After receiving a newToken event, we already have name/ticker/mint/dev
directly in the message — no getTransaction parsing needed.

Usage:
  from collectors.pumpportal_ws import listen
  await listen(on_new_token=..., on_graduation=...)
"""
import asyncio
import json
import logging
from typing import Callable, Optional

import websockets

from collectors.rugcheck import check_token, format_for_prompt

log = logging.getLogger("nathanai.crypto.pumpportal_ws")

WS_URL = "wss://pumpportal.fun/api/data"


async def listen(
    on_new_token:  Callable,
    on_trade:      Optional[Callable] = None,
    on_graduation: Optional[Callable] = None,
):
    """
    Connect to PumpPortal WebSocket and stream pump.fun events.
    Automatically reconnects with exponential backoff on disconnect.

    on_new_token  — called for every new token (post rugcheck gate)
    on_trade      — called for bonding curve trades (optional)
    on_graduation — called when token graduates to PumpSwap (optional)
    """
    backoff = 3
    max_backoff = 60

    while True:
        try:
            log.info("Connecting to PumpPortal WebSocket: %s", WS_URL)
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=15,
                max_size=5 * 1024 * 1024,
            ) as ws:
                log.info("PumpPortal connected — subscribing to events")
                backoff = 3

                # Subscribe to new token launches
                await ws.send(json.dumps({"method": "subscribeNewToken"}))

                # Subscribe to migrations (graduations) if handler provided
                if on_graduation:
                    await ws.send(json.dumps({"method": "subscribeMigration"}))

                log.info("Subscribed. Listening for pump.fun token launches...")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        await _dispatch(msg, on_new_token, on_trade, on_graduation)
                    except json.JSONDecodeError:
                        log.debug("non-JSON message: %s", str(raw)[:60])
                    except Exception as e:
                        log.debug("dispatch error: %s", e)

        except websockets.ConnectionClosed as e:
            log.warning("PumpPortal disconnected (%s) — retrying in %ds", e.code, backoff)
        except Exception as e:
            log.error("PumpPortal error: %s — retrying in %ds", e, backoff)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)


async def _dispatch(msg: dict, on_new_token, on_trade, on_graduation):
    """Route a PumpPortal message to the correct handler."""
    txtype = msg.get("txType", "")

    if txtype == "create":
        await _handle_new_token(msg, on_new_token)

    elif txtype in ("buy", "sell") and on_trade:
        import inspect
        if inspect.iscoroutinefunction(on_trade):
            await on_trade(msg)
        else:
            on_trade(msg)

    elif txtype == "migrate" and on_graduation:
        mint = msg.get("mint", "")
        name = msg.get("name", "")
        log.info("GRADUATION: %s mint=%s", name or "?", mint[:8])
        import inspect
        if inspect.iscoroutinefunction(on_graduation):
            await on_graduation(msg)
        else:
            on_graduation(msg)


async def _handle_new_token(msg: dict, on_new_token: Callable):
    """
    Process a new token creation event from PumpPortal.
    PumpPortal provides name/ticker/mint/dev directly — no RPC parsing needed.

    PumpPortal newToken message fields:
      mint, name, symbol, uri, traderPublicKey (dev), signature,
      initialBuy, solAmount, marketCapSol, bondingCurveKey, vSolInBondingCurve,
      vTokensInBondingCurve, tokenTotalSupply, timestamp
    """
    mint   = msg.get("mint", "")
    name   = msg.get("name", "") or ""
    ticker = msg.get("symbol", "") or ""
    dev    = msg.get("traderPublicKey", "") or ""
    uri    = msg.get("uri", "") or ""

    if not mint:
        return

    log.info("NEW TOKEN: %s (%s) mint=%s dev=%s",
             name or "?", ticker or "?", mint[:8], dev[:8] if dev else "?")

    # ── rugcheck hard skip gate ───────────────────────────────────────────
    rc = check_token(mint)

    token_event = {
        "mint":              mint,
        "name":              name,
        "ticker":            ticker,
        "dev":               dev,
        "uri":               uri,
        "signature":         msg.get("signature", ""),
        "initial_buy_sol":   msg.get("solAmount", 0.0),
        "market_cap_sol":    msg.get("marketCapSol", 0.0),
        "rugcheck":          rc,
        "rugcheck_prompt":   format_for_prompt(rc),
        "hard_skip":         rc["hard_skip"],
    }

    if rc["hard_skip"]:
        log.warning("HARD SKIP %s (%s) — rug signals: %s",
                    name or mint[:8], ticker, rc.get("risk_flags", []))
        token_event["skip_reason"] = "rugcheck_hard_skip"

    import inspect
    if inspect.iscoroutinefunction(on_new_token):
        await on_new_token(token_event)
    else:
        on_new_token(token_event)
