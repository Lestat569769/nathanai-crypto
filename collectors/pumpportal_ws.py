"""
collectors/pumpportal_ws.py
===========================
Subscribes to PumpPortal's free real-time WebSocket API for pump.fun events.
No API key, no credits, no Cloudflare — purpose-built for pump.fun data.

Docs: https://pumpportal.fun/data-api/real-time

Events we subscribe to:
  - subscribeNewToken      : every new token launch (name, ticker, mint, dev)
  - subscribeMigration     : token graduated to PumpSwap/Raydium
  - subscribeTokenTrade    : bonding curve trades for non-skipped tokens

After receiving a newToken event we already have name/ticker/mint/dev/solAmount/
marketCapSol/vTokensInBondingCurve/vSolInBondingCurve directly in the message.
We also fetch the token's metadata URI (twitter, telegram, website, description)
asynchronously so the main loop is never blocked.

IMPORTANT: PumpPortal bans clients that open multiple WebSocket connections.
All subscriptions (newToken, migration, tokenTrade) share one connection.

Usage:
  from collectors.pumpportal_ws import listen
  await listen(on_new_token=..., on_graduation=..., on_trade=...)
"""
import asyncio
import json
import logging
from typing import Callable, Optional

import httpx
import websockets

from collectors.rugcheck import check_token, format_for_prompt

log = logging.getLogger("nathanai.crypto.pumpportal_ws")

WS_URL = "wss://pumpportal.fun/api/data"

# How long to wait for the metadata URI fetch (pump.fun IPFS can be slow)
URI_FETCH_TIMEOUT = 8.0


# ── URI metadata fetcher ──────────────────────────────────────────────────────

async def _fetch_uri_metadata(uri: str) -> dict:
    """
    Fetch token metadata from the pump.fun URI (IPFS or pump.fun CDN).
    Returns a dict with social links and description, or empty dict on failure.

    Typical URI JSON structure:
      {
        "name": "Token Name",
        "symbol": "TKR",
        "description": "...",
        "image": "https://...",
        "showName": true,
        "twitter": "https://twitter.com/...",
        "telegram": "https://t.me/...",
        "website": "https://...",
        "createdOn": "https://pump.fun"
      }
    """
    if not uri:
        return {}
    try:
        async with httpx.AsyncClient(timeout=URI_FETCH_TIMEOUT) as client:
            resp = await client.get(uri)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "description": data.get("description", ""),
                    "image":       data.get("image", ""),
                    "twitter":     data.get("twitter", ""),
                    "telegram":    data.get("telegram", ""),
                    "website":     data.get("website", ""),
                    "created_on":  data.get("createdOn", ""),
                }
    except Exception as e:
        log.debug("URI fetch failed (%s): %s", uri[:60], e)
    return {}


# ── Main listener ─────────────────────────────────────────────────────────────

async def listen(
    on_new_token:  Callable,
    on_trade:      Optional[Callable] = None,
    on_graduation: Optional[Callable] = None,
):
    """
    Connect to PumpPortal WebSocket and stream pump.fun events.
    Automatically reconnects with exponential backoff on disconnect.

    on_new_token  — called for every new token (post rugcheck gate).
                    Event dict includes rugcheck result + all PumpPortal fields.
    on_trade      — called for bonding curve buy/sell events. We subscribe to
                    trades for every token that passes the rugcheck hard-skip.
    on_graduation — called when a token graduates to PumpSwap/Raydium.

    Token-trade subscriptions are sent on the SAME connection to avoid bans.
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
                        await _dispatch(msg, ws, on_new_token, on_trade, on_graduation)
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


async def _dispatch(msg: dict, ws, on_new_token, on_trade, on_graduation):
    """Route a PumpPortal message to the correct handler."""
    txtype = msg.get("txType", "")

    if txtype == "create":
        await _handle_new_token(msg, ws, on_new_token, on_trade)

    elif txtype in ("buy", "sell") and on_trade:
        await _call(on_trade, _enrich_trade(msg))

    elif txtype == "migrate" and on_graduation:
        _log_migration_fields(msg)
        await _call(on_graduation, msg)


def _enrich_trade(msg: dict) -> dict:
    """
    Normalise a PumpPortal trade event.

    PumpPortal trade fields (buy/sell):
      mint, name, symbol, traderPublicKey, txType, tokenAmount,
      solAmount, newTokenBalance, bondingCurveKey,
      vTokensInBondingCurve, vSolInBondingCurve, marketCapSol,
      signature, timestamp (optional)
    """
    return {
        "mint":                    msg.get("mint", ""),
        "name":                    msg.get("name", "") or "",
        "ticker":                  msg.get("symbol", "") or "",
        "trader":                  msg.get("traderPublicKey", "") or "",
        "side":                    msg.get("txType", ""),        # "buy" | "sell"
        "token_amount":            msg.get("tokenAmount", 0),
        "sol_amount":              msg.get("solAmount", 0.0),
        "new_token_balance":       msg.get("newTokenBalance", 0),
        "bonding_curve_key":       msg.get("bondingCurveKey", ""),
        "v_tokens_in_curve":       msg.get("vTokensInBondingCurve", 0),
        "v_sol_in_curve":          msg.get("vSolInBondingCurve", 0.0),
        "market_cap_sol":          msg.get("marketCapSol", 0.0),
        "signature":               msg.get("signature", ""),
    }


def _log_migration_fields(msg: dict):
    """
    Log all fields in a migration event so we can discover the schema.
    PumpPortal has not published the migration event schema — we log it raw.
    """
    mint = msg.get("mint", "")
    name = msg.get("name", "")
    log.info("GRADUATION: %s mint=%s | full_event=%s",
             name or "?", mint[:8] if mint else "?", json.dumps(msg))


async def _handle_new_token(msg: dict, ws, on_new_token: Callable, on_trade):
    """
    Process a new token creation event from PumpPortal.

    PumpPortal provides all fields inline — no getTransaction needed:
      mint, name, symbol, uri, traderPublicKey (dev), signature,
      initialBuy, solAmount, marketCapSol, bondingCurveKey,
      vSolInBondingCurve, vTokensInBondingCurve, tokenTotalSupply
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

    # ── rugcheck hard skip gate ───────────────────────────────────────────────
    rc = check_token(mint)

    token_event = {
        # Identity
        "mint":                    mint,
        "name":                    name,
        "ticker":                  ticker,
        "dev":                     dev,
        "uri":                     uri,
        "signature":               msg.get("signature", ""),
        # Market data at launch
        "initial_buy_sol":         msg.get("solAmount", 0.0),
        "market_cap_sol":          msg.get("marketCapSol", 0.0),
        "bonding_curve_key":       msg.get("bondingCurveKey", ""),
        "v_sol_in_curve":          msg.get("vSolInBondingCurve", 0.0),
        "v_tokens_in_curve":       msg.get("vTokensInBondingCurve", 0),
        "token_total_supply":      msg.get("tokenTotalSupply", 0),
        # Social metadata — populated async below (may be empty initially)
        "description":             "",
        "twitter":                 "",
        "telegram":                "",
        "website":                 "",
        # Rugcheck
        "rugcheck":                rc,
        "rugcheck_prompt":         format_for_prompt(rc),
        "hard_skip":               rc["hard_skip"],
    }

    if rc["hard_skip"]:
        log.warning("HARD SKIP %s (%s) — rug signals: %s",
                    name or mint[:8], ticker, rc.get("risk_flags", []))
        token_event["skip_reason"] = "rugcheck_hard_skip"
    else:
        # Subscribe to bonding curve trades for this token (same WS connection)
        if on_trade:
            try:
                await ws.send(json.dumps({
                    "method": "subscribeTokenTrade",
                    "keys":   [mint],
                }))
                log.debug("Subscribed to trades for %s (%s)", ticker or "?", mint[:8])
            except Exception as e:
                log.debug("subscribeTokenTrade failed for %s: %s", mint[:8], e)

    # Emit the event immediately — don't block on URI fetch
    await _call(on_new_token, token_event)

    # Fetch metadata URI in the background so it doesn't delay the pipeline
    if uri:
        asyncio.create_task(_fetch_and_update_metadata(uri, token_event, on_new_token))


async def _fetch_and_update_metadata(uri: str, original_event: dict, on_new_token: Callable):
    """
    Fetch URI metadata and emit an updated event with social fields populated.
    Called as a background task so the main pipeline is not delayed.
    """
    meta = await _fetch_uri_metadata(uri)
    if not meta:
        return

    updated = {**original_event, **meta, "_metadata_update": True}
    log.debug("URI metadata fetched for %s: twitter=%s telegram=%s website=%s",
              original_event.get("mint", "")[:8],
              bool(meta.get("twitter")),
              bool(meta.get("telegram")),
              bool(meta.get("website")))
    await _call(on_new_token, updated)


async def _call(fn: Callable, arg):
    """Invoke a callback that may be sync or async."""
    import inspect
    if inspect.iscoroutinefunction(fn):
        await fn(arg)
    else:
        fn(arg)
