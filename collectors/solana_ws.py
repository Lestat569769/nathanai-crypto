"""
collectors/solana_ws.py
=======================
Subscribes to Solana public RPC WebSocket and filters for pump.fun
new token creation events. No Cloudflare, no paid API — uses free
public RPC or Helius free-tier RPC WebSocket.

Flow:
  1. logsSubscribe → mentions pump.fun program ID
  2. Filter for "Instruction: Create" log entries
  3. getTransaction → parse accounts + instruction data (Borsh decode)
  4. Extract: mint, name, ticker, dev wallet, uri
  5. rugcheck hard skip gate
  6. Emit to on_new_token / on_graduation callbacks

pump.fun program ID: 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P

Account layout for pump.fun Create instruction:
  [0] mint            — new token mint address
  [1] mint_metadata   — Metaplex metadata PDA
  [2] bonding_curve   — bonding curve state account
  [3] assoc_bonding   — associated bonding curve token account
  [4] global          — pump.fun global config
  [5] mpl_program     — Metaplex token metadata program
  [6] system_program
  [7] token_program
  [8] rent
  [9] event_authority
  [10] program
  [11] creator        — dev wallet (fee payer / signer)
"""
import asyncio
import base64
import json
import logging
import os
import struct
import httpx
import base58 as base58lib
import websockets
from typing import Callable, Optional

from collectors.rugcheck import check_token, format_for_prompt

log = logging.getLogger("nathanai.crypto.solana_ws")

PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# Use Helius RPC if API key is set — better rate limits on free tier
# Otherwise fall back to public Solana RPC (still works, lower rate limits)
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")

def _get_ws_url() -> str:
    # Use Helius WebSocket for reliable event delivery.
    # logsSubscribe on public RPC is unreliable — events drop silently.
    # Credit cost: only logsSubscribe notifications, NOT getTransaction
    # (getTransaction uses public RPC = 0 credits).
    # Estimated: ~2000 pump.fun events/hour × 1 credit = 2000/hr = 48K/day
    # = 1.44M/month. Over free limit of 1M.
    # Mitigation: filter events BEFORE counting credits by checking logs
    # for "Instruction:" patterns via WebSocket message itself (no extra call).
    # Real fix: upgrade to Helius paid or use QuickNode free tier.
    # For now: Helius free tier + accept potential monthly limit hit.
    if HELIUS_API_KEY:
        return f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    return "wss://api.mainnet-beta.solana.com"

def _get_rpc_url() -> str:
    # Public RPC — free for getTransaction too.
    # Helius free tier burns 1 credit per getTransaction call.
    # pump.fun creates ~1000+ tokens/day = 30K credits/month minimum.
    # Keep Helius as emergency fallback only (set USE_HELIUS_RPC=true to enable).
    if os.getenv("USE_HELIUS_RPC", "false").lower() == "true" and HELIUS_API_KEY:
        return f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    return "https://api.mainnet-beta.solana.com"


# ── Borsh instruction data parser ─────────────────────────────────────────

def _decode_borsh_string(data: bytes, offset: int) -> tuple[str, int]:
    """Decode a Borsh-encoded string. Returns (value, new_offset)."""
    if offset + 4 > len(data):
        raise ValueError("truncated borsh string length")
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if offset + length > len(data):
        raise ValueError(f"borsh string data truncated: need {length} bytes")
    value = data[offset : offset + length].decode("utf-8", errors="replace")
    return value, offset + length


def _parse_create_instruction(data_str: str) -> Optional[dict]:
    """
    Decode pump.fun Create instruction data (Anchor/Borsh encoding).
    Returns {"name": ..., "symbol": ..., "uri": ...} or None on failure.

    Solana JSON encoding: instruction.data is base58-encoded.
    Instruction layout (after 8-byte Anchor discriminator):
      name   : Borsh string (4-byte LE length + UTF-8 bytes)
      symbol : Borsh string
      uri    : Borsh string
    """
    try:
        # Try base58 first (Solana JSON encoding default)
        try:
            data = base58lib.b58decode(data_str)
        except Exception:
            # Fallback: try base64 (some RPC responses use this)
            data = base64.b64decode(data_str + "==")

        if len(data) < 8:
            return None
        offset = 8  # skip 8-byte Anchor discriminator
        name,   offset = _decode_borsh_string(data, offset)
        symbol, offset = _decode_borsh_string(data, offset)
        uri,    offset = _decode_borsh_string(data, offset)
        return {"name": name, "symbol": symbol, "uri": uri}
    except Exception as e:
        log.debug("instruction parse failed: %s", e)
        return None


# ── Transaction fetcher ───────────────────────────────────────────────────

async def _get_transaction(signature: str) -> Optional[dict]:
    """Fetch full transaction from RPC. Returns parsed tx dict or None."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            signature,
            {
                "encoding":                       "json",
                "maxSupportedTransactionVersion": 0,
                "commitment":                     "confirmed",
            },
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_get_rpc_url(), json=payload)
            resp.raise_for_status()
            result = resp.json().get("result")
            return result
    except Exception as e:
        log.debug("getTransaction failed for %s: %s", signature[:12], e)
        return None


def _extract_token_event(tx: dict, signature: str) -> Optional[dict]:
    """
    Parse a getTransaction response into a token event dict.
    Returns None if this doesn't look like a pump.fun Create tx.
    """
    try:
        msg = tx["transaction"]["message"]
        accounts = msg.get("accountKeys", [])
        instructions = msg.get("instructions", [])

        # Find the pump.fun program instruction
        pf_ix = None
        for ix in instructions:
            prog_idx = ix.get("programIdIndex", -1)
            if prog_idx < len(accounts) and accounts[prog_idx] == PUMPFUN_PROGRAM:
                pf_ix = ix
                break

        if not pf_ix:
            return None

        # Decode instruction data → name, symbol, uri
        data_b64 = pf_ix.get("data", "")
        parsed = _parse_create_instruction(data_b64)
        if not parsed:
            log.debug("could not parse instruction data for %s", signature[:12])
            # Still emit event with empty name/ticker — mint is more important
            parsed = {"name": "", "symbol": "", "uri": ""}

        # Extract accounts by position
        ix_accounts = pf_ix.get("accounts", [])

        def account_at(pos: int) -> str:
            if pos < len(ix_accounts):
                idx = ix_accounts[pos]
                if idx < len(accounts):
                    return accounts[idx]
            return ""

        mint = account_at(0)

        # Dev wallet: try instruction account[11] (creator), fall back to
        # fee payer (accounts[0]) which is always the transaction signer
        dev = account_at(11) or (accounts[0] if accounts else "")

        if not mint:
            return None

        return {
            "signature": signature,
            "mint":      mint,
            "name":      parsed["name"],
            "ticker":    parsed["symbol"],
            "dev":       dev,
            "uri":       parsed["uri"],
        }

    except Exception as e:
        log.debug("_extract_token_event error: %s", e)
        return None


# ── Main listener ─────────────────────────────────────────────────────────

async def listen(
    on_new_token:  Callable[[dict], None],
    on_trade:      Optional[Callable[[dict], None]] = None,
    on_graduation: Optional[Callable[[dict], None]] = None,
):
    """
    Subscribe to Solana RPC WebSocket and emit pump.fun token events.
    Uses free public RPC or Helius free-tier RPC (set HELIUS_API_KEY in .env).

    on_new_token  — called for every new token (post rugcheck gate)
    on_trade      — not implemented via this feed (use DexScreener polling)
    on_graduation — called when migration log detected
    """
    backoff = 5
    max_backoff = 120
    ws_url = _get_ws_url()
    source = "Helius" if HELIUS_API_KEY else "public Solana RPC"

    subscribe_msg = json.dumps({
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "logsSubscribe",
        "params":  [
            {"mentions": [PUMPFUN_PROGRAM]},
            {"commitment": "confirmed"},
        ],
    })

    while True:
        try:
            log.info("Connecting to Solana WebSocket (%s): %s",
                     source, ws_url[:50] + "...")
            async with websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=20,
                max_size=10 * 1024 * 1024,  # 10MB — large transactions
            ) as ws:
                await ws.send(subscribe_msg)
                log.info("Subscribed to pump.fun program logs via %s", source)
                backoff = 5  # reset on successful connect

                async for raw in ws:
                    try:
                        await _handle_message(
                            json.loads(raw),
                            on_new_token,
                            on_graduation,
                        )
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        log.debug("message handling error: %s", e)

        except websockets.ConnectionClosed as e:
            log.warning("WebSocket closed (%s) — retrying in %ds", e.code, backoff)
        except Exception as e:
            log.error("WebSocket error: %s — retrying in %ds", e, backoff)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)


async def _handle_message(
    msg: dict,
    on_new_token: Callable,
    on_graduation: Optional[Callable],
) -> None:
    """Route a WebSocket notification to the appropriate handler."""
    # Subscription confirmation
    if "result" in msg and isinstance(msg["result"], int):
        log.info("Subscription confirmed, id=%d — waiting for events...", msg["result"])
        return

    params = msg.get("params", {})
    value  = params.get("result", {}).get("value", {})
    logs   = value.get("logs", [])
    sig    = value.get("signature", "")

    if not logs or not sig:
        return

    logs_str = " ".join(logs)

    # Debug: print first 3 log lines of every pump.fun tx to learn format
    log.info("PUMPFUN TX sig=%s logs=%s", sig[:12], logs[:3])

    # ── New token creation ────────────────────────────────────────────────
    if "Instruction: Create" in logs_str:
        await _handle_create(sig, on_new_token)

    # ── Graduation (bonding curve filled → Raydium) ───────────────────────
    elif on_graduation and (
        "Instruction: Migrate" in logs_str
        or "migration" in logs_str.lower()
    ):
        log.info("GRADUATION detected: sig=%s", sig[:12])
        import inspect
        if inspect.iscoroutinefunction(on_graduation):
            await on_graduation({"signature": sig, "logs": logs})
        else:
            on_graduation({"signature": sig, "logs": logs})


async def _handle_create(sig: str, on_new_token: Callable) -> None:
    """Fetch + parse a create transaction and emit token event."""
    tx = await _get_transaction(sig)
    if not tx:
        log.debug("no tx data for sig %s", sig[:12])
        return

    event = _extract_token_event(tx, sig)
    if not event:
        return

    mint   = event["mint"]
    name   = event["name"]
    ticker = event["ticker"]
    dev    = event["dev"]

    log.info("NEW TOKEN: %s (%s) mint=%s dev=%s",
             name or "?", ticker or "?", mint[:8], dev[:8] if dev else "?")

    # ── rugcheck hard skip gate ───────────────────────────────────────────
    rc = await check_token(mint)

    token_event = {
        **event,
        "rugcheck":       rc,
        "rugcheck_prompt": format_for_prompt(rc),
        "hard_skip":      rc["hard_skip"],
    }

    if rc["hard_skip"]:
        log.warning("HARD SKIP %s (%s) — rug signals: %s",
                    name or mint[:8], ticker, rc.get("risk_flags", []))
        token_event["skip_reason"] = "rugcheck_hard_skip"

    # Support both sync and async callbacks
    import inspect
    if inspect.iscoroutinefunction(on_new_token):
        await on_new_token(token_event)
    else:
        on_new_token(token_event)
