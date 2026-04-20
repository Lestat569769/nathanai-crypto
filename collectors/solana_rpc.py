"""
collectors/solana_rpc.py
========================
Dev wallet history and reputation lookup via Solana public RPC + rugcheck wallet API.

For each new token, we look up the dev wallet to answer:
  - How many tokens has this wallet launched before?
  - How many graduated vs died vs rugged?
  - Is this wallet flagged as a known whale or sniper?
  - What is their historical grad rate?

Data sources:
  1. rugcheck.xyz /wallet/{address}/risk  — creator history, risk flags
  2. Solana RPC getSignaturesForAddress    — tx history to count launches
  3. Neo4j graph                           — our own observed history

All calls use Helius RPC for getTransaction reliability,
public RPC for signature listing (no credits consumed).

Usage:
  from collectors.solana_rpc import get_wallet_profile
  profile = await get_wallet_profile(address, neo4j_driver)
"""
import asyncio
import logging
import os
from typing import Optional

import httpx

log = logging.getLogger("nathanai.crypto.solana_rpc")

HELIUS_API_KEY  = os.getenv("HELIUS_API_KEY", "")
PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# Metaplex Token Metadata program — used to identify graduation txs
RAYDIUM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

RUGCHECK_BASE = "https://api.rugcheck.xyz/v1"

def _rpc_url() -> str:
    """Always use public RPC to avoid burning Helius credits."""
    return "https://api.mainnet-beta.solana.com"

def _pub_rpc_url() -> str:
    return "https://api.mainnet-beta.solana.com"


# ── rugcheck wallet endpoint ──────────────────────────────────────────────

async def _rugcheck_wallet(address: str) -> dict:
    """
    GET /wallet/{address}/risk from rugcheck.xyz
    Returns risk profile for the creator wallet.
    """
    url = f"{RUGCHECK_BASE}/wallet/{address}/risk"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "rc_wallet_risk":      data.get("risk", "unknown"),
                    "rc_creator_rugs":     data.get("rugCount", 0),
                    "rc_creator_launches": data.get("tokenCount", 0),
                    "rc_flagged":          data.get("flagged", False),
                }
            log.debug("rugcheck wallet %s: HTTP %s", address[:8], resp.status_code)
    except Exception as e:
        log.debug("rugcheck wallet error for %s: %s", address[:8], e)
    return {}


# ── Solana RPC helpers ────────────────────────────────────────────────────

async def _get_signatures(address: str, limit: int = 50) -> list[str]:
    """
    Get recent transaction signatures for a wallet address.
    Uses public RPC — no credits consumed.
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "getSignaturesForAddress",
        "params":  [address, {"limit": limit, "commitment": "confirmed"}],
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(_pub_rpc_url(), json=payload)
            resp.raise_for_status()
            result = resp.json().get("result", [])
            return [r["signature"] for r in result if r.get("err") is None]
    except Exception as e:
        log.debug("getSignaturesForAddress failed for %s: %s", address[:8], e)
        return []


async def _get_transaction(sig: str) -> Optional[dict]:
    """Fetch a single transaction. Uses Helius for reliability."""
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "getTransaction",
        "params":  [sig, {
            "encoding":                       "json",
            "maxSupportedTransactionVersion": 0,
            "commitment":                     "confirmed",
        }],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_rpc_url(), json=payload)
            resp.raise_for_status()
            return resp.json().get("result")
    except Exception as e:
        log.debug("getTransaction failed for %s: %s", sig[:12], e)
        return None


def _tx_involves_program(tx: dict, program_id: str) -> bool:
    """Check if a transaction involves a specific program."""
    try:
        accounts = tx["transaction"]["message"].get("accountKeys", [])
        return program_id in accounts
    except Exception:
        return False


async def _count_wallet_launches(address: str, sig_limit: int = 100) -> dict:
    """
    Count how many pump.fun tokens this wallet has created by scanning tx history.
    Returns {"launches_observed": int, "raydium_interactions": int}
    """
    sigs = await _get_signatures(address, limit=sig_limit)
    if not sigs:
        return {"launches_observed": 0, "raydium_interactions": 0}

    launches      = 0
    raydium_hits  = 0

    # Sample up to 20 transactions to avoid burning credits
    # Full history scan reserved for known high-activity wallets
    sample = sigs[:20]

    async def check_tx(sig: str):
        nonlocal launches, raydium_hits
        tx = await _get_transaction(sig)
        if not tx:
            return
        if _tx_involves_program(tx, PUMPFUN_PROGRAM):
            # Check if it's a create (signer = address = dev wallet position)
            try:
                accounts = tx["transaction"]["message"].get("accountKeys", [])
                # In pump.fun create: account[11] is creator
                if len(accounts) > 11 and accounts[11] == address:
                    launches += 1
            except Exception:
                pass
        if _tx_involves_program(tx, RAYDIUM_PROGRAM):
            raydium_hits += 1

    # Run tx checks concurrently (max 5 at a time to respect rate limits)
    semaphore = asyncio.Semaphore(5)

    async def bounded_check(sig):
        async with semaphore:
            await check_tx(sig)

    await asyncio.gather(*[bounded_check(s) for s in sample])

    return {"launches_observed": launches, "raydium_interactions": raydium_hits}


# ── Neo4j history query ───────────────────────────────────────────────────

async def _get_neo4j_wallet_history(address: str, driver) -> dict:
    """
    Query our own Neo4j graph for observed history of this wallet.
    More reliable than on-chain scanning for wallets we've seen before.
    """
    try:
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Token)-[:CREATED_BY]->(w:Wallet {address: $address})
                RETURN
                    count(t)                                       AS total_launched,
                    sum(CASE WHEN t.graduated = true  THEN 1 ELSE 0 END) AS total_graduated,
                    sum(CASE WHEN t.rc_hard_skip = true THEN 1 ELSE 0 END) AS total_skipped
                """,
                {"address": address},
            )
            record = await result.single()
            if record and record["total_launched"] > 0:
                total     = record["total_launched"]
                graduated = record["total_graduated"]
                return {
                    "neo4j_launches":   total,
                    "neo4j_graduated":  graduated,
                    "neo4j_grad_rate":  round(graduated / total, 3) if total > 0 else 0.0,
                    "neo4j_skipped":    record["total_skipped"],
                }
    except Exception as e:
        log.debug("Neo4j wallet history error for %s: %s", address[:8], e)
    return {}


# ── Main entry point ──────────────────────────────────────────────────────

async def get_wallet_profile(address: str, driver=None) -> dict:
    """
    Build a full wallet reputation profile for a dev wallet.
    Combines: rugcheck wallet API + Solana tx history + Neo4j observed history.

    Returns a dict ready to pass to GraphIngester.upsert_wallet():
      address, tokens_launched, graduates_launched, rugs_launched,
      grad_rate, known_whale, known_sniper, rc_wallet_risk, rc_flagged
    """
    if not address:
        return {}

    log.info("Profiling wallet: %s", address[:8])

    # rugcheck wallet API gives us launch/rug counts for free (1 HTTP call).
    # Chain scan (_count_wallet_launches) makes up to 20 Solana RPC calls per wallet
    # and causes 429s on the free public RPC — skip it entirely.
    rc_task    = _rugcheck_wallet(address)
    neo4j_task = _get_neo4j_wallet_history(address, driver) if driver else asyncio.sleep(0)

    rc_data, neo4j_data = await asyncio.gather(rc_task, neo4j_task)
    if neo4j_data is None:
        neo4j_data = {}

    rc_launches  = rc_data.get("rc_creator_launches", 0)
    rc_rugs      = rc_data.get("rc_creator_rugs", 0)
    neo_launches = neo4j_data.get("neo4j_launches", 0)
    neo_grad     = neo4j_data.get("neo4j_graduated", 0)

    total_launches = max(rc_launches, neo_launches)
    rugs_launched  = rc_rugs
    grad_launched  = neo_grad
    grad_rate      = round(grad_launched / total_launches, 3) if total_launches > 0 else 0.0

    profile = {
        "address":           address,
        "tokens_launched":   total_launches,
        "graduates_launched":grad_launched,
        "rugs_launched":     rugs_launched,
        "grad_rate":         grad_rate,
        "known_whale":       False,       # updated later by smart_money.py
        "known_sniper":      False,
        "rc_wallet_risk":    rc_data.get("rc_wallet_risk", "unknown"),
        "rc_flagged":        rc_data.get("rc_flagged", False),
    }

    log.info(
        "Wallet %s: launches=%d grad_rate=%.1f%% rugs=%d risk=%s flagged=%s",
        address[:8],
        total_launches,
        grad_rate * 100,
        rugs_launched,
        profile["rc_wallet_risk"],
        profile["rc_flagged"],
    )

    return profile
