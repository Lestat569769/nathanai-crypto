"""
collectors/rugcheck.py
======================
Fetches token risk report from rugcheck.xyz free API.

Endpoint: GET https://api.rugcheck.xyz/v1/tokens/{mint}/report
No API key required. Rate limit: be respectful, add delay between calls.

Risk signals extracted:
  - score_normalised  : 0-100 (higher = safer)
  - mint_authority    : if set, dev can print unlimited tokens → HIGH RISK
  - freeze_authority  : if set, dev can freeze holder wallets → HIGH RISK
  - top_holder_pct    : % held by single largest wallet
  - insider_detected  : coordinated insider buying network found
  - lp_locked         : liquidity provider tokens locked (can't rug pull)
  - rugged            : already confirmed rug
  - risks             : list of flagged issues with severity levels
  - creator_history   : prior tokens by same dev (graduates vs rugs)
"""
import asyncio
import httpx
import logging
import time
from typing import Optional

log = logging.getLogger("nathanai.crypto.rugcheck")

API_BASE    = "https://api.rugcheck.xyz/v1"
TIMEOUT     = 10.0
RETRY_DELAY = 2.0
MAX_RETRIES = 3

# Severity weights for risk score aggregation
SEVERITY_WEIGHTS = {
    "danger":  10,
    "warning":  5,
    "info":     1,
}


async def fetch_report(mint: str) -> Optional[dict]:
    """Fetch raw rugcheck report for a token mint address (async — non-blocking)."""
    url = f"{API_BASE}/tokens/{mint}/report"
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                log.warning("rugcheck rate limited, sleeping %ds", RETRY_DELAY * 2)
                await asyncio.sleep(RETRY_DELAY * 2)
            elif resp.status_code == 404:
                log.debug("rugcheck: token %s not indexed yet", mint[:8])
                return None
            else:
                log.warning("rugcheck HTTP %d for %s", resp.status_code, mint[:8])
                return None
        except httpx.TimeoutException:
            log.warning("rugcheck timeout (attempt %d/%d) for %s", attempt + 1, MAX_RETRIES, mint[:8])
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            log.error("rugcheck error for %s: %s", mint[:8], e)
            return None
    return None


def parse_report(report: dict) -> dict:
    """
    Extract the signals we care about from a raw rugcheck report.
    Returns a flat dict ready for Neo4j storage and prompt building.
    """
    if not report:
        return _empty_result()

    token       = report.get("token", {})
    top_holders = report.get("topHolders", [])
    risks       = report.get("risks", [])
    lockers     = report.get("lockers", [])

    # ── Core danger flags ─────────────────────────────────────────────
    mint_authority   = token.get("mintAuthority") not in (None, "")
    freeze_authority = token.get("freezeAuthority") not in (None, "")
    already_rugged   = report.get("rugged", False)

    # ── Holder concentration ──────────────────────────────────────────
    top_holder_pct = 0.0
    if top_holders:
        top_holder_pct = top_holders[0].get("pct", 0.0) * 100  # convert to %

    # ── Liquidity lock ────────────────────────────────────────────────
    lp_locked = len(lockers) > 0

    # ── Insider network detection ─────────────────────────────────────
    insider_detected = report.get("graphInsidersDetected", False)
    insider_networks = report.get("insiderNetworks", [])

    # ── Risk list (danger / warning / info) ───────────────────────────
    risk_flags = []
    risk_score_penalty = 0
    for r in risks:
        name     = r.get("name", "unknown")
        level    = r.get("level", "info").lower()
        risk_flags.append({"name": name, "level": level})
        risk_score_penalty += SEVERITY_WEIGHTS.get(level, 1)

    # ── Creator history ───────────────────────────────────────────────
    creator_tokens    = report.get("creatorTokens", [])
    creator_graduates = sum(1 for t in creator_tokens if t.get("graduated", False))
    creator_rugs      = sum(1 for t in creator_tokens if t.get("rugged", False))
    creator_total     = len(creator_tokens)

    # ── rugcheck's own score ──────────────────────────────────────────
    score_normalised = report.get("score_normalised", 50)

    # ── Hard skip decision ────────────────────────────────────────────
    # Any of these alone is disqualifying — don't even let the LLM see it
    hard_skip = (
        already_rugged
        or mint_authority
        or freeze_authority
        or top_holder_pct > 30.0          # single wallet holds >30% = trap
        or risk_score_penalty >= 10       # one danger flag or two warnings
    )

    return {
        # Safety flags
        "mint_authority":    mint_authority,
        "freeze_authority":  freeze_authority,
        "already_rugged":    already_rugged,
        "hard_skip":         hard_skip,

        # Score
        "score_normalised":  score_normalised,
        "risk_penalty":      risk_score_penalty,
        "risk_flags":        risk_flags,

        # Holder data
        "top_holder_pct":    round(top_holder_pct, 2),

        # Liquidity
        "lp_locked":         lp_locked,
        "lp_locker_count":   len(lockers),

        # Insider activity
        "insider_detected":  insider_detected,
        "insider_networks":  len(insider_networks),

        # Creator track record
        "creator_total":     creator_total,
        "creator_graduates": creator_graduates,
        "creator_rugs":      creator_rugs,
        "creator_grad_rate": round(creator_graduates / creator_total, 2) if creator_total > 0 else 0.0,

        # Raw for Neo4j storage
        "raw_risks":         risk_flags,
    }


def _empty_result() -> dict:
    """Returned when rugcheck has no data yet (very new token)."""
    return {
        "mint_authority":    None,
        "freeze_authority":  None,
        "already_rugged":    False,
        "hard_skip":         False,
        "score_normalised":  None,
        "risk_penalty":      0,
        "risk_flags":        [],
        "top_holder_pct":    0.0,
        "lp_locked":         False,
        "lp_locker_count":   0,
        "insider_detected":  False,
        "insider_networks":  0,
        "creator_total":     0,
        "creator_graduates": 0,
        "creator_rugs":      0,
        "creator_grad_rate": 0.0,
        "raw_risks":         [],
    }


async def check_token(mint: str) -> dict:
    """
    Main entry point. Fetch + parse a rugcheck report (async — non-blocking).

    Returns parsed signals dict. Callers should check `hard_skip` first:
        result = await check_token(mint)
        if result["hard_skip"]:
            return SKIP  # don't waste LLM inference on this
    """
    log.info("rugcheck: fetching report for %s...", mint[:8])
    raw = await fetch_report(mint)
    result = parse_report(raw)

    if result["hard_skip"]:
        reasons = []
        if result["already_rugged"]:    reasons.append("ALREADY RUGGED")
        if result["mint_authority"]:    reasons.append("MINT AUTHORITY SET")
        if result["freeze_authority"]:  reasons.append("FREEZE AUTHORITY SET")
        if result["top_holder_pct"] > 30: reasons.append(f"TOP HOLDER {result['top_holder_pct']:.0f}%")
        if result["risk_penalty"] >= 10:  reasons.append(f"RISK PENALTY={result['risk_penalty']}")
        log.warning("rugcheck HARD SKIP %s: %s", mint[:8], " | ".join(reasons))
    else:
        log.info(
            "rugcheck OK %s: score=%s  top_holder=%.1f%%  insider=%s  lp_locked=%s  creator=%d/%d grad",
            mint[:8],
            result["score_normalised"],
            result["top_holder_pct"],
            result["insider_detected"],
            result["lp_locked"],
            result["creator_graduates"],
            result["creator_total"],
        )

    return result


def format_for_prompt(result: dict) -> str:
    """
    Render rugcheck signals as a concise block for the LLM prompt.
    Called by adapter/prompt_builder.py.
    """
    if result["hard_skip"]:
        return "[RUGCHECK] HARD SKIP — do not trade this token."

    lines = [
        f"[RUGCHECK] score={result['score_normalised']}/100",
        f"  mint_authority={result['mint_authority']}  freeze_authority={result['freeze_authority']}",
        f"  top_holder={result['top_holder_pct']:.1f}%  lp_locked={result['lp_locked']}",
        f"  insider_detected={result['insider_detected']}",
        f"  creator: {result['creator_graduates']}/{result['creator_total']} prior tokens graduated"
            + (f"  {result['creator_rugs']} rugs" if result["creator_rugs"] > 0 else ""),
    ]

    if result["risk_flags"]:
        flags = ", ".join(f"{r['level']}:{r['name']}" for r in result["risk_flags"])
        lines.append(f"  risks: {flags}")

    return "\n".join(lines)
