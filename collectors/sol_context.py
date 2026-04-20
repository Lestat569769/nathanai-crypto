"""
collectors/sol_context.py
=========================
Periodic market context collector — no API key, no paid credits.

Three data sources:
  1. CoinGecko public API  — SOL price (USD) + 24h % change
  2. alternative.me API    — Crypto Fear & Greed Index (0-100)
  3. RSS feeds             — Recent crypto / Solana news headlines

All fetches are cached for a configurable TTL so the LLM prompt always
gets fresh-ish context without hammering free-tier endpoints.

Usage:
  from collectors.sol_context import get_market_context
  ctx = await get_market_context()
  # ctx = {
  #   "sol_price_usd":    174.23,
  #   "sol_change_24h":   -2.1,
  #   "fear_greed_value": 42,
  #   "fear_greed_label": "Fear",
  #   "news_headlines":   ["Solana TVL hits record...", ...],
  #   "context_summary":  "SOL $174 (-2.1% 24h). Market: Fear (42/100)...",
  #   "fetched_at":       1714000000.0,
  # }

The `context_summary` field is pre-formatted for direct injection into
the LLM prompt so prompt_builder.py doesn't need to know the structure.
"""
import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

log = logging.getLogger("nathanai.crypto.sol_context")

# ── Endpoints ─────────────────────────────────────────────────────────────────

COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=solana&vs_currencies=usd&include_24hr_change=true"
)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"

# Free RSS feeds — no API key needed
RSS_FEEDS = [
    "https://cointelegraph.com/rss/tag/solana",           # Solana-specific
    "https://decrypt.co/feed",                             # General crypto
    "https://www.coindesk.com/arc/outboundfeeds/rss/",    # General crypto
]

MAX_HEADLINES   = 5      # headlines per feed × feeds (deduplicated)
HEADLINES_CAP   = 10     # total headlines in the final context

# ── Cache ─────────────────────────────────────────────────────────────────────

_CACHE: dict = {}
_CACHE_TTL   = 300.0   # 5 minutes — CoinGecko free tier allows ~30 req/min


def _cache_get(key: str, ttl: float = _CACHE_TTL) -> Optional[dict]:
    entry = _CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["data"]
    return None


def _cache_set(key: str, data) -> None:
    _CACHE[key] = {"ts": time.time(), "data": data}


# ── SOL Price ─────────────────────────────────────────────────────────────────

async def get_sol_price() -> dict:
    """
    Fetch SOL price from CoinGecko public API.
    Returns {"sol_price_usd": float, "sol_change_24h": float}
    Falls back to cached value if the request fails.
    """
    cached = _cache_get("sol_price")
    if cached:
        return cached

    result = {"sol_price_usd": 0.0, "sol_change_24h": 0.0}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(COINGECKO_URL)
            resp.raise_for_status()
            data = resp.json()
            sol = data.get("solana", {})
            result = {
                "sol_price_usd":  round(sol.get("usd", 0.0), 4),
                "sol_change_24h": round(sol.get("usd_24h_change", 0.0), 2),
            }
            _cache_set("sol_price", result)
            log.debug("SOL price: $%.2f (%+.1f%% 24h)",
                      result["sol_price_usd"], result["sol_change_24h"])
    except Exception as e:
        log.warning("CoinGecko fetch failed: %s", e)
        # Return stale cache if available
        stale = _CACHE.get("sol_price")
        if stale:
            return stale["data"]

    return result


# ── Fear & Greed ──────────────────────────────────────────────────────────────

async def get_fear_greed() -> dict:
    """
    Fetch the Crypto Fear & Greed Index from alternative.me.
    Returns {"fear_greed_value": int, "fear_greed_label": str}

    Labels: Extreme Fear (0-24), Fear (25-49), Neutral (50-54),
            Greed (55-74), Extreme Greed (75-100)
    """
    cached = _cache_get("fear_greed", ttl=600.0)   # 10 min — updated daily
    if cached:
        return cached

    result = {"fear_greed_value": 50, "fear_greed_label": "Neutral"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(FEAR_GREED_URL)
            resp.raise_for_status()
            data = resp.json()
            entry = data.get("data", [{}])[0]
            result = {
                "fear_greed_value": int(entry.get("value", 50)),
                "fear_greed_label": entry.get("value_classification", "Neutral"),
            }
            _cache_set("fear_greed", result)
            log.debug("Fear & Greed: %d (%s)",
                      result["fear_greed_value"], result["fear_greed_label"])
    except Exception as e:
        log.warning("Fear & Greed fetch failed: %s", e)
        stale = _CACHE.get("fear_greed")
        if stale:
            return stale["data"]

    return result


# ── News Headlines ────────────────────────────────────────────────────────────

async def _fetch_rss(url: str, client: httpx.AsyncClient) -> list[str]:
    """Fetch one RSS feed and return a list of headline strings."""
    try:
        resp = await client.get(url, timeout=8.0)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        headlines = []
        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            if title:
                headlines.append(title)
            if len(headlines) >= MAX_HEADLINES:
                break
        return headlines
    except Exception as e:
        log.debug("RSS fetch failed (%s): %s", url[:50], e)
        return []


async def get_news_headlines() -> list[str]:
    """
    Fetch recent crypto headlines from RSS feeds.
    Returns a deduplicated list of up to HEADLINES_CAP headline strings.
    """
    cached = _cache_get("news", ttl=900.0)   # 15 min
    if cached:
        return cached

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "NathanAI-CryptoBot/1.0"},
    ) as client:
        results = await asyncio.gather(
            *[_fetch_rss(url, client) for url in RSS_FEEDS],
            return_exceptions=True,
        )

    seen = set()
    headlines = []
    for batch in results:
        if isinstance(batch, list):
            for h in batch:
                if h not in seen:
                    seen.add(h)
                    headlines.append(h)
                if len(headlines) >= HEADLINES_CAP:
                    break
        if len(headlines) >= HEADLINES_CAP:
            break

    _cache_set("news", headlines)
    log.debug("RSS headlines fetched: %d total", len(headlines))
    return headlines


# ── Combined context ──────────────────────────────────────────────────────────

async def get_market_context() -> dict:
    """
    Fetch all market signals concurrently and return a combined context dict.

    The `context_summary` field is a compact prose string ready to inject
    into the LLM system prompt or token-evaluation prompt:

      "SOL $174.23 (-2.1% 24h). Market sentiment: Fear (42/100). "
      "Recent news: Solana TVL hits record | SEC eyes Solana ETF | ..."

    Keys:
      sol_price_usd    float   — current SOL price in USD
      sol_change_24h   float   — 24h price change %
      fear_greed_value int     — 0-100
      fear_greed_label str     — "Extreme Fear" … "Extreme Greed"
      news_headlines   list    — up to HEADLINES_CAP strings
      context_summary  str     — pre-formatted for LLM prompt injection
      fetched_at       float   — unix timestamp
    """
    price_task = asyncio.create_task(get_sol_price())
    fg_task    = asyncio.create_task(get_fear_greed())
    news_task  = asyncio.create_task(get_news_headlines())

    price, fg, news = await asyncio.gather(price_task, fg_task, news_task)

    change_str = f"{price['sol_change_24h']:+.1f}%"
    news_str   = " | ".join(news[:5]) if news else "No recent headlines"

    summary = (
        f"SOL ${price['sol_price_usd']:.2f} ({change_str} 24h). "
        f"Market sentiment: {fg['fear_greed_label']} ({fg['fear_greed_value']}/100). "
        f"Recent news: {news_str}"
    )

    return {
        "sol_price_usd":    price["sol_price_usd"],
        "sol_change_24h":   price["sol_change_24h"],
        "fear_greed_value": fg["fear_greed_value"],
        "fear_greed_label": fg["fear_greed_label"],
        "news_headlines":   news,
        "context_summary":  summary,
        "fetched_at":       time.time(),
    }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    async def _test():
        ctx = await get_market_context()
        pprint.pprint(ctx)

    asyncio.run(_test())
