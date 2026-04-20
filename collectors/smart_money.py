"""
collectors/smart_money.py
=========================
Smart money wallet reputation system.

Two signal layers:
  1. Manual curated list  (data/smart_money.json) — operator-controlled
  2. Auto-discovered      (Neo4j)                 — wallets we observe with
                                                    high grad rates promoted
                                                    automatically

Used in two places:
  A. Dev wallet check — is the token creator smart money?
     Called in main.py on_new_token after the rugcheck gate.
  B. Early buyer check — did a known whale buy into a token early?
     Called in main.py on_trade from subscribeTokenTrade events.

Labels:
  WHALE           — high-capital trader, picks winners consistently
  KOL             — Key Opinion Leader, social influence
  SERIAL_DEPLOYER — dev wallet with proven grad rate (auto-promoted)
  SNIPER          — consistently first buyer (often inside info)
  BANNED          — known rug deployer (hard-skip their launches)

Auto-promotion criteria (from Neo4j observed data):
  tokens_launched  >= MIN_LAUNCHES  (default 5)
  grad_rate        >= MIN_GRAD_RATE (default 0.40)
  rc_flagged       = false

Usage:
  from collectors.smart_money import SmartMoneyCache
  sm = SmartMoneyCache()
  await sm.refresh(driver)          # load from Neo4j

  hit = sm.is_smart_money(address)  # O(1) — use on every trade event
  # hit = {"label": "WHALE", "tier": 1, "grad_rate": 0.72, ...} or None

  hit = sm.check_dev(token_event)   # check token's dev wallet
  hit = sm.check_trade(trade_event) # check trade's buyer/seller
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("nathanai.crypto.smart_money")

# ── Thresholds for Neo4j auto-promotion ──────────────────────────────────────

MIN_LAUNCHES  = 5     # must have launched at least this many tokens
MIN_GRAD_RATE = 0.40  # at least 40% of launches graduated

# How often to refresh from Neo4j (seconds)
REFRESH_INTERVAL = 300  # 5 min

# Where the curated list lives
_DATA_FILE = Path(__file__).parent.parent / "data" / "smart_money.json"


class SmartMoneyCache:
    """
    In-memory cache of smart money wallet reputations.
    Backed by a JSON file (manual curation) and Neo4j (auto-discovered).
    Thread-safe for asyncio — all mutations happen in the event loop.
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}   # address → reputation dict
        self._last_refresh: float = 0.0
        self._curated_count: int = 0
        self._discovered_count: int = 0
        self._load_json()

    # ── Load / save curated list ──────────────────────────────────────────────

    def _load_json(self) -> None:
        """Load manually curated wallets from data/smart_money.json."""
        try:
            if _DATA_FILE.exists():
                data = json.loads(_DATA_FILE.read_text())
                for address, info in data.get("wallets", {}).items():
                    self._cache[address] = {
                        "label":     info.get("label", "WHALE"),
                        "tier":      1,
                        "notes":     info.get("notes", ""),
                        "added":     info.get("added", ""),
                        "grad_rate": info.get("grad_rate", None),
                        "source":    "manual",
                    }
                self._curated_count = len(self._cache)
                log.info("smart_money: loaded %d curated wallets", self._curated_count)
        except Exception as e:
            log.warning("smart_money: failed to load JSON: %s", e)

    def add_wallet(
        self,
        address: str,
        label: str = "WHALE",
        notes: str = "",
        grad_rate: Optional[float] = None,
    ) -> None:
        """
        Add a wallet to the curated list and save to disk immediately.
        Call this from scripts or the future admin interface.
        """
        entry = {
            "label":     label,
            "notes":     notes,
            "grad_rate": grad_rate,
            "added":     datetime.now(timezone.utc).date().isoformat(),
        }
        self._cache[address] = {**entry, "tier": 1, "source": "manual"}

        # Persist to JSON
        try:
            data = json.loads(_DATA_FILE.read_text()) if _DATA_FILE.exists() else {}
            data.setdefault("wallets", {})[address] = entry
            _DATA_FILE.write_text(json.dumps(data, indent=2))
            log.info("smart_money: saved %s as %s", address[:8], label)
        except Exception as e:
            log.warning("smart_money: failed to save JSON: %s", e)

    def ban_wallet(self, address: str, notes: str = "") -> None:
        """Mark a wallet as BANNED so all their future launches are hard-skipped."""
        self.add_wallet(address, label="BANNED", notes=notes)

    # ── Neo4j auto-discovery ──────────────────────────────────────────────────

    async def refresh(self, driver) -> None:
        """
        Query Neo4j for high-performing dev wallets and add them as
        SERIAL_DEPLOYER tier-2 entries. Called periodically from the main loop.
        """
        if not driver:
            return
        try:
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (t:Token)-[:CREATED_BY]->(w:Wallet)
                    WHERE w.rc_flagged <> true
                    WITH  w,
                          count(t)                                        AS launches,
                          sum(CASE WHEN t.graduated = true THEN 1 ELSE 0 END) AS grads
                    WHERE launches >= $min_launches
                      AND (toFloat(grads) / launches) >= $min_grad_rate
                    RETURN w.address AS address,
                           launches,
                           grads,
                           round(toFloat(grads) / launches, 3) AS grad_rate
                    ORDER BY grad_rate DESC
                    """,
                    {
                        "min_launches":  MIN_LAUNCHES,
                        "min_grad_rate": MIN_GRAD_RATE,
                    },
                )
                rows = await result.data()

            promoted = 0
            for row in rows:
                address   = row["address"]
                grad_rate = row["grad_rate"]
                # Don't overwrite tier-1 manual entries
                if address not in self._cache or self._cache[address]["tier"] > 1:
                    self._cache[address] = {
                        "label":     "SERIAL_DEPLOYER",
                        "tier":      2,
                        "grad_rate": grad_rate,
                        "launches":  row["launches"],
                        "grads":     row["grads"],
                        "notes":     f"auto-promoted: {row['grads']}/{row['launches']} graduated",
                        "source":    "neo4j",
                    }
                    promoted += 1

            self._discovered_count = promoted
            self._last_refresh = time.time()
            log.info(
                "smart_money: refresh done — %d curated, %d auto-discovered (%d total)",
                self._curated_count, promoted, len(self._cache),
            )
        except Exception as e:
            log.warning("smart_money: Neo4j refresh failed: %s", e)

    async def refresh_if_stale(self, driver) -> None:
        """Refresh from Neo4j only if the cache is older than REFRESH_INTERVAL."""
        if time.time() - self._last_refresh > REFRESH_INTERVAL:
            await self.refresh(driver)

    # ── Lookup API ────────────────────────────────────────────────────────────

    def is_smart_money(self, address: str) -> Optional[dict]:
        """
        O(1) cache lookup. Returns the reputation dict if address is known,
        None otherwise. The caller decides what to do with BANNED vs WHALE etc.
        """
        return self._cache.get(address)

    def is_banned(self, address: str) -> bool:
        """True if this wallet is explicitly banned (known rug deployer)."""
        entry = self._cache.get(address)
        return entry is not None and entry.get("label") == "BANNED"

    def check_dev(self, token_event: dict) -> Optional[dict]:
        """
        Check if the token's dev wallet is smart money.
        Returns hit dict with label/tier/grad_rate, or None.

        Adds "dev_smart_money" key to the event in-place so downstream
        code (prompt_builder) can read it directly from the event dict.
        """
        dev = token_event.get("dev", "")
        if not dev:
            return None
        hit = self.is_smart_money(dev)
        if hit:
            token_event["dev_smart_money"] = hit
            log.info(
                "SMART MONEY DEV: %s (%s) — %s (grad_rate=%.0f%%)",
                token_event.get("name", "?"),
                token_event.get("ticker", "?"),
                hit["label"],
                (hit.get("grad_rate") or 0) * 100,
            )
        return hit

    def check_trade(self, trade_event: dict) -> Optional[dict]:
        """
        Check if a bonding-curve trade's buyer/seller is smart money.
        Returns hit dict, or None.

        Call this on every subscribeTokenTrade event coming from
        pumpportal_ws — the check is O(1) and safe at high throughput.
        """
        trader = trade_event.get("trader", "") or trade_event.get("traderPublicKey", "")
        if not trader:
            return None
        hit = self.is_smart_money(trader)
        if hit and hit.get("label") != "BANNED":
            side = trade_event.get("side", trade_event.get("txType", "?"))
            mint = trade_event.get("mint", "")
            log.info(
                "SMART MONEY TRADE: %s %s %s | trader=%s (%s) grad_rate=%.0f%%",
                hit["label"], side.upper(),
                trade_event.get("ticker") or mint[:8],
                trader[:8],
                hit["label"],
                (hit.get("grad_rate") or 0) * 100,
            )
        return hit

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line status string for the dashboard."""
        total   = len(self._cache)
        banned  = sum(1 for v in self._cache.values() if v.get("label") == "BANNED")
        whales  = sum(1 for v in self._cache.values() if v.get("label") == "WHALE")
        deployers = sum(1 for v in self._cache.values() if v.get("label") == "SERIAL_DEPLOYER")
        age_min = int((time.time() - self._last_refresh) / 60) if self._last_refresh else -1
        return (
            f"smart_money: {total} wallets "
            f"(whale={whales} deployer={deployers} banned={banned}) "
            f"refresh={age_min}min ago"
        )

    def top_wallets(self, n: int = 10) -> list[dict]:
        """Return the top N wallets by grad_rate for the dashboard."""
        ranked = sorted(
            [{"address": addr, **info} for addr, info in self._cache.items()
             if info.get("label") != "BANNED" and info.get("grad_rate") is not None],
            key=lambda x: x.get("grad_rate", 0),
            reverse=True,
        )
        return ranked[:n]


# ── Module-level singleton ────────────────────────────────────────────────────

_cache = SmartMoneyCache()


def get_cache() -> SmartMoneyCache:
    """Return the module-level singleton cache (no async needed)."""
    return _cache


async def refresh(driver) -> None:
    """Refresh the singleton cache from Neo4j."""
    await _cache.refresh(driver)


def is_smart_money(address: str) -> Optional[dict]:
    """Fast O(1) singleton lookup."""
    return _cache.is_smart_money(address)


def check_dev(token_event: dict) -> Optional[dict]:
    """Check token dev wallet and annotate event in-place."""
    return _cache.check_dev(token_event)


def check_trade(trade_event: dict) -> Optional[dict]:
    """Check a trade's buyer against the smart money cache."""
    return _cache.check_trade(trade_event)


def is_banned(address: str) -> bool:
    """True if wallet is explicitly banned."""
    return _cache.is_banned(address)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(_cache.summary())
    print("Top wallets:", _cache.top_wallets())
