"""
graph/ingest.py
===============
Write Token, Wallet, Signal, Trade, and NewsEvent nodes to Neo4j.
Also generates and stores embeddings via nomic-embed-text (Ollama).

All write functions are idempotent — safe to call multiple times for the same
node (MERGE ensures no duplicates).

Embedding strategy:
  - Token: embed the full context string assembled by prompt_builder
  - Wallet: embed a summary of wallet stats
  - Both stored as float lists on the node property
  - Neo4j vector index allows similarity search without external vector store

Usage:
  from graph.ingest import GraphIngester
  ingester = GraphIngester(driver)
  await ingester.upsert_token(token_event)
  await ingester.upsert_wallet(wallet_data)
  await ingester.write_signal(signal)
  await ingester.write_trade(trade)
"""
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx

log = logging.getLogger("nathanai.crypto.graph.ingest")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBED_MODEL     = "nomic-embed-text"


# ── Embedding helper ───────────────────────────────────────────────────────

async def embed(text: str) -> list[float]:
    """Generate embedding via nomic-embed-text through Ollama. Returns 768-dim vector."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text}
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
    except Exception as e:
        log.warning("Embedding failed for text '%s...': %s", text[:40], e)
        return []  # empty list = no vector stored, index skips gracefully


# ── GraphIngester class ────────────────────────────────────────────────────

class GraphIngester:

    def __init__(self, driver):
        self.driver = driver

    async def upsert_token(self, event: dict) -> None:
        """
        Write or update a Token node from a pump.fun event dict.
        Generates context_embedding via nomic-embed-text.

        event keys (from pumpfun_ws + rugcheck):
          mint, name, ticker, dev, uri, hard_skip, rugcheck (dict)
        """
        rc = event.get("rugcheck", {})
        mint   = event["mint"]
        name   = event.get("name", "")
        ticker = event.get("ticker", "")
        dev    = event.get("dev", "")

        # Build embedding context — same text the LLM will reason about
        context_text = (
            f"{name} {ticker} "
            f"dev:{dev[:8]} "
            f"rc_score:{rc.get('score', 0)} "
            f"rc_hard_skip:{event.get('hard_skip', False)} "
            f"top_holder:{rc.get('top_holder_pct', 0):.1f}% "
            f"mint_auth:{rc.get('mint_authority', False)} "
            f"freeze_auth:{rc.get('freeze_authority', False)}"
        )
        embedding = await embed(context_text)

        query = """
            MERGE (t:Token {mint: $mint})
            SET
                t.name                = $name,
                t.ticker              = $ticker,
                t.dev_wallet          = $dev,
                t.created_at          = $created_at,
                t.uri                 = $uri,
                t.rc_hard_skip        = $hard_skip,
                t.rc_score            = $rc_score,
                t.rc_mint_authority   = $mint_authority,
                t.rc_freeze_authority = $freeze_authority,
                t.rc_top_holder_pct   = $top_holder_pct,
                t.rc_lp_locked        = $lp_locked,
                t.rc_insider_detected = $insider_detected,
                t.rc_risk_penalty     = $risk_penalty,
                t.rc_risk_flags       = $risk_flags,
                t.context_text        = $context_text,
                t.context_embedding   = $embedding
            MERGE (w:Wallet {address: $dev})
            MERGE (t)-[:CREATED_BY]->(w)
        """
        params = {
            "mint":             mint,
            "name":             name,
            "ticker":           ticker,
            "dev":              dev,
            "created_at":       datetime.now(timezone.utc).isoformat(),
            "uri":              event.get("uri", ""),
            "hard_skip":        event.get("hard_skip", False),
            "rc_score":         rc.get("score", 0),
            "mint_authority":   rc.get("mint_authority", False),
            "freeze_authority": rc.get("freeze_authority", False),
            "top_holder_pct":   rc.get("top_holder_pct", 0.0),
            "lp_locked":        rc.get("lp_locked", False),
            "insider_detected": rc.get("insider_detected", False),
            "risk_penalty":     rc.get("risk_penalty", 0),
            "risk_flags":       str(rc.get("risk_flags", [])),
            "context_text":     context_text,
            "embedding":        embedding,
        }

        async with self.driver.session() as session:
            await session.run(query, params)
            log.debug("upsert_token OK: %s (%s)", name, mint[:8])

    async def upsert_wallet(self, address: str, stats: dict) -> None:
        """
        Write or update a Wallet node with reputation stats.
        Generates profile_embedding via nomic-embed-text.

        stats keys: tokens_launched, graduates_launched, rugs_launched,
                    grad_rate, known_whale, known_sniper
        """
        profile_text = (
            f"wallet:{address[:8]} "
            f"launches:{stats.get('tokens_launched', 0)} "
            f"grad_rate:{stats.get('grad_rate', 0.0):.2f} "
            f"rugs:{stats.get('rugs_launched', 0)} "
            f"whale:{stats.get('known_whale', False)} "
            f"sniper:{stats.get('known_sniper', False)}"
        )
        embedding = await embed(profile_text)

        query = """
            MERGE (w:Wallet {address: $address})
            SET
                w.tokens_launched    = $tokens_launched,
                w.graduates_launched = $graduates_launched,
                w.rugs_launched      = $rugs_launched,
                w.grad_rate          = $grad_rate,
                w.known_whale        = $known_whale,
                w.known_sniper       = $known_sniper,
                w.last_seen          = $last_seen,
                w.profile_text       = $profile_text,
                w.profile_embedding  = $embedding
        """
        params = {
            "address":           address,
            "tokens_launched":   stats.get("tokens_launched", 0),
            "graduates_launched":stats.get("graduates_launched", 0),
            "rugs_launched":     stats.get("rugs_launched", 0),
            "grad_rate":         stats.get("grad_rate", 0.0),
            "known_whale":       stats.get("known_whale", False),
            "known_sniper":      stats.get("known_sniper", False),
            "last_seen":         datetime.now(timezone.utc).isoformat(),
            "profile_text":      profile_text,
            "embedding":         embedding,
        }
        async with self.driver.session() as session:
            await session.run(query, params)
            log.debug("upsert_wallet OK: %s", address[:8])

    async def write_signal(self, signal: dict) -> None:
        """
        Write a Signal node (LLM BUY/SKIP decision) linked to its Token.

        signal keys: token_mint, decision, confidence, reasoning, model_version
        """
        query = """
            MATCH (t:Token {mint: $token_mint})
            CREATE (s:Signal {
                id:            $id,
                token_mint:    $token_mint,
                timestamp:     $timestamp,
                decision:      $decision,
                confidence:    $confidence,
                reasoning:     $reasoning,
                model_version: $model_version
            })
            CREATE (s)-[:EVALUATED]->(t)
        """
        params = {
            "id":            str(uuid.uuid4()),
            "token_mint":    signal["token_mint"],
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "decision":      signal["decision"],
            "confidence":    signal["confidence"],
            "reasoning":     signal.get("reasoning", "")[:2000],  # truncate
            "model_version": signal.get("model_version", "unknown"),
        }
        async with self.driver.session() as session:
            await session.run(query, params)
            log.info("signal written: %s %.2f for %s",
                     signal["decision"], signal["confidence"], signal["token_mint"][:8])

    async def write_trade(self, trade: dict) -> None:
        """
        Write a Trade node (executed or paper trade) linked to its Token.

        trade keys: token_mint, entry_time, entry_price_sol, entry_amount_sol,
                    exit_time, exit_price_sol, pnl_sol, pnl_pct, exit_reason
        """
        query = """
            MATCH (t:Token {mint: $token_mint})
            CREATE (tr:Trade {
                id:               $id,
                token_mint:       $token_mint,
                entry_time:       $entry_time,
                entry_price_sol:  $entry_price_sol,
                entry_amount_sol: $entry_amount_sol,
                exit_time:        $exit_time,
                exit_price_sol:   $exit_price_sol,
                pnl_sol:          $pnl_sol,
                pnl_pct:          $pnl_pct,
                exit_reason:      $exit_reason
            })
            CREATE (tr)-[:FOR_TOKEN]->(t)
        """
        params = {**trade, "id": str(uuid.uuid4())}
        async with self.driver.session() as session:
            await session.run(query, params)
            log.info("trade written: pnl=%.4f SOL (%s)", trade["pnl_sol"], trade["exit_reason"])

    async def mark_graduated(self, mint: str, raydium_pool: Optional[str] = None) -> None:
        """Mark a token as graduated to Raydium."""
        async with self.driver.session() as session:
            await session.run(
                """
                MATCH (t:Token {mint: $mint})
                SET t.graduated       = true,
                    t.graduation_time = $ts,
                    t.raydium_pool    = $pool
                """,
                {"mint": mint, "ts": datetime.now(timezone.utc).isoformat(),
                 "pool": raydium_pool or ""}
            )
            log.info("marked graduated: %s", mint[:8])

    async def find_similar_tokens(
        self, embedding: list[float], limit: int = 5, graduated_only: bool = True
    ) -> list[dict]:
        """
        Vector similarity search: find tokens most similar to the given embedding.
        Used to ask: "Have we seen tokens like this before? Did they graduate?"

        Returns list of {mint, name, ticker, graduated, score} dicts.
        """
        where_clause = "WHERE t.graduated = true" if graduated_only else ""
        query = f"""
            CALL db.index.vector.queryNodes(
                'token_context_vector', $limit, $embedding
            ) YIELD node AS t, score
            {where_clause}
            RETURN t.mint AS mint, t.name AS name, t.ticker AS ticker,
                   t.graduated AS graduated, score
            ORDER BY score DESC
            LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(query, {"embedding": embedding, "limit": limit})
            return [dict(r) for r in await result.data()]

    async def find_similar_wallets(
        self, embedding: list[float], limit: int = 5
    ) -> list[dict]:
        """
        Vector similarity search: find wallets with similar profile to this one.
        Used to classify unknown dev wallets by similarity to known good/bad wallets.
        """
        query = """
            CALL db.index.vector.queryNodes(
                'wallet_profile_vector', $limit, $embedding
            ) YIELD node AS w, score
            RETURN w.address AS address, w.grad_rate AS grad_rate,
                   w.known_whale AS known_whale, w.known_sniper AS known_sniper, score
            ORDER BY score DESC
            LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(query, {"embedding": embedding, "limit": limit})
            return [dict(r) for r in await result.data()]
