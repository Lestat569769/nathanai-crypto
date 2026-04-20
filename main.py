"""
NathanAI Crypto Bot — Entry Point
Usage:
  python main.py --mode collect    # collect only — no LLM, pure data gathering
  python main.py --mode paper      # full pipeline: collect + LLM + paper trade logging
  python main.py --mode validate   # batch re-evaluate stored tokens (backfill)
  python main.py --mode live       # full pipeline + real on-chain execution (Phase 4)
"""
import argparse
import asyncio
import logging
import sys

log = logging.getLogger("nathanai.crypto.main")

# ── Minimum confidence for execution ──────────────────────────────────────────
BUY_CONFIDENCE_THRESHOLD = 0.75


def parse_args():
    parser = argparse.ArgumentParser(description="NathanAI Crypto Bot")
    parser.add_argument(
        "--mode",
        choices=["collect", "paper", "validate", "live"],
        required=True,
        help="Operating mode",
    )
    return parser.parse_args()


# ── Shared background tasks ───────────────────────────────────────────────────

async def _profile_and_store_wallet(dev: str, ingester, driver):
    """Background task: profile dev wallet and update Neo4j."""
    from collectors.solana_rpc import get_wallet_profile
    try:
        profile = await get_wallet_profile(dev, driver)
        if profile:
            await ingester.upsert_wallet(dev, profile)
    except Exception as e:
        log.warning("wallet profile error for %s: %s", dev[:8], e)


async def _refresh_smart_money_loop(driver):
    """Refresh smart money cache from Neo4j every 5 minutes."""
    from collectors import smart_money as sm
    while True:
        await asyncio.sleep(300)
        await sm.refresh(driver)


# ── LLM decision pipeline ─────────────────────────────────────────────────────

async def _run_llm_decision(event: dict, ingester, market_ctx_ref: list):
    """
    Full LLM decision pipeline for a single token event.
    Runs as an asyncio background task — never blocks the WebSocket listener.

    Steps:
      1. Build prompt from all available signals
      2. Call qwen3:8b (primary decision)
      3. Parse + validate schema and calibration
      4. If BUY + confidence ≥ 0.75: call qwen3:4b hallucination gate
      5. Write Signal node to Neo4j
      6. Log decision clearly for dashboard monitoring
    """
    from adapter.prompt_builder  import build_messages
    from adapter.inference       import call_primary
    from adapter.validator       import validate
    from adapter.decision_parser import parse

    mint   = event.get("mint", "")
    name   = event.get("name", "?") or "?"
    ticker = event.get("ticker", "?") or "?"

    # ── 1. Assemble prompt ────────────────────────────────────────────────
    market_ctx = market_ctx_ref[0] if market_ctx_ref else None

    # Fetch similar graduated tokens from Neo4j vector index
    similar_tokens = []
    try:
        from graph.ingest import embed
        context_text = (
            f"{name} {ticker} "
            f"rc_score:{event.get('rugcheck', {}).get('score_normalised', 0)}"
        )
        embedding = await embed(context_text)
        if embedding:
            similar_tokens = await ingester.find_similar_tokens(embedding, limit=5)
    except Exception as e:
        log.debug("vector search failed for %s: %s", mint[:8], e)

    # Whale buy signals for this token (written by on_trade)
    whale_signals = []
    try:
        async with ingester.driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Signal)-[:EVALUATED]->(t:Token {mint: $mint})
                WHERE s.decision STARTS WITH 'WHALE_'
                RETURN s.reasoning AS reasoning, s.decision AS decision
                LIMIT 5
                """,
                {"mint": mint},
            )
            whale_signals = await result.data()
    except Exception as e:
        log.debug("whale signal fetch failed for %s: %s", mint[:8], e)

    messages = build_messages(event, market_ctx, similar_tokens, whale_signals)

    # ── 2. Primary decision ───────────────────────────────────────────────
    raw = await call_primary(messages)
    if not raw:
        log.warning("LLM: no response from primary model for %s", mint[:8])
        return

    parsed = parse(raw, token_event=event)
    if not parsed:
        log.warning("LLM: unparseable response for %s | raw=%s", mint[:8], raw[:150])
        return

    # ── 3. Hallucination gate (BUY only, above threshold) ─────────────────
    gate_result = {"valid": True, "reason": "not_required"}
    if parsed.decision == "BUY" and parsed.confidence >= BUY_CONFIDENCE_THRESHOLD:
        gate_result = await validate(event, parsed)

    # ── 4. Final decision ─────────────────────────────────────────────────
    final_decision = parsed.decision
    if not gate_result["valid"]:
        final_decision = "SKIP"

    _log_decision(name, ticker, mint, parsed, final_decision, gate_result)

    # ── 5. Write signal to Neo4j ──────────────────────────────────────────
    await ingester.write_signal({
        "token_mint":    mint,
        "decision":      final_decision,
        "confidence":    parsed.confidence,
        "reasoning":     (
            parsed.reasoning
            + (f" [GATE REJECTED: {gate_result['reason']}]" if not gate_result["valid"] else "")
        ),
        "model_version": "qwen3:8b+4b_v1",
    })


def _log_decision(name, ticker, mint, parsed, final_decision, gate_result):
    """Print a clear, consistently-formatted decision line."""
    conf_pct = int(parsed.confidence * 100)
    gate_str = ""
    if parsed.decision == "BUY" and not gate_result["valid"]:
        gate_str = f" [GATE REJECTED: {gate_result['reason'][:60]}]"

    if final_decision == "BUY":
        log.info(
            "★ BUY  | %s (%s) | conf=%d%% | %s",
            name, ticker, conf_pct, parsed.reasoning[:80],
        )
    else:
        log.info(
            "  SKIP | %s (%s) | conf=%d%% | %s%s",
            name, ticker, conf_pct, parsed.reasoning[:80], gate_str,
        )


# ── Market context refresh loop ───────────────────────────────────────────────

async def _refresh_market_ctx_loop(market_ctx_ref: list):
    """Update market context every 5 minutes. market_ctx_ref is a 1-element list."""
    from collectors.sol_context import get_market_context
    while True:
        try:
            ctx = await get_market_context()
            market_ctx_ref[0] = ctx
            log.debug("market context refreshed: %s", ctx.get("context_summary", "")[:80])
        except Exception as e:
            log.warning("market context refresh failed: %s", e)
        await asyncio.sleep(300)


# ── Collect mode — data only, no LLM ─────────────────────────────────────────

async def run_collect():
    import os
    from neo4j import AsyncGraphDatabase
    from collectors.pumpportal_ws import listen
    from collectors import smart_money as sm
    from graph.schema import init_schema
    from graph.ingest import GraphIngester

    neo4j_uri  = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
    neo4j_user = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "crypto123")
    driver     = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    await init_schema(driver)
    ingester = GraphIngester(driver)

    await sm.refresh(driver)
    print("[collect] Neo4j ready — data collection mode (no LLM)")
    print(f"[collect] {sm.get_cache().summary()}")

    asyncio.create_task(_refresh_smart_money_loop(driver))

    async def on_new_token(event):
        if event.get("_metadata_update"):
            await ingester.upsert_token(event)
            return

        mint   = event.get("mint", "")
        name   = event.get("name", "?") or "?"
        ticker = event.get("ticker", "?") or "?"
        dev    = event.get("dev", "")

        sm.check_dev(event)

        status = "HARD SKIP" if event.get("hard_skip") else "QUEUED  "
        label  = f" ★ {event['dev_smart_money']['label']}" if event.get("dev_smart_money") else ""
        print(f"[collect] {status} | {name} ({ticker}) | mint={mint[:8]}{label}")

        await ingester.upsert_token(event)
        if dev:
            asyncio.create_task(_profile_and_store_wallet(dev, ingester, driver))

    async def on_trade(event):
        from collectors import smart_money as sm
        hit = sm.check_trade(event)
        if hit:
            side   = event.get("side", "?").upper()
            ticker = event.get("ticker", "?") or "?"
            sol    = event.get("sol_amount", 0.0)
            mcap   = event.get("market_cap_sol", 0.0)
            print(f"[collect] ★ {hit['label']} {side} {ticker} | {sol:.2f} SOL | mcap={mcap:.0f} SOL")
            await ingester.write_signal({
                "token_mint":    event.get("mint", ""),
                "decision":      f"WHALE_{side}",
                "confidence":    0.0,
                "reasoning":     (
                    f"{hit['label']} {event.get('trader','')[:8]} {side} "
                    f"{sol:.2f} SOL at mcap={mcap:.0f} SOL"
                ),
                "model_version": "smart_money_v1",
            })

    async def on_graduation(event):
        mint = event.get("mint", "")
        name = event.get("name", "") or "?"
        print(f"[collect] GRADUATED | {name} | mint={mint[:8] if mint else '?'}")
        if mint:
            await ingester.mark_graduated(mint)

    await listen(on_new_token=on_new_token, on_trade=on_trade, on_graduation=on_graduation)


# ── Paper mode — full pipeline, no execution ──────────────────────────────────

async def run_paper():
    import os
    from neo4j import AsyncGraphDatabase
    from collectors.pumpportal_ws import listen
    from collectors import smart_money as sm
    from collectors.sol_context import get_market_context
    from adapter.inference import check_ollama_available
    from graph.schema import init_schema
    from graph.ingest import GraphIngester

    neo4j_uri  = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
    neo4j_user = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "crypto123")
    driver     = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    await init_schema(driver)
    ingester = GraphIngester(driver)

    # Pre-flight checks
    await sm.refresh(driver)
    ollama_ok = await check_ollama_available()
    if not ollama_ok:
        print("[paper] WARNING: Ollama not available or models missing. Decisions will be skipped.")

    # Shared market context (refreshed every 5 min)
    market_ctx_ref = [None]
    try:
        market_ctx_ref[0] = await get_market_context()
    except Exception as e:
        log.warning("initial market context fetch failed: %s", e)

    print("[paper] Pipeline ready — collecting + LLM decisions (paper trade mode)")
    print(f"[paper] {sm.get_cache().summary()}")
    if market_ctx_ref[0]:
        print(f"[paper] {market_ctx_ref[0].get('context_summary', '')}")

    asyncio.create_task(_refresh_smart_money_loop(driver))
    asyncio.create_task(_refresh_market_ctx_loop(market_ctx_ref))

    async def on_new_token(event):
        if event.get("_metadata_update"):
            await ingester.upsert_token(event)
            return

        mint   = event.get("mint", "")
        name   = event.get("name", "?") or "?"
        ticker = event.get("ticker", "?") or "?"
        dev    = event.get("dev", "")

        sm.check_dev(event)

        if sm.is_banned(dev):
            print(f"[paper] BANNED DEV | {name} ({ticker}) | mint={mint[:8]}")
        elif event.get("hard_skip"):
            print(f"[paper] HARD SKIP  | {name} ({ticker}) | mint={mint[:8]}")
        else:
            label = f" ★ {event['dev_smart_money']['label']}" if event.get("dev_smart_money") else ""
            print(f"[paper] QUEUED     | {name} ({ticker}) | mint={mint[:8]}{label}")

        await ingester.upsert_token(event)
        if dev:
            asyncio.create_task(_profile_and_store_wallet(dev, ingester, driver))

        # Run LLM pipeline for non-skipped tokens
        if not event.get("hard_skip") and ollama_ok:
            asyncio.create_task(_run_llm_decision(event, ingester, market_ctx_ref))

    async def on_trade(event):
        hit = sm.check_trade(event)
        if hit:
            side   = event.get("side", "?").upper()
            ticker = event.get("ticker", "?") or "?"
            sol    = event.get("sol_amount", 0.0)
            mcap   = event.get("market_cap_sol", 0.0)
            print(f"[paper] ★ {hit['label']} {side} {ticker} | {sol:.2f} SOL | mcap={mcap:.0f} SOL")
            await ingester.write_signal({
                "token_mint":    event.get("mint", ""),
                "decision":      f"WHALE_{side}",
                "confidence":    0.0,
                "reasoning":     (
                    f"{hit['label']} {event.get('trader','')[:8]} {side} "
                    f"{sol:.2f} SOL at mcap={mcap:.0f} SOL"
                ),
                "model_version": "smart_money_v1",
            })

    async def on_graduation(event):
        mint = event.get("mint", "")
        name = event.get("name", "") or "?"
        print(f"[paper] GRADUATED  | {name} | mint={mint[:8] if mint else '?'}")
        if mint:
            await ingester.mark_graduated(mint)

    await listen(on_new_token=on_new_token, on_trade=on_trade, on_graduation=on_graduation)


# ── Validate mode — batch re-evaluate stored tokens ───────────────────────────

async def run_validate():
    print("[validate] Batch LLM evaluation — not yet implemented")
    print("[validate] Will re-evaluate stored tokens without rugcheck hard-skip constraint")
    await asyncio.sleep(3600)


# ── Live mode — real execution (Phase 4) ──────────────────────────────────────

async def run_live():
    print("[live] Live trading — requires Phase 4 execution layer (not yet built)")
    print("[live] Use --mode paper to run the full pipeline without real trades")
    await asyncio.sleep(3600)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    print(f"[nathanai-crypto] Starting — mode={args.mode}")

    modes = {
        "collect":  run_collect,
        "paper":    run_paper,
        "validate": run_validate,
        "live":     run_live,
    }
    await modes[args.mode]()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[nathanai-crypto] Shutting down.")
        sys.exit(0)
