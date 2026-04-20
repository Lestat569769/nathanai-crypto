"""
NathanAI Crypto Bot — Entry Point
Usage:
  python main.py --mode collect    # collect only — no LLM, pure data gathering
  python main.py --mode paper      # full pipeline: collect + LLM + paper trade logging
  python main.py --mode validate   # batch re-evaluate stored tokens (backfill)
  python main.py --mode live       # full pipeline + real on-chain execution (Phase 4)

LLM Evaluation Strategy (paper/live mode)
==========================================
qwen3:8b takes 15-30s per decision. Pump.fun launches 3-5 tokens/min.
Evaluating every token would create a queue backlog of hundreds of items.

Two-stage filter:
  Stage 1 — Launch quality gate (at token creation):
    Only queue for LLM if: initial_buy >= 0.1 SOL OR dev is known smart money
    Filters out the ~70% of tokens where the dev barely invested anything.

  Stage 2 — Graduation progress trigger (from trade events):
    subscribeTokenTrade gives us v_sol_in_curve in real time.
    Re-evaluate ANY token (including filtered ones) when it crosses:
      10 SOL → 25 SOL → 50 SOL → 70 SOL in the bonding curve
    This catches "sleeper" tokens that gain traction after a quiet launch.

LLM queue: single asyncio.Queue worker — one inference at a time.
    Prevents Ollama timeout cascade. Max 10 pending items; older tokens dropped
    when capacity is exceeded (stale decisions are worthless anyway).
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Load .env before anything else so all os.getenv() calls pick up the values.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

log = logging.getLogger("nathanai.crypto.main")

# ── Thresholds ────────────────────────────────────────────────────────────────

BUY_CONFIDENCE_THRESHOLD = 0.75   # minimum LLM confidence to act on BUY

# Stage 1 — launch filter
MIN_LAUNCH_BUY_SOL = 0.1          # dev must invest at least this much at launch

# Stage 2 — bonding curve milestone triggers (SOL in curve)
GRADUATION_TARGET  = 85.0         # ~$69K market cap = graduation to Raydium
GRAD_THRESHOLDS    = [10.0, 25.0, 50.0, 70.0]  # evaluate at each milestone

# LLM queue capacity — drop excess to avoid stale decisions
LLM_QUEUE_MAX = 10


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


async def _refresh_market_ctx_loop(market_ctx_ref: list):
    """Update market context every 5 minutes."""
    from collectors.sol_context import get_market_context
    while True:
        try:
            ctx = await get_market_context()
            market_ctx_ref[0] = ctx
        except Exception as e:
            log.warning("market context refresh failed: %s", e)
        await asyncio.sleep(300)


# ── LLM decision pipeline ─────────────────────────────────────────────────────

async def _run_llm_decision(event: dict, ingester, market_ctx_ref: list, trigger: str = "launch"):
    """
    Full LLM decision pipeline for one token event.
    Called by the single LLM queue worker — never directly as a create_task.

    trigger: "launch" | "10sol" | "25sol" | "50sol" | "70sol"
    """
    from adapter.prompt_builder  import build_messages
    from adapter.inference       import call_primary
    from adapter.validator       import validate
    from adapter.decision_parser import parse

    mint   = event.get("mint", "")
    name   = event.get("name", "?") or "?"
    ticker = event.get("ticker", "?") or "?"

    market_ctx = market_ctx_ref[0] if market_ctx_ref else None

    # Similar graduated tokens from vector index
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

    # Smart money buys already recorded for this token
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

    # Primary decision
    raw = await call_primary(messages)
    if not raw:
        log.warning("LLM: no response for %s [%s]", mint[:8], trigger)
        return

    parsed = parse(raw, token_event=event)
    if not parsed:
        log.warning("LLM: unparseable response for %s: %s", mint[:8], raw[:100])
        return

    # Hallucination gate (only for high-confidence BUY)
    gate_result = {"valid": True, "reason": "not_required"}
    if parsed.decision == "BUY" and parsed.confidence >= BUY_CONFIDENCE_THRESHOLD:
        gate_result = await validate(event, parsed)

    final_decision = parsed.decision if gate_result["valid"] else "SKIP"

    _log_decision(name, ticker, mint, parsed, final_decision, gate_result, trigger)

    await ingester.write_signal({
        "token_mint":    mint,
        "decision":      final_decision,
        "confidence":    parsed.confidence,
        "reasoning":     (
            f"[{trigger}] " + parsed.reasoning
            + (f" [GATE REJECTED: {gate_result['reason']}]" if not gate_result["valid"] else "")
        ),
        "model_version": f"qwen3:8b+4b_v1",
    })


def _log_decision(name, ticker, mint, parsed, final_decision, gate_result, trigger):
    conf_pct  = int(parsed.confidence * 100)
    gate_str  = f" [GATE REJECTED: {gate_result['reason'][:50]}]" if not gate_result["valid"] else ""
    trig_str  = f"[{trigger}] "

    if final_decision == "BUY":
        log.info("★ BUY  %s| %s (%s) | conf=%d%% | %s",
                 trig_str, name, ticker, conf_pct, parsed.reasoning[:80])
    else:
        log.info("  SKIP %s| %s (%s) | conf=%d%% | %s%s",
                 trig_str, name, ticker, conf_pct, parsed.reasoning[:80], gate_str)


# ── Collect mode — data only, no LLM ─────────────────────────────────────────

async def run_collect():
    import os
    from neo4j import AsyncGraphDatabase
    from collectors.pumpportal_ws import listen
    from collectors import smart_money as sm
    from graph.schema import init_schema
    from graph.ingest import GraphIngester

    neo4j_uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
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
        hit = sm.check_trade(event)
        if hit:
            side, ticker = event.get("side","?").upper(), event.get("ticker","?") or "?"
            print(f"[collect] ★ {hit['label']} {side} {ticker} | {event.get('sol_amount',0):.2f} SOL")
            await ingester.write_signal({
                "token_mint": event.get("mint",""), "decision": f"WHALE_{side}",
                "confidence": 0.0, "model_version": "smart_money_v1",
                "reasoning": f"{hit['label']} {event.get('trader','')[:8]} {side} {event.get('sol_amount',0):.2f} SOL",
            })

    async def on_graduation(event):
        mint = event.get("mint","")
        print(f"[collect] GRADUATED | {event.get('name','?')} | mint={mint[:8] if mint else '?'}")
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

    neo4j_uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "crypto123")
    driver     = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    await init_schema(driver)
    ingester = GraphIngester(driver)

    await sm.refresh(driver)
    ollama_ok = await check_ollama_available()
    if not ollama_ok:
        print("[paper] WARNING: Ollama not available. LLM decisions disabled.")

    market_ctx_ref = [None]
    try:
        market_ctx_ref[0] = await get_market_context()
    except Exception as e:
        log.warning("initial market context fetch failed: %s", e)

    print("[paper] Pipeline ready — quality-filtered LLM decisions (paper trade mode)")
    print(f"[paper] {sm.get_cache().summary()}")
    print(f"[paper] Launch filter: initial_buy >= {MIN_LAUNCH_BUY_SOL} SOL")
    print(f"[paper] Grad triggers: {GRAD_THRESHOLDS} SOL in bonding curve")
    if market_ctx_ref[0]:
        print(f"[paper] {market_ctx_ref[0].get('context_summary','')}")

    asyncio.create_task(_refresh_smart_money_loop(driver))
    asyncio.create_task(_refresh_market_ctx_loop(market_ctx_ref))

    # ── LLM queue (single worker) ─────────────────────────────────────────
    llm_queue   = asyncio.Queue()
    llm_sent    = {}       # mint → last trigger queued (avoids duplicate evals)
    token_cache = {}       # mint → event dict (for graduation re-evaluation)

    async def _llm_worker():
        while True:
            event, trigger = await llm_queue.get()
            try:
                await _run_llm_decision(event, ingester, market_ctx_ref, trigger)
            except Exception as e:
                log.warning("LLM worker error [%s] %s: %s",
                            trigger, event.get("mint","")[:8], e)
            finally:
                llm_queue.task_done()

    asyncio.create_task(_llm_worker())

    async def _enqueue(event: dict, trigger: str):
        """Add to LLM queue, respecting deduplication and capacity limits."""
        mint = event.get("mint", "")
        if not mint or not ollama_ok:
            return
        if llm_sent.get(mint) == trigger:
            return   # already queued for this trigger
        if llm_queue.qsize() >= LLM_QUEUE_MAX:
            log.debug("LLM queue full — dropping %s [%s]", mint[:8], trigger)
            return
        llm_sent[mint] = trigger
        await llm_queue.put((event, trigger))
        log.info("LLM queued [%s] %s (%s) — %d in queue",
                 trigger, event.get("name","?"), event.get("ticker","?"), llm_queue.qsize())

    # ── New token handler ─────────────────────────────────────────────────
    async def on_new_token(event):
        if event.get("_metadata_update"):
            await ingester.upsert_token(event)
            return

        mint        = event.get("mint", "")
        name        = event.get("name", "?") or "?"
        ticker      = event.get("ticker", "?") or "?"
        dev         = event.get("dev", "")
        initial_buy = event.get("initial_buy_sol", 0.0)

        sm.check_dev(event)

        if sm.is_banned(dev):
            print(f"[paper] BANNED DEV | {name} ({ticker}) | mint={mint[:8]}")
        elif event.get("hard_skip"):
            print(f"[paper] HARD SKIP  | {name} ({ticker}) | mint={mint[:8]}")
        else:
            label = f" ★ {event['dev_smart_money']['label']}" if event.get("dev_smart_money") else ""
            is_sm = bool(event.get("dev_smart_money"))
            passes_filter = initial_buy >= MIN_LAUNCH_BUY_SOL or is_sm

            if passes_filter:
                print(f"[paper] QUEUED     | {name} ({ticker}) | mint={mint[:8]} | "
                      f"{initial_buy:.3f} SOL buy{label}")
                await _enqueue(event, "launch")
            else:
                print(f"[paper] FILTERED   | {name} ({ticker}) | mint={mint[:8]} | "
                      f"{initial_buy:.3f} SOL buy (< {MIN_LAUNCH_BUY_SOL} SOL)")

            # Cache for graduation trigger regardless of filter
            token_cache[mint] = dict(event)

        await ingester.upsert_token(event)
        if dev:
            asyncio.create_task(_profile_and_store_wallet(dev, ingester, driver))

    # ── Trade event handler ───────────────────────────────────────────────
    async def on_trade(event):
        mint  = event.get("mint", "")
        v_sol = event.get("v_sol_in_curve", 0.0)

        # Whale buy detection
        hit = sm.check_trade(event)
        if hit:
            side   = event.get("side", "?").upper()
            ticker = event.get("ticker", "?") or "?"
            sol    = event.get("sol_amount", 0.0)
            mcap   = event.get("market_cap_sol", 0.0)
            print(f"[paper] ★ {hit['label']} {side} {ticker} | "
                  f"{sol:.2f} SOL | mcap={mcap:.0f} SOL | curve={v_sol:.0f}/{GRADUATION_TARGET:.0f}")
            await ingester.write_signal({
                "token_mint":    mint,
                "decision":      f"WHALE_{side}",
                "confidence":    0.0,
                "model_version": "smart_money_v1",
                "reasoning":     (
                    f"{hit['label']} {event.get('trader','')[:8]} {side} "
                    f"{sol:.2f} SOL at mcap={mcap:.0f} SOL"
                ),
            })

        # Graduation milestone trigger — re-evaluate when curve crosses thresholds
        if mint and v_sol > 0 and mint in token_cache:
            cached = token_cache[mint]
            # Update with latest market data
            cached["v_sol_in_curve"] = v_sol
            cached["market_cap_sol"] = event.get("market_cap_sol", cached.get("market_cap_sol", 0))

            for threshold in GRAD_THRESHOLDS:
                trig = f"{threshold:.0f}sol"
                if v_sol >= threshold and llm_sent.get(mint, "") != trig:
                    pct = v_sol / GRADUATION_TARGET * 100
                    name   = cached.get("name", "?") or "?"
                    ticker = cached.get("ticker", "?") or "?"
                    print(f"[paper] MILESTONE  | {name} ({ticker}) | "
                          f"{v_sol:.0f}/{GRADUATION_TARGET:.0f} SOL ({pct:.0f}%) — re-evaluating")
                    await _enqueue(cached, trig)
                    break  # only trigger lowest unmet threshold per trade event

    # ── Graduation handler ────────────────────────────────────────────────
    async def on_graduation(event):
        mint = event.get("mint", "")
        name = event.get("name", "") or "?"
        pct  = "100"
        print(f"[paper] GRADUATED  | {name} | mint={mint[:8] if mint else '?'} | 🎓 {pct}%")
        if mint:
            await ingester.mark_graduated(mint)
            token_cache.pop(mint, None)   # free memory

    await listen(on_new_token=on_new_token, on_trade=on_trade, on_graduation=on_graduation)


# ── Validate / Live modes (stubs) ─────────────────────────────────────────────

async def run_validate():
    print("[validate] Batch LLM evaluation — not yet implemented")
    await asyncio.sleep(3600)

async def run_live():
    print("[live] Live trading — requires Phase 4 execution layer (not yet built)")
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
