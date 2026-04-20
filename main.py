"""
NathanAI Crypto Bot — Entry Point
Usage:
  python main.py --mode collect    # run pump.fun collector + neo4j ingest
  python main.py --mode validate   # run LLM decision engine
  python main.py --mode paper      # run paper trading (no real execution)
  python main.py --mode live       # run live trading (requires wallet + funds)
"""
import argparse
import asyncio
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="NathanAI Crypto Bot")
    parser.add_argument(
        "--mode",
        choices=["collect", "validate", "paper", "live"],
        required=True,
        help="Operating mode"
    )
    return parser.parse_args()


async def _profile_and_store_wallet(dev: str, ingester, driver):
    """Background task: profile dev wallet and update Neo4j."""
    from collectors.solana_rpc import get_wallet_profile
    try:
        profile = await get_wallet_profile(dev, driver)
        if profile:
            await ingester.upsert_wallet(dev, profile)
    except Exception as e:
        print(f"[collector] wallet profile error for {dev[:8]}: {e}")


async def run_collect():
    import os
    from neo4j import AsyncGraphDatabase
    from collectors.pumpportal_ws import listen
    from collectors import smart_money as sm
    from graph.schema import init_schema
    from graph.ingest import GraphIngester

    # Connect to Neo4j and ensure schema is ready
    neo4j_uri  = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
    neo4j_user = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "crypto123")
    driver     = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    await init_schema(driver)
    ingester = GraphIngester(driver)

    # Initial smart money cache load from Neo4j
    await sm.refresh(driver)
    print("[collector] Neo4j ready — listening for pump.fun token launches...")
    print(f"[collector] {sm.get_cache().summary()}")

    # ── Periodic smart money refresh task ──────────────────────────────────
    async def _refresh_smart_money_loop():
        while True:
            await asyncio.sleep(300)   # every 5 min
            await sm.refresh(driver)

    asyncio.create_task(_refresh_smart_money_loop())

    # ── Token event handler ────────────────────────────────────────────────
    async def on_new_token(event):
        # Metadata updates come through the same callback — just upsert quietly
        if event.get("_metadata_update"):
            await ingester.upsert_token(event)
            return

        mint   = event.get("mint", "")
        name   = event.get("name", "?") or "?"
        ticker = event.get("ticker", "?") or "?"
        dev    = event.get("dev", "")

        # Smart money dev check — annotates event["dev_smart_money"] in-place
        dev_hit = sm.check_dev(event)

        if event.get("hard_skip"):
            # Extra hard-skip if dev is banned
            if sm.is_banned(dev):
                print(f"[collector] BANNED DEV | {name} ({ticker}) | mint={mint[:8]}")
            else:
                print(f"[collector] HARD SKIP  | {name} ({ticker}) | mint={mint[:8]}")
        else:
            label = f" ★ {dev_hit['label']}" if dev_hit else ""
            print(f"[collector] QUEUED     | {name} ({ticker}) | mint={mint[:8]}{label}")

        # Write token to Neo4j
        await ingester.upsert_token(event)

        # Profile the dev wallet async (non-blocking)
        if dev:
            asyncio.create_task(_profile_and_store_wallet(dev, ingester, driver))

    # ── Trade event handler (from subscribeTokenTrade) ─────────────────────
    async def on_trade(event):
        # Check if the buyer is smart money — log prominently if so
        hit = sm.check_trade(event)
        if hit:
            mint   = event.get("mint", "")
            ticker = event.get("ticker", "?") or "?"
            side   = event.get("side", "?").upper()
            sol    = event.get("sol_amount", 0.0)
            mcap   = event.get("market_cap_sol", 0.0)
            print(
                f"[smart_money] ★ {hit['label']} {side} "
                f"{ticker} | {sol:.2f} SOL | mcap={mcap:.0f} SOL | "
                f"trader={event.get('trader','')[:8]}"
            )
            # Write a signal node so this appears in paper trading analysis
            await ingester.write_signal({
                "token_mint":    mint,
                "decision":      f"WHALE_{side}",
                "confidence":    0.0,   # confidence reserved for LLM signals
                "reasoning":     (
                    f"{hit['label']} wallet {event.get('trader','')[:8]} "
                    f"made a {side} of {sol:.2f} SOL at mcap={mcap:.0f} SOL. "
                    f"Grad rate: {(hit.get('grad_rate') or 0)*100:.0f}%"
                ),
                "model_version": "smart_money_v1",
            })

    # ── Graduation handler ─────────────────────────────────────────────────
    async def on_graduation(event):
        mint = event.get("mint", "")
        name = event.get("name", "")
        print(f"[collector] GRADUATED  | {name or '?'} | mint={mint[:8] if mint else '?'}")
        if mint:
            await ingester.mark_graduated(mint)

    await listen(
        on_new_token=on_new_token,
        on_trade=on_trade,
        on_graduation=on_graduation,
    )


async def run_validate():
    print("[validator] LLM decision engine — not yet implemented")
    await asyncio.sleep(3600)


async def run_paper():
    print("[paper] Paper trading mode — not yet implemented")
    await asyncio.sleep(3600)


async def run_live():
    print("[live] Live trading — not yet implemented")
    await asyncio.sleep(3600)


async def main():
    args = parse_args()
    print(f"[nathanai-crypto] Starting in mode: {args.mode}")

    modes = {
        "collect": run_collect,
        "validate": run_validate,
        "paper": run_paper,
        "live": run_live,
    }

    await modes[args.mode]()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[nathanai-crypto] Shutting down.")
        sys.exit(0)
