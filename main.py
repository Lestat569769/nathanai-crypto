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


async def run_collect():
    import os
    from neo4j import AsyncGraphDatabase
    from collectors.solana_ws import listen
    from graph.schema import init_schema
    from graph.ingest import GraphIngester

    # Connect to Neo4j and ensure schema is ready
    neo4j_uri  = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
    neo4j_user = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "crypto123")
    driver     = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    await init_schema(driver)
    ingester = GraphIngester(driver)

    print("[collector] Neo4j ready — listening for pump.fun token launches...")

    async def on_new_token(event):
        status = "HARD SKIP" if event.get("hard_skip") else "QUEUED "
        name   = event.get("name", "?") or "?"
        ticker = event.get("ticker", "?") or "?"
        mint   = event.get("mint", "")
        print(f"[collector] {status} | {name} ({ticker}) | mint={mint[:8]}")
        # Write to Neo4j (both skipped and queued — all are training data)
        await ingester.upsert_token(event)

    async def on_graduation(event):
        sig  = event.get("signature", "")[:12]
        print(f"[collector] GRADUATED | sig={sig}")

    await listen(on_new_token=on_new_token, on_graduation=on_graduation)


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
