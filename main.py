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
    from collectors.pumpfun_ws import listen

    def on_new_token(event):
        status = "HARD SKIP" if event.get("hard_skip") else "QUEUED"
        print(f"[collector] {status} | {event['name']} ({event['ticker']}) | mint={event['mint'][:8]}")

    def on_trade(event):
        pass  # TODO: update bonding curve snapshot in Neo4j

    def on_graduation(event):
        print(f"[collector] GRADUATED | {event.get('name')} | mint={event.get('mint','')[:8]}")

    await listen(on_new_token=on_new_token, on_trade=on_trade, on_graduation=on_graduation)


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
