"""
monitor/dashboard.py
====================
Live terminal dashboard using Rich.
Polls Neo4j every 5 seconds and displays:
  - Open positions
  - Recent signals (BUY/SKIP)
  - P&L summary
  - Hard skip stats
  - Ollama model status
"""
import asyncio
import os
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "crypto123")

console = Console()


def make_header() -> Panel:
    return Panel(
        Text("NathanAI Crypto Bot — Live Dashboard", justify="center", style="bold green"),
        style="green"
    )


def make_stats_table(stats: dict) -> Table:
    table = Table(title="Session Stats", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Tokens seen", str(stats.get("tokens_seen", 0)))
    table.add_row("Hard skipped", str(stats.get("hard_skipped", 0)))
    table.add_row("BUY signals", str(stats.get("buys", 0)))
    table.add_row("SKIP signals", str(stats.get("skips", 0)))
    table.add_row("Graduated (held)", str(stats.get("graduated", 0)))
    table.add_row("P&L (SOL)", str(stats.get("pnl_sol", 0.0)))
    return table


def make_layout(stats: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(make_header(), size=3),
        Layout(make_stats_table(stats)),
    )
    return layout


async def poll_neo4j_stats() -> dict:
    """Try to pull live stats from Neo4j. Returns zeros if not connected yet."""
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Token)
                RETURN
                  count(t)                               AS tokens_seen,
                  sum(CASE WHEN t.rc_hard_skip THEN 1 ELSE 0 END) AS hard_skipped,
                  sum(CASE WHEN t.graduated     THEN 1 ELSE 0 END) AS graduated
                """
            )
            record = await result.single()
            if record:
                return {
                    "tokens_seen":  record["tokens_seen"],
                    "hard_skipped": record["hard_skipped"],
                    "graduated":    record["graduated"],
                    "buys":         0,
                    "skips":        0,
                    "pnl_sol":      0.0,
                }
        await driver.close()
    except Exception:
        pass
    return {"tokens_seen": 0, "hard_skipped": 0, "graduated": 0,
            "buys": 0, "skips": 0, "pnl_sol": 0.0}


async def main():
    console.print("[green]NathanAI Crypto Dashboard starting...[/green]")
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            stats = await poll_neo4j_stats()
            live.update(make_layout(stats))
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
