"""
adapter/prompt_builder.py
=========================
Assemble all available signals into a structured prompt for qwen3:8b.

Every field is explicitly labelled so the LLM has grounded facts to reason
about — no vague descriptions, only numbers and flags.

Output format: list of Ollama chat messages
  [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": <assembled token context>},
  ]

Usage:
  from adapter.prompt_builder import build_messages
  messages = build_messages(token_event, market_ctx, similar_tokens, whale_signals)
  # pass messages directly to inference.call_primary(messages)
"""
import logging
from typing import Optional

log = logging.getLogger("nathanai.crypto.adapter.prompt_builder")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are NathanAI, an expert Solana trading analyst specializing in pump.fun token launches.

Your job: decide whether a newly launched token will graduate from the pump.fun bonding curve (~$69K market cap) to Raydium within 24 hours.

Graduation base rate: ~1-2% of all pump.fun launches. You should be selective.

Key graduation signals (positive):
- Dev wallet with 40%+ historical grad rate and 5+ prior launches
- No mint/freeze authority (immutable supply)
- Top holder below 15% (no whale trap)
- Twitter + Telegram both present
- Initial buy ≥ 1 SOL from dev (skin in the game)
- Known whale/smart money wallet early buy
- Similar tokens to past graduates (vector search match ≥ 0.85)
- Bonding curve already 20%+ filled at launch
- Fear & Greed index ≥ 55 (Greed) — market momentum supports meme launches

Key rug/fail signals (negative):
- Unknown dev wallet (0 prior launches)
- Top holder > 20%
- No social presence
- Risk penalty ≥ 5 from rugcheck
- Insider buying network detected
- Market Fear & Greed < 30 (Extreme Fear)

Output ONLY valid JSON. No markdown. No text outside the JSON.
Schema: {"decision": "BUY"|"SKIP", "confidence": 0.0-1.0, "reasoning": "1-2 sentences max"}"""


# ── Section builders ──────────────────────────────────────────────────────────

def _token_identity(event: dict) -> str:
    name   = event.get("name", "?") or "?"
    ticker = event.get("ticker", "?") or "?"
    mint   = event.get("mint", "")
    uri    = event.get("uri", "")

    twitter  = "Yes" if event.get("twitter")  else "No"
    telegram = "Yes" if event.get("telegram") else "No"
    website  = "Yes" if event.get("website")  else "No"
    desc     = (event.get("description") or "")[:120]

    lines = [
        "TOKEN IDENTITY",
        f"  name:     {name}",
        f"  ticker:   {ticker}",
        f"  mint:     {mint[:16]}...",
        f"  twitter:  {twitter}  telegram: {telegram}  website: {website}",
    ]
    if desc:
        lines.append(f"  description: {desc}")
    return "\n".join(lines)


def _rugcheck_section(event: dict) -> str:
    rc = event.get("rugcheck", {})
    if not rc:
        return "RUGCHECK\n  No report available (very new token)"

    score     = rc.get("score_normalised", "?")
    mint_auth = rc.get("mint_authority", "?")
    frz_auth  = rc.get("freeze_authority", "?")
    top_pct   = rc.get("top_holder_pct", 0.0)
    lp_locked = rc.get("lp_locked", False)
    insider   = rc.get("insider_detected", False)
    penalty   = rc.get("risk_penalty", 0)
    c_total   = rc.get("creator_total", 0)
    c_grads   = rc.get("creator_graduates", 0)
    c_rugs    = rc.get("creator_rugs", 0)

    lines = [
        "RUGCHECK",
        f"  score:             {score}/100  (risk_penalty={penalty})",
        f"  mint_authority:    {mint_auth}",
        f"  freeze_authority:  {frz_auth}",
        f"  top_holder:        {top_pct:.1f}%",
        f"  lp_locked:         {lp_locked}",
        f"  insider_detected:  {insider}",
        f"  creator_history:   {c_grads}/{c_total} prior launches graduated"
          + (f", {c_rugs} rugs" if c_rugs else ""),
    ]

    flags = rc.get("risk_flags", [])
    if flags:
        flag_str = "  |  ".join(f"{f['level']}:{f['name']}" for f in flags[:5])
        lines.append(f"  risk_flags:        {flag_str}")

    return "\n".join(lines)


def _dev_wallet_section(event: dict) -> str:
    dev     = event.get("dev", "")
    if not dev:
        return "DEV WALLET\n  address: unknown"

    # Check if wallet profile was fetched (upserted from solana_rpc)
    # These fields get written to Neo4j and can be carried in the event
    launches  = event.get("dev_launches",  event.get("tokens_launched",   "unknown"))
    grads     = event.get("dev_graduates", event.get("graduates_launched", "unknown"))
    rugs      = event.get("dev_rugs",      event.get("rugs_launched",      0))
    grad_rate = event.get("dev_grad_rate", event.get("grad_rate",          None))
    rc_risk   = event.get("dev_rc_risk",   event.get("rc_wallet_risk",     "unknown"))
    flagged   = event.get("dev_flagged",   event.get("rc_flagged",         False))

    sm_hit = event.get("dev_smart_money")
    sm_str = f"  smart_money_label: {sm_hit['label']}" if sm_hit else ""

    grad_str = f"{grad_rate*100:.0f}%" if isinstance(grad_rate, float) else "unknown"

    lines = [
        "DEV WALLET",
        f"  address:   {dev[:16]}...",
        f"  launches:  {launches}  graduates: {grads}  rugs: {rugs}",
        f"  grad_rate: {grad_str}",
        f"  rc_risk:   {rc_risk}  flagged: {flagged}",
    ]
    if sm_str:
        lines.append(sm_str)
    return "\n".join(lines)


def _bonding_curve_section(event: dict) -> str:
    init_buy  = event.get("initial_buy_sol", 0.0)
    mcap      = event.get("market_cap_sol", 0.0)
    v_sol     = event.get("v_sol_in_curve", 0.0)
    target    = 85.0  # ~$69K SOL target to graduate

    progress = (v_sol / target * 100) if v_sol and target else 0.0

    lines = [
        "BONDING CURVE",
        f"  initial_buy_sol:   {init_buy:.3f} SOL",
        f"  market_cap_sol:    {mcap:.1f} SOL",
        f"  sol_in_curve:      {v_sol:.1f} SOL  (target ~{target:.0f} SOL to graduate)",
        f"  curve_progress:    {progress:.1f}%",
    ]
    return "\n".join(lines)


def _similar_tokens_section(similar: list) -> str:
    if not similar:
        return "SIMILAR PAST TOKENS\n  No similar tokens found in history yet"

    lines = ["SIMILAR PAST TOKENS (vector similarity to past launches)"]
    for t in similar[:5]:
        graduated = "graduated ✓" if t.get("graduated") else "did not graduate ✗"
        score     = t.get("score", 0.0)
        name      = t.get("name", "?") or "?"
        ticker    = t.get("ticker", "?") or "?"
        lines.append(f"  - {name} ({ticker})  {graduated}  similarity={score:.3f}")
    return "\n".join(lines)


def _whale_signals_section(whale_signals: list) -> str:
    if not whale_signals:
        return "SMART MONEY SIGNALS\n  No smart money buys detected"

    lines = [f"SMART MONEY SIGNALS ({len(whale_signals)} detected)"]
    for sig in whale_signals[:5]:
        reasoning = sig.get("reasoning", "")
        lines.append(f"  - {reasoning}")
    return "\n".join(lines)


def _market_context_section(ctx: Optional[dict]) -> str:
    if not ctx:
        return "MARKET CONTEXT\n  Not available"

    price  = ctx.get("sol_price_usd", 0.0)
    change = ctx.get("sol_change_24h", 0.0)
    fg_val = ctx.get("fear_greed_value", 50)
    fg_lbl = ctx.get("fear_greed_label", "Neutral")
    news   = ctx.get("news_headlines", [])

    change_str = f"{change:+.1f}%"
    news_str   = " | ".join(news[:3]) if news else "None"

    lines = [
        "MARKET CONTEXT",
        f"  SOL price:     ${price:.2f} ({change_str} 24h)",
        f"  fear_greed:    {fg_val}/100 ({fg_lbl})",
        f"  recent_news:   {news_str}",
    ]
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_messages(
    token_event:   dict,
    market_ctx:    Optional[dict]  = None,
    similar_tokens: list           = None,
    whale_signals:  list           = None,
) -> list[dict]:
    """
    Assemble the full prompt as an Ollama chat messages list.

    Parameters
    ----------
    token_event    : dict from pumpportal_ws + rugcheck + solana_rpc + smart_money
    market_ctx     : dict from sol_context.get_market_context() — may be None
    similar_tokens : list from graph.ingest.find_similar_tokens() — may be None/empty
    whale_signals  : list of Signal dicts for WHALE_BUY signals on this token

    Returns
    -------
    [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    Ready to pass to inference.call_primary(messages).
    """
    similar_tokens = similar_tokens or []
    whale_signals  = whale_signals  or []

    sections = [
        _token_identity(token_event),
        "",
        _rugcheck_section(token_event),
        "",
        _dev_wallet_section(token_event),
        "",
        _bonding_curve_section(token_event),
        "",
        _similar_tokens_section(similar_tokens),
        "",
        _whale_signals_section(whale_signals),
        "",
        _market_context_section(market_ctx),
        "",
        'Output JSON: {"decision": "BUY"|"SKIP", "confidence": 0.0-1.0, "reasoning": "1-2 sentences"}',
    ]

    user_content = "\n".join(sections)
    log.debug("prompt built: %d chars for mint=%s",
              len(user_content), token_event.get("mint", "")[:8])

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
