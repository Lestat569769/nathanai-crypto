"""
adapter/validator.py
====================
Hallucination gate — qwen3:4b cross-checks the primary decision.

Called ONLY when qwen3:8b returns BUY with confidence ≥ 0.75.
Cost: one extra inference call per BUY signal (skipped for SKIP decisions).

What it checks:
  1. Fabrication  — does reasoning reference statistics absent from the facts?
  2. Contradiction — does reasoning contradict the actual token facts?
  3. Overconfidence — "no rug signals" when risk_penalty ≥ 5?

If the validator says invalid → the BUY is downgraded to SKIP.
Both models must agree BUY for a signal to proceed to execution.

Usage:
  from adapter.validator import build_validation_messages, validate
  msgs = build_validation_messages(token_event, parsed_decision)
  result = await validate(token_event, parsed_decision)
  if result["valid"]:
      # proceed to risk manager
"""
import logging
from typing import Optional

from adapter.decision_parser import ParsedDecision, parse_validator_response
from adapter.inference import call_validator

log = logging.getLogger("nathanai.crypto.adapter.validator")

VALIDATION_SYSTEM = """You are a fact-checker for AI trading decisions.

You will receive:
1. FACTS: structured data about a pump.fun token
2. REASONING: a trading decision + explanation from an AI analyst

Your job: check whether the REASONING accurately reflects the FACTS.

Flag as INVALID if the reasoning:
- References statistics not present in the facts (hallucination)
- Contradicts the actual numbers (e.g., calls top_holder "low" when it is 25%)
- Claims "no rug signals" when risk flags are present
- Claims a "proven dev" for an unknown wallet with 0 prior launches
- Is internally inconsistent with the confidence score

Output ONLY valid JSON. No markdown. No text outside the JSON.
Schema: {"valid": true|false, "reason": "1 sentence explanation"}"""


def _build_facts_block(token_event: dict) -> str:
    """
    Render the token facts as plain text for the validator.
    Deliberately structured — numbers are explicit so the validator can
    catch any discrepancy in the primary model's reasoning.
    """
    rc       = token_event.get("rugcheck", {})
    sm_hit   = token_event.get("dev_smart_money")

    name      = token_event.get("name", "?") or "?"
    ticker    = token_event.get("ticker", "?") or "?"
    dev       = (token_event.get("dev") or "")[:16]

    score     = rc.get("score_normalised", "N/A")
    penalty   = rc.get("risk_penalty", 0)
    mint_auth = rc.get("mint_authority", "N/A")
    frz_auth  = rc.get("freeze_authority", "N/A")
    top_pct   = rc.get("top_holder_pct", 0.0)
    insider   = rc.get("insider_detected", False)
    lp_locked = rc.get("lp_locked", False)
    c_total   = rc.get("creator_total", 0)
    c_grads   = rc.get("creator_graduates", 0)
    c_rugs    = rc.get("creator_rugs", 0)
    flags     = rc.get("risk_flags", [])

    twitter   = bool(token_event.get("twitter"))
    telegram  = bool(token_event.get("telegram"))
    website   = bool(token_event.get("website"))
    init_buy  = token_event.get("initial_buy_sol", 0.0)
    mcap      = token_event.get("market_cap_sol", 0.0)

    grad_rate = token_event.get("dev_grad_rate", token_event.get("grad_rate", None))
    launches  = token_event.get("dev_launches", token_event.get("tokens_launched", "unknown"))
    dev_rugs  = token_event.get("dev_rugs", token_event.get("rugs_launched", 0))
    sm_label  = sm_hit["label"] if sm_hit else "None"

    flag_str  = ", ".join(f"{f['level']}:{f['name']}" for f in flags) if flags else "none"
    grad_str  = f"{grad_rate*100:.0f}%" if isinstance(grad_rate, float) else "unknown"

    return f"""FACTS:
  name:              {name}  ticker: {ticker}
  dev_wallet:        {dev}...
  rugcheck_score:    {score}/100  risk_penalty: {penalty}
  mint_authority:    {mint_auth}
  freeze_authority:  {frz_auth}
  top_holder_pct:    {top_pct:.1f}%
  lp_locked:         {lp_locked}
  insider_detected:  {insider}
  risk_flags:        {flag_str}
  creator_history:   {c_grads}/{c_total} prior launches graduated, {c_rugs} rugs
  dev_grad_rate:     {grad_str}  (from {launches} observed launches)
  dev_rugs:          {dev_rugs}
  dev_smart_money:   {sm_label}
  twitter:           {twitter}  telegram: {telegram}  website: {website}
  initial_buy_sol:   {init_buy:.3f} SOL
  market_cap_sol:    {mcap:.1f} SOL"""


def build_validation_messages(
    token_event:    dict,
    parsed:         ParsedDecision,
) -> list[dict]:
    """
    Build the qwen3:4b validation prompt.

    Parameters
    ----------
    token_event : the full token event dict (same one passed to prompt_builder)
    parsed      : ParsedDecision from the primary model

    Returns
    -------
    Ollama chat messages list, ready for inference.call_validator()
    """
    facts_block = _build_facts_block(token_event)

    user_content = f"""{facts_block}

DECISION TO CHECK:
  decision:    {parsed.decision}
  confidence:  {parsed.confidence:.2f}
  reasoning:   {parsed.reasoning}

Does the REASONING accurately reflect the FACTS?
Output JSON: {{"valid": true|false, "reason": "1 sentence"}}"""

    return [
        {"role": "system", "content": VALIDATION_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


async def validate(
    token_event: dict,
    parsed:      ParsedDecision,
) -> dict:
    """
    Run the hallucination gate check.

    Returns
    -------
    {"valid": bool, "reason": str}
    On inference failure, returns {"valid": False, "reason": "validator_unavailable"}
    so the BUY is conservatively rejected rather than blindly executed.
    """
    messages = build_validation_messages(token_event, parsed)
    raw      = await call_validator(messages)

    if not raw:
        log.warning("validator: inference failed — conservatively rejecting BUY")
        return {"valid": False, "reason": "validator_unavailable"}

    result = parse_validator_response(raw)
    if result is None:
        log.warning("validator: could not parse response — rejecting BUY | raw=%s", raw[:200])
        return {"valid": False, "reason": "parse_failure"}

    if result["valid"]:
        log.info("validator: APPROVED — %s", result["reason"])
    else:
        log.warning("validator: REJECTED — %s", result["reason"])

    return result
