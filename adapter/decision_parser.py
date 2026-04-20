"""
adapter/decision_parser.py
==========================
Parse and validate BUY/SKIP decisions from raw LLM output.

Handles all the messy real-world output formats:
  - Clean JSON:     {"decision": "BUY", "confidence": 0.85, "reasoning": "..."}
  - Markdown block: ```json\n{...}\n```
  - JSON in prose:  "After analysis: {"decision": ...}"

Validation layers after parsing:
  1. Schema   — decision in ["BUY","SKIP"], confidence is float 0.0-1.0
  2. Sanity   — confidence matches decision (SKIP at 0.95 = suspicious)
  3. Calibration — BUY + confidence ≥ 0.95 when rug flags are present → reject

Usage:
  from adapter.decision_parser import parse
  result = parse(raw_llm_output, token_event)
  if result:
      print(result.decision, result.confidence, result.reasoning)
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("nathanai.crypto.adapter.decision_parser")


@dataclass
class ParsedDecision:
    decision:   str    # "BUY" | "SKIP"
    confidence: float  # 0.0 – 1.0
    reasoning:  str
    raw:        str    # original model output (for debugging)


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """
    Try several strategies to find a JSON object in model output.
    Returns the parsed dict or None.
    """
    text = text.strip()

    # Strategy 1: clean JSON (whole response is the object)
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strategy 2: markdown code block ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Strategy 3: first { ... } in the text
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Strategy 4: last { ... } in the text (some models put JSON at the end)
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    if matches:
        try:
            return json.loads(matches[-1].group(0))
        except Exception:
            pass

    return None


# ── Schema validation ─────────────────────────────────────────────────────────

def _validate_schema(data: dict) -> Optional[str]:
    """
    Return an error string if the parsed dict violates the required schema.
    Returns None if valid.
    """
    decision = data.get("decision", "")
    if decision not in ("BUY", "SKIP"):
        return f"decision must be 'BUY' or 'SKIP', got: {repr(decision)}"

    confidence = data.get("confidence")
    if confidence is None:
        return "missing 'confidence' field"
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return f"confidence must be a float, got: {repr(confidence)}"
    if not (0.0 <= confidence <= 1.0):
        return f"confidence {confidence} out of range [0.0, 1.0]"

    if "reasoning" not in data:
        return "missing 'reasoning' field"

    return None


# ── Calibration checks ────────────────────────────────────────────────────────

def _calibration_check(data: dict, token_event: Optional[dict]) -> Optional[str]:
    """
    Flag decisions that are almost certainly hallucinated or miscalibrated.
    Returns an error string to reject the decision, or None to accept it.

    Checks:
      - BUY at confidence ≥ 0.95 when rug signals are present
      - SKIP at confidence < 0.40 (model is barely uncertain — likely confused)
      - BUY at confidence < 0.40 (too uncertain to trade)
    """
    decision   = data["decision"]
    confidence = float(data["confidence"])

    # BUY with suspiciously high confidence when rug flags exist
    if decision == "BUY" and confidence >= 0.95 and token_event:
        rc = token_event.get("rugcheck", {})
        risk_penalty = rc.get("risk_penalty", 0)
        insider      = rc.get("insider_detected", False)
        if risk_penalty >= 5 or insider:
            return (
                f"calibration_reject: BUY at confidence={confidence:.2f} but "
                f"risk_penalty={risk_penalty} insider={insider} — overconfident"
            )

    # BUY confidence below execution threshold — treat as SKIP
    if decision == "BUY" and confidence < 0.40:
        return f"calibration_reject: BUY confidence {confidence:.2f} too low to act on"

    return None


# ── Main parse entry point ────────────────────────────────────────────────────

def parse(
    raw: str,
    token_event: Optional[dict] = None,
) -> Optional[ParsedDecision]:
    """
    Parse raw LLM output into a validated ParsedDecision.

    Returns None if the output cannot be parsed or fails validation.
    Logs the specific failure reason at WARNING level.

    token_event is used for calibration checks — pass None to skip them.
    """
    if not raw or not raw.strip():
        log.warning("decision_parser: empty model output")
        return None

    data = _extract_json(raw)
    if not data:
        log.warning("decision_parser: could not extract JSON from: %s", raw[:200])
        return None

    schema_error = _validate_schema(data)
    if schema_error:
        log.warning("decision_parser: schema error — %s | raw=%s", schema_error, raw[:200])
        return None

    calibration_error = _calibration_check(data, token_event)
    if calibration_error:
        log.warning("decision_parser: %s", calibration_error)
        return None

    return ParsedDecision(
        decision   = data["decision"],
        confidence = float(data["confidence"]),
        reasoning  = str(data.get("reasoning", ""))[:1000],
        raw        = raw,
    )


def parse_validator_response(raw: str) -> Optional[dict]:
    """
    Parse the qwen3:4b validator response.
    Expected: {"valid": true|false, "reason": "..."}
    Returns {"valid": bool, "reason": str} or None on failure.
    """
    data = _extract_json(raw)
    if not data:
        log.warning("validator_parser: could not extract JSON from: %s", raw[:200])
        return None

    valid = data.get("valid")
    if not isinstance(valid, bool):
        # Some models return "true"/"false" strings
        if str(valid).lower() in ("true", "yes", "1"):
            valid = True
        elif str(valid).lower() in ("false", "no", "0"):
            valid = False
        else:
            log.warning("validator_parser: 'valid' is not bool: %s", repr(valid))
            return None

    return {
        "valid":  valid,
        "reason": str(data.get("reason", data.get("reasoning", "")))[:500],
    }
