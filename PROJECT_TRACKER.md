# NathanAI Crypto Bot — Project Tracker
**Last Updated:** 2026-04-20
**Status:** Active Development — Phase 1 (Data Pipeline)

---

## Overview

A fully automated Solana trading bot that:
1. Watches every new token launch on pump.fun in real time
2. Screens each token through rugcheck.xyz (hard skip gate)
3. Reasons about graduation probability using Qwen3-4B via Ollama
4. Uses a two-model hallucination gate (qwen3:8b decision + qwen3:4b validator)
5. Stores everything in Neo4j with built-in vector search (nomic-embed-text, 768-dim)
6. Executes buys on-chain via pump.fun program + Jupiter DEX for sells

**Constraints:** US-accessible only, zero paid API subscriptions, SOL-only, on-chain execution

---

## Repository

| Repo | URL | Local Path |
|------|-----|------------|
| nathanai-crypto | https://github.com/Lestat569769/nathanai-crypto | `D:\Docker\ComfyUI_windows_portable\n8n\nathanai-crypto` |
| nathanai-evolution | https://github.com/Lestat569769/nathanai-evolution | `D:\Docker\ComfyUI_windows_portable\n8n\ai neuron` |

---

## Infrastructure

### Docker Stack (all running locally)

| Container | Image | Ports | Status |
|-----------|-------|-------|--------|
| nathanai-crypto-neo4j | neo4j:5.18 | 7474 (browser), 7687 (bolt) | ✅ Healthy |
| nathanai-crypto-collector | nathanai-crypto-collector | — | ✅ Up (backoff on pump.fun 530) |
| nathanai-crypto-validator | nathanai-crypto-validator | — | ✅ Up |
| nathanai-crypto-dashboard | nathanai-crypto-dashboard | 8765 | ✅ Up |

**Neo4j Browser:** http://localhost:7474 (neo4j / crypto123)

### Ollama Models (existing, no new instances needed)

| Model | Purpose | Size |
|-------|---------|------|
| qwen3:8b | Primary decision engine (BUY/SKIP) | 5.2 GB |
| qwen3:4b | Hallucination gate validator | 2.5 GB |
| nomic-embed-text | Token + wallet embeddings (768-dim) | 274 MB |

### External Services

| Service | Purpose | Status |
|---------|---------|--------|
| rugcheck.xyz | Rug detection hard-skip gate | ✅ Working |
| **PumpPortal WebSocket** | New token events — free, no API key, no credits | ✅ LIVE — receiving real tokens |
| Helius RPC | getTransaction fallback (0 credits if unused) | ✅ Configured (key: Stagbog) |
| PumpPortal Trading API | On-chain buy execution (Phase 4) | Pending |
| Solana public RPC | Dev wallet history lookup | ✅ Built |
| DexScreener | Post-graduation price tracking | Pending build |
| CoinGecko | SOL price + market context | Pending build |
| alternative.me | Fear & Greed index | Pending build |

---

## Neo4j Schema

### Vector Indexes (built-in Neo4j 5.18, 768-dim cosine)

| Index Name | Node | Property | Purpose |
|------------|------|----------|---------|
| token_context_vector | Token | context_embedding | Find similar past graduates |
| wallet_profile_vector | Wallet | profile_embedding | Classify unknown dev wallets |

### Standard Indexes & Constraints
- 5 uniqueness constraints (Token.mint, Wallet.address, Signal.id, Trade.id, NewsEvent.id)
- 8 range indexes (token dates, graduation status, ticker, wallet grad_rate, signal fields)

---

## LLM Architecture

```
Token passes rugcheck
       ↓
Context Builder assembles structured prompt
       ↓
qwen3:8b (no-think) → BUY/SKIP + confidence + JSON reasoning
       ↓
  If BUY + confidence ≥ 0.75:
       ↓
  Validator: qwen3:4b — "Does this reasoning contradict the facts? Y/N"
       ↓
  Both agree BUY → Risk Manager → Execute (or log for paper trading)
```

**Hallucination gate checks:**
1. Schema validation (confidence float 0.0-1.0, decision must be BUY or SKIP)
2. Fact grounding (no fabricated wallet stats in reasoning)
3. Confidence calibration (BUY 0.95 with rug signals present → auto-reject)

---

## Risk Management (hardcoded, no LLM override)

| Setting | Value |
|---------|-------|
| Max position per token | 1% of portfolio |
| Max simultaneous positions | 5 |
| Min confidence to execute | 0.75 |
| Auto-exit win | Sell at Raydium graduation |
| Auto-exit loss | Cut if bonding curve stalls > 30 min |
| Session kill switch | Halt if portfolio drops 10% in one session |
| Daily loss limit | 15% |
| Max daily trades | 50 |

---

## Build Progress

### ✅ COMPLETE

| Item | Date | Notes |
|------|------|-------|
| MASTER_BUILD_PLAN.md | 2026-04-19 | Full 14-section plan |
| collectors/rugcheck.py | 2026-04-19 | Hard skip gate — mint_auth, freeze_auth, top_holder, risk_penalty |
| collectors/pumpfun_ws.py | 2026-04-19 | WebSocket listener (kept as reference — blocked by Cloudflare 530) |
| collectors/pumpportal_ws.py | 2026-04-20 | **ACTIVE** — PumpPortal WebSocket, free, live, real token names |
| collectors/solana_ws.py | 2026-04-20 | Solana RPC fallback (kept for reference) |
| collectors/solana_rpc.py | 2026-04-20 | Dev wallet profiler — rugcheck wallet API + tx history |
| Docker stack | 2026-04-20 | docker-compose.yml, Dockerfile, requirements.txt |
| graph/schema.py | 2026-04-20 | All constraints, indexes, vector indexes — applied to live Neo4j |
| graph/ingest.py | 2026-04-20 | GraphIngester: upsert_token, upsert_wallet, write_signal, write_trade, vector similarity search |
| monitor/dashboard.py | 2026-04-20 | Rich live dashboard — polls Neo4j every 5s |
| main.py | 2026-04-20 | Entrypoint: --mode collect|validate|paper|live |

### 🔄 IN PROGRESS

| Item | Blocked By | Notes |
|------|-----------|-------|
| collectors/sol_context.py | — | SOL price + fear/greed + RSS news |

### 📋 PENDING — Phase 1 (Data Pipeline)

| Item | Priority | Notes |
|------|----------|-------|
| collectors/sol_context.py | MEDIUM | CoinGecko SOL price + alternative.me fear/greed + RSS |
| collectors/dexscreener.py | MEDIUM | Post-graduation price/volume tracking |
| collectors/smart_money.py | HIGH | Whale/KOL wallet reputation cache |
| scripts/verify_setup.py | LOW | Connection health check script |

### 📋 PENDING — Phase 2 (Historical Backfill)

| Item | Priority | Notes |
|------|----------|-------|
| collectors/backfill.py | MEDIUM | Pull 90 days of pump.fun launches from on-chain |
| scripts/backfill_history.py | MEDIUM | One-time run: load historical data into Neo4j |
| graph/queries.py | MEDIUM | Analytics queries — grad rate by rugcheck score, etc. |

### 📋 PENDING — Phase 3 (LLM Adapter)

| Item | Priority | Notes |
|------|----------|-------|
| adapter/prompt_builder.py | HIGH | Assemble all signals → structured LLM prompt |
| adapter/inference.py | HIGH | qwen3:8b via Ollama — BUY/SKIP + confidence |
| adapter/validator.py | HIGH | qwen3:4b hallucination gate |
| adapter/decision_parser.py | HIGH | Parse BUY/SKIP + confidence from model output |
| training/nero_extract.py | MEDIUM | Generate thinking-chain Q&A from Neo4j history |
| training/grpo_reward.py | MEDIUM | Graduation prediction reward function |

### 📋 PENDING — Phase 4 (Execution)

| Item | Priority | Notes |
|------|----------|-------|
| risk/position_sizer.py | HIGH | 1% of portfolio in SOL |
| risk/circuit_breaker.py | HIGH | Session loss tracking + kill switch |
| execution/wallet.py | HIGH | Keypair load, balance check, sign tx |
| execution/pumpfun_buyer.py | HIGH | On-chain buy on bonding curve |
| execution/jupiter_seller.py | HIGH | Sell via Jupiter post-graduation |

### 📋 PENDING — Phase 5 (Paper Trading, 14 days minimum)

| Item | Notes |
|------|-------|
| scripts/run_paper_trade.py | Full live pipeline, no real execution |
| Paper trading go/no-go review | Accuracy > 55% required to proceed to live |

### 📋 PENDING — Phase 6 (Live Trading)

| Item | Notes |
|------|-------|
| Solana wallet setup + funded | Only capital you can afford to lose |
| Live trading with 0.5% position size (first week) | Monitor daily |
| GRPO feedback loop from real trade outcomes | Month 2+ |

---

## Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Vector store | Neo4j built-in (5.18) | No external Qdrant/Pinecone needed |
| Embedding model | nomic-embed-text (768-dim) | Already in Ollama, Apache 2.0, fast |
| Primary LLM | qwen3:8b no-think mode | 0.880 function-calling score, fast |
| Gate LLM | qwen3:4b no-think mode | Cheap cross-check, same model family |
| Real-time data | Helius LaserStream gRPC | Bypasses Cloudflare, used by serious bots |
| Execution | pump.fun program + Jupiter | On-chain only, no CEX account |
| Training pipeline | NERO → SFT → GRPO | Proven in medical domain, adapts to crypto |

---

## Known Issues / Blockers

| Issue | Severity | Fix |
|-------|----------|-----|
| pump.fun WebSocket HTTP 530 | HIGH | Replace with Helius LaserStream gRPC |
| Solana wallet not yet created | HIGH | Phase 4 prerequisite |
| No historical training data yet | MEDIUM | Need 90-day backfill before NERO training |

---

## Milestone Tracker

| # | Milestone | Status | Target |
|---|-----------|--------|--------|
| 1 | Docker stack running | ✅ DONE | 2026-04-20 |
| 2 | Neo4j schema + vector indexes live | ✅ DONE | 2026-04-20 |
| 3 | Helius gRPC collector live | 🔄 IN PROGRESS | 2026-04-20 |
| 4 | All collectors running (Solana RPC, SOL context, DexScreener) | Pending | Week 1 |
| 5 | 90-day historical backfill complete | Pending | Week 1 |
| 6 | Adapter layer (prompt builder + inference + validator) | Pending | Week 2 |
| 7 | Paper trading live — 14 day run | Pending | Week 3-4 |
| 8 | Go/no-go review (accuracy > 55%) | Pending | Week 4 |
| 9 | Live trading — minimum capital | Pending | Month 2 |
| 10 | GRPO feedback loop from real trades | Pending | Month 2+ |

---

## File Structure (current state)

```
nathanai-crypto/
├── PROJECT_TRACKER.md          ← this file
├── MASTER_BUILD_PLAN.md        ← full architecture + phase plan
├── docker-compose.yml          ← 4 services: neo4j, collector, validator, dashboard
├── Dockerfile                  ← Python 3.11 + Rust (for solders)
├── requirements.txt            ← all Python deps
├── .env.example                ← config template (copy to .env)
├── .gitignore
├── main.py                     ← entrypoint --mode collect|validate|paper|live
│
├── collectors/
│   ├── __init__.py
│   ├── rugcheck.py             ✅ hard skip gate
│   ├── pumpfun_ws.py           ✅ WebSocket (blocked — Cloudflare 530)
│   ├── helius_grpc.py          🔄 NEXT — LaserStream gRPC
│   ├── solana_rpc.py           📋 pending
│   ├── sol_context.py          📋 pending
│   ├── dexscreener.py          📋 pending
│   └── smart_money.py          📋 pending
│
├── graph/
│   ├── __init__.py
│   ├── schema.py               ✅ constraints + indexes + vector indexes
│   ├── ingest.py               ✅ GraphIngester + embedding + similarity search
│   └── queries.py              📋 pending
│
├── adapter/
│   ├── __init__.py
│   ├── prompt_builder.py       📋 pending
│   ├── inference.py            📋 pending
│   ├── validator.py            📋 pending
│   └── decision_parser.py      📋 pending
│
├── risk/
│   ├── __init__.py
│   ├── position_sizer.py       📋 pending
│   └── circuit_breaker.py      📋 pending
│
├── execution/
│   ├── __init__.py
│   ├── wallet.py               📋 pending
│   ├── pumpfun_buyer.py        📋 pending
│   └── jupiter_seller.py       📋 pending
│
├── backtest/
│   └── __init__.py             📋 pending
│
├── training/
│   ├── __init__.py
│   ├── nero_extract.py         📋 pending
│   └── grpo_reward.py          📋 pending
│
├── monitor/
│   ├── __init__.py
│   └── dashboard.py            ✅ Rich live dashboard
│
└── scripts/
    ├── verify_setup.py         📋 pending
    ├── backfill_history.py     📋 pending
    └── run_paper_trade.py      📋 pending
```
