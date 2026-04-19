# NathanAI Crypto Bot — Master Build Plan
# SOL / pump.fun Edition

**Created:** 2026-04-19  
**Author:** NathanAI Project  
**Status:** Pre-build — planning complete, ready to execute  

---

## Table of Contents

1. [What We Are Building](#1-what-we-are-building)
2. [Full System Architecture](#2-full-system-architecture)
3. [Tools & Dependencies](#3-tools--dependencies)
4. [Environment Setup](#4-environment-setup)
5. [Repository Structure](#5-repository-structure)
6. [Data Sources & Settings](#6-data-sources--settings)
7. [Neo4j Schema](#7-neo4j-schema)
8. [Build Order — Phase by Phase](#8-build-order--phase-by-phase)
9. [NERO + GRPO Training Plan](#9-nero--grpo-training-plan)
10. [Risk Management Settings](#10-risk-management-settings)
11. [Configuration File Reference](#11-configuration-file-reference)
12. [Paper Trading Protocol](#12-paper-trading-protocol)
13. [Live Trading Checklist](#13-live-trading-checklist)
14. [Milestone Tracker](#14-milestone-tracker)

---

## 1. What We Are Building

A fully automated Solana trading bot that:

1. **Watches** every new token launch on pump.fun in real time via WebSocket
2. **Screens** each token through rugcheck.xyz before spending any compute
3. **Reasons** about whether the token will graduate using a fine-tuned Qwen3-4B LLM adapter trained via the NERO → GRPO pipeline
4. **Executes** buys directly on-chain (no exchange account needed) using the pump.fun program + Jupiter DEX aggregator
5. **Stores** every token, wallet, signal, and outcome in Neo4j for pattern mining and model improvement
6. **Learns** continuously: real trade outcomes feed back into GRPO as additional reward signal

**Target edge:**
> Other bots pattern-match price candles. This bot reasons about *why* a token will graduate — combining dev reputation, early buyer identity, bonding curve momentum, rug signals, and market context simultaneously.

**Constraints:**
- US-accessible only
- Zero paid API subscriptions
- SOL-only (no BTC/ETH complexity)
- On-chain execution only (no CEX account needed beyond a Solana wallet)

---

## 2. Full System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                        DATA LAYER (all free)                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  pump.fun WebSocket    wss://frontend-api.pump.fun/ws               ║
║    → every new token launch + every bonding curve trade             ║
║                                                                      ║
║  rugcheck.xyz REST     api.rugcheck.xyz/v1/tokens/{mint}/report     ║
║    → mint authority, freeze authority, insider networks,            ║
║      creator history, risk flags, rug confirmation                  ║
║                                                                      ║
║  Solana public RPC     api.mainnet-beta.solana.com                  ║
║    → wallet transaction history, token balances, on-chain state     ║
║                                                                      ║
║  DexScreener           api.dexscreener.com/latest/dex/tokens/{addr} ║
║    → post-graduation price, volume, liquidity on Raydium            ║
║                                                                      ║
║  CoinGecko (free)      api.coingecko.com/api/v3/simple/price        ║
║    → SOL/USD price, 24h volume, market cap                          ║
║                                                                      ║
║  alternative.me        api.alternative.me/fng/                      ║
║    → Crypto Fear & Greed index (market context)                     ║
║                                                                      ║
║  yfinance (library)    Python — no HTTP call                        ║
║    → historical SOL OHLCV for training data backfill                ║
║                                                                      ║
║  RSS feeds             CoinDesk, Solana Foundation blog             ║
║    → news headlines for sentiment context                           ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║                     RUGCHECK HARD SKIP GATE                         ║
║                   (runs BEFORE any LLM call)                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Check immediately on every new token launch:                       ║
║                                                                      ║
║  HARD SKIP if ANY of:                                               ║
║    ✗  mint_authority is set    (dev can print more tokens)          ║
║    ✗  freeze_authority is set  (dev can freeze your wallet)         ║
║    ✗  already_rugged = true    (confirmed rug)                      ║
║    ✗  top_holder_pct > 30%     (one wallet dominates)               ║
║    ✗  risk_penalty >= 10       (one danger or two warning flags)     ║
║                                                                      ║
║  → Hard skipped tokens are logged to Neo4j (training data)         ║
║  → Hard skipped tokens NEVER reach the LLM                         ║
╚══════════════════════════════════════════════════════════════════════╝
                              │ (passed gate)
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║                      NEO4J GRAPH DATABASE                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Stores: Token, Wallet, BondingCurveSnapshot, Signal,               ║
║          Trade, Outcome, NewsEvent                                   ║
║                                                                      ║
║  Key graph queries:                                                  ║
║  · "Find wallets that bought 3+ graduating tokens in first 60s"     ║
║  · "Show all tokens created by this dev wallet + their outcomes"    ║
║  · "What % of tokens with insider_detected=true graduated?"         ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║                   CONTEXT BUILDER                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Assembles structured prompt from:                                  ║
║   · Token name, ticker, mint address                                ║
║   · rugcheck summary (score, flags, creator history)                ║
║   · Dev wallet: prior launches, grad rate, rug count                ║
║   · Early buyer wallets: whale/sniper flags, prior behavior         ║
║   · Bonding curve: % filled, fill speed (tokens/min)               ║
║   · SOL market: price, 24h change, fear/greed, trend               ║
║   · Recent news headlines mentioning SOL or meme coins              ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║             NATHANAI CRYPTO ADAPTER — Qwen3-4B LoRA                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Trained via NERO → SFT → GRPO pipeline                             ║
║  Thinking budget: 4096 tokens                                       ║
║                                                                      ║
║  Reasoning steps:                                                    ║
║   0. rugcheck already filtered — safe to reason about               ║
║   1. Dev wallet reputation score and pattern                        ║
║   2. Early buyer wallet classification                              ║
║   3. Bonding curve momentum vs historical graduates                 ║
║   4. Token name/ticker meme potential for current market            ║
║   5. SOL macro context: risk-on or risk-off?                        ║
║   6. Decision: BUY / SKIP + confidence 0.0–1.0                     ║
║                                                                      ║
║  Output format:                                                      ║
║   <think>...reasoning...</think>                                    ║
║   BUY — confidence: 0.83                                            ║
║   [or]                                                              ║
║   SKIP — confidence: 0.91                                           ║
╚══════════════════════════════════════════════════════════════════════╝
                              │  BUY + confidence ≥ 0.75
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║                     RISK MANAGER (deterministic)                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Rules (all hardcoded, no LLM override):                            ║
║   · Max position size:  1.0% of portfolio per token                 ║
║   · Max open positions: 5 simultaneous                              ║
║   · Confidence gate:    only execute if confidence ≥ 0.75           ║
║   · Auto-exit (win):    sell at graduation (Raydium listing)        ║
║   · Auto-exit (loss):   cut if bonding curve stalls > 30 min        ║
║   · Session kill switch: halt if portfolio drops 10% in one session ║
║   · Never DCA into a loser — one entry, one exit, done              ║
╚══════════════════════════════════════════════════════════════════════╝
                              │  approved order
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║                   EXECUTION LAYER (on-chain)                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  BUY:  pump.fun bonding curve program (Solana on-chain tx)          ║
║  SELL: Jupiter DEX aggregator (best route post-graduation)          ║
║  Gas:  ~0.000005 SOL per tx — $5 covers 1,000,000+ transactions    ║
║  Libs: solders, anchorpy-core, httpx                                ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║                   P&L FEEDBACK LOOP (Phase 5)                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Trade outcome → Neo4j → GRPO additional reward signal              ║
║  Real trading makes the model better over time                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. Tools & Dependencies

### Python Packages

```txt
# Data collection
httpx>=0.27.0           # async HTTP for rugcheck, DexScreener, CoinGecko
websockets>=12.0        # pump.fun WebSocket connection
yfinance>=0.2.38        # historical SOL OHLCV (no API key)
feedparser>=6.0.11      # RSS news feeds

# Neo4j
neo4j>=5.19.0           # official Neo4j Python driver

# Solana / on-chain execution
solders>=0.21.0         # Rust-backed Solana types (fast, official)
anchorpy-core>=0.1.0    # Anchor program interaction (pump.fun program)

# LLM inference
vllm>=0.4.0             # serve Qwen3-4B LoRA adapter (same as NERO pipeline)
transformers>=4.41.0    # tokenizer + model loading
peft>=0.11.0            # LoRA adapter loading

# Training (reuses nathanai-evolution pipeline)
trl>=0.8.6              # GRPOTrainer
torch>=2.3.0            # CUDA training

# Utilities
pydantic>=2.7.0         # config validation
python-dotenv>=1.0.0    # .env file for wallet key + RPC URL
schedule>=1.2.1         # cron-style polling jobs
rich>=13.7.0            # terminal dashboard
```

### External Services (all free, no account required except Solana wallet)

| Service | Purpose | Account Required |
|---|---|---|
| pump.fun | Token launch data | No |
| rugcheck.xyz | Rug detection | No |
| Solana mainnet RPC | On-chain data | No |
| DexScreener | Post-grad data | No |
| CoinGecko | SOL price | No |
| alternative.me | Fear & Greed | No |
| **Solana wallet** | **Execute trades** | **Yes — self-custodied** |

### Infrastructure

| Component | Option A (local) | Option B (cloud free) |
|---|---|---|
| Neo4j | Neo4j Desktop (free) | Neo4j Aura Free Tier (1 instance) |
| LLM inference | Local GPU (H200 already running) | — |
| Bot process | Same machine as GPU | — |
| Wallet | Phantom / solana-keygen | — |

---

## 4. Environment Setup

### Step 1 — Python environment

```bash
cd /workspace/nathanai-crypto
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Neo4j

**Option A: Local (recommended for dev)**
```bash
# Download Neo4j Desktop from neo4j.com/download — free
# Create a new database, set password
# Default: bolt://localhost:7687
```

**Option B: Neo4j Aura (free cloud)**
```
Go to: console.neo4j.io
Create free instance → copy connection URI + password
```

### Step 3 — Solana wallet

```bash
# Install Solana CLI
sh -c "$(curl -sSfL https://release.solana.com/v1.18.0/install)"

# Generate a new trading wallet (store keypair securely)
solana-keygen new --outfile ~/.config/solana/trading_wallet.json

# Fund it with a small amount of SOL for gas + trading capital
# Transfer from your main wallet or buy SOL via Coinbase
solana balance  # confirm balance
```

### Step 4 — Environment variables

```bash
# Create .env file (NEVER commit this to git)
cat > .env << 'EOF'
# Solana
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_WALLET_PATH=/root/.config/solana/trading_wallet.json

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Model
MODEL_PATH=/workspace/nathanai-evolution/data/nero/crypto_adapter
BASE_MODEL=Qwen/Qwen3-4B

# Trading limits (safety defaults)
MAX_POSITION_PCT=0.01       # 1% per token
MAX_OPEN_POSITIONS=5
MIN_CONFIDENCE=0.75
SESSION_LOSS_LIMIT=0.10     # halt if -10% in one session
EOF
```

### Step 5 — Verify connections

```bash
python3 scripts/verify_setup.py
# Checks: Neo4j reachable, Solana RPC live, rugcheck API responding,
#         pump.fun WebSocket connectable, wallet balance readable
```

---

## 5. Repository Structure

```
nathanai-crypto/
│
├── collectors/
│   ├── __init__.py
│   ├── pumpfun_ws.py        # pump.fun WebSocket → new token events
│   ├── pumpfun_rest.py      # REST: bonding curve state, token details
│   ├── rugcheck.py          # rugcheck.xyz risk report (hard skip gate)
│   ├── solana_rpc.py        # wallet history, token account balances
│   ├── dexscreener.py       # post-graduation price/volume on Raydium
│   ├── sol_context.py       # CoinGecko SOL price + fear/greed + RSS news
│   └── backfill.py          # historical pump.fun data loader for training
│
├── graph/
│   ├── __init__.py
│   ├── schema.py            # Neo4j constraints + indexes setup
│   ├── ingest.py            # write Token, Wallet, Trade, Outcome nodes
│   └── queries.py           # training data extraction, analytics queries
│
├── adapter/
│   ├── __init__.py
│   ├── prompt_builder.py    # assemble all signals → structured LLM prompt
│   ├── inference.py         # Qwen3-4B LoRA inference via vLLM
│   └── decision_parser.py   # extract BUY/SKIP + confidence from response
│
├── risk/
│   ├── __init__.py
│   ├── position_sizer.py    # calculate SOL amount from portfolio %
│   └── circuit_breaker.py   # session loss tracking + kill switch
│
├── execution/
│   ├── __init__.py
│   ├── pumpfun_buyer.py     # buy on bonding curve via pump.fun program
│   ├── jupiter_seller.py    # sell via Jupiter post-graduation
│   └── wallet.py            # keypair loading, balance checking, signing
│
├── backtest/
│   ├── __init__.py
│   ├── historical_loader.py # pull labeled launches from Neo4j
│   ├── simulate.py          # replay history: would our model have profited?
│   └── metrics.py           # win rate, avg ROI, Sharpe, max drawdown
│
├── training/
│   ├── __init__.py
│   ├── nero_extract.py      # generate Q&A thinking chains from Neo4j history
│   └── grpo_reward.py       # graduation prediction reward function
│
├── monitor/
│   ├── __init__.py
│   └── dashboard.py         # live terminal: open positions, P&L, signals
│
├── scripts/
│   ├── verify_setup.py      # connection health check
│   ├── backfill_history.py  # one-time: load 90 days of pump.fun history
│   └── run_paper_trade.py   # paper trading mode (log predictions, no execution)
│
├── config.yaml              # all tuneable settings (see Section 11)
├── .env                     # secrets: wallet path, Neo4j password (not in git)
├── .gitignore               # exclude .env, __pycache__, venv
├── requirements.txt         # all Python dependencies
├── MASTER_BUILD_PLAN.md     # this document
└── main.py                  # entry point: start collectors → reason → execute
```

---

## 6. Data Sources & Settings

### pump.fun WebSocket

```python
WS_URL          = "wss://frontend-api.pump.fun/ws"
PING_INTERVAL   = 20    # seconds
RECONNECT_DELAY = 3     # seconds on disconnect
```

**Events we handle:**
- `create` → new token launched (trigger full evaluation pipeline)
- `buy` / `sell` → bonding curve trade (update curve snapshot in Neo4j)
- `migration` → token graduated to Raydium (mark outcome, trigger sell if holding)

### rugcheck.xyz

```python
API_BASE         = "https://api.rugcheck.xyz/v1"
TIMEOUT          = 10.0   # seconds
MAX_RETRIES      = 3
RETRY_DELAY      = 2.0    # seconds between retries

# Hard skip thresholds
HARD_SKIP_TOP_HOLDER_PCT  = 30.0   # skip if single wallet > 30%
HARD_SKIP_RISK_PENALTY    = 10     # skip if danger(10) OR warning+warning(10)
```

**Severity weights:**
```python
SEVERITY_WEIGHTS = {"danger": 10, "warning": 5, "info": 1}
```

### Solana RPC

```python
RPC_URL          = "https://api.mainnet-beta.solana.com"  # free public
# For higher rate limits (optional, still free):
# RPC_URL        = "https://solana-mainnet.g.alchemy.com/v2/demo"
WALLET_HISTORY_LIMIT = 50   # recent txns to check per wallet
```

### DexScreener

```python
API_BASE = "https://api.dexscreener.com/latest/dex/tokens"
# No key, no rate limit specified — be polite: 1 req/sec max
POLL_INTERVAL = 60  # seconds: check graduated token price every 60s
```

### CoinGecko (free tier)

```python
API_BASE         = "https://api.coingecko.com/api/v3"
SOL_ID           = "solana"
REFRESH_INTERVAL = 300   # seconds: update SOL price every 5 min
# Free tier: 10-30 calls/min — plenty for our polling rate
```

### Fear & Greed

```python
API_URL          = "https://api.alternative.me/fng/?limit=1"
REFRESH_INTERVAL = 3600  # hourly is fine (index updates daily)
```

---

## 7. Neo4j Schema

### Constraints & Indexes (run once on setup)

```cypher
CREATE CONSTRAINT token_mint IF NOT EXISTS
  FOR (t:Token) REQUIRE t.mint IS UNIQUE;

CREATE CONSTRAINT wallet_address IF NOT EXISTS
  FOR (w:Wallet) REQUIRE w.address IS UNIQUE;

CREATE INDEX token_created IF NOT EXISTS
  FOR (t:Token) ON (t.created_at);

CREATE INDEX token_graduated IF NOT EXISTS
  FOR (t:Token) ON (t.graduated);
```

### Node Definitions

```cypher
// ── Token ─────────────────────────────────────────────────────────────
(Token {
    mint:                string,   // Solana mint address (primary key)
    name:                string,
    ticker:              string,
    created_at:          datetime,
    dev_wallet:          string,   // creator wallet address

    // Outcome (filled in when known)
    graduated:           boolean,
    graduation_time:     datetime,
    peak_multiplier:     float,    // e.g. 3.5 = 3.5x from entry
    rug_pull:            boolean,

    // rugcheck.xyz (filled on launch)
    rc_score:            int,      // 0-100, higher = safer
    rc_hard_skip:        boolean,
    rc_mint_authority:   boolean,
    rc_freeze_authority: boolean,
    rc_top_holder_pct:   float,
    rc_lp_locked:        boolean,
    rc_insider_detected: boolean,
    rc_risk_penalty:     int,
    rc_risk_flags:       string,   // JSON array

    // Bonding curve snapshot at T+60s
    bc_pct_filled_60s:   float,
    bc_buy_count_60s:    int,
    bc_volume_sol_60s:   float,
})

// ── Wallet ────────────────────────────────────────────────────────────
(Wallet {
    address:             string,
    tokens_launched:     int,      // as dev
    graduates_launched:  int,
    rugs_launched:       int,
    grad_rate:           float,
    known_whale:         boolean,
    known_sniper:        boolean,
    first_seen:          datetime,
    last_seen:           datetime,
})

// ── Signal ────────────────────────────────────────────────────────────
(Signal {
    id:                  string,   // uuid
    token_mint:          string,
    timestamp:           datetime,
    decision:            string,   // "BUY" or "SKIP"
    confidence:          float,
    reasoning:           string,   // full thinking chain (truncated)
    model_version:       string,   // e.g. "crypto_grpo_v1"
})

// ── Trade ─────────────────────────────────────────────────────────────
(Trade {
    id:                  string,
    token_mint:          string,
    entry_time:          datetime,
    entry_price_sol:     float,
    entry_amount_sol:    float,
    exit_time:           datetime,
    exit_price_sol:      float,
    pnl_sol:             float,
    pnl_pct:             float,
    exit_reason:         string,   // "graduation" | "stall_cut" | "kill_switch"
})

// ── NewsEvent ─────────────────────────────────────────────────────────
(NewsEvent {
    timestamp:           datetime,
    headline:            string,
    source:              string,
    sentiment:           string,   // "positive" | "negative" | "neutral"
})
```

### Relationships

```cypher
(Token)-[:CREATED_BY]->(Wallet)
(Wallet)-[:BOUGHT {amount_sol: float, seconds_since_launch: int}]->(Token)
(Token)-[:GRADUATED_TO {raydium_pool: string}]->(Token)
(Signal)-[:EVALUATED]->(Token)
(Signal)-[:LED_TO]->(Trade)
(Trade)-[:RESULTED_IN {pnl_sol: float}]->(Signal)
(NewsEvent)-[:PRECEDED {hours: float}]->(Token)
```

### Key Training Queries

```cypher
// Wallets that consistently buy graduates early (these are signals)
MATCH (w:Wallet)-[b:BOUGHT]->(t:Token)
WHERE b.seconds_since_launch < 60 AND t.graduated = true
WITH w, count(t) AS early_grad_count
WHERE early_grad_count >= 3
RETURN w.address, early_grad_count ORDER BY early_grad_count DESC;

// Dev wallets with best graduate rate
MATCH (t:Token)-[:CREATED_BY]->(w:Wallet)
WHERE w.tokens_launched >= 3
RETURN w.address, w.grad_rate, w.tokens_launched
ORDER BY w.grad_rate DESC LIMIT 50;

// Does rugcheck score correlate with graduation?
MATCH (t:Token)
WHERE t.rc_hard_skip = false AND t.graduated IS NOT NULL
RETURN t.rc_score, avg(toFloat(t.graduated)) AS grad_rate, count(t) AS n
ORDER BY t.rc_score DESC;
```

---

## 8. Build Order — Phase by Phase

### Phase 0 — Prerequisites (before writing one line of bot code)

```
[ ] Neo4j instance running and accessible
[ ] Solana wallet created and funded (≥ 0.1 SOL for gas)
[ ] .env file configured with all credentials
[ ] python verify_setup.py passes all checks
[ ] Medical NERO pipeline proven (grpo_v12+ shows no regression)
      → This is already done as of 2026-04-19
```

### Phase 1 — Data Pipeline (Week 1, Days 1-3)

Build in this exact order — each depends on the previous:

```
Day 1:
[ ] graph/schema.py          — Neo4j constraints + indexes
[ ] graph/ingest.py          — write functions for Token, Wallet nodes
[ ] collectors/rugcheck.py   — DONE (already written)
[ ] collectors/pumpfun_ws.py — DONE (already written, rugcheck gate included)

Day 2:
[ ] collectors/solana_rpc.py — wallet transaction history lookup
[ ] collectors/sol_context.py — CoinGecko + fear/greed + RSS polling
[ ] collectors/dexscreener.py — post-graduation price tracking

Day 3:
[ ] Wire together: main collector loop
    pumpfun_ws → rugcheck gate → solana_rpc enrich → neo4j ingest
[ ] Test: run for 1 hour, verify tokens appearing in Neo4j
[ ] collectors/backfill.py   — historical data loader
[ ] scripts/backfill_history.py — one-time: load 90 days of data
```

### Phase 2 — Historical Backfill (Week 1, Days 3-4)

```
[ ] Run backfill: pull all pump.fun launches from on-chain history
    → target: 90 days of labeled data (graduated / dead)
    → store in Neo4j with full rugcheck enrichment
[ ] Run graph/queries.py analytics to verify data quality:
    - How many tokens total?
    - What % graduated?
    - Does rugcheck score correlate with graduation?
[ ] Export training dataset: graph queries → JSONL files
```

### Phase 3 — LLM Adapter Training (Week 2)

Uses existing nathanai-evolution NERO → GRPO pipeline.

```
Day 5-6: NERO Extraction
[ ] training/nero_extract.py
    — reads labeled tokens from Neo4j
    — generates thinking-chain Q&A pairs using Qwen3-32B teacher
    — saves to data/nero/crypto_pumpfun_sft.jsonl
[ ] Target: 10,000+ thinking-chain examples

Day 6-7: SFT Training
[ ] python nero/train.py --domain crypto_pumpfun
    — same pipeline as medical
    — LoRA r=64 alpha=128, 3 epochs
    — adapter saved to data/nero/crypto_pumpfun_adapter_v1_sft

Day 8-9: GRPO Refinement
[ ] Register crypto_pumpfun domain in phase8/domains/
[ ] training/grpo_reward.py — graduation prediction reward
[ ] python phase8/grpo_online_run.py --domains crypto_pumpfun
    — 500 steps, cold start from base (same lesson as medical v12)
    — target: >58% graduation prediction accuracy (>50% random)
```

### Phase 4 — Adapter Integration (Week 2, Days 9-10)

```
[ ] adapter/prompt_builder.py
    — assembles Token + rugcheck + wallet + context → prompt
[ ] adapter/inference.py
    — loads Qwen3-4B + LoRA via vLLM
    — returns BUY/SKIP + confidence
[ ] adapter/decision_parser.py
    — regex parse decision + confidence from model output
[ ] Test: run inference on 100 held-out historical tokens
    — does the model agree with ground truth?
```

### Phase 5 — Risk + Execution Layer (Week 2-3)

```
[ ] risk/position_sizer.py   — 1% of portfolio in SOL
[ ] risk/circuit_breaker.py  — session loss tracking
[ ] execution/wallet.py      — keypair load, balance check, sign tx
[ ] execution/pumpfun_buyer.py — on-chain buy on bonding curve
[ ] execution/jupiter_seller.py — sell via Jupiter post-graduation
[ ] Test execution on devnet first (Solana devnet = free fake SOL)
```

### Phase 6 — Paper Trading (Week 3-4, minimum 14 days)

```
[ ] scripts/run_paper_trade.py
    — full live pipeline: collect → rugcheck → context → LLM → decide
    — log every decision + outcome to Neo4j
    — NO real trades executed
[ ] Run for minimum 14 days
[ ] Review metrics weekly:
    - Prediction accuracy (BUY vs actual graduation)
    - How many hard skips were correct?
    - What did we miss? What did we avoid?
[ ] Only proceed to Phase 7 if paper accuracy > 55%
```

### Phase 7 — Live Trading (Month 2+)

```
[ ] Start with absolute minimum capital (only what you can lose entirely)
[ ] 1% per token, 5 positions max — enforced by circuit breaker
[ ] Monitor every day for first 2 weeks
[ ] Review model weekly: is it improving? where does it fail?
[ ] Phase 3 feedback loop: real outcomes → GRPO additional reward
```

---

## 9. NERO + GRPO Training Plan

### Training Data Format (same as medical)

```json
{
  "question": "Given this Solana token launch, predict if it will graduate:\nToken: BONKCAT (BCAT)\nDev wallet: 3 prior launches, 2 graduated (67% rate), 0 rugs\nrugcheck: score=82, no mint authority, no freeze, top holder 8.2%\nEarly buyers (first 60s): wallet_A (bought 4 prior graduates early), wallet_B (unknown)\nBonding curve: 11% filled in 60 seconds (avg is 5%)\nSOL: $148, up 8% this week. Fear & Greed: 71 (Greed)\nWill this token graduate? A) Yes  B) No",
  "answer": "A",
  "thinking": "Dev wallet has 67% grad rate — solid. No mint/freeze authority — safe to trade. Top holder at 8.2% is healthy. Wallet_A is a known graduate buyer — strong signal. 11% bonding curve fill in 60s is 2x the average — high momentum. SOL in uptrend with greed sentiment — meme season active. Multiple confirming signals. Graduate probability high."
}
```

### Reward Function

```python
def graduation_reward(prediction: str, graduated: bool) -> float:
    if prediction == "A" and graduated:      return  1.0   # correct BUY
    if prediction == "B" and not graduated:  return  1.0   # correct SKIP
    if prediction == "A" and not graduated:  return -1.0   # bought a rug
    if prediction == "B" and graduated:      return -0.5   # missed opportunity
    return 0.0
```

### GRPO Settings (crypto domain)

```python
grpo_steps     = 500       # more than medical — new domain
lr             = 1e-5      # cold start from base
n_rollouts     = 16        # same as medical v12
max_comp_len   = 8192      # allow full thinking chains
beta           = 0.04      # KL penalty
thinking       = True      # thinking-ON (reasoning is the moat)
thinking_budget= 4096      # tokens for reasoning
```

### Target Metrics

| Metric | Baseline (random) | Target | Stretch |
|---|---|---|---|
| Graduation prediction accuracy | 50% | 58% | 65% |
| Hard skip precision (skipped rugs / total skipped) | — | >80% | >90% |
| Missed graduation rate | — | <40% | <25% |

---

## 10. Risk Management Settings

### Position Sizing

```yaml
max_position_pct:     0.01    # 1% of portfolio per token
max_open_positions:   5       # never more than 5 tokens at once
min_confidence:       0.75    # LLM must be ≥75% confident to execute
```

### Exit Rules

```yaml
exit_on_graduation:   true    # sell immediately when token hits Raydium
stall_exit_minutes:   30      # cut position if bonding curve stalls 30 min
stall_threshold_pct:  0.5     # "stalled" = bonding curve grew < 0.5% in 30 min
```

### Session Limits

```yaml
session_loss_limit:   0.10    # halt all trading if -10% portfolio in one session
daily_loss_limit:     0.15    # halt for the day if -15% total
max_daily_trades:     50      # prevent runaway trading on bad model day
```

### Hard Rules (never override, regardless of model confidence)

1. Never trade a token that failed the rugcheck hard skip gate
2. Never invest more than 1% per token
3. Always sell at graduation — never hold for "more gains"
4. Never DCA into a position — one entry, one exit
5. Stop all trading if session loss limit hit — review before resuming

---

## 11. Configuration File Reference

```yaml
# config.yaml — all tuneable settings
# Hot-reloaded every 60 seconds — no restart needed

# ── Data Sources ──────────────────────────────────────────────────────
pumpfun:
  ws_url:             "wss://frontend-api.pump.fun/ws"
  ping_interval:      20
  reconnect_delay:    3

rugcheck:
  api_base:           "https://api.rugcheck.xyz/v1"
  timeout:            10.0
  max_retries:        3
  hard_skip_top_holder_pct:  30.0
  hard_skip_risk_penalty:    10

solana:
  rpc_url:            "${SOLANA_RPC_URL}"       # from .env
  wallet_path:        "${SOLANA_WALLET_PATH}"   # from .env
  wallet_history_limit: 50

dexscreener:
  api_base:           "https://api.dexscreener.com/latest/dex/tokens"
  poll_interval:      60

coingecko:
  sol_id:             "solana"
  refresh_interval:   300

fear_greed:
  api_url:            "https://api.alternative.me/fng/?limit=1"
  refresh_interval:   3600

# ── Neo4j ─────────────────────────────────────────────────────────────
neo4j:
  uri:                "${NEO4J_URI}"
  user:               "${NEO4J_USER}"
  password:           "${NEO4J_PASSWORD}"

# ── LLM Adapter ───────────────────────────────────────────────────────
adapter:
  model_path:         "${MODEL_PATH}"
  base_model:         "Qwen/Qwen3-4B"
  thinking:           true
  thinking_budget:    4096
  max_new_tokens:     4608       # thinking_budget + answer
  temperature:        0.6
  repetition_penalty: 1.1

# ── Risk Management ───────────────────────────────────────────────────
risk:
  max_position_pct:   0.01
  max_open_positions: 5
  min_confidence:     0.75
  stall_exit_minutes: 30
  stall_threshold_pct: 0.5
  session_loss_limit: 0.10
  daily_loss_limit:   0.15
  max_daily_trades:   50

# ── Training ──────────────────────────────────────────────────────────
training:
  grpo_steps:         500
  lr:                 1.0e-5
  n_rollouts:         16
  max_comp_len:       8192
  beta:               0.04
```

---

## 12. Paper Trading Protocol

Before risking any real SOL, run paper trading for **minimum 14 days**.

### What paper trading does

- Connects to all live data sources (pump.fun WS, rugcheck, Solana RPC)
- Runs the full pipeline: collect → gate → context → LLM → decision
- Logs every decision + actual outcome to Neo4j
- Does NOT execute any on-chain transactions

### Metrics to track during paper trading

| Metric | Check daily | Pass threshold |
|---|---|---|
| Graduation prediction accuracy | Yes | >55% |
| Hard skip precision | Yes | >80% |
| False BUY rate (bought a rug) | Yes | <20% |
| Missed graduation rate | Weekly | <45% |
| Estimated P&L (if we had traded) | Weekly | Positive |
| Model confidence calibration | Weekly | Confidence correlates with accuracy |

### Go / No-Go decision

**Go live only if ALL of these are true:**
1. Paper ran for ≥ 14 consecutive days without interruption
2. Graduation accuracy > 55% across all 14 days (not just best day)
3. Estimated P&L positive after accounting for 1% fees per trade
4. No period of >5 consecutive wrong BUY calls
5. Hard skip gate correctly avoided ≥80% of rugs

---

## 13. Live Trading Checklist

Run this checklist before starting live trading:

```
Pre-flight:
[ ] Paper trading ran 14+ days with positive metrics (see Section 12)
[ ] Wallet balance confirmed (only capital you can afford to lose entirely)
[ ] config.yaml reviewed — especially risk limits
[ ] Neo4j has 90+ days of historical data for pattern context
[ ] circuit_breaker.py tested: does it actually halt trading?
[ ] Test one real buy+sell on devnet with tiny amount

Go-live:
[ ] Start with max_position_pct = 0.005 (0.5%) for first week
[ ] Monitor terminal dashboard every hour for first 3 days
[ ] Review every trade in Neo4j at end of each day
[ ] After 1 week positive: raise to max_position_pct = 0.01 (1%)

Ongoing:
[ ] Weekly: review worst losses — was the model wrong or unlucky?
[ ] Weekly: retrain GRPO with new real trade outcomes
[ ] Monthly: full backtest on new data to check for model drift
```

---

## 14. Milestone Tracker

| # | Milestone | Status | Target |
|---|---|---|---|
| 1 | Medical NERO pipeline proven (v12 no regression) | **DONE** | 2026-04-19 |
| 2 | Medical GRPO v13 benchmarked | In progress | Today |
| 3 | Phase 0: environment setup (Neo4j, wallet, .env) | Pending | Week 0 |
| 4 | Phase 1: pump.fun WS + rugcheck collector live | Partial (rugcheck done) | Week 1 Day 1-2 |
| 5 | Phase 1: Solana RPC + SOL context + DexScreener | Pending | Week 1 Day 2 |
| 6 | Phase 1: Neo4j schema + ingestion wired end-to-end | Pending | Week 1 Day 3 |
| 7 | Phase 2: 90-day historical backfill complete | Pending | Week 1 Day 3-4 |
| 8 | Phase 3: NERO thinking-chain extraction (10K+ examples) | Pending | Week 2 Day 5-6 |
| 9 | Phase 3: SFT training complete | Pending | Week 2 Day 6-7 |
| 10 | Phase 3: GRPO v1 — graduation accuracy > 58% | Pending | Week 2 Day 8-9 |
| 11 | Phase 4: prompt builder + inference + parser integrated | Pending | Week 2 Day 9-10 |
| 12 | Phase 5: risk + execution on devnet (fake SOL) | Pending | Week 3 |
| 13 | Phase 6: 14-day paper trading — metrics pass go/no-go | Pending | Week 3-4 |
| 14 | Phase 7: live trading with minimum capital | Pending | Month 2 |
| 15 | Phase 7: GRPO feedback loop from real trades | Pending | Month 2+ |
