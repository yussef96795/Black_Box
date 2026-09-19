# Block A: Ingestion & Feasibility Engine

## Step 1: Document Processing Pipeline
- [ ] Implement Docling PDF parser integration for layout, math, and table extraction
- [ ] Configure structure-aware semantic chunking (Docling Hybrid Chunker / Chonkie)
- [ ] Build Variable Resolution Module to map LaTeX math symbols to text definitions
- [ ] Implement table schema validator to verify parameter bounds and numerical ranges

## Step 2: Block A.5 Feasibility Auditor Gate
- [ ] Define System Capabilities JSON contract (data granularity, supported primitives)
- [ ] Build Hard Dependency Checker (L3 data, latency, non-crypto TradFi dependencies)
- [ ] Build Algorithmic & Compute Checker (black-box ML models, compute bounds)
- [ ] Build Domain Mapping Engine (TradFi primitives to crypto analogs: rates, trading hours)
- [ ] Implement Feasibility Result Schema output (`PASSED`, `REQUIRES_HITL`, `REJECTED`)

---

# Block B: Formulation Engine & Human-In-The-Loop

## Step 1: LangGraph State Machine Core
- [ ] Setup LangGraph DAG runtime and PostgreSQL/SQLite state checkpointing
- [ ] Implement Node 1: Quant Extractor (PydanticAI math/rule extractor with `NULL_AMBIGUOUS` flag)
- [ ] Implement Node 2: Schema Builder (Pydantic V2 AST transpiler + SymPy math validation)
- [ ] Implement Node 3: Cynical Auditor (adversarial validator for lookahead, exit rules, bounds)
- [ ] Enforce deterministic iteration cap ($N \le 2$ retry loop)

## Step 2: Angular HITL Gate & Streaming Layer
- [ ] Implement Node 4: HITL Payload Generator (batches missing rules & audit flags)
- [ ] Implement Node 5: LangGraph `interrupt()` gate for state persistence
- [ ] Setup FastAPI SSE / WebSocket endpoints for real-time Angular communication
- [ ] Build State Resumption endpoint (`POST /api/v1/strategy/resume`) to inject user answers

---

# Block C: Statistical Stress Validation Engine

## Step 1: Vectorized Surface Sweep & CPCV
- [ ] Implement VectorBT Pro parameter sweep engine
- [ ] Build 3D Neighborhood Surface Decay Metric for parameter stability scoring
- [ ] Implement Combinatorial Purged & Embargoed Cross-Validation (CPCV) framework
- [ ] Build feature leakage sanity checker to detect lookahead bias across folds

## Step 2: Monte Carlo & Portfolio Allocation
- [ ] Build Block Bootstrap Return Resampling engine (volatility clustering test)
- [ ] Build Execution Perturbation engine (order skipping & dynamic slippage penalty)
- [ ] Integrate Riskfolio-Lib Abstract Allocation Matrix Interface (HRP, Risk Parity, MVO)
- [ ] Implement Validation Report Payload serializer for frontend streaming

---

# Block D: Execution Transpiler & Display Factory

## Step 1: AST Transpiler & Risk Guardrail Injection
- [ ] Build Jinja2 AST Transpiler Engine for target code generation (.mq5 EA / Python)
- [ ] Inject Global Max Drawdown Circuit Breaker & Kill-Switch module
- [ ] Inject Daily Loss Limit & Consecutive Loss Cooldown timers
- [ ] Inject Dynamic Position Sizing (ATR/Volatility adjusted lot sizes)
- [ ] Inject Market Microstructure Protection (Spread/Slippage guards)
- [ ] Inject Unique Magic Number & Local State Persistence handler

## Step 2: Backend Data Aggregation & Angular Display
- [ ] Implement LTOB (Largest-Triangle-Three-Buckets) downsampling for time-series charts
- [ ] Implement Monte Carlo percentile band aggregator (P10, P50, P90 density reduction)
- [ ] Build Angular WebGL / ECharts display components for 3D Heatmaps and MC distributions
- [ ] Implement final strategy export bundle packager (.mq5/.py files + JSON report)d