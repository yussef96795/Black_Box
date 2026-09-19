# Black_Box — Build Plan & Session Handoff

> **Repo** `yussef96795/Black_Box` (public) · **Baseline commit** `d244c73`
> **Stack** FastAPI backend (uv) + Docling 2.129.0 + Angular 22 frontend scaffold
> **Today** Sat Sep 19 2026
> **Doc refs** Context7 `/websites/docling` (converter + HybridChunker), `/libraries/fastapi`

---

## Current Status (start-of-session handoff)

### ✅ Finished
- **Git repo**: public repo `yussef96795/Black_Box` pushed to `main`, baseline commit `d244c73`, clean `origin` URL, no secrets in tree (verified), root `.gitignore` (venv/node_modules/.env/caches).
- **Dependency tooling**: backend scaffold created with `uv`; `pyproject.toml` with FastAPI 0.141.1, pydantic 2.13, pydantic-settings 2.15, python-multipart, uvicorn 0.53, and **docling 2.129.0 installed** (torch/CUDA deps resolved).
- **Backend package skeleton** (`backend/src/black_box/`):
  - `core/config.py` — `Settings` dataclass (env/.env driven) + `get_settings()` (`lru_cache`).
  - `services/docling_service.py` — `Chunk` dataclass, `DoclingService` (lazy singleton `DocumentConverter`, async-safe lifecycle `_ensure`/`aclose`, `accel()` probe), `_BytesSource` (in-memory spool), `_ServiceHolder` singleton + `get_docling_service()` DI factory.
  - `api/routes.py` — single `api_router` (duplicate decl removed), `POST /documents/ingest` (validates file, depends on service, maps errors → 422/502).
  - `main.py` — FastAPI app, lifespan owns/tears-down Docling service on `app.state`, mounts `api_router` under `settings.api_prefix`, `/health` liveness probe.
- **Import graph reconciles**: single authoritative read + grep confirm every name lines up top-to-bottom. All four modules pass a direct import check with no `ImportError`.

### 🔍 Verified by Agent
- `uv pip list` confirms all runtime deps installed at pinned versions.
- Git log confirms single commit `d244c73` = baseline.
- No `.env` file exists yet — must be created before boot.
- Frontend is Angular 22.1.x scaffold only (`app.ts`, `app.routes.ts` empty routes, no custom components).

### ⏭️ Next / Remaining (in priority order)
1. **Boot + smoke test** — create `.env`, run `uvicorn black_box.main:app`, curl `/health`, `POST /documents/ingest` with a sample PDF/DOCX.
2. **Reconcile DoclingService ownership** — two instances possible (lifespan `app.state.docling` vs `_ServiceHolder` singleton). Pick one owner (recommended: make `get_docling_service` serve the lifespan instance, drop the module singleton) per Rules.md §SRP.
3. **Land real semantic chunking** — replace fallback `iterate_items` text splits with Docling `HybridChunker(chunk_by_documents=True)`.
4. Then continue Block A Step 2 → B/C/D below.

---

## Testing Strategy (Cross-Cutting)

### Unit Tests
- **Framework**: `pytest` + `pytest-asyncio` + `httpx` (TestClient for FastAPI)
- **Coverage target**: ≥80% backend, ≥70% frontend
- **Location**: `backend/tests/` (mirrors package structure), `frontend/src/app/**/*.spec.ts`
- **Mock strategy**: `DoclingService` mocked at unit level; `DocumentConverter` instantiated only in integration tests
- **Fixtures**: sample PDFs/DOCXs in `backend/tests/fixtures/` (generated via headless LibreOffice or pre-committed)

### Integration Tests
- **Scope**: Full `/api/v1/documents/ingest` pipeline with real Docling converter
- **Assertions**: response shape, chunk count > 0, chunk fields populated, status codes
- **Database**: SQLite in-memory for Block B state checkpointing tests

### E2E Tests
- **Framework**: `@playwright/test` for Angular frontend → FastAPI backend
- **Scenarios**: file upload → chunk display → HITL interaction → strategy export
- **Location**: `e2e/` at repo root

### CI/CD
- **Platform**: GitHub Actions (repo is public on GitHub)
- **Pipeline**:
  1. `lint` — `ruff check` + `ruff format --check` for Python; `npx eslint` + `npx prettier --check` for TS
  2. `test:unit` — `pytest tests/unit` with coverage
  3. `test:integration` — `pytest tests/integration` (requires Docling)
  4. `test:e2e` — `npx playwright test`
  5. `build` — `uv build` + `ng build`
  6. `deploy` — optional, Dockerfile (see below)
- **Pre-commit hooks**: `pre-commit` with ruff, prettier, mypy, lint-staged

### Observability
- **Logging**: Python `logging` module with structured JSON format (`python-json-logger`), correlation IDs per request
- **Metrics**: Prometheus endpoint `/metrics` via `slowapi` or custom middleware
- **Tracing**: OpenTelemetry SDK for request tracing (Block C/D when distributed)
- **Health**: `/health` liveness + `/ready` readiness probe (Docling converter initialized?)

---

## Cross-Cutting Concerns

### Security
- **File upload validation**: MIME type allowlist (`application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `text/html`, `text/markdown`), size cap (default 25MB), magic-byte sniff before parse (Rules.md §5)
- **CORS**: `fastapi.middleware.cors.CORSMiddleware` — restrict origins to frontend URL
- **Rate limiting**: `slowapi` on ingestion endpoint (default 10 req/min per IP)
- **Input sanitization**: all text chunks HTML-escaped before rendering in Angular
- **No secrets on disk**: `_BytesSource` uses `io.BytesIO` (already implemented); `.env` in `.gitignore`

### Data Persistence
- **Block A**: in-memory chunks (stateless ingestion), optional Redis cache for chunk results
- **Block B/C**: PostgreSQL (not just SQLite) for state checkpointing + strategy artifacts
- **Block D**: filesystem export bundles + optional S3/MinIO for distributed storage
- **Migrations**: `alembic` for PostgreSQL schema evolution

### API Contract
- **Versioning**: URL prefix `/api/v1/` (configured via `Settings.api_prefix`)
- **Error format**: `{"detail": "...", "status_code": N, "type": "..."}` (FastAPI default)
- **Content-Type**: `application/json` for all responses
- **Pagination**: `limit`/`offset` query params for chunk listings (Block C/D)

### Deployment
- **Docker**: multi-stage build (uv Python layer + Angular build), `Dockerfile` at repo root
- **Docker Compose**: `docker-compose.yml` with app + PostgreSQL + Redis (optional)
- **Runtime**: `uvicorn black_box.main:app --host 0.0.0.0 --port 8000 --workers 1`
- **Frontend**: `ng build --configuration production` served via Nginx or FastAPI static files

---

# Block A: Ingestion & Feasibility Engine

## Step 1: Document Processing Pipeline

### 1a · Thin FastAPI ingestion endpoint — ✅ DONE (skeleton)
- [x] `Settings` + `get_settings` from env/.env (`core/config.py`)
- [x] `DoclingService` lazy singleton with async `_ensure()` (converter constructed off-loop in executor), `aclose()`, `accel()`
- [x] In-memory `_BytesSource` upload wrapper (no tempfile, no secrets on disk) — Rules.md §5
- [x] `POST /documents/ingest` route: validates part presence → 422; maps service failure → 502; returns structured chunks payload
- [x] FastAPI DI wiring: `Depends(get_docling_service)` + lifespan lifecycle
- [x] `/health` probe exposing service/version/docling-ready state

**Acceptance Criteria**: `uvicorn` boots without error; `GET /health` returns `{"service": "Black_Box", "version": "0.1.0", "docling": "ready|pending"}`; `POST /documents/ingest` with no file returns 422; `POST /documents/ingest` with file returns 201 with `document_id`, `chunk_count`, `chunks`.

### 1b · Structural semantic chunking — ⏳ NEXT (in progress)
- [ ] **Resolve DoclingService ownership** — consolidate lifespan instance and `_ServiceHolder` singleton into one owner (acceptance: single `DoclingService` instance per worker, verified via `id()` comparison)
- [ ] Replace fallback text-split with Docling **HybridChunker** (`chunk_by_documents=True`) → `chunker.chunk(doc)`
  - **Implementation**: In `DoclingService.ingest()`, after `converter.convert()`, instantiate `HybridChunker(chunk_by_documents=True)` and call `chunker.chunk(doc)`. Map Docling `Chunk` objects to our `Chunk` dataclass fields.
  - **HybridChunker API** (per Context7 `/websites/docling`): `HybridChunker(chunk_by_documents=True)` → `.chunk(document)` returns `List[Chunk]` where each chunk has `.text`, `.page`, `.heading`, `.id`, `.meta`
- [ ] Emit `Chunk` fields fully: `id`, `text`, `page`, `heading`, `tokens`, `meta`
  - **`id`**: `f"{doc_id}:{chunk_index}"` UUID-based
  - **`text`**: chunk text content
  - **`page`**: source page number (from Docling chunk)
  - **`heading`**: nearest heading context (from Docling chunk metadata)
  - **`tokens`**: `len(text.split())` as proxy (or `tiktoken` if added to deps)
  - **`meta`**: `{"table": bool, "equation": bool, "section": str}` flags from Docling item labels
- [ ] Attach heading/page/section provenance per chunk (structure-aware)
- [ ] Document type allowlist (PDF/DOCX/HTML/MD) + size cap + magic-byte sniff before parse
  - **Allowlist**: `application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `text/html`, `text/markdown`
  - **Size cap**: `Settings.max_upload_size` (default 25MB, env-configurable)
  - **Magic-byte sniff**: check first 4-8 bytes before passing to Docling (PDF = `%PDF`, DOCX = ZIP header `PK\x03\x04`, HTML = `<`, MD = `#` or `---`)
- [ ] **Add `.env` file** with defaults: `APP_NAME=Black_Box`, `DEBUG=false`, `API_PREFIX=/api/v1`, `MAX_UPLOAD_SIZE=26214400`, `DATA_DIR=data`, `UPLOAD_DIR=data/uploads`, `CHUNK_CACHE_DIR=data/chunks`
- [ ] Boot + end-to-end smoke test with a sample PDF → confirm chunks in response
  - **Sample file**: `backend/tests/fixtures/sample.pdf` (create or download a small test PDF)
  - **Test command**: `curl -X POST localhost:8000/api/v1/documents/ingest -F "file=@sample.pdf"`
  - **Assertions**: 201 status, `chunk_count > 0`, each chunk has `id`, `text`, `page`, `heading`, `tokens`, `meta`

**Acceptance Criteria**: `POST /documents/ingest` with valid PDF returns 201 with `chunk_count >= 1`; every chunk object has all 6 fields populated; magic-byte rejection returns 422; oversized file returns 413.

**Risk**: Docling `HybridChunker` API may differ slightly from documented signature. Mitigate by running a smoke test immediately after implementation and checking against Context7 `/websites/docling` docs.

### 1c · Variable Resolution Module (LaTeX → text definitions) — not started
- [ ] Map LaTeX math symbols/expressions to canonical text definitions
  - **Approach**: Parse LaTeX expressions found in Docling `MATH` items; resolve symbols using a symbol table (e.g., `\alpha` → "alpha", `\sum` → "summation")
  - **Library**: `sympy` for symbolic math parsing; `latex2text` or custom regex for symbol resolution
- [ ] Keep original token stream + resolved symbol table (Block C consumes)
  - **Output schema**: `{"original_latex": str, "resolved_text": str, "symbol_table": dict[str, str], "context": str}`

**Acceptance Criteria**: LaTeX expressions in document chunks are resolved to human-readable text; symbol table is emitted alongside each chunk containing math content.

**Dependencies**: Block A Step 1b (chunking) must land first.

### 1d · Table Schema Validator — not started
- [ ] Parse Docling table structures (`TABLE` items from `iterate_items` or chunk metadata)
  - **Approach**: Extract `TABLE` items from Docling document; parse rows/columns/cells into structured format
- [ ] Verify parameter bounds & numerical ranges against expected schema
  - **Schema definition**: JSON schema file (`tables/schema.json`) defining expected columns, types, min/max values
  - **Validation**: check each cell against its column's type and bounds
- [ ] Emit validation report per table (fit/orphan/parse-error)
  - **Report format**: `{"table_id": str, "status": "fit" | "orphan" | "parse-error", "rows": int, "columns": int, "errors": list[str]}`

**Acceptance Criteria**: Table items are extracted from documents; validation report identifies fit/orphan/parse-error tables; report is included in ingestion response when tables present.

**Dependencies**: Block A Step 1b.

---

## Step 2: Block A.5 Feasibility Auditor Gate
- [ ] **Define System Capabilities JSON contract** (data granularity, supported primitives)
  - **Schema**: `{"capabilities": [{"name": str, "type": "data" | "compute" | "domain", "granularity": str, "primitives": [str]}], "version": str}`
- [ ] **Build Hard Dependency Checker** (L3 data, latency, non-crypto TradFi dependencies)
  - Checks for required data sources, latency SLA, non-crypto dependencies (e.g., Bloomberg, Reuters feeds)
  - Output: `{"dependencies": [{"name": str, "status": "satisfied" | "missing" | "partial", "latency_ms": int}]}`
- [ ] **Build Algorithmic & Compute Checker** (black-box ML models, compute bounds)
  - Validates that strategies don't rely on unsupported black-box models
  - Checks compute bounds (GPU memory, inference time) against available hardware
  - Output: `{"models": [...], "compute_budget": {"gpu_memory_mb": int, "inference_ms": int}}`
- [ ] **Build Domain Mapping Engine** (TradFi primitives to crypto analogs: rates, trading hours)
  - Maps TradFi concepts (interest rates, market hours, order types) to crypto equivalents
  - Output: `{"mapping": [{"tradfi": str, "crypto": str, "confidence": float}]}`
- [ ] **Implement Feasibility Result Schema output** (`PASSED`, `REQUIRES_HITL`, `REJECTED`)
  - **Endpoint**: `POST /api/v1/feasibility/audit`
  - **Result schema**: `{"status": "PASSED" | "REQUIRES_HITL" | "REJECTED", "checks": [...], "summary": str}`

**Acceptance Criteria**: Feasibility audit endpoint returns structured result; all 4 checkers execute in sequence; result schema matches contract; `PASSED` requires all hard dependencies satisfied.

**Dependencies**: Block A Step 1 (full pipeline) must be stable.

---

# Block B: Formulation Engine & Human-In-The-Loop

## Step 1: LangGraph State Machine Core

### Technical Architecture
- **Framework**: `langgraph` (LangGraph) + `langchain` for LLM integration
- **State backend**: PostgreSQL via `langgraph-checkpoint-postgres` (not SQLite — PostgreSQL for production reliability)
- **State schema**: Typed Pydantic model defining the full strategy formulation state
- **Retry cap**: Deterministic iteration limit `N ≤ 2` enforced by LangGraph `max_iterations`

### State Schema (TypeScript/Python shared)
```typescript
// frontend/src/app/strategy/strategy-state.ts
interface StrategyState {
  sessionId: string;
  round: number; // 0 = initial, increments per HITL cycle
  extractedQuantities: Quantity[];
  schemaDefinition: SchemaDefinition | null;
  auditFlags: AuditFlag[];
  userResponses: Record<string, string>;
  status: "idle" | "extracting" | "building" | "auditing" | "hitl" | "complete" | "rejected";
  iterationCount: number;
  maxIterations: number; // = 2
}
```

```python
# backend/src/black_box/state/schema.py
class StrategyState(BaseModel):
    session_id: str
    round: int = 0
    extracted_quantities: list[Quantity] = []
    schema_definition: Optional[SchemaDefinition] = None
    audit_flags: list[AuditFlag] = []
    user_responses: dict[str, str] = {}
    status: Literal["idle", "extracting", "building", "auditing", "hitl", "complete", "rejected"] = "idle"
    iteration_count: int = 0
    max_iterations: int = 2
```

### Node Implementations

- [ ] **Setup LangGraph DAG runtime and PostgreSQL checkpointing**
  - **Implementation**: `langgraph.graph.StateGraph` with `postgres` checkpoint saver
  - **Connection**: `DATABASE_URL` from `.env` → `AsyncEngine` → `AsyncSession`
  - **Migration**: `alembic` for `langgraph_checkpoint` tables
  - **Acceptance**: State persists across server restarts; `POST /api/v1/strategy/submit` creates a new session

- [ ] **Node 1: Quant Extractor** (PydanticAI math/rule extractor with `NULL_AMBIGUOUS` flag)
  - **Implementation**: PydanticAI agent with `NULL_AMBIGUOUS` tool flag; extracts numerical quantities, constraints, and rules from document chunks
  - **Output**: `Quantity[]` with fields `{name, value, unit, ambiguity_level, source_chunk_id}`
  - **Acceptance**: Extracts ≥90% of numerical values from a sample strategy doc; ambiguous values flagged `NULL_AMBIGUOUS`

- [ ] **Node 2: Schema Builder** (Pydantic V2 AST transpiler + SymPy math validation)
  - **Implementation**: Pydantic V2 model auto-generated from extracted quantities; SymPy validates mathematical relationships
  - **Output**: `SchemaDefinition` with Pydantic model class + SymPy expression tree
  - **Acceptance**: Generated schema validates against sample data; SymPy confirms mathematical consistency

- [ ] **Node 3: Cynical Auditor** (adversarial validator for lookahead, exit rules, bounds)
  - **Implementation**: Adversarial agent that tries to find lookahead bias, missing exit rules, unbounded parameters
  - **Output**: `AuditFlag[]` with fields `{type, severity, description, suggestion}`
  - **Acceptance**: Detects all seeded lookahead biases in test documents; flags missing exit rules

- [ ] **Enforce deterministic iteration cap** ($N \le 2$ retry loop)
  - **Implementation**: LangGraph `ConditionalEdge` checking `iteration_count < max_iterations`
  - **Acceptance**: Workflow terminates after exactly 2 retries regardless of audit results; `iteration_count` is monotonically increasing

### Block B API Endpoints
- [ ] `POST /api/v1/strategy/submit` — creates new strategy session, kicks off LangGraph DAG
- [ ] `GET /api/v1/strategy/{session_id}/state` — retrieves current state
- [ ] `POST /api/v1/strategy/{session_id}/resume` — injects user answers, resumes DAG

**Acceptance Criteria**: Full DAG executes end-to-end with a sample document; state persists in PostgreSQL; each node produces expected output; iteration cap enforced.

**Dependencies**: Block A Step 1b+1c+1d must be stable (extraction depends on chunking + variable resolution + table validation).

---

## Step 2: Angular HITL Gate & Streaming Layer

### Technical Architecture
- **Real-time transport**: FastAPI SSE (Server-Sent Events) for state updates + WebSocket for bidirectional HITL responses
- **LangGraph interrupt**: `langgraph interrupt()` gate for state preservation during HITL pauses
- **Angular components**: `StrategyBuilderComponent`, `HITLDialogComponent`, `StateStreamComponent`

### Node Implementations

- [ ] **Node 4: HITL Payload Generator** (batches missing rules & audit flags)
  - **Implementation**: Collects `AuditFlag[]` and missing `Quantity` from Node 3 output; batches into HITL payload
  - **Output**: `HITLPayload {missing_rules: Rule[], audit_flags: AuditFlag[], current_schema: SchemaDefinition}`
  - **Acceptance**: Payload contains all actionable items; batches are ≤ 20 items per page

- [ ] **Node 5: LangGraph `interrupt()` gate for state persistence**
  - **Implementation**: `interrupt()` call in LangGraph DAG after Node 3 → state saved to PostgreSQL → waits for user response
  - **Acceptance**: State is persisted when interrupt fires; `POST /resume` correctly resumes from checkpoint

- [ ] **Setup FastAPI SSE / WebSocket endpoints for real-time Angular communication**
  - **SSE endpoint**: `GET /api/v1/strategy/{session_id}/stream` — emits `StateUpdate` events as SSE
  - **WebSocket endpoint**: `WS /api/v1/strategy/{session_id}/ws` — bidirectional for HITL responses
  - **Message format**: JSON `{"type": "state_update" | "hitl_request" | "hitl_response", "payload": {...}}`
  - **Acceptance**: Angular receives real-time state updates; HITL responses flow back through WebSocket

- [ ] **Build State Resumption endpoint** (`POST /api/v1/strategy/resume`) to inject user answers
  - **Implementation**: Accepts `{session_id, answers: Record<string, string>}`; injects into LangGraph state; resumes DAG
  - **Acceptance**: User answers are merged into state; DAG continues from interrupt point

### Angular Frontend Components

- [ ] **`StrategyBuilderComponent`** — main workflow display, shows current state, extracted quantities, schema
- [ ] **`HITLDialogComponent`** — modal/dialog for missing rules & audit flags, submits responses via WebSocket
- [ ] **`StateStreamComponent`** — SSE consumer, displays real-time state updates, iteration count
- [ ] **`StrategyExportComponent`** — final export bundle display (links to Block D)

**Acceptance Criteria**: User can upload document → see extraction → respond to HITL prompts → see state update in real-time → strategy reaches complete/rejected state. WebSocket latency < 100ms.

**Dependencies**: Block B Step 1 (LangGraph core) must be functional.

---

# Block C: Statistical Stress Validation Engine

## Step 1: Vectorized Surface Sweep & CPCV

### Technical Architecture
- **Core library**: `vectorbt-pro` (or `vectorbt` free tier) for parameter sweeps
- **Validation framework**: Custom CPCV implementation with purging and embargoing
- **Visualization**: 3D surface plots via ECharts/Three.js in Angular

### Implementations

- [ ] **Implement VectorBT Pro parameter sweep engine**
  - **Implementation**: `vectorbt.Portfolio` with parameter grid; sweep over strategy parameters (e.g., lookback period, threshold, position size)
  - **Input**: Strategy parameters from Block B schema definition
  - **Output**: `SweepResult {parameter_grid: dict, metrics: dict[str, list[float]]}`
  - **Acceptance**: Sweep covers all parameter combinations; execution time < 60s for ≤1000 combinations

- [ ] **Build 3D Neighborhood Surface Decay Metric for parameter stability scoring**
  - **Implementation**: Compute parameter neighborhood (adjacent grid points); calculate performance decay as function of distance from optimal
  - **Metric**: `decay_score = 1 - (std(neighborhood_performance) / mean(neighborhood_performance))`
  - **Output**: `SurfaceMetric {optimal_params: dict, decay_score: float, neighborhood: list[dict]}`
  - **Acceptance**: Decay score ∈ [0, 1]; higher score = more stable parameter region

- [ ] **Implement Combinatorial Purged & Embargoed Cross-Validation (CPCV) framework**
  - **Implementation**: K-fold CV with purging (remove overlapping data between folds) and embargoing (gap between train/test)
  - **Configuration**: `n_splits=5`, `purge_window=30 days`, `embargo_window=15 days`
  - **Feature leakage checker**: Scan for lookahead bias across folds (e.g., same data in train and test)
  - **Output**: `CPCVResult {fold_metrics: list[dict], leakage_detected: bool, overall_score: float}`
  - **Acceptance**: No data leakage between folds; each fold has purged/embargoed windows; overall score is mean of fold metrics

- [ ] **Build feature leakage sanity checker** to detect lookahead bias across folds
  - **Implementation**: Check if any sample appears in both train and test; check if future data leaks into features
  - **Acceptance**: Flags all leakage scenarios in test suite; clean on valid CPCV configuration

**Acceptance Criteria**: Parameter sweep completes with valid metrics; CPCV shows no leakage; surface decay score is computed for every parameter region.

**Dependencies**: Block B (strategy schema) must define parameters to sweep.

---

## Step 2: Monte Carlo & Portfolio Allocation

### Technical Architecture
- **Simulation**: Block Bootstrap + Monte Carlo return resampling
- **Allocation**: `riskfolio-lib` for HRP, Risk Parity, MVO
- **Visualization**: Monte Carlo percentile bands in Angular (P10/P50/P90)

### Implementations

- [ ] **Build Block Bootstrap Return Resampling engine** (volatility clustering test)
  - **Implementation**: Block bootstrap with block size = 20 trading days; resample blocks with replacement; test for volatility clustering (GARCH effects)
  - **Output**: `BootstrapResult {returns: list[float], volatility_clusters: list[list[float]], garch_effects_detected: bool}`
  - **Acceptance**: 1000 bootstrap samples generated; GARCH effects detected if present in original returns

- [ ] **Build Execution Perturbation engine** (order skipping & dynamic slippage penalty)
  - **Implementation**: Simulate order execution with random skip probability (5-15%); apply dynamic slippage based on volatility (ATR-based)
  - **Output**: `PerturbationResult {skipped_orders: int, slippage_bps: float, adjusted_returns: list[float]}`
  - **Acceptance**: Slippage scales with volatility; skip rate is configurable; adjusted returns reflect realistic execution

- [ ] **Integrate Riskfolio-Lib Abstract Allocation Matrix Interface** (HRP, Risk Parity, MVO)
  - **Implementation**: `riskfolio-lib` `Portfolio` class with three allocation methods; compare Sharpe ratios
  - **Input**: Covariance matrix from bootstrap results; expected returns from CPCV
  - **Output**: `AllocationResult {method: str, weights: dict[str, float], sharpe: float, max_drawdown: float}`
  - **Acceptance**: All three methods produce valid weights; HRP shows best robustness, MVO shows highest return but highest drawdown

- [ ] **Implement Validation Report Payload serializer** for frontend streaming
  - **Implementation**: Serialize all Block C results into `ValidationReport` JSON; stream via SSE
  - **Schema**: `ValidationReport {cpcv: CPCVResult, bootstrap: BootstrapResult, perturbation: PerturbationResult, allocation: AllocationResult, surface: SurfaceMetric}`
  - **Acceptance**: Report serializes without error; SSE streams complete report to Angular; report is < 1MB

**Acceptance Criteria**: Monte Carlo simulation produces P10/P50/P90 bands; portfolio allocation returns valid weights for all three methods; validation report is streamed to frontend.

**Dependencies**: Block B (strategy definition) and Block C Step 1 (parameter sweep) must be complete.

---

# Block D: Execution Transpiler & Display Factory

## Step 1: AST Transpiler & Risk Guardrail Injection

### Technical Architecture
- **Transpiler**: Jinja2 templates for code generation (.mq5 EA / Python)
- **Guardrails**: Decorator/injection pattern for risk management modules
- **Export**: Bundled .mq5 + .py files + JSON report

### Risk Guardrail Modules (injected into generated code)

- [ ] **Build Jinja2 AST Transpiler Engine** for target code generation (.mq5 EA / Python)
  - **Implementation**: Jinja2 templates parameterized by strategy schema + validation results; generates MetaTrader 5 MQL5 EA or Python algo
  - **Input**: `StrategyDefinition` + `AllocationResult` + `ValidationReport`
  - **Output**: `.mq5` or `.py` source file
  - **Acceptance**: Generated code compiles/runs without syntax errors; all strategy logic is present

- [ ] **Inject Global Max Drawdown Circuit Breaker & Kill-Switch module**
  - **Implementation**: Generated code includes `if current_drawdown > max_drawdown_threshold: close_all_positions()` + `kill_switch = True`
  - **Configuration**: `max_drawdown_pct` from strategy schema (default 20%)
  - **Acceptance**: Circuit breaker triggers at threshold; all positions closed; kill-switch locks account

- [ ] **Inject Daily Loss Limit & Consecutive Loss Cooldown timers**
  - **Implementation**: `daily_loss_limit = account_balance * max_daily_loss_pct`; `consecutive_loss_count` tracker; cooldown = N minutes after max consecutive losses
  - **Configuration**: `max_daily_loss_pct` (default 5%), `max_consecutive_losses` (default 5), `cooldown_minutes` (default 30)
  - **Acceptance**: Daily loss limit enforced; cooldown activates after threshold; timers reset at midnight

- [ ] **Inject Dynamic Position Sizing** (ATR/Volatility adjusted lot sizes)
  - **Implementation**: `lot_size = (account_risk_pct * account_balance) / (atr * point_value)`; volatility-adjusted multiplier
  - **Configuration**: `risk_per_trade_pct` (default 1%), `atr_period` (default 14)
  - **Acceptance**: Lot sizes scale inversely with ATR; minimum lot = broker minimum; maximum lot = risk cap

- [ ] **Inject Market Microstructure Protection** (Spread/Slippage guards)
  - **Implementation**: `if spread > max_spread_points: skip_trade()`; `slippage_limit = max_slippage_pips`; order rejection on violation
  - **Configuration**: `max_spread_points` (default 3), `max_slippage_pips` (default 5)
  - **Acceptance**: Trades skipped when spread exceeds limit; slippage capped; orders rejected on violation

- [ ] **Inject Unique Magic Number & Local State Persistence handler**
  - **Implementation**: `magic_number = hash(strategy_id + session_id)`; local state file stores open positions, P&L, trade history
  - **Configuration**: State file at `data/state/{session_id}.json`
  - **Acceptance**: Unique magic number per strategy; state persists across sessions; state file is valid JSON

**Acceptance Criteria**: Generated code includes all 6 guardrail modules; magic numbers are unique; state persists across restarts; generated .mq5/.py files are syntactically valid.

**Dependencies**: Block C (validation results) must be complete.

---

## Step 2: Backend Data Aggregation & Angular Display

### Technical Architecture
- **Downsampling**: LTOB (Largest-Triangle-Three-Buckets) for time-series
- **Aggregation**: Monte Carlo percentile bands (P10/P50/P90)
- **Visualization**: Angular components with ECharts/Three.js
- **Export**: Final strategy bundle (.mq5/.py + JSON report)

### Implementations

- [ ] **Implement LTOB (Largest-Triangle-Three-Buckets) downsampling** for time-series charts
  - **Implementation**: `ltob(data, threshold)` — reduces N points to `threshold` buckets; preserves visual shape
  - **Input**: Full time-series (equity curve, drawdown, P&L)
  - **Output**: Downsampled array of `{x, y}` points ≤ threshold
  - **Acceptance**: Downsampled curve visually matches original within 5% tolerance; threshold is configurable

- [ ] **Implement Monte Carlo percentile band aggregator** (P10, P50, P90 density reduction)
  - **Implementation**: Sort all MC paths; extract P10, P50, P90 at each time step; compress into 3-band visualization data
  - **Input**: 1000+ Monte Carlo paths
  - **Output**: `{upper: list[float], median: list[float], lower: list[float], x_axis: list[int]}`
  - **Acceptance**: Bands are monotonic; median equals mean of symmetric distribution; compression ratio ≥ 100x

- [ ] **Build Angular WebGL / ECharts display components** for 3D Heatmaps and MC distributions
  - **Components**: `Heatmap3DComponent` (parameter surface), `MCDistributionComponent` (percentile bands), `EquityCurveComponent` (LTOB downsampled), `ValidationDashboardComponent` (aggregate all Block C/D results)
  - **Library**: ECharts for 2D charts; Three.js/WebGL for 3D heatmaps
  - **Acceptance**: All components render without errors; 3D heatmap is interactive (rotate/zoom); MC bands update in real-time

- [ ] **Implement final strategy export bundle packager** (.mq5/.py files + JSON report)
  - **Implementation**: Zip bundle containing `.mq5`, `.py`, `report.json`, `README.md`; downloadable via `GET /api/v1/strategy/{session_id}/export`
  - **Report schema**: `ExportBundle {strategy_definition, validation_results, allocation, guardrails, generated_code}`
  - **Acceptance**: Bundle downloads successfully; all files present; JSON report is complete and valid

**Acceptance Criteria**: All display components render correctly; LTOB/MC aggregation works; export bundle is downloadable and complete.

**Dependencies**: Block C Step 2 must be complete.

---

## Quick orientation for the next session
- Working dir: `/home/_7oss/Desktop/Black_Box`
- Backend: `backend/` (uv venv at `backend/.venv`, package `black_box` under `backend/src/black_box`)
- Frontend: `frontend/` (Angular 22 scaffold)
- **Immediate action**: Create `.env`, run boot test, resolve DoclingService singleton, land HybridChunker
- Docling docs from **Context7 MCP** (`/websites/docling`); FastAPI docs from **Context7 MCP** (`/libraries/fastapi`)
- Testing: pytest + pytest-asyncio + httpx (backend), Vitest + Angular tests (frontend), Playwright (e2e)
- CI: GitHub Actions (lint → unit → integration → e2e → build)
- DB: PostgreSQL (not SQLite) for Block B/C state
- Deployment: Docker + Docker Compose
