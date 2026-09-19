# Black_Box — Build Plan & Session Handoff

> **Repo** `yussef96795/Black_Box` (public) · **Baseline commit** `d244c73` → `9be5fee`
> **Stack** FastAPI backend (uv) + Docling 2.129.0 + Angular 22 frontend scaffold
> **Today** Sat Sep 19 2026
> **Doc refs** Docling official docs `https://docling-project.github.io/docling/` (hybrid_chunking, concepts/chunking, reference/document_converter) — ground truth for the installed 2.129.0 API

---

## Current Status (block 1 of Block A complete)

### ✅ Finished
- **Git repo**: public repo `yussef96795/Black_Box` on `main`; baseline `d244c73` + feature commit `9be5fee` (backend pipeline). No secrets in tree (`.env` gitignored).
- **Boot + smoke test** — uvicorn boots clean; `GET /health` → `{"service":"Black_Box","version":"0.1.0","docling":"ready"}`; markdown strategy doc → 201 with 5 heading-provenanced chunks; garbage PDF → 422.
- **DoclingService ownership resolved** (SRP): `get_docling_service` now serves the lifespan-managed instance via `request.app.state.docling`; module-level `_ServiceHolder` singleton dropped. Single owner per worker.
- **Real HybridChunker semantic chunking** landed with the **installed Docling 2.129.0 API** (supersedes earlier plan claims):
  - `HybridChunker()` takes **no** `chunk_by_documents` param; `chunker.chunk(dl_doc=doc)` returns an iterator.
  - `DocumentStream(name=…, stream=BytesIO)` replaces hand-rolled `_BytesSource` (no tempfile — Rules.md §5).
  - `DocumentConverter(allowed_formats=[…])` **is** built in (no hand-rolled magic-byte sniff needed) + `max_file_size`/`max_num_pages` caps.
  - `ConversionError` → **422** (client parse failure), not 502.
- **1c · Variable Resolution Module** — `services/math_resolver.py`: curated unicode + LaTeX symbol table, resolves σ/×/`$\sigma$` etc. in chunk text, emits `MathResolution{original, resolved, symbol_table, context}`; no heavyweight math engine (KISS, Rules.md §1).
- **1d · Table Schema Validator** — `services/table_validator.py`: extracts `TABLE` items via `TableItem.data.grid`, validates rows against expected `Param/Value/Bounds` schema (columns matched **by header name**), emits per-table `fit | orphan | parse-error` reports.
- **API contract** (`schemas.py`): `ChunkOut`, `TableValidationReport`, `MathResolution`, `IngestResponse` (+ `tables`, `math` fields). `_UNPROCESSABLE = HTTP_422_UNPROCESSABLE_CONTENT` (deprecation fix).
- **Test suite**: 17 tests pass (`pytest`), ruff lint + format clean. Unit: math resolution, table validation. Integration (real Docling converter): health, markdown ingest with table-tagged chunks + `fit` report + σ resolution, missing/empty/garbage → 422, settings defaults.
- **README + `.env.example`**: backend docs and config template; dev deps (pytest, pytest-asyncio, httpx, ruff) in `pyproject.toml`.

### ⏭️ Next / Remaining (in priority order)
1. **Block A Step 2 — Feasibility Auditor Gate**: capabilities JSON contract, hard-dependency checker, compute checker, domain mapping engine, `PASSED|REQUIRES_HITL|REJECTED` result schema at `POST /api/v1/feasibility/audit`.
2. **Parallel track (recommended, see "Parallel Tracks" below)**: Angular infra skeleton — routing, `StrategyState` shared types, SSE/WS clients, component shells.
3. **Block B**: LangGraph state machine → HITL gate → dashboard visualization (post-Block C, when data contracts exist).
4. **Block C → Block D**, then CI/CD hardening (see Testing Strategy).

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

## Step 1: Document Processing Pipeline — ✅ COMPLETE (commits `9be5fee`)

### 1a · Thin FastAPI ingestion endpoint — ✅ DONE
- [x] `Settings` + `get_settings` from env/.env (`core/config.py`) — incl. `max_upload_size`, `max_num_pages`, `allowed_formats`/`allowed_format_list`
- [x] `DoclingService` with single owner: **DI serves the lifespan-managed instance** (`request.app.state.docling`), lazy async converter init, `aclose()`, `accel()` probe
- [x] `POST /api/v1/documents/ingest` route: validates part presence → 422; empty file → 422; oversized → 413; `ConversionError` → 422 (client parse failure)
- [x] FastAPI DI wiring: `Depends(get_docling_service)` + lifespan lifecycle
- [x] `/health` probe → `{"service", "version", "docling": "ready|pending"}`
- [x] Pydantic response contract: `ChunkOut` / `IngestResponse` (stable for B/C/D)
- [x] `_UNPROCESSABLE = HTTP_422_UNPROCESSABLE_CONTENT` (Starlette deprecation fix)

**Acceptance Criteria (met)**: boots clean; `/health` 200; no-file → 422; markdown strategy → 201 with `document_id`, `chunk_count`, `chunks`; garbage PDF → 422.

### 1b · Structural semantic chunking — ✅ DONE
- [x] **Ownership consolidated** — single `DoclingService` instance per worker via lifespan (verified by tests; module singleton dropped)
- [x] **Real Docling `HybridChunker`** (installed 2.129.0 API):
  - `HybridChunker()` — **no `chunk_by_documents` param in 2.129.0** (older docs were stale); `chunker.chunk(dl_doc=doc)` returns an **iterator** over `DocChunk`
  - Real token counts via `chunker.tokenizer.count_tokens(text)` (HuggingFaceTokenizer)
- [x] `Chunk` fields fully emitted: `id` (`{doc_id}:{idx}`), `text`, `page` (from item provenance, PDFs), `heading` (` | `-joined headings), `tokens` (real), `meta` (incl. `doc_item_labels` for table/equation/section tagging)
- [x] `DocumentStream(name=…, stream=BytesIO)` — Docling's first-class in-memory input; **no tempfile, no secrets on disk** (Rules.md §5)
- [x] Type allowlist via **built-in** `DocumentConverter(allowed_formats=[InputFormat…])` (settings-driven `ALLOWED_FORMATS`) + `max_file_size`/`max_num_pages` caps — hand-rolled magic-byte sniff NOT needed
- [x] `.env.example` with defaults; `.env` (gitignored) active for dev
- [x] Boot + end-to-end smoke: 201, 5 chunks, table-tagged chunk present, `doc_item_labels: ["table"]`, code preserved

**Acceptance Criteria (met)**: valid MD/PDF → 201 with all 6 chunk fields; empty → 422; oversized → 413.

### 1c · Variable Resolution Module (LaTeX → text definitions) — ✅ DONE
- [x] Curated `SYMBOL_TABLE`: unicode symbols Docling emits (σ→sigma(annualized volatility), ×→multiplied by, ≤, ≥, Σ, Δ…) **and** single-backslash LaTeX keys (`r"\sigma"`, `r"\sum"`…) for `$…$`/`\(…\)` spans
- [x] `resolve_math_in_text(text, context)` → `MathResolution | None`; **actually replaces** symbols in text (original kept, resolved emitted alongside)
- [x] Emits `{"original", "resolved", "symbol_table", "context"}` per occurrence; Block C consumes
- [x] Unit-tested: unicode symbol, inline LaTeX, no-math → None, context carried

**Acceptance Criteria (met)**: σ and × in strategy smoke doc resolved to human-readable text with symbol table + context; original token stream preserved.

*Note*: deliberately no `sympy`/`latex2text` — regex + curated table covers the quant-domain vocabulary (KISS, Rules.md §1). Upgrade if LLM extraction surfaces broader LaTeX.

### 1d · Table Schema Validator — ✅ DONE
- [x] Extracts `TABLE` items from Docling document via `TableItem.data.grid` (rows of cells with `.text`) — verified with real HTML/MD probes
- [x] `TableSpec`/`ColumnSpec` schema config; columns matched **by header name** (specs may cover a subset of grid columns). Default schema: `Param/Value/Bounds` with number + `[min, max]` range kinds
- [x] Per-table report: `{"table_id", "status": "fit"|"orphan"|"parse-error", "rows", "columns", "errors"}` — included in `IngestResponse.tables`
- [x] Unit-tested: fit, bad value, inverted range, orphan headers, broken grid, custom spec; integration asserts `fit` on the smoke doc

**Acceptance Criteria (met)**: strategy smoke doc table reported `fit` (5 rows, 3 cols, no errors); orphan/parse-error paths tested.

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

## Parallel Tracks: Dashboard Sequencing (critical review finding)

**Rejected**: strict waterfall A → B → C → D, with the entire Angular dashboard waiting until Block D.

**Why**: Block B is a **human-in-the-loop** workflow — the whole point is a user reviewing/editing strategy formulations. Without any UI, that validation can only happen via raw curl against LangGraph endpoints, which defeats the HITL purpose and delays the riskiest feedback loop (does the strategy spec actually match what a quant expects?) until the very end.

**Adopted — hybrid approach**:

| Track | Timeline | Contents |
|---|---|---|
| **A — Backend** | Block A → B → C → D (unchanged critical path) | Ingestion + feasibility → formulation state machine → statistical validation → transpiler |
| **B — Angular infrastructure** | **Starts in parallel with Block A Step 2** | routing shells (`/upload`, `/formulate`, `/validate`, `/dashboard`), `StrategyState` TS types mirroring the Python state schema, HTTP + SSE/WS client services (mock providers first), component skeletons with loading/error states |
| **C — Dashboard visualization** | **After Block C data contracts exist** | heatmaps, Monte Carlo bands, equity curves — rendered from real Block C payloads once the contracts are frozen |

**Guardrails** (prevent the frontend from running ahead of the API):
1. Angular track is **infra-only until Block B endpoints exist**: no hard-coded business logic, no fake strategy semantics in components. Mock providers are explicitly labeled and swappable via token injection.
2. `StrategyState` TS types are generated/vendor-synced from the Python pydantic schema (single source of truth, Block B Step 1 §State Schema).
3. Visualization widgets are **shelved after the Block C contract freeze** (`ValidationReport`/`SurfaceSweepResult` shapes) — never built against guesswork.
4. Each backend block ships a **contract-first OpenAPI update**; the Angular services track it so drift is caught by tests (see Testing Strategy ‑ API Contract).

**Why this is the right trade**: it front-loads the HITL interface (the actual product), keeps the backend contract the only authority, and avoids a giant "frontend month" at the end. Cost: disciplined scope control on track B (infra ≠ features).

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
