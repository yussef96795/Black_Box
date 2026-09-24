# Black_Box — Build Plan & Session Handoff

> **Repo** `yussef96795/Black_Box` (public) · **Baseline commit** `d244c73` → `b068a35` (Block A complete)
> **Stack** FastAPI backend (uv) + Docling 2.129.0 + Angular 22 frontend scaffold
> **Today** Sat Sep 19 2026
> **Doc refs** Docling official docs `https://docling-project.github.io/docling/` (hybrid_chunking, concepts/chunking, reference/document_converter) — ground truth for the installed 2.129.0 API

---

## Current Status (Block A complete + Alpha engine implemented)

### ✅ Finished
- **Git repo**: public repo `yussef96795/Black_Box` on `main`; baseline `d244c73` + commits `9be5fee`, `7d207f2`, `b068a35` (Block A complete). No secrets in tree (`.env` gitignored).
- **Block A Alpha engine (Stages A1–A6)** — LLM extraction → catalog hard-stop → 8-operator sweep → DSL compiler → human gatekeeper → `out/block_a_specs.json` (see Step 3, section below). 93 Block A tests; full suite 137 passing; real-Ollama live run emits 4 validated specs.
- **Boot + smoke test** — uvicorn boots clean; `GET /health` → `{"service":"Black_Box","version":"0.1.0","docling":"ready"}`; markdown strategy doc → 201 with 5 heading-provenanced chunks; garbage PDF → 422.
- **DoclingService ownership resolved** (SRP): `get_docling_service` now serves the lifespan-managed instance via `request.app.state.docling`; module-level `_ServiceHolder` singleton dropped. Single owner per worker.
- **Real HybridChunker semantic chunking** landed with the **installed Docling 2.129.0 API** (supersedes earlier plan claims):
  - `HybridChunker()` takes **no** `chunk_by_documents` param; `chunker.chunk(dl_doc=doc)` returns an iterator.
  - `DocumentStream(name=…, stream=BytesIO)` replaces hand-rolled `_BytesSource` (no tempfile — Rules.md §5).
  - `DocumentConverter(allowed_formats=[…])` **is** built in (no hand-rolled magic-byte sniff needed) + `max_file_size`/`max_num_pages` caps.
  - `ConversionError` → **422** (client parse failure), not 502.
- **1c · Variable Resolution Module** — `services/math_resolver.py`: curated unicode + LaTeX symbol table, resolves σ/×/`$\sigma$` etc. in chunk text, emits `MathResolution{original, resolved, symbol_table, context}`; no heavyweight math engine (KISS, Rules.md §1).
- **1d · Table Schema Validator** — `services/table_validator.py`: extracts `TABLE` items via `TableItem.data.grid`, validates rows against expected `Param/Value/Bounds` schema (columns matched **by header name**), emits per-table `fit | orphan | parse-error` reports.
- **API contract** (`schemas.py`): `ChunkOut`, `TableValidationReport`, `MathResolution`, `IngestResponse` (+ `tables`, `math`), feasibility models (`DataDependency`, `ModelCheck`, `DomainMapping`, `FeasibilityAuditRequest`, `FeasibilityCheck`, `FeasibilityResult`). `_UNPROCESSABLE = HTTP_422_UNPROCESSABLE_CONTENT` (deprecation fix).
- **Test suite**: 29 tests pass (`pytest`), ruff lint + format clean. Unit: math resolution, table validation, feasibility checkers. Integration (real Docling converter): health, markdown ingest with table-tagged chunks + `fit` report + σ resolution, missing/empty/garbage → 422, settings defaults, capabilities + audit endpoints.
- **README + `.env.example`**: backend docs and config template; dev deps (pytest, pytest-asyncio, httpx, ruff) in `pyproject.toml`.
- **2 · Feasibility Auditor Gate** — `core/capabilities.py` (capability contract, `GET /api/v1/feasibility/capabilities`) + `services/feasibility_auditor.py` (4 rule-based checkers: hard dependencies, algorithmic/compute, domain mapping, tables) + `POST /api/v1/feasibility/audit` → `PASSED | REQUIRES_HITL | REJECTED`. Live-verified: crypto-native → PASSED, LSTM → REJECTED.

### ⏭️ Next / Remaining (in priority order)
1. **Block B: Formulation Engine** — LangGraph state machine core (`state/schema.py`, nodes, PostgreSQL checkpointing) → HITL gate. **(deps staged by coordination port; ownership TBD — see Port Coordination)**
2. **Block C** statistical stress validation (contract freeze gates dashboard work).
3. **Block D** transpiler + backend aggregation, then CI/CD hardening (see Testing Strategy).
4. **Frontend (PAUSED)** — restart only after Stitch/Figma design handoff (user decision).

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

## Step 2: Block A.5 Feasibility Auditor Gate — ✅ DONE
- [x] **System Capabilities JSON contract** (`core/capabilities.py`, `Capabilities` pydantic model, `version: 1.0.0`) — data (crypto L3 vs unsupported equity/rates), compute (vectorized backtest only; ML inference unsupported), domain (24/7 perpetuals, funding-rate analog…). Exposed at `GET /api/v1/feasibility/capabilities`.
- [x] **Hard Dependency Checker** — keyword taxonomy over the strategy text: crypto-native providers (Binance/Coinbase/OKX/CCXT…) → `satisfied`; TradFi providers (Bloomberg/Reuters/FRED/Yahoo…) → `partial` when a mentioned domain concept offers an analog (needs HITL), else `missing`; granularity demands (tick/L3/orderbook/daily…) checked against capability table. Output: `[{name, status: satisfied|missing|partial, latency_ms, evidence}]` (`latency_ms: null` until real feeds land).
- [x] **Algorithmic & Compute Checker** — black-box models (neural network/LSTM/xgboost/LLM…) → REJECTED; compute hints (GPU / real-time inference / high-frequency) → REQUIRES_HITL. Output: `{models: [{name, supported, reason}], compute_budget: {gpu_memory_mb, inference_ms, hints}}`.
- [x] **Domain Mapping Engine** — 12 TradFi→crypto analogs with confidence (interest rate→funding rate 0.9, settlement→instant 0.85, trading hours→24/7 0.95, dark pool→RFQ 0.7, FOMC→macro/on-chain 0.6…). Confidence < 0.7 → REQUIRES_HITL. Output: `[{tradfi, crypto, confidence, note}]`.
- [x] **Table gate** — 1d validation reports from ingest feed the audit: non-`fit` tables (parse-error/orphan) → REQUIRES_HITL.
- [x] **Feasibility Result schema** — `POST /api/v1/feasibility/audit` body `{text, tables?}` → `{status: PASSED|REQUIRES_HITL|REJECTED, checks: [{name, status, details}], summary}`. Aggregate: any REJECTED → REJECTED; else any REQUIRES_HITL → REQUIRES_HITL; else PASSED.

**Acceptance Criteria (met)**: audit endpoint returns structured result; all 4 checkers execute in sequence; result schema matches contract; `PASSED` requires all hard dependencies satisfied. 12 new tests (unit + API integration); live curl verified PASSED (crypto-native) and REJECTED (LSTM) paths.

**Dependencies**: Block A Step 1 (full pipeline) stable — required, and satisfied.

---

## Step 3: Alpha Feasibility & Strategy Ingestion Engine (Stages A1–A6) — ✅ DONE

Implements the Alpha engine per `BLOCK_A_SPECIFICATION.md`: LLM-driven mechanism
extraction bound to Pydantic v2 via Instructor, DuckDB/polars resource-catalog
checks, an annotations-only 8-operator sweep (zero deletion authority), a DSL
spec compiler (≤5 pruned specs, Tiers 0–3), a human gatekeeper, emitting a
validated `ExecutableStrategySpec` JSON array to `/out/block_a_specs.json`.

- [x] **Schema-first LLM contract** — every stage binds to Pydantic v2 via Instructor (`extra="forbid"` everywhere): `PaperExtractionSchema` (A1–A2), `CausalAbstractionSchema` (A4), `OperatorAnnotations` (A5, `min_length=8`). Local Ollama via `instructor.from_provider("ollama/<model>", mode=Mode.JSON_SCHEMA)`; `max_retries` on `create` only (spec §6: max 3), verified hermetic via `from_openai(OpenAI(http_client=MockTransport…))`.
- [x] **Stage A3 data catalog** — `catalog.py` + packaged `data_catalog.parquet` (18-row seed: BTC/ETH/SOL full L2+orderflow tick/1m/1h/1d from 2020; ADA/DOGE OHLCV-only from 2021); DuckDB/polars `check_catalog` with L2→`volume_delta:1m` proxy rule; **hard-stop authority** for Stage A5 emission (incomplete data → `RESOURCE_INSUFFICIENT_ERROR`, no specs emitted). Catalog is authoritative for `asset_class` in verdicts.
- [x] **Stage A5 operator sweep (no-drop)** — annotations-only; `StrategyAnnotation` has no `is_testable` field; the compiler never emits when the anchor dataset is unverified (structural no-drop mirror of the hard stop).
- [x] **Stage A6 DSL spec compiler** — `spec_compiler.py`: primitives registry (`triggers/indicators/risk_filters`), ≤5 pruned specs across `TIER_0_LITERAL` / `TIER_1_PARAMETRIC` / `TIER_2_GENERALIZED` / `TIER_3_AUGMENTED`, `is_testable` set only by data-resource position; `validate_and_export` writes `block_a_specs.json`.
- [x] **Human gatekeeper + CLI** — `gatekeeper.py` rich review (per-spec approve/reject), `cli.py` exit codes 0/1/2/3/4, `--llm fake` for CI, sync Docling parser.
- [x] **Stateless processing** — fresh LangGraph graph + MemorySaver per run (no cross-paper leakage; tested with unique paper markers); `ui = llm evaluator` recorded call counts assert exactly 3 fresh calls per run.
- [x] **Test suite (93 Block A tests)** — catalog, models, LLM/evaluator (incl. hermetic Instructor retry loop), compiler, DAG (hard-stop QQQ path, gatekeeper veto, real Docling integration, stateless isolation), CLI. Full suite 137 passed; ruff lint + format clean on all touched files.
- [x] **Live-verified** — real Ollama (llama3.2 3B) run on `tests/fixtures/block_a/vwap_trend.md` emits 4 specs (T0/T1 BTCUSDT, T2 ETHUSDT, T3 BTCUSDT) with exit 0 (T2's alternate symbol is now chosen from the catalog — granularity-first, no hardcoded target; see the deterministic-hardening batch below). `Mode.JSON_SCHEMA` (structured output with the actual Pydantic schema) was required for small-model compliance; `Mode.JSON` double-encodes arrays on the 3B model.
- [x] **Feasibility auditor integration** — granularity authority rewired from the static capability table to the DuckDB catalog (`audit(..., catalog_path=…)`, graceful fallback keeps existing contract/tests green).
- [x] **Deployments/config** — 6 new `BLOCK_A_*` settings in `core/config.py` + `.env.example`; deps `instructor 1.17.0`, `duckdb 1.5.5`, `polars 1.44.2`, `rich 14.3.4` via `uv add`; lockfile updated.
- [x] **Remediation P1 — recursive DSL AST** — hybrid `signal_ast: GenericPrimitiveNode | None` on `ExecutableStrategySpec` (flat fields stay the Block C/D boundary): `DataStreamNode`/`OperandNode`/`OperatorNode` discriminated union, `extra="forbid"`, `MAX_AST_DEPTH=8` recursive ceiling, Literal ops + arity rules, compile-time registry guard `validate_ast_registry`; Tier 3 emits `ZScore(<secondary indicator>)`.
- [x] **Remediation P2 — graceful LLM failure** — exhausted retries/`ValidationError` on any LLM stage → `status="llm_extraction_failed"` + `stage_error {stage, error_type, detail, trace_hash}`, terminal `llm_failed` DAG node (partial state preserved, nothing dropped), CLI exit `4 EXIT_LLM_FAILED`.
- [x] **Remediation P3 — live observability** — per-paper JSONL traces (`<out_dir>/traces/<paper_id>.jsonl`, `BLOCK_A_TRACE_ENABLED` default off) recording request/response/error frames + instructor event-hook raw completions/retries/usage via `TraceWriter`; `BLOCK_A_TRACE_DIR` setting.
- [x] **Remediation P4 — granular CLI gatekeeper** — per-spec `[a]pprove/[r]eject/[t]oggle risk tags/[q]uit` loop returning a curated `GatekeeperDecision {status, specs}` (D4): `confirm_gatekeeper` drives the rich loop, re-validates edited specs, back-compat preserved (`"n"` reject-all, `"y"`/`approve=False` veto paths unchanged); `BlockAResult.risk_tags` recomputed from the curated array; full rejection → no file, exit 0.
- [x] **Block A deterministic-hardening batch** — one conventional commit per item, full suite + ruff green after each (`a18bca7`→`f561e32`): P0 `5m`/`15m` in `DataGranularity` + `core_mechanism` length bounds (50–2000); P1.1 `RISK_PRIORITY` single-sourced in models and drives the gatekeeper menu; P1.2 `check_resources` structured through `_call_stage` with a terminal `llm_failed` DAG edge; P1.3 Tier 3 secondary signal derived from the registry; P1.4 Tier 2 alternate symbol chosen granularity-first from the catalog (no fixed symbol, L2 filter dropped); P1.5 entry/exit phrase tables split per clause so keywords never cross-contaminate; P2.1 phrase/risk-guard/data-series tables sourced from `primitives_registry.json` with module fallbacks; P2.2 `AssetClass` enum on `DatasetRequirement`, `DataStreamNode.granularity: DataGranularity`, `operator_name` Literal derived from `OPERATOR_NAMES`.

**Acceptance Criteria (met)**: zero raw-text output (all LLM stages schema-bound); stateless isolation across papers; retry loop ≤3 (hermetic test: garbage-forever raises `InstructorRetryException` after 3 attempts); export format validated array; no-drop semantics enforced structurally and via the A3 hard stop.

## Parallel Tracks: Dashboard Sequencing (critical review finding)

**Rejected**: strict waterfall A → B → C → D, with the entire Angular dashboard waiting until Block D.

**Why**: Block B is a **human-in-the-loop** workflow — the whole point is a user reviewing/editing strategy formulations. Without any UI, that validation can only happen via raw curl against LangGraph endpoints, which defeats the HITL purpose and delays the riskiest feedback loop (does the strategy spec actually match what a quant expects?) until the very end.

**USER DECISION (Sep 19 2026) — frontend deferred**: the user will design the dashboard in **Stitch / Figma first**. Until designs are handed off:
- **NO frontend work in any port** (Angular scaffold stays untouched at baseline).
- Backend-only focus across all ports.
- Dashboard *backend* data aggregation (Block D Step 2 backend half) may proceed without the UI.

**Backend track (unchanged critical path)**: Block A → B → C → D, all backend.

| Track | Timeline | Contents |
|---|---|---|
| **A — Backend** | Block A → B → C → D | Ingestion + feasibility → formulation state machine → statistical validation → transpiler + aggregation |
| **B — Frontend** | **PAUSED — gated on Stitch/Figma design handoff (user decision)** | routing shells, `StrategyState` TS types, SSE/WS clients, component shells |
| **C — Dashboard visualization** | **After Block C data contracts exist** (and design handoff) | heatmaps, Monte Carlo bands, equity curves — rendered from real Block C payloads once contracts are frozen |

**Guardrails** (prevent the frontend from running ahead of the API):
1. No frontend work at all until the user hands off Stitch/Figma designs.
2. `StrategyState` TS types are generated/vendor-synced from the Python pydantic schema (single source of truth, Block B Step 1 §State Schema) — never hand-drifted.
3. Visualization widgets are built only after the Block C contract freeze (`ValidationReport`/`SurfaceSweepResult` shapes) — never against guesswork.
4. Each backend block ships a **contract-first OpenAPI update**; when frontend restarts, its services track it so drift is caught by tests (see Testing Strategy ‑ API Contract).

---

## Port Coordination (3 active sessions, shared repo `yussef96795/Black_Box`)

**Shared source of truth**: this `plan.md` (mirrored to `/home/_7oss/.opencode/plan/plan.md`). Always `git fetch` + update from `main` before starting new work; never commit another port's uncommitted files.

| Session | Role | Scope / status |
|---|---|---|
| `ses_f45281638ffeaQ71eobzNxX0kA` | **plan.md owner + Block A implementer** | plan.md maintained; Block A complete (commits `9be5fee`, `7d207f2`, `b068a35`); CI workflow landed; next: uncontested backend work (contract-first), keeps plan.md + `.opencode/plan` mirrored |
| `ses_f4501726affee6p7oFKTwmH5hh` | **Backend coordination** | coordinates ports; Block B deps staged in `backend/pyproject.toml` (uncommitted — do NOT commit); likely owns LangGraph state machine |
| `ses_f45016b1cffe9r5F2v810pGS1X` | **Backend-only dashboard development** | Block D Step 2 backend aggregation — needs Block C contract freeze before consuming; build service skeleton against provisional contract, mark clearly |

**Handoff conventions**:
1. `git pull --rebase` origin/main before pushing; small, focused commits with conventional prefixes.
2. Uncommitted foreign changes (e.g. `backend/pyproject.toml`) are left untouched.
3. Block B state schema lives at `backend/src/black_box/state/schema.py` (only one port may own it — flag who takes it).
4. Update `plan.md` ✅/⏳ marks when a block lands; push quickly so others see progress.

---

# Block B: Formulation Engine & Human-In-The-Loop

> **Status: ✅ shipped (v1, deterministic core).** Architecture note: the
> original PydanticAI/SymPy/WS design was replaced mid-implementation — see
> "Design vs. implemented" at the end of Step 1. The API contract below is
> frozen; the frontend (Step 2) is paused on the Stitch/Figma handoff.

## Step 1: LangGraph State Machine Core ✅

### Technical Architecture
- **Framework**: `langgraph` (v1.2.11) `StateGraph` over a typed Pydantic
  `StrategyState`; `interrupt()`/`Command(resume=...)` for HITL
- **Checkpointing**: in-process `MemorySaver` (live resume) + disk JSON
  snapshot `out/strategy/<session_id>.json` (crash-safe read store).
  After a process restart a parked session is re-parked deterministically
  by replaying the pre-audit chain onto `hitl_gate` (`replay_park`).
  **PostgreSQL checkpointing is the documented upgrade path, not built**
  (ponytail: one durable store for v1).
- **State schema**: single source of truth at
  `backend/src/black_box/state/schema.py` (msgpack-safe primitives) —
  interface contract for the TS mirror `frontend/src/app/strategy/strategy-state.ts`
- **Round cap**: `round < max_rounds (= 2)` enforced by a `ConditionalEdge`;
  `iteration_count` is a monotonic observability counter (NOT the control —
  a parked HITL stream would trip a graph-level `max_iterations` dead-man).

### Data flow
```
hydrate_spec → structural_gate → cynical_auditor ─ router ─► mark_complete ─► END
                                                  │          └► mark_rejected ─► END
                                                  └► hitl_payload → hitl_gate (interrupt)
                                                     Command(resume) → apply_answers ─► structural_gate (loop)
```

### Node Implementations

- [x] **Setup LangGraph DAG runtime + restart-resilient checkpointing**
  - **Implementation**: `state/graph.py` (`build_graph` / `replay_park`),
    `state/nodes.py`; MemorySaver + disk snapshot `out/strategy/`
  - **Acceptance**: state survives a process restart; `replay_park`
    restores the parked thread from disk and resume completes

- [x] **strategylib — native, self-extending component library**
  - **Implementation**: executable Python components (not JSON metadata):
    `strategylib/components/{entry*,exit*,sizing*}`, index-only manifest
    `manifest.json`, pure registry (`registry.py`: resolve / by_pillar /
    matching / append_manifest). Curated components: `entry.ema_cross`,
    `entry.vwap_band`, `exit.atr_stop`, `exit.time_exit`,
    `sizing.fractional`, `sizing.atr_scaled`
  - **Synthesis**: `strategylib/synthesize.py` compiles a spec `signal_ast`
    subtree into a real module under `components/generated/`, runs a
    hermetic self-check (import + shape + finiteness), and only a human
    `approve` on the `missingPrimitive` card registers it
    (`append_manifest`) — mandatory review before generated code enters the
    trusted library

- [x] **Node: Hydrator** (A6 spec → formulations)
  - **Implementation**: `block_b/formulation.py::hydrate_spec` — resolves
    entry/exit candidates from the registry by trigger `implements`,
    designs the sweep grid from manifest param schemas (A6 `parameters`
    carry intent/invariants, not bounds), caps variants at
    `MAX_VARIANTS = 4`, flags missing pillars / unbacked primitives /
    bad_data as structural errors

- [x] **Node: Structural Gate** (deterministic well-posedness, re-run on
  every re-audit so an approved registration/human fix clears naturally)

- [x] **Node: Cynical Auditor** (adversarial well-posedness — v1 rule-based)
  - **Implementation**: `block_b/auditor.py` — lookahead/lead references in
    the signal AST (negative lag/returns literals), missing exit, no
    sizing, unbounded sweep params, degenerate EMA windows
  - **Acceptance**: all seeded lookahead biases in test documents flagged;
    missing exit rules flagged; `AuditFlag` contract unchanged so the
    **LLM-backed adversarial pass is a documented node-internal swap** (not
    built — ponytail: rule → LLM upgrade path)

- [x] **Node: HITL gate (LangGraph `interrupt()`)** — parks with a typed
  card batch; `Command(resume={"answers": ...})` resumes

- [x] **Node: Apply Answers** (deterministic keyed mutations)
  - **Implementation**: `block_b/hitl.py::apply_answers` — cards are typed
    (missingPillar / missingPrimitive / remediation / universe), each with
    a keyed `mutation` path (e.g. `variants.0.sizing`); `accept` re-derives
    the full pillar payload (component + params + sweep grid) from the
    registry; idempotent via `answers_applied` + `resume_key` (a duplicate
    batch is a no-op and does not burn a round)

### Block B API Endpoints ✅ (frozen)
- [x] `POST /api/v1/strategy/submit` — `{spec_id}` (curated A6 spec from
  `out/block_a_specs.json`) or inline `{spec}` → new session, runs DAG to
  first interrupt/terminal
- [x] `GET /api/v1/strategy/{session_id}/state` — current state (live
  checkpoint, or disk snapshot + re-park after restart)
- [x] `GET /api/v1/strategy/{session_id}/stream` — SSE events
  (`formulation` → `hitl_request` | `done`; synchronous snapshot stream
  for v1, no sse-starlette dep)
- [x] `POST /api/v1/strategy/{session_id}/resume` — `{answers: {card_id:
  {action, value}}}`; idempotent per answers batch (identical batch →
  graceful no-op)

**Acceptance**: mechanical spec → `hitl` with missingPillar cards →
`accept` resume → `complete` with swept variants; lookahead spec →
`rejected` after the 2-round cap; restart replay resumes a parked session
(covered by `tests/test_block_b.py` + `tests/strategylib/`).

---

## Step 2: Angular HITL Gate & Streaming Layer ⏳ (frontend paused)

### Technical Architecture
- **Real-time transport**: FastAPI SSE (`GET /strategy/{id}/stream`)
  shipped; **WebSocket deferred** — the `resume` REST contract carries HITL
  responses (API contract frozen, WS not required by it)
- **LangGraph interrupt**: `interrupt()` gate ships (Step 1)
- **Angular components**: `StrategyBuilderComponent`, `HITLDialogComponent`,
  `StateStreamComponent` — paused on Stitch/Figma handoff

### Node Implementations

- [x] **Node: HITL Payload Generator** (batches structural errors + audit
  flags into typed ClarificationCards, ≤ 20 per batch)
- [x] **Node: LangGraph `interrupt()` gate** — state saved at interrupt;
  `POST /resume` resumes from checkpoint (idempotent)
- [x] **SSE endpoint** `GET /api/v1/strategy/{session_id}/stream` — emits
  `StateUpdate` events as SSE
- [ ] **WebSocket endpoint** `WS /api/v1/strategy/{session_id}/ws` —
  deferred (not required by the frozen REST contract; revisit with live
  per-node streaming when async graph runs land)

### Angular Frontend Components (paused on Stitch/Figma handoff)

- [ ] **`StrategyBuilderComponent`** — main workflow display, shows current
  state, variants, cards
- [ ] **`HITLDialogComponent`** — typed cards + keyed answers, submits via
  `POST /resume`
- [ ] **`StateStreamComponent`** — SSE consumer, displays real-time state
  updates, round counter
- [ ] **`StrategyExportComponent`** — final export bundle display (Block D)

**Acceptance Criteria** (post-handoff): user submits a curated spec → sees
variants → answers typed HITL cards → state reaches complete/rejected.
WebSocket latency requirement moot until WS ships.

**Dependencies**: Block A Step 6 (`block_a_specs.json`) is the submit
input; frontend depends on the frozen `StrategyState` contract.

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
