# Black_Box Backend

FastAPI + Docling 2.x document engineering backend for the Black_Box quant
strategy platform (see `../plan.md` for the full roadmap, blocks A–D).

## Current Status

**Block A** (Ingestion & Feasibility Auditor) ✅ + **Block B scaffold** (LangGraph DAG, state schema, strategy API) ✅ + **Block A Alpha engine** (Stages A1–A6: LLM extraction → catalog check → 8-operator sweep → DSL spec compiler → human gatekeeper → validated `ExecutableStrategySpec` JSON) ✅

## Quickstart

```bash
# 1. Create local env config
cp .env.example .env

# 2. Install deps + sync lockfile
uv sync

# 3. Boot
uv run uvicorn black_box.main:app --reload --port 8000
# or: .venv/bin/uvicorn black_box.main:app

# 4. Run the Block A Alpha engine on a research paper (PDF/MD/HTML)
uv run python -m black_box.block_a.cli tests/fixtures/block_a/vwap_trend.md --yes
```

The CLI drives the full A1–A6 pipeline: Docling parse → schema-first LLM
extraction (Instructor, local Ollama) → DuckDB/polars data-catalog check
(Stage A3 hard stop) → 8 quant operators (annotations only, no deletion
authority) → DSL spec compiler → rich-text human gatekeeper review → validated
spec matrix written to `out/block_a_specs.json`.

```bash
# Deterministic dry-run without any LLM (CI-safe)
uv run python -m black_box.block_a.cli tests/fixtures/block_a/vwap_trend.md --llm fake --yes

# Interactive gatekeeper review (approve/reject per spec)
uv run python -m black_box.block_a.cli paper.pdf
```

Exit codes: `0` complete · `1` unexpected failure · `2` RESOURCE_INSUFFICIENT (hard Stage A3 data stop) · `3` unparseable paper · `4` LLM_EXTRACTION_FAILED (a stage exhausted its validation retries).

## Smoke test

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/api/v1/documents/ingest \
  -F "file=@tests/fixtures/strategy_smoke.md;type=text/markdown"
curl -s -X POST localhost:8000/api/v1/strategy/submit
curl -s localhost:8000/api/v1/feasibility/capabilities
```

## Endpoints

| Method | Path                       | Description                        |
| ------ | -------------------------- | ---------------------------------- |
| GET    | `/health`                  | Liveness probe (docling readiness) |
| POST   | `/api/v1/documents/ingest` | Parse PDF/DOCX/HTML/MD → chunks    |
| POST   | `/api/v1/strategy/submit`  | Create strategy session + kick off DAG |
| GET    | `/api/v1/strategy/{id}/state` | Retrieve session state          |
| POST   | `/api/v1/strategy/{id}/resume` | Inject HITL responses + resume |
| GET    | `/api/v1/feasibility/capabilities` | Platform capability contract |
| POST   | `/api/v1/feasibility/audit` | Run feasibility audit          |

## Configuration (`Settings`, env or `.env`)

- `ALLOWED_FORMATS` — comma-separated Docling `InputFormat` allowlist (default `pdf,docx,html,md`)
- `MAX_UPLOAD_SIZE` — bytes (default 25 MB), enforced pre-parse
- `MAX_NUM_PAGES` — pages per document (default 500)
- `API_PREFIX` — API version prefix (default `/api/v1`)
- `DATABASE_URL` — PostgreSQL connection string for Block B+ checkpointing (default `postgresql+asyncpg://postgres:postgres@localhost:5432/black_box`)

### Block A — Alpha Feasibility & Strategy Ingestion Engine

- `BLOCK_A_LLM_BACKEND` — `ollama` (default) | `fake` (deterministic tests)
- `BLOCK_A_OLLAMA_MODEL` — local model (default `llama3.2`)
- `BLOCK_A_OLLAMA_BASE_URL` — OpenAI-compatible endpoint (default `http://localhost:11434/v1`)
- `BLOCK_A_MAX_RETRIES` — Instructor Pydantic retry cap, max 3 per spec
- `BLOCK_A_CONFIG_DIR` — `primitives_registry.json` + `data_catalog.parquet` (defaults to packaged seed catalog)
- `BLOCK_A_OUT_DIR` — output dir for `block_a_specs.json` (default `out`)
- `BLOCK_A_TRACE_ENABLED` — JSONL prompt/response traces per paper, off by default (set `true` to debug)
- `BLOCK_A_TRACE_DIR` — trace dir for runs without an explicit `--out` (default `out/traces`)

The seed catalog (`src/black_box/block_a/config/data_catalog.parquet`) covers
BTC/ETH/SOL L2+orderflow and ADA/DOGE OHLCV rows across tick/1m/1h/1d from
2020–2021. The feasibility auditor's granularity authority was rewired to this
catalog (with static-capability fallback when the parquet is absent).

## Tests

```bash
uv run pytest tests/ -v
```

Integration tests convert a real Markdown strategy document with the actual
Docling converter and assert chunk structure + error paths. Block B tests
validate the LangGraph DAG topology and strategy API endpoints.

## Layout

```
src/black_box/
  main.py                 # FastAPI app + lifespan (owns DoclingService)
  schemas.py              # API response contracts (ChunkOut, IngestResponse, etc.)
  core/config.py          # Settings (pydantic-settings, env/.env)
  core/capabilities.py    # Platform capability contract
  api/routes.py           # Block A routes (ingest, feasibility)
  api/strategy_routes.py  # Block B routes (strategy submit/state/resume)
  services/docling_service.py  # Docling converter + HybridChunker
  services/math_resolver.py    # Symbol/LaTeX resolution
  services/table_validator.py  # Table schema validation
  services/feasibility_auditor.py # Feasibility gate (4 checkers)
  state/schema.py         # StrategyState Pydantic model (Block B)
  state/nodes.py          # DAG node implementations (Block B)
  state/graph.py          # LangGraph StateGraph topology (Block B)
  state/__init__.py       # State module exports
tests/                    # pytest suite (44 tests)
```

## Roadmap

- **Block B**: LangGraph DAG with PostgreSQL checkpointing, real PydanticAI
  extractor nodes, WebSocket/SSE streaming, HITL gate
- **Block C**: Statistical stress validation (vectorbt, CPCV, Monte Carlo)
- **Block D**: Execution transpiler (Jinja2 → .mq5/.py), guardrail injection
