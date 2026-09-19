# Black_Box Backend

FastAPI + Docling 2.x document engineering backend for the Black_Box quant
strategy platform (see `../plan.md` for the full roadmap, blocks A–D).

## Quickstart

```bash
# 1. Create local env config (values already match defaults)
cp .env.example .env

# 2. Boot
uv run uvicorn black_box.main:app --reload --port 8000
# or: .venv/bin/uvicorn black_box.main:app
```

## Smoke test

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/api/v1/documents/ingest \
  -F "file=@tests/fixtures/strategy_smoke.md;type=text/markdown"
```

Returns structure-aware chunks (`id`, `text`, `page`, `heading`, `tokens`,
`meta`) produced by Docling's `HybridChunker`.

## Endpoints

| Method | Path                       | Description                        |
| ------ | -------------------------- | ---------------------------------- |
| GET    | `/health`                  | Liveness probe (docling readiness) |
| POST   | `/api/v1/documents/ingest` | Parse PDF/DOCX/HTML/MD → chunks    |

## Configuration (`Settings`, env or `.env`)

- `ALLOWED_FORMATS` — comma-separated Docling `InputFormat` allowlist (default `pdf,docx,html,md`)
- `MAX_UPLOAD_SIZE` — bytes (default 25 MB), enforced pre-parse
- `MAX_NUM_PAGES` — pages per document (default 500)
- `API_PREFIX` — API version prefix (default `/api/v1`)

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Integration tests convert a real Markdown strategy document with the actual
Docling converter and assert chunk structure + error paths.

## Layout

```
src/black_box/
  main.py                 # FastAPI app + lifespan (owns DoclingService)
  schemas.py              # API response contracts (ChunkOut, IngestResponse)
  core/config.py          # Settings (pydantic-settings, env/.env)
  api/routes.py           # thin route handlers
  services/docling_service.py  # Docling converter + HybridChunker singleton
```