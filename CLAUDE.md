# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# Install dependencies
pip3 install -r requirements.txt

# Configure API keys (copy template, then edit .env)
cp .env.example .env

# Run the web app (port 5001)
python3 app.py
# Open http://localhost:5001
```

No build step for the frontend.

## Linting & Testing

```bash
# Lint (ruff)
python3 -m ruff check .          # check for issues
python3 -m ruff check --fix .    # auto-fix safe issues
python3 -m ruff format .         # format all Python files

# Tests (pytest, 20 smoke tests)
python3 -m pytest tests/ -v      # run all tests
python3 -m pytest tests/test_smoke.py::test_config -v  # run single test
```

Config lives in `pyproject.toml`. Legacy files (`lead_finder.py`, `config.py`) are excluded from linting.

**Legacy CLI mode** (lead_finder.py via run.py):
```bash
python run.py                                    # default brief
python run.py briefs/client.yaml --test          # 5 leads only
python run.py briefs/client.yaml --no-enrich     # skip enrichment
```

## Architecture

### Backend (modular)

```
app.py              (376 lines) — thin Flask route handlers, API key loading
utils.py            — log_dev (thread-safe), dev_logs, extract_domain, clean_domain, is_business_domain, SKIP_DOMAINS
models.py           — Lead dataclass (30+ fields)
scoring.py          — LeadScorer class (0-100 fit_score)
jobs.py             — JobManager class (thread-safe with Lock, TTL cleanup)

api_clients/
  base.py           — BaseAPIClient with _get(), _post(), is_configured, _log()
  google_places.py  — GooglePlacesClient (query-param auth override)
  apollo.py         — ApolloClient (search_contacts, search_organizations, enrich_organization)
  firecrawl.py      — FirecrawlClient (scrape, search)
  hunter.py         — HunterClient (domain_search, email_finder, email_verifier, email_count)
  scraper.py        — WebsiteScraper (requests-based, extracts social links + emails)
  __init__.py       — re-exports all client classes

services/
  search.py         — run_search() pipeline: _step_search → _step_details → _step_enrich → _step_score
  discovery.py      — discover_prospects(), run_web_search(), run_apollo_search(), parse_maps_markdown()
  enrichment.py     — enrich_company(), enrich_batch(), enrich_contacts(), extract_company_data()
  brief.py          — generate_brief() via Firecrawl + OpenAI
  export.py         — export_leads_csv() → BytesIO
  __init__.py
```

### Frontend: templates/index.html (~4500 lines)

Single-page vanilla JS app with tabbed UI: **Prospecting** (unified search with auto-routing), **Enrichment** (domain enrichment), **Brief** (AI GTM generation). Dark theme, no framework, no build step.

## Enrichment Pipeline

```
Google Places search → Place details → Firecrawl scrape → Apollo org+contacts → Hunter emails → Score → CSV export
```

Background jobs use `threading.Thread` (daemon=True) with polling (`/api/status/<job_id>` at 1s intervals). Job state managed by `JobManager` in `jobs.py` (thread-safe, auto-expires completed jobs after 1 hour).

## Key Endpoints

| Route | Purpose |
|---|---|
| `/api/config` | Which APIs are configured (bool per key) |
| `/api/search` | Start background search job |
| `/api/status/<job_id>` | Poll job progress |
| `/api/cancel/<job_id>` | Cancel running job |
| `/api/export/<job_id>` | Download CSV |
| `/api/brief` | AI brief generation (OpenAI + Firecrawl) |
| `/api/discover` | Smart search / Apollo search / Maps URL scraping |
| `/api/enrich` | Single domain enrichment |
| `/api/enrich-batch` | Batch domain enrichment |
| `/api/enrich-contacts` | Find decision makers |
| `/api/hunter/domain-search` | Hunter domain search |
| `/api/hunter/email-finder` | Hunter email finder |
| `/api/hunter/verify` | Hunter email verification |
| `/api/dev/logs` | Dev panel log stream |
| `/api/dev/stats` | Enrichment statistics |

## API Client Pattern

All clients inherit from `BaseAPIClient` (`api_clients/base.py`). Return `{}` on failure, log via `log_dev()`, guard with `if not self.api_key: return {}`.

```python
from api_clients import BaseAPIClient

class NewClient(BaseAPIClient):
    BASE_URL = "https://api.example.com/v1"

    def __init__(self, api_key: str):
        super().__init__(api_key, log_fn=log_dev, category="NEW_API")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def search(self, query: str) -> dict:
        return self._get("/search", params={"q": query})
```

Note: GooglePlacesClient and HunterClient override `_get()` to pass API key as a query parameter instead of header.

## Service Layer Pattern

Route handlers in `app.py` are thin — they parse request data, call service functions, and return jsonify'd results. Business logic lives in `services/`.

```python
# app.py route
@app.route("/api/search", methods=["POST"])
def start_search():
    config = request.json
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    thread = threading.Thread(
        target=search_service.run_search,
        args=(job_id, config, _api_keys(), job_mgr),
        daemon=True
    )
    thread.start()
    return jsonify({"job_id": job_id})
```

## Dev Logging

```python
from utils import log_dev
log_dev("CATEGORY", "message", "info|success|warning|error")
```
Thread-safe (uses `threading.Lock`). Categories: SEARCH, GOOGLE, APOLLO, FIRECRAWL, HUNTER, ENRICH, CONFIG, DISCOVER, BRIEF. Rolling buffer of 500 entries, visible in the UI dev panel.

## Important Conventions

- **Graceful degradation** — app works with only `GOOGLE_PLACES_API_KEY`. All other APIs are optional and guarded.
- **Domain filtering** — `extract_domain()` normalizes URLs; `is_business_domain()` + `SKIP_DOMAINS` set filters out social/platform domains before enrichment.
- **Scoring** — `LeadScorer` produces 0-100 `fit_score` based on rating, reviews, website presence, emails found, decision makers, and signal keywords. Excluded brands get score 0.
- **No auth on endpoints** — designed for local/internal use. CORS open to all origins.
- **API keys** — loaded from `.env` (gitignored). `.env.example` has placeholders. Keys: `GOOGLE_PLACES_API_KEY` (required), `APOLLO_API_KEY`, `HUNTER_API_KEY`, `FIRECRAWL_API_KEY`, `OPENAI_API_KEY`, `INSTANTLY_API_KEY` (all optional).
- **XSS prevention** — all dynamic values in `displayFullBrief()` and `refreshDevLogs()` are wrapped with `escapeHtml()`.
- **Input debouncing** — strategy indicator and search summary updates use `debounce()` (200ms) to avoid excessive DOM updates.
