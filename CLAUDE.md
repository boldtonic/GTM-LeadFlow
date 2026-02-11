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

There are no tests. No linter is configured. No build step for the frontend.

**Legacy CLI mode** (lead_finder.py via run.py):
```bash
python run.py                                    # default brief
python run.py briefs/client.yaml --test          # 5 leads only
python run.py briefs/client.yaml --no-enrich     # skip enrichment
```

## Architecture

### Backend: app.py (monolithic, ~2300 lines)

Single Flask file organized in sections:

1. **Config & globals** — API key loading from `.env`, `job_status` dict, `dev_logs` list, `log_dev()` function
2. **Lead dataclass** — 30+ fields covering identity, location, contact, reputation, enrichment, scoring
3. **API clients** — `GooglePlacesClient`, `ApolloClient`, `FirecrawlClient`, `HunterClient` (all inline)
4. **Utilities** — `WebsiteScraper`, `LeadScorer`, `extract_domain()`, `is_business_domain()`
5. **Background jobs** — `run_search()` orchestrates the full pipeline via `threading.Thread`
6. **Flask routes** — all endpoints

### Frontend: templates/index.html (~3700 lines)

Single-page vanilla JS app with 4 tabs: **Brief** (AI GTM generation), **Discovery** (prospect search), **Enrichment** (domain enrichment), **Finder** (legacy Places search). Dark theme, no framework, no build step.

### api_clients/ module (in progress)

`BaseAPIClient` in `api_clients/base.py` — reusable base with `_get()`, `_post()`, `is_configured`, `_log()`. New API integrations should inherit from this instead of being added inline to app.py.

## Enrichment Pipeline

```
Google Places search → Place details → Firecrawl scrape → Apollo org+contacts → Hunter emails → Score → CSV export
```

Background jobs use threading with polling (`/api/status/<job_id>` at 1s intervals). Job state lives in the `job_status` global dict.

## Key Endpoints

| Route | Purpose |
|---|---|
| `/api/config` | Which APIs are configured (bool per key) |
| `/api/search` | Start background search job |
| `/api/status/<job_id>` | Poll job progress |
| `/api/export/<job_id>` | Download CSV |
| `/api/brief` | AI brief generation (OpenAI + Firecrawl) |
| `/api/discover` | Smart search / Apollo search / Maps URL scraping |
| `/api/enrich` | Single/batch domain enrichment |
| `/api/hunter/*` | Hunter.io email operations |
| `/api/dev/logs` | Dev panel log stream |

## API Client Pattern

Existing clients in app.py all follow this pattern — return `{}` on failure, log via `log_dev(CATEGORY, msg, level)`, guard with `if not self.api_key: return {}`.

New clients should use BaseAPIClient:
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

## Dev Logging

```python
log_dev("CATEGORY", "message", "info|success|warning|error")
```
Categories: SEARCH, GOOGLE, APOLLO, FIRECRAWL, HUNTER, ENRICH, CONFIG. Rolling buffer of 500 entries, visible in the UI dev panel.

## Important Conventions

- **Graceful degradation** — app works with only `GOOGLE_PLACES_API_KEY`. All other APIs are optional and guarded.
- **Domain filtering** — `extract_domain()` normalizes URLs; `is_business_domain()` + `SKIP_DOMAINS` set filters out social/platform domains before enrichment.
- **Scoring** — `LeadScorer` produces 0-100 `fit_score` based on rating, reviews, website presence, emails found, decision makers, and signal keywords. Excluded brands get score 0.
- **No auth on endpoints** — designed for local/internal use. CORS open to all origins.
- **API keys** — loaded from `.env` (gitignored). `.env.example` has placeholders. `config.py` is legacy and not used at runtime.
