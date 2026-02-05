# GTM LeadFlow

B2B lead discovery and enrichment tool for Go-To-Market campaigns. Find retail partners, enrich company data, discover decision makers, and export actionable lead lists.

## Tech Stack

- **Backend:** Python 3.x + Flask (port 5001)
- **Frontend:** Single-page HTML/CSS/JS (vanilla, dark theme)
- **Data:** YAML briefs for client configs, CSV export

## API Integrations

| API | Purpose | Required |
|-----|---------|----------|
| Google Places | Lead discovery (location search, place details) | Yes |
| Apollo.io | Company enrichment (size, revenue, LinkedIn) + contact search | Optional |
| Hunter.io | Email finding, verification, domain search | Optional |
| Firecrawl | Web scraping (markdown extraction, email/social discovery) | Optional |
| OpenAI | AI-powered GTM brief generation | Optional |

## Project Structure

```
GTM-LEADFLOW/
├── app.py                  # Main Flask app (~2200 lines)
│                           #   - API client classes (Google, Apollo, Hunter, Firecrawl)
│                           #   - WebsiteScraper (Firecrawl + basic fallback)
│                           #   - LeadScorer (fit scoring with signals)
│                           #   - Lead dataclass (30+ fields)
│                           #   - All Flask routes / endpoints
│                           #   - Background job processing (threading)
│                           #   - Dev logging system
├── api_clients/            # New modular API client module
│   ├── __init__.py
│   └── base.py             # BaseAPIClient (reusable GET/POST, auth, logging)
├── config.py               # API keys config + rate limits
├── lead_finder.py          # Original CLI tool
├── run.py                  # CLI batch runner
├── brief_template.yaml     # YAML template for client briefs
├── briefs/                 # Client brief files
├── templates/
│   └── index.html          # Web UI (single-page, ~156KB)
├── requirements.txt        # Python deps
├── .env.example            # Environment variable template
└── START.command           # macOS quick launch
```

## Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web UI |
| `/api/config` | GET | Check configured APIs |
| `/api/search` | POST | Start background search job |
| `/api/status/<job_id>` | GET | Job progress & results |
| `/api/cancel/<job_id>` | POST | Cancel running job |
| `/api/export/<job_id>` | GET | Export results as CSV |
| `/api/brief` | POST | Generate GTM brief (OpenAI) |
| `/api/discover` | POST | Discover leads from web |
| `/api/enrich` | POST | Enrich single company by domain |
| `/api/enrich-batch` | POST | Enrich multiple companies |
| `/api/enrich-contacts` | POST | Find decision makers |
| `/api/hunter/*` | POST | Hunter.io direct endpoints |
| `/api/dev/logs` | GET | Debug logs |
| `/api/dev/stats` | GET | Enrichment statistics |

## Enrichment Pipeline

```
Search (Google Places)
  -> Scrape website (Firecrawl / basic fallback)
    -> Extract emails, social links, description
  -> Enrich company (Apollo: size, revenue, LinkedIn)
  -> Find contacts (Apollo: decision makers by title)
  -> Find emails (Hunter: domain search, verification)
  -> Score lead (fit score 0-100 based on signals)
  -> Export CSV
```

## Setup

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your keys

# 3. Run
python3 app.py
# Open http://localhost:5001
```

## Architecture Notes

- **Monolithic app.py** contains all API clients, routes, and logic inline. The `api_clients/` module is the start of modularizing this.
- **BaseAPIClient** (`api_clients/base.py`) provides reusable `_get()` / `_post()` with auth, logging, error handling, and timeouts. New API integrations should inherit from this.
- **Background jobs** run in threads, tracked by `job_status` dict. Real-time progress via polling `/api/status/<job_id>`.
- **Dev panel** built-in logging (`log_dev()`) with categories per API. Last 500 entries.
- **Graceful degradation** — app works with only Google Places key; other APIs enhance results when configured.
- **Lead dataclass** has 30+ fields covering: identity, location (7 fields), contact, reputation, operations, web scraping results, Apollo enrichment, scoring, and metadata.
