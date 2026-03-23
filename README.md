# GTM BYOK LeadFlow v0.1.3

**Find the right actionable prospects: companies and people.

LeadFlow is a self-hosted BYOK B2B prospecting tool that helps you build qualified lead lists fast — without the price tag of enterprise sales tools, just your API cost. Search by location and industry, and several other filters, including signals; Discover decision makers, score every lead automatically, and export a clean CSV ready for your outreach campaigns.

 [Enrich them in Kanban to Messaging Soon]
---

## Features

### Company Search
Search for businesses by location, industry, or intent. LeadFlow finds matching companies, scrapes their websites for contact info and social links, identifies decision makers, and scores each lead based on how well they match your ideal customer profile.

### People Search
Search for professionals by job title, seniority, department, and location. Get a scored list of contacts with their LinkedIn profiles, company info, and fit scores.

### Lead Scoring
Every result gets a **fit score (0–100)** based on real signals: ratings, review volume, website quality, email availability, decision maker presence, and keyword relevance.

### Decision Maker Discovery
For each company, LeadFlow finds the right people — CEOs, Marketing Directors, Owners — with their emails, LinkedIn profiles, and titles. Enrichment data flows directly into the CSV export.

### GTM Brief Generator
Paste a company URL and get an AI-generated Go-To-Market brief: positioning, target audience, competitive landscape, and outreach hooks.

### CSV Export
Two export formats built for real workflows:
- **Prospect CSV** — one row per company, with social links, scores, and contact info
- **Enriched CSV** — one row per decision maker, ready to import into any email sequencer

---

## Getting Started

### Prerequisites

- Python 3.9+
- A [Google Places API key](https://developers.google.com/maps/documentation/places/web-service/get-api-key) (required)
- Additional API keys are optional — see [Data Sources](#data-sources)

### Installation

```bash
git clone https://github.com/boldtonic/GTM-LeadFlow.git
cd GTM-LeadFlow
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your API keys, then run:

```bash
python app.py
```

Open `http://localhost:5001` in your browser.

---

## Data Sources

LeadFlow connects to best-in-class data providers. Only Google Places is required — all others enhance results when configured.

| Provider | What it adds |
|---|---|
| **Google Places** | Location-based company discovery (required) |
| **Apollo.io** | Company size, revenue, LinkedIn, contact search |
| **Hunter.io** | Email finding and verification |
| **Firecrawl** | Deep website scraping for emails, social links, and company descriptions |
| **OpenAI** | GTM brief generation |

LeadFlow runs entirely on your machine. No data leaves your environment except through your own API keys.

---

## Scoring

Leads are scored from 0 to 100 based on:
- Google rating and review volume
- Website and social presence
- Email and contact availability
- Decision maker count
- Keyword relevance to your search

High-fit leads (≥ 70) are surfaced at the top of every results list.

---

## License

MIT — see [LICENSE](LICENSE) for details.
