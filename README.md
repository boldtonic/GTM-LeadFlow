# Fogata 🔥

**Self-hosted BYOK prospecting: find companies, discover decision makers, research them with AI — all on your machine.**

![Fogata](assets/coolvid.gif)

[![Version](https://img.shields.io/badge/version-0.1.3-brightgreen?style=flat-square)](https://github.com/boldtonic/fogata/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-gold?style=flat-square)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Self-hosted](https://img.shields.io/badge/self--hosted-yes-8b5cf6?style=flat-square)]()
[![BYOK](https://img.shields.io/badge/BYOK-bring%20your%20own%20keys-f59e0b?style=flat-square)]()

Fogata is a self-hosted B2B prospecting tool built on your own API keys. Search for companies and people, score every lead automatically, enrich them through a Kanban pipeline, get per-person AI intelligence briefs, and export a clean CSV — without paying for an enterprise sales platform.

No SaaS pricing. No data leaving your machine except through your own keys.

---

## What you get

- **Company search** — find businesses by location, industry, and keyword signals
- **People search** — discover professionals by title, seniority, department, and location
- **Lead scoring** — automatic 0–100 fit score based on real signals (ratings, contacts, website, keywords)
- **Decision maker discovery** — CEO, Marketing Director, Owner — with emails and LinkedIn
- **Enrichment Kanban** — drag leads from *Imported → Contacts Found → Deep Researched*
- **AI deep research** — per-person intelligence brief: summary, pain points, outreach angle, and icebreakers (OpenAI + Firecrawl)
- **GTM Brief generator** — paste a URL and get a full ICP + go-to-market analysis
- **CSV export** — two formats: one row per company, or one row per decision maker ready for sequencers

---

## Installation

```bash
git clone https://github.com/boldtonic/fogata.git
cd fogata
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your API keys, then:

```bash
python app.py
```

Open `http://localhost:5001`.

---

## Quick start

**1. Find companies in a city:**
Search for *"marketing agencies in Warsaw"* → get a scored list of companies with website, social, and contact signals.

**2. Discover contacts at a company:**
Drop a company into the Kanban → drag it to *Contacts Found* → Fogata finds decision makers via Apollo with emails and LinkedIn.

**3. Research a person with AI:**
Click *Deep Research* on a person card → get a structured brief: who they are, their pain points, the best outreach angle, and two icebreakers. Powered by OpenAI + Firecrawl.

**4. Export to your sequencer:**
Export enriched leads as CSV (one row per person) and import directly into Instantly, Apollo, or any email tool.

![Fogata](assets/fogata_lkd.png)
---

## Data sources

Only Google Places is required — all others enhance results when configured.

| Provider | What it adds | Required |
|---|---|---|
| **Google Places** | Location-based company discovery | ✅ Yes |
| **Apollo.io** | Company size, revenue, LinkedIn, contact search | Optional |
| **Hunter.io** | Email finding and verification | Optional |
| **Firecrawl** | Website + LinkedIn scraping for deep research | Optional |
| **OpenAI** | GTM brief and AI intelligence briefs | Optional |
| **Instantly** | Direct outreach campaign push | Optional |

Fogata runs entirely on your machine. No data leaves your environment except through your own keys.

---

## Lead scoring

Every result is scored 0–100 based on:

- Google rating and review volume
- Website and social presence
- Email and contact availability
- Decision maker count
- Keyword relevance to your search

Leads scoring ≥ 70 are surfaced at the top of every list.

---

## License

MIT — see [LICENSE](LICENSE) for details.
