# GTM LeadFlow

**Find the right companies and people. Enrich them. Export and go.**

LeadFlow is a local B2B prospecting tool that helps you build qualified lead lists fast — without the price tag of enterprise sales tools. Search by location and industry, discover decision makers, score every lead automatically, and export a clean CSV ready for your outreach campaigns.

---

## What It Does

### Company Search
Search for businesses by location, industry, or intent. LeadFlow finds matching companies, scrapes their websites for contact info and social links, identifies decision makers, and scores each lead based on how well they match your ideal customer profile.

### People Search
Search Apollo's people database by job title, seniority, department, and location. Get a scored list of contacts with their LinkedIn profiles, company info, and fit scores — ready to work.

### Lead Scoring
Every result gets a **fit score (0–100)** based on real signals: ratings, review volume, website quality, email availability, decision maker presence, and keyword relevance. No more sorting through noise manually.

### Decision Maker Discovery
For each company, LeadFlow finds the right people — CEOs, Marketing Directors, Owners — with their emails, LinkedIn profiles, and titles. Enrichment data flows directly into the CSV export.

### GTM Brief Generator
Paste a company URL and get an AI-generated Go-To-Market brief: positioning, target audience, competitive landscape, and outreach hooks. Powered by web scraping and AI analysis.

### CSV Export
Two export formats built for real workflows:
- **Prospect CSV** — one row per company, with social links, scores, and contact info
- **Enriched CSV** — one row per decision maker, ready to paste into any email sequencer

---

## Setup

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Configure your keys
cp .env.example .env
# Edit .env — only Google Places is required to start

# 3. Launch
python3 app.py
# Open http://localhost:5001
```

LeadFlow runs entirely on your machine. No SaaS subscription, no usage dashboard, no data leaving your environment except through your own API keys.

---

## Data Sources

LeadFlow connects to best-in-class data providers — but you're always in control of which ones you use. Every integration is optional except Google Places:

| Source | What it adds |
|---|---|
| **Google Places** | Location-based company discovery (required) |
| **Apollo.io** | Company size, revenue, LinkedIn, contact search |
| **Hunter.io** | Email finding and verification |
| **Firecrawl** | Deep website scraping for emails, social links, and about text |
| **OpenAI** | GTM brief generation |

---

## Scoring Logic

Leads are scored from 0 to 100 based on:
- Google rating and review count
- Website presence and quality
- Email and social link availability
- Decision maker count
- Signal keywords in name/category
- Specific exclusions (chains, platforms, aggregators)

High-fit leads (≥70) are the ones worth prioritizing for outreach.

