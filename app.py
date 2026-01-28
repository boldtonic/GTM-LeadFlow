"""
GTM Lead Finder - Web Application
Flask backend for the lead finding tool
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
import time
import re
import csv
import io
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import requests
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION - Load API keys from environment variables
# ============================================================
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY", "")
HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

# Global state for progress tracking
job_status = {}


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Lead:
    """Structured lead data"""
    place_id: str
    google_maps_url: str
    name: str
    category: str
    subcategories: list = field(default_factory=list)
    address_full: str = ""
    street: str = ""
    city: str = ""
    country: str = ""
    postal_code: str = ""
    lat: float = 0.0
    lng: float = 0.0
    phone: str = ""
    website: str = ""
    rating: float = 0.0
    reviews_count: int = 0
    hours: dict = field(default_factory=dict)
    price_level: int = 0
    emails_found: list = field(default_factory=list)
    social_links: dict = field(default_factory=dict)
    brands_carried: list = field(default_factory=list)
    about_text: str = ""
    apollo_org_id: str = ""
    company_size: str = ""
    estimated_revenue: str = ""
    linkedin_url: str = ""
    decision_makers: list = field(default_factory=list)
    fit_score: int = 0
    fit_reasons: list = field(default_factory=list)
    scraped_at: str = ""
    enriched_at: str = ""
    source_query: str = ""


# ============================================================
# API CLIENTS
# ============================================================

class GooglePlacesClient:
    BASE_URL = "https://maps.googleapis.com/maps/api/place"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def text_search(self, query: str) -> list[dict]:
        if not self.api_key:
            return []

        url = f"{self.BASE_URL}/textsearch/json"
        params = {"query": query, "key": self.api_key}
        results = []
        next_page_token = None

        while True:
            if next_page_token:
                params["pagetoken"] = next_page_token
                time.sleep(2)

            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()

                if data.get("status") not in ["OK", "ZERO_RESULTS"]:
                    break

                results.extend(data.get("results", []))
                next_page_token = data.get("next_page_token")

                if not next_page_token or len(results) >= 60:
                    break
            except Exception as e:
                print(f"Search error: {e}")
                break

        return results

    def get_place_details(self, place_id: str) -> dict:
        if not self.api_key:
            return {}

        url = f"{self.BASE_URL}/details/json"
        params = {
            "place_id": place_id,
            "key": self.api_key,
            "fields": "place_id,name,formatted_address,formatted_phone_number,international_phone_number,website,url,rating,user_ratings_total,price_level,opening_hours,types,address_components,geometry"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if data.get("status") == "OK":
                return data.get("result", {})
        except:
            pass
        return {}


class ApolloClient:
    BASE_URL = "https://api.apollo.io/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}

    def enrich_organization(self, domain: str) -> dict:
        if not self.api_key:
            return {}

        url = f"{self.BASE_URL}/organizations/enrich"
        params = {"api_key": self.api_key, "domain": domain}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            data = response.json()
            return data.get("organization", {})
        except:
            return {}

    def search_contacts(self, domain: str, titles: list = None) -> list[dict]:
        if not self.api_key:
            return []

        url = f"{self.BASE_URL}/mixed_people/search"
        data = {
            "api_key": self.api_key,
            "q_organization_domains": domain,
            "page": 1,
            "per_page": 5,
        }
        if titles:
            data["person_titles"] = titles

        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=10)
            result = response.json()
            return result.get("people", [])
        except:
            return []


class FirecrawlClient:
    """Firecrawl API client for powerful web scraping"""
    BASE_URL = "https://api.firecrawl.dev/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def scrape(self, url: str) -> dict:
        """Scrape a single URL and return markdown + metadata"""
        if not self.api_key:
            return {}

        try:
            response = requests.post(
                f"{self.BASE_URL}/scrape",
                headers=self.headers,
                json={
                    "url": url,
                    "formats": ["markdown", "links"],
                    "onlyMainContent": True
                },
                timeout=30
            )
            data = response.json()
            if response.ok:
                return data.get("data", {})
        except Exception as e:
            print(f"Firecrawl error: {e}")
        return {}

    def search(self, query: str, limit: int = 10) -> list:
        """Search the web and return results with scraped content"""
        if not self.api_key:
            return []

        try:
            response = requests.post(
                f"{self.BASE_URL}/search",
                headers=self.headers,
                json={
                    "query": query,
                    "limit": limit,
                    "scrapeOptions": {"formats": ["markdown", "links"]}
                },
                timeout=30
            )
            data = response.json()
            if response.ok:
                return data.get("data", [])
        except Exception as e:
            print(f"Firecrawl search error: {e}")
        return []


class WebsiteScraper:
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    SOCIAL_PATTERNS = {
        "instagram": re.compile(r'instagram\.com/([a-zA-Z0-9_.]+)'),
        "facebook": re.compile(r'facebook\.com/([a-zA-Z0-9_.]+)'),
        "linkedin": re.compile(r'linkedin\.com/company/([a-zA-Z0-9_-]+)'),
        "twitter": re.compile(r'(?:twitter|x)\.com/([a-zA-Z0-9_]+)'),
    }

    def __init__(self, firecrawl_key: str = None):
        self.firecrawl = FirecrawlClient(firecrawl_key) if firecrawl_key else None
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })

    def scrape(self, url: str) -> dict:
        result = {"emails": [], "social_links": {}, "description": "", "scrape_method": "basic"}

        # Try Firecrawl first (better results)
        if self.firecrawl:
            fc_result = self.firecrawl.scrape(url)
            if fc_result:
                result["scrape_method"] = "firecrawl"
                markdown = fc_result.get("markdown", "")
                links = fc_result.get("links", [])

                # Extract emails from markdown
                emails = list(set(self.EMAIL_PATTERN.findall(markdown)))
                result["emails"] = [e for e in emails if not any(x in e.lower() for x in ['example', 'noreply', 'wordpress', 'wixpress'])][:5]

                # Extract social links
                all_text = markdown + " " + " ".join(links)
                for platform, pattern in self.SOCIAL_PATTERNS.items():
                    matches = pattern.findall(all_text)
                    if matches:
                        result["social_links"][platform] = matches[0]

                # Extract description (first substantial paragraph)
                paragraphs = markdown.split('\n\n')
                for p in paragraphs:
                    cleaned = p.strip().replace('#', '').strip()
                    if 50 < len(cleaned) < 500 and not cleaned.startswith('['):
                        result["description"] = cleaned
                        break

                return result

        # Fallback to basic scraping
        try:
            pages = [url, f"{url.rstrip('/')}/contact", f"{url.rstrip('/')}/about"]
            all_text = ""

            for page_url in pages:
                try:
                    resp = self.session.get(page_url, timeout=5)
                    if resp.status_code == 200:
                        all_text += resp.text + "\n"
                except:
                    continue

            emails = list(set(self.EMAIL_PATTERN.findall(all_text)))
            result["emails"] = [e for e in emails if not any(x in e.lower() for x in ['example', 'noreply', 'wordpress'])][:3]

            for platform, pattern in self.SOCIAL_PATTERNS.items():
                matches = pattern.findall(all_text)
                if matches:
                    result["social_links"][platform] = matches[0]

        except:
            pass

        return result


# ============================================================
# LEAD SCORING
# ============================================================

class LeadScorer:
    def __init__(self, config: dict):
        self.positive_signals = config.get("positive_signals", [])
        self.negative_signals = config.get("negative_signals", [])
        self.exclude_brands = [b.lower() for b in config.get("exclude_brands", [])]
        self.min_rating = config.get("min_rating", 0)
        self.target_segment = config.get("segment", "").lower()

    def score(self, lead: Lead) -> tuple[int, list[str]]:
        score = 50
        reasons = []

        name_lower = lead.name.lower()
        for excluded in self.exclude_brands:
            if excluded in name_lower:
                return 0, [f"Excluded: {excluded}"]

        if lead.rating >= 4.5:
            score += 15
            reasons.append(f"Great rating ({lead.rating})")
        elif lead.rating >= 4.0:
            score += 10
            reasons.append(f"Good rating ({lead.rating})")
        elif 0 < lead.rating < self.min_rating:
            score -= 20

        if lead.reviews_count >= 100:
            score += 10
            reasons.append(f"Established ({lead.reviews_count} reviews)")
        elif lead.reviews_count >= 30:
            score += 5

        if lead.website:
            score += 10
            reasons.append("Has website")
        else:
            score -= 15

        if lead.phone:
            score += 5

        if lead.emails_found:
            score += 10
            reasons.append("Email found")

        if lead.decision_makers:
            score += 10
            reasons.append("Decision maker found")

        searchable = f"{lead.name} {lead.category} {' '.join(lead.subcategories)}".lower()

        for signal in self.positive_signals:
            if signal.lower() in searchable:
                score += 5
                reasons.append(f"+{signal}")

        for signal in self.negative_signals:
            if signal.lower() in searchable:
                score -= 10
                reasons.append(f"-{signal}")

        if lead.social_links.get("instagram"):
            score += 5
            reasons.append("Has Instagram")

        return max(0, min(100, score)), reasons


# ============================================================
# MAIN FINDER
# ============================================================

def extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        return domain.replace("www.", "")
    except:
        return ""


def run_search(job_id: str, config: dict):
    """Run the search in background"""
    global job_status

    job_status[job_id] = {
        "status": "running",
        "step": "Initializing...",
        "progress": 0,
        "leads": [],
        "errors": [],
        "cancelled": False
    }

    def is_cancelled():
        return job_status[job_id].get("cancelled", False)

    try:
        google = GooglePlacesClient(GOOGLE_PLACES_API_KEY)
        apollo = ApolloClient(APOLLO_API_KEY)
        scraper = WebsiteScraper(FIRECRAWL_API_KEY)

        scorer_config = {
            "positive_signals": config.get("positive_signals", []),
            "negative_signals": config.get("negative_signals", []),
            "exclude_brands": config.get("exclude_brands", []),
            "min_rating": config.get("min_rating", 4.0),
            "segment": config.get("segment", "mid-premium"),
        }
        scorer = LeadScorer(scorer_config)

        leads = {}
        queries = config.get("queries", [])
        total_queries = len(queries)

        # Step 1: Search
        job_status[job_id]["step"] = "Searching Google Maps..."

        for i, query in enumerate(queries):
            if is_cancelled():
                job_status[job_id]["status"] = "cancelled"
                job_status[job_id]["step"] = "Cancelled by user"
                return

            job_status[job_id]["progress"] = int((i / total_queries) * 30)
            job_status[job_id]["step"] = f"Searching: {query[:40]}..."

            results = google.text_search(query)

            for place in results:
                place_id = place.get("place_id")
                if place_id and place_id not in leads:
                    lead = Lead(
                        place_id=place_id,
                        google_maps_url=f"https://www.google.com/maps/place/?q=place_id:{place_id}",
                        name=place.get("name", ""),
                        category=", ".join(place.get("types", [])[:2]),
                        address_full=place.get("formatted_address", ""),
                        rating=place.get("rating", 0),
                        reviews_count=place.get("user_ratings_total", 0),
                        lat=place.get("geometry", {}).get("location", {}).get("lat", 0),
                        lng=place.get("geometry", {}).get("location", {}).get("lng", 0),
                        scraped_at=datetime.now().isoformat(),
                        source_query=query,
                    )
                    leads[place_id] = lead

            time.sleep(0.2)

        if not leads:
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["step"] = "No results found"
            job_status[job_id]["progress"] = 100
            return

        # Step 2: Get details
        job_status[job_id]["step"] = "Getting place details..."
        lead_list = list(leads.values())
        total_leads = len(lead_list)

        for i, lead in enumerate(lead_list):
            if is_cancelled():
                job_status[job_id]["status"] = "cancelled"
                job_status[job_id]["step"] = "Cancelled by user"
                return

            job_status[job_id]["progress"] = 30 + int((i / total_leads) * 30)
            job_status[job_id]["step"] = f"Details: {lead.name[:30]}..."

            details = google.get_place_details(lead.place_id)
            if details:
                lead.phone = details.get("formatted_phone_number", "") or details.get("international_phone_number", "")
                lead.website = details.get("website", "")
                lead.google_maps_url = details.get("url", lead.google_maps_url)
                lead.subcategories = details.get("types", [])

                for comp in details.get("address_components", []):
                    types = comp.get("types", [])
                    if "locality" in types:
                        lead.city = comp.get("long_name", "")
                    elif "country" in types:
                        lead.country = comp.get("long_name", "")

            time.sleep(0.05)

        # Step 3: Enrich
        if config.get("enrich", True):
            job_status[job_id]["step"] = "Enriching leads..."

            for i, lead in enumerate(lead_list):
                if is_cancelled():
                    job_status[job_id]["status"] = "cancelled"
                    job_status[job_id]["step"] = "Cancelled by user"
                    return

                if not lead.website:
                    continue

                job_status[job_id]["progress"] = 60 + int((i / total_leads) * 30)
                job_status[job_id]["step"] = f"Enriching: {lead.name[:30]}..."

                domain = extract_domain(lead.website)
                if not domain:
                    continue

                # Website scraping
                scrape_result = scraper.scrape(lead.website)
                lead.emails_found = scrape_result.get("emails", [])
                lead.social_links = scrape_result.get("social_links", {})

                # Apollo
                if APOLLO_API_KEY:
                    org = apollo.enrich_organization(domain)
                    if org:
                        lead.company_size = str(org.get("estimated_num_employees", ""))
                        lead.linkedin_url = org.get("linkedin_url", "")

                    contacts = apollo.search_contacts(domain, ["owner", "founder", "director", "manager", "buyer"])
                    for contact in contacts[:2]:
                        lead.decision_makers.append({
                            "name": contact.get("name", ""),
                            "title": contact.get("title", ""),
                            "email": contact.get("email", ""),
                        })

                lead.enriched_at = datetime.now().isoformat()
                time.sleep(0.1)

        # Step 4: Score
        job_status[job_id]["step"] = "Scoring leads..."
        job_status[job_id]["progress"] = 95

        for lead in lead_list:
            score, reasons = scorer.score(lead)
            lead.fit_score = score
            lead.fit_reasons = reasons

        # Sort by score
        lead_list.sort(key=lambda x: x.fit_score, reverse=True)

        # Convert to dicts
        job_status[job_id]["leads"] = [asdict(l) for l in lead_list]
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["step"] = f"Found {len(lead_list)} leads"
        job_status[job_id]["progress"] = 100

    except Exception as e:
        job_status[job_id]["status"] = "error"
        job_status[job_id]["step"] = f"Error: {str(e)}"
        job_status[job_id]["errors"].append(str(e))


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def get_config():
    """Check if API keys are configured"""
    return jsonify({
        "google_places": bool(GOOGLE_PLACES_API_KEY),
        "apollo": bool(APOLLO_API_KEY),
        "hunter": bool(HUNTER_API_KEY),
        "firecrawl": bool(FIRECRAWL_API_KEY),
    })


@app.route("/api/search", methods=["POST"])
def start_search():
    """Start a new search job"""
    config = request.json
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Start search in background
    thread = threading.Thread(target=run_search, args=(job_id, config))
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def get_status(job_id):
    """Get status of a search job"""
    if job_id in job_status:
        return jsonify(job_status[job_id])
    return jsonify({"status": "not_found"}), 404


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel_search(job_id):
    """Cancel a running search job"""
    if job_id in job_status:
        job_status[job_id]["cancelled"] = True
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Job not found"}), 404


@app.route("/api/export/<job_id>")
def export_csv(job_id):
    """Export leads as CSV"""
    if job_id not in job_status:
        return "Not found", 404

    leads = job_status[job_id].get("leads", [])

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Score", "Name", "City", "Country", "Address", "Phone", "Website",
        "Email", "Rating", "Reviews", "Instagram", "LinkedIn",
        "Decision Maker", "DM Email", "Fit Reasons", "Google Maps URL"
    ])

    for lead in leads:
        dm = lead.get("decision_makers", [{}])[0] if lead.get("decision_makers") else {}
        writer.writerow([
            lead.get("fit_score", 0),
            lead.get("name", ""),
            lead.get("city", ""),
            lead.get("country", ""),
            lead.get("address_full", ""),
            lead.get("phone", ""),
            lead.get("website", ""),
            lead.get("emails_found", [""])[0] if lead.get("emails_found") else "",
            lead.get("rating", 0),
            lead.get("reviews_count", 0),
            lead.get("social_links", {}).get("instagram", ""),
            lead.get("linkedin_url", ""),
            dm.get("name", ""),
            dm.get("email", ""),
            "; ".join(lead.get("fit_reasons", [])),
            lead.get("google_maps_url", ""),
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'leads_{job_id}.csv'
    )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("GTM Lead Finder")
    print("="*50)
    print("\nOpen http://localhost:5001 in your browser")
    print("\nAPI Keys configured:")
    print(f"  Google Places: {'✓' if GOOGLE_PLACES_API_KEY else '✗ (required)'}")
    print(f"  Apollo.io:     {'✓' if APOLLO_API_KEY else '✗ (optional)'}")
    print(f"  Firecrawl:     {'✓' if FIRECRAWL_API_KEY else '✗ (optional)'}")
    print(f"  Hunter.io:     {'✓' if HUNTER_API_KEY else '✗ (optional)'}")
    print("\n" + "="*50 + "\n")

    app.run(debug=True, port=5001, host='0.0.0.0')
