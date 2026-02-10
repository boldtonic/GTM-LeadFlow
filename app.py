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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
INSTANTLY_API_KEY = os.getenv("INSTANTLY_API_KEY", "")
NEW_API_KEY = os.getenv("NEW_API_KEY", "")

# Global state for progress tracking
job_status = {}

# Dev/Debug logging
dev_logs = []
MAX_LOGS = 500

def log_dev(category: str, message: str, level: str = "info"):
    """Add a log entry for dev panel"""
    global dev_logs
    entry = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "message": message,
        "level": level  # info, success, warning, error
    }
    dev_logs.append(entry)
    # Keep only last MAX_LOGS
    if len(dev_logs) > MAX_LOGS:
        dev_logs = dev_logs[-MAX_LOGS:]
    # Also print to console
    print(f"[{level.upper()}] [{category}] {message}")


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
            if not response.ok:
                log_dev("APOLLO", f"Org enrich failed for {domain}: {data.get('error', response.status_code)}", "error")
                return {}
            return data.get("organization", {})
        except Exception as e:
            log_dev("APOLLO", f"Org enrich exception for {domain}: {str(e)[:50]}", "error")
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
            if not response.ok:
                log_dev("APOLLO", f"Contact search failed for {domain}: {result.get('error', response.status_code)}", "error")
                return []
            return result.get("people", [])
        except Exception as e:
            log_dev("APOLLO", f"Contact search exception for {domain}: {str(e)[:50]}", "error")
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
            else:
                error_msg = data.get("error", response.status_code)
                log_dev("FIRECRAWL", f"Failed for {url[:40]}: {error_msg}", "error")
        except Exception as e:
            log_dev("FIRECRAWL", f"Exception for {url[:40]}: {str(e)[:50]}", "error")
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


class HunterClient:
    """Hunter.io API client for email finding and verification"""
    BASE_URL = "https://api.hunter.io/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def domain_search(self, domain: str, limit: int = 10) -> dict:
        """Find emails associated with a domain"""
        if not self.api_key:
            return {}

        try:
            response = requests.get(
                f"{self.BASE_URL}/domain-search",
                params={
                    "domain": domain,
                    "api_key": self.api_key,
                    "limit": limit
                },
                timeout=10
            )
            data = response.json()
            if response.ok and data.get("data"):
                return data.get("data", {})
        except Exception as e:
            print(f"Hunter domain search error: {e}")
        return {}

    def email_finder(self, domain: str, first_name: str = None, last_name: str = None) -> dict:
        """Find a specific person's email at a company"""
        if not self.api_key:
            return {}

        params = {"domain": domain, "api_key": self.api_key}
        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name

        try:
            response = requests.get(
                f"{self.BASE_URL}/email-finder",
                params=params,
                timeout=10
            )
            data = response.json()
            if response.ok and data.get("data"):
                return data.get("data", {})
        except Exception as e:
            print(f"Hunter email finder error: {e}")
        return {}

    def email_verifier(self, email: str) -> dict:
        """Verify if an email address is valid and deliverable"""
        if not self.api_key:
            return {}

        try:
            response = requests.get(
                f"{self.BASE_URL}/email-verifier",
                params={"email": email, "api_key": self.api_key},
                timeout=10
            )
            data = response.json()
            if response.ok and data.get("data"):
                return data.get("data", {})
        except Exception as e:
            print(f"Hunter email verifier error: {e}")
        return {}

    def email_count(self, domain: str) -> dict:
        """Get the number of emails found for a domain (free endpoint)"""
        if not self.api_key:
            return {}

        try:
            response = requests.get(
                f"{self.BASE_URL}/email-count",
                params={"domain": domain, "api_key": self.api_key},
                timeout=10
            )
            data = response.json()
            if response.ok and data.get("data"):
                return data.get("data", {})
        except Exception as e:
            print(f"Hunter email count error: {e}")
        return {}


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


# Domains that are platforms, not actual businesses — skip for enrichment
SKIP_DOMAINS = {
    "instagram.com", "facebook.com", "twitter.com", "x.com",
    "linkedin.com", "tiktok.com", "youtube.com", "pinterest.com",
    "linktr.ee", "linktree.com", "bit.ly", "goo.gl",
    "yelp.com", "tripadvisor.com", "tripadvisor.es", "google.com",
    "maps.google.com", "foursquare.com", "booking.com",
    "ubereats.com", "deliveroo.com", "glovo.com", "justeat.com",
    "thefork.com", "eltenedor.es", "opentable.com",
}


def is_business_domain(domain: str) -> bool:
    """Return True if domain belongs to actual business, not a social/platform site."""
    if not domain:
        return False
    d = domain.lower().strip()
    for skip in SKIP_DOMAINS:
        if d == skip or d.endswith("." + skip):
            return False
    return True


def run_search(job_id: str, config: dict):
    """Run the search in background"""
    global job_status

    job_status[job_id] = {
        "status": "running",
        "step": "Initializing...",
        "progress": 0,
        "leads": [],
        "errors": [],
        "cancelled": False,
        "enrichment_stats": {}
    }

    def is_cancelled():
        return job_status[job_id].get("cancelled", False)

    try:
        # Log search start
        queries = config.get("queries", [])
        log_dev("SEARCH", f"Starting search job {job_id} with {len(queries)} queries", "info")
        log_dev("CONFIG", f"APIs: Google={'✓' if GOOGLE_PLACES_API_KEY else '✗'}, Apollo={'✓' if APOLLO_API_KEY else '✗'}, Firecrawl={'✓' if FIRECRAWL_API_KEY else '✗'}, Hunter={'✓' if HUNTER_API_KEY else '✗'}", "info")

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
        total_queries = len(queries)

        # Step 1: Search
        job_status[job_id]["step"] = "Searching Google Maps..."
        log_dev("GOOGLE", f"Starting Google Places search with {total_queries} queries", "info")

        for i, query in enumerate(queries):
            if is_cancelled():
                job_status[job_id]["status"] = "cancelled"
                job_status[job_id]["step"] = "Cancelled by user"
                log_dev("SEARCH", "Search cancelled by user", "warning")
                return

            job_status[job_id]["progress"] = int((i / total_queries) * 30)
            job_status[job_id]["step"] = f"Searching: {query[:40]}..."

            results = google.text_search(query)
            log_dev("GOOGLE", f"Query '{query[:30]}...' returned {len(results)} results", "info")

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
            log_dev("ENRICH", f"Starting enrichment for {total_leads} leads", "info")

            # Enrichment statistics
            enrich_stats = {
                "firecrawl_success": 0,
                "firecrawl_failed": 0,
                "basic_scrape": 0,
                "apollo_org_success": 0,
                "apollo_org_failed": 0,
                "apollo_contacts_found": 0,
                "hunter_emails_found": 0,
                "hunter_verified": 0,
                "hunter_failed": 0,
                "emails_found": 0,
                "social_found": 0,
                "no_website": 0,
            }

            for i, lead in enumerate(lead_list):
                if is_cancelled():
                    job_status[job_id]["status"] = "cancelled"
                    job_status[job_id]["step"] = "Cancelled by user"
                    log_dev("ENRICH", "Enrichment cancelled by user", "warning")
                    return

                if not lead.website:
                    enrich_stats["no_website"] += 1
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

                # Track scraping method
                if scrape_result.get("scrape_method") == "firecrawl":
                    enrich_stats["firecrawl_success"] += 1
                else:
                    enrich_stats["basic_scrape"] += 1

                if lead.emails_found:
                    enrich_stats["emails_found"] += 1
                if lead.social_links:
                    enrich_stats["social_found"] += 1

                # Only enrich via Apollo/Hunter if domain is a real business site
                biz_domain = is_business_domain(domain)

                # Apollo
                if APOLLO_API_KEY and biz_domain:
                    try:
                        org = apollo.enrich_organization(domain)
                        if org:
                            lead.company_size = str(org.get("estimated_num_employees", ""))
                            lead.linkedin_url = org.get("linkedin_url", "")
                            enrich_stats["apollo_org_success"] += 1
                        else:
                            enrich_stats["apollo_org_failed"] += 1
                    except Exception as e:
                        enrich_stats["apollo_org_failed"] += 1
                        log_dev("APOLLO", f"Org enrich failed for {domain}: {str(e)[:50]}", "error")

                    try:
                        contacts = apollo.search_contacts(domain, ["owner", "founder", "director", "manager", "buyer"])
                        for contact in contacts[:2]:
                            lead.decision_makers.append({
                                "name": contact.get("name", ""),
                                "title": contact.get("title", ""),
                                "email": contact.get("email", ""),
                            })
                        if contacts:
                            enrich_stats["apollo_contacts_found"] += 1
                    except Exception as e:
                        log_dev("APOLLO", f"Contact search failed for {domain}: {str(e)[:50]}", "error")

                # Hunter.io email enrichment
                if HUNTER_API_KEY and biz_domain:
                    try:
                        hunter = HunterClient(HUNTER_API_KEY)
                        hunter_data = hunter.domain_search(domain, limit=5)
                        if hunter_data:
                            hunter_emails = hunter_data.get("emails", [])
                            existing_emails = {e.lower() for e in lead.emails_found}
                            # Also collect emails from decision makers
                            for dm in lead.decision_makers:
                                if dm.get("email"):
                                    existing_emails.add(dm["email"].lower())

                            for he in hunter_emails:
                                email = he.get("value", "")
                                if email and email.lower() not in existing_emails:
                                    lead.emails_found.append(email)
                                    existing_emails.add(email.lower())
                                    # Add as decision maker if has name + title
                                    name = f"{he.get('first_name', '')} {he.get('last_name', '')}".strip()
                                    if name and he.get("position"):
                                        lead.decision_makers.append({
                                            "name": name,
                                            "title": he.get("position", ""),
                                            "email": email,
                                            "source": "hunter",
                                            "confidence": he.get("confidence", 0),
                                        })

                            if hunter_emails:
                                enrich_stats["hunter_emails_found"] += len(hunter_emails)
                                log_dev("HUNTER", f"Found {len(hunter_emails)} emails for {domain}", "info")
                    except Exception as e:
                        log_dev("HUNTER", f"Domain search failed for {domain}: {str(e)[:50]}", "error")

                # Email verification (verify best email)
                if HUNTER_API_KEY and biz_domain and lead.emails_found:
                    try:
                        hunter = HunterClient(HUNTER_API_KEY)
                        best_email = lead.emails_found[0]
                        verify_result = hunter.email_verifier(best_email)
                        if verify_result:
                            status = verify_result.get("result", "unknown")
                            if status == "deliverable":
                                enrich_stats["hunter_verified"] += 1
                            elif status == "undeliverable":
                                # Demote bad email to end of list
                                lead.emails_found.append(lead.emails_found.pop(0))
                                enrich_stats["hunter_failed"] += 1
                                log_dev("HUNTER", f"Undeliverable email demoted for {domain}: {best_email}", "warning")
                    except Exception as e:
                        log_dev("HUNTER", f"Verify failed for {domain}: {str(e)[:50]}", "error")

                lead.enriched_at = datetime.now().isoformat()
                time.sleep(0.1)

                # Log progress every 50 leads
                if (i + 1) % 50 == 0:
                    log_dev("ENRICH", f"Progress: {i + 1}/{total_leads} leads enriched", "info")

            # Save enrichment stats
            job_status[job_id]["enrichment_stats"] = enrich_stats
            log_dev("ENRICH", f"Enrichment complete: Firecrawl={enrich_stats['firecrawl_success']}, Basic={enrich_stats['basic_scrape']}, Apollo={enrich_stats['apollo_org_success']}, Hunter={enrich_stats['hunter_emails_found']} emails/{enrich_stats['hunter_verified']} verified", "success")

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
        "openai": bool(OPENAI_API_KEY),
        "instantly": bool(INSTANTLY_API_KEY),
        "new_api": bool(NEW_API_KEY),
    })


@app.route("/api/dev/logs")
def get_dev_logs():
    """Get dev logs for debugging panel"""
    limit = request.args.get("limit", 100, type=int)
    category = request.args.get("category", None)

    logs = dev_logs[-limit:]
    if category:
        logs = [l for l in logs if l["category"] == category]

    return jsonify({"logs": logs})


@app.route("/api/dev/logs/clear", methods=["POST"])
def clear_dev_logs():
    """Clear all dev logs"""
    global dev_logs
    dev_logs = []
    return jsonify({"success": True})


@app.route("/api/dev/stats")
def get_dev_stats():
    """Get enrichment statistics from last job"""
    # Find the most recent completed job
    recent_job = None
    for job_id, job in sorted(job_status.items(), reverse=True):
        if job.get("status") == "completed":
            recent_job = job
            break

    if not recent_job:
        return jsonify({"hasData": False})

    leads = recent_job.get("leads", [])
    stats = recent_job.get("enrichment_stats", {})

    # Calculate data quality metrics
    total = len(leads)
    with_email = sum(1 for l in leads if l.get("emails_found") or any(dm.get("email") for dm in l.get("decision_makers", [])))
    with_social = sum(1 for l in leads if l.get("social_links"))
    with_instagram = sum(1 for l in leads if l.get("social_links", {}).get("instagram"))
    with_linkedin = sum(1 for l in leads if l.get("linkedin_url") or l.get("social_links", {}).get("linkedin"))
    with_decision_makers = sum(1 for l in leads if l.get("decision_makers"))
    with_website = sum(1 for l in leads if l.get("website"))

    # Score distribution
    high_fit = sum(1 for l in leads if l.get("fit_score", 0) >= 70)
    medium_fit = sum(1 for l in leads if 50 <= l.get("fit_score", 0) < 70)
    low_fit = sum(1 for l in leads if l.get("fit_score", 0) < 50)

    return jsonify({
        "hasData": True,
        "total_leads": total,
        "data_quality": {
            "with_email": with_email,
            "with_social": with_social,
            "with_instagram": with_instagram,
            "with_linkedin": with_linkedin,
            "with_decision_makers": with_decision_makers,
            "with_website": with_website,
        },
        "score_distribution": {
            "high": high_fit,
            "medium": medium_fit,
            "low": low_fit
        },
        "enrichment_stats": stats
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
        "Score", "Name", "Category", "City", "Country", "Address", "Postal Code",
        "Phone", "Website", "Domain",
        "Email (Best)", "Email (All)",
        "Rating", "Reviews", "Price Level",
        "Instagram", "Facebook", "Twitter", "LinkedIn",
        "Company Size", "Revenue", "About",
        "DM1 Name", "DM1 Title", "DM1 Email",
        "DM2 Name", "DM2 Title", "DM2 Email",
        "Fit Reasons", "Google Maps URL", "Source Query", "Enriched At"
    ])

    for lead in leads:
        dms = lead.get("decision_makers", [])
        dm1 = dms[0] if len(dms) > 0 else {}
        dm2 = dms[1] if len(dms) > 1 else {}
        social = lead.get("social_links", {})
        emails = lead.get("emails_found", [])
        website = lead.get("website", "")
        # Extract domain from website
        domain = ""
        if website:
            domain = website.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/").split("/")[0]

        writer.writerow([
            lead.get("fit_score", 0),
            lead.get("name", ""),
            lead.get("category", ""),
            lead.get("city", ""),
            lead.get("country", ""),
            lead.get("address_full", ""),
            lead.get("postal_code", ""),
            lead.get("phone", ""),
            website,
            domain,
            emails[0] if emails else "",
            "; ".join(emails) if len(emails) > 1 else "",
            lead.get("rating", 0),
            lead.get("reviews_count", 0),
            lead.get("price_level", ""),
            social.get("instagram", ""),
            social.get("facebook", ""),
            social.get("twitter", ""),
            lead.get("linkedin_url", "") or social.get("linkedin", ""),
            lead.get("company_size", ""),
            lead.get("estimated_revenue", ""),
            lead.get("about_text", "")[:200],
            dm1.get("name", ""),
            dm1.get("title", ""),
            dm1.get("email", ""),
            dm2.get("name", ""),
            dm2.get("title", ""),
            dm2.get("email", ""),
            "; ".join(lead.get("fit_reasons", [])),
            lead.get("google_maps_url", ""),
            lead.get("source_query", ""),
            lead.get("enriched_at", ""),
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'leads_{job_id}.csv'
    )


# ============================================================
# NEW ENDPOINTS: GTM Brief, Discovery, Enrichment
# ============================================================

@app.route("/api/brief", methods=["POST"])
def generate_brief():
    """Generate comprehensive GTM Brief using Firecrawl + OpenAI"""
    data = request.json
    url = data.get("url", "").strip()
    additional_context = data.get("additionalContext", "")

    if not url:
        return jsonify({"success": False, "error": "URL is required"})

    if not FIRECRAWL_API_KEY:
        return jsonify({"success": False, "error": "Firecrawl API not configured"})

    if not OPENAI_API_KEY:
        return jsonify({"success": False, "error": "OpenAI API not configured"})

    # Format URL
    if not url.startswith('http://') and not url.startswith('https://'):
        url = f'https://{url}'

    try:
        # Step 1: Scrape website with Firecrawl
        print(f"Scraping website: {url}")
        firecrawl = FirecrawlClient(FIRECRAWL_API_KEY)
        scrape_result = firecrawl.scrape(url)

        if not scrape_result:
            return jsonify({"success": False, "error": "Failed to scrape website"})

        markdown = scrape_result.get("markdown", "")
        links = scrape_result.get("links", [])
        print(f"Scraped content length: {len(markdown)}")

        # Step 2: Generate GTM Brief using OpenAI
        system_prompt = """You are a B2B go-to-market research analyst. Analyze the provided website content and generate a comprehensive GTM brief. Be specific, actionable, and data-driven in your analysis.

IMPORTANT: For target personas, classify each into one of three roles:
- decision_maker: C-suite, VPs, Directors who approve purchases
- influencer: Managers who research and recommend solutions
- worker: End-users who can advocate internally and relay messages to decision-makers

Include at least one persona from each category when relevant."""

        user_prompt = f"""Analyze this company website content and generate a detailed GTM (Go-To-Market) brief.

Website: {url}
{f'Additional Context: {additional_context}' if additional_context else ''}

Website Content:
{markdown[:15000]}

Social/External Links Found:
{chr(10).join(links[:20]) if links else 'None'}

Generate a comprehensive GTM brief with:
1. Company snapshot (name, tagline, industry, location if found)
2. Product intelligence (products/services, features, pricing if available, tech stack)
3. Market positioning (market tension they address, value proposition, differentiators, proof points)
4. Ideal Customer Profile - include THREE types of personas:
   - Decision Makers (C-suite, VPs who approve)
   - Influencers (Managers who research/recommend)
   - Workers/Users (End-users who can advocate internally)
5. Search strategy for finding PROSPECTS (businesses that would BUY from this company):
   - suggestedQueries: Google Maps/Places search queries to find POTENTIAL CUSTOMER BUSINESSES (e.g., if analyzing a shoe brand, queries should be "multi-brand shoe store", "women's footwear boutique", "luxury shoe retailer" - NOT queries about the brand itself)
   - geographies: target locations/cities for prospecting
   - outreachAngles: messaging angles for each persona

CRITICAL: The suggestedQueries must be searches to find BUSINESSES THAT WOULD BUY FROM the analyzed company, not searches about the company itself. Think: "What type of business would be a customer of this company?" and create Google Maps search queries for those businesses."""

        print("Calling OpenAI...")

        # OpenAI function definition for structured output
        tools = [{
            "type": "function",
            "function": {
                "name": "generate_gtm_brief",
                "description": "Generate a structured GTM brief from website analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "companySnapshot": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "tagline": {"type": "string"},
                                "website": {"type": "string"},
                                "industry": {"type": "string"},
                                "founded": {"type": "string"},
                                "location": {"type": "string"},
                                "employees": {"type": "string"},
                                "revenue": {"type": "string"}
                            },
                            "required": ["name", "tagline", "industry"]
                        },
                        "productIntelligence": {
                            "type": "object",
                            "properties": {
                                "products": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "description": {"type": "string"},
                                            "category": {"type": "string"},
                                            "pricing": {"type": "string"}
                                        }
                                    }
                                },
                                "keyFeatures": {"type": "array", "items": {"type": "string"}},
                                "integrations": {"type": "array", "items": {"type": "string"}},
                                "techStack": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "positioning": {
                            "type": "object",
                            "properties": {
                                "marketTension": {"type": "string"},
                                "valueProposition": {"type": "string"},
                                "differentiators": {"type": "array", "items": {"type": "string"}},
                                "proofPoints": {"type": "array", "items": {"type": "string"}},
                                "targetOutcome": {"type": "string"}
                            }
                        },
                        "icp": {
                            "type": "object",
                            "properties": {
                                "primaryProfiles": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "seniority": {"type": "string"},
                                            "department": {"type": "string"},
                                            "role": {"type": "string", "enum": ["decision_maker", "influencer", "worker"]},
                                            "painPoints": {"type": "array", "items": {"type": "string"}},
                                            "goals": {"type": "array", "items": {"type": "string"}}
                                        }
                                    }
                                },
                                "companyCharacteristics": {
                                    "type": "object",
                                    "properties": {
                                        "size": {"type": "string"},
                                        "industries": {"type": "array", "items": {"type": "string"}, "description": "Industries of PROSPECT companies that would buy from the analyzed company"},
                                        "technologies": {"type": "array", "items": {"type": "string"}},
                                        "signals": {"type": "array", "items": {"type": "string"}, "description": "Positive signals to look for in PROSPECT businesses (e.g., 'premium positioning', 'multi-brand', 'curated selection')"}
                                    }
                                },
                                "qualificationCriteria": {"type": "array", "items": {"type": "string"}},
                                "disqualifiers": {"type": "array", "items": {"type": "string"}, "description": "Negative signals - types of businesses to EXCLUDE as prospects (e.g., 'discount retailers', 'outlet stores', 'sports-only stores')"}
                            }
                        },
                        "searchStrategy": {
                            "type": "object",
                            "properties": {
                                "suggestedQueries": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Google Maps search queries to find POTENTIAL CUSTOMER BUSINESSES - businesses that would buy from the analyzed company. E.g., for a shoe brand: 'multi-brand shoe store', 'women's footwear boutique', 'luxury footwear retailer'"
                                },
                                "geographies": {"type": "array", "items": {"type": "string"}},
                                "outreachAngles": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "angle": {"type": "string"},
                                            "targetPersona": {"type": "string"},
                                            "keyMessage": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "required": ["companySnapshot", "productIntelligence", "positioning", "icp", "searchStrategy"]
                }
            }
        }]

        # Retry logic for rate limits (429)
        max_retries = 3
        retry_delay = 5  # seconds
        openai_response = None

        for attempt in range(max_retries):
            openai_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "tools": tools,
                    "tool_choice": {"type": "function", "function": {"name": "generate_gtm_brief"}}
                },
                timeout=60
            )

            if openai_response.status_code == 429:
                error_data = openai_response.json() if openai_response.text else {}
                error_msg = error_data.get("error", {}).get("message", "Rate limit exceeded")
                print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {error_msg}")

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 5, 10, 20 seconds
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
            break

        if not openai_response.ok:
            error_text = openai_response.text
            print(f"OpenAI error: {openai_response.status_code} - {error_text}")

            # Parse error for better message
            try:
                error_data = json.loads(error_text)
                error_message = error_data.get("error", {}).get("message", f"API error: {openai_response.status_code}")
            except:
                error_message = f"OpenAI API error: {openai_response.status_code}"

            return jsonify({"success": False, "error": error_message})

        ai_data = openai_response.json()

        # Extract function call result
        tool_call = ai_data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [{}])[0]
        if not tool_call or tool_call.get("function", {}).get("name") != "generate_gtm_brief":
            print("No tool call in response")
            return jsonify({"success": False, "error": "Failed to parse AI response"})

        brief_data = json.loads(tool_call["function"]["arguments"])

        # Add metadata
        now = datetime.now().isoformat()
        brief = {
            "id": str(hash(url + now))[:12],
            **brief_data,
            "createdAt": now,
            "updatedAt": now
        }

        # Ensure website is set
        if "companySnapshot" in brief:
            brief["companySnapshot"]["website"] = url

        print("GTM Brief generated successfully")
        return jsonify({"success": True, "brief": brief})

    except Exception as e:
        print(f"Error generating brief: {e}")
        return jsonify({"success": False, "error": str(e)})


def run_web_search(search_queries: list, category: str, location: str) -> list:
    """Run web search using Firecrawl"""
    if not FIRECRAWL_API_KEY:
        return []

    prospects = []
    now = datetime.now().isoformat()
    firecrawl = FirecrawlClient(FIRECRAWL_API_KEY)

    for query in search_queries[:3]:
        print(f"Web searching: {query}")
        try:
            results = firecrawl.search(query, limit=15)

            for result in results:
                title = result.get("title", "")
                url = result.get("url", "")
                markdown = result.get("markdown", "")

                # Extract business name from title
                name_match = re.match(r'^([^-–|]+)', title)
                name = name_match.group(1).strip() if name_match else title.split(' - ')[0]

                if not name or len(name) < 2 or len(name) > 100:
                    continue

                # Skip non-business results
                skip_domains = ['wikipedia.org', 'facebook.com', 'linkedin.com/company',
                               'yelp.com', 'tripadvisor.com', 'yellowpages.com']
                if any(d in url for d in skip_domains):
                    continue

                # Extract rating
                rating_match = re.search(r'(\d\.?\d?)\s*(?:stars?|rating|★)', markdown, re.I)
                rating = float(rating_match.group(1)) if rating_match else None

                # Extract review count
                review_match = re.search(r'(\d+)\s*reviews?', markdown, re.I)
                review_count = int(review_match.group(1)) if review_match else None

                # Extract phone
                phone_match = re.search(r'(?:\+1|1)?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', markdown)
                phone = phone_match.group(0) if phone_match else None

                # Get website
                website = url
                if 'google.com' in url or 'bing.com' in url:
                    links = result.get("links", [])
                    for link in links:
                        if link.startswith('http') and not any(d in link for d in ['google.com', 'facebook.com']):
                            website = link
                            break

                prospects.append({
                    "id": str(len(prospects) + 1),
                    "name": name[:100],
                    "category": category or "Business",
                    "location": location or "Unknown",
                    "rating": rating,
                    "reviewCount": review_count,
                    "website": website,
                    "phone": phone,
                    "sourceUrl": url,
                    "discoveredAt": now
                })

        except Exception as e:
            print(f"Error searching '{query}': {e}")

    return prospects


def run_apollo_search(category: str, location: str, industry: str = None) -> dict:
    """Run Apollo company search with error handling"""
    if not APOLLO_API_KEY:
        return {"prospects": [], "error": "Apollo API not configured"}

    print(f"Searching Apollo for: {industry or category} in {location}")

    try:
        response = requests.post(
            "https://api.apollo.io/api/v1/mixed_companies/search",
            headers={"Content-Type": "application/json", "X-Api-Key": APOLLO_API_KEY},
            json={
                "q_organization_name": category or industry,
                "organization_locations": [location] if location else None,
                "per_page": 25
            },
            timeout=15
        )

        data = response.json()

        if not response.ok:
            print(f"Apollo API error: {data}")
            if data.get("error_code") == "API_INACCESSIBLE":
                return {"prospects": [], "error": "Apollo API requires a paid plan for company search"}
            return {"prospects": [], "error": data.get("error", "Failed to search Apollo")}

        now = datetime.now().isoformat()
        prospects = []

        for org in data.get("organizations", []):
            loc = f"{org.get('city')}, {org.get('country')}" if org.get('city') else location
            prospects.append({
                "id": str(len(prospects) + 1),
                "name": org.get("name", "Unknown"),
                "category": industry or category or "Business",
                "location": loc or "Unknown",
                "address": org.get("street_address"),
                "website": org.get("website_url"),
                "phone": org.get("phone"),
                "sourceUrl": org.get("linkedin_url") or f"https://apollo.io/companies/{org.get('id')}",
                "discoveredAt": now
            })

        print(f"Found {len(prospects)} companies via Apollo")
        return {"prospects": prospects}

    except Exception as e:
        print(f"Apollo search error: {e}")
        return {"prospects": [], "error": str(e)}


def deduplicate_prospects(prospects: list) -> list:
    """Remove duplicate prospects by name or website"""
    seen_names = set()
    seen_websites = set()
    unique = []

    for p in prospects:
        name_lower = p.get("name", "").lower()
        website_lower = (p.get("website") or "").lower()

        if name_lower in seen_names:
            continue
        if website_lower and website_lower in seen_websites:
            continue

        seen_names.add(name_lower)
        if website_lower:
            seen_websites.add(website_lower)
        unique.append(p)

    return unique


def parse_maps_markdown(markdown: str, links: list, location: str) -> list:
    """Parse Google Maps markdown to extract business listings"""
    businesses = []
    seen_names = set()
    lines = markdown.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip empty or irrelevant lines
        if not line or len(line) < 3 or len(line) > 100:
            continue
        if line.startswith('http') or line.startswith('#'):
            continue
        if any(x in line.lower() for x in ['google maps', 'sign in', 'privacy', 'terms']):
            continue

        # Look for business name patterns
        is_business = re.match(r'^[A-Z]', line) and '·' not in line and len(line) <= 60

        if is_business:
            context = ' '.join(lines[i:i+6])

            # Rating
            rating_match = re.search(r'(\d\.?\d?)\s*(?:stars?|★|⭐)', context, re.I)
            rating = float(rating_match.group(1)) if rating_match else None

            # Review count
            review_match = re.search(r'\((\d[\d,]*)\s*(?:reviews?|reseñas?)\)', context, re.I)
            review_count = int(review_match.group(1).replace(',', '')) if review_match else None

            # Phone
            phone_match = re.search(r'(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})', context)
            phone = phone_match.group(1).strip() if phone_match else None

            # Address
            address_match = re.search(r'(\d+\s+[A-Za-z0-9\s,.-]+(?:St|Ave|Rd|Blvd|Dr|Way|Ln|Street|Avenue|Road)[^,]*)', context, re.I)
            address = address_match.group(1).strip() if address_match else None

            # Category
            cat_match = re.search(r'·\s*([A-Za-z\s&]+?)(?:\s*·|$)', context)
            category = cat_match.group(1).strip() if cat_match else None

            # Hours
            hours_match = re.search(r'(?:Open|Closed)\s*(?:until|at)?\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM)?', context, re.I)
            hours = hours_match.group(0) if hours_match else None

            # Website from links
            website = None
            name_parts = line.lower().split()[:3]
            for link in links:
                if any(d in link for d in ['google.com', 'facebook.com', 'yelp.com']):
                    continue
                if any(p in link.lower() for p in name_parts if len(p) > 3):
                    website = link
                    break

            # Only add if we have enough context
            if (rating or review_count or phone or address) and line.lower() not in seen_names:
                seen_names.add(line.lower())
                businesses.append({
                    "name": line,
                    "category": category,
                    "rating": rating,
                    "reviewCount": review_count,
                    "address": address,
                    "phone": phone,
                    "website": website,
                    "hours": hours
                })

    return businesses


@app.route("/api/discover", methods=["POST"])
def discover_prospects():
    """Discover prospects using various methods - matches LeadFlow functionality"""
    data = request.json
    mode = data.get("mode", "smart_search")
    queries = data.get("queries", [])
    query = data.get("query", "")
    category = data.get("category", "")
    location = data.get("location", "")
    industry = data.get("industry", "")
    maps_url = data.get("mapsUrl", "")

    try:
        now = datetime.now().isoformat()

        # Build search queries if not provided
        if not queries and query:
            queries = [query]
        elif not queries and category and location:
            queries = [
                f"{category} companies {location}",
                f"{category} stores {location}",
                f"best {category} businesses {location}",
                f"{category} retailers {location} contact"
            ]

        # Mode: Maps URL scraping
        if mode == "maps_url" or mode == "maps":
            if not maps_url:
                return jsonify({"success": False, "error": "Google Maps URL is required"})

            if not FIRECRAWL_API_KEY:
                return jsonify({"success": False, "error": "Firecrawl not configured"})

            print(f"Scraping Google Maps URL: {maps_url}")
            firecrawl = FirecrawlClient(FIRECRAWL_API_KEY)

            # Try structured extraction first
            try:
                response = requests.post(
                    f"{firecrawl.BASE_URL}/scrape",
                    headers=firecrawl.headers,
                    json={
                        "url": maps_url,
                        "formats": ["extract", "markdown", "links"],
                        "extract": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "businesses": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "category": {"type": "string"},
                                                "rating": {"type": "number"},
                                                "reviewCount": {"type": "number"},
                                                "address": {"type": "string"},
                                                "phone": {"type": "string"},
                                                "website": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["businesses"]
                            },
                            "systemPrompt": "Extract all business listings from this Google Maps search results page."
                        },
                        "waitFor": 5000
                    },
                    timeout=45
                )

                scrape_data = response.json()
                extract_data = scrape_data.get("data", {}).get("extract", {})
                businesses = extract_data.get("businesses", [])

                if not businesses:
                    # Fallback to markdown parsing
                    markdown = scrape_data.get("data", {}).get("markdown", "")
                    links = scrape_data.get("data", {}).get("links", [])
                    businesses = parse_maps_markdown(markdown, links, location)

            except Exception as e:
                print(f"Structured extraction failed: {e}")
                # Try basic scrape
                scrape_result = firecrawl.scrape(maps_url)
                if scrape_result:
                    markdown = scrape_result.get("markdown", "")
                    links = scrape_result.get("links", [])
                    businesses = parse_maps_markdown(markdown, links, location)
                else:
                    businesses = []

            # Convert to prospects
            prospects = []
            for b in businesses:
                if not b.get("name") or len(b.get("name", "")) < 2:
                    continue
                prospects.append({
                    "id": str(len(prospects) + 1),
                    "name": b.get("name", "")[:100],
                    "category": b.get("category") or category or "Business",
                    "location": location or "Unknown",
                    "address": b.get("address"),
                    "rating": b.get("rating"),
                    "reviewCount": b.get("reviewCount"),
                    "website": b.get("website"),
                    "phone": b.get("phone"),
                    "hours": b.get("hours"),
                    "sourceUrl": maps_url,
                    "discoveredAt": now
                })

            # Sort by quality
            prospects.sort(key=lambda x: (
                (3 if x.get("phone") else 0) +
                (2 if x.get("website") else 0) +
                (1 if x.get("rating") else 0) +
                (1 if x.get("address") else 0)
            ), reverse=True)

            return jsonify({
                "success": True,
                "prospects": prospects,
                "totalFound": len(prospects),
                "source": "maps_scrape"
            })

        # Mode: Smart Search - try all sources
        if mode == "smart_search":
            print("Running Smart Search across all sources")
            all_prospects = []
            sources = []
            fallback_reason = None

            # Web search
            if FIRECRAWL_API_KEY and queries:
                web_prospects = run_web_search(queries, category, location)
                all_prospects.extend(web_prospects)
                if web_prospects:
                    sources.append("web")
                print(f"Web search found {len(web_prospects)} prospects")

            # Apollo search
            if APOLLO_API_KEY:
                apollo_result = run_apollo_search(category, location, industry)
                if apollo_result.get("prospects"):
                    all_prospects.extend(apollo_result["prospects"])
                    sources.append("apollo")
                elif apollo_result.get("error"):
                    fallback_reason = apollo_result["error"]

            unique_prospects = deduplicate_prospects(all_prospects)
            print(f"Smart search returning {len(unique_prospects)} unique prospects from: {', '.join(sources)}")

            return jsonify({
                "success": True,
                "prospects": unique_prospects,
                "totalFound": len(unique_prospects),
                "queriesUsed": queries[:3],
                "source": "smart_search",
                "fallbackReason": fallback_reason
            })

        # Mode: Apollo Search (with fallback)
        if mode == "apollo_search":
            apollo_result = run_apollo_search(category, location, industry)

            if apollo_result.get("prospects"):
                return jsonify({
                    "success": True,
                    "prospects": apollo_result["prospects"],
                    "totalFound": len(apollo_result["prospects"]),
                    "source": "apollo"
                })

            # Fallback to web search
            if FIRECRAWL_API_KEY and queries:
                print(f"Apollo failed: {apollo_result.get('error')}. Falling back to web search.")
                web_prospects = run_web_search(queries, category, location)
                unique_prospects = deduplicate_prospects(web_prospects)

                return jsonify({
                    "success": True,
                    "prospects": unique_prospects,
                    "totalFound": len(unique_prospects),
                    "queriesUsed": queries[:3],
                    "source": "web_search_fallback",
                    "fallbackReason": apollo_result.get("error")
                })

            return jsonify({"success": False, "error": apollo_result.get("error", "Apollo search failed")})

        # Mode: Web Search
        if not FIRECRAWL_API_KEY:
            return jsonify({"success": False, "error": "Firecrawl not configured"})

        if not queries:
            return jsonify({"success": False, "error": "Search query or queries[] required"})

        print(f"Running web search with {len(queries)} queries")
        all_prospects = run_web_search(queries, category, location)
        unique_prospects = deduplicate_prospects(all_prospects)

        return jsonify({
            "success": True,
            "prospects": unique_prospects,
            "totalFound": len(unique_prospects),
            "queriesUsed": queries[:3],
            "source": "web_search"
        })

    except Exception as e:
        print(f"Error discovering prospects: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/enrich-contacts", methods=["POST"])
def enrich_contacts():
    """Dedicated endpoint for finding contacts at a company via Apollo"""
    data = request.json
    domain = data.get("domain", "").strip()
    company_name = data.get("companyName", "").strip()

    if not domain and not company_name:
        return jsonify({"success": False, "error": "Domain or company name is required"})

    if not APOLLO_API_KEY:
        return jsonify({"success": False, "error": "Apollo API not configured"})

    try:
        print(f"Enriching contacts for: {domain or company_name}")

        # First, find the organization
        org_response = requests.post(
            "https://api.apollo.io/api/v1/mixed_companies/search",
            headers={"Content-Type": "application/json", "X-Api-Key": APOLLO_API_KEY},
            json={
                "q_organization_domains": [domain] if domain else None,
                "q_organization_name": company_name if not domain else None,
                "page": 1,
                "per_page": 1
            },
            timeout=15
        )

        org_data = org_response.json()

        if not org_response.ok:
            return jsonify({"success": False, "error": "Failed to search organization in Apollo"})

        organization = org_data.get("organizations", [{}])[0] if org_data.get("organizations") else None

        if not organization:
            return jsonify({
                "success": True,
                "data": {
                    "domain": domain,
                    "contacts": [],
                    "apolloEnriched": False
                }
            })

        print(f"Found organization: {organization.get('name')}")

        # Search for people at this organization
        people_response = requests.post(
            "https://api.apollo.io/api/v1/mixed_people/search",
            headers={"Content-Type": "application/json", "X-Api-Key": APOLLO_API_KEY},
            json={
                "q_organization_domains": [domain] if domain else None,
                "organization_ids": [organization.get("id")],
                "person_seniorities": ["director", "vp", "c_suite", "founder", "owner"],
                "page": 1,
                "per_page": 10
            },
            timeout=15
        )

        people_data = people_response.json()

        if not people_response.ok:
            return jsonify({"success": False, "error": "Failed to search contacts in Apollo"})

        people = people_data.get("people", [])
        print(f"Found {len(people)} contacts")

        contacts = [{
            "name": p.get("name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "title": p.get("title", "Unknown"),
            "email": p.get("email"),
            "phone": p.get("phone_numbers", [{}])[0].get("sanitized_number") if p.get("phone_numbers") else None,
            "linkedin": p.get("linkedin_url")
        } for p in people]

        return jsonify({
            "success": True,
            "data": {
                "domain": domain or organization.get("primary_domain"),
                "contacts": contacts,
                "apolloEnriched": True
            }
        })

    except Exception as e:
        print(f"Error enriching contacts: {e}")
        return jsonify({"success": False, "error": str(e)})


def clean_markdown_text(text: str) -> str:
    """Remove markdown formatting from text"""
    # Remove markdown links but keep link text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove markdown images
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
    # Remove bold/italic markers
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    # Remove headers markers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_company_data(markdown: str, links: list, domain: str) -> dict:
    """Extract comprehensive company data from scraped content"""
    data = {"domain": domain}

    # Unwanted patterns in names
    skip_patterns = [
        'skip to', 'sign in', 'sign up', 'log in', 'signed', 'click', 'chat with',
        'menu', 'navigation', 'footer', 'header', 'cookie', 'privacy'
    ]

    def is_valid_name(name):
        if not name or len(name) > 60 or len(name) < 2:
            return False
        name_lower = name.lower()
        if any(skip in name_lower for skip in skip_patterns):
            return False
        # Should start with a capital letter or number
        if not name[0].isupper() and not name[0].isdigit():
            return False
        return True

    # Use the domain name as the primary name (most reliable)
    domain_name = domain.split('.')[0]
    # Capitalize properly (stripe -> Stripe, openAI -> OpenAI)
    if domain_name.isupper() or domain_name.islower():
        data["name"] = domain_name.capitalize()
    else:
        data["name"] = domain_name

    # Try to find a better name from the content
    name = None

    # Pattern 1: Look for company name in meta-like patterns
    meta_match = re.search(r'(?:company|about|welcome to)\s+([A-Z][A-Za-z0-9\s&.]+?)(?:\s*[|\-.,]|\n|$)', markdown, re.IGNORECASE)
    if meta_match:
        candidate = clean_markdown_text(meta_match.group(1).strip())
        if is_valid_name(candidate):
            name = candidate

    # Pattern 2: H1 header (but be careful)
    if not name:
        title_match = re.search(r'^#\s+([A-Z][A-Za-z0-9\s&.]+?)(?:\s*[-|•·]|\n|$)', markdown, re.MULTILINE)
        if title_match:
            candidate = clean_markdown_text(title_match.group(1).strip())
            if is_valid_name(candidate):
                name = candidate

    # Only use extracted name if it's valid and different from domain
    if name and is_valid_name(name):
        data["name"] = name

    # Extract description (first substantial paragraph)
    paragraphs = markdown.split('\n\n')
    for p in paragraphs:
        cleaned = clean_markdown_text(re.sub(r'^[#*\-\s]+', '', p).strip())
        # Skip if it's a navigation or action text
        if any(skip in cleaned.lower() for skip in ['signed in', 'signed out', 'reload', 'click here', 'sign up', 'log in']):
            continue
        if 50 < len(cleaned) < 500 and not cleaned.startswith('['):
            data["description"] = cleaned
            break

    # Extract LinkedIn
    linkedin_match = re.search(r'linkedin\.com/company/([a-zA-Z0-9\-]+)', markdown, re.IGNORECASE)
    if linkedin_match:
        data["linkedin"] = f"https://linkedin.com/company/{linkedin_match.group(1)}"
    elif links:
        for link in links:
            if 'linkedin.com/company' in link:
                data["linkedin"] = link
                break

    # Extract Twitter/X
    twitter_match = re.search(r'(?:twitter\.com|x\.com)/([a-zA-Z0-9_]+)', markdown, re.IGNORECASE)
    if twitter_match:
        data["twitter"] = f"https://twitter.com/{twitter_match.group(1)}"
    elif links:
        for link in links:
            if 'twitter.com' in link or 'x.com' in link:
                data["twitter"] = link
                break

    # Extract Instagram
    instagram_match = re.search(r'instagram\.com/([a-zA-Z0-9_.]+)', markdown, re.IGNORECASE)
    if instagram_match:
        data["instagram"] = f"https://instagram.com/{instagram_match.group(1)}"
    elif links:
        for link in links:
            if 'instagram.com' in link:
                data["instagram"] = link
                break

    # Extract Facebook
    facebook_match = re.search(r'facebook\.com/([a-zA-Z0-9_.]+)', markdown, re.IGNORECASE)
    if facebook_match:
        data["facebook"] = f"https://facebook.com/{facebook_match.group(1)}"

    # Extract email
    email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', markdown)
    if email_match:
        email = email_match.group(1)
        if not any(x in email.lower() for x in ['example', 'noreply', 'wordpress', 'wixpress']):
            data["email"] = email

    # Extract phone
    phone_match = re.search(r'(?:\+1|1)?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', markdown)
    if phone_match:
        data["phone"] = phone_match.group(0)

    # Industry detection based on keywords
    industry_keywords = {
        'Technology': ['software', 'saas', 'tech', 'platform', 'api', 'cloud', 'ai', 'machine learning', 'data'],
        'E-commerce': ['ecommerce', 'e-commerce', 'shop', 'store', 'retail', 'marketplace', 'buy', 'sell'],
        'Finance': ['fintech', 'banking', 'payments', 'financial', 'insurance', 'invest'],
        'Healthcare': ['health', 'medical', 'healthcare', 'pharma', 'biotech', 'clinic'],
        'Marketing': ['marketing', 'advertising', 'agency', 'seo', 'digital marketing', 'branding'],
        'Education': ['education', 'learning', 'edtech', 'courses', 'training', 'school'],
        'Real Estate': ['real estate', 'property', 'realty', 'housing', 'homes'],
        'Manufacturing': ['manufacturing', 'factory', 'production', 'industrial'],
        'Consulting': ['consulting', 'consultancy', 'advisory', 'professional services'],
    }

    lower_markdown = markdown.lower()
    for industry, keywords in industry_keywords.items():
        if any(keyword in lower_markdown for keyword in keywords):
            data["industry"] = industry
            break

    # Extract technologies from common tech stack mentions
    tech_patterns = [
        'react', 'vue', 'angular', 'nextjs', 'nuxt', 'svelte',
        'node', 'python', 'java', 'ruby', 'go', 'rust', 'php', 'django', 'flask',
        'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'vercel', 'netlify',
        'postgresql', 'mongodb', 'mysql', 'redis', 'elasticsearch',
        'stripe', 'shopify', 'salesforce', 'hubspot', 'segment', 'mixpanel',
        'wordpress', 'webflow', 'squarespace', 'wix'
    ]
    found_tech = [tech for tech in tech_patterns if tech in lower_markdown]
    if found_tech:
        data["technologies"] = [t.capitalize() for t in list(set(found_tech))]

    # Extract location hints
    location_patterns = [
        r'(?:headquartered|located|based)\s+in\s+([A-Z][a-zA-Z\s,]+)',
        r'([A-Z][a-zA-Z]+,\s*[A-Z]{2})\s*\d{5}',  # City, ST ZIP
    ]
    for pattern in location_patterns:
        loc_match = re.search(pattern, markdown)
        if loc_match:
            data["location"] = loc_match.group(1).strip()
            break

    # Extract founded year
    founded_match = re.search(r'(?:founded|established|since)\s*(?:in\s*)?(\d{4})', markdown, re.IGNORECASE)
    if founded_match:
        data["founded"] = founded_match.group(1)

    # Collect social links
    social_links = [link for link in links if any(x in link for x in ['linkedin.com', 'twitter.com', 'x.com', 'facebook.com', 'instagram.com'])]
    if social_links:
        data["socialLinks"] = social_links

    return data


@app.route("/api/enrich", methods=["POST"])
def enrich_company():
    """Enrich a single company by domain"""
    data = request.json
    domain = data.get("domain", "").strip()

    if not domain:
        return jsonify({"success": False, "error": "Domain is required"})

    try:
        # Clean domain
        domain = domain.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
        result = {"domain": domain}

        # Use WebsiteScraper for web scraping (Firecrawl with fallback)
        if FIRECRAWL_API_KEY:
            url = f"https://{domain}"
            scraper = WebsiteScraper(FIRECRAWL_API_KEY)
            scrape_result = scraper.scrape(url)

            if scrape_result:
                markdown = scrape_result.get("markdown", "")
                links = scrape_result.get("links", [])

                # Use comprehensive extraction
                result = extract_company_data(markdown, links, domain)

        # Also try Apollo if available for additional data
        if APOLLO_API_KEY:
            apollo = ApolloClient(APOLLO_API_KEY)
            try:
                org = apollo.enrich_organization(domain)
                if org:
                    # Only override name if we don't have one or if Apollo has a better one
                    apollo_name = org.get("name")
                    if apollo_name and (not result.get("name") or result.get("name") == domain.split('.')[0].capitalize()):
                        result["name"] = apollo_name

                    result["industry"] = result.get("industry") or org.get("industry")
                    result["linkedin"] = result.get("linkedin") or org.get("linkedin_url")

                    # Employee count - convert to readable format
                    employees = org.get("estimated_num_employees")
                    if employees:
                        result["employees"] = f"{employees:,}" if isinstance(employees, int) else str(employees)

                    result["revenue"] = org.get("annual_revenue_printed")

                    # Location
                    city = org.get("city")
                    state = org.get("state")
                    country = org.get("country")
                    if city or state or country:
                        loc_parts = [p for p in [city, state, country] if p]
                        result["location"] = result.get("location") or ", ".join(loc_parts)

                    # Founded year
                    result["founded"] = result.get("founded") or org.get("founded_year")

                # Get contacts from Apollo
                contacts = apollo.search_contacts(domain, ["ceo", "founder", "owner", "director", "manager", "head", "vp", "president"])
                if contacts:
                    result["contacts"] = [{
                        "name": c.get("name", ""),
                        "title": c.get("title", ""),
                        "email": c.get("email", ""),
                        "phone": c.get("phone_numbers", [{}])[0].get("sanitized_number", "") if c.get("phone_numbers") else "",
                        "linkedin": c.get("linkedin_url", "")
                    } for c in contacts[:5]]
            except Exception as e:
                print(f"Apollo enrichment error: {e}")

        # Use Hunter.io for additional email finding
        if HUNTER_API_KEY:
            try:
                hunter = HunterClient(HUNTER_API_KEY)
                hunter_data = hunter.domain_search(domain, limit=5)

                if hunter_data:
                    # Get emails found by Hunter
                    hunter_emails = hunter_data.get("emails", [])
                    if hunter_emails:
                        # Add Hunter emails to existing contacts or create new ones
                        existing_emails = set()
                        if result.get("contacts"):
                            existing_emails = {c.get("email", "").lower() for c in result["contacts"] if c.get("email")}

                        for he in hunter_emails:
                            email = he.get("value", "")
                            if email and email.lower() not in existing_emails:
                                # Add as a new contact
                                if "contacts" not in result:
                                    result["contacts"] = []
                                result["contacts"].append({
                                    "name": f"{he.get('first_name', '')} {he.get('last_name', '')}".strip() or "Unknown",
                                    "title": he.get("position", ""),
                                    "email": email,
                                    "phone": he.get("phone_number", ""),
                                    "linkedin": he.get("linkedin", ""),
                                    "source": "hunter.io",
                                    "confidence": he.get("confidence", 0)
                                })
                                existing_emails.add(email.lower())

                    # Add organization info if not already present
                    if not result.get("name") and hunter_data.get("organization"):
                        result["name"] = hunter_data.get("organization")

                    result["hunter_email_count"] = hunter_data.get("total", 0)
            except Exception as e:
                print(f"Hunter.io enrichment error: {e}")

        return jsonify({"success": True, "data": result})

    except Exception as e:
        print(f"Enrichment error: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/enrich-batch", methods=["POST"])
def enrich_batch():
    """Enrich multiple companies in batch"""
    data = request.json
    domains = data.get("domains", [])

    if not domains or not isinstance(domains, list):
        return jsonify({"success": False, "error": "Domains array is required"})

    if not FIRECRAWL_API_KEY:
        return jsonify({"success": False, "error": "Firecrawl API not configured"})

    results = []
    for domain in domains:
        try:
            # Clean domain
            domain = domain.strip().replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
            if not domain:
                continue

            url = f"https://{domain}"
            firecrawl = FirecrawlClient(FIRECRAWL_API_KEY)
            scrape_result = firecrawl.scrape(url)

            if scrape_result:
                markdown = scrape_result.get("markdown", "")
                links = scrape_result.get("links", [])
                company_data = extract_company_data(markdown, links, domain)

                # Add Apollo data if available
                if APOLLO_API_KEY:
                    apollo = ApolloClient(APOLLO_API_KEY)
                    org = apollo.enrich_organization(domain)
                    if org:
                        company_data["name"] = company_data.get("name") or org.get("name")
                        company_data["employees"] = org.get("estimated_num_employees")
                        company_data["revenue"] = org.get("annual_revenue_printed")

                results.append(company_data)
            else:
                results.append({"domain": domain, "error": "Failed to scrape"})

        except Exception as e:
            results.append({"domain": domain, "error": str(e)})

    return jsonify({"success": True, "data": results})


@app.route("/api/hunter/domain-search", methods=["POST"])
def hunter_domain_search():
    """Search for emails at a domain using Hunter.io"""
    if not HUNTER_API_KEY:
        return jsonify({"success": False, "error": "Hunter.io API not configured"})

    data = request.json
    domain = data.get("domain", "").strip()

    if not domain:
        return jsonify({"success": False, "error": "Domain is required"})

    # Clean domain
    domain = domain.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")

    try:
        hunter = HunterClient(HUNTER_API_KEY)
        result = hunter.domain_search(domain, limit=data.get("limit", 10))

        if result:
            return jsonify({
                "success": True,
                "data": {
                    "domain": domain,
                    "organization": result.get("organization"),
                    "emails": result.get("emails", []),
                    "total": result.get("total", 0),
                    "pattern": result.get("pattern")
                }
            })
        else:
            return jsonify({"success": False, "error": "No results found"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/hunter/email-finder", methods=["POST"])
def hunter_email_finder():
    """Find a specific person's email using Hunter.io"""
    if not HUNTER_API_KEY:
        return jsonify({"success": False, "error": "Hunter.io API not configured"})

    data = request.json
    domain = data.get("domain", "").strip()
    first_name = data.get("first_name", "").strip()
    last_name = data.get("last_name", "").strip()

    if not domain:
        return jsonify({"success": False, "error": "Domain is required"})

    # Clean domain
    domain = domain.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")

    try:
        hunter = HunterClient(HUNTER_API_KEY)
        result = hunter.email_finder(domain, first_name, last_name)

        if result:
            return jsonify({
                "success": True,
                "data": {
                    "email": result.get("email"),
                    "confidence": result.get("score", 0),
                    "first_name": result.get("first_name"),
                    "last_name": result.get("last_name"),
                    "position": result.get("position"),
                    "domain": domain
                }
            })
        else:
            return jsonify({"success": False, "error": "Email not found"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/hunter/verify", methods=["POST"])
def hunter_verify_email():
    """Verify an email address using Hunter.io"""
    if not HUNTER_API_KEY:
        return jsonify({"success": False, "error": "Hunter.io API not configured"})

    data = request.json
    email = data.get("email", "").strip()

    if not email:
        return jsonify({"success": False, "error": "Email is required"})

    try:
        hunter = HunterClient(HUNTER_API_KEY)
        result = hunter.email_verifier(email)

        if result:
            return jsonify({
                "success": True,
                "data": {
                    "email": email,
                    "status": result.get("status"),  # valid, invalid, accept_all, webmail, disposable, unknown
                    "result": result.get("result"),  # deliverable, undeliverable, risky, unknown
                    "score": result.get("score", 0),
                    "regexp": result.get("regexp"),
                    "gibberish": result.get("gibberish"),
                    "disposable": result.get("disposable"),
                    "webmail": result.get("webmail"),
                    "mx_records": result.get("mx_records"),
                    "smtp_server": result.get("smtp_server"),
                    "smtp_check": result.get("smtp_check"),
                    "accept_all": result.get("accept_all"),
                    "block": result.get("block")
                }
            })
        else:
            return jsonify({"success": False, "error": "Verification failed"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🔥 LeadFlow")
    print("="*50)
    print("\nOpen http://localhost:5001 in your browser")
    print("\nAPI Keys configured:")
    print(f"  Google Places: {'✓' if GOOGLE_PLACES_API_KEY else '✗ (required)'}")
    print(f"  Apollo.io:     {'✓' if APOLLO_API_KEY else '✗ (optional)'}")
    print(f"  Firecrawl:     {'✓' if FIRECRAWL_API_KEY else '✗ (optional)'}")
    print(f"  Hunter.io:     {'✓' if HUNTER_API_KEY else '✗ (email finding)'}")
    print(f"  OpenAI:        {'✓' if OPENAI_API_KEY else '✗ (for GTM Brief)'}")
    print(f"  Instantly:     {'✓' if INSTANTLY_API_KEY else '✗ (for outreach)'}")
    print("\n" + "="*50 + "\n")

    app.run(debug=True, port=5001, host='0.0.0.0')
