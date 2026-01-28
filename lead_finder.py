"""
GTM Lead Finder - Main Tool
Finds and enriches retail partner leads from Google Maps for fashion brands.
"""

import json
import time
import re
import csv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml
import requests

from config import (
    GOOGLE_PLACES_API_KEY,
    APOLLO_API_KEY,
    HUNTER_API_KEY,
    OUTPUT_DIR,
)


@dataclass
class Lead:
    """Structured lead data"""
    # Identifiers
    place_id: str
    google_maps_url: str

    # Basic Info
    name: str
    category: str
    subcategories: list = field(default_factory=list)

    # Location
    address_full: str = ""
    street: str = ""
    city: str = ""
    country: str = ""
    postal_code: str = ""
    lat: float = 0.0
    lng: float = 0.0

    # Contact (from Google)
    phone: str = ""
    website: str = ""

    # Reputation
    rating: float = 0.0
    reviews_count: int = 0

    # Operations
    hours: dict = field(default_factory=dict)
    price_level: int = 0

    # Enrichment: Website scraping
    emails_found: list = field(default_factory=list)
    social_links: dict = field(default_factory=dict)
    brands_carried: list = field(default_factory=list)
    about_text: str = ""

    # Enrichment: Apollo
    apollo_org_id: str = ""
    company_size: str = ""
    estimated_revenue: str = ""
    linkedin_url: str = ""
    decision_makers: list = field(default_factory=list)

    # Scoring
    fit_score: int = 0  # 0-100
    fit_reasons: list = field(default_factory=list)

    # Metadata
    scraped_at: str = ""
    enriched_at: str = ""
    source_query: str = ""


class GooglePlacesClient:
    """Google Places API client for finding stores"""

    BASE_URL = "https://maps.googleapis.com/maps/api/place"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def text_search(self, query: str, location: str = None) -> list[dict]:
        """
        Search for places using text query.
        Returns list of place results.
        """
        url = f"{self.BASE_URL}/textsearch/json"
        params = {
            "query": query,
            "key": self.api_key,
        }
        if location:
            params["query"] = f"{query} in {location}"

        results = []
        next_page_token = None

        while True:
            if next_page_token:
                params["pagetoken"] = next_page_token
                time.sleep(2)  # Required delay for pagination

            response = requests.get(url, params=params)
            data = response.json()

            if data.get("status") != "OK":
                print(f"  Search error: {data.get('status')} - {data.get('error_message', '')}")
                break

            results.extend(data.get("results", []))

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

        return results

    def get_place_details(self, place_id: str) -> dict:
        """Get detailed information about a place"""
        url = f"{self.BASE_URL}/details/json"
        params = {
            "place_id": place_id,
            "key": self.api_key,
            "fields": ",".join([
                "place_id",
                "name",
                "formatted_address",
                "formatted_phone_number",
                "international_phone_number",
                "website",
                "url",
                "rating",
                "user_ratings_total",
                "price_level",
                "opening_hours",
                "types",
                "address_components",
                "geometry",
            ])
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("status") == "OK":
            return data.get("result", {})
        return {}

    def nearby_search(self, lat: float, lng: float, radius: int = 5000,
                      keyword: str = None, place_type: str = "store") -> list[dict]:
        """Search for places near a location"""
        url = f"{self.BASE_URL}/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "type": place_type,
            "key": self.api_key,
        }
        if keyword:
            params["keyword"] = keyword

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("status") == "OK":
            return data.get("results", [])
        return []


class ApolloClient:
    """Apollo.io API client for B2B enrichment"""

    BASE_URL = "https://api.apollo.io/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }

    def enrich_organization(self, domain: str) -> dict:
        """
        Enrich company data from domain.
        Returns company info including size, industry, contacts.
        """
        url = f"{self.BASE_URL}/organizations/enrich"
        params = {
            "api_key": self.api_key,
            "domain": domain,
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            return data.get("organization", {})
        except Exception as e:
            print(f"  Apollo error for {domain}: {e}")
            return {}

    def search_contacts(self, domain: str, titles: list = None) -> list[dict]:
        """
        Search for contacts at a company.
        Useful for finding decision makers (owners, buyers, managers).
        """
        url = f"{self.BASE_URL}/mixed_people/search"

        data = {
            "api_key": self.api_key,
            "q_organization_domains": domain,
            "page": 1,
            "per_page": 10,
        }

        if titles:
            data["person_titles"] = titles

        try:
            response = requests.post(url, headers=self.headers, json=data)
            result = response.json()
            return result.get("people", [])
        except Exception as e:
            print(f"  Apollo contacts error for {domain}: {e}")
            return []


class WebsiteScraper:
    """Simple website scraper for contact info and brand lists"""

    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )

    SOCIAL_PATTERNS = {
        "instagram": re.compile(r'instagram\.com/([a-zA-Z0-9_.]+)'),
        "facebook": re.compile(r'facebook\.com/([a-zA-Z0-9_.]+)'),
        "linkedin": re.compile(r'linkedin\.com/company/([a-zA-Z0-9_-]+)'),
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })

    def scrape(self, url: str) -> dict:
        """
        Scrape a website for contact info and brand mentions.
        Returns dict with emails, social links, and text content.
        """
        result = {
            "emails": [],
            "social_links": {},
            "brands": [],
            "about_text": "",
        }

        try:
            # Try main page and common contact pages
            pages_to_try = [
                url,
                f"{url.rstrip('/')}/contact",
                f"{url.rstrip('/')}/kontakt",  # Polish
                f"{url.rstrip('/')}/about",
                f"{url.rstrip('/')}/o-nas",    # Polish: about us
                f"{url.rstrip('/')}/brands",
                f"{url.rstrip('/')}/marki",    # Polish: brands
            ]

            all_text = ""

            for page_url in pages_to_try:
                try:
                    resp = self.session.get(page_url, timeout=10)
                    if resp.status_code == 200:
                        all_text += resp.text + "\n"
                except:
                    continue

            # Extract emails
            emails = list(set(self.EMAIL_PATTERN.findall(all_text)))
            # Filter out common non-contact emails
            result["emails"] = [
                e for e in emails
                if not any(x in e.lower() for x in ['example', 'test', 'noreply', 'wordpress', 'wix'])
            ]

            # Extract social links
            for platform, pattern in self.SOCIAL_PATTERNS.items():
                matches = pattern.findall(all_text)
                if matches:
                    result["social_links"][platform] = matches[0]

            # Could add brand detection here with a list of known brands

        except Exception as e:
            print(f"  Scrape error for {url}: {e}")

        return result


class HunterClient:
    """Hunter.io API for email finding"""

    BASE_URL = "https://api.hunter.io/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def domain_search(self, domain: str) -> list[dict]:
        """Find all emails for a domain"""
        if not self.api_key:
            return []

        url = f"{self.BASE_URL}/domain-search"
        params = {
            "domain": domain,
            "api_key": self.api_key,
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()
            return data.get("data", {}).get("emails", [])
        except:
            return []


class LeadScorer:
    """Score leads based on fit with client brief"""

    def __init__(self, brief: dict):
        self.brief = brief
        self.positive_signals = brief.get("filters", {}).get("positive_signals", [])
        self.negative_signals = brief.get("filters", {}).get("negative_signals", [])
        self.exclude_brands = [b.lower() for b in brief.get("exclude_brands", [])]
        self.min_rating = brief.get("filters", {}).get("min_rating", 0)
        self.min_reviews = brief.get("filters", {}).get("min_reviews", 0)
        self.target_segment = brief.get("target", {}).get("segment", "").lower()

    def score(self, lead: Lead) -> tuple[int, list[str]]:
        """
        Score a lead 0-100 based on fit.
        Returns (score, list of reasons).
        """
        score = 50  # Base score
        reasons = []

        # Check for exclusions first
        name_lower = lead.name.lower()
        for excluded in self.exclude_brands:
            if excluded in name_lower:
                return 0, [f"Excluded brand: {excluded}"]

        # Rating bonus
        if lead.rating >= 4.5:
            score += 15
            reasons.append(f"Excellent rating: {lead.rating}")
        elif lead.rating >= 4.0:
            score += 10
            reasons.append(f"Good rating: {lead.rating}")
        elif lead.rating < self.min_rating and lead.rating > 0:
            score -= 20
            reasons.append(f"Low rating: {lead.rating}")

        # Reviews indicate established business
        if lead.reviews_count >= 100:
            score += 10
            reasons.append(f"Well-established ({lead.reviews_count} reviews)")
        elif lead.reviews_count >= 50:
            score += 5
            reasons.append(f"Established ({lead.reviews_count} reviews)")
        elif lead.reviews_count < self.min_reviews and lead.reviews_count > 0:
            score -= 10
            reasons.append(f"Few reviews: {lead.reviews_count}")

        # Has website (essential for B2B outreach)
        if lead.website:
            score += 10
            reasons.append("Has website")
        else:
            score -= 15
            reasons.append("No website found")

        # Has phone
        if lead.phone:
            score += 5
            reasons.append("Has phone")

        # Contact info found through enrichment
        if lead.emails_found:
            score += 10
            reasons.append(f"Email found: {lead.emails_found[0]}")

        if lead.decision_makers:
            score += 10
            reasons.append(f"Decision maker found: {lead.decision_makers[0].get('name', 'Unknown')}")

        # Positive signals in name or category
        searchable_text = f"{lead.name} {lead.category} {' '.join(lead.subcategories)} {lead.about_text}".lower()

        for signal in self.positive_signals:
            if signal.lower() in searchable_text:
                score += 5
                reasons.append(f"Positive signal: {signal}")

        # Negative signals
        for signal in self.negative_signals:
            if signal.lower() in searchable_text:
                score -= 10
                reasons.append(f"Negative signal: {signal}")

        # Price level match (for mid-premium, we want 2-3)
        if self.target_segment == "mid-premium":
            if lead.price_level in [2, 3]:
                score += 10
                reasons.append(f"Price level matches ({lead.price_level})")
            elif lead.price_level == 1:
                score -= 5
                reasons.append("May be too budget-focused")
            elif lead.price_level == 4:
                score += 5
                reasons.append("Premium positioning")

        # Social presence
        if lead.social_links.get("instagram"):
            score += 5
            reasons.append("Has Instagram presence")

        # Cap score
        score = max(0, min(100, score))

        return score, reasons


class GTMLeadFinder:
    """Main orchestrator for the GTM Lead Finding process"""

    def __init__(self, brief_path: str):
        with open(brief_path, 'r') as f:
            self.brief = yaml.safe_load(f)

        self.google = GooglePlacesClient(GOOGLE_PLACES_API_KEY)
        self.apollo = ApolloClient(APOLLO_API_KEY)
        self.scraper = WebsiteScraper()
        self.hunter = HunterClient(HUNTER_API_KEY)
        self.scorer = LeadScorer(self.brief)

        self.leads: dict[str, Lead] = {}  # Keyed by place_id to dedupe

        # Create output directory
        Path(OUTPUT_DIR).mkdir(exist_ok=True)

    def run(self, enrich: bool = True, limit: int = None):
        """
        Run the full lead finding pipeline.

        Args:
            enrich: Whether to enrich leads (adds API calls)
            limit: Max leads to process (for testing)
        """
        print(f"\n{'='*60}")
        print(f"GTM Lead Finder - {self.brief['client']['brand_name']}")
        print(f"{'='*60}\n")

        # Step 1: Search Google Maps
        print("STEP 1: Searching Google Maps")
        print("-" * 40)
        self._search_google_maps()

        if limit:
            # Limit for testing
            place_ids = list(self.leads.keys())[:limit]
            self.leads = {pid: self.leads[pid] for pid in place_ids}

        print(f"\nFound {len(self.leads)} unique places\n")

        # Step 2: Get detailed info from Google
        print("STEP 2: Getting place details")
        print("-" * 40)
        self._get_place_details()

        # Step 3: Enrich (optional but recommended)
        if enrich:
            print("\nSTEP 3: Enriching leads")
            print("-" * 40)
            self._enrich_leads()

        # Step 4: Score leads
        print("\nSTEP 4: Scoring leads")
        print("-" * 40)
        self._score_leads()

        # Step 5: Export
        print("\nSTEP 5: Exporting results")
        print("-" * 40)
        self._export_results()

        # Summary
        self._print_summary()

    def _search_google_maps(self):
        """Execute all search queries from the brief"""
        queries = self.brief.get("search_queries", [])

        for query in queries:
            print(f"  Searching: '{query}'")
            results = self.google.text_search(query)

            for place in results:
                place_id = place.get("place_id")
                if place_id and place_id not in self.leads:
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
                        price_level=place.get("price_level", 0),
                        scraped_at=datetime.now().isoformat(),
                        source_query=query,
                    )
                    self.leads[place_id] = lead

            print(f"    Found {len(results)} results, {len(self.leads)} unique total")
            time.sleep(0.1)  # Small delay between queries

    def _get_place_details(self):
        """Get detailed info for each lead"""
        total = len(self.leads)

        for i, (place_id, lead) in enumerate(self.leads.items(), 1):
            print(f"  [{i}/{total}] {lead.name[:40]}...")

            details = self.google.get_place_details(place_id)

            if details:
                lead.phone = details.get("formatted_phone_number", "") or details.get("international_phone_number", "")
                lead.website = details.get("website", "")
                lead.google_maps_url = details.get("url", lead.google_maps_url)

                # Parse address components
                for component in details.get("address_components", []):
                    types = component.get("types", [])
                    if "street_number" in types or "route" in types:
                        lead.street += component.get("long_name", "") + " "
                    elif "locality" in types:
                        lead.city = component.get("long_name", "")
                    elif "country" in types:
                        lead.country = component.get("long_name", "")
                    elif "postal_code" in types:
                        lead.postal_code = component.get("long_name", "")

                lead.street = lead.street.strip()

                # Opening hours
                if details.get("opening_hours"):
                    lead.hours = {
                        "periods": details["opening_hours"].get("periods", []),
                        "weekday_text": details["opening_hours"].get("weekday_text", []),
                    }

                # More specific types
                lead.subcategories = details.get("types", [])

            time.sleep(0.05)  # Rate limiting

    def _enrich_leads(self):
        """Enrich leads with website scraping and Apollo data"""
        total = len(self.leads)

        for i, (place_id, lead) in enumerate(self.leads.items(), 1):
            if not lead.website:
                continue

            print(f"  [{i}/{total}] Enriching {lead.name[:40]}...")

            # Extract domain from website
            domain = self._extract_domain(lead.website)
            if not domain:
                continue

            # Website scraping
            scrape_result = self.scraper.scrape(lead.website)
            lead.emails_found = scrape_result.get("emails", [])
            lead.social_links = scrape_result.get("social_links", {})
            lead.about_text = scrape_result.get("about_text", "")

            # Hunter.io for more emails
            if HUNTER_API_KEY:
                hunter_emails = self.hunter.domain_search(domain)
                for email_data in hunter_emails:
                    email = email_data.get("value")
                    if email and email not in lead.emails_found:
                        lead.emails_found.append(email)

            # Apollo enrichment
            if APOLLO_API_KEY:
                org_data = self.apollo.enrich_organization(domain)
                if org_data:
                    lead.apollo_org_id = org_data.get("id", "")
                    lead.company_size = org_data.get("estimated_num_employees", "")
                    lead.estimated_revenue = org_data.get("annual_revenue_printed", "")
                    lead.linkedin_url = org_data.get("linkedin_url", "")

                # Find decision makers
                contacts = self.apollo.search_contacts(
                    domain,
                    titles=["owner", "founder", "director", "manager", "buyer", "purchasing"]
                )
                for contact in contacts[:3]:  # Top 3
                    lead.decision_makers.append({
                        "name": contact.get("name", ""),
                        "title": contact.get("title", ""),
                        "email": contact.get("email", ""),
                        "linkedin": contact.get("linkedin_url", ""),
                    })

            lead.enriched_at = datetime.now().isoformat()
            time.sleep(0.2)  # Rate limiting

    def _score_leads(self):
        """Score all leads"""
        for lead in self.leads.values():
            score, reasons = self.scorer.score(lead)
            lead.fit_score = score
            lead.fit_reasons = reasons
            print(f"  {lead.name[:40]}: {score}/100")

    def _export_results(self):
        """Export results to CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        brand_slug = self.brief['client']['brand_name'].lower().replace(" ", "_")

        # Sort by score
        sorted_leads = sorted(
            self.leads.values(),
            key=lambda x: x.fit_score,
            reverse=True
        )

        # CSV Export (for easy use in spreadsheets)
        csv_path = Path(OUTPUT_DIR) / f"{brand_slug}_leads_{timestamp}.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Score", "Name", "City", "Address", "Phone", "Website",
                "Email", "Rating", "Reviews", "Instagram", "LinkedIn",
                "Decision Maker", "DM Email", "Fit Reasons", "Google Maps URL"
            ])

            for lead in sorted_leads:
                dm = lead.decision_makers[0] if lead.decision_makers else {}
                writer.writerow([
                    lead.fit_score,
                    lead.name,
                    lead.city,
                    lead.address_full,
                    lead.phone,
                    lead.website,
                    lead.emails_found[0] if lead.emails_found else "",
                    lead.rating,
                    lead.reviews_count,
                    lead.social_links.get("instagram", ""),
                    lead.linkedin_url,
                    dm.get("name", ""),
                    dm.get("email", ""),
                    "; ".join(lead.fit_reasons),
                    lead.google_maps_url,
                ])

        print(f"  CSV: {csv_path}")

        # JSON Export (full data)
        json_path = Path(OUTPUT_DIR) / f"{brand_slug}_leads_{timestamp}.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(lead) for lead in sorted_leads],
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"  JSON: {json_path}")

    def _print_summary(self):
        """Print summary of results"""
        sorted_leads = sorted(
            self.leads.values(),
            key=lambda x: x.fit_score,
            reverse=True
        )

        high_fit = [l for l in sorted_leads if l.fit_score >= 70]
        medium_fit = [l for l in sorted_leads if 50 <= l.fit_score < 70]
        low_fit = [l for l in sorted_leads if l.fit_score < 50]

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total leads found: {len(self.leads)}")
        print(f"  High fit (70+):   {len(high_fit)}")
        print(f"  Medium fit (50-69): {len(medium_fit)}")
        print(f"  Low fit (<50):    {len(low_fit)}")

        if high_fit:
            print(f"\nTOP 10 LEADS:")
            print("-" * 60)
            for i, lead in enumerate(high_fit[:10], 1):
                email = lead.emails_found[0] if lead.emails_found else "No email"
                print(f"  {i}. [{lead.fit_score}] {lead.name}")
                print(f"      {lead.city} | {lead.phone} | {email}")

        print(f"\n{'='*60}\n")

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            domain = domain.replace("www.", "")
            return domain
        except:
            return ""


if __name__ == "__main__":
    import sys

    brief_path = sys.argv[1] if len(sys.argv) > 1 else "brief_template.yaml"

    finder = GTMLeadFinder(brief_path)
    finder.run(enrich=True, limit=None)
