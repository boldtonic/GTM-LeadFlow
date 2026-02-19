"""
Background search pipeline for GTM LeadFlow.
Orchestrates: Google Places search → details → enrichment → scoring.
"""

import time
from dataclasses import asdict
from datetime import datetime

from api_clients import ApolloClient, GooglePlacesClient, HunterClient
from api_clients.scraper import WebsiteScraper
from models import Lead
from scoring import LeadScorer
from services.enrichment import compute_enrichment_signals
from utils import extract_domain, is_business_domain, log_dev


def run_search(job_id: str, config: dict, api_keys: dict, job_mgr):
    """Run the full search pipeline in a background thread."""
    job_mgr.create(
        job_id,
        {
            "status": "running",
            "step": "Initializing...",
            "progress": 0,
            "leads": [],
            "errors": [],
            "cancelled": False,
            "enrichment_stats": {},
            "max_leads": config.get("max_leads", 50),
        },
    )

    try:
        clients = _init_clients(api_keys)
        queries = config.get("queries", [])
        max_leads = config.get("max_leads", 50)
        target_titles = config.get("target_titles", ["owner", "founder", "director", "manager", "buyer"])

        log_dev("SEARCH", f"Starting search job {job_id} with {len(queries)} queries", "info")
        log_dev("SEARCH", f"Max leads: {max_leads}, Enrich: {config.get('enrich', True)}", "info")
        if target_titles != ["owner", "founder", "director", "manager", "buyer"]:
            log_dev("SEARCH", f"Custom target titles: {', '.join(target_titles)}", "info")
        log_dev(
            "CONFIG",
            f"APIs: Google={'✓' if api_keys.get('google_places') else '✗'}, "
            f"Apollo={'✓' if api_keys.get('apollo') else '✗'}, "
            f"Firecrawl={'✓' if api_keys.get('firecrawl') else '✗'}, "
            f"Hunter={'✓' if api_keys.get('hunter') else '✗'}",
            "info",
        )

        # Step 1: Search Google Places
        leads = _step_search(job_id, queries, max_leads, clients["google"], job_mgr)
        if not leads or job_mgr.is_cancelled(job_id):
            if not leads and not job_mgr.is_cancelled(job_id):
                job_mgr.update(job_id, status="completed", step="No results found", progress=100)
            return

        # Step 2: Get place details
        _step_details(job_id, leads, clients["google"], job_mgr)
        if job_mgr.is_cancelled(job_id):
            return

        # Step 3: Enrich (or just compute lightweight signals)
        if config.get("enrich", True):
            _step_enrich(job_id, leads, clients, target_titles, job_mgr)
            if job_mgr.is_cancelled(job_id):
                return
        else:
            # Even without enrichment, compute free signals (email count, grade)
            hunter = clients["hunter"]
            for lead in leads.values():
                if lead.domain:
                    compute_enrichment_signals(lead, hunter)

        # Step 4: Score and finalize
        _step_score(job_id, leads, config, job_mgr)

    except Exception as e:
        job_mgr.update(job_id, status="error", step=f"Error: {str(e)}", errors=[str(e)])


def _init_clients(api_keys: dict) -> dict:
    """Create API client instances from keys."""
    return {
        "google": GooglePlacesClient(api_keys.get("google_places", ""), log_fn=log_dev),
        "apollo": ApolloClient(api_keys.get("apollo", ""), log_fn=log_dev),
        "scraper": WebsiteScraper(api_keys.get("firecrawl", ""), log_fn=log_dev),
        "hunter": HunterClient(api_keys.get("hunter", ""), log_fn=log_dev),
    }


def _step_search(job_id, queries, max_leads, google, job_mgr):
    """Step 1: Search Google Places for leads."""
    job_mgr.update(job_id, step="Searching Google Maps...")
    log_dev("GOOGLE", f"Starting Google Places search with {len(queries)} queries", "info")

    leads = {}
    total_queries = len(queries)

    for i, query in enumerate(queries):
        if job_mgr.is_cancelled(job_id):
            job_mgr.update(job_id, status="cancelled", step="Cancelled by user")
            log_dev("SEARCH", "Search cancelled by user", "warning")
            return {}

        job_mgr.update(job_id, progress=int((i / total_queries) * 30), step=f"Searching: {query[:40]}...")

        results = google.text_search(query)
        log_dev("GOOGLE", f"Query '{query[:30]}...' returned {len(results)} results", "info")

        for place in results:
            place_id = place.get("place_id")
            if place_id and place_id not in leads:
                leads[place_id] = Lead(
                    place_id=place_id,
                    google_maps_url=f"https://www.google.com/maps/place/?q=place_id:{place_id}",
                    name=place.get("name", ""),
                    category=", ".join(place.get("types", [])[:2]),
                    address_full=place.get("formatted_address", ""),
                    rating=place.get("rating", 0),
                    reviews_count=place.get("user_ratings_total", 0),
                    lat=place.get("geometry", {}).get("location", {}).get("lat", 0),
                    lng=place.get("geometry", {}).get("location", {}).get("lng", 0),
                    source="google_places",
                    scraped_at=datetime.now().isoformat(),
                    source_query=query,
                )

        if len(leads) >= max_leads:
            log_dev("SEARCH", f"Reached max leads cap ({max_leads}), stopping search", "info")
            break

        time.sleep(0.2)

    # Enforce max_leads cap
    if len(leads) > max_leads:
        lead_keys = list(leads.keys())[:max_leads]
        leads = {k: leads[k] for k in lead_keys}
        log_dev("SEARCH", f"Truncated to {max_leads} leads", "info")

    return leads


def _step_details(job_id, leads, google, job_mgr):
    """Step 2: Get detailed info for each place."""
    job_mgr.update(job_id, step="Getting place details...")
    lead_list = list(leads.values())
    total = len(lead_list)

    for i, lead in enumerate(lead_list):
        if job_mgr.is_cancelled(job_id):
            job_mgr.update(job_id, status="cancelled", step="Cancelled by user")
            return

        job_mgr.update(job_id, progress=30 + int((i / total) * 30), step=f"Details: {lead.name[:30]}...")

        details = google.get_place_details(lead.place_id)
        if details:
            lead.phone = details.get("formatted_phone_number", "") or details.get("international_phone_number", "")
            lead.website = details.get("website", "")
            if lead.website:
                lead.domain = extract_domain(lead.website)
            lead.google_maps_url = details.get("url", lead.google_maps_url)
            lead.subcategories = details.get("types", [])

            for comp in details.get("address_components", []):
                types = comp.get("types", [])
                if "locality" in types:
                    lead.city = comp.get("long_name", "")
                elif "country" in types:
                    lead.country = comp.get("long_name", "")

        time.sleep(0.05)


def _step_enrich(job_id, leads, clients, target_titles, job_mgr):
    """Step 3: Enrich leads with website scraping, Apollo, and Hunter."""
    job_mgr.update(job_id, step="Enriching leads...")
    lead_list = list(leads.values())
    total = len(lead_list)
    log_dev("ENRICH", f"Starting enrichment for {total} leads", "info")

    apollo = clients["apollo"]
    scraper = clients["scraper"]
    hunter = clients["hunter"]

    stats = {
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
        if job_mgr.is_cancelled(job_id):
            job_mgr.update(job_id, status="cancelled", step="Cancelled by user")
            log_dev("ENRICH", "Enrichment cancelled by user", "warning")
            return

        if not lead.website:
            stats["no_website"] += 1
            continue

        job_mgr.update(job_id, progress=60 + int((i / total) * 30), step=f"Enriching: {lead.name[:30]}...")

        domain = extract_domain(lead.website)
        if not domain:
            continue

        # Website scraping
        scrape_result = scraper.scrape(lead.website)
        lead.emails_found = scrape_result.get("emails", [])
        lead.social_links = scrape_result.get("social_links", {})
        if scrape_result.get("description") and not lead.about_text:
            lead.about_text = scrape_result["description"][:500]

        if scrape_result.get("scrape_method") == "firecrawl":
            stats["firecrawl_success"] += 1
        else:
            stats["basic_scrape"] += 1

        if lead.emails_found:
            stats["emails_found"] += 1
        if lead.social_links:
            stats["social_found"] += 1

        biz_domain = is_business_domain(domain)

        # Apollo enrichment
        if apollo.is_configured and biz_domain:
            _enrich_apollo(lead, domain, target_titles, apollo, stats)

        # Hunter email_finder for Apollo contacts without emails
        if hunter.is_configured and biz_domain:
            _find_emails_for_contacts(lead, domain, hunter, stats)

        # Hunter email enrichment
        if hunter.is_configured and biz_domain:
            _enrich_hunter(lead, domain, hunter, stats)

        # Enrichment signals (email count is free)
        compute_enrichment_signals(lead, hunter)

        lead.enriched_at = datetime.now().isoformat()
        time.sleep(0.1)

        if (i + 1) % 50 == 0:
            log_dev("ENRICH", f"Progress: {i + 1}/{total} leads enriched", "info")

    job_mgr.update(job_id, enrichment_stats=stats)
    log_dev(
        "ENRICH",
        f"Enrichment complete: Firecrawl={stats['firecrawl_success']}, "
        f"Basic={stats['basic_scrape']}, Apollo={stats['apollo_org_success']}, "
        f"Hunter={stats['hunter_emails_found']} emails/{stats['hunter_verified']} verified",
        "success",
    )


def _enrich_apollo(lead, domain, target_titles, apollo, stats):
    """Enrich a single lead with Apollo org + contacts data."""
    try:
        org = apollo.enrich_organization(domain)
        if org:
            lead.company_size = str(org.get("estimated_num_employees", ""))
            lead.linkedin_url = org.get("linkedin_url", "")
            lead.estimated_revenue = lead.estimated_revenue or (org.get("annual_revenue_printed") or "")
            lead.industry = lead.industry or (org.get("industry") or "")
            lead.founded_year = lead.founded_year or str(org.get("founded_year") or "")
            stats["apollo_org_success"] += 1
        else:
            stats["apollo_org_failed"] += 1
    except Exception as e:
        stats["apollo_org_failed"] += 1
        log_dev("APOLLO", f"Org enrich failed for {domain}: {str(e)[:50]}", "error")

    try:
        contacts = apollo.search_contacts(domain, titles=target_titles)
        for contact in contacts[:3]:
            first = contact.get("first_name", "")
            last = contact.get("last_name", "")
            full = contact.get("name", "") or f"{first} {last}".strip()
            lead.decision_makers.append(
                {
                    "name": full,
                    "first_name": first,
                    "last_name": last,
                    "title": contact.get("title", ""),
                    "email": contact.get("email", ""),
                    "linkedin": contact.get("linkedin_url", ""),
                    "seniority": contact.get("seniority", ""),
                    "phone": (contact.get("phone_numbers") or [{}])[0].get("sanitized_number", "")
                    if contact.get("phone_numbers")
                    else "",
                    "source": "apollo",
                }
            )
        if contacts:
            stats["apollo_contacts_found"] += 1
    except Exception as e:
        log_dev("APOLLO", f"Contact search failed for {domain}: {str(e)[:50]}", "error")


def _find_emails_for_contacts(lead, domain, hunter, stats):
    """Use Hunter email_finder to find emails for Apollo contacts that lack them."""
    for dm in lead.decision_makers:
        if dm.get("email") or dm.get("source") != "apollo":
            continue
        first = dm.get("first_name", "")
        last = dm.get("last_name", "")
        if not first or not last:
            continue
        try:
            result = hunter.email_finder(domain, first_name=first, last_name=last)
            if result and result.get("email"):
                email = result["email"]
                dm["email"] = email
                dm["confidence"] = result.get("score", 0)
                dm["email_source"] = "hunter_finder"
                if email.lower() not in {e.lower() for e in lead.emails_found}:
                    lead.emails_found.append(email)
                stats["hunter_emails_found"] += 1
                log_dev("HUNTER", f"email_finder: {first} {last}@{domain} → {email}", "success")
        except Exception as e:
            log_dev("HUNTER", f"email_finder failed for {first} {last}@{domain}: {str(e)[:50]}", "error")


def _enrich_hunter(lead, domain, hunter, stats):
    """Enrich a single lead with Hunter email data + verification."""
    try:
        hunter_data = hunter.domain_search(domain, limit=5)
        if hunter_data:
            hunter_emails = hunter_data.get("emails", [])
            existing_emails = {e.lower() for e in lead.emails_found}
            for dm in lead.decision_makers:
                if dm.get("email"):
                    existing_emails.add(dm["email"].lower())

            for he in hunter_emails:
                email = he.get("value", "")
                if email and email.lower() not in existing_emails:
                    lead.emails_found.append(email)
                    existing_emails.add(email.lower())
                    first = he.get("first_name", "")
                    last = he.get("last_name", "")
                    name = f"{first} {last}".strip()
                    if name and he.get("position"):
                        lead.decision_makers.append(
                            {
                                "name": name,
                                "first_name": first,
                                "last_name": last,
                                "title": he.get("position", ""),
                                "email": email,
                                "linkedin": he.get("linkedin", ""),
                                "phone": he.get("phone_number", ""),
                                "source": "hunter",
                                "confidence": he.get("confidence", 0),
                                "email_type": he.get("type", ""),
                            }
                        )

            if hunter_emails:
                stats["hunter_emails_found"] += len(hunter_emails)
                log_dev("HUNTER", f"Found {len(hunter_emails)} emails for {domain}", "info")
    except Exception as e:
        log_dev("HUNTER", f"Domain search failed for {domain}: {str(e)[:50]}", "error")

    # Verify all emails (up to 5), prioritize personal over generic
    _verify_all_emails(lead, hunter, stats)


def _verify_all_emails(lead, hunter, stats):
    """Verify all found emails (up to 5). Prioritize personal emails first."""
    if not lead.emails_found:
        return

    dm_emails = {dm.get("email", "").lower() for dm in lead.decision_makers if dm.get("email")}
    personal = [e for e in lead.emails_found if e.lower() in dm_emails]
    generic = [e for e in lead.emails_found if e.lower() not in dm_emails]
    ordered = personal + generic

    verified_any = False
    for email in ordered[:5]:
        try:
            result = hunter.email_verifier(email)
            if not result:
                continue
            status = result.get("result", "unknown")
            score = result.get("score", 0)

            for dm in lead.decision_makers:
                if dm.get("email", "").lower() == email.lower():
                    dm["email_status"] = status
                    dm["confidence"] = score
                    break

            if not verified_any or status == "deliverable":
                lead.email_status = status
                lead.email_confidence = score
                verified_any = True

            if status == "deliverable":
                stats["hunter_verified"] += 1
            elif status == "undeliverable":
                stats["hunter_failed"] += 1
                if email in lead.emails_found:
                    lead.emails_found.remove(email)
                    lead.emails_found.append(email)
                log_dev("HUNTER", f"Undeliverable: {email}", "warning")

        except Exception as e:
            log_dev("HUNTER", f"Verify failed for {email}: {str(e)[:50]}", "error")


def _step_score(job_id, leads, config, job_mgr):
    """Step 4: Score all leads and finalize results."""
    job_mgr.update(job_id, step="Scoring leads...", progress=95)

    scorer = LeadScorer(
        {
            "positive_signals": config.get("positive_signals", []),
            "negative_signals": config.get("negative_signals", []),
            "exclude_brands": config.get("exclude_brands", []),
            "min_rating": config.get("min_rating", 4.0),
            "segment": config.get("segment", "mid-premium"),
        }
    )

    lead_list = list(leads.values())
    for lead in lead_list:
        score, reasons = scorer.score(lead)
        lead.fit_score = score
        lead.fit_reasons = reasons

    lead_list.sort(key=lambda x: x.fit_score, reverse=True)

    job_mgr.update(
        job_id,
        leads=[asdict(lead) for lead in lead_list],
        status="completed",
        step=f"Found {len(lead_list)} leads",
        progress=100,
    )
