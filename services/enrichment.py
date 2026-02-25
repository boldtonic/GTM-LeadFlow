"""
Enrichment service for GTM LeadFlow.
Single-domain, batch, contact enrichment, and universal prospect enrichment pipeline.
"""

import re
import time
from dataclasses import asdict
from datetime import datetime

from api_clients import ApolloClient, FirecrawlClient, HunterClient
from api_clients.scraper import WebsiteScraper
from models import Lead
from scoring import LeadScorer
from services import discovery as discovery_service
from utils import clean_domain, clean_markdown_text, extract_domain, is_business_domain, log_dev

# ============================================================
# ENRICHMENT SIGNALS (prospect-stage, lightweight)
# ============================================================


def compute_enrichment_signals(lead, hunter=None):
    """Set has_website, email_count, and enrichment_grade on a Lead.
    Hunter email count is a FREE API call — no credits consumed."""
    lead.has_website = bool(lead.domain and is_business_domain(lead.domain))

    if hunter and hunter.is_configured and lead.domain and lead.has_website:
        try:
            result = hunter.email_count(lead.domain)
            lead.email_count = result.get("total", 0) or 0
        except Exception:
            lead.email_count = 0

    # Enrichment grade: A (high potential), B (moderate), C (low)
    has_apollo = bool(lead.company_size or lead.industry or lead.linkedin_url)
    if lead.has_website and lead.email_count > 5 and has_apollo:
        lead.enrichment_grade = "A"
    elif lead.has_website and (lead.email_count > 0 or has_apollo):
        lead.enrichment_grade = "B"
    else:
        lead.enrichment_grade = "C"


# ============================================================
# UNIVERSAL ENRICHMENT PIPELINE
# ============================================================


def run_discover_and_enrich(job_id: str, config: dict, api_keys: dict, job_mgr):
    """Background job: discover prospects via any method, then enrich all of them.
    Same output shape as the Google Places pipeline."""
    job_mgr.create(
        job_id,
        {
            "status": "running",
            "step": "Discovering prospects...",
            "progress": 0,
            "leads": [],
            "errors": [],
            "cancelled": False,
            "enrichment_stats": {},
            "max_leads": config.get("max_leads", 50),
        },
    )

    try:
        # Step 1: Discovery (0-25%)
        log_dev("DISCOVER", f"Starting discovery job {job_id}, mode={config.get('mode')}", "info")
        discovery_result = discovery_service.discover_prospects(config, api_keys)

        if job_mgr.is_cancelled(job_id):
            return

        if not discovery_result.get("success") or not discovery_result.get("prospects"):
            error = discovery_result.get("error", "No prospects found")
            job_mgr.update(job_id, status="completed", step=error, progress=100, leads=[])
            return

        prospects = discovery_result["prospects"]
        source = discovery_result.get("source", "unknown")
        log_dev("DISCOVER", f"Found {len(prospects)} raw prospects via {source}", "info")
        job_mgr.update(job_id, progress=25, step=f"Found {len(prospects)} prospects. Enriching...")

        # Step 2: Convert prospects to Lead objects (25-30%)
        leads = _prospects_to_leads(prospects, config, source=source)
        log_dev("ENRICH", f"Converted {len(leads)} prospects to leads for enrichment", "info")
        job_mgr.update(job_id, progress=30, step="Starting enrichment pipeline...")

        if job_mgr.is_cancelled(job_id):
            return

        # Step 3: Enrich all leads (30-90%)
        target_titles = config.get("target_titles", ["owner", "founder", "director", "manager", "buyer"])
        stats = _enrich_lead_list(leads, api_keys, target_titles, job_mgr, job_id, progress_range=(30, 90))

        if job_mgr.is_cancelled(job_id):
            return

        # Step 4: Score (90-100%)
        job_mgr.update(job_id, progress=92, step="Scoring leads...")
        scorer = LeadScorer(
            {
                "positive_signals": config.get("positive_signals", []),
                "negative_signals": config.get("negative_signals", []),
                "exclude_brands": config.get("exclude_brands", []),
                "min_rating": config.get("min_rating", 4.0),
                "segment": config.get("segment", "mid-premium"),
            }
        )
        for lead in leads:
            score, reasons = scorer.score(lead)
            lead.fit_score = score
            lead.fit_reasons = reasons

        leads.sort(key=lambda x: x.fit_score, reverse=True)

        job_mgr.update(
            job_id,
            leads=[asdict(lead) for lead in leads],
            enrichment_stats=stats,
            status="completed",
            step=f"Found {len(leads)} enriched leads",
            progress=100,
        )
        log_dev("ENRICH", f"Job {job_id} complete: {len(leads)} enriched leads", "success")

    except Exception as e:
        log_dev("ENRICH", f"Job {job_id} error: {e}", "error")
        job_mgr.update(job_id, status="error", step=f"Error: {str(e)}", errors=[str(e)])


def _prospects_to_leads(prospects: list, config: dict, source: str = "") -> list:
    """Convert raw prospect dicts from any discovery source into Lead dataclass objects."""
    leads = []
    seen_domains = set()

    for p in prospects:
        name = p.get("name", "").strip()
        if not name or len(name) < 2:
            continue

        # Resolve domain
        website = p.get("website", "") or ""
        domain = ""
        if website:
            domain = extract_domain(website)
            if not domain or not is_business_domain(domain):
                domain = ""

        # Deduplicate by domain
        if domain and domain in seen_domains:
            continue
        if domain:
            seen_domains.add(domain)

        # Parse location
        location = p.get("location", "")
        city = p.get("city", "")
        country = p.get("country", "")
        if not city and not country and location:
            parts = [x.strip() for x in location.split(",")]
            if len(parts) >= 2:
                city = parts[0]
                country = parts[-1]
            elif len(parts) == 1:
                city = parts[0]

        lead = Lead(
            place_id=p.get("place_id", p.get("id", "")),
            google_maps_url=p.get("sourceUrl", ""),
            name=name[:100],
            category=p.get("category", "Business"),
            address_full=p.get("address", ""),
            city=city,
            country=country,
            phone=p.get("phone", "") or "",
            website=website,
            domain=domain,
            rating=float(p.get("rating") or 0),
            reviews_count=int(p.get("reviewCount") or p.get("reviews_count") or 0),
            linkedin_url=p.get("linkedin_url", ""),
            company_size=p.get("company_size", ""),
            source=source,
            scraped_at=datetime.now().isoformat(),
            source_query=config.get("query", "") or config.get("category", ""),
        )
        leads.append(lead)

    return leads


def _enrich_lead_list(
    leads: list, api_keys: dict, target_titles: list, job_mgr=None, job_id=None, progress_range=(0, 100)
) -> dict:
    """Run the full enrichment pipeline on a list of Lead objects.
    This is the SHARED enrichment engine used by both Google Places and discovery paths."""
    apollo = ApolloClient(api_keys.get("apollo", ""), log_fn=log_dev)
    scraper = WebsiteScraper(api_keys.get("firecrawl", ""), log_fn=log_dev)
    hunter = HunterClient(api_keys.get("hunter", ""), log_fn=log_dev)

    total = len(leads)
    start_pct, end_pct = progress_range
    log_dev("ENRICH", f"Starting enrichment for {total} leads", "info")

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

    for i, lead in enumerate(leads):
        if job_mgr and job_id and job_mgr.is_cancelled(job_id):
            job_mgr.update(job_id, status="cancelled", step="Cancelled by user")
            log_dev("ENRICH", "Enrichment cancelled by user", "warning")
            return stats

        if not lead.domain and lead.website:
            lead.domain = extract_domain(lead.website)

        if not lead.domain:
            stats["no_website"] += 1
            continue

        pct = start_pct + int((i / total) * (end_pct - start_pct))
        if job_mgr and job_id:
            job_mgr.update(job_id, progress=pct, step=f"Enriching: {lead.name[:30]}...")

        domain = lead.domain
        biz_domain = is_business_domain(domain)

        # Website scraping
        if biz_domain:
            scrape_result = scraper.scrape(lead.website or f"https://{domain}")
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

        # Apollo enrichment
        if apollo.is_configured and biz_domain:
            _enrich_lead_apollo(lead, domain, target_titles, apollo, stats)

        # Hunter email_finder for Apollo contacts without emails
        if hunter.is_configured and biz_domain:
            _find_emails_for_contacts(lead, domain, hunter, stats)

        # Hunter email enrichment
        if hunter.is_configured and biz_domain:
            _enrich_lead_hunter(lead, domain, hunter, stats)

        # Enrichment signals (email count is free)
        compute_enrichment_signals(lead, hunter)

        lead.enriched_at = datetime.now().isoformat()
        time.sleep(0.1)

        if (i + 1) % 50 == 0:
            log_dev("ENRICH", f"Progress: {i + 1}/{total} leads enriched", "info")

    log_dev(
        "ENRICH",
        f"Enrichment complete: Firecrawl={stats['firecrawl_success']}, "
        f"Basic={stats['basic_scrape']}, Apollo={stats['apollo_org_success']}, "
        f"Hunter={stats['hunter_emails_found']} emails/{stats['hunter_verified']} verified",
        "success",
    )
    return stats


def _enrich_lead_apollo(lead, domain, target_titles, apollo, stats):
    """Enrich a Lead object with Apollo org + contacts data."""
    try:
        org = apollo.enrich_organization(domain)
        if org:
            lead.company_size = lead.company_size or str(org.get("estimated_num_employees", ""))
            lead.linkedin_url = lead.linkedin_url or org.get("linkedin_url", "")
            lead.estimated_revenue = lead.estimated_revenue or (org.get("annual_revenue_printed") or "")
            lead.industry = lead.industry or (org.get("industry") or "")
            lead.founded_year = lead.founded_year or str(org.get("founded_year") or "")
            # Always prefer Apollo name — it's the canonical company name
            apollo_name = org.get("name", "")
            if apollo_name:
                lead.name = apollo_name
            # Prefer Apollo location — more accurate than search query location
            if org.get("city"):
                lead.city = org["city"]
            if org.get("country"):
                lead.country = org["country"]
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


def _enrich_lead_hunter(lead, domain, hunter, stats):
    """Enrich a Lead object with Hunter email data + verification."""
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
    """Verify all found emails (up to 5). Prioritize personal emails first.
    Stores email_status on each matching decision_maker and on lead."""
    if not lead.emails_found:
        return

    # Sort: personal emails first (those matching a decision maker), then generic
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

            # Store on matching decision maker
            for dm in lead.decision_makers:
                if dm.get("email", "").lower() == email.lower():
                    dm["email_status"] = status
                    dm["confidence"] = score
                    break

            # Set lead-level status from the best verified email
            if not verified_any or status == "deliverable":
                lead.email_status = status
                lead.email_confidence = score
                verified_any = True

            if status == "deliverable":
                stats["hunter_verified"] += 1
            elif status == "undeliverable":
                stats["hunter_failed"] += 1
                # Demote undeliverable to end of list
                if email in lead.emails_found:
                    lead.emails_found.remove(email)
                    lead.emails_found.append(email)
                log_dev("HUNTER", f"Undeliverable: {email}", "warning")

        except Exception as e:
            log_dev("HUNTER", f"Verify failed for {email}: {str(e)[:50]}", "error")


def get_socials(people: list, api_keys: dict) -> dict:
    """Find LinkedIn (and Twitter) URLs for a list of people.

    Each person dict: {first_name, last_name, title, company}
    Strategy: Apollo find_person first, then Firecrawl web search as fallback.
    Returns: {success, results: [...], found, total}
    """
    apollo = ApolloClient(api_keys.get("apollo", ""), log_fn=log_dev)
    firecrawl = FirecrawlClient(api_keys.get("firecrawl", ""), log_fn=log_dev)

    if not apollo.is_configured and not firecrawl.is_configured:
        return {"success": False, "error": "Apollo or Firecrawl API required to find social profiles"}

    results = []
    for person in people:
        first = person.get("first_name", "").strip()
        last = person.get("last_name", "").strip()
        company = person.get("company", "").strip()
        title = person.get("title", "").strip()

        existing_url = person.get("linkedin_url", "").strip()
        result = {
            "first_name": first,
            "last_name": last,
            "title": title,
            "company": company,
            "linkedin_url": existing_url,
            "twitter_url": "",
            "source": "csv" if existing_url else "not_found",
        }

        # If LinkedIn URL already in CSV, skip API lookups
        if existing_url:
            log_dev("SOCIALS", f"CSV: using existing LinkedIn URL for {first} {last}", "success")
            results.append(result)
            continue

        # Apollo: primary source — has LinkedIn URLs for most B2B professionals
        if apollo.is_configured and (first or last) and company:
            try:
                person_data = apollo.find_person(first, last, company, title or None)
                if person_data:
                    result["linkedin_url"] = person_data.get("linkedin_url", "") or ""
                    result["twitter_url"] = person_data.get("twitter_url", "") or ""
                    if result["linkedin_url"]:
                        result["source"] = "apollo"
                        log_dev("SOCIALS", f"Apollo: found LinkedIn for {first} {last}", "success")
            except Exception as e:
                log_dev("SOCIALS", f"Apollo find_person failed for {first} {last}: {str(e)[:60]}", "error")

        # Firecrawl fallback: web search for LinkedIn profile URL
        if not result["linkedin_url"] and firecrawl.is_configured and (first or last):
            try:
                query = f'"{first} {last}" {company} site:linkedin.com/in'
                search_results = firecrawl.search(query, limit=3)
                for item in search_results:
                    url = item.get("url", "")
                    if "linkedin.com/in/" in url:
                        result["linkedin_url"] = url.split("?")[0]
                        result["source"] = "firecrawl"
                        log_dev("SOCIALS", f"Firecrawl: found LinkedIn for {first} {last}", "success")
                        break
                    # Also scan markdown content for linkedin.com/in/ links
                    markdown = item.get("markdown", "")
                    li_match = re.search(r"linkedin\.com/in/([a-zA-Z0-9\-_%]+)", markdown)
                    if li_match:
                        result["linkedin_url"] = f"https://www.linkedin.com/in/{li_match.group(1)}"
                        result["source"] = "firecrawl"
                        log_dev("SOCIALS", f"Firecrawl (markdown): found LinkedIn for {first} {last}", "success")
                        break
            except Exception as e:
                log_dev("SOCIALS", f"Firecrawl search failed for {first} {last}: {str(e)[:60]}", "error")

        if not result["linkedin_url"]:
            log_dev("SOCIALS", f"No LinkedIn found for {first} {last} ({company})", "warning")

        results.append(result)
        time.sleep(0.3)  # Be gentle on APIs

    found = sum(1 for r in results if r["linkedin_url"])
    log_dev("SOCIALS", f"Get Socials complete: {found}/{len(results)} profiles found", "success")
    return {"success": True, "results": results, "found": found, "total": len(results)}


def enrich_company(domain_input: str, api_keys: dict) -> dict:
    """Enrich a single company by domain. Returns dict for jsonify."""
    domain = clean_domain(domain_input)
    if not domain:
        return {"success": False, "error": "Domain is required"}

    try:
        result = {"domain": domain}
        firecrawl_key = api_keys.get("firecrawl", "")
        apollo_key = api_keys.get("apollo", "")
        hunter_key = api_keys.get("hunter", "")

        # Scrape website
        if firecrawl_key:
            url = f"https://{domain}"
            scraper = WebsiteScraper(firecrawl_key, log_fn=log_dev)
            scrape_result = scraper.scrape(url)

            if scrape_result:
                markdown = scrape_result.get("markdown", "")
                links = scrape_result.get("links", [])
                result = extract_company_data(markdown, links, domain)

        # Apollo enrichment
        if apollo_key:
            apollo = ApolloClient(apollo_key, log_fn=log_dev)
            _enrich_with_apollo(result, domain, apollo)

        # Hunter email enrichment
        if hunter_key:
            hunter = HunterClient(hunter_key, log_fn=log_dev)
            _enrich_with_hunter(result, domain, hunter)

        return {"success": True, "data": result}

    except Exception as e:
        log_dev("ENRICH", f"Enrichment error for {domain}: {e}", "error")
        return {"success": False, "error": str(e)}


def enrich_batch(domains: list, api_keys: dict) -> dict:
    """Enrich multiple companies in batch."""
    firecrawl_key = api_keys.get("firecrawl", "")
    apollo_key = api_keys.get("apollo", "")

    if not firecrawl_key:
        return {"success": False, "error": "Firecrawl API not configured"}

    results = []
    for domain_input in domains:
        domain = clean_domain(domain_input)
        if not domain:
            continue

        try:
            url = f"https://{domain}"
            firecrawl = FirecrawlClient(firecrawl_key, log_fn=log_dev)
            scrape_result = firecrawl.scrape(url)

            if scrape_result:
                markdown = scrape_result.get("markdown", "")
                links = scrape_result.get("links", [])
                company_data = extract_company_data(markdown, links, domain)

                if apollo_key:
                    apollo = ApolloClient(apollo_key, log_fn=log_dev)
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

    return {"success": True, "data": results}


def enrich_contacts(data: dict, apollo_key: str) -> dict:
    """Find contacts at a company via Apollo."""
    domain = data.get("domain", "").strip()
    company_name = data.get("companyName", "").strip()

    if not domain and not company_name:
        return {"success": False, "error": "Domain or company name is required"}
    if not apollo_key:
        return {"success": False, "error": "Apollo API not configured"}

    try:
        log_dev("ENRICH", f"Enriching contacts for: {domain or company_name}", "info")
        apollo = ApolloClient(apollo_key, log_fn=log_dev)

        # Find the organization
        orgs = apollo.search_organizations(name=company_name if not domain else None, domain=domain or None, per_page=1)

        organization = orgs[0] if orgs else None

        if not organization:
            return {"success": True, "data": {"domain": domain, "contacts": [], "apolloEnriched": False}}

        log_dev("ENRICH", f"Found organization: {organization.get('name')}", "info")

        # Search for people
        people = apollo.search_contacts(
            domain=domain or None,
            organization_ids=[organization.get("id")],
            seniorities=["director", "vp", "c_suite", "founder", "owner"],
            per_page=10,
        )

        log_dev("ENRICH", f"Found {len(people)} contacts", "info")

        contacts = [
            {
                "name": p.get("name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                "title": p.get("title", "Unknown"),
                "email": p.get("email"),
                "phone": p.get("phone_numbers", [{}])[0].get("sanitized_number") if p.get("phone_numbers") else None,
                "linkedin": p.get("linkedin_url"),
            }
            for p in people
        ]

        return {
            "success": True,
            "data": {
                "domain": domain or organization.get("primary_domain"),
                "contacts": contacts,
                "apolloEnriched": True,
            },
        }

    except Exception as e:
        log_dev("ENRICH", f"Error enriching contacts: {e}", "error")
        return {"success": False, "error": str(e)}


def _enrich_with_apollo(result: dict, domain: str, apollo: ApolloClient):
    """Add Apollo org + contacts data to result dict."""
    try:
        org = apollo.enrich_organization(domain)
        if org:
            apollo_name = org.get("name")
            if apollo_name and (not result.get("name") or result.get("name") == domain.split(".")[0].capitalize()):
                result["name"] = apollo_name

            result["industry"] = result.get("industry") or org.get("industry")
            result["linkedin"] = result.get("linkedin") or org.get("linkedin_url")

            employees = org.get("estimated_num_employees")
            if employees:
                result["employees"] = f"{employees:,}" if isinstance(employees, int) else str(employees)

            result["revenue"] = org.get("annual_revenue_printed")

            city = org.get("city")
            state = org.get("state")
            country = org.get("country")
            if city or state or country:
                loc_parts = [p for p in [city, state, country] if p]
                result["location"] = result.get("location") or ", ".join(loc_parts)

            result["founded"] = result.get("founded") or org.get("founded_year")

        contacts = apollo.search_contacts(
            domain, titles=["ceo", "founder", "owner", "director", "manager", "head", "vp", "president"]
        )
        if contacts:
            result["contacts"] = [
                {
                    "name": c.get("name", ""),
                    "title": c.get("title", ""),
                    "email": c.get("email", ""),
                    "phone": c.get("phone_numbers", [{}])[0].get("sanitized_number", "")
                    if c.get("phone_numbers")
                    else "",
                    "linkedin": c.get("linkedin_url", ""),
                }
                for c in contacts[:5]
            ]
    except Exception as e:
        log_dev("ENRICH", f"Apollo enrichment error for {domain}: {e}", "error")


def _enrich_with_hunter(result: dict, domain: str, hunter: HunterClient):
    """Add Hunter email data to result dict."""
    try:
        hunter_data = hunter.domain_search(domain, limit=5)

        if hunter_data:
            hunter_emails = hunter_data.get("emails", [])
            if hunter_emails:
                existing_emails = set()
                if result.get("contacts"):
                    existing_emails = {c.get("email", "").lower() for c in result["contacts"] if c.get("email")}

                for he in hunter_emails:
                    email = he.get("value", "")
                    if email and email.lower() not in existing_emails:
                        if "contacts" not in result:
                            result["contacts"] = []
                        result["contacts"].append(
                            {
                                "name": f"{he.get('first_name', '')} {he.get('last_name', '')}".strip() or "Unknown",
                                "title": he.get("position", ""),
                                "email": email,
                                "phone": he.get("phone_number", ""),
                                "linkedin": he.get("linkedin", ""),
                                "source": "hunter.io",
                                "confidence": he.get("confidence", 0),
                            }
                        )
                        existing_emails.add(email.lower())

            if not result.get("name") and hunter_data.get("organization"):
                result["name"] = hunter_data.get("organization")

            result["hunter_email_count"] = hunter_data.get("total", 0)
    except Exception as e:
        log_dev("ENRICH", f"Hunter.io enrichment error for {domain}: {e}", "error")


def extract_company_data(markdown: str, links: list, domain: str) -> dict:
    """Extract comprehensive company data from scraped content."""
    data = {"domain": domain}

    skip_patterns = [
        "skip to",
        "sign in",
        "sign up",
        "log in",
        "signed",
        "click",
        "chat with",
        "menu",
        "navigation",
        "footer",
        "header",
        "cookie",
        "privacy",
    ]

    def is_valid_name(name):
        if not name or len(name) > 60 or len(name) < 2:
            return False
        name_lower = name.lower()
        if any(skip in name_lower for skip in skip_patterns):
            return False
        if not name[0].isupper() and not name[0].isdigit():
            return False
        return True

    # Domain-based name as fallback
    domain_name = domain.split(".")[0]
    if domain_name.isupper() or domain_name.islower():
        data["name"] = domain_name.capitalize()
    else:
        data["name"] = domain_name

    # Try better name from content
    name = None
    meta_match = re.search(
        r"(?:company|about|welcome to)\s+([A-Z][A-Za-z0-9\s&.]+?)(?:\s*[|\-.,]|\n|$)", markdown, re.IGNORECASE
    )
    if meta_match:
        candidate = clean_markdown_text(meta_match.group(1).strip())
        if is_valid_name(candidate):
            name = candidate

    if not name:
        title_match = re.search(r"^#\s+([A-Z][A-Za-z0-9\s&.]+?)(?:\s*[-|•·]|\n|$)", markdown, re.MULTILINE)
        if title_match:
            candidate = clean_markdown_text(title_match.group(1).strip())
            if is_valid_name(candidate):
                name = candidate

    if name and is_valid_name(name):
        data["name"] = name

    # Description
    paragraphs = markdown.split("\n\n")
    for p in paragraphs:
        cleaned = clean_markdown_text(re.sub(r"^[#*\-\s]+", "", p).strip())
        if any(
            skip in cleaned.lower() for skip in ["signed in", "signed out", "reload", "click here", "sign up", "log in"]
        ):
            continue
        if 50 < len(cleaned) < 500 and not cleaned.startswith("["):
            data["description"] = cleaned
            break

    # Social links
    linkedin_match = re.search(r"linkedin\.com/company/([a-zA-Z0-9\-]+)", markdown, re.IGNORECASE)
    if linkedin_match:
        data["linkedin"] = f"https://linkedin.com/company/{linkedin_match.group(1)}"
    elif links:
        for link in links:
            if "linkedin.com/company" in link:
                data["linkedin"] = link
                break

    twitter_match = re.search(r"(?:twitter\.com|x\.com)/([a-zA-Z0-9_]+)", markdown, re.IGNORECASE)
    if twitter_match:
        data["twitter"] = f"https://twitter.com/{twitter_match.group(1)}"
    elif links:
        for link in links:
            if "twitter.com" in link or "x.com" in link:
                data["twitter"] = link
                break

    instagram_match = re.search(r"instagram\.com/([a-zA-Z0-9_.]+)", markdown, re.IGNORECASE)
    if instagram_match:
        data["instagram"] = f"https://instagram.com/{instagram_match.group(1)}"
    elif links:
        for link in links:
            if "instagram.com" in link:
                data["instagram"] = link
                break

    facebook_match = re.search(r"facebook\.com/([a-zA-Z0-9_.]+)", markdown, re.IGNORECASE)
    if facebook_match:
        data["facebook"] = f"https://facebook.com/{facebook_match.group(1)}"

    # Email
    email_match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", markdown)
    if email_match:
        email = email_match.group(1)
        if not any(x in email.lower() for x in ["example", "noreply", "wordpress", "wixpress"]):
            data["email"] = email

    # Phone
    phone_match = re.search(r"(?:\+1|1)?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}", markdown)
    if phone_match:
        data["phone"] = phone_match.group(0)

    # Industry detection
    industry_keywords = {
        "Technology": ["software", "saas", "tech", "platform", "api", "cloud", "ai", "machine learning", "data"],
        "E-commerce": ["ecommerce", "e-commerce", "shop", "store", "retail", "marketplace", "buy", "sell"],
        "Finance": ["fintech", "banking", "payments", "financial", "insurance", "invest"],
        "Healthcare": ["health", "medical", "healthcare", "pharma", "biotech", "clinic"],
        "Marketing": ["marketing", "advertising", "agency", "seo", "digital marketing", "branding"],
        "Education": ["education", "learning", "edtech", "courses", "training", "school"],
        "Real Estate": ["real estate", "property", "realty", "housing", "homes"],
        "Manufacturing": ["manufacturing", "factory", "production", "industrial"],
        "Consulting": ["consulting", "consultancy", "advisory", "professional services"],
    }

    lower_markdown = markdown.lower()
    for industry, keywords in industry_keywords.items():
        if any(keyword in lower_markdown for keyword in keywords):
            data["industry"] = industry
            break

    # Technologies
    tech_patterns = [
        "react",
        "vue",
        "angular",
        "nextjs",
        "nuxt",
        "svelte",
        "node",
        "python",
        "java",
        "ruby",
        "go",
        "rust",
        "php",
        "django",
        "flask",
        "aws",
        "azure",
        "gcp",
        "kubernetes",
        "docker",
        "vercel",
        "netlify",
        "postgresql",
        "mongodb",
        "mysql",
        "redis",
        "elasticsearch",
        "stripe",
        "shopify",
        "salesforce",
        "hubspot",
        "segment",
        "mixpanel",
        "wordpress",
        "webflow",
        "squarespace",
        "wix",
    ]
    found_tech = [tech for tech in tech_patterns if tech in lower_markdown]
    if found_tech:
        data["technologies"] = [t.capitalize() for t in list(set(found_tech))]

    # Location
    location_patterns = [
        r"(?:headquartered|located|based)\s+in\s+([A-Z][a-zA-Z\s,]+)",
        r"([A-Z][a-zA-Z]+,\s*[A-Z]{2})\s*\d{5}",
    ]
    for pattern in location_patterns:
        loc_match = re.search(pattern, markdown)
        if loc_match:
            data["location"] = loc_match.group(1).strip()
            break

    # Founded year
    founded_match = re.search(r"(?:founded|established|since)\s*(?:in\s*)?(\d{4})", markdown, re.IGNORECASE)
    if founded_match:
        data["founded"] = founded_match.group(1)

    # Social links list
    social_links = [
        link
        for link in links
        if any(x in link for x in ["linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com"])
    ]
    if social_links:
        data["socialLinks"] = social_links

    return data
