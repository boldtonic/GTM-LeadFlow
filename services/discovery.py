"""
Discovery service for GTM LeadFlow.
Handles smart search, Apollo search, Maps URL scraping.
"""

import re
from datetime import datetime

from api_clients import ApolloClient, FirecrawlClient
from utils import log_dev


def discover_prospects(data: dict, api_keys: dict) -> dict:
    """Discover prospects using various methods. Returns dict for jsonify."""
    mode = data.get("mode", "smart_search")
    queries = data.get("queries", [])
    query = data.get("query", "")
    category = data.get("category", "")
    location = data.get("location", "")
    industry = data.get("industry", "")
    maps_url = data.get("mapsUrl", "")
    company_size = data.get("company_size", "")
    max_results = data.get("max_results", 25)

    # Build search queries if not provided
    if not queries and query:
        queries = [query]
    elif not queries and category and location:
        queries = [
            f"{category} companies {location}",
            f"{category} stores {location}",
            f"best {category} businesses {location}",
            f"{category} retailers {location} contact",
        ]

    firecrawl_key = api_keys.get("firecrawl", "")
    apollo_key = api_keys.get("apollo", "")

    # Mode: Maps URL scraping
    if mode in ("maps_url", "maps"):
        return _discover_maps(maps_url, location, category, firecrawl_key)

    # Mode: Smart Search
    if mode == "smart_search":
        return _discover_smart(queries, category, location, industry, max_results, firecrawl_key, apollo_key)

    # Mode: Apollo Search (with fallback)
    if mode == "apollo_search":
        return _discover_apollo(
            queries, category, location, industry, company_size, max_results, firecrawl_key, apollo_key
        )

    # Mode: Web Search (default)
    if not firecrawl_key:
        return {"success": False, "error": "Firecrawl not configured"}
    if not queries:
        return {"success": False, "error": "Search query or queries[] required"}

    log_dev("DISCOVER", f"Running web search with {len(queries)} queries", "info")
    all_prospects = run_web_search(queries, category, location, firecrawl_key)
    unique_prospects = deduplicate_prospects(all_prospects)

    return {
        "success": True,
        "prospects": unique_prospects,
        "totalFound": len(unique_prospects),
        "queriesUsed": queries[:3],
        "source": "web_search",
    }


def _discover_maps(maps_url, location, category, firecrawl_key):
    """Discover prospects from a Google Maps URL."""
    if not maps_url:
        return {"success": False, "error": "Google Maps URL is required"}
    if not firecrawl_key:
        return {"success": False, "error": "Firecrawl not configured"}

    log_dev("DISCOVER", f"Scraping Google Maps URL: {maps_url[:60]}", "info")
    firecrawl = FirecrawlClient(firecrawl_key, log_fn=log_dev)
    now = datetime.now().isoformat()

    # Try structured extraction first
    import requests

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
                                        "website": {"type": "string"},
                                    },
                                    "required": ["name"],
                                },
                            }
                        },
                        "required": ["businesses"],
                    },
                    "systemPrompt": "Extract all business listings from this Google Maps search results page.",
                },
                "waitFor": 5000,
            },
            timeout=45,
        )

        scrape_data = response.json()
        extract_data = scrape_data.get("data", {}).get("extract", {})
        businesses = extract_data.get("businesses", [])

        if not businesses:
            markdown = scrape_data.get("data", {}).get("markdown", "")
            links = scrape_data.get("data", {}).get("links", [])
            businesses = parse_maps_markdown(markdown, links, location)

    except Exception as e:
        log_dev("DISCOVER", f"Structured extraction failed: {e}", "warning")
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
        prospects.append(
            {
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
                "discoveredAt": now,
            }
        )

    # Sort by quality
    prospects.sort(
        key=lambda x: (
            (3 if x.get("phone") else 0)
            + (2 if x.get("website") else 0)
            + (1 if x.get("rating") else 0)
            + (1 if x.get("address") else 0)
        ),
        reverse=True,
    )

    return {"success": True, "prospects": prospects, "totalFound": len(prospects), "source": "maps_scrape"}


def _discover_smart(queries, category, location, industry, max_results, firecrawl_key, apollo_key):
    """Smart Search: try all available sources."""
    log_dev("DISCOVER", "Running Smart Search across all sources", "info")
    all_prospects = []
    sources = []
    fallback_reason = None

    if firecrawl_key and queries:
        web_prospects = run_web_search(queries, category, location, firecrawl_key)
        all_prospects.extend(web_prospects)
        if web_prospects:
            sources.append("web")
        log_dev("DISCOVER", f"Web search found {len(web_prospects)} prospects", "info")

    if apollo_key:
        apollo_result = run_apollo_search(category, location, industry, apollo_key, per_page=max_results)
        if apollo_result.get("prospects"):
            all_prospects.extend(apollo_result["prospects"])
            sources.append("apollo")
        elif apollo_result.get("error"):
            fallback_reason = apollo_result["error"]

    unique_prospects = deduplicate_prospects(all_prospects)[:max_results]
    log_dev(
        "DISCOVER",
        f"Smart search returning {len(unique_prospects)} unique prospects from: {', '.join(sources)}",
        "info",
    )

    return {
        "success": True,
        "prospects": unique_prospects,
        "totalFound": len(unique_prospects),
        "queriesUsed": queries[:3],
        "source": "smart_search",
        "fallbackReason": fallback_reason,
    }


def _discover_apollo(queries, category, location, industry, company_size, max_results, firecrawl_key, apollo_key):
    """Apollo Search with web search fallback."""
    apollo_result = run_apollo_search(
        category, location, industry, apollo_key, company_size=company_size, per_page=max_results
    )

    if apollo_result.get("prospects"):
        return {
            "success": True,
            "prospects": apollo_result["prospects"],
            "totalFound": len(apollo_result["prospects"]),
            "source": "apollo",
        }

    # Fallback to web search
    if firecrawl_key and queries:
        log_dev("DISCOVER", f"Apollo failed: {apollo_result.get('error')}. Falling back to web search.", "info")
        web_prospects = run_web_search(queries, category, location, firecrawl_key)
        unique_prospects = deduplicate_prospects(web_prospects)
        return {
            "success": True,
            "prospects": unique_prospects,
            "totalFound": len(unique_prospects),
            "queriesUsed": queries[:3],
            "source": "web_search_fallback",
            "fallbackReason": apollo_result.get("error"),
        }

    return {"success": False, "error": apollo_result.get("error", "Apollo search failed")}


def run_web_search(search_queries: list, category: str, location: str, firecrawl_key: str) -> list:
    """Run web search using Firecrawl."""
    if not firecrawl_key:
        return []

    firecrawl = FirecrawlClient(firecrawl_key, log_fn=log_dev)
    prospects = []
    now = datetime.now().isoformat()

    for query in search_queries[:3]:
        log_dev("DISCOVER", f"Web searching: {query}", "info")
        try:
            results = firecrawl.search(query, limit=15)

            for result in results:
                title = result.get("title", "")
                url = result.get("url", "")
                markdown = result.get("markdown", "")

                name_match = re.match(r"^([^-–|]+)", title)
                name = name_match.group(1).strip() if name_match else title.split(" - ")[0]

                if not name or len(name) < 2 or len(name) > 100:
                    continue

                skip_domains = [
                    "wikipedia.org",
                    "facebook.com",
                    "linkedin.com/company",
                    "yelp.com",
                    "tripadvisor.com",
                    "yellowpages.com",
                ]
                if any(d in url for d in skip_domains):
                    continue

                rating_match = re.search(r"(\d\.?\d?)\s*(?:stars?|rating|★)", markdown, re.I)
                rating = float(rating_match.group(1)) if rating_match else None

                review_match = re.search(r"(\d+)\s*reviews?", markdown, re.I)
                review_count = int(review_match.group(1)) if review_match else None

                phone_match = re.search(r"(?:\+1|1)?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}", markdown)
                phone = phone_match.group(0) if phone_match else None

                website = url
                if "google.com" in url or "bing.com" in url:
                    links = result.get("links", [])
                    for link in links:
                        if link.startswith("http") and not any(d in link for d in ["google.com", "facebook.com"]):
                            website = link
                            break

                prospects.append(
                    {
                        "id": str(len(prospects) + 1),
                        "name": name[:100],
                        "category": category or "Business",
                        "location": location or "Unknown",
                        "rating": rating,
                        "reviewCount": review_count,
                        "website": website,
                        "phone": phone,
                        "sourceUrl": url,
                        "discoveredAt": now,
                    }
                )

        except Exception as e:
            log_dev("DISCOVER", f"Error searching '{query}': {e}", "error")

    return prospects


def run_apollo_search(category, location, industry, apollo_key, company_size=None, per_page=25) -> dict:
    """Run Apollo company search with error handling."""
    if not apollo_key:
        return {"prospects": [], "error": "Apollo API not configured"}

    log_dev("DISCOVER", f"Searching Apollo for: {industry or category} in {location}", "info")
    apollo = ApolloClient(apollo_key, log_fn=log_dev)

    try:
        employee_ranges = None
        if company_size:
            parts = company_size.split(",")
            if len(parts) == 2:
                employee_ranges = [f"{parts[0]},{parts[1]}"]

        orgs = apollo.search_organizations(
            name=category or industry, location=location or None, employee_ranges=employee_ranges, per_page=per_page
        )

        if not orgs and not isinstance(orgs, list):
            return {"prospects": [], "error": "Failed to search Apollo"}

        now = datetime.now().isoformat()
        prospects = []
        for org in orgs:
            loc = f"{org.get('city')}, {org.get('country')}" if org.get("city") else location
            prospects.append(
                {
                    "id": str(len(prospects) + 1),
                    "name": org.get("name", "Unknown"),
                    "category": industry or category or "Business",
                    "location": loc or "Unknown",
                    "address": org.get("street_address"),
                    "website": org.get("website_url"),
                    "phone": org.get("phone"),
                    "sourceUrl": org.get("linkedin_url") or f"https://apollo.io/companies/{org.get('id')}",
                    "discoveredAt": now,
                }
            )

        log_dev("DISCOVER", f"Found {len(prospects)} companies via Apollo", "info")
        return {"prospects": prospects}

    except Exception as e:
        log_dev("DISCOVER", f"Apollo search error: {e}", "error")
        return {"prospects": [], "error": str(e)}


def deduplicate_prospects(prospects: list) -> list:
    """Remove duplicate prospects by name or website."""
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
    """Parse Google Maps markdown to extract business listings."""
    businesses = []
    seen_names = set()
    lines = markdown.split("\n")

    for i, line in enumerate(lines):
        line = line.strip()

        if not line or len(line) < 3 or len(line) > 100:
            continue
        if line.startswith("http") or line.startswith("#"):
            continue
        if any(x in line.lower() for x in ["google maps", "sign in", "privacy", "terms"]):
            continue

        is_business = re.match(r"^[A-Z]", line) and "·" not in line and len(line) <= 60

        if is_business:
            context = " ".join(lines[i : i + 6])

            rating_match = re.search(r"(\d\.?\d?)\s*(?:stars?|★|⭐)", context, re.I)
            rating = float(rating_match.group(1)) if rating_match else None

            review_match = re.search(r"\((\d[\d,]*)\s*(?:reviews?|reseñas?)\)", context, re.I)
            review_count = int(review_match.group(1).replace(",", "")) if review_match else None

            phone_match = re.search(r"(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})", context)
            phone = phone_match.group(1).strip() if phone_match else None

            address_match = re.search(
                r"(\d+\s+[A-Za-z0-9\s,.-]+(?:St|Ave|Rd|Blvd|Dr|Way|Ln|Street|Avenue|Road)[^,]*)", context, re.I
            )
            address = address_match.group(1).strip() if address_match else None

            cat_match = re.search(r"·\s*([A-Za-z\s&]+?)(?:\s*·|$)", context)
            cat = cat_match.group(1).strip() if cat_match else None

            hours_match = re.search(r"(?:Open|Closed)\s*(?:until|at)?\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM)?", context, re.I)
            hours = hours_match.group(0) if hours_match else None

            website = None
            name_parts = line.lower().split()[:3]
            for link in links:
                if any(d in link for d in ["google.com", "facebook.com", "yelp.com"]):
                    continue
                if any(p in link.lower() for p in name_parts if len(p) > 3):
                    website = link
                    break

            if (rating or review_count or phone or address) and line.lower() not in seen_names:
                seen_names.add(line.lower())
                businesses.append(
                    {
                        "name": line,
                        "category": cat,
                        "rating": rating,
                        "reviewCount": review_count,
                        "address": address,
                        "phone": phone,
                        "website": website,
                        "hours": hours,
                    }
                )

    return businesses
