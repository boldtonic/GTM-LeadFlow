"""
GTM Brief generation service for GTM LeadFlow.
Uses Firecrawl for scraping + OpenAI for analysis.
"""

import json
import time
from datetime import datetime

import requests

from api_clients import FirecrawlClient
from utils import log_dev


def generate_brief(data: dict, firecrawl_key: str, openai_key: str) -> dict:
    """Generate comprehensive GTM Brief. Returns dict for jsonify."""
    url = data.get("url", "").strip()
    additional_context = data.get("additionalContext", "")

    if not url:
        return {"success": False, "error": "URL is required"}
    if not firecrawl_key:
        return {"success": False, "error": "Firecrawl API not configured"}
    if not openai_key:
        return {"success": False, "error": "OpenAI API not configured"}

    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"

    try:
        # Step 1: Scrape website
        log_dev("BRIEF", f"Scraping website: {url}", "info")
        firecrawl = FirecrawlClient(firecrawl_key, log_fn=log_dev)
        scrape_result = firecrawl.scrape(url)

        if not scrape_result:
            return {"success": False, "error": "Failed to scrape website"}

        markdown = scrape_result.get("markdown", "")
        links = scrape_result.get("links", [])
        log_dev("BRIEF", f"Scraped content length: {len(markdown)}", "info")

        # Step 2: Generate via OpenAI
        brief_data = _call_openai(url, markdown, links, additional_context, openai_key)
        if isinstance(brief_data, dict) and "error" in brief_data and not brief_data.get("companySnapshot"):
            return {"success": False, "error": brief_data["error"]}

        # Add metadata
        now = datetime.now().isoformat()
        brief = {"id": str(hash(url + now))[:12], **brief_data, "createdAt": now, "updatedAt": now}

        if "companySnapshot" in brief:
            brief["companySnapshot"]["website"] = url

        log_dev("BRIEF", "GTM Brief generated successfully", "success")
        return {"success": True, "brief": brief}

    except Exception as e:
        log_dev("BRIEF", f"Error generating brief: {e}", "error")
        return {"success": False, "error": str(e)}


def _call_openai(url, markdown, links, additional_context, openai_key):
    """Call OpenAI API with structured output for GTM brief."""
    system_prompt = """You are a B2B go-to-market research analyst. Analyze the provided website content and generate a comprehensive GTM brief. Be specific, actionable, and data-driven in your analysis.

IMPORTANT: For target personas, classify each into one of three roles:
- decision_maker: C-suite, VPs, Directors who approve purchases
- influencer: Managers who research and recommend solutions
- worker: End-users who can advocate internally and relay messages to decision-makers

Include at least one persona from each category when relevant."""

    user_prompt = f"""Analyze this company website content and generate a detailed GTM (Go-To-Market) brief.

Website: {url}
{f"Additional Context: {additional_context}" if additional_context else ""}

Website Content:
{markdown[:15000]}

Social/External Links Found:
{chr(10).join(links[:20]) if links else "None"}

Generate a comprehensive GTM brief with:
1. Company snapshot (name, tagline, industry, location if found)
2. Product intelligence (products/services, features, pricing if available, tech stack)
3. Market positioning (market tension they address, value proposition, differentiators, proof points)
4. Ideal Customer Profile - include THREE types of personas:
   - Decision Makers (C-suite, VPs who approve)
   - Influencers (Managers who research/recommend)
   - Workers/Users (End-users who can advocate internally)
5. Search strategy for finding PROSPECTS (businesses that would BUY from this company):
   - suggestedQueries: Google Maps/Places search queries to find POTENTIAL CUSTOMER BUSINESSES
   - geographies: target locations/cities for prospecting
   - outreachAngles: messaging angles for each persona

CRITICAL: The suggestedQueries must be searches to find BUSINESSES THAT WOULD BUY FROM the analyzed company, not searches about the company itself."""

    tools = [_build_tool_schema()]

    # Retry logic for rate limits
    max_retries = 3
    openai_response = None

    for attempt in range(max_retries):
        openai_response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "generate_gtm_brief"}},
            },
            timeout=60,
        )

        if openai_response.status_code == 429:
            if attempt < max_retries - 1:
                wait_time = 5 * (2**attempt)
                log_dev("BRIEF", f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})", "warning")
                time.sleep(wait_time)
                continue
        break

    if not openai_response.ok:
        try:
            error_data = openai_response.json()
            error_message = error_data.get("error", {}).get("message", f"API error: {openai_response.status_code}")
        except (json.JSONDecodeError, Exception):
            error_message = f"OpenAI API error: {openai_response.status_code}"
        return {"error": error_message}

    ai_data = openai_response.json()
    tool_call = ai_data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [{}])[0]

    if not tool_call or tool_call.get("function", {}).get("name") != "generate_gtm_brief":
        return {"error": "Failed to parse AI response"}

    return json.loads(tool_call["function"]["arguments"])


def _build_tool_schema():
    """Build the OpenAI function tool schema for GTM brief generation."""
    return {
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
                            "revenue": {"type": "string"},
                        },
                        "required": ["name", "tagline", "industry"],
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
                                        "pricing": {"type": "string"},
                                    },
                                },
                            },
                            "keyFeatures": {"type": "array", "items": {"type": "string"}},
                            "integrations": {"type": "array", "items": {"type": "string"}},
                            "techStack": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "positioning": {
                        "type": "object",
                        "properties": {
                            "marketTension": {"type": "string"},
                            "valueProposition": {"type": "string"},
                            "differentiators": {"type": "array", "items": {"type": "string"}},
                            "proofPoints": {"type": "array", "items": {"type": "string"}},
                            "targetOutcome": {"type": "string"},
                        },
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
                                        "goals": {"type": "array", "items": {"type": "string"}},
                                    },
                                },
                            },
                            "companyCharacteristics": {
                                "type": "object",
                                "properties": {
                                    "size": {"type": "string"},
                                    "industries": {"type": "array", "items": {"type": "string"}},
                                    "technologies": {"type": "array", "items": {"type": "string"}},
                                    "signals": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                            "qualificationCriteria": {"type": "array", "items": {"type": "string"}},
                            "disqualifiers": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "searchStrategy": {
                        "type": "object",
                        "properties": {
                            "suggestedQueries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Google Maps search queries to find POTENTIAL CUSTOMER BUSINESSES",
                            },
                            "geographies": {"type": "array", "items": {"type": "string"}},
                            "outreachAngles": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "angle": {"type": "string"},
                                        "targetPersona": {"type": "string"},
                                        "keyMessage": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
                "required": ["companySnapshot", "productIntelligence", "positioning", "icp", "searchStrategy"],
            },
        },
    }
