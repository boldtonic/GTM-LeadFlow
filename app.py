"""
GTM Lead Finder - Web Application
Flask backend with thin route handlers.
Business logic lives in services/, API clients in api_clients/.
"""

import os
import threading
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

from api_clients import HunterClient
from jobs import JobManager
from services import brief as brief_service
from services import discovery as discovery_service
from services import enrichment as enrichment_service
from services import export as export_service
from services import people as people_service
from services import search as search_service
from utils import clean_domain, dev_logs, log_dev

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

# Thread-safe job manager (replaces bare job_status dict)
job_mgr = JobManager(ttl_seconds=3600)


def _api_keys() -> dict:
    """Bundle API keys for passing to services."""
    return {
        "google_places": GOOGLE_PLACES_API_KEY,
        "apollo": APOLLO_API_KEY,
        "hunter": HUNTER_API_KEY,
        "firecrawl": FIRECRAWL_API_KEY,
        "openai": OPENAI_API_KEY,
    }


# ============================================================
# ROUTES
# ============================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def get_config():
    """Check if API keys are configured."""
    return jsonify(
        {
            "google_places": bool(GOOGLE_PLACES_API_KEY),
            "apollo": bool(APOLLO_API_KEY),
            "hunter": bool(HUNTER_API_KEY),
            "firecrawl": bool(FIRECRAWL_API_KEY),
            "openai": bool(OPENAI_API_KEY),
            "instantly": bool(INSTANTLY_API_KEY),
        }
    )


# --- Dev/Debug ---


@app.route("/api/dev/logs")
def get_dev_logs():
    """Get dev logs for debugging panel."""
    limit = request.args.get("limit", 100, type=int)
    category = request.args.get("category", None)

    logs = dev_logs[-limit:]
    if category:
        logs = [entry for entry in logs if entry["category"] == category]

    return jsonify({"logs": logs})


@app.route("/api/dev/logs/clear", methods=["POST"])
def clear_dev_logs():
    """Clear all dev logs."""
    dev_logs.clear()
    return jsonify({"success": True})


@app.route("/api/dev/stats")
def get_dev_stats():
    """Get enrichment statistics from last job."""
    recent_job = job_mgr.find_recent_completed()

    if not recent_job:
        return jsonify({"hasData": False})

    leads = recent_job.get("leads", [])
    stats = recent_job.get("enrichment_stats", {})

    total = len(leads)
    with_email = sum(
        1
        for lead in leads
        if lead.get("emails_found") or any(dm.get("email") for dm in lead.get("decision_makers", []))
    )
    with_social = sum(1 for lead in leads if lead.get("social_links"))
    with_instagram = sum(1 for lead in leads if lead.get("social_links", {}).get("instagram"))
    with_linkedin = sum(1 for lead in leads if lead.get("linkedin_url") or lead.get("social_links", {}).get("linkedin"))
    with_decision_makers = sum(1 for lead in leads if lead.get("decision_makers"))
    with_website = sum(1 for lead in leads if lead.get("website"))

    high_fit = sum(1 for lead in leads if lead.get("fit_score", 0) >= 70)
    medium_fit = sum(1 for lead in leads if 50 <= lead.get("fit_score", 0) < 70)
    low_fit = sum(1 for lead in leads if lead.get("fit_score", 0) < 50)

    return jsonify(
        {
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
            "score_distribution": {"high": high_fit, "medium": medium_fit, "low": low_fit},
            "enrichment_stats": stats,
        }
    )


# --- Search ---


@app.route("/api/search", methods=["POST"])
def start_search():
    """Start a new background search job."""
    config = request.json
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    thread = threading.Thread(
        target=search_service.run_search, args=(job_id, config, _api_keys(), job_mgr), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def get_status(job_id):
    """Get status of a search job."""
    job = job_mgr.get(job_id)
    if job is not None:
        return jsonify(job)
    return jsonify({"status": "not_found"}), 404


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel_search(job_id):
    """Cancel a running search job."""
    if job_mgr.exists(job_id):
        job_mgr.set_cancelled(job_id)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Job not found"}), 404


@app.route("/api/people-search", methods=["POST"])
def start_people_search():
    """Start a background people search job (Apollo mixed_people/search)."""
    config = request.json
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_people"

    thread = threading.Thread(
        target=people_service.run_people_search,
        args=(job_id, config, _api_keys(), job_mgr),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/export/<job_id>")
def export_csv(job_id):
    """Export leads as CSV.
    ?mode=prospect — company-level
    ?mode=enriched — contact-level (default)
    ?mode=people   — people search results
    """
    leads = job_mgr.get_leads(job_id)
    if not leads:
        return "Not found", 404

    mode = request.args.get("mode", "prospect")
    if mode == "people":
        csv_data = export_service.export_people_csv(leads)
        prefix = "people"
    else:
        prefix = "prospects" if mode == "prospect" else "leads"
        csv_data = export_service.export_leads_csv(leads, mode=mode)
    return send_file(csv_data, mimetype="text/csv", as_attachment=True, download_name=f"{prefix}_{job_id}.csv")


# --- GTM Brief ---


@app.route("/api/brief", methods=["POST"])
def generate_brief():
    """Generate comprehensive GTM Brief using Firecrawl + OpenAI."""
    result = brief_service.generate_brief(request.json, FIRECRAWL_API_KEY, OPENAI_API_KEY)
    return jsonify(result)


# --- Discovery ---


@app.route("/api/discover", methods=["POST"])
def discover_prospects():
    """Discover prospects using various methods."""
    try:
        result = discovery_service.discover_prospects(request.json, _api_keys())
        return jsonify(result)
    except Exception as e:
        log_dev("DISCOVER", f"Error discovering prospects: {e}", "error")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/discover-and-enrich", methods=["POST"])
def discover_and_enrich():
    """Discover prospects and enrich them — background job, same output as /api/search."""
    config = request.json
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    thread = threading.Thread(
        target=enrichment_service.run_discover_and_enrich,
        args=(job_id, config, _api_keys(), job_mgr),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


# --- Enrichment ---


@app.route("/api/get-socials", methods=["POST"])
def get_socials():
    """Find LinkedIn/Twitter URLs for a list of people from a CSV upload.
    Expects JSON: {people: [{first_name, last_name, title, company}, ...]}
    """
    data = request.json
    people = data.get("people", [])
    if not people or not isinstance(people, list):
        return jsonify({"success": False, "error": "people array is required"})
    if len(people) > 200:
        return jsonify({"success": False, "error": "Maximum 200 people per request"})
    result = enrichment_service.get_socials(people, _api_keys())
    return jsonify(result)


@app.route("/api/enrich-contacts", methods=["POST"])
def enrich_contacts():
    """Find decision makers at a company via Apollo."""
    result = enrichment_service.enrich_contacts(request.json, APOLLO_API_KEY)
    return jsonify(result)


@app.route("/api/enrich", methods=["POST"])
def enrich_company():
    """Enrich a single company by domain."""
    domain = request.json.get("domain", "").strip()
    result = enrichment_service.enrich_company(domain, _api_keys())
    return jsonify(result)


@app.route("/api/enrich-batch", methods=["POST"])
def enrich_batch():
    """Enrich multiple companies in batch."""
    data = request.json
    domains = data.get("domains", [])

    if not domains or not isinstance(domains, list):
        return jsonify({"success": False, "error": "Domains array is required"})

    result = enrichment_service.enrich_batch(domains, _api_keys())
    return jsonify(result)


# --- Hunter.io direct endpoints ---


@app.route("/api/hunter/domain-search", methods=["POST"])
def hunter_domain_search():
    """Search for emails at a domain using Hunter.io."""
    if not HUNTER_API_KEY:
        return jsonify({"success": False, "error": "Hunter.io API not configured"})

    data = request.json
    domain = clean_domain(data.get("domain", ""))

    if not domain:
        return jsonify({"success": False, "error": "Domain is required"})

    try:
        hunter = HunterClient(HUNTER_API_KEY, log_fn=log_dev)
        result = hunter.domain_search(domain, limit=data.get("limit", 10))

        if result:
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "domain": domain,
                        "organization": result.get("organization"),
                        "emails": result.get("emails", []),
                        "total": result.get("total", 0),
                        "pattern": result.get("pattern"),
                    },
                }
            )
        return jsonify({"success": False, "error": "No results found"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/hunter/email-finder", methods=["POST"])
def hunter_email_finder():
    """Find a specific person's email using Hunter.io."""
    if not HUNTER_API_KEY:
        return jsonify({"success": False, "error": "Hunter.io API not configured"})

    data = request.json
    domain = clean_domain(data.get("domain", ""))
    first_name = data.get("first_name", "").strip()
    last_name = data.get("last_name", "").strip()

    if not domain:
        return jsonify({"success": False, "error": "Domain is required"})

    try:
        hunter = HunterClient(HUNTER_API_KEY, log_fn=log_dev)
        result = hunter.email_finder(domain, first_name, last_name)

        if result:
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "email": result.get("email"),
                        "confidence": result.get("score", 0),
                        "first_name": result.get("first_name"),
                        "last_name": result.get("last_name"),
                        "position": result.get("position"),
                        "domain": domain,
                    },
                }
            )
        return jsonify({"success": False, "error": "Email not found"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/hunter/verify", methods=["POST"])
def hunter_verify_email():
    """Verify an email address using Hunter.io."""
    if not HUNTER_API_KEY:
        return jsonify({"success": False, "error": "Hunter.io API not configured"})

    data = request.json
    email = data.get("email", "").strip()

    if not email:
        return jsonify({"success": False, "error": "Email is required"})

    try:
        hunter = HunterClient(HUNTER_API_KEY, log_fn=log_dev)
        result = hunter.email_verifier(email)

        if result:
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "email": email,
                        "status": result.get("status"),
                        "result": result.get("result"),
                        "score": result.get("score", 0),
                        "regexp": result.get("regexp"),
                        "gibberish": result.get("gibberish"),
                        "disposable": result.get("disposable"),
                        "webmail": result.get("webmail"),
                        "mx_records": result.get("mx_records"),
                        "smtp_server": result.get("smtp_server"),
                        "smtp_check": result.get("smtp_check"),
                        "accept_all": result.get("accept_all"),
                        "block": result.get("block"),
                    },
                }
            )
        return jsonify({"success": False, "error": "Verification failed"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🔥 LeadFlow")
    print("=" * 50)
    print("\nOpen http://localhost:5001 in your browser")
    print("\nAPI Keys configured:")
    print(f"  Google Places: {'✓' if GOOGLE_PLACES_API_KEY else '✗ (required)'}")
    print(f"  Apollo.io:     {'✓' if APOLLO_API_KEY else '✗ (optional)'}")
    print(f"  Firecrawl:     {'✓' if FIRECRAWL_API_KEY else '✗ (optional)'}")
    print(f"  Hunter.io:     {'✓' if HUNTER_API_KEY else '✗ (email finding)'}")
    print(f"  OpenAI:        {'✓' if OPENAI_API_KEY else '✗ (for GTM Brief)'}")
    print(f"  Instantly:     {'✓' if INSTANTLY_API_KEY else '✗ (for outreach)'}")
    print("\n" + "=" * 50 + "\n")

    app.run(debug=True, port=5001, host="0.0.0.0")
