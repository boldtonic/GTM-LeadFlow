"""
Smoke tests for Fogata endpoints.
Uses Flask test client — no external API calls, no running server needed.
"""

import pytest

from app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# --- Static / GET endpoints ---


def test_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Fogata" in resp.data


def test_config(client):
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "google_places" in data
    assert "apollo" in data
    assert "hunter" in data
    assert "firecrawl" in data
    assert "openai" in data
    assert "instantly" in data


def test_dev_logs(client):
    resp = client.get("/api/dev/logs")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "logs" in data
    assert isinstance(data["logs"], list)


def test_dev_stats(client):
    resp = client.get("/api/dev/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "hasData" in data


def test_status_not_found(client):
    resp = client.get("/api/status/nonexistent_job")
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["status"] == "not_found"


def test_cancel_not_found(client):
    resp = client.post("/api/cancel/nonexistent_job")
    assert resp.status_code == 404


def test_export_not_found(client):
    resp = client.get("/api/export/nonexistent_job")
    assert resp.status_code == 404


# --- POST endpoints (validation / empty-input paths) ---


def test_discover_and_enrich_starts_job(client):
    resp = client.post(
        "/api/discover-and-enrich",
        json={"mode": "smart_search", "query": "test", "max_leads": 1},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "job_id" in data


def test_brief_missing_url(client):
    resp = client.post("/api/brief", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is False
    assert "URL" in data["error"] or "url" in data["error"].lower()


def test_enrich_missing_domain(client):
    resp = client.post("/api/enrich", json={"domain": ""})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is False


def test_enrich_batch_missing_domains(client):
    resp = client.post("/api/enrich-batch", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is False


def test_hunter_domain_search_missing(client):
    resp = client.post("/api/hunter/domain-search", json={"domain": ""})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is False


def test_hunter_email_finder_missing(client):
    resp = client.post("/api/hunter/email-finder", json={"domain": ""})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is False


def test_hunter_verify_missing(client):
    resp = client.post("/api/hunter/verify", json={"email": ""})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is False


def test_clear_dev_logs(client):
    resp = client.post("/api/dev/logs/clear")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True


# --- Unit tests for shared utilities ---


def test_extract_domain():
    from utils import extract_domain

    assert extract_domain("https://www.example.com/page") == "example.com"
    assert extract_domain("http://example.com") == "example.com"
    assert extract_domain("") == ""


def test_clean_domain():
    from utils import clean_domain

    assert clean_domain("https://www.example.com/path") == "example.com"
    assert clean_domain("example.com") == "example.com"
    assert clean_domain("") == ""


def test_is_business_domain():
    from utils import is_business_domain

    assert is_business_domain("acme.com") is True
    assert is_business_domain("facebook.com") is False
    assert is_business_domain("instagram.com") is False


def test_lead_dataclass():
    from models import Lead

    lead = Lead(place_id="abc123", google_maps_url="", name="Test Co", category="Test")
    assert lead.name == "Test Co"
    assert lead.fit_score == 0
    assert lead.decision_makers == []


def test_lead_scorer():
    from models import Lead
    from scoring import LeadScorer

    scorer = LeadScorer({"min_rating": 4.0, "segment": "mid-premium"})
    lead = Lead(
        place_id="x",
        google_maps_url="",
        name="Test",
        category="Shop",
        rating=4.5,
        reviews_count=50,
        website="https://test.com",
    )
    score, reasons = scorer.score(lead)
    assert isinstance(score, int)
    assert 0 <= score <= 100
    assert isinstance(reasons, list)


def test_job_manager():
    from jobs import JobManager

    mgr = JobManager(ttl_seconds=60)
    mgr.create("test1", {"status": "running", "leads": []})

    job = mgr.get("test1")
    assert job is not None
    assert job["status"] == "running"

    mgr.update("test1", status="completed", leads=[{"name": "A"}])
    job = mgr.get("test1")
    assert job["status"] == "completed"
    assert len(job["leads"]) == 1

    assert mgr.get("nonexistent") is None
    assert mgr.is_cancelled("test1") is False

    mgr.set_cancelled("test1")
    assert mgr.is_cancelled("test1") is True
