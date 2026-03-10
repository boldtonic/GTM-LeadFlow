"""
Data models for GTM LeadFlow.
"""

from dataclasses import dataclass, field


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
    municipality: str = ""
    country: str = ""
    postal_code: str = ""
    lat: float = 0.0
    lng: float = 0.0
    phone: str = ""
    website: str = ""
    domain: str = ""
    rating: float = 0.0
    reviews_count: int = 0
    hours: dict = field(default_factory=dict)
    price_level: int = 0
    emails_found: list = field(default_factory=list)
    social_links: dict = field(default_factory=dict)
    brands_carried: list = field(default_factory=list)
    about_text: str = ""
    apollo_org_id: str = ""
    industry: str = ""
    company_size: str = ""
    estimated_revenue: str = ""
    founded_year: str = ""
    linkedin_url: str = ""
    decision_makers: list = field(default_factory=list)
    email_count: int = 0
    has_website: bool = False
    enrichment_grade: str = ""
    email_status: str = ""
    email_confidence: int = 0
    fit_score: int = 0
    fit_reasons: list = field(default_factory=list)
    source: str = ""
    scraped_at: str = ""
    enriched_at: str = ""
    source_query: str = ""


@dataclass
class EnrichedLead:
    """A lead on the Enrichment Kanban board."""

    # Identity
    id: str = ""
    source_type: str = ""           # "company" or "person"
    parent_company_id: str = ""     # non-empty when nested under a company card

    # Original prospect data (full dict as received from Prospecting)
    source_data: dict = field(default_factory=dict)

    # Status
    column: str = "imported"        # "imported" | "contacts_found" | "deep_researched"

    # Contact enrichment layer
    persons: list = field(default_factory=list)         # decision makers found inside a company
    contact_email: str = ""
    contact_phone: str = ""
    all_socials: dict = field(default_factory=dict)
    company_size_verified: str = ""
    contact_enriched_at: str = ""

    # Deep research layer
    linkedin_summary: str = ""
    recent_activity: str = ""
    company_news: str = ""
    pain_points: list = field(default_factory=list)
    outreach_angle: str = ""
    tech_stack: list = field(default_factory=list)
    deep_researched_at: str = ""

    # Organisation
    tags: list = field(default_factory=list)
    notes: str = ""
    activity_log: list = field(default_factory=list)

    # Scoring
    fit_score: int = 0
    enriched_score: int = 0         # updated after enrichment (0 = not yet updated)

    # Meta
    added_at: str = ""
    updated_at: str = ""


@dataclass
class Person:
    """A person found via Apollo People Search."""

    id: str = ""
    first_name: str = ""
    last_name: str = ""
    name: str = ""
    title: str = ""
    seniority: str = ""
    departments: list = field(default_factory=list)
    city: str = ""
    state: str = ""
    country: str = ""
    email: str = ""
    linkedin_url: str = ""
    company_name: str = ""
    company_website: str = ""
    company_domain: str = ""
    company_industry: str = ""
    company_size: str = ""
    company_linkedin: str = ""
    fit_score: int = 0
    fit_reasons: list = field(default_factory=list)
    source_query: str = ""
