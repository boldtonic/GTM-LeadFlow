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
