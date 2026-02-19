"""
Shared utilities for GTM LeadFlow.
Logging, domain helpers, text cleaning.
"""

import re
import threading
from datetime import datetime
from urllib.parse import urlparse

# --- Dev/Debug logging (thread-safe) ---

dev_logs = []
MAX_LOGS = 500
_log_lock = threading.Lock()


def log_dev(category: str, message: str, level: str = "info"):
    """Add a log entry for dev panel. Thread-safe."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "message": message,
        "level": level,  # info, success, warning, error
    }
    with _log_lock:
        dev_logs.append(entry)
        if len(dev_logs) > MAX_LOGS:
            del dev_logs[: len(dev_logs) - MAX_LOGS]
    print(f"[{level.upper()}] [{category}] {message}")


# --- Domain utilities ---

SKIP_DOMAINS = {
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "tiktok.com",
    "youtube.com",
    "pinterest.com",
    "linktr.ee",
    "linktree.com",
    "bit.ly",
    "goo.gl",
    "yelp.com",
    "tripadvisor.com",
    "tripadvisor.es",
    "google.com",
    "maps.google.com",
    "foursquare.com",
    "booking.com",
    "ubereats.com",
    "deliveroo.com",
    "glovo.com",
    "justeat.com",
    "thefork.com",
    "eltenedor.es",
    "opentable.com",
}


def extract_domain(url: str) -> str:
    """Extract bare domain from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        return domain.replace("www.", "")
    except Exception:
        return ""


def clean_domain(value: str) -> str:
    """Normalize a domain or URL string to bare domain."""
    value = value.strip()
    if "://" in value:
        return extract_domain(value)
    return value.replace("www.", "").rstrip("/")


def is_business_domain(domain: str) -> bool:
    """Return True if domain belongs to actual business, not a social/platform site."""
    if not domain:
        return False
    d = domain.lower().strip()
    for skip in SKIP_DOMAINS:
        if d == skip or d.endswith("." + skip):
            return False
    return True


def clean_markdown_text(text: str) -> str:
    """Remove markdown formatting from text."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
