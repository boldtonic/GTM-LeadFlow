"""Firecrawl API client for web scraping."""

from .base import BaseAPIClient


class FirecrawlClient(BaseAPIClient):
    BASE_URL = "https://api.firecrawl.dev/v1"

    def __init__(self, api_key: str, log_fn=None):
        super().__init__(api_key, log_fn, category="FIRECRAWL")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def scrape(self, url: str, only_main_content: bool = True) -> dict:
        """Scrape a single URL and return markdown + metadata.
        Set only_main_content=False to include headers/footers (needed for social link extraction)."""
        if not self.is_configured:
            return {}

        data = self._post(
            "/scrape",
            payload={"url": url, "formats": ["markdown", "links"], "onlyMainContent": only_main_content},
            timeout=30,
        )

        return data.get("data", {}) if data else {}

    def search(self, query: str, limit: int = 10) -> list:
        """Search the web and return results with scraped content."""
        if not self.is_configured:
            return []

        data = self._post(
            "/search",
            payload={"query": query, "limit": limit, "scrapeOptions": {"formats": ["markdown", "links"]}},
            timeout=30,
        )

        return data.get("data", []) if data else []
