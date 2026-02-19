"""Hunter.io API client for email finding and verification."""

from .base import BaseAPIClient


class HunterClient(BaseAPIClient):
    BASE_URL = "https://api.hunter.io/v2"

    def __init__(self, api_key: str, log_fn=None):
        super().__init__(api_key, log_fn, category="HUNTER")

    def _get(self, path, params=None, timeout=None):
        """Override to inject API key as query parameter."""
        params = params or {}
        params["api_key"] = self.api_key
        return super()._get(path, params, timeout)

    def domain_search(self, domain: str, limit: int = 10) -> dict:
        """Find emails associated with a domain."""
        if not self.is_configured:
            return {}

        data = self._get("/domain-search", params={"domain": domain, "limit": limit}, timeout=10)

        return data.get("data", {}) if data else {}

    def email_finder(self, domain: str, first_name: str = None, last_name: str = None) -> dict:
        """Find a specific person's email at a company."""
        if not self.is_configured:
            return {}

        params = {"domain": domain}
        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name

        data = self._get("/email-finder", params=params, timeout=10)
        return data.get("data", {}) if data else {}

    def email_verifier(self, email: str) -> dict:
        """Verify if an email address is valid and deliverable."""
        if not self.is_configured:
            return {}

        data = self._get("/email-verifier", params={"email": email}, timeout=10)
        return data.get("data", {}) if data else {}

    def email_count(self, domain: str) -> dict:
        """Get the number of emails found for a domain (free endpoint)."""
        if not self.is_configured:
            return {}

        data = self._get("/email-count", params={"domain": domain}, timeout=10)
        return data.get("data", {}) if data else {}
