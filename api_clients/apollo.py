"""Apollo.io API client."""

from .base import BaseAPIClient


class ApolloClient(BaseAPIClient):
    BASE_URL = "https://api.apollo.io/v1"

    def __init__(self, api_key: str, log_fn=None):
        super().__init__(api_key, log_fn, category="APOLLO")
        self.headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
        }

    def enrich_organization(self, domain: str) -> dict:
        """Enrich a company by domain."""
        if not self.is_configured:
            return {}

        data = self._get("/organizations/enrich", params={"domain": domain}, timeout=10)
        return data.get("organization", {}) if data else {}

    def search_contacts(
        self,
        domain: str = None,
        titles: list = None,
        seniorities: list = None,
        organization_ids: list = None,
        page: int = 1,
        per_page: int = 5,
    ) -> list:
        """Search for people at a company."""
        if not self.is_configured:
            return []

        payload = {"page": page, "per_page": per_page}
        if domain:
            payload["q_organization_domains"] = domain
        if titles:
            payload["person_titles"] = titles
        if seniorities:
            payload["person_seniorities"] = seniorities
        if organization_ids:
            payload["organization_ids"] = organization_ids

        data = self._post("/mixed_people/search", payload=payload, timeout=10)
        return data.get("people", []) if data else []

    def find_person(self, first_name: str, last_name: str, company_name: str, title: str = None) -> dict:
        """Find a specific person by name + company via /people/match. Returns full Apollo record (linkedin_url, etc)."""
        if not self.is_configured:
            return {}

        params = {}
        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name
        if company_name:
            params["organization_name"] = company_name

        data = self._post("/people/match", params=params, timeout=10)
        return data.get("person", {}) if data else {}

    def search_organizations(
        self,
        name: str = None,
        domain: str = None,
        location: str = None,
        employee_ranges: list = None,
        page: int = 1,
        per_page: int = 25,
    ) -> list:
        """Search Apollo's organization database with filters."""
        if not self.is_configured:
            return []

        payload = {"page": page, "per_page": min(per_page, 100)}
        if name:
            payload["q_organization_name"] = name
        if domain:
            payload["q_organization_domains"] = [domain] if isinstance(domain, str) else domain
        if location:
            payload["organization_locations"] = [location] if isinstance(location, str) else location
        if employee_ranges:
            payload["organization_num_employees_ranges"] = employee_ranges

        data = self._post("/mixed_companies/search", payload=payload, timeout=15)
        if data:
            orgs = data.get("organizations", [])
            self._log(f"Org search returned {len(orgs)} results", "info")
            return orgs
        return []
