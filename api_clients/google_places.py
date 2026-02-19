"""Google Places API client."""

import time

from .base import BaseAPIClient


class GooglePlacesClient(BaseAPIClient):
    BASE_URL = "https://maps.googleapis.com/maps/api/place"

    def __init__(self, api_key: str, log_fn=None):
        super().__init__(api_key, log_fn, category="GOOGLE")

    def _get(self, path, params=None, timeout=None):
        """Override to inject API key as query parameter."""
        params = params or {}
        params["key"] = self.api_key
        return super()._get(path, params, timeout)

    def text_search(self, query: str) -> list:
        """Search Google Places by text query. Handles pagination."""
        if not self.is_configured:
            return []

        params = {"query": query}
        results = []
        next_page_token = None

        while True:
            if next_page_token:
                params["pagetoken"] = next_page_token
                time.sleep(2)

            data = self._get("/textsearch/json", params=params, timeout=10)
            if not data or data.get("status") not in ["OK", "ZERO_RESULTS"]:
                break

            results.extend(data.get("results", []))
            next_page_token = data.get("next_page_token")

            if not next_page_token or len(results) >= 60:
                break

        return results

    def get_place_details(self, place_id: str) -> dict:
        """Get detailed info for a specific place."""
        if not self.is_configured:
            return {}

        data = self._get(
            "/details/json",
            params={
                "place_id": place_id,
                "fields": "place_id,name,formatted_address,formatted_phone_number,"
                "international_phone_number,website,url,rating,"
                "user_ratings_total,price_level,opening_hours,types,"
                "address_components,geometry",
            },
            timeout=10,
        )

        if data and data.get("status") == "OK":
            return data.get("result", {})
        return {}
