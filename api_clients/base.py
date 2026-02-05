"""
Base API Client for GTM LeadFlow.
All new API clients should inherit from this class.
"""

import requests
import time
from datetime import datetime


class BaseAPIClient:
    """
    Base class for all API clients.
    Provides common patterns: auth, logging, error handling, rate limiting.

    Usage:
        class NewClient(BaseAPIClient):
            BASE_URL = "https://api.example.com/v1"

            def __init__(self, api_key: str, log_fn=None):
                super().__init__(api_key, log_fn, category="NEW_API")
                self.headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

            def some_endpoint(self, param: str) -> dict:
                return self._get("/endpoint", params={"q": param})
    """

    BASE_URL: str = ""
    DEFAULT_TIMEOUT: int = 15

    def __init__(self, api_key: str, log_fn=None, category: str = "API"):
        self.api_key = api_key
        self.log_fn = log_fn
        self.category = category
        self.headers = {"Content-Type": "application/json"}

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _log(self, message: str, level: str = "info"):
        if self.log_fn:
            self.log_fn(self.category, message, level)

    def _get(self, path: str, params: dict = None, timeout: int = None) -> dict:
        if not self.is_configured:
            return {}
        try:
            response = requests.get(
                f"{self.BASE_URL}{path}",
                headers=self.headers,
                params=params or {},
                timeout=timeout or self.DEFAULT_TIMEOUT,
            )
            data = response.json()
            if not response.ok:
                self._log(f"GET {path} failed: {data.get('error', response.status_code)}", "error")
                return {}
            return data
        except Exception as e:
            self._log(f"GET {path} exception: {str(e)[:80]}", "error")
            return {}

    def _post(self, path: str, payload: dict = None, timeout: int = None) -> dict:
        if not self.is_configured:
            return {}
        try:
            response = requests.post(
                f"{self.BASE_URL}{path}",
                headers=self.headers,
                json=payload or {},
                timeout=timeout or self.DEFAULT_TIMEOUT,
            )
            data = response.json()
            if not response.ok:
                self._log(f"POST {path} failed: {data.get('error', response.status_code)}", "error")
                return {}
            return data
        except Exception as e:
            self._log(f"POST {path} exception: {str(e)[:80]}", "error")
            return {}
