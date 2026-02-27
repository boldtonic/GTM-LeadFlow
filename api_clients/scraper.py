"""Website scraper with Firecrawl + basic fallback."""

import re

import requests

from .firecrawl import FirecrawlClient


class WebsiteScraper:
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    # Non-profile paths to exclude from social matches
    _IG_SKIP = {"p", "reel", "reels", "explore", "stories", "tv", "accounts", "about", "web", "static"}
    _FB_SKIP = {"sharer", "share", "dialog", "plugins", "tr", "pg", "help", "login", "photo", "video", "events", "groups", "pages"}

    SOCIAL_PATTERNS = {
        # Instagram: min 3-char handle, skip utility paths
        "instagram": (re.compile(r"instagram\.com/([a-zA-Z0-9_.]{3,30})"), "https://instagram.com/"),
        # Facebook: min 3-char handle, skip utility paths
        "facebook": (re.compile(r"facebook\.com/([a-zA-Z0-9_.]{3,})"), "https://facebook.com/"),
        "linkedin": (re.compile(r"linkedin\.com/company/([a-zA-Z0-9_-]+)"), "https://linkedin.com/company/"),
        "twitter": (re.compile(r"(?:twitter|x)\.com/([a-zA-Z0-9_]{2,})"), "https://twitter.com/"),
    }

    @classmethod
    def _filter_social_match(cls, platform: str, handle: str) -> bool:
        """Return True if the handle is a valid profile (not a utility path)."""
        handle_lower = handle.lower().split("?")[0].split(".")[0]
        if platform == "instagram" and handle_lower in cls._IG_SKIP:
            return False
        if platform == "facebook" and handle_lower in cls._FB_SKIP:
            return False
        return True

    def __init__(self, firecrawl_key: str = None, log_fn=None):
        self.firecrawl = FirecrawlClient(firecrawl_key, log_fn=log_fn) if firecrawl_key else None
        self.log_fn = log_fn
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )

    def _log(self, message, level="info"):
        if self.log_fn:
            self.log_fn("SCRAPER", message, level)

    def scrape_social_links(self, url: str) -> dict:
        """Lightweight single-page scrape to extract social links only (1 Firecrawl credit).
        Used at prospect stage — no emails or description extracted."""
        if not (self.firecrawl and self.firecrawl.is_configured):
            return {}
        try:
            fc_result = self.firecrawl.scrape(url)
            if not fc_result:
                return {}
            all_text = fc_result.get("markdown", "") + " " + " ".join(fc_result.get("links", []))
            social_links = {}
            for platform, (pattern, base_url) in self.SOCIAL_PATTERNS.items():
                for handle in pattern.findall(all_text):
                    if self._filter_social_match(platform, handle):
                        social_links[platform] = base_url + handle
                        break
            return social_links
        except Exception as e:
            self._log(f"Social scrape error for {url}: {e}", "warning")
            return {}

    def scrape(self, url: str) -> dict:
        result = {"emails": [], "social_links": {}, "description": "", "scrape_method": "basic"}

        # Try Firecrawl first (better results) — scrape homepage + contact + about
        if self.firecrawl:
            base = url.rstrip("/")
            pages_to_try = [url, f"{base}/contact", f"{base}/about"]
            combined_markdown = ""
            combined_links = []
            result["scrape_method"] = "firecrawl"

            for page_url in pages_to_try:
                fc_result = self.firecrawl.scrape(page_url)
                if fc_result:
                    combined_markdown += "\n\n" + fc_result.get("markdown", "")
                    combined_links += fc_result.get("links", [])
                # Stop early once we have emails — saves API credits
                if page_url == url:
                    early_emails = [
                        e for e in self.EMAIL_PATTERN.findall(combined_markdown)
                        if not any(x in e.lower() for x in ["example", "noreply", "wordpress", "wixpress"])
                    ]
                    if len(early_emails) >= 2:
                        break

            if combined_markdown.strip():
                # Extract emails
                emails = list(set(self.EMAIL_PATTERN.findall(combined_markdown)))
                result["emails"] = [
                    e
                    for e in emails
                    if not any(x in e.lower() for x in ["example", "noreply", "wordpress", "wixpress"])
                ][:5]

                # Extract social links
                all_text = combined_markdown + " " + " ".join(combined_links)
                for platform, (pattern, base_url) in self.SOCIAL_PATTERNS.items():
                    for handle in pattern.findall(all_text):
                        if self._filter_social_match(platform, handle):
                            result["social_links"][platform] = base_url + handle
                            break

                # Extract description from homepage markdown (first substantial paragraph)
                first_page_md = combined_markdown.split("\n\n")
                for p in first_page_md:
                    cleaned = p.strip().replace("#", "").strip()
                    if 50 < len(cleaned) < 500 and not cleaned.startswith("["):
                        result["description"] = cleaned
                        break

                return result

        # Fallback to basic scraping
        try:
            pages = [url, f"{url.rstrip('/')}/contact", f"{url.rstrip('/')}/about"]
            all_text = ""

            for page_url in pages:
                try:
                    resp = self.session.get(page_url, timeout=5)
                    if resp.status_code == 200:
                        all_text += resp.text + "\n"
                except Exception as e:
                    self._log(f"Page fetch error {page_url}: {e}", "warning")
                    continue

            emails = list(set(self.EMAIL_PATTERN.findall(all_text)))
            result["emails"] = [
                e for e in emails if not any(x in e.lower() for x in ["example", "noreply", "wordpress"])
            ][:3]

            for platform, (pattern, base_url) in self.SOCIAL_PATTERNS.items():
                for handle in pattern.findall(all_text):
                    if self._filter_social_match(platform, handle):
                        result["social_links"][platform] = base_url + handle
                        break

        except Exception as e:
            self._log(f"Scrape error for {url}: {e}", "error")

        return result
