"""Website scraper with Firecrawl + basic fallback."""

import re

import requests

from .firecrawl import FirecrawlClient


class WebsiteScraper:
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    SOCIAL_PATTERNS = {
        "instagram": re.compile(r"instagram\.com/([a-zA-Z0-9_.]+)"),
        "facebook": re.compile(r"facebook\.com/([a-zA-Z0-9_.]+)"),
        "linkedin": re.compile(r"linkedin\.com/company/([a-zA-Z0-9_-]+)"),
        "twitter": re.compile(r"(?:twitter|x)\.com/([a-zA-Z0-9_]+)"),
    }

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

    def scrape(self, url: str) -> dict:
        result = {"emails": [], "social_links": {}, "description": "", "scrape_method": "basic"}

        # Try Firecrawl first (better results)
        if self.firecrawl:
            fc_result = self.firecrawl.scrape(url)
            if fc_result:
                result["scrape_method"] = "firecrawl"
                markdown = fc_result.get("markdown", "")
                links = fc_result.get("links", [])

                # Extract emails from markdown
                emails = list(set(self.EMAIL_PATTERN.findall(markdown)))
                result["emails"] = [
                    e
                    for e in emails
                    if not any(x in e.lower() for x in ["example", "noreply", "wordpress", "wixpress"])
                ][:5]

                # Extract social links
                all_text = markdown + " " + " ".join(links)
                for platform, pattern in self.SOCIAL_PATTERNS.items():
                    matches = pattern.findall(all_text)
                    if matches:
                        result["social_links"][platform] = matches[0]

                # Extract description (first substantial paragraph)
                paragraphs = markdown.split("\n\n")
                for p in paragraphs:
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

            for platform, pattern in self.SOCIAL_PATTERNS.items():
                matches = pattern.findall(all_text)
                if matches:
                    result["social_links"][platform] = matches[0]

        except Exception as e:
            self._log(f"Scrape error for {url}: {e}", "error")

        return result
