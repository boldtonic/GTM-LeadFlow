"""
API Clients module for Fogata.
Modular API integrations for lead enrichment and discovery.
"""

from .apollo import ApolloClient as ApolloClient
from .base import BaseAPIClient as BaseAPIClient
from .firecrawl import FirecrawlClient as FirecrawlClient
from .google_places import GooglePlacesClient as GooglePlacesClient
from .hunter import HunterClient as HunterClient
from .scraper import WebsiteScraper as WebsiteScraper
