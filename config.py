"""
Configuration for GTM Lead Finder
Add your API keys here
"""

# Google Places API
# Get from: https://console.cloud.google.com/apis/credentials
GOOGLE_PLACES_API_KEY = "YOUR_GOOGLE_PLACES_API_KEY"

# Apollo.io API
# Get from: https://app.apollo.io/#/settings/integrations/api
APOLLO_API_KEY = "YOUR_APOLLO_API_KEY"

# Optional: Hunter.io for additional email finding
# Get from: https://hunter.io/api
HUNTER_API_KEY = ""

# Optional: Firecrawl for website scraping
# Get from: https://firecrawl.dev
FIRECRAWL_API_KEY = ""

# Rate limiting (requests per second)
RATE_LIMITS = {
    "google_places": 10,  # Google allows up to 100 QPS
    "apollo": 5,          # Apollo is more restrictive
    "hunter": 10,
    "new_api": 10,        # Placeholder for new API integration
}

# Output settings
OUTPUT_DIR = "output"
