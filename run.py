#!/usr/bin/env python3
"""
GTM Lead Finder - Quick Runner

Usage:
    python run.py                           # Use default brief template
    python run.py briefs/andres_machado_poland.yaml  # Use specific brief
    python run.py briefs/client.yaml --test # Test mode (5 leads only)
    python run.py briefs/client.yaml --no-enrich  # Skip enrichment
"""

import argparse
from lead_finder import GTMLeadFinder


def main():
    parser = argparse.ArgumentParser(description="GTM Lead Finder")
    parser.add_argument(
        "brief",
        nargs="?",
        default="brief_template.yaml",
        help="Path to client brief YAML file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - process only 5 leads"
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip enrichment (faster, less API calls)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of leads to process"
    )

    args = parser.parse_args()

    limit = 5 if args.test else args.limit
    enrich = not args.no_enrich

    print(f"\nLoading brief: {args.brief}")
    print(f"Enrichment: {'ON' if enrich else 'OFF'}")
    if limit:
        print(f"Limit: {limit} leads")

    finder = GTMLeadFinder(args.brief)
    finder.run(enrich=enrich, limit=limit)


if __name__ == "__main__":
    main()
