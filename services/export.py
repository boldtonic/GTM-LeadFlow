"""
CSV export service for GTM LeadFlow.
Two formats: prospect (company-level) and enriched (contact-level).
"""

import csv
import io


def _resolve_domain(lead):
    """Get domain from lead, falling back to website."""
    domain = lead.get("domain", "")
    if not domain:
        website = lead.get("website", "")
        if website:
            domain = website.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/").split("/")[0]
    return domain


def export_leads_csv(leads: list, mode: str = "enriched") -> io.BytesIO:
    """Generate CSV from leads list. Returns BytesIO ready for send_file.

    mode='prospect' — company-level, one row per company (21 cols)
    mode='enriched' — contact-level, one row per decision maker (35 cols)
    """
    output = io.StringIO()
    writer = csv.writer(output)

    if mode == "prospect":
        _write_prospect_csv(writer, leads)
    else:
        _write_enriched_csv(writer, leads)

    output.seek(0)
    return io.BytesIO(output.getvalue().encode("utf-8"))


def _write_prospect_csv(writer, leads):
    """Company-level CSV for review. One row per company."""
    writer.writerow(
        [
            "enrichment_grade",
            "company_name",
            "domain",
            "website",
            "category",
            "industry",
            "city",
            "country",
            "address",
            "phone",
            "rating",
            "reviews",
            "company_size",
            "estimated_revenue",
            "linkedin_url",
            "founded_year",
            "email_count",
            "has_website",
            "fit_score",
            "fit_reasons",
            "google_maps_url",
            "source",
            "source_query",
            "discovered_at",
        ]
    )

    for lead in leads:
        writer.writerow(
            [
                lead.get("enrichment_grade", ""),
                lead.get("name", ""),
                _resolve_domain(lead),
                lead.get("website", ""),
                lead.get("category", ""),
                lead.get("industry", ""),
                lead.get("city", ""),
                lead.get("country", ""),
                lead.get("address_full", ""),
                lead.get("phone", ""),
                lead.get("rating", 0),
                lead.get("reviews_count", 0),
                lead.get("company_size", ""),
                lead.get("estimated_revenue", ""),
                lead.get("linkedin_url", ""),
                lead.get("founded_year", ""),
                lead.get("email_count", 0),
                lead.get("has_website", False),
                lead.get("fit_score", 0),
                "; ".join(lead.get("fit_reasons", [])),
                lead.get("google_maps_url", ""),
                lead.get("source", ""),
                lead.get("source_query", ""),
                lead.get("scraped_at", ""),
            ]
        )


def _write_enriched_csv(writer, leads):
    """Contact-level CSV for outreach tools. One row per decision maker.
    Companies with zero contacts still get one row."""
    writer.writerow(
        [
            # Contact
            "first_name",
            "last_name",
            "email",
            "email_status",
            "email_confidence",
            "email_type",
            "title",
            "seniority",
            "contact_linkedin",
            "contact_phone",
            "email_source",
            # Company
            "company_name",
            "domain",
            "website",
            "category",
            "industry",
            "city",
            "country",
            "address",
            "phone",
            "rating",
            "reviews",
            "company_size",
            "estimated_revenue",
            "linkedin_url",
            "founded_year",
            "instagram",
            "facebook",
            "twitter",
            "about",
            # Meta
            "enrichment_grade",
            "email_count",
            "fit_score",
            "fit_reasons",
            "google_maps_url",
            "source_query",
            "enriched_at",
        ]
    )

    for lead in leads:
        domain = _resolve_domain(lead)
        social = lead.get("social_links", {})
        dms = lead.get("decision_makers", [])

        # Shared company fields
        company_row = [
            lead.get("name", ""),
            domain,
            lead.get("website", ""),
            lead.get("category", ""),
            lead.get("industry", ""),
            lead.get("city", ""),
            lead.get("country", ""),
            lead.get("address_full", ""),
            lead.get("phone", ""),
            lead.get("rating", 0),
            lead.get("reviews_count", 0),
            lead.get("company_size", ""),
            lead.get("estimated_revenue", ""),
            lead.get("linkedin_url", "") or social.get("linkedin", ""),
            lead.get("founded_year", ""),
            social.get("instagram", ""),
            social.get("facebook", ""),
            social.get("twitter", ""),
            (lead.get("about_text", "") or "")[:200],
        ]

        meta_row = [
            lead.get("enrichment_grade", ""),
            lead.get("email_count", 0),
            lead.get("fit_score", 0),
            "; ".join(lead.get("fit_reasons", [])),
            lead.get("google_maps_url", ""),
            lead.get("source_query", ""),
            lead.get("enriched_at", ""),
        ]

        if dms:
            # One row per decision maker
            for dm in dms:
                first = dm.get("first_name", "")
                last = dm.get("last_name", "")
                # Split name if first/last not stored
                if not first and not last and dm.get("name"):
                    parts = dm["name"].split(" ", 1)
                    first = parts[0]
                    last = parts[1] if len(parts) > 1 else ""

                contact_row = [
                    first,
                    last,
                    dm.get("email", ""),
                    dm.get("email_status", ""),
                    dm.get("confidence", "") or lead.get("email_confidence", ""),
                    dm.get("email_type", ""),
                    dm.get("title", ""),
                    dm.get("seniority", ""),
                    dm.get("linkedin", ""),
                    dm.get("phone", ""),
                    dm.get("source", ""),
                ]
                writer.writerow(contact_row + company_row + meta_row)
        else:
            # Company with no contacts — still one row with best email
            emails = lead.get("emails_found", [])
            contact_row = [
                "",  # first_name
                "",  # last_name
                emails[0] if emails else "",
                lead.get("email_status", ""),
                lead.get("email_confidence", ""),
                "",  # email_type
                "",  # title
                "",  # seniority
                "",  # contact_linkedin
                "",  # contact_phone
                "",  # email_source
            ]
            writer.writerow(contact_row + company_row + meta_row)
