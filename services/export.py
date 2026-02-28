"""
CSV export service for GTM LeadFlow.
Two formats: prospect (company-level) and enriched (contact-level).
People search has its own 9-column format.
"""

import csv
import io


def export_people_csv(people: list) -> io.BytesIO:
    """Generate CSV from people search results. 9 columns, email excluded."""
    cols = [
        "name", "title", "city", "country", "company_name",
        "linkedin_url", "fit_score", "fit_reasons", "source_query",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for person in people:
        row = {col: person.get(col, "") for col in cols}
        row["fit_reasons"] = "; ".join(person.get("fit_reasons", []))
        writer.writerow(row)
    output.seek(0)
    return io.BytesIO(output.getvalue().encode("utf-8"))


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
    """Company-level CSV. One row per company, includes primary email + first decision maker."""
    writer.writerow(
        [
            # Core identity
            "name",
            "industry",
            "full_address",
            "street_address",
            "city",
            "country",
            "zip",
            "municipality",
            "website",
            "linkedin_url",
            "instagram_url",
            "facebook_url",
            # Signals
            "reviews",
            "rating",
            "company_size",
            "fit_score",
            "fit_reasons",
            # Meta
            "google_maps_url",
            "source_query",
        ]
    )

    for lead in leads:
        social = lead.get("social_links", {})

        writer.writerow(
            [
                lead.get("name", ""),
                lead.get("industry", "") or lead.get("category", ""),
                lead.get("address_full", ""),
                lead.get("street", ""),
                lead.get("city", ""),
                lead.get("country", ""),
                lead.get("postal_code", ""),
                lead.get("municipality", ""),
                lead.get("website", ""),
                lead.get("linkedin_url", "") or social.get("linkedin", ""),
                social.get("instagram", ""),
                social.get("facebook", ""),
                lead.get("reviews_count", 0),
                lead.get("rating", 0),
                lead.get("company_size", ""),
                lead.get("fit_score", 0),
                "; ".join(lead.get("fit_reasons", [])),
                lead.get("google_maps_url", ""),
                lead.get("source_query", ""),
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
