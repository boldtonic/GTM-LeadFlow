"""People search pipeline using Apollo /mixed_people/search."""

from dataclasses import asdict

from api_clients import ApolloClient
from models import Person
from utils import log_dev

SENIORITY_MAP = {
    "Entry": "entry",
    "Senior": "senior",
    "Manager": "manager",
    "Director": "director",
    "VP": "vp",
    "C-Suite": "c_suite",
    "Owner": "owner",
}

SIZE_MAP = {
    "1-10": "1,10",
    "11-50": "11,50",
    "51-200": "51,200",
    "201-1K": "201,1000",
    "1K+": "1001,10000",
}

HIGH_SENIORITY = {"owner", "founder", "c_suite", "partner", "vp", "director"}


def run_people_search(job_id: str, config: dict, api_keys: dict, job_mgr):
    job_mgr.create(
        job_id,
        {
            "status": "running",
            "step": "Initializing...",
            "progress": 0,
            "leads": [],
            "errors": [],
            "cancelled": False,
            "job_type": "people",
        },
    )
    try:
        apollo = ApolloClient(api_keys.get("apollo", ""), log_fn=log_dev)
        if not apollo.is_configured:
            job_mgr.update(job_id, status="error", step="Apollo API key required for People Search")
            return

        # Parse config
        raw_titles = config.get("titles", "")
        titles = [t.strip() for t in raw_titles.split(",") if t.strip()]
        seniorities = [SENIORITY_MAP[s] for s in config.get("seniorities", []) if s in SENIORITY_MAP]
        departments = [d.lower().replace(" ", "_") for d in config.get("departments", [])]
        location = config.get("location", "")
        size_labels = config.get("company_size_ranges", [])
        employee_ranges = [SIZE_MAP[s] for s in size_labels if s in SIZE_MAP]
        max_leads = min(config.get("max_leads", 25), 100)

        if not titles and not seniorities and not departments and not location:
            job_mgr.update(job_id, status="error", step="Please enter a title, location, or select a seniority / department filter")
            return

        person_locations = [location] if location else None
        job_mgr.update(job_id, step="Searching Apollo...", progress=20)

        people_raw = apollo.search_people(
            titles=titles,
            seniorities=seniorities or None,
            departments=departments or None,
            person_locations=person_locations,
            employee_ranges=employee_ranges or None,
            per_page=max_leads,
        )

        log_dev("PEOPLE", f"Apollo returned {len(people_raw)} results", "info")
        job_mgr.update(job_id, step=f"Processing {len(people_raw)} results...", progress=60)

        persons = []
        for p in people_raw:
            if job_mgr.is_cancelled(job_id):
                return
            org = p.get("organization") or {}
            website = org.get("website_url", "")
            domain = (
                website.replace("https://", "").replace("http://", "").split("/")[0]
                if website
                else ""
            )
            person = Person(
                id=p.get("id", ""),
                first_name=p.get("first_name", ""),
                last_name=p.get("last_name", ""),
                name=p.get("name", "")
                or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                title=p.get("title", ""),
                seniority=p.get("seniority", ""),
                departments=p.get("departments", []),
                city=p.get("city", ""),
                state=p.get("state", ""),
                country=p.get("country", ""),
                email=p.get("email", ""),
                linkedin_url=p.get("linkedin_url", ""),
                company_name=org.get("name", ""),
                company_website=website,
                company_domain=domain,
                company_industry=org.get("industry", ""),
                company_size=str(org.get("estimated_num_employees", "")),
                company_linkedin=org.get("linkedin_url", ""),
                source_query=raw_titles,
            )
            score, reasons = _score_person(person)
            person.fit_score = score
            person.fit_reasons = reasons
            persons.append(person)

        persons.sort(key=lambda x: x.fit_score, reverse=True)
        job_mgr.update(
            job_id,
            leads=[asdict(p) for p in persons],
            status="completed",
            step=f"Found {len(persons)} people",
            progress=100,
        )

    except Exception as e:
        log_dev("PEOPLE", f"Pipeline error: {e}", "error")
        job_mgr.update(job_id, status="error", step=f"Error: {str(e)}", errors=[str(e)])


def _score_person(person: Person) -> tuple:
    score = 40  # base
    reasons = []

    if person.seniority in HIGH_SENIORITY:
        score += 20
        reasons.append(f"Senior role ({person.seniority})")

    if person.linkedin_url:
        score += 10
        reasons.append("Has LinkedIn")

    if person.company_name:
        score += 5
        reasons.append("Company identified")

    if person.company_website:
        score += 5
        reasons.append("Company has website")

    return min(score, 100), reasons
