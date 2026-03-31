"""Kanban board state management for GTM LeadFlow.
In-memory, thread-safe. Leads persist within a server session.
Migration to SQLite planned when deployed to VPS.
"""

import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from models import EnrichedLead

VALID_COLUMNS = {"imported", "contacts_found", "deep_researched"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KanbanManager:
    def __init__(self):
        self._leads: dict = {}  # id → EnrichedLead
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_log(self, lead: EnrichedLead, action: str, detail: str = ""):
        lead.activity_log.append({"action": action, "timestamp": _now(), "detail": detail})
        lead.updated_at = _now()

    def _prospect_uid(self, source_data: dict) -> str:
        """Return a stable identifier from raw prospect data for deduplication."""
        return (
            source_data.get("id")
            or source_data.get("place_id")
            or source_data.get("company_domain")
            or source_data.get("domain")
            or ""
        )

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def check_duplicates(self, prospect_ids: list) -> list:
        """Return the subset of prospect_ids already present in the Kanban."""
        with self._lock:
            existing = {
                self._prospect_uid(lead.source_data)
                for lead in self._leads.values()
            }
            return [pid for pid in prospect_ids if pid and pid in existing]

    def import_leads(self, prospects: list, source_type: str) -> list:
        """Import a list of raw prospect dicts. Returns list of created lead dicts."""
        created = []
        now = _now()
        with self._lock:
            for p in prospects:
                lead_id = str(uuid.uuid4())
                lead = EnrichedLead(
                    id=lead_id,
                    source_type=source_type,
                    source_data=p,
                    fit_score=p.get("fit_score", 0),
                    added_at=now,
                    updated_at=now,
                )
                self._append_log(lead, "Imported from Prospect", f"Score: {lead.fit_score}")
                self._leads[lead_id] = lead
                created.append(asdict(lead))
        return created

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self, column: str = None, tags: list = None) -> list:
        """Return all leads as dicts, newest first, with optional filters."""
        with self._lock:
            leads = list(self._leads.values())

        if column:
            leads = [lead for lead in leads if lead.column == column]
        if tags:
            leads = [lead for lead in leads if any(t in lead.tags for t in tags)]

        leads.sort(key=lambda lead: lead.added_at, reverse=True)
        return [asdict(lead) for lead in leads]

    def get(self, lead_id: str) -> Optional[dict]:
        with self._lock:
            lead = self._leads.get(lead_id)
            return asdict(lead) if lead else None

    def count_by_column(self) -> dict:
        with self._lock:
            counts = {col: 0 for col in VALID_COLUMNS}
            for lead in self._leads.values():
                if lead.column in counts:
                    counts[lead.column] += 1
            return counts

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_tag(self, lead_id: str, tag: str) -> bool:
        with self._lock:
            lead = self._leads.get(lead_id)
            if not lead:
                return False
            if tag not in lead.tags:
                lead.tags.append(tag)
                self._append_log(lead, f"Tag added: {tag}")
            return True

    def remove_tag(self, lead_id: str, tag: str) -> bool:
        with self._lock:
            lead = self._leads.get(lead_id)
            if not lead:
                return False
            if tag in lead.tags:
                lead.tags.remove(tag)
                self._append_log(lead, f"Tag removed: {tag}")
            return True

    def update_notes(self, lead_id: str, notes: str) -> bool:
        with self._lock:
            lead = self._leads.get(lead_id)
            if not lead:
                return False
            lead.notes = notes
            self._append_log(lead, "Note updated")
            return True

    def move(self, lead_id: str, column: str) -> bool:
        if column not in VALID_COLUMNS:
            return False
        with self._lock:
            lead = self._leads.get(lead_id)
            if not lead:
                return False
            old_col = lead.column
            lead.column = column
            self._append_log(lead, f"Moved: {old_col} → {column}")
            return True

    def set_persons(self, company_lead_id: str, persons: list) -> Optional[list]:
        """Create individual person cards in contacts_found for each contact found at a company.
        The company card stays in 'imported'.
        Returns list of created person lead dicts, or None if company_lead_id not found."""
        now = _now()
        created = []
        with self._lock:
            company_lead = self._leads.get(company_lead_id)
            if not company_lead:
                return None  # distinct from [] (empty persons list)
            company_lead.contact_enriched_at = now
            company_lead.persons = persons  # keep a reference on the company card
            self._append_log(company_lead, f"Contacts found: {len(persons)}", "Person cards created in Contacts Found")

            csd = company_lead.source_data or {}
            company_name = csd.get("name") or csd.get("company_name") or ""

            for p in persons:
                pid = str(uuid.uuid4())
                # Inject company_name into person source_data so cards show "Title · Company"
                person_data = {**p, "company_name": company_name}
                person_lead = EnrichedLead(
                    id=pid,
                    source_type="person",
                    parent_company_id=company_lead_id,
                    source_data=person_data,
                    column="contacts_found",
                    fit_score=p.get("fit_score", 0),
                    added_at=now,
                    updated_at=now,
                )
                self._append_log(person_lead, f"Found at {company_name}")
                self._leads[pid] = person_lead
                created.append(asdict(person_lead))
        return created

    def set_intelligence(self, lead_id: str, intelligence: dict) -> bool:
        """Store AI intelligence brief on a person lead and move it to deep_researched."""
        with self._lock:
            lead = self._leads.get(lead_id)
            if not lead:
                return False
            lead.intelligence = intelligence
            lead.deep_researched_at = _now()
            old_col = lead.column
            lead.column = "deep_researched"
            name = (lead.source_data or {}).get("name", "contact")
            self._append_log(lead, f"Deep researched: {name}", f"Moved from {old_col}")
            return True

    def delete(self, lead_id: str) -> bool:
        with self._lock:
            if lead_id in self._leads:
                del self._leads[lead_id]
                return True
            return False

    def delete_many(self, lead_ids: list) -> int:
        """Delete multiple leads. Returns count deleted."""
        count = 0
        with self._lock:
            for lead_id in lead_ids:
                if lead_id in self._leads:
                    del self._leads[lead_id]
                    count += 1
        return count
