"""
Thread-safe job management for GTM LeadFlow.
Replaces the bare job_status dict with locked access and TTL cleanup.
"""

import threading
import time


class JobManager:
    def __init__(self, ttl_seconds=3600):
        self._jobs = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds

    def create(self, job_id: str, initial_state: dict):
        """Create a new job entry. Cleans up expired jobs first."""
        with self._lock:
            self._cleanup_expired()
            self._jobs[job_id] = {**initial_state, "_created_at": time.time()}

    def get(self, job_id: str) -> dict:
        """Return a snapshot (copy) of job state, or None."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # Return copy without internal fields
            return {k: v for k, v in job.items() if not k.startswith("_")}

    def update(self, job_id: str, **kwargs):
        """Update fields on an existing job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)

    def exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.get(job_id, {}).get("cancelled", False)

    def set_cancelled(self, job_id: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["cancelled"] = True

    def get_leads(self, job_id: str) -> list:
        """Return a copy of leads list for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            return list(job.get("leads", [])) if job else []

    def find_recent_completed(self) -> dict:
        """Find the most recent completed job. Returns copy or None."""
        with self._lock:
            for job_id in sorted(self._jobs.keys(), reverse=True):
                job = self._jobs[job_id]
                if job.get("status") == "completed":
                    return {k: v for k, v in job.items() if not k.startswith("_")}
            return None

    def _cleanup_expired(self):
        """Remove completed/errored/cancelled jobs older than TTL."""
        now = time.time()
        expired = [
            jid
            for jid, j in self._jobs.items()
            if j.get("status") in ("completed", "error", "cancelled") and now - j.get("_created_at", 0) > self.ttl
        ]
        for jid in expired:
            del self._jobs[jid]
