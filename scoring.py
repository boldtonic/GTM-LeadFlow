"""
Lead scoring for Fogata.
"""

from models import Lead


class LeadScorer:
    def __init__(self, config: dict):
        self.positive_signals = config.get("positive_signals", [])
        self.negative_signals = config.get("negative_signals", [])
        self.exclude_brands = [b.lower() for b in config.get("exclude_brands", [])]
        self.min_rating = config.get("min_rating", 0)
        self.target_segment = config.get("segment", "").lower()

    def score(self, lead: Lead) -> tuple[int, list[str]]:
        score = 50
        reasons = []

        name_lower = lead.name.lower()
        for excluded in self.exclude_brands:
            if excluded in name_lower:
                return 0, [f"Excluded: {excluded}"]

        if lead.rating >= 4.5:
            score += 15
            reasons.append(f"Great rating ({lead.rating})")
        elif lead.rating >= 4.0:
            score += 10
            reasons.append(f"Good rating ({lead.rating})")
        elif 0 < lead.rating < self.min_rating:
            score -= 20

        if lead.reviews_count >= 100:
            score += 10
            reasons.append(f"Established ({lead.reviews_count} reviews)")
        elif lead.reviews_count >= 30:
            score += 5

        if lead.website:
            score += 10
            reasons.append("Has website")
        else:
            score -= 15

        if lead.phone:
            score += 5

        if lead.emails_found:
            score += 10
            reasons.append("Email found")

        if lead.decision_makers:
            score += 10
            reasons.append("Decision maker found")

        searchable = f"{lead.name} {lead.category} {' '.join(lead.subcategories)}".lower()

        for signal in self.positive_signals:
            if signal.lower() in searchable:
                score += 5
                reasons.append(f"+{signal}")

        for signal in self.negative_signals:
            if signal.lower() in searchable:
                score -= 10
                reasons.append(f"-{signal}")

        if lead.social_links.get("instagram"):
            score += 5
            reasons.append("Has Instagram")

        return max(0, min(100, score)), reasons
