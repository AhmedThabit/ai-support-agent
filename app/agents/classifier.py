from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    signals: Dict[str, int]


CATEGORIES = ("Billing", "Account", "Bug", "General")


_KEYWORDS = {
    "Billing": [
        "invoice", "refund", "charged", "charge", "billing", "payment", "card",
        "subscription", "renew", "plan", "price"
    ],
    "Account": [
        "password", "login", "sign in", "account", "email", "verification",
        "verify", "2fa", "two factor", "reset"
    ],
    "Bug": [
        "error", "bug", "crash", "broken", "not working", "blank", "issue",
        "fails", "failure", "slow", "latency"
    ],
}


def classify_ticket(text: str) -> ClassificationResult:
    """
    MVP baseline classifier (keyword scoring).
    Later we'll replace/augment this with an LLM classifier.
    """
    t = (text or "").lower().strip()
    if not t:
        return ClassificationResult(category="General", confidence=0.2, signals={})

    scores: Dict[str, int] = {c: 0 for c in CATEGORIES}
    signals: Dict[str, int] = {}

    for cat, kws in _KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[cat] += 1
                signals[f"{cat}:{kw}"] = signals.get(f"{cat}:{kw}", 0) + 1

    best_cat, best_score = _argmax(scores)
    total = sum(scores.values())

    if total == 0:
        return ClassificationResult(category="General", confidence=0.35, signals={})

    confidence = min(0.95, 0.5 + (best_score / max(1, total)) * 0.45)
    return ClassificationResult(category=best_cat, confidence=confidence, signals=signals)


def _argmax(scores: Dict[str, int]) -> Tuple[str, int]:
    best_cat = "General"
    best_score = -1
    for cat, score in scores.items():
        if score > best_score:
            best_cat, best_score = cat, score
    return best_cat, best_score
