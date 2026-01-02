import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

CHUNKS_PATH = Path("data/kb/chunks.json")

def load_chunks() -> List[Dict[str, Any]]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

def score_keyword(query: str, text: str) -> int:
    q = (query or "").lower()
    t = (text or "").lower()
    tokens = [w for w in re.findall(r"[a-zA-Z0-9]+", q) if len(w) > 2]
    return sum(1 for w in tokens if w in t)

def top_k_keyword(query: str, k: int = 3) -> List[Tuple[int, Dict[str, Any]]]:
    chunks = load_chunks()
    scored = [(score_keyword(query, c["content"]), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, c) for s, c in scored[:k] if s > 0]
