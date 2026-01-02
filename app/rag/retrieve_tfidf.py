import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

CHUNKS_PATH = Path("data/kb/chunks.json")

def load_chunks() -> List[Dict[str, Any]]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

def tokenize(s: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", s or "") if len(w) > 2]

def build_idf(docs_tokens: List[List[str]]) -> Dict[str, float]:
    N = len(docs_tokens)
    df: Dict[str, int] = {}
    for toks in docs_tokens:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    return {w: math.log((N + 1) / (df[w] + 1)) + 1.0 for w in df}

def tfidf_vec(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    return {w: tf[w] * idf.get(w, 0.0) for w in tf}

def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(w, 0.0) * b.get(w, 0.0) for w in a.keys())
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

def top_k_tfidf(query: str, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    chunks = load_chunks()
    docs_tokens = [tokenize(c["content"]) for c in chunks]
    idf = build_idf(docs_tokens)

    qv = tfidf_vec(tokenize(query), idf)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for c, toks in zip(chunks, docs_tokens):
        dv = tfidf_vec(toks, idf)
        scored.append((cosine(qv, dv), c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, c) for s, c in scored[:k] if s > 0.0]
