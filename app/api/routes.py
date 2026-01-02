from fastapi import FastAPI, Query
from pydantic import BaseModel

from app.agents.classifier import classify_ticket
from app.rag.retrieve_keyword import top_k_keyword
from app.rag.retrieve_tfidf import top_k_tfidf

app = FastAPI(title="AI Support Agent", version="0.1.0")


class TicketIn(BaseModel):
    text: str


class ClassificationOut(BaseModel):
    category: str
    confidence: float


class SourceOut(BaseModel):
    category: str
    question: str
    score: float
    mode: str


class AnswerOut(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationOut)
def classify(payload: TicketIn):
    r = classify_ticket(payload.text)
    return {"category": r.category, "confidence": r.confidence}


def _clean_md_answer(s: str) -> str:
    return (
        (s or "")
        .replace("### Q:", "")
        .replace("**A:**", "")
        .replace("**Steps:**", "\nSteps:")
        .replace("**Notes:**", "\nNotes:")
        .replace("**", "")
        .strip()
    )


@app.post("/answer", response_model=AnswerOut)
def answer(payload: TicketIn, mode: str = Query(default="keyword", pattern="^(keyword|tfidf)$")):
    text = payload.text

    if mode == "tfidf":
        hits = top_k_tfidf(text, k=3)  # [(score(float), chunk), ...]
        best = hits[0] if hits else None
        score = float(best[0]) if best else 0.0
        chunk = best[1] if best else None
    else:
        hits = top_k_keyword(text, k=3)  # [(score(int), chunk), ...]
        best = hits[0] if hits else None
        score = float(best[0]) if best else 0.0
        chunk = best[1] if best else None

    if not chunk:
        return {"answer": "No relevant KB article found. Please share more details.", "sources": []}

    clean_answer = _clean_md_answer(chunk["content"])

    return {
        "answer": clean_answer,
        "sources": [{
            "category": chunk["category"],
            "question": chunk["question"],
            "score": score,
            "mode": mode
        }]
    }
