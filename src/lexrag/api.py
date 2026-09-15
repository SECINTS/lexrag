"""HTTP-Schnittstelle.

Bewusst ohne eigenes Frontend: LexRAG ist ein Dienst, kein Produkt. Andere
Anwendungen -- etwa AI Act Radar -- klinken sich hier ein und liefern zu ihrer
eigenen Ausgabe die belegte Rechtsgrundlage nach.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .agent import Agent
from .embed import embed_query
from .store import Store

app = FastAPI(
    title="LexRAG",
    version="0.1.0",
    description="Belegter Zugriff auf EU AI Act und NIS-2-Richtlinie.",
)

_store = Store()
_agent: Agent | None = None


def agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(store=_store)
    return _agent


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=config.TOP_K, ge=1, le=20)
    doc: str | None = Field(default=None, pattern="^(ai_act|nis2)$")


class AskRequest(BaseModel):
    question: str
    model: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "korpus": _store.stats()}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    """Reines Retrieval -- kein Modellaufruf, keine Kosten."""
    hits = _store.search(req.query, embed_query(req.query), k=req.k, doc=req.doc)
    return {"query": req.query, "treffer": [h.to_dict() for h in hits]}


@app.get("/article/{doc}/{article}")
def article(doc: str, article: str) -> dict:
    if doc not in ("ai_act", "nis2"):
        raise HTTPException(404, "Unbekannter Rechtsakt.")
    rows = _store.get_article(doc, article)
    if not rows:
        raise HTTPException(404, f"Artikel {article} nicht im Korpus.")
    return {"doc": doc, "article": article, "absaetze": rows}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Agentische Antwort mit geprueften Belegen."""
    a = agent() if not req.model else Agent(store=_store, model=req.model)
    res = a.ask(req.question)
    return res.to_dict()
