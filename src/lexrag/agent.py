"""Agentische Retrieval-Schleife ueber dem EU-Rechtskorpus.

Der Unterschied zu klassischem RAG: das Modell entscheidet selbst, wonach es
sucht, darf mehrfach suchen und einen vollstaendigen Artikel nachladen, wenn
ein einzelner Absatz nicht traegt. Beendet wird immer ueber `submit_answer`,
damit jede Antwort ein pruefbares Schema hat.

Guardrail: belegt wird mit Chunk-IDs. Eine ID, die nicht im tatsaechlich
Abgerufenen vorkommt, wird deterministisch als unbelegt markiert -- dafuer
braucht es kein zweites Modell.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from anthropic import Anthropic

from . import config
from .embed import embed_query
from .store import Store
from .trace import tracer

SYSTEM = """Du bist ein Fachassistent für EU-Regulatorik. Dein Korpus umfasst ausschließlich:
- Verordnung (EU) 2024/1689 — KI-Verordnung (EU AI Act)
- Richtlinie (EU) 2022/2555 — NIS-2-Richtlinie

Arbeitsweise:
1. Suche mit `search_corpus`, bevor du etwas behauptest. Formuliere die Suchanfrage mit
   den Fachbegriffen des Normtexts, nicht mit denen der Frage.
2. Reicht ein Absatz nicht, lade den ganzen Artikel mit `get_article` nach.
3. Schließe immer mit `submit_answer` ab.

Harte Regeln:
- Antworte ausschließlich auf Grundlage der abgerufenen Textstellen. Dein Vorwissen zu
  diesen Rechtsakten ist unzulässig als Quelle.
- Jede inhaltliche Aussage braucht mindestens einen Beleg. Belege sind die `chunk_id`-Werte
  aus den Suchergebnissen — nie frei formulierte Fundstellen.
- Trägt der Korpus die Antwort nicht, setze `grounded` auf false und sage, was fehlt.
  Das ist ein korrektes Ergebnis, kein Versagen.
- Erfinde keine Artikel-, Absatz- oder Anhangnummern.
- Antworte auf Deutsch, knapp und präzise. Nenne die Fundstellen im Fließtext in der
  üblichen Zitierweise (z. B. „AI Act Art. 6 Abs. 2")."""

TOOLS = [
    {
        "name": "search_corpus",
        "description": (
            "Hybride Suche (BM25 + Vektoren) über den Normtext. Liefert Textstellen "
            "mit chunk_id und Zitierweise. Mehrfach aufrufbar mit verschiedenen Formulierungen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Suchbegriffe in der Sprache des Normtexts."},
                "doc": {"type": "string", "enum": ["ai_act", "nis2"],
                        "description": "Optional auf einen Rechtsakt einschränken."},
                "k": {"type": "integer", "description": "Anzahl Treffer (Standard 6, max 12)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_article",
        "description": "Lädt einen vollständigen Artikel mit allen Absätzen. Für Kontext um einen Treffer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc": {"type": "string", "enum": ["ai_act", "nis2"]},
                "article": {"type": "string", "description": "Artikelnummer, z. B. \"6\"."},
            },
            "required": ["doc", "article"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Schließt die Bearbeitung ab. Immer als letzter Schritt aufzurufen.",
        # Reihenfolge ist Absicht: erst die kurzen Felder, dann der lange
        # Fliesstext. Stand `answer` vorn, serialisierte das Modell die
        # restlichen Felder gelegentlich als Pseudo-XML *in* den Antwortstring.
        "input_schema": {
            "type": "object",
            "properties": {
                "grounded": {
                    "type": "boolean",
                    "description": "false, wenn der Korpus die Frage nicht beantwortet.",
                },
                "citations": {
                    "type": "array", "items": {"type": "string"},
                    "description": "chunk_id-Werte aus den Suchergebnissen, die die Antwort belegen.",
                },
                "answer": {"type": "string", "description": "Die Antwort auf Deutsch."},
            },
            "required": ["grounded", "citations", "answer"],
        },
    },
]


def _salvage(payload: dict) -> tuple[dict, bool]:
    """Rettet Felder, die das Modell als Pseudo-XML in `answer` geschrieben hat.

    Beobachtetes Fehlbild: der gesamte Tool-Input landet im ersten String-Feld,
    inklusive `</answer><citations>[...]</citations><grounded>true</grounded>`.
    Die Reihenfolge im Schema beugt dem vor; dies ist das Netz darunter. Jeder
    Eingriff wird gezaehlt, damit die Haeufigkeit sichtbar bleibt.
    """
    answer = payload.get("answer") or ""
    if "</answer>" not in answer and "<citations>" not in answer:
        return payload, False

    fixed = dict(payload)
    fixed["answer"] = re.split(r"</answer>|<citations>", answer, maxsplit=1)[0].strip()
    if not payload.get("citations"):
        m = re.search(r"<citations>\s*(\[.*?\])\s*</citations>", answer, re.S)
        if m:
            try:
                fixed["citations"] = json.loads(m.group(1))
            except json.JSONDecodeError:
                fixed["citations"] = re.findall(r"[\w]+:[\w.]+(?::[\w.]+)?", m.group(1))
    if not payload.get("grounded"):
        m = re.search(r"<grounded>\s*(true|false)\s*</grounded>", answer, re.I)
        if m:
            fixed["grounded"] = m.group(1).lower() == "true"
    return fixed, True


@dataclass
class AgentResult:
    answer: str
    citations: list[str]
    grounded: bool
    retrieved: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    steps: int = 0
    truncated: bool = False
    repaired: bool = False
    searches: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    trace_id: str = ""

    @property
    def citations_valid(self) -> bool:
        return not self.invalid_citations

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["citations_valid"] = self.citations_valid
        return d


class Agent:
    def __init__(self, store: Store | None = None, client: Anthropic | None = None,
                 model: str | None = None):
        self.store = store or Store()
        self.client = client or Anthropic()
        self.model = model or config.ANSWER_MODEL

    # ---------- Werkzeuge ----------

    def _search(self, args: dict) -> tuple[str, list[str]]:
        query = (args.get("query") or "").strip()
        k = min(int(args.get("k") or config.TOP_K), 12)
        hits = self.store.search(query, embed_query(query), k=k, doc=args.get("doc"))
        payload = [
            {"chunk_id": h.chunk_id, "fundstelle": h.citation,
             "titel": h.article_title, "text": h.text}
            for h in hits
        ]
        return json.dumps(payload, ensure_ascii=False), [h.chunk_id for h in hits]

    def _article(self, args: dict) -> tuple[str, list[str]]:
        rows = self.store.get_article(args.get("doc", ""), str(args.get("article", "")))
        payload = [
            {"chunk_id": r["chunk_id"], "fundstelle": r["citation"],
             "titel": r["article_title"], "text": r["text"]}
            for r in rows
        ]
        if not payload:
            return json.dumps({"hinweis": "Artikel nicht im Korpus."}, ensure_ascii=False), []
        return json.dumps(payload, ensure_ascii=False), [r["chunk_id"] for r in rows]

    # ---------- Schleife ----------

    def ask(self, question: str, write_trace: bool = True) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": question}]
        retrieved: list[str] = []
        searches: list[str] = []

        with tracer(question, self.model, write=write_trace) as tr:
            for step in range(config.MAX_AGENT_STEPS):
                last = step == config.MAX_AGENT_STEPS - 1
                kwargs = dict(
                    model=self.model, max_tokens=config.MAX_TOKENS, system=SYSTEM,
                    tools=TOOLS, messages=messages,
                )
                # Im letzten erlaubten Schritt den Abschluss erzwingen -> die
                # Schleife endet garantiert mit einer schemakonformen Antwort.
                kwargs["tool_choice"] = (
                    {"type": "tool", "name": "submit_answer"} if last else {"type": "auto"}
                )
                resp = self.client.messages.create(**kwargs)
                tr.add_usage(resp.usage)

                # Ein am Token-Limit abgeschnittener Tool-Aufruf liefert ein
                # unvollstaendiges JSON. Ohne diese Pruefung wird daraus
                # klammheimlich eine leere Antwort mit grounded=false --
                # ununterscheidbar von einer echten Verweigerung.
                truncated = resp.stop_reason == "max_tokens"
                if truncated:
                    tr.add_step("truncated", stop_reason=resp.stop_reason,
                                out_tokens=resp.usage.output_tokens)

                tool_uses = [b for b in resp.content if b.type == "tool_use"]
                if not tool_uses:
                    text = " ".join(b.text for b in resp.content if b.type == "text")
                    tr.add_step("text_only", chars=len(text))
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append({"role": "user", "content":
                                     "Bitte schließe mit submit_answer ab."})
                    continue

                messages.append({"role": "assistant", "content": resp.content})
                results = []
                finished: AgentResult | None = None

                for tu in tool_uses:
                    if tu.name == "submit_answer":
                        payload, repaired = _salvage(tu.input)
                        if repaired:
                            tr.add_step("repaired_schema_leak")
                        cites = list(payload.get("citations") or [])
                        invalid = [c for c in cites if c not in set(retrieved)]
                        answer_text = payload.get("answer", "")
                        if truncated and not answer_text:
                            # Abgeschnitten, bevor ein Feld stand: das ist ein
                            # Fehler, keine Verweigerung.
                            finished = AgentResult(
                                answer="", citations=[], grounded=False,
                                retrieved=retrieved, invalid_citations=[],
                                steps=step + 1, searches=searches,
                                trace_id=tr.trace_id, truncated=True,
                            )
                            tr.add_step("submit_answer_truncated")
                            break
                        finished = AgentResult(
                            answer=answer_text,
                            citations=cites,
                            grounded=bool(payload.get("grounded")),
                            retrieved=retrieved, invalid_citations=invalid,
                            steps=step + 1, searches=searches, trace_id=tr.trace_id,
                            truncated=truncated, repaired=repaired,
                        )
                        tr.add_step("submit_answer", citations=cites,
                                    grounded=finished.grounded, invalid=invalid)
                        break

                    if tu.name == "search_corpus":
                        body, ids = self._search(tu.input)
                        searches.append(tu.input.get("query", ""))
                        tr.add_step("search", query=tu.input.get("query"), hits=len(ids))
                    elif tu.name == "get_article":
                        body, ids = self._article(tu.input)
                        tr.add_step("get_article", doc=tu.input.get("doc"),
                                    article=tu.input.get("article"), paragraphs=len(ids))
                    else:
                        body, ids = json.dumps({"fehler": "unbekanntes Werkzeug"}), []

                    for i in ids:
                        if i not in retrieved:
                            retrieved.append(i)
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": body})

                if finished:
                    tr.retrieved = retrieved
                    tr.grounded = finished.grounded
                    tr.finalize()
                    finished.tokens_in, finished.tokens_out = tr.tokens_in, tr.tokens_out
                    finished.cost_usd, finished.latency_ms = tr.cost_usd, tr.latency_ms
                    return finished

                messages.append({"role": "user", "content": results})

            tr.retrieved = retrieved
            return AgentResult(answer="", citations=[], grounded=False, retrieved=retrieved,
                               steps=config.MAX_AGENT_STEPS, searches=searches,
                               trace_id=tr.trace_id)
