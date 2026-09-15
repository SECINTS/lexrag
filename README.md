# LexRAG — belegter Retrieval-Agent für EU-Regulatorik

Ein agentisches RAG-System über den **EU AI Act** (VO (EU) 2024/1689) und die
**NIS-2-Richtlinie** (RL (EU) 2022/2555). Beantwortet Fachfragen ausschließlich
aus dem Normtext, belegt jede Aussage artikel- und absatzgenau — und schweigt
nachweislich, wenn der Korpus die Frage nicht trägt.

Gebaut als Referenzimplementierung für die vier Fähigkeiten, die 2026 in
KI-Ausschreibungen am häufigsten verlangt werden: **Evaluation**, **Agenten**,
**RAG** und **MCP**.

> **Hinweis:** Kein Rechtsrat. Das System gibt den amtlichen Normtext wieder
> und ersetzt keine juristische Prüfung.

---

## Kennzahlen

<!-- EVAL:START -->
Stand 20260915 · Modell `claude-sonnet-5` · Embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` · k=6 · 24 Fälle (18 beantwortbar, 6 bewusst unbeantwortbar)

| Metrik | Wert | Bedeutung |
|---|---|---|
| Antwortquote | **100%** | beantwortbare Fragen, die auch beantwortet wurden |
| Retrieval-Recall@k | **100%** | Anteil der erwarteten Fundstellen, die das Retrieval liefert |
| Zitat-Gültigkeit | **100%** | Anteil Antworten ohne erfundene Fundstelle |
| Zitat-Deckung | **100%** | jede erwartete Fundstelle ist auch belegt |
| Zusatzzitate | **1.1** | zusätzlich belegte Nachbarnormen je Antwort (informativ) |
| Faithfulness | **89%** | Aussagen sind vom zitierten Text gedeckt (LLM-Judge) |
| Verweigerungsquote | **100%** | korrektes Schweigen bei Fragen außerhalb des Korpus |
| Kosten je Anfrage | **0.0630 USD** | USD, Mittel über alle Fälle |
| Latenz (Mittel) | **17757 ms** | ms |
| Latenz (p95) | **28484 ms** | ms |
| Agenten-Schritte | **2.5** | Mittel je Anfrage |

Gesamtkosten des Laufs: 1.512 USD.
<!-- EVAL:END -->

Reproduzieren:

```bash
python evals/run_eval.py --retrieval-only   # deterministisch, ohne API-Kosten
python evals/run_eval.py                    # inkl. Agent und LLM-Judge
```

---

## Was das System kann

| Fähigkeit | Umsetzung |
|---|---|
| **Hybride Suche** | BM25 (SQLite FTS5) + Vektoren, vereint per Reciprocal Rank Fusion |
| **Agentische Schleife** | Modell wählt Suchanfragen selbst, lädt bei Bedarf ganze Artikel nach |
| **Geprüfte Belege** | Zitate sind Chunk-IDs; erfundene Fundstellen werden deterministisch erkannt |
| **Verweigerung** | `grounded=false`, wenn der Korpus nicht trägt — als Metrik gemessen |
| **Evaluation** | 24-Fall-Golden-Set, sechs Metriken, zwei Ebenen (mit/ohne API) |
| **Observability** | JSONL-Trace je Anfrage: Schritte, Token, Kosten, Latenz |
| **MCP-Server** | Korpus als Werkzeug für Claude Desktop / Claude Code |
| **HTTP-API** | `/search`, `/ask`, `/article/{doc}/{nr}` für Fremdsysteme |

---

## Schnellstart

```bash
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env          # ANTHROPIC_API_KEY eintragen
./.venv/bin/python -m lexrag.cli fetch    # Amtstexte von EUR-Lex
./.venv/bin/python -m lexrag.cli index    # Chunking + Einbettung
```

Der erste Indexlauf lädt das Embedding-Modell (~330 MB) und rechnet auf CPU.
Danach liegt alles in `corpus/lexrag.db`; weitere Läufe brauchen das nicht.

```bash
./.venv/bin/python -m lexrag.cli search "Meldefristen erhebliche Sicherheitsvorfälle"
./.venv/bin/python -m lexrag.cli ask "Wann müssen KI-erzeugte Inhalte gekennzeichnet werden?"
./.venv/bin/python -m lexrag.cli stats
```

HTTP-Dienst:

```bash
./.venv/bin/uvicorn lexrag.api:app --app-dir src --port 8730
```

---

## Als MCP-Server einbinden

In `claude_desktop_config.json` (Pfade absolut):

```json
{
  "mcpServers": {
    "lexrag": {
      "command": "/PFAD/lexrag/.venv/bin/python",
      "args": ["-m", "lexrag.mcp_server"],
      "env": { "PYTHONPATH": "/PFAD/lexrag/src" }
    }
  }
}
```

Danach stehen `search_eu_law`, `get_eu_article` und `corpus_info` als Werkzeuge
bereit — das Modell des Clients antwortet, der Korpus liefert die Belege.

---

## Korpus

| Rechtsakt | Artikel | Chunks |
|---|---|---|
| VO (EU) 2024/1689 — KI-Verordnung | 113 | 524 |
| RL (EU) 2022/2555 — NIS-2-Richtlinie | 46 | 200 |

Quelle: EUR-Lex, deutsche Sprachfassung. Aufgenommen sind Artikel und Anhänge,
**nicht** die Erwägungsgründe — Begründung in [ARCHITECTURE.md](ARCHITECTURE.md).

Amtliche EU-Rechtstexte sind frei weiterverwendbar; das Repository enthält
keine geschützten Normwerke (ISO-Normen o. Ä.) und lädt die Texte zur Laufzeit.

---

## Aufbau

```
src/lexrag/
  ingest.py      EUR-Lex-HTML -> zitierfähige Chunks (Artikel/Absatz)
  embed.py       lokale Embeddings (ONNX, kein zusätzlicher API-Key)
  store.py       SQLite: FTS5 + Vektoren, Hybrid-Suche mit RRF
  agent.py       Tool-Use-Schleife, erzwungenes Antwortschema, Beleg-Prüfung
  trace.py       JSONL-Trace mit Token-, Kosten- und Latenzrechnung
  api.py         FastAPI
  mcp_server.py  MCP
  cli.py         Kommandozeile
evals/
  golden.yaml    24 Fälle (18 beantwortbar, 6 bewusst unbeantwortbar)
  run_eval.py    Harness, zwei Ebenen
tests/           15 Unit-Tests
```

Details und Begründungen: [ARCHITECTURE.md](ARCHITECTURE.md).
Bedrohungsmodell und bekannte Schwächen: [SECURITY.md](SECURITY.md).
Messaufbau, gefundene Fehler und Grenzen: [EVAL.md](EVAL.md).

---

## Lizenz und Haftung

Code: siehe LICENSE. Die Rechtstexte stammen von EUR-Lex und unterliegen der
Wiederverwendungspolitik der EU. Das System ersetzt keine Rechtsberatung.
