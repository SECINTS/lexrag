# Architektur

## Überblick

```
EUR-Lex (amtlich)          ┌──────────────── Clients ────────────────┐
        │                  │                                          │
        ▼                  │   HTTP /search /ask      MCP-Werkzeuge   │
   ingest.py               │   (AI Act Radar,         (Claude Desktop,│
   Artikel/Absatz-         │    eigene Apps)           Claude Code)   │
   genaues Chunking        └──────┬───────────────────────┬───────────┘
        │                         │                       │
        ▼                     api.py                 mcp_server.py
   embed.py ──► store.py ◄───────┴───────────────────────┘
   (lokal,      SQLite:
    ONNX)       FTS5 (BM25) + Vektoren
                     │
                     ▼
                 agent.py ──► Claude (Tool-Use-Schleife)
                     │
                     ▼
                 trace.py (JSONL: Schritte, Token, Kosten, Latenz)
                     │
                     ▼
              evals/run_eval.py (Kennzahlen)
```

## Entwurfsentscheidungen

**Korpus = operativer Normtext.** Aufgenommen werden Artikel und Anhänge,
nicht die Erwägungsgründe. Erwägungsgründe entfalten keine unmittelbare
Rechtswirkung und erzeugen semantische Beinahe-Duplikate, die das Retrieval
messbar verschlechtern. Wer sie braucht, schaltet sie in `ingest.py` frei.

**Chunk = ein Absatz.** EUR-Lex liefert ELI-Markup mit der Hierarchie
`art_57 → 057.001`. Damit fällt die Chunk-Grenze mit der juristischen
Zitiereinheit zusammen — „AI Act Art. 57 Abs. 1" ist prüfbar, „irgendwo in
Kapitel VI" nicht. Kurze Absätze werden bis ~200 Zeichen zusammengefasst,
lange an Satzgrenzen bis 1800 Zeichen geteilt, nie über Artikelgrenzen hinweg.

**Kontextualisierte Einbettung.** Eingebettet wird nicht der nackte Absatz,
sondern sein Pfad davor: *„KI-Verordnung > KAPITEL VI > Artikel 57
(KI-Reallabore) Absatz 1: …"*. Der Vektor kennt damit seine Verortung;
angezeigt wird weiterhin der rohe Text.

**Hybride Suche statt reiner Vektorsuche.** Rechtsfragen enthalten exakte
Termini und Normzitate, bei denen semantische Nähe versagt. BM25 (FTS5) und
Vektoren laufen parallel und werden per *Reciprocal Rank Fusion* vereinigt:

    score(d) = Σ_r  1 / (k + rang_r(d)),  k = 60

RRF wertet nur Ränge, nicht Scores — die beiden völlig unterschiedlich
skalierten Maße müssen deshalb nicht normalisiert werden. Das macht das
Verfahren robust gegen Ausreißer in beiden Kanälen.

*Gemessener Beleg:* Auf die Frage „Wann müssen KI-erzeugte Inhalte
gekennzeichnet werden?" liegt **AI Act Art. 50 Abs. 4** (die Deepfake-
Kennzeichnungspflicht) auf **BM25-Rang 1, aber Vektor-Rang 24**. Eine reine
Vektorsuche hätte genau die einschlägige Norm verfehlt. Umgekehrt findet der
Vektorkanal Absatz 2 auf Rang 1, den BM25 nur auf Rang 2 führt. Beide Kanäle
tragen nachweislich bei.

**Deutsche Komposita.** FTS5 tokenisiert nicht morphologisch. Tokens ab fünf
Zeichen werden deshalb als Präfixsuche gestellt, damit „Risikomanagement" auch
„Risikomanagementsystem" findet.

**Kein Vektordienst.** 724 Chunks × 768 Dimensionen × 4 Byte = 2,2 MB. Die
Matrix liegt im RAM, Brute-Force-Kosinus ist bei dieser Größe exakt und
schneller als jeder Index. Der Wechsel auf pgvector oder sqlite-vec wird ab
grob 100 000 Chunks nötig und berührt nur `store.py`.

**Agentische Schleife statt Einmal-Retrieval.** Das Modell wählt seine
Suchanfragen selbst, darf nachsuchen und mit `get_article` den vollständigen
Artikel nachladen, wenn ein Absatz nicht trägt. Im letzten erlaubten Schritt
wird `submit_answer` per `tool_choice` erzwungen — die Schleife endet damit
garantiert schemakonform statt im Leerlauf.

**Belege sind IDs, nicht Prosa.** `submit_answer` verlangt `chunk_id`-Werte.
Eine ID, die nicht im tatsächlich Abgerufenen vorkommt, wird deterministisch
als ungültig markiert. Erfundene Fundstellen fallen damit ohne zweites Modell
auf — der einzige Halluzinationstyp, der sich mechanisch ausschließen lässt.

**Verweigerung ist ein Ergebnis.** Trägt der Korpus die Frage nicht, ist
`grounded=false` das korrekte Resultat. Der Golden Set enthält sechs Fälle,
die genau das prüfen (DSGVO, ISO 27001, CRA, DORA, nationale Umsetzung,
Off-Topic).

## Schichten und Austauschbarkeit

| Schicht | Modul | Wechsel berührt |
|---|---|---|
| Einbettung | `embed.py` | nur dieses Modul (lokal ↔ gehostet) |
| Speicher/Suche | `store.py` | nur dieses Modul (SQLite ↔ pgvector) |
| Antwortmodell | `config.ANSWER_MODEL` | Umgebungsvariable |
| Korpus | `ingest.DOCS` | eine Liste |

## Observability

Jede Anfrage schreibt eine JSONL-Zeile nach `traces/`: Schrittfolge, gestellte
Suchanfragen, abgerufene Chunk-IDs, Token ein/aus, Kosten in USD, Latenz.
Die Preistabelle steht zentral in `config.PRICES`, damit die Kostenrechnung
nicht an drei Stellen auseinanderläuft.
