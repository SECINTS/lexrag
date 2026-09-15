# Sicherheitsbetrachtung

Bedrohungsmodell, getroffene Maßnahmen und bewusst offen gelassene Punkte.
Geprüft vor der Veröffentlichung.

## Annahmen

LexRAG ist als **lokaler Dienst und Bibliothek** entworfen, nicht als
öffentlich exponierte Anwendung. Alle Aussagen unten gelten für diesen
Betriebsfall; Abweichungen sind markiert.

## Geheimnisse

| Punkt | Zustand |
|---|---|
| `ANTHROPIC_API_KEY` | ausschließlich aus `.env` oder Umgebung; `.env` ist in `.gitignore` |
| Schlüssel in Logs oder Traces | nein — `trace.py` schreibt Frage, Schritte, Token, Kosten, nie Header oder Schlüssel |
| Schlüssel im Repository | nein; `.env.example` enthält nur den Platzhalter `sk-ant-...` |
| Embeddings | laufen lokal über ONNX — für die Indexierung verlässt kein Text den Rechner |

Geprüft wurde der gesamte zu committende Baum auf die Muster
`sk-ant-`, `sk-…`, `ghp_`, `github_pat_`, `AKIA…`, `xox…`, `AIza…` und
`BEGIN … PRIVATE KEY`. Einziger Treffer: der Platzhalter in `.env.example`.

## Eingaben

**SQL.** Jede Abfrage mit Nutzereingabe ist parametrisiert (`?`). Es gibt keine
String-Interpolation in SQL.

**FTS5.** Freitext wird nicht direkt an `MATCH` gereicht. `_fts_query()`
zerlegt die Eingabe in Wort-Token und quotet jedes einzeln — sonst führen
Zeichen wie `"`, `*`, `:` oder `-` zu Syntaxfehlern oder ungewollten
Operatoren. Abgedeckt durch parametrisierte Tests in `tests/test_units.py`.

**HTTP-API.** `doc` ist gegen eine Allowlist geprüft (`^(ai_act|nis2)$`),
`k` auf 1–20 begrenzt. `article` fließt ausschließlich als gebundener
Parameter in die Abfrage; Pfad-Traversal ist dadurch gegenstandslos.

## Modell- und Korpusrisiken

**Prompt Injection.** Der Korpus besteht aus amtlichem EU-Recht, das über
HTTPS von EUR-Lex bezogen wird — die Angriffsfläche ist gering, aber nicht
null: wer den Korpus austauscht, steuert die Antworten. Zwei Maßnahmen
begrenzen den Schaden:

- Der Abschluss erfolgt über ein erzwungenes Schema (`submit_answer`), nicht
  über freien Text. Das Modell kann das Ausgabeformat nicht verlassen.
- Belege sind `chunk_id`-Werte und werden **deterministisch** gegen das
  tatsächlich Abgerufene geprüft. Eine erfundene Fundstelle wird als solche
  markiert, unabhängig davon, was im Text steht.

**Korpusintegrität.** Die Texte werden zur Laufzeit geladen, ohne
Hash-Pinning. Für den Einsatz mit Rechtsfolgen sollte die Fassung
eingefroren und der Hash mitgeführt werden (offen, siehe unten).

**Halluzination.** Nicht vollständig ausschließbar. Messbar gemacht statt
behauptet: Zitat-Gültigkeit deterministisch, Faithfulness per LLM-Judge,
Verweigerungsquote über sechs bewusst unbeantwortbare Fälle. Werte in der
README.

## Bekannte Schwächen (bewusst offen)

| # | Punkt | Wirkung | Wann relevant |
|---|---|---|---|
| 1 | **Keine Authentifizierung an der HTTP-API** | Wer `/ask` erreicht, verbraucht fremdes API-Guthaben | Nur bei öffentlicher Exposition. Vor einem Deployment zwingend nachrüsten. |
| 2 | **Keine Ratenbegrenzung** | Kosten-DoS durch Anfrageflut | wie 1 |
| 3 | **Kein Hash-Pinning des Korpus** | stille Änderung der Rechtsgrundlage | bei rechtsverbindlicher Nutzung |
| 4 | **Traces enthalten die Fragen im Klartext** | Rückschlüsse auf Nutzer möglich | bei Mehrbenutzerbetrieb → Aufbewahrung regeln |
| 5 | **Keine Mandantentrennung** | ein Korpus, ein Traceverzeichnis | bei Mehrmandantenbetrieb |

Punkte 1 und 2 sind der Grund, warum diese Anwendung **ohne vorgelagerte
Authentifizierung nicht ins offene Netz gehört**. Für den lokalen Betrieb und
als MCP-Server (stdio, kein Netzwerkport) sind sie gegenstandslos.

## Datenschutz

Verarbeitet werden ausschließlich die Fragen der Nutzenden. Diese gehen an die
Anthropic-API; Embeddings bleiben lokal. Es werden keine personenbezogenen
Daten aus dem Korpus verarbeitet — amtliche Rechtstexte enthalten keine.
Für einen produktiven Einsatz mit Dritten wäre ein Auftragsverarbeitungs-
vertrag mit dem Modellanbieter erforderlich.

## Meldung von Schwachstellen

Sicherheitsrelevante Funde bitte an **office@secints.com**, nicht als
öffentliches Issue.
