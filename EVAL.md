# Evaluation

Wie gemessen wird, was dabei herauskam, und welche Fehler die Messung
aufgedeckt hat.

## Aufbau

**Golden Set** (`evals/golden.yaml`): 24 Fälle, davon 18 beantwortbar und
**6 bewusst unbeantwortbar** (DSGVO, ISO 27001, Cyber Resilience Act, DORA,
nationale NIS-Umsetzung, Off-Topic). Die Negativfälle messen, ob das System
schweigt statt zu raten — eine Eigenschaft, die in kaum einem RAG-Projekt
erhoben wird und in der Regulatorik die wichtigste ist.

**Zwei Ebenen, bewusst getrennt:**

| Ebene | Was sie misst | Kosten | Wofür |
|---|---|---|---|
| `--retrieval-only` | Recall@k, MRR, Latenz | **0 USD** | Retrieval tunen, Modelle vergleichen |
| voll | zusätzlich Antwortquote, Zitate, Faithfulness, Verweigerung, Kosten | ~1,50 USD | Gesamturteil |

Die Trennung ist der Grund, warum die Chunking-Reparaturen unten überhaupt
bezahlbar waren: Retrieval lässt sich beliebig oft messen, ohne ein Modell zu
bezahlen.

**Wiederaufnahme:** Jeder fertige Fall wird einzeln zwischengespeichert, der
Cache-Schlüssel trägt einen Korpus-Fingerabdruck. Ein Abbruch kostet damit nur
den laufenden Fall; ein geänderter Korpus entwertet alte Läufe automatisch.

## Ergebnisse

Siehe [README](README.md#kennzahlen) — die Zahlen dort werden von
`evals/update_readme.py` aus dem jüngsten Lauf eingetragen, nicht von Hand
gepflegt.

## Was die Messung gefunden hat

Vier Fehler, von denen **drei ohne Messung unsichtbar geblieben wären**. Die
Kette ist hier vollständig dokumentiert, weil sie mehr über das System aussagt
als die Endzahlen.

### 1 · Anhänge waren praktisch unauffindbar

*Symptom:* Recall@6 = 83 %. Vier Fälle scheiterten, alle mit demselben Muster —
Artikel trafen, Anhänge nie.

*Ursachen:* Der Anhangstitel war das Label („ANHANG III") statt der Überschrift
(„Hochrisiko-KI-Systeme gemäß Artikel 6 Absatz 2"); Anhänge wurden als
1800-Zeichen-Blöcke eingebettet, obwohl sie Aufzählungen sind — die Einbettung
mittelte über Biometrie, Bildung, Strafverfolgung und Migration zu einem
Schwerpunkt, der zu nichts mehr passte; Füllwörter verrauschten BM25.

*Behebung:* Titel korrigiert, Splitting an nummerierten Listenpunkten,
Stoppwortfilter. **Recall 83 % → 97 %.**

### 2 · 165 Absätze wurden nie gelesen

*Symptom:* Der LLM-Judge meldete unbelegte Aussagen: „Bankwesen", „Trinkwasser",
„Abwasser" als NIS2-Sektoren.

*Ursache:* EUR-Lex setzt Fließtext als `oj-normal`, Tabellenzellen aber als
`oj-tbl-txt`. NIS2 führt die Spalte „Art der Einrichtung" durchgängig als
`oj-tbl-txt` — der Parser las sie nie. Vier Sektoren fehlten vollständig,
inklusive ihrer Bezeichnung, die als 12-Zeichen-Fragment („3. Bankwesen") der
Mindestlänge zum Opfer fiel.

*Warum das gefährlich war:* Das Modell kannte die Sektoren aus eigenem Wissen
und nannte sie — mit **gültigen** Zitaten auf benachbarte Stellen. Die Antwort
sah vollständig und belegt aus. Nur der Faithfulness-Judge sah, dass die
Aussagen nicht aus den zitierten Stellen folgten.

*Behebung:* `oj-tbl-txt` wird mitgelesen, Kurzüberschriften werden an ihren
Abschnitt gehängt statt verworfen. NIS2-Anhänge 11 → 20 Chunks.

### 3 · Zwölf von achtzehn Antworten waren leer — und liefen als „ok" durch

*Symptom:* `grounded=false` ohne Zitate bei 12 von 18 beantwortbaren Fragen.

*Ursache:* `max_tokens=2000`. Eine Antwort über einen vollständigen
Rechtsartikel überschreitet das; der Tool-Aufruf brach mitten im JSON ab und
lieferte ein leeres Input-Objekt. Der Code las daraus `answer=""` und
`grounded=false` — **ununterscheidbar von einer legitimen Verweigerung**.

*Behebung:* Budget auf 8000, und `stop_reason == "max_tokens"` wird ausgewertet:
ein abgeschnittener Abschluss ist jetzt ein Fehler, keine Verweigerung.

### 4 · Der eigentliche Fehler lag in der Evaluation

Fehler 3 blieb über mehrere vollständige Läufe unentdeckt, weil die Eval keine
**Antwortquote** erhob. Gemessen wurden Faithfulness und Zitat-Gültigkeit —
beides Kriterien, die eine leere Antwort mühelos besteht. Zwölf Fehlschläge
wurden als „ok" ausgewiesen.

> Eine Metrik, die nicht erhoben wird, beschreibt einen Fehler, der nicht
> gesehen wird.

*Behebung:* `antwortquote` und `abgeschnitten` als eigene Metriken; das
Erfolgs-Flag je Fall prüft jetzt auch, ob überhaupt geantwortet wurde.

### Nachspiel · Das Modell serialisierte alle Felder in ein einziges

Nach der Budgeterhöhung schrieb das Modell gelegentlich den gesamten
Tool-Input als Pseudo-XML in das Feld `answer`
(`…</answer><citations>[…]</citations><grounded>true</grounded>`). Die Antwort
stand da, aber Zitate und `grounded` blieben leer.

*Behebung:* Reihenfolge im Schema umgedreht — `grounded` und `citations` vor
dem langen Fließtext. Zusätzlich ein Rettungsparser, der solche Fälle repariert
**und zählt**, damit die Häufigkeit sichtbar bleibt statt stillschweigend
kaschiert zu werden.

## Eine Metrik wurde nach ihrem Ergebnis geändert — offengelegt

Die ursprüngliche Metrik „Zitat-Treffsicherheit" verlangte, dass **alle**
Zitate innerhalb der erwarteten Artikel liegen. Sie ergab 60 %. Die Prüfung der
Fehlschläge zeigte: In beiden Fällen war der erwartete Artikel zitiert, und
zusätzlich eine sachlich richtige Nachbarnorm — bei NIS2-Governance etwa
Art. 21, auf den Art. 20 unmittelbar verweist.

Die Metrik maß damit Exklusivität, nicht Richtigkeit. Bei Rechtstexten sind
Querverweise der Normalfall; zusätzliche korrekte Fundstellen sind ein Vorzug.
Sie wurde ersetzt durch **Zitat-Deckung** (ist jede erwartete Fundstelle
belegt?) plus **Zusatzzitate** als getrennte, nicht bestrafte Kennzahl.

Das ist eine Änderung nach Sichtung des Ergebnisses und wird deshalb hier
ausdrücklich genannt.

## Wahl des Embedding-Modells — gemessen, nicht angenommen

Naheliegende Annahme: ein größeres, eigens für Deutsch trainiertes Modell
liefert auf einem deutschen Rechtskorpus bessere Treffer. Gemessen trifft das
nicht zu.

Reproduzieren mit `python evals/compare_embeddings.py` (kostet nichts —
deterministische Ebene, keine Modellaufrufe):

| Modell | Dim | Recall@6 | MRR | vollständig | Latenz | Indexbau |
|---|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | **0,972** | 0,835 | **0,944** | **40 ms** | **140 s** |
| `jina-embeddings-v2-base-de` | 768 | 0,889 | **0,843** | 0,833 | 76 ms | 772 s |

Das kleinere, generische Modell findet häufiger das Richtige, bei 5,5-fach
kürzerem Indexbau und halber Abfragelatenz. Jina liegt allein beim MRR knapp
vorn: Wenn es trifft, platziert es minimal besser — es trifft nur seltener.

**Woran es liegt.** Der Unterschied besteht aus genau zwei Fällen. Der
deutlichere: auf *„Welche Pflicht zur KI-Kompetenz trifft Anbieter und
Betreiber?"* liefert Jina im Vektorkanal Art. 23, 24, 25, 26 und 50 — lauter
Normen über *Pflichten von Anbietern, Betreibern, Händlern und Einführern*.
Es gewichtet die generische Rollen- und Pflichtensprache und verliert dabei den
kennzeichnenden Begriff „KI-Kompetenz"; Art. 4 fällt aus den ersten sechs
heraus. MiniLM führt ihn auf Rang 3.

Da beide Läufe denselben BM25-Kanal und denselben Index benutzen, ist der
Vektorkanal die einzige Variable — er hat den Fall entschieden.

**Wie belastbar das ist.** Retrieval ist deterministisch: derselbe Index und
dieselbe Anfrage ergeben stets dasselbe Ergebnis, es gibt also keine Streuung
zwischen Wiederholungen. Der Unterschied ist damit reproduzierbar und kein
Messrauschen. Er beruht aber auf **zwei von achtzehn Fällen**. Die belastbare
Aussage lautet deshalb: *MiniLM ist hier mindestens so gut wie Jina, bei einem
Bruchteil der Kosten* — nicht: *MiniLM ist neun Punkte besser*. Für eine
Aussage in dieser Schärfe wäre ein deutlich größerer Golden Set nötig.

**Entscheidung:** MiniLM bleibt Vorgabe. Das Modell ist über
`LEXRAG_EMBED_MODEL` austauschbar, der Vergleich jederzeit wiederholbar.

## Bekannte Grenzen

**Der Agent zitiert repräsentativ, nicht erschöpfend.** Bei
`aia_anhang3_bereiche` beantwortete er alle acht Bereiche des Anhangs III
korrekt, führte aber nur 5 der 10 einschlägigen Chunks als Beleg. Der Korpus
ist nachweislich vollständig; die Zitatliste ist es nicht. Das drückt die
Faithfulness und ist die Hauptursache für die verbleibenden 11 %.

**Leichte interpretatorische Zusätze.** In einem Fall bezeichnete die Antwort
die Aufzählung in Art. 5 als „abschließend" — sachlich vertretbar, aber so
nicht im zitierten Text.

**Der Judge ist ein Modell.** Faithfulness ist die einzige Metrik ohne
deterministische Grundlage und entsprechend mit Rauschen behaftet. Die
deterministischen Kennzahlen — Antwortquote, Zitat-Gültigkeit, Zitat-Deckung,
Verweigerung — sind reproduzierbar.

**Ein Golden Set von 24 Fällen ist klein.** Die Werte zeigen Größenordnungen,
keine Nachkommastellen. `--repeat` schätzt die Varianz.
