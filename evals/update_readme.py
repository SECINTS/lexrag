"""Traegt die Kennzahlen des juengsten Eval-Laufs in die README ein.

Damit stehen dort keine handkopierten Zahlen, die still veralten.

    python evals/update_readme.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"

LABELS = {
    "antwortquote": ("Antwortquote", "beantwortbare Fragen, die auch beantwortet wurden"),
    "recall_at_k": ("Retrieval-Recall@k", "Anteil der erwarteten Fundstellen, die das Retrieval liefert"),
    "zitate_gueltig": ("Zitat-Gültigkeit", "Anteil Antworten ohne erfundene Fundstelle"),
    "zitat_deckung": ("Zitat-Deckung", "jede erwartete Fundstelle ist auch belegt"),
    "zusatzzitate_mittel": ("Zusatzzitate", "zusätzlich belegte Nachbarnormen je Antwort (informativ)"),
    "faithfulness": ("Faithfulness", "Aussagen sind vom zitierten Text gedeckt (LLM-Judge)"),
    "verweigerung_korrekt": ("Verweigerungsquote", "korrektes Schweigen bei Fragen außerhalb des Korpus"),
    "kosten_usd_mittel": ("Kosten je Anfrage", "USD, Mittel über alle Fälle"),
    "latenz_ms_mittel": ("Latenz (Mittel)", "ms"),
    "latenz_ms_p95": ("Latenz (p95)", "ms"),
    "schritte_mittel": ("Agenten-Schritte", "Mittel je Anfrage"),
}


def fmt(key: str, val) -> str:
    if val is None:
        return "—"
    if key.startswith("kosten"):
        return f"{val:.4f} USD"
    if key.startswith("latenz"):
        return f"{int(val)} ms"
    if key in ("schritte_mittel", "zusatzzitate_mittel"):
        return f"{val:.1f}"
    return f"{val:.0%}" if isinstance(val, float) and val <= 1 else str(val)


def main() -> None:
    runs = sorted(RESULTS.glob("*.json"))
    full = [p for p in runs if "retrieval" not in p.name]
    if not full:
        raise SystemExit("Kein vollstaendiger Eval-Lauf in evals/results/.")
    data = json.loads(full[-1].read_text(encoding="utf-8"))
    s = data["zusammenfassung"]

    lines = [
        f"Stand {data['zeitpunkt'][:8]} · Modell `{data['modell']}` · Embedding "
        f"`{data['embedding']}` · k={data['k']} · {s['faelle']} Fälle "
        f"({s['beantwortbar']} beantwortbar, {s['negativ']} bewusst unbeantwortbar)",
        "",
        "| Metrik | Wert | Bedeutung |",
        "|---|---|---|",
    ]
    for key, (label, meaning) in LABELS.items():
        if key in s:
            lines.append(f"| {label} | **{fmt(key, s[key])}** | {meaning} |")
    lines += ["", f"Gesamtkosten des Laufs: {s['kosten_usd_gesamt']:.3f} USD."]

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- EVAL:START -->", "<!-- EVAL:END -->"
    a, b = text.index(start) + len(start), text.index(end)
    readme.write_text(text[:a] + "\n" + "\n".join(lines) + "\n" + text[b:], encoding="utf-8")
    print(f"README aktualisiert aus {full[-1].name}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
