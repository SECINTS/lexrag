"""Eval-Harness fuer LexRAG.

Zwei Ebenen, bewusst getrennt:

1. Deterministisch, ohne jeden API-Call -- Retrieval-Recall, Zitat-Gueltigkeit,
   Zitat-Genauigkeit, Verweigerungsquote. Schnell, kostenlos, reproduzierbar.
   Damit laesst sich Retrieval tunen, ohne das Modell zu bezahlen.
2. LLM-Judge -- nur fuer Faithfulness: ist jede Aussage vom zitierten Text
   gedeckt? Das laesst sich mechanisch nicht pruefen.

Aufruf:
    python evals/run_eval.py --retrieval-only   # Ebene 1, keine Kosten
    python evals/run_eval.py                    # Ebene 1 + 2
    python evals/run_eval.py --k 10 --repeat 3  # Parametersuche / Varianz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexrag import config              # noqa: E402
from lexrag.embed import embed_query   # noqa: E402
from lexrag.store import Store         # noqa: E402

JUDGE_SYSTEM = """Du prüfst, ob eine Antwort vollständig durch die mitgelieferten Belegstellen
gedeckt ist. Du bewertest ausschließlich die Deckung, nicht Stil oder Vollständigkeit.

Eine Aussage gilt als gedeckt, wenn sie sich aus den Belegstellen ableiten lässt.
Allgemeine Einleitungssätze ohne Tatsachengehalt ignorierst du.

Antworte ausschließlich über das Werkzeug `urteil`."""

JUDGE_TOOL = {
    "name": "urteil",
    "description": "Gibt das Deckungsurteil ab.",
    "input_schema": {
        "type": "object",
        "properties": {
            "gedeckt": {"type": "boolean", "description": "Sind alle Tatsachenaussagen gedeckt?"},
            "ungedeckte_aussagen": {"type": "array", "items": {"type": "string"}},
            "begruendung": {"type": "string"},
        },
        "required": ["gedeckt", "ungedeckte_aussagen", "begruendung"],
    },
}


def hit_expected(retrieved: list[str], expected: list[str]) -> tuple[int, int]:
    """Wie viele der erwarteten Fundstellen wurden abgerufen?"""
    if not expected:
        return 0, 0
    found = sum(1 for e in expected if any(r.startswith(e) for r in retrieved))
    return found, len(expected)


def eval_retrieval(cases: list[dict], store: Store, k: int) -> list[dict]:
    rows = []
    for c in cases:
        if not c["answerable"]:
            continue
        t0 = time.time()
        hits = store.search(c["question"], embed_query(c["question"]), k=k)
        ms = int((time.time() - t0) * 1000)
        ids = [h.chunk_id for h in hits]
        found, total = hit_expected(ids, c["expect_articles"])
        # Reciprocal Rank: Position des ersten korrekten Treffers
        rr = 0.0
        for i, cid in enumerate(ids, start=1):
            if any(cid.startswith(e) for e in c["expect_articles"]):
                rr = 1.0 / i
                break
        rows.append({
            "id": c["id"], "recall": found / total if total else 0.0,
            "found": found, "expected": total, "rr": rr,
            "latency_ms": ms, "top": ids[:3],
        })
    return rows


def judge(client, answer: str, cited_texts: list[str], model: str) -> dict:
    if not answer.strip():
        return {"gedeckt": False, "ungedeckte_aussagen": ["leere Antwort"], "begruendung": "keine Antwort"}
    belege = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(cited_texts)) or "(keine)"
    resp = client.messages.create(
        model=model, max_tokens=800, system=JUDGE_SYSTEM,
        tools=[JUDGE_TOOL], tool_choice={"type": "tool", "name": "urteil"},
        messages=[{"role": "user", "content": f"ANTWORT:\n{answer}\n\nBELEGSTELLEN:\n{belege}"}],
    )
    for b in resp.content:
        if b.type == "tool_use":
            return b.input
    return {"gedeckt": False, "ungedeckte_aussagen": [], "begruendung": "kein Urteil"}


def _corpus_fingerprint(store: Store) -> str:
    """Aendert sich der Korpus, sind zwischengespeicherte Laeufe wertlos."""
    st = store.stats()
    raw = f"{st.get('n_chunks')}|{st.get('embed_model')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def _cache_path(case_id: str, k: int, fp: str) -> Path:
    """Ein Fall, eine Datei. Ein Abbruch kostet damit nur den laufenden Fall,
    nicht den ganzen Lauf -- Agentenlaeufe sind bezahlte Arbeit."""
    d = ROOT / "evals" / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{case_id}__{config.ANSWER_MODEL}__k{k}__{fp}.json"


def eval_full(cases: list[dict], store: Store, k: int, use_cache: bool = True) -> list[dict]:
    from anthropic import Anthropic
    from lexrag.agent import Agent

    client = Anthropic()
    agent = Agent(store=store, client=client)
    fp = _corpus_fingerprint(store)
    rows = []
    for i, c in enumerate(cases, start=1):
        print(f"  [{i:>2}/{len(cases)}] {c['id']:<28}", end="", flush=True)
        cache = _cache_path(c["id"], k, fp)
        if use_cache and cache.exists():
            row = json.loads(cache.read_text(encoding="utf-8"))
            rows.append(row)
            print(f" (Cache)  {row['latency_ms']:>5} ms  {row['cost_usd']:.4f} USD")
            continue
        res = agent.ask(c["question"], write_trace=True)
        found, total = hit_expected(res.retrieved, c["expect_articles"])
        # Deckung statt Exklusivitaet: gefordert ist, dass jede erwartete
        # Fundstelle belegt wird. Zusaetzlich zitierte Nachbarnormen sind bei
        # Rechtstexten der Normalfall (Querverweise) und kein Fehler -- sie
        # werden separat gezaehlt, nicht bestraft.
        cited_covers = (
            all(any(cid.startswith(e) for cid in res.citations) for e in c["expect_articles"])
            if (c["answerable"] and c["expect_articles"]) else None
        )
        extra = (
            len([cid for cid in res.citations
                 if not any(cid.startswith(e) for e in c["expect_articles"])])
            if c["answerable"] else 0
        )
        row = {
            "id": c["id"], "answerable": c["answerable"],
            "grounded": res.grounded,
            "beantwortet": bool(res.grounded and res.answer.strip()),
            "truncated": res.truncated,
            "refusal_correct": (res.grounded is False) if not c["answerable"] else None,
            "recall": found / total if total else None,
            "citations": res.citations,
            "citations_valid": res.citations_valid,
            "invalid_citations": res.invalid_citations,
            "cited_covers_expected": cited_covers,
            "extra_citations": extra,
            "steps": res.steps, "searches": res.searches,
            "cost_usd": res.cost_usd, "latency_ms": res.latency_ms,
            "tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
            "answer": res.answer,
        }
        # Faithfulness nur dort pruefen, wo ueberhaupt etwas behauptet wurde
        if c["answerable"] and res.grounded and res.citations:
            texts = [r["text"] for r in (store.get(cid) for cid in res.citations) if r]
            v = judge(client, res.answer, texts, config.JUDGE_MODEL)
            row["faithful"] = bool(v.get("gedeckt"))
            row["unsupported"] = v.get("ungedeckte_aussagen", [])
        else:
            row["faithful"] = None
        rows.append(row)
        cache.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = (row["citations_valid"] and row.get("faithful") is not False
              and (row["beantwortet"] or not c["answerable"]))
        flag = "ok" if ok else "PRUEFEN"
        print(f" {flag:<8} {row['latency_ms']:>5} ms  {row['cost_usd']:.4f} USD")
    return rows


def summarize(rows: list[dict], retrieval_only: bool) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 3) if xs else None

    def p95(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[min(len(xs) - 1, int(len(xs) * 0.95))] if xs else None

    if retrieval_only:
        return {
            "faelle": len(rows),
            "recall_at_k": mean([r["recall"] for r in rows]),
            "mrr": mean([r["rr"] for r in rows]),
            "vollstaendig_getroffen": round(
                sum(1 for r in rows if r["recall"] == 1.0) / len(rows), 3) if rows else None,
            "latenz_ms_mittel": mean([r["latency_ms"] for r in rows]),
        }

    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    return {
        "faelle": len(rows),
        "beantwortbar": len(ans),
        "negativ": len(neg),
        "antwortquote": mean([1.0 if r["beantwortet"] else 0.0 for r in ans]),
        "abgeschnitten": sum(1 for r in rows if r.get("truncated")),
        "recall_at_k": mean([r["recall"] for r in ans]),
        "zitate_gueltig": round(sum(1 for r in rows if r["citations_valid"]) / len(rows), 3),
        "zitat_deckung": mean(
            [1.0 if r["cited_covers_expected"] else 0.0 for r in ans
             if r["cited_covers_expected"] is not None]),
        "zusatzzitate_mittel": mean([r["extra_citations"] for r in ans]),
        "faithfulness": mean([1.0 if r["faithful"] else 0.0 for r in ans
                              if r["faithful"] is not None]),
        "verweigerung_korrekt": mean([1.0 if r["refusal_correct"] else 0.0 for r in neg]),
        "schritte_mittel": mean([r["steps"] for r in rows]),
        "kosten_usd_mittel": round(mean([r["cost_usd"] for r in rows]) or 0, 5),
        "kosten_usd_gesamt": round(sum(r["cost_usd"] for r in rows), 4),
        "latenz_ms_mittel": mean([r["latency_ms"] for r in rows]),
        "latenz_ms_p95": p95([r["latency_ms"] for r in rows]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="nur Ebene 1, keine API-Kosten")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--repeat", type=int, default=1, help="Wiederholungen fuer Varianzschaetzung")
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "results")
    ap.add_argument("--no-cache", action="store_true", help="fertige Faelle neu rechnen")
    args = ap.parse_args()

    cases = yaml.safe_load((ROOT / "evals" / "golden.yaml").read_text(encoding="utf-8"))["cases"]
    store = Store()
    stats = store.stats()
    if not stats.get("n_chunks"):
        raise SystemExit("Index leer -- zuerst `python -m lexrag.store` ausfuehren.")

    print(f"Korpus: {stats['n_chunks']} Chunks {stats['by_doc']}")
    print(f"Modell: {config.ANSWER_MODEL} | Embedding: {stats['embed_model']} | k={args.k}\n")

    runs = []
    for r in range(args.repeat):
        if args.repeat > 1:
            print(f"--- Durchlauf {r+1}/{args.repeat} ---")
        rows = (eval_retrieval(cases, store, args.k) if args.retrieval_only
                else eval_full(cases, store, args.k, use_cache=not args.no_cache))
        runs.append(summarize(rows, args.retrieval_only))

    summary = runs[0] if args.repeat == 1 else {
        k: (round(statistics.mean([r[k] for r in runs if r[k] is not None]), 4)
            if isinstance(runs[0][k], (int, float)) else runs[0][k])
        for k in runs[0]
    }

    print("\n" + "=" * 58)
    print(f"{'ERGEBNIS':<40}")
    print("=" * 58)
    for key, val in summary.items():
        print(f"  {key:<34} {val}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "zeitpunkt": stamp, "modell": config.ANSWER_MODEL,
        "embedding": stats["embed_model"], "k": args.k, "wiederholungen": args.repeat,
        "korpus": stats, "zusammenfassung": summary,
        "einzelergebnisse": rows,
    }
    path = args.out / f"{stamp}{'-retrieval' if args.retrieval_only else ''}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ngeschrieben: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
