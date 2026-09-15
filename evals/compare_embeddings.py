"""Vergleicht Embedding-Modelle auf der deterministischen Eval-Ebene.

Kostet nichts: gemessen wird nur Retrieval, kein Modellaufruf. Damit laesst
sich die Modellwahl belegen statt begruenden.

    python evals/compare_embeddings.py

Jede Konfiguration laeuft in einem eigenen Prozess -- die Konfiguration wird
beim Import gelesen, ein Wechsel im selben Prozess waere unsauber.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"

KONFIGURATIONEN = [
    {
        "name": "MiniLM-multilingual",
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "db": ROOT / "corpus" / "lexrag-minilm.db",
    },
    {
        "name": "Jina-v2-base-de",
        "model": "jinaai/jina-embeddings-v2-base-de",
        "dim": 768,
        "db": ROOT / "corpus" / "lexrag-jina.db",
    },
]


def env_for(cfg: dict) -> dict:
    e = dict(os.environ)
    e.update({
        "LEXRAG_EMBED_MODEL": cfg["model"],
        "LEXRAG_EMBED_DIM": str(cfg["dim"]),
        "LEXRAG_DB": str(cfg["db"]),
        "PYTHONPATH": str(ROOT / "src"),
    })
    return e


def build(cfg: dict) -> float:
    t0 = time.time()
    subprocess.run(
        [str(PY), "-c", "from lexrag.store import build_index; build_index()"],
        env=env_for(cfg), check=True, capture_output=True,
    )
    return round(time.time() - t0, 1)


def measure(cfg: dict) -> dict:
    subprocess.run(
        [str(PY), str(ROOT / "evals" / "run_eval.py"), "--retrieval-only"],
        env=env_for(cfg), check=True, capture_output=True,
    )
    neu = sorted((ROOT / "evals" / "results").glob("*-retrieval.json"))[-1]
    return json.loads(neu.read_text(encoding="utf-8"))["zusammenfassung"]


def main() -> None:
    nur_messen = "--no-build" in sys.argv
    zeilen = []
    for cfg in KONFIGURATIONEN:
        print(f"› {cfg['name']} …", flush=True)
        bauzeit = None
        if not nur_messen or not cfg["db"].exists():
            bauzeit = build(cfg)
        s = measure(cfg)
        zeilen.append({**cfg, "db": str(cfg["db"]), "bauzeit_s": bauzeit, **s})

    kopf = ["Modell", "Dim", "Recall@6", "MRR", "vollst.", "Latenz", "Indexbau"]
    print("\n| " + " | ".join(kopf) + " |")
    print("|" + "|".join("---" for _ in kopf) + "|")
    for z in zeilen:
        print(f"| {z['name']} | {z['dim']} | {z['recall_at_k']:.3f} | {z['mrr']:.3f} "
              f"| {z['vollstaendig_getroffen']:.3f} | {z['latenz_ms_mittel']:.0f} ms "
              f"| {(str(z['bauzeit_s']) + ' s') if z['bauzeit_s'] else '—'} |")

    out = ROOT / "evals" / "results" / "embedding-vergleich.json"
    out.write_text(json.dumps(zeilen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ngeschrieben: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
