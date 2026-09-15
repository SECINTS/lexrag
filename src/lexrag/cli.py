"""Kommandozeile fuer LexRAG.

    python -m lexrag.cli fetch     # Amtstexte von EUR-Lex holen
    python -m lexrag.cli index     # Chunking + Einbettung + Index
    python -m lexrag.cli search "Meldefristen NIS2"
    python -m lexrag.cli ask "Wann muss KI-Inhalt gekennzeichnet werden?"
    python -m lexrag.cli stats
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

from . import config

SOURCES = {
    "ai_act_de.html": "https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=OJ:L_202401689",
    "nis2_de.html": "https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32022L2555",
}


def cmd_fetch(_args) -> None:
    raw = config.ROOT / "corpus" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        r = httpx.get(url, follow_redirects=True, timeout=60)
        r.raise_for_status()
        (raw / name).write_bytes(r.content)
        print(f"  {name}: {len(r.content):,} Bytes")


def cmd_index(_args) -> None:
    from .store import build_index
    print(json.dumps(build_index(), indent=2, ensure_ascii=False))


def cmd_search(args) -> None:
    from .embed import embed_query
    from .store import Store
    store = Store()
    for h in store.search(args.query, embed_query(args.query), k=args.k, doc=args.doc):
        ranks = f"bm25={h.bm25_rank or '-'} vec={h.vec_rank or '-'}"
        print(f"\n[{h.citation}] {h.article_title}  ({ranks}, rrf={h.score:.4f})")
        print(f"  {h.text[:260]}{'...' if len(h.text) > 260 else ''}")


def cmd_ask(args) -> None:
    from .agent import Agent
    res = Agent().ask(args.question)
    print(f"\n{res.answer}\n")
    print(f"belegt: {res.grounded} | Fundstellen: {', '.join(res.citations) or '-'}")
    if res.invalid_citations:
        print(f"UNGUELTIGE BELEGE: {res.invalid_citations}")
    print(f"Schritte: {res.steps} | Suchen: {res.searches}")
    print(f"{res.tokens_in}+{res.tokens_out} Token | {res.cost_usd:.5f} USD | {res.latency_ms} ms")


def cmd_stats(_args) -> None:
    from .store import Store
    print(json.dumps(Store().stats(), indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(prog="lexrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    sub.add_parser("index").set_defaults(fn=cmd_index)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)

    p = sub.add_parser("search"); p.set_defaults(fn=cmd_search)
    p.add_argument("query"); p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--doc", choices=["ai_act", "nis2"])

    p = sub.add_parser("ask"); p.set_defaults(fn=cmd_ask)
    p.add_argument("question")

    args = ap.parse_args()
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
