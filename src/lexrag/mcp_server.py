"""MCP-Server: macht den Korpus fuer jeden MCP-Client nutzbar.

Einbinden in Claude Desktop / Claude Code (claude_desktop_config.json):

    {
      "mcpServers": {
        "lexrag": {
          "command": "/ABSOLUTER/PFAD/lexrag/.venv/bin/python",
          "args": ["-m", "lexrag.mcp_server"],
          "env": {"PYTHONPATH": "/ABSOLUTER/PFAD/lexrag/src"}
        }
      }
    }

Bewusst nur die Retrieval-Werkzeuge: der Client bringt sein eigenes Modell mit,
eine zweite Agentenschleife waere doppelt gemoppelt und doppelt bezahlt.
"""
from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from .embed import embed_query
from .store import Store

mcp = MCPServer(
    "lexrag",
    instructions=(
        "Belegter Zugriff auf den EU AI Act (VO (EU) 2024/1689) und die "
        "NIS-2-Richtlinie (RL (EU) 2022/2555). Jede Fundstelle ist zitierfaehig. "
        "Antworte nur mit dem, was die Werkzeuge liefern."
    ),
)
_store = Store()


@mcp.tool()
def search_eu_law(query: str, doc: str | None = None, k: int = 6) -> str:
    """Durchsucht EU AI Act und NIS-2-Richtlinie (hybrid: BM25 + Vektoren).

    Args:
        query: Suchbegriffe, moeglichst in der Sprache des Normtexts.
        doc: Optional 'ai_act' oder 'nis2' zum Einschraenken.
        k: Anzahl Treffer (1-20).
    """
    hits = _store.search(query, embed_query(query), k=max(1, min(k, 20)), doc=doc)
    return json.dumps(
        [{"fundstelle": h.citation, "titel": h.article_title,
          "text": h.text, "chunk_id": h.chunk_id} for h in hits],
        ensure_ascii=False, indent=2,
    )


@mcp.tool()
def get_eu_article(doc: str, article: str) -> str:
    """Laedt einen vollstaendigen Artikel mit allen Absaetzen.

    Args:
        doc: 'ai_act' (VO (EU) 2024/1689) oder 'nis2' (RL (EU) 2022/2555).
        article: Artikelnummer, z. B. '6'.
    """
    rows = _store.get_article(doc, article)
    if not rows:
        return json.dumps({"hinweis": f"Artikel {article} nicht im Korpus."}, ensure_ascii=False)
    return json.dumps(
        [{"fundstelle": r["citation"], "titel": r["article_title"], "text": r["text"]}
         for r in rows], ensure_ascii=False, indent=2,
    )


@mcp.tool()
def corpus_info() -> str:
    """Nennt Umfang und Stand des Korpus."""
    return json.dumps(_store.stats(), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
