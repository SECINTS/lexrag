"""Zentrale Konfiguration. Alles ueber Umgebungsvariablen ueberschreibbar."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# --- Retrieval ---
EMBED_MODEL = os.getenv("LEXRAG_EMBED_MODEL", "jinaai/jina-embeddings-v2-base-de")
EMBED_DIM = int(os.getenv("LEXRAG_EMBED_DIM", "768"))
DB_PATH = Path(os.getenv("LEXRAG_DB", ROOT / "corpus" / "lexrag.db"))
RRF_K = int(os.getenv("LEXRAG_RRF_K", "60"))
TOP_K = int(os.getenv("LEXRAG_TOP_K", "6"))
CANDIDATES = int(os.getenv("LEXRAG_CANDIDATES", "25"))

# --- Agent ---
ANSWER_MODEL = os.getenv("LEXRAG_ANSWER_MODEL", "claude-sonnet-5")
JUDGE_MODEL = os.getenv("LEXRAG_JUDGE_MODEL", "claude-sonnet-5")
MAX_AGENT_STEPS = int(os.getenv("LEXRAG_MAX_STEPS", "4"))
# Eine Antwort ueber einen ganzen Rechtsartikel braucht Platz. Bei 2000
# brach `submit_answer` mitten im JSON ab -- das Ergebnis war ein leeres
# Antwortfeld, das wie eine legitime Verweigerung aussah.
MAX_TOKENS = int(os.getenv("LEXRAG_MAX_TOKENS", "8000"))

# --- Observability ---
TRACE_DIR = Path(os.getenv("LEXRAG_TRACE_DIR", ROOT / "traces"))

# Preise USD je 1 Mio Token (Stand 2026-09; zentral gepflegt, damit die
# Kostenrechnung im Trace nicht an drei Stellen auseinanderlaeuft).
PRICES = {
    "claude-sonnet-5":  {"in": 3.00, "out": 15.00},
    "claude-opus-4-8":  {"in": 5.00, "out": 25.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
}


def price_for(model: str) -> dict[str, float]:
    for key, val in PRICES.items():
        if model.startswith(key):
            return val
    return {"in": 0.0, "out": 0.0}
