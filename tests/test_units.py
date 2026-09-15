"""Unit-Tests fuer die Teile, die stillschweigend brechen koennen."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexrag.ingest import Chunk, _pack, _split_long  # noqa: E402
from lexrag.store import Store, _fts_query           # noqa: E402


# ---------- FTS5-Eingabe ----------

@pytest.mark.parametrize("raw", [
    'Was gilt bei "Hochrisiko"?',
    "Artikel 6 Absatz 3 -- Derogation",
    "NIS2: Meldefrist 24h?",
    "A* OR B AND NOT C",
    "   ",
    "()",
])
def test_fts_query_ist_immer_syntaktisch_gueltig(tmp_path, raw):
    """FTS5 deutet \" * : - als Operatoren. Ungepruefte Eingabe wirft OperationalError."""
    store = Store(tmp_path / "t.db")
    q = _fts_query(raw)
    if not q:
        return
    store.conn.execute("INSERT INTO chunks_fts VALUES ('x','Hochrisiko Derogation Meldefrist')")
    store.conn.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ?", (q,)).fetchall()


def test_fts_query_praefix_fuer_komposita():
    """Deutsche Komposita: 'Risikomanagement' muss 'Risikomanagementsystem' finden."""
    assert '"Risikomanagement"*' in _fts_query("Risikomanagement")
    assert '"und"*' not in _fts_query("und")  # kurze Tokens ohne Praefix


def test_fts_query_leer_bei_nur_sonderzeichen():
    assert _fts_query("!!! ??? ***") == ""


# ---------- Chunking ----------

def test_split_long_haelt_obergrenze_auch_ohne_satzgrenzen():
    text = "Wort " * 1000  # keine Satzzeichen -> Notfall-Hartschnitt muss greifen
    for piece in _split_long(text, limit=300):
        assert len(piece) <= 300


def test_split_long_laesst_kurzes_unangetastet():
    assert _split_long("Kurzer Satz.", limit=300) == ["Kurzer Satz."]


def test_pack_fasst_kurze_absaetze_zusammen():
    units = [("1", "Kurz."), ("2", "Auch kurz."), ("3", "x" * 400)]
    packed = _pack(units)
    assert len(packed) < len(units)


def test_contextual_text_enthaelt_pfad():
    c = Chunk("ai_act:art_57:1", "ai_act", "KI-Verordnung", "article", "57",
              "KI-Reallabore", "1", "KAPITEL VI", "AI Act Art. 57 Abs. 1", "Inhalt.")
    ctx = c.contextual_text()
    assert "KI-Reallabore" in ctx and "KAPITEL VI" in ctx and "Absatz 1" in ctx
    assert ctx.endswith("Inhalt.")


# ---------- Fusion ----------

def test_rrf_bevorzugt_treffer_aus_beiden_kanaelen(tmp_path):
    """Kernaussage von RRF: wer in beiden Kanaelen auftaucht, schlaegt den,
    der nur in einem auftaucht.

    Symmetrische Raenge (Rang 1 hier, Rang 2 dort und umgekehrt) ergeben
    dagegen bewusst ein Unentschieden -- das ist korrektes RRF-Verhalten und
    wird hier nicht als Rangfolge geprueft.
    """
    store = Store(tmp_path / "t.db")
    chunks = [
        # findet nur BM25 (Stichwort da, Vektor orthogonal zur Anfrage)
        Chunk("nur_bm25", "ai_act", "T", "article", "1", "", "", "", "AI Act Art. 1",
              "Risikomanagement als Begriff."),
        # findet beides (Stichwort mehrfach + Vektor deckungsgleich)
        Chunk("beides", "ai_act", "T", "article", "2", "", "", "", "AI Act Art. 2",
              "Risikomanagement, Risikomanagement, Risikomanagement."),
        # findet nur der Vektor (kein Stichwort, Vektor nahe der Anfrage)
        Chunk("nur_vec", "ai_act", "T", "article", "3", "", "", "", "AI Act Art. 3",
              "Voellig anderer Inhalt ohne das gesuchte Wort."),
    ]
    vecs = np.zeros((3, 768), dtype=np.float32)
    vecs[0, 1] = 1.0          # nur_bm25: orthogonal
    vecs[1, 0] = 1.0          # beides:   deckungsgleich
    vecs[2, 0] = 0.9
    vecs[2, 1] = 0.44         # nur_vec:  nah dran, aber nicht identisch
    store.rebuild(chunks, vecs)

    qvec = np.zeros(768, dtype=np.float32)
    qvec[0] = 1.0
    hits = store.search("Risikomanagement", qvec, k=3)

    assert hits[0].chunk_id == "beides", [h.chunk_id for h in hits]
    assert hits[0].bm25_rank is not None and hits[0].vec_rank is not None
    nur_vec = next(h for h in hits if h.chunk_id == "nur_vec")
    assert nur_vec.bm25_rank is None, "ohne Stichwort darf BM25 nicht treffen"
    assert hits[0].score > nur_vec.score


def test_suche_ohne_vektor_faellt_auf_bm25_zurueck(tmp_path):
    store = Store(tmp_path / "t.db")
    chunks = [Chunk("d:0", "ai_act", "T", "article", "1", "", "", "", "AI Act Art. 1",
                    "Verbotene Praktiken im KI-Bereich.")]
    store.rebuild(chunks, np.zeros((1, 768), dtype=np.float32))
    hits = store.search("verbotene Praktiken", None, k=3)
    assert hits and hits[0].chunk_id == "d:0"
    assert hits[0].vec_rank is None


# ---------- Store ----------

def test_get_article_sortiert_absaetze_numerisch(tmp_path):
    store = Store(tmp_path / "t.db")
    chunks = [
        Chunk("a:3", "ai_act", "T", "article", "6", "Titel", "3", "", "AI Act Art. 6 Abs. 3", "drei"),
        Chunk("a:1", "ai_act", "T", "article", "6", "Titel", "1", "", "AI Act Art. 6 Abs. 1", "eins"),
        Chunk("a:10", "ai_act", "T", "article", "6", "Titel", "10", "", "AI Act Art. 6 Abs. 10", "zehn"),
    ]
    store.rebuild(chunks, np.zeros((3, 768), dtype=np.float32))
    paras = [r["paragraph"] for r in store.get_article("ai_act", "6")]
    assert paras == ["1", "3", "10"], "Absatz 10 darf nicht zwischen 1 und 3 landen"


# ---------- Nebenlaeufigkeit ----------

def test_store_ist_aus_fremdem_thread_nutzbar(tmp_path):
    """Regression: MCP-Server und FastAPI rufen synchrone Funktionen im
    Threadpool auf. Eine an den Hauptthread gebundene SQLite-Verbindung wirft
    dort ProgrammingError -- der Fehler trat real im MCP-Server auf."""
    import threading

    store = Store(tmp_path / "t.db")
    chunks = [Chunk("d:0", "ai_act", "T", "article", "1", "Titel", "1", "",
                    "AI Act Art. 1", "Verbotene Praktiken im KI-Bereich.")]
    store.rebuild(chunks, np.zeros((1, 768), dtype=np.float32))

    ergebnis: dict = {}

    def aus_anderem_thread():
        try:
            ergebnis["hits"] = store.search("verbotene Praktiken", None, k=3)
            ergebnis["article"] = store.get_article("ai_act", "1")
            ergebnis["stats"] = store.stats()
        except Exception as exc:  # noqa: BLE001
            ergebnis["fehler"] = exc

    t = threading.Thread(target=aus_anderem_thread)
    t.start()
    t.join(timeout=10)

    assert "fehler" not in ergebnis, f"Fremdthread scheiterte: {ergebnis.get('fehler')}"
    assert ergebnis["hits"] and ergebnis["hits"][0].chunk_id == "d:0"
    assert len(ergebnis["article"]) == 1
