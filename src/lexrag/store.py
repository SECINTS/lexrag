"""Hybrider Retrieval-Speicher auf reinem SQLite.

BM25 (FTS5) fuer exakte Rechtsbegriffe und Normzitate, Vektoren fuer Bedeutung,
zusammengefuehrt per Reciprocal Rank Fusion. Kein Docker, kein Dienst.

Skalierung: die Vektoren liegen als eine numpy-Matrix im RAM (724 Chunks x 768
Dim x 4 B = 2,2 MB). Brute-force-Kosinus traegt bis grob 100k Chunks; darueber
ist der Wechsel auf pgvector oder sqlite-vec faellig und beruehrt nur dieses
Modul.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config
from .ingest import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc           TEXT NOT NULL,
    doc_title     TEXT NOT NULL,
    unit          TEXT NOT NULL,
    article       TEXT,
    article_title TEXT,
    paragraph     TEXT,
    chapter       TEXT,
    citation      TEXT NOT NULL,
    text          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    haystack,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS vectors (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id),
    vec      BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class Hit:
    chunk_id: str
    citation: str
    doc: str
    article_title: str
    text: str
    score: float
    bm25_rank: int | None
    vec_rank: int | None

    def to_dict(self) -> dict:
        return asdict(self)


STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines",
    "und", "oder", "im", "am", "auf", "fuer", "für", "von", "zu", "zum", "zur",
    "mit", "bei", "nach", "aus", "ist", "sind", "war", "wie", "was", "wann",
    "wo", "welche", "welchen", "welcher", "welches", "wer", "wird", "werden",
    "sich", "als", "auch", "nicht", "nur", "an", "in", "es", "sie", "man",
    "ueber", "über", "unter", "vor", "durch", "gegen", "ohne", "um", "dass",
}


def _fts_query(raw: str) -> str:
    """Nutzereingabe in eine sichere FTS5-Query uebersetzen.

    FTS5 deutet Zeichen wie " * : - als Operatoren; ungeprueft fuehrt das zu
    Syntaxfehlern bei harmlosen Fragen. Jeder Token wird deshalb gequotet.
    Deutsche Komposita werden zusaetzlich per Praefixsuche abgefangen
    ("Risikomanagement" findet auch "Risikomanagementsystem"). Fuellwoerter
    fliegen raus -- bei reiner OR-Verknuepfung gewinnen sonst Dokumente, die
    viele Allerweltswoerter enthalten statt der Fachbegriffe.
    """
    tokens = re.findall(r"\w+", raw, flags=re.UNICODE)
    tokens = [t for t in tokens if len(t) > 1 and t.lower() not in STOPWORDS]
    if not tokens:
        return ""
    parts = []
    for t in tokens:
        q = t.replace('"', '')
        parts.append(f'"{q}"*' if len(q) >= 5 else f'"{q}"')
    return " OR ".join(parts)


class Store:
    def __init__(self, db_path: Path | None = None):
        self.path = Path(db_path or config.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # SQLite-Verbindungen sind an ihren Erzeugerthread gebunden. MCP-Server
        # und FastAPI fuehren synchrone Aufrufe im Threadpool aus -- eine
        # gemeinsam genutzte Verbindung wirft dort ProgrammingError. Deshalb
        # bekommt jeder Thread seine eigene; SQLite vertraegt parallele Leser.
        self._local = threading.local()
        self._lock = threading.Lock()
        self.conn.executescript(SCHEMA)
        self._ids: list[str] | None = None
        self._matrix: np.ndarray | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ---------- Schreiben ----------

    def rebuild(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == len(vectors), "Chunk- und Vektorzahl muessen uebereinstimmen"
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM chunks_fts")
        cur.execute("DELETE FROM vectors")
        for c, v in zip(chunks, vectors):
            cur.execute(
                "INSERT INTO chunks VALUES (:chunk_id,:doc,:doc_title,:unit,:article,"
                ":article_title,:paragraph,:chapter,:citation,:text)",
                asdict(c),
            )
            haystack = " ".join([c.citation, c.article_title or "", c.text])
            cur.execute("INSERT INTO chunks_fts VALUES (?,?)", (c.chunk_id, haystack))
            cur.execute("INSERT INTO vectors VALUES (?,?)",
                        (c.chunk_id, np.asarray(v, dtype=np.float32).tobytes()))
        cur.execute("INSERT OR REPLACE INTO meta VALUES ('embed_model',?)", (config.EMBED_MODEL,))
        cur.execute("INSERT OR REPLACE INTO meta VALUES ('embed_dim',?)", (str(config.EMBED_DIM),))
        cur.execute("INSERT OR REPLACE INTO meta VALUES ('n_chunks',?)", (str(len(chunks)),))
        self.conn.commit()
        self._ids = self._matrix = None

    # ---------- Lesen ----------

    def stats(self) -> dict:
        rows = dict(self.conn.execute("SELECT key, value FROM meta").fetchall())
        rows["by_doc"] = {
            r["doc"]: r["n"]
            for r in self.conn.execute("SELECT doc, COUNT(*) n FROM chunks GROUP BY doc")
        }
        return rows

    def get(self, chunk_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def get_article(self, doc: str, article: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE doc=? AND article=? AND unit='article' "
            "ORDER BY CAST(paragraph AS REAL), paragraph",
            (doc, str(article)),
        ).fetchall()
        return [dict(r) for r in rows]

    def _load_matrix(self) -> tuple[list[str], np.ndarray]:
        if self._matrix is None:
            with self._lock:
                if self._matrix is None:  # zweite Pruefung unter dem Lock
                    rows = self.conn.execute("SELECT chunk_id, vec FROM vectors").fetchall()
                    ids = [r["chunk_id"] for r in rows]
                    self._matrix = (
                        np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
                        if rows else np.zeros((0, config.EMBED_DIM), dtype=np.float32)
                    )
                    self._ids = ids
        return self._ids or [], self._matrix

    # ---------- Suche ----------

    def search_bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        q = _fts_query(query)
        if not q:
            return []
        rows = self.conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (q, k),
        ).fetchall()
        return [(r["chunk_id"], -float(r["score"])) for r in rows]  # bm25(): kleiner = besser

    def search_vector(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        ids, mat = self._load_matrix()
        if not ids:
            return []
        sims = mat @ np.asarray(qvec, dtype=np.float32)
        top = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in top]

    def search(self, query: str, qvec: np.ndarray | None = None, k: int | None = None,
               doc: str | None = None) -> list[Hit]:
        """Hybrid: BM25 + Vektor, fusioniert per RRF.

        RRF bewertet nur den *Rang*, nicht den Score. Damit muessen die beiden
        voellig unterschiedlich skalierten Scores nicht normalisiert werden --
        genau deshalb ist es robust.
        """
        k = k or config.TOP_K
        pool = max(config.CANDIDATES, k * 4)
        bm = self.search_bm25(query, pool)
        vec = self.search_vector(qvec, pool) if qvec is not None else []

        bm_rank = {cid: i + 1 for i, (cid, _) in enumerate(bm)}
        vec_rank = {cid: i + 1 for i, (cid, _) in enumerate(vec)}

        fused: dict[str, float] = {}
        for ranks in (bm_rank, vec_rank):
            for cid, rank in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (config.RRF_K + rank)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        hits: list[Hit] = []
        for cid, score in ordered:
            row = self.get(cid)
            if not row:
                continue
            if doc and row["doc"] != doc:
                continue
            hits.append(Hit(
                chunk_id=cid, citation=row["citation"], doc=row["doc"],
                article_title=row["article_title"] or "", text=row["text"],
                score=round(score, 6),
                bm25_rank=bm_rank.get(cid), vec_rank=vec_rank.get(cid),
            ))
            if len(hits) >= k:
                break
        return hits


def build_index(raw_dir: Path | None = None, db_path: Path | None = None) -> dict:
    from .ingest import ingest_all
    from .embed import embed_passages

    raw_dir = raw_dir or (config.ROOT / "corpus" / "raw")
    chunks = ingest_all(raw_dir)
    vectors = embed_passages([c.contextual_text() for c in chunks])
    store = Store(db_path)
    store.rebuild(chunks, vectors)
    return store.stats()


if __name__ == "__main__":
    print(json.dumps(build_index(), indent=2, ensure_ascii=False))
