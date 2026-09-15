"""Lokale Embeddings via fastembed (ONNX) -- laeuft ohne zusaetzlichen API-Key.

Bewusst hinter einer schmalen Schnittstelle: ein Wechsel auf einen gehosteten
Anbieter (Voyage, OpenAI-kompatibel) beruehrt nur dieses Modul.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding
    config.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=config.EMBED_MODEL,
        threads=config.ONNX_THREADS,
        cache_dir=str(config.MODEL_CACHE),
    )


def embed_passages(texts: list[str], progress: bool = False) -> np.ndarray:
    """Bettet in kleinen Stapeln ein, damit der Spitzenspeicher beschraenkt bleibt.

    Das Ergebnis selbst ist winzig (768 Chunks x 768 Dim x 4 B = 2,4 MB); teuer
    sind allein die Aktivierungspuffer waehrend der Inferenz. Die Groesse steht
    in `config.EMBED_BATCH`, die Prozesszahl in `config.EMBED_PARALLEL`.
    """
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

    bs = max(1, config.EMBED_BATCH)
    out: list[np.ndarray] = []
    model = _model()
    for start in range(0, len(texts), bs):
        stapel = texts[start:start + bs]
        vecs = list(model.embed(stapel, batch_size=bs, parallel=config.EMBED_PARALLEL))
        out.append(_normalize(np.asarray(vecs, dtype=np.float32)))
        if progress:
            print(f"    eingebettet {min(start + bs, len(texts))}/{len(texts)}",
                  end="\r", flush=True)
    if progress:
        print()
    return np.vstack(out)


def embed_query(text: str) -> np.ndarray:
    vec = np.array(list(_model().query_embed([text], parallel=1))[0], dtype=np.float32)
    return _normalize(vec.reshape(1, -1))[0]


def _normalize(m: np.ndarray) -> np.ndarray:
    """L2-Normierung -- danach ist das Skalarprodukt die Kosinus-Aehnlichkeit."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
