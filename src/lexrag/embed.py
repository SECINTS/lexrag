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
    return TextEmbedding(model_name=config.EMBED_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(_model().embed(texts)), dtype=np.float32)
    return _normalize(vecs)


def embed_query(text: str) -> np.ndarray:
    vec = np.array(list(_model().query_embed([text]))[0], dtype=np.float32)
    return _normalize(vec.reshape(1, -1))[0]


def _normalize(m: np.ndarray) -> np.ndarray:
    """L2-Normierung -- danach ist das Skalarprodukt die Kosinus-Aehnlichkeit."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
