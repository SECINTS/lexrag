"""Tracing und Kostenrechnung.

Jede Anfrage erzeugt eine JSONL-Zeile: Schritte, abgerufene Chunks, Token,
Kosten, Latenz. Das ist die Grundlage fuer die Eval-Kennzahlen und fuer die
Frage, die in Produktion immer kommt: was kostet eine Anfrage?
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import config


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    model: str = ""
    steps: list[dict] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    grounded: bool | None = None
    error: str | None = None

    def add_step(self, kind: str, **payload) -> None:
        self.steps.append({"kind": kind, "at_ms": int((time.time() - self._t0) * 1000), **payload})

    def add_usage(self, usage) -> None:
        self.tokens_in += getattr(usage, "input_tokens", 0) or 0
        self.tokens_out += getattr(usage, "output_tokens", 0) or 0

    def finalize(self) -> None:
        p = config.price_for(self.model)
        self.cost_usd = round(
            self.tokens_in / 1_000_000 * p["in"] + self.tokens_out / 1_000_000 * p["out"], 6
        )
        self.latency_ms = int((time.time() - self._t0) * 1000)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_t0", None)
        return d


@contextmanager
def tracer(question: str, model: str, write: bool = True):
    t = Trace(question=question, model=model)
    t._t0 = time.time()
    try:
        yield t
    except Exception as exc:  # Fehler gehoeren in den Trace, nicht nur in stderr
        t.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        t.finalize()
        if write:
            config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
            path = config.TRACE_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
