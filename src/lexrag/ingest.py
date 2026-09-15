"""Parse offizielle EUR-Lex-Rechtstexte in zitierfaehige Chunks.

Korpusentscheidung: nur operativer Normtext (Artikel + Anhaenge).
Erwaegungsgruende (rct_*) bleiben draussen -- sie entfalten keine unmittelbare
Rechtswirkung und erzeugen Beinahe-Duplikate, die das Retrieval verschlechtern.
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MIN_CHARS = 200
MAX_CHARS = 1800
# Anhaenge sind Aufzaehlungen: an den nummerierten Punkten trennen, statt
# 1800-Zeichen-Bloecke zu bilden, deren Einbettung zu einem nichtssagenden
# Schwerpunkt mittelt.
ANNEX_ITEM = re.compile(r"(?=(?:^|\s)\d{1,2}\.\s+[A-ZÄÖÜ])")
# EUR-Lex setzt Fliesstext als oj-normal, Tabellenzellen aber als oj-tbl-txt.
# NIS2 fuehrt die Spalte "Art der Einrichtung" durchgaengig als oj-tbl-txt --
# wer nur oj-normal liest, verliert dort 165 Absaetze und damit ganze Sektoren.
TEXT_CLASSES = ("oj-normal", "oj-tbl-txt")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    doc_title: str
    unit: str
    article: str
    article_title: str
    paragraph: str
    chapter: str
    citation: str
    text: str

    def contextual_text(self) -> str:
        """Was eingebettet wird: Pfad + Text, damit der Vektor den Kontext kennt."""
        parts = [self.doc_title]
        if self.chapter:
            parts.append(self.chapter)
        if self.unit == "article":
            head = f"Artikel {self.article}"
            if self.article_title:
                head += f" ({self.article_title})"
            if self.paragraph:
                head += f" Absatz {self.paragraph}"
            parts.append(head)
        else:
            head = f"Anhang {self.article}"
            if self.article_title:
                head += f" — {self.article_title}"
            parts.append(head)
        return " > ".join(parts) + ": " + self.text


def _is_substantive(text: str) -> bool:
    """Verwirft Fragmente ohne Textgehalt.

    EUR-Lex-Tabellen fuehren Listenmarker und Inhalt in getrennten Spalten;
    beim Einsammeln entstehen daraus Ketten wie "1. 2. 3. 4.". Solche Chunks
    tragen nichts bei und verwaessern nur den Index.
    """
    if not text:
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    return letters / len(text) >= 0.5


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.replace("\xa0", " "))
    return re.sub(r"\s+", " ", s).strip()


def _split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.;:!?])\s+", text):
        if cur and len(cur) + len(sent) + 1 > limit:
            out.append(cur.strip())
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur.strip())
    hard: list[str] = []
    for piece in out:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit) or limit
            hard.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            hard.append(piece)
    return hard


def _pack(units: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Kurze Absaetze zusammenfassen, lange aufteilen. Nie ueber Artikelgrenzen."""
    packed: list[tuple[str, str]] = []
    buf_no, buf_txt = "", ""
    for no, txt in units:
        if buf_txt and len(buf_txt) >= MIN_CHARS:
            packed.append((buf_no, buf_txt))
            buf_no, buf_txt = no, txt
        elif buf_txt:
            buf_no = buf_no or no
            buf_txt = f"{buf_txt} {txt}".strip()
        else:
            buf_no, buf_txt = no, txt
    if buf_txt:
        packed.append((buf_no, buf_txt))

    out: list[tuple[str, str]] = []
    for no, txt in packed:
        for i, piece in enumerate(_split_long(txt)):
            out.append((no if i == 0 else f"{no}.{i+1}", piece))
    return out


def _chapter_index(soup: BeautifulSoup) -> dict[int, str]:
    """Position im Dokument -> zugehoeriges Kapitel."""
    marks: list[tuple[int, str]] = []
    for p in soup.find_all("p", class_="oj-ti-section-1"):
        label = _clean(p.get_text())
        holder = p.find_parent("div")
        title = ""
        if holder:
            t = holder.find("p", class_="oj-ti-section-2")
            if t:
                title = _clean(t.get_text())
        marks.append((id(p), f"{label} — {title}" if title else label))
    return dict(marks)


def parse_document(html_path: Path, doc: str, doc_title: str, short: str) -> list[Chunk]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    chunks: list[Chunk] = []

    # --- Kapitelzuordnung ueber Dokumentreihenfolge ---
    order = list(soup.find_all(["p", "div"]))
    pos = {id(el): i for i, el in enumerate(order)}
    chapter_marks: list[tuple[int, str]] = []
    for p in soup.find_all("p", class_="oj-ti-section-1"):
        label = _clean(p.get_text())
        title = ""
        parent = p.parent
        if parent:
            t = parent.find("p", class_="oj-ti-section-2")
            if t:
                title = _clean(t.get_text())
        chapter_marks.append((pos.get(id(p), 0), f"{label} — {title}" if title else label))
    chapter_marks.sort()

    def chapter_for(i: int) -> str:
        cur = ""
        for at, name in chapter_marks:
            if at <= i:
                cur = name
            else:
                break
        return cur

    # --- Artikel ---
    for div in soup.find_all("div", class_="eli-subdivision"):
        did = div.get("id", "")
        if not did.startswith("art_"):
            continue
        num_el = div.find("p", class_="oj-ti-art")
        if not num_el:
            continue
        art_no = _clean(num_el.get_text()).replace("Artikel", "").strip()
        title_el = div.find("p", class_="oj-sti-art")
        art_title = _clean(title_el.get_text()) if title_el else ""
        chapter = chapter_for(pos.get(id(div), 0))

        units: list[tuple[str, str]] = []
        seen_para_divs = [d for d in div.find_all("div", recursive=False) if re.match(r"^\d+\.\d+$", d.get("id", ""))]
        if seen_para_divs:
            for pd in seen_para_divs:
                txt = " ".join(_clean(p.get_text()) for p in pd.find_all("p", class_=list(TEXT_CLASSES)))
                txt = _clean(txt)
                if not txt:
                    continue
                m = re.match(r"^\((\d+)\)\s*", txt)
                no = m.group(1) if m else pd["id"].split(".")[-1].lstrip("0")
                units.append((no, re.sub(r"^\(\d+\)\s*", "", txt)))
        else:
            body = " ".join(_clean(p.get_text()) for p in div.find_all("p", class_=list(TEXT_CLASSES)))
            if _clean(body):
                units.append(("", _clean(body)))

        for para_no, text in _pack(units):
            if len(text) < 40 or not _is_substantive(text):
                continue
            cid = f"{doc}:art_{art_no}" + (f":{para_no}" if para_no else "")
            cit = f"{short} Art. {art_no}" + (f" Abs. {para_no}" if para_no else "")
            chunks.append(Chunk(cid, doc, doc_title, "article", art_no, art_title,
                                para_no, chapter, cit, text))

    # --- Anhaenge ---
    for div in soup.find_all("div", class_="eli-container"):
        ti = div.find("p", class_="oj-doc-ti")
        head = div.find("p", class_="oj-ti-annex") or div.find("p", class_="oj-ti-grseq-1")
        label_el = div.find(string=re.compile(r"ANHANG", re.I))
        if not (ti or label_el):
            continue
        raw_label = _clean(label_el) if label_el else ""
        m = re.search(r"ANHANG\s+([IVXL]+|\d+)", raw_label, re.I)
        anx_no = m.group(1) if m else ""
        if not anx_no:
            continue
        titles = [_clean(t.get_text()) for t in div.find_all("p", class_="oj-doc-ti")]
        titles = [t for t in titles if t and not re.match(r"^ANHANG\b", t, re.I)]
        anx_title = titles[0] if titles else ""
        paras = [_clean(p.get_text()) for p in div.find_all("p", class_=list(TEXT_CLASSES))]
        body = _clean(" ".join(x for x in paras if x))
        if len(body) < 60:
            continue
        raw_items = [x.strip() for x in ANNEX_ITEM.split(body) if x.strip()]
        # "3. Bankwesen" allein ist 12 Zeichen lang und fiele sonst der
        # Mindestlaenge zum Opfer -- samt des Sektors, den es benennt.
        items: list[str] = []
        for it in raw_items:
            if items and len(items[-1]) < 60:
                items[-1] = f"{items[-1]} {it}".strip()
            else:
                items.append(it)
        items = [x for x in items if len(x) >= 60]
        pieces: list[str] = []
        for item in (items or [body]):
            pieces.extend(_split_long(item))
        keep = [x for x in pieces if len(x) >= 60 and _is_substantive(x)]
        for i, piece in enumerate(keep, start=1):
            cid = f"{doc}:anx_{anx_no}:{i}"
            cit = f"{short} Anhang {anx_no}"
            chunks.append(Chunk(cid, doc, doc_title, "annex", anx_no, anx_title,
                                str(i), "", cit, piece))

    return chunks


DOCS = [
    ("ai_act", "ai_act_de.html",
     "Verordnung (EU) 2024/1689 (KI-Verordnung / EU AI Act)", "AI Act"),
    ("nis2", "nis2_de.html",
     "Richtlinie (EU) 2022/2555 (NIS-2-Richtlinie)", "NIS2"),
]


def ingest_all(raw_dir: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for doc, fname, title, short in DOCS:
        out.extend(parse_document(raw_dir / fname, doc, title, short))
    return out


if __name__ == "__main__":
    import json
    import sys

    raw = Path(__file__).resolve().parents[2] / "corpus" / "raw"
    cs = ingest_all(raw)
    by_doc: dict[str, int] = {}
    by_unit: dict[str, int] = {}
    for c in cs:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1
        by_unit[c.unit] = by_unit.get(c.unit, 0) + 1
    print(f"Chunks gesamt: {len(cs)}")
    print(f"  nach Dokument: {by_doc}")
    print(f"  nach Einheit:  {by_unit}")
    lens = sorted(len(c.text) for c in cs)
    if lens:
        print(f"  Laenge  min={lens[0]}  median={lens[len(lens)//2]}  max={lens[-1]}")
    print("\nBeispiele:")
    for c in cs[:2] + cs[len(cs)//2:len(cs)//2+1]:
        print(f"  [{c.citation}] {c.text[:110]}...")
    if "--dump" in sys.argv:
        out = Path(__file__).resolve().parents[2] / "corpus" / "chunks.jsonl"
        out.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in cs), encoding="utf-8")
        print(f"\ngeschrieben: {out}")
