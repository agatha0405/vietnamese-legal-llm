"""Extract traceable article and clause text from the pinned official PDFs."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
import re
import unicodedata
from urllib.request import urlopen

import pymupdf


ARTICLE = re.compile(r"^Điều\s+(\d+[a-z]?)\.\s*(.*)$")
CLAUSE = re.compile(r"^(\d+[a-z]?)\.\s+(.*)$")
SECTION = re.compile(r"^(?:Chương\s+[IVX]+|Mục\s+\d+)(?:\b|$)", re.I)
DATE = re.compile(r"(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})")
BASE_DATE = "2025-01-01"


def clean(text):
    text = unicodedata.normalize("NFC", text.replace("\u00ad", "").replace("\u200b", ""))
    return " ".join(text.split())


@dataclass
class Passage:
    lines: list[str] = field(default_factory=list)
    pages: set[int] = field(default_factory=set)
    notes: set[str] = field(default_factory=set)

    def append(self, text, page, notes):
        self.lines.append(text)
        self.pages.add(page)
        self.notes.update(notes)

    @property
    def text(self):
        return clean(" ".join(self.lines))


@dataclass
class Article:
    heading: Passage = field(default_factory=Passage)
    clauses: dict[str, Passage] = field(default_factory=dict)


def source_file(source, raw_dir, download):
    path = raw_dir / f"{source['key']}.pdf"
    if not path.is_file() and download:
        with urlopen(source["pdf_url"], timeout=60) as response:
            content = response.read()
        if sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"Official PDF checksum changed: {source['edition']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError(f"PDF checksum mismatch: {path}")
    return path


def split_articles(path, last_article):
    articles = {}
    footnotes = defaultdict(list)
    current = None
    clause = None
    active_note = None
    finished = False
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, 1):
            blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
            for block in blocks["blocks"]:
                for line in block["lines"]:
                    spans = line["spans"]
                    nonempty = [s for s in spans if s["text"].strip()]
                    if not nonempty:
                        continue
                    body = [s for s in spans if s["size"] >= 13.5 and not s["flags"] & 1]
                    if not any(s["text"].strip() for s in body):
                        first = nonempty[0]
                        if len(nonempty) > 1 and first["size"] <= 9 and first["text"].strip().isdigit():
                            active_note = first["text"].strip()
                            footnotes[active_note].append(clean("".join(s["text"] for s in spans[spans.index(first) + 1:])))
                        elif active_note and 9.5 <= first["size"] <= 12.5:
                            footnotes[active_note].append(clean("".join(s["text"] for s in spans)))
                        continue
                    text = clean("".join(s["text"] for s in body))
                    notes = {
                        s["text"].strip()
                        for s in spans
                        if s["flags"] & 1 and s["text"].strip().isdigit()
                    }
                    bold = all(s["flags"] & 16 for s in body if s["text"].strip())
                    heading = ARTICLE.match(text) if bold else None
                    if heading:
                        number, title = heading.groups()
                        if int(re.match(r"\d+", number)[0]) > last_article:
                            finished = True
                        if finished:
                            continue
                        if number in articles:
                            raise ValueError(f"Duplicate article {number} in {path}")
                        current = articles[number] = Article()
                        current.heading.append(title, page_number, notes)
                        clause = None
                    elif SECTION.match(text) and bold:
                        current = None
                    elif current is not None and not finished:
                        if bold and clause is None:
                            current.heading.append(text, page_number, notes)
                            continue
                        match = CLAUSE.match(text) if line["bbox"][0] >= 100 else None
                        if match:
                            clause, text = match.groups()
                            if clause in current.clauses:
                                raise ValueError(f"Duplicate clause {clause} in {current.heading.text}")
                        elif clause is None:
                            clause = "0"
                        passage = current.clauses.setdefault(clause, Passage())
                        passage.append(text, page_number, notes)
    missing = {str(i) for i in range(1, last_article + 1)} - articles.keys()
    if missing:
        raise ValueError(f"Article extraction incomplete for {path}: {sorted(missing)}")
    return articles, {key: clean(" ".join(lines)) for key, lines in footnotes.items()}


def effective_date(source, article, clause, notes, footnotes):
    overrides = source["effective_overrides"]
    dates = [overrides.get(f"{article}.{clause}", overrides.get(article, BASE_DATE))]
    for note in sorted(notes):
        text = footnotes.get(note, "")
        marker = re.search(r"có hiệu lực.{0,40}?ngày\s+" + DATE.pattern, text)
        if marker is None:
            return None
        day, month, year = map(int, marker.groups())
        dates.append(date(year, month, day).isoformat())
    return max(dates)
