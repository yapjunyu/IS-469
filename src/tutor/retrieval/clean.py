"""Topic 1 text normalisation: tidy extracted pages before they are chunked."""
import re
import unicodedata
from collections import Counter
from dataclasses import replace

from .types import PageRecord

MAX_BOILERPLATE_WORDS = 8  # headers and footers are short; longer lines are always kept

_INVISIBLE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f­​-‍﻿]")
_PAGE_NUMBER = re.compile(r"^(page|slide)?\s*\d+(\s*(of|/)\s*\d+)?$", re.I)
_BULLET = re.compile(r"^([-•▪◦*–]|\d+[.)])\s+")


def clean_pages(pages: list[PageRecord]) -> list[PageRecord]:
    """Normalise characters, drop repeated headers/footers and page numbers,
    rejoin wrapped lines in documents, and drop pages left empty.
    Case is kept as-is; BM25 lowercases its own tokens.
    """
    pages = [replace(p, text=_normalize(p.text)) for p in pages]
    pages = _strip_repeated_lines(pages)
    cleaned = []
    for p in pages:
        text = p.text
        # Slide line breaks are bullets; PDF page line breaks are mostly just wrapping.
        if p.kind == "page":
            text = _join_wrapped_lines(text)
        text = _tidy_spaces(text)
        if text:
            cleaned.append(replace(p, text=text))
    return cleaned


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # ligatures, non-breaking spaces, full-width chars
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _INVISIBLE.sub("", text)


def _strip_repeated_lines(pages: list[PageRecord]) -> list[PageRecord]:
    """Remove short lines that appear on at least half the pages, and page numbers.
    A bare number only counts as a page number on a page's first or last line;
    elsewhere it is usually a list marker ("01  What RAG is") and is kept.
    """
    counts = Counter()
    for p in pages:
        counts.update({line.strip() for line in p.text.split("\n") if _is_short(line)})

    threshold = max(3, len(pages) // 2)
    repeated = {line for line, n in counts.items() if n >= threshold}

    stripped = []
    for p in pages:
        lines = [line for line in p.text.split("\n") if line.strip() not in repeated]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and _PAGE_NUMBER.match(lines[-1].strip()):
            lines.pop()
        if lines and _PAGE_NUMBER.match(lines[0].strip()):
            lines.pop(0)
        stripped.append(replace(p, text="\n".join(lines)))
    return stripped


def _is_short(line: str) -> bool:
    return 0 < len(line.split()) <= MAX_BOILERPLATE_WORDS


def _join_wrapped_lines(text: str) -> str:
    # Undo words split across a line break ("informa-\ntion"). A real hyphenated
    # word that happens to break at its hyphen gets merged too; that trade-off is accepted.
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            out.append("")  # keep paragraph breaks
        elif out and out[-1] and not _BULLET.match(line):
            out[-1] += " " + line  # continuation of the previous line
        else:
            out.append(line)
    return "\n".join(out)


def _tidy_spaces(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
