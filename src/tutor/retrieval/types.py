"""Records passed between retrieval stages.

extract.py makes PageRecords, chunk.py turns them into Chunks, and search.py
returns Passages. Other packages (agent, app) only ever see Chunk and Passage.
"""
from dataclasses import dataclass
from typing import Literal


Kind = Literal["page", "slide"]


@dataclass(frozen=True)
class PageRecord:
    """The text of one PDF page or one slide, before it is cut into chunks."""
    file: str
    page: int
    text: str
    kind: Kind


@dataclass(frozen=True)
class Chunk:
    """One searchable passage: the unit that is embedded, indexed and cited.

    A passage from a PDF can run across a page break, so it records both the
    first and last page it covers. For slides, page and page_end are the same.
    """
    id: str
    file: str
    page: int
    page_end: int
    text: str
    kind: Kind

    def citation(self) -> str:
        """The label shown to the student, e.g. 'Lecture3.pdf, page 12' or 'Week5.pptx, slides 4–6'."""
        if self.page == self.page_end:
            return f"{self.file}, {self.kind} {self.page}"
        return f"{self.file}, {self.kind}s {self.page}–{self.page_end}"


@dataclass(frozen=True)
class Passage(Chunk):
    """A chunk returned by search, with how well it matched the question."""
    score: float = 0.0
