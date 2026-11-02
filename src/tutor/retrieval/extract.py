from pathlib import Path
import pymupdf
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from .types import PageRecord
import hashlib


class FileRejected(ValueError):
    """Raised with a message that can be shown to the student as-is."""


def file_sha256(path) -> str:
    """A fingerprint of the file's contents: the same bytes always give the same hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(path,max_file_mb: int=50) -> list[PageRecord]:
    """Return the text of each page or slide, or raise FileRejected."""
    path=Path(path)
    suffix=path.suffix.lower()

    if suffix not in (".pdf",".pptx"):
        raise FileRejected(f"{path.name}: only PowerPoint (.pptx) and PDF files are supported.")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_file_mb:
        raise FileRejected(f"{path.name} is {size_mb:.0f} MB; the limit is {max_file_mb} MB.")

    if suffix == ".pdf":
        return _extract_pdf(path)
    
    return _extract_pptx(path)

def _extract_pdf(path: Path) -> list[PageRecord]:
    try:
        doc=pymupdf.open(path)
    except Exception as e:
        raise FileRejected(f"{path.name} could not be opened as a PDF.") from e

    with doc:
        kind = "slide" if _looks_like_slides(doc) else "page"
        pages = []
        for number, page in enumerate(doc, start=1):
            text = page.get_text()
            pages.append(PageRecord(file=path.name, page=number, text=text, kind=kind))
    return pages


def _looks_like_slides(doc) -> bool:
    """A PDF counts as a slide deck if most of its pages are landscape."""
    landscape = sum(1 for page in doc if page.rect.width > page.rect.height)
    return landscape > len(doc) / 2

def _extract_pptx(path: Path) -> list[PageRecord]:
    try:
        prs = Presentation(str(path))
    except Exception as e:
        raise FileRejected(f"{path.name} could not be opened as a PowerPoint file.") from e

    pages = []
    for number, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            parts.extend(_shape_text(shape))

        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"Speaker notes: {notes}")

        pages.append(PageRecord(file=path.name, page=number, text="\n\n".join(parts), kind="slide"))
    return pages


def _shape_text(shape) -> list[str]:
    """All text inside one shape, including groups and tables."""
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        texts = []
        for inner in shape.shapes:
            texts.extend(_shape_text(inner))
        return texts

    if shape.has_text_frame:
        text = shape.text_frame.text.strip()
        return [text] if text else []

    if shape.has_table:
        rows = []
        for row in shape.table.rows:
            rows.append(" | ".join(cell.text.strip() for cell in row.cells))
        return ["\n".join(rows)]

    return []