from collections import Counter
from pathlib import Path
import pymupdf
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from .types import PageRecord
import hashlib
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "data" / "images"   # the PNGs that get embedded as image chunks
IMAGE_DPI = 100     # whole slides / pages
FIGURE_DPI = 150    # a cropped picture is small, so it is rendered sharper
MIN_DRAWINGS = 20   # a page with more drawn shapes than this counts as a diagram (a guess; tune on real decks)
MIN_SHAPE_PT = 15   # a drawn shape smaller than this on both sides (about 5 mm) is a bullet marker, not part of a diagram
MIN_FIGURE_PT = 60  # a picture smaller than this on the page (about 2 cm) is an icon or bullet, not a figure
# macOS installs LibreOffice here without putting it on PATH
_SOFFICE_MAC = "/Applications/LibreOffice.app/Contents/MacOS/soffice"


class FileRejected(ValueError):
    """Raised with a message that can be shown to the student as-is."""


def file_sha256(path) -> str:
    """A fingerprint of the file's contents: the same bytes always give the same hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(path,max_file_mb: int=50, images: bool=True) -> list[PageRecord]:
    """Return the text of each page or slide, or raise FileRejected.

    images=False skips saving page images (a text-only run has no use for them)."""
    path=Path(path)
    suffix=path.suffix.lower()

    if suffix not in (".pdf",".pptx"):
        raise FileRejected(f"{path.name}: only PowerPoint (.pptx) and PDF files are supported.")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_file_mb:
        raise FileRejected(f"{path.name} is {size_mb:.0f} MB; the limit is {max_file_mb} MB.")

    digest = file_sha256(path)[:12]  # names the folder this file's page images are saved in
    if suffix == ".pdf":
        return _extract_pdf(path, digest, images)

    return _extract_pptx(path, digest, images)


def _extract_pdf(path: Path, digest: str, images: bool) -> list[PageRecord]:
    try:
        doc=pymupdf.open(path)
    except Exception as e:
        raise FileRejected(f"{path.name} could not be opened as a PDF.") from e

    with doc:
        kind = "slide" if _looks_like_slides(doc) else "page"
        repeated = _repeated_images(doc) if images else set()  # a logo on every page or slide says nothing about it
        pages = []
        for number, page in enumerate(doc, start=1):
            _remove_notes(page)  # first, so neither the text nor the images include them
            text = page.get_text()
            if not images:
                paths = ()
            elif kind == "slide":
                # if visual then whole slide is 1PNG, as a slide is mostly one picture
                whole = _render_if_visual(page, digest, number, repeated)
                paths = (whole,) if whole else ()
            else:
                # if is a longdoc pdf not slide then cropped each figure to its own PNG as a page is mostly text with a figure in corner
                paths = _page_figures(page, digest, number, repeated)
            pages.append(PageRecord(file=path.name, page=number, text=text, kind=kind,
                                    image_paths=paths))
    return pages


def _remove_notes(page) -> None:
    """Remove student annotations (notes, highlights, pen marks) from the page in memory,
    so only the original course content is indexed. The file on disk is not changed."""
    for annot in list(page.annots()):
        page.delete_annot(annot)


def _n_shapes(page) -> int:
    """How many drawn shapes on the page are big enough to be part of a diagram.

    A bulleted text page can hold 20+ tiny drawn bullet markers, so counting every
    drawing would take it for a diagram; only shapes with a side of MIN_SHAPE_PT or more count."""
    return sum(1 for d in page.get_drawings() if max(d["rect"].width, d["rect"].height) >= MIN_SHAPE_PT)


def _png_path(digest: str, name: str) -> Path:
    out = IMAGE_DIR / digest / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _pictures(page, repeated: set[int]) -> list:
    """Where the page's real pictures sit (clipped to the page): not a repeated logo,
    and not smaller than MIN_FIGURE_PT, which is an icon or bullet."""
    found = []
    for info in page.get_image_info(xrefs=True):
        rect = pymupdf.Rect(info["bbox"]) & page.rect  # keep only the part that is on the page
        if info["xref"] in repeated or rect.width < MIN_FIGURE_PT or rect.height < MIN_FIGURE_PT:
            continue
        found.append(rect)
    return found


def _render_if_visual(page, digest: str, number: int, repeated: frozenset | set = frozenset()) -> str | None:
    """Slides: save the whole page as a PNG and return its path, but only if it has
    a real picture (not the template's logo) or many substantial drawn shapes. Plain
    text slides return None, since their text chunks already cover them. Rendering the
    whole slide (not just the embedded pictures) also catches diagrams that are drawn
    as shapes."""
    if not _pictures(page, repeated) and _n_shapes(page) <= MIN_DRAWINGS:
        return None
    out = _png_path(digest, f"p{number:04d}")
    page.get_pixmap(dpi=IMAGE_DPI).save(str(out))
    return str(out)


def _repeated_images(doc) -> set[int]:
    """Pictures that sit on at least half the pages (a logo or letterhead), which
    say nothing about the content. Same rule clean.py uses for repeated lines.

    Counts pictures actually placed on each page (get_image_info), not get_images(): in a PDF
    LibreOffice makes from a .pptx, get_images() lists every picture in the deck on every page."""
    pages_with = Counter(xref for page in doc for xref in {i["xref"] for i in page.get_image_info(xrefs=True)})
    threshold = max(3, len(doc) // 2)
    return {xref for xref, n in pages_with.items() if xref and n >= threshold}


def _page_figures(page, digest: str, number: int, repeated: set[int]) -> tuple[str, ...]:
    """Document pages: one PNG per picture on the page, cropped to where it sits, with
    icons and repeated logos skipped. A page whose figures are drawn as shapes (no
    embedded pictures) is saved whole instead."""
    paths = []
    for rect in _pictures(page, repeated):
        out = _png_path(digest, f"p{number:04d}_{len(paths) + 1}")
        page.get_pixmap(dpi=FIGURE_DPI, clip=rect).save(str(out))
        paths.append(str(out))
    if not paths and _n_shapes(page) > MIN_DRAWINGS:
        out = _png_path(digest, f"p{number:04d}")
        page.get_pixmap(dpi=IMAGE_DPI).save(str(out))
        paths.append(str(out))
    return tuple(paths)

def _looks_like_slides(doc) -> bool:
    """A PDF counts as a slide deck if most of its pages are landscape."""
    landscape = sum(1 for page in doc if page.rect.width > page.rect.height)
    return landscape > len(doc) / 2

def _render_pptx_slides(path: Path, digest: str, n_slides: int) -> dict[int, str]:
    """Slide number -> PNG path, for the slides that have pictures or diagrams.

    python-pptx can't draw slides, so LibreOffice converts the deck to a PDF first
    and the PDF pages are rendered like any other PDF. Returns {} (text-only deck)
    if LibreOffice is missing, fails, or exports a different number of pages than
    the deck has slides (it leaves out hidden slides, which would shift the numbers).
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None and Path(_SOFFICE_MAC).exists():
        soffice = _SOFFICE_MAC
    if soffice is None:
        return {}

    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(path)],
                           check=True, timeout=120, capture_output=True)
            doc = pymupdf.open(Path(tmp) / f"{path.stem}.pdf")
        except Exception:  # not installed properly, timed out, or produced no PDF
            return {}
        with doc:
            if len(doc) != n_slides:
                return {}
            repeated = _repeated_images(doc)  # the template logo that sits on nearly every slide
            rendered = {}
            for number, page in enumerate(doc, start=1):
                image_path = _render_if_visual(page, digest, number, repeated)
                if image_path:
                    rendered[number] = image_path
            return rendered


def _extract_pptx(path: Path, digest: str, with_images: bool) -> list[PageRecord]:
    try:
        prs = Presentation(str(path))
    except Exception as e:
        raise FileRejected(f"{path.name} could not be opened as a PowerPoint file.") from e

    images = _render_pptx_slides(path, digest, len(prs.slides)) if with_images else {}
    pages = []
    for number, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            parts.extend(_shape_text(shape))

        pages.append(PageRecord(file=path.name, page=number, text="\n\n".join(parts), kind="slide",
                                image_paths=(images[number],) if number in images else ()))
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