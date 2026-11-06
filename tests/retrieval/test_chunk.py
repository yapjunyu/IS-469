import pytest

from tutor.retrieval.chunk import chunk_pages
from tutor.retrieval.types import PageRecord

HASH = "abcdef1234567890"


def doc(*texts):
    return [PageRecord("notes.pdf", i, t, "page") for i, t in enumerate(texts, start=1)]


def deck(*texts):
    return [PageRecord("deck.pptx", i, t, "slide") for i, t in enumerate(texts, start=1)]


def numbered(tag, n):
    return " ".join(f"{tag}{i}" for i in range(n)) + "."


# fixed

def test_fixed_hand_example():
    pages = doc("LoRA freezes the weights", "and trains small matrices")
    chunks = chunk_pages(pages, HASH, chunk_size=4, chunk_overlap=1, method="fixed")
    assert [(c.text, c.citation()) for c in chunks] == [
        ("LoRA freezes the weights", "notes.pdf, page 1"),
        ("weights and trains small", "notes.pdf, pages 1–2"),
        ("small matrices", "notes.pdf, page 2"),
    ]


def test_fixed_without_overlap_has_no_repeated_words():
    pages = doc(numbered("w", 250))
    words = [w for c in chunk_pages(pages, HASH, chunk_size=100, chunk_overlap=0) for w in c.text.split()]
    assert words == numbered("w", 250).split()


# recursive

def test_recursive_cuts_at_sentence_ends_with_word_overlap():
    pages = doc("LoRA is a cheap fine-tuning method.\n\nIt freezes the original weights.",
                "It trains two small matrices called A and B. They are added to each layer.")
    chunks = chunk_pages(pages, HASH, chunk_size=14, chunk_overlap=6, method="recursive")
    # each chunk ends at a sentence end; the next one repeats the last 6 words
    assert [c.text for c in chunks] == [
        "LoRA is a cheap fine-tuning method. It freezes the original weights.",
        "It freezes the original weights. It trains two small matrices called A and B.",
        "small matrices called A and B. They are added to each layer.",
    ]
    assert chunks[1].citation() == "notes.pdf, pages 1–2"


# slides

def test_slides_default_to_one_chunk_per_slide():
    chunks = chunk_pages(deck("Intro", "LoRA basics", "Summary"), HASH)
    assert [c.citation() for c in chunks] == ["deck.pptx, slide 1", "deck.pptx, slide 2", "deck.pptx, slide 3"]


def test_slides_can_be_packed_with_recursive():
    pages = deck(numbered("a", 5), numbered("b", 40), numbered("c", 8), numbered("d", 50))
    chunks = chunk_pages(pages, HASH, chunk_size=60, chunk_overlap=0, slide_method="recursive")
    assert [c.citation() for c in chunks] == ["deck.pptx, slides 1–3", "deck.pptx, slide 4"]
    assert all(c.kind == "slide" for c in chunks)


# validation

@pytest.mark.parametrize("kwargs", [dict(chunk_overlap=200), dict(method="recurisve")])
def test_bad_settings_raise(kwargs):
    with pytest.raises(ValueError):
        chunk_pages(doc("some words here"), HASH, chunk_size=200, **kwargs)


# images

def test_every_image_gets_its_own_chunk_after_the_text_chunks():
    pages = [PageRecord("notes.pdf", 1, "LoRA freezes the weights", "page", image_paths=("a.png", "b.png")),
             PageRecord("notes.pdf", 2, "", "page", image_paths=("c.png",))]  # a page that is only a figure
    chunks = chunk_pages(pages, HASH, chunk_size=10, chunk_overlap=0)
    assert [(c.text, c.image_path, c.page) for c in chunks] == [
        ("LoRA freezes the weights", None, 1), ("", "a.png", 1), ("", "b.png", 1), ("", "c.png", 2)]
    assert len({c.id for c in chunks}) == 4
