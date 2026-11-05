from tutor.retrieval.clean import clean_pages
from tutor.retrieval.types import PageRecord


def page(n, text, kind="page"):
    return PageRecord(file="notes.pdf", page=n, text=text, kind=kind)


def test_repeated_footer_and_page_numbers_are_removed():
    pages = [page(i, f"IS469 Lecture Notes\nBody text {i} about LoRA.\n{i}", kind="slide") for i in range(1, 5)]
    assert [p.text for p in clean_pages(pages)] == [f"Body text {i} about LoRA." for i in range(1, 5)]


def test_list_numbers_inside_a_page_are_kept():
    [p] = clean_pages([page(1, "Agenda\n01\nWhat RAG is\n02\nWhy it fails\n7", kind="slide")])
    assert p.text == "Agenda\n01\nWhat RAG is\n02\nWhy it fails"


def test_long_repeated_line_is_kept():
    sentence = "Retrieval augmented generation looks things up in your documents before answering."
    pages = [page(i, f"{sentence}\nPage {i}") for i in range(1, 5)]
    assert all(p.text == sentence for p in clean_pages(pages))


def test_line_on_few_pages_is_kept():
    pages = [page(1, "Agenda\nText one."), page(2, "Agenda\nText two."), page(3, "Other\nText three."),
             page(4, "More\nText four."), page(5, "Last\nText five."), page(6, "End\nText six.")]
    assert clean_pages(pages)[0].text == "Agenda Text one."


def test_wrapped_lines_and_hyphenation_are_rejoined():
    [p] = clean_pages([page(1, "Retrieval-augmented genera-\ntion looks things\nup first.\n\nNew paragraph.")])
    assert p.text == "Retrieval-augmented generation looks things up first.\n\nNew paragraph."


def test_bullets_stay_on_their_own_lines():
    [p] = clean_pages([page(1, "Two kinds:\n- dense search\n- keyword search")])
    assert p.text == "Two kinds:\n- dense search\n- keyword search"


def test_slide_lines_are_not_joined():
    [p] = clean_pages([page(1, "LoRA\nLow-rank adapters\nCheap fine-tuning", kind="slide")])
    assert p.text == "LoRA\nLow-rank adapters\nCheap fine-tuning"


def test_unicode_is_normalised_and_empty_pages_dropped():
    cleaned = clean_pages([page(1, "eﬃcient search​"), page(2, "  \n\n 7 ")])
    assert [(p.page, p.text) for p in cleaned] == [(1, "efficient search")]


def test_page_with_only_an_image_is_kept():
    chart_only = PageRecord("notes.pdf", 2, "", "page", image_paths=("chart.png",))
    kept = clean_pages([page(1, "Some text."), chart_only, page(3, "")])
    assert [p.page for p in kept] == [1, 2]  # the empty page with no image is dropped
    assert kept[1].image_paths == ("chart.png",)
