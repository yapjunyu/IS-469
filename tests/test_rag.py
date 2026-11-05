from tutor.rag import NOT_FOUND, Answer, answer, build_prompt
from tutor.retrieval.types import Passage


def passage(n, text, image_path=None):
    return Passage(id=f"h:{n}", file="deck.pptx", page=n, page_end=n, text=text, kind="slide",
                   image_path=image_path, score=0.5)


class FakeIndex:
    def __init__(self, *passages):
        self.passages = list(passages)

    def search(self, query, k=5):
        return self.passages[:k]


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt, images=None):
        self.prompts.append(prompt)
        self.images = images
        return self.reply


def test_prompt_numbers_the_passages_and_includes_the_question():
    prompt = build_prompt("What is LoRA?", [passage(2, "LoRA adds small matrices."), passage(5, "It freezes weights.")])
    assert "[1] (deck.pptx, slide 2)\nLoRA adds small matrices." in prompt
    assert "[2] (deck.pptx, slide 5)\nIt freezes weights." in prompt
    assert prompt.rstrip().endswith("Answer:") and "Question: What is LoRA?" in prompt


def test_answer_and_its_citations():
    llm = FakeLLM("LoRA adds small matrices [1] and freezes the rest [2]. Again [1]. Bogus [9].")
    result = answer("What is LoRA?", FakeIndex(passage(2, "a"), passage(5, "b")), llm)
    assert result.citations() == ["deck.pptx, slide 2", "deck.pptx, slide 5"]  # in order, no repeats, [9] ignored
    assert len(llm.prompts) == 1


def test_text_only_search_calls_the_llm_without_images():
    llm = FakeLLM("Answer [1].")
    result = answer("q", FakeIndex(passage(1, ""), passage(2, "text")), llm)  # slide 1 has no text
    assert [p.page for p in result.sources] == [2]
    assert llm.images is None and "Pictures" not in llm.prompts[-1]


def test_pictures_are_attached_numbered_after_the_passages_and_citable():
    llm = FakeLLM("Text says A [1]; the diagram shows B [2].")
    picture = passage(3, "", image_path="fig.png")
    result = answer("q", FakeIndex(passage(2, "text"), picture), llm)
    assert result.figures == [picture] and result.images_sent == 1
    assert llm.images == ["fig.png"]
    assert "[2] (deck.pptx, slide 3)" in llm.prompts[-1] and "Pictures" in llm.prompts[-1]
    assert result.citations() == ["deck.pptx, slide 2", "deck.pptx, slide 3"]


def test_only_the_top_pictures_are_sent_but_all_are_returned():
    llm = FakeLLM("A [1].")
    pictures = [passage(n, "", image_path=f"fig{n}.png") for n in (3, 4, 5)]
    result = answer("q", FakeIndex(passage(2, "text"), *pictures), llm, max_images=2)
    assert llm.images == ["fig3.png", "fig4.png"]
    assert result.figures == pictures and result.images_sent == 2


def test_a_picture_alone_is_enough_to_answer():
    llm = FakeLLM("The diagram shows it [1].")
    result = answer("q", FakeIndex(passage(3, "", image_path="fig.png")), llm)
    assert result.sources == [] and llm.images == ["fig.png"]
    assert result.citations() == ["deck.pptx, slide 3"]


def test_nothing_to_read_means_no_llm_call():
    llm = FakeLLM("should not be used")
    result = answer("q", FakeIndex(), llm)
    assert result == Answer(NOT_FOUND) and llm.prompts == []


def test_citations_in_gpt_oss_style_are_understood():
    result = Answer("A【1】 and B【2†L1-L4】 and C [1].", sources=[passage(2, "a"), passage(5, "b")])
    assert result.citations() == ["deck.pptx, slide 2", "deck.pptx, slide 5"]
