from sentence_transformers.base.modality import is_audio_url_or_path, is_image_url_or_path, is_video_url_or_path

from tutor.retrieval.embed import _as_text


def test_media_links_stay_text():
    for link in ("https://www.youtube.com/watch?v=VKFKypbvoUI", "https://example.com/figure.png",
                 "https://example.com/clip.mp4", "https://example.com/talk.mp3", "data:image/png;base64,AAAA"):
        text = _as_text(link)
        assert not (is_image_url_or_path(text) or is_video_url_or_path(text) or is_audio_url_or_path(text))


def test_text_starting_with_a_link_gets_a_leading_space():
    assert _as_text("https://example.com/talk and some words").startswith(" ")


def test_ordinary_text_is_unchanged():
    sentence = "Containers share the host operating system."
    assert _as_text(sentence) == sentence
