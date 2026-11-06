"""Turn text and images into embedding vectors with sentence-transformers (plan step 4).

Two multimodal models, so one question can match a text chunk or a figure:
  gemma2     embeddinggemma-2    740M params, reads 8,192 tokens (Google, open weights)
  jinaclip2  jina-clip-v2        865M params, reads 8,192 tokens (Jina AI, CC BY-NC 4.0: non-commercial)

jinaclip2 runs custom code from its Hugging Face repo and does not load on transformers 5, so it has
its own environment, .venv-jina (see requirements-jina.txt).

One small text-only model, used as the retrieval baseline:
  minilm   all-MiniLM-L6-v2      22M params, reads 256 tokens    (sentence-transformers)
"""
from functools import lru_cache

import numpy as np

MODELS = {
    "gemma2": dict(name="google/embeddinggemma-2",
                   query=dict(prompt_name="SearchQuery"), doc=dict(prompt_name="Document"),
                   max_tokens=8192,  
                   skip_audio=True),  # text + image only: don't load the 300M audio encoder
    "jinaclip2": dict(name="jinaai/jina-clip-v2",
                      query=dict(prompt_name="retrieval.query"), doc=dict(batch_size=8),
                      max_tokens=8192,
                      load=dict(revision="e10d47f5691d0454a0fb5d13f46f2199b74cb436", trust_remote_code=True)),
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2", query=dict(), doc=dict(), images=False),
}
DEFAULT_MODEL = "gemma2"


def _as_text(text: str) -> str:
    """Keep text as text. sentence-transformers treats a string with no spaces that looks like a media
    link or file (a slide that is only a YouTube URL) as that media, and tries to load it. jinaclip2
    goes further and downloads any string starting with "http". Tokenizers ignore the added space."""
    return " " + text if " " not in text or text.startswith(("data:", "http")) else text


class Embedder:
    """embedder(texts) embeds passages; embedder.embed_query(text) embeds a question;
    embedder.embed_images(paths) embeds picture files.

    Vectors are normalised to length 1, so a dot product between two of them is
    their cosine similarity.
    """
    supports_images = True

    def __init__(self, key: str = DEFAULT_MODEL):
        spec = MODELS[key]
        self.key = key
        self.supports_images = spec.get("images", True)
        self.model_name = spec["name"]
        self.query_args = spec["query"]
        self.doc_args = spec["doc"]
        self._max_tokens = spec.get("max_tokens")
        self.model = _load(key)

    @property
    def max_tokens(self) -> int:
        """Tokens the model reads per text; anything longer is silently cut off."""
        return self._max_tokens or self.model.max_seq_length

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Embed passages (chunks)."""
        return self._encode([_as_text(t) for t in texts], self.doc_args)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question."""
        return self._encode([_as_text(text)], self.query_args)[0]

    def embed_images(self, paths: list[str]) -> np.ndarray:
        """Embed picture files into the same space as the text. Images get no prompt."""
        from PIL import Image
        images = []
        for path in paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        batch = {k: v for k, v in self.doc_args.items() if k == "batch_size"}  # no prompt, but same batch limit
        return self._encode(images, batch)

    def _encode(self, inputs, extra) -> np.ndarray:
        return self.model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False, **extra)


@lru_cache(maxsize=None)
def _load(key: str):
    # Imported here so modules that only chunk don't pay the slow torch import.
    from sentence_transformers import SentenceTransformer
    spec = MODELS[key]
    config = {"audio_config": None} if spec.get("skip_audio") else {}
    return SentenceTransformer(spec["name"], config_kwargs=config, **spec.get("load", {}))
