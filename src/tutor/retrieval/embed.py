"""Turn text and images into embedding vectors with sentence-transformers (plan step 4).

Two multimodal models, so one question can match a text chunk or a figure:
  gemma2   embeddinggemma-2      740M params, reads 8,192 tokens (Google, open weights)
  qwen3vl  Qwen3-VL-Embedding-2B 2B params, reads 32K tokens     (Alibaba, open weights)

One small text-only model, used as the retrieval baseline:
  minilm   all-MiniLM-L6-v2      22M params, reads 256 tokens    (sentence-transformers)
"""
from functools import lru_cache

import numpy as np

# Each model is trained to embed questions and passages slightly differently.
# query / doc hold the extrai lo arguments passed to encode(): Gemma 2 has named
# prompts built in, Qwen3-VL takes an instruction string (passages use its default).
MODELS = {
    "gemma2": dict(name="google/embeddinggemma-2",
                   query=dict(prompt_name="SearchQuery"), doc=dict(prompt_name="Document"),
                   max_tokens=8192,  # the model reports no limit of its own
                   skip_audio=True),  # text + image only: don't load the 300M audio encoder
    "qwen3vl": dict(name="Qwen/Qwen3-VL-Embedding-2B",
                    query=dict(prompt="Retrieve study material that answers the question."),
                    doc=dict(batch_size=8),  # small batches and half precision fit in 16 GB of memory
                    half=True),
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2", query=dict(), doc=dict(), images=False),
}
DEFAULT_MODEL = "gemma2"


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
        self.model = _load(self.model_name, spec.get("skip_audio", False), spec.get("half", False))

    @property
    def max_tokens(self) -> int:
        """Tokens the model reads per text; anything longer is silently cut off."""
        return self._max_tokens or self.model.max_seq_length

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Embed passages (chunks)."""
        return self._encode(texts, self.doc_args)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question."""
        return self._encode([text], self.query_args)[0]

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
def _load(model_name: str, skip_audio: bool, half: bool = False):
    # Imported here so modules that only chunk don't pay the slow torch import.
    from sentence_transformers import SentenceTransformer
    config = {"audio_config": None} if skip_audio else {}
    model_kwargs = {"torch_dtype": "float16"} if half else {}
    return SentenceTransformer(model_name, config_kwargs=config, model_kwargs=model_kwargs)
