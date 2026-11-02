"""Turn text into embedding vectors with sentence-transformers (plan step 4).

Three models to compare, from the course-lab baseline to a large modern model:
  minilm  all-MiniLM-L6-v2      22M params, reads 256 tokens   (course labs)
  gemma   embeddinggemma-300m   308M params, reads 2,048 tokens (Google, on-device)
  qwen3   Qwen3-Embedding-0.6B  600M params, reads 32K tokens   (slide 16)
"""
from functools import lru_cache

import numpy as np

# Gemma and Qwen3 are trained to embed questions and passages slightly differently,
# using named prompts built into each model. MiniLM uses none.
MODELS = {
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2", query_prompt=None, doc_prompt=None),
    "gemma": dict(name="google/embeddinggemma-300m", query_prompt="query", doc_prompt="document"),
    "qwen3": dict(name="Qwen/Qwen3-Embedding-0.6B", query_prompt="query", doc_prompt=None),
}


class Embedder:
    """embedder(texts) embeds passages; embedder.embed_query(text) embeds a question.

    Vectors are normalised to length 1, so a dot product between two of them is
    their cosine similarity.
    """

    def __init__(self, key: str = "minilm"):
        spec = MODELS[key]
        self.key = key
        self.model_name = spec["name"]
        self.query_prompt = spec["query_prompt"]
        self.doc_prompt = spec["doc_prompt"]
        self.model = _load(self.model_name)

    @property
    def max_tokens(self) -> int:
        """Tokens the model reads per text; anything longer is silently cut off."""
        return self.model.max_seq_length

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Embed passages (chunks, or sentences for semantic chunking)."""
        return self._encode(texts, self.doc_prompt)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question."""
        return self._encode([text], self.query_prompt)[0]

    def _encode(self, texts, prompt_name) -> np.ndarray:
        extra = {"prompt_name": prompt_name} if prompt_name else {}
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False, **extra)


@lru_cache(maxsize=None)
def _load(model_name: str):
    # Imported here so modules that only chunk don't pay the slow torch import.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)




