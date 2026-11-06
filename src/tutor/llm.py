"""
Calls a model through OpenRouter's OpenAI-compatible API & cache reply on disk
Needs OPENROUTER_API_KEY (environment or .env) only when a reply is not cached yet.
"""

import base64
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data" / "llm_cache"
BASE_URL = "https://openrouter.ai/api/v1"


def strip_thinking(text: str | None) -> str:
    """Gemma 4 can write its reasoning into the reply as <thought>...</thought>
    before the answer, and the thinking can't be switched off. Keep only the answer."""
    text = re.sub(r"<thought>.*?</thought>", "", text or "", flags=re.DOTALL)
    return re.sub(r"<thought>.*", "", text, flags=re.DOTALL).strip()  # cut off before it closed


class LLM:
    """Callable: llm("prompt") -> reply text, or llm("prompt", images=["slide.png"]) for a vision model."""

    def __init__(self, model_id: str, temperature: float = 0.0,
                 cache_dir: str | Path = DEFAULT_CACHE, max_retries: int = 8,
                 reasoning_effort: str | None = None):
        self.model_id = model_id
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort  # "low" | "medium" | "high" for reasoning models
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self._client = None

    def __call__(self, prompt: str, images: list[str | Path] | None = None) -> str:
        images = list(images or [])
        path = self._cache_path(prompt, images)
        if path.exists():
            return json.loads(path.read_text())["response"]

        start = time.perf_counter()
        reply = self._call_api(prompt, images)
        record = {
            "model": self.model_id,
            "temperature": self.temperature,
            "prompt": prompt,
            "images": [str(i) for i in images],
            "response": strip_thinking(reply.choices[0].message.content),
            "input_tokens": getattr(reply.usage, "prompt_tokens", None),  # some replies carry no usage
            "output_tokens": getattr(reply.usage, "completion_tokens", None),
            "latency_s": round(time.perf_counter() - start, 3),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
        return record["response"]

    def _cache_path(self, prompt: str, images: list[str | Path]) -> Path:
        # images are keyed by their bytes, so a re-rendered slide gets a fresh answer
        pictures = [hashlib.sha256(Path(i).read_bytes()).hexdigest() for i in images]
        key = json.dumps([self.model_id, self.temperature, self.reasoning_effort, prompt, pictures])
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def _message(self, prompt: str, images: list[str | Path]) -> dict:
        """Plain text, or text followed by one image_url part per picture."""
        if not images:
            return {"role": "user", "content": prompt}
        parts = [{"type": "text", "text": prompt}]
        for image in images:
            b64 = base64.b64encode(Path(image).read_bytes()).decode()
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        return {"role": "user", "content": parts}

    def _call_api(self, prompt: str, images: list[str | Path]):
        import os

        import openai
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")  # lets OPENROUTER_API_KEY live in .env
        if self._client is None:
            self._client = openai.OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], max_retries=0)
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        for attempt in range(self.max_retries):
            try:
                return self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[self._message(prompt, images)],
                    temperature=self.temperature,
                    **extra,
                )
            # 429 rate limit; 500/503 the host is busy or failed (the same request often works next time)
            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                response = getattr(e, "response", None)
                wait = response.headers.get("retry-after") if response is not None else None
                time.sleep(float(wait) + 1 if wait else min(2 ** attempt * 2, 30))  # 2s, 4s, 8s, ... at most 30s
