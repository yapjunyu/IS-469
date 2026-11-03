"""Cached LLM calls through Groq (plan step 9: save every AI response).

Every prompt and its reply is saved to disk, keyed by a hash of (model, temperature,
prompt). Asking the same thing again reads the saved reply instead of calling the
API, so reruns are free, identical, and don't hit Groq's rate limits.

Needs the GROQ_API_KEY environment variable only when a reply is not cached yet.
"""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data" / "llm_cache"


class LLM:
    """Callable: llm("prompt") -> reply text."""

    def __init__(self, model_id: str, temperature: float = 0.0,
                 cache_dir: str | Path = DEFAULT_CACHE, max_retries: int = 8,
                 reasoning_effort: str | None = None):
        self.model_id = model_id
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort  # "low" | "medium" | "high" for reasoning models
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self._client = None

    def __call__(self, prompt: str) -> str:
        path = self._cache_path(prompt)
        if path.exists():
            return json.loads(path.read_text())["response"]

        start = time.perf_counter()
        reply = self._call_api(prompt)
        record = {
            "model": self.model_id,
            "temperature": self.temperature,
            "prompt": prompt,
            "response": reply.choices[0].message.content,
            "input_tokens": reply.usage.prompt_tokens,
            "output_tokens": reply.usage.completion_tokens,
            "latency_s": round(time.perf_counter() - start, 3),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
        return record["response"]

    def _cache_path(self, prompt: str) -> Path:
        key = json.dumps([self.model_id, self.temperature, self.reasoning_effort, prompt])
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def _call_api(self, prompt: str):
        import groq

        if self._client is None:
            self._client = groq.Groq()  # reads GROQ_API_KEY
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        for attempt in range(self.max_retries):
            try:
                return self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    **extra,
                )
            except groq.RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                # Groq says how long to wait; otherwise back off 5s, 10s, 20s, ...
                wait = e.response.headers.get("retry-after")
                time.sleep(float(wait) + 1 if wait else 2 ** attempt * 5)
