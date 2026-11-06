"""Step 9: the only file that talks to an LLM provider.

- Route by model name: OpenAI model IDs (gpt-...) go to OpenAI, all others to Groq.
- GPT-5 models reject a custom temperature and take max_completion_tokens instead of
  max_tokens, so their request is adapted; reruns stay identical through the cache.
- Cache every successful response to disk, keyed by a hash of the full request.
- On 429: both SDKs' own retries (max_retries) honour `retry-after`.
- On 400 `tool_use_failed`: retry; never append the bad call to history.
- Return plain dicts (not SDK objects) so cache and trace can JSON them.
"""
import hashlib
import json
from pathlib import Path

import groq
import openai
from dotenv import load_dotenv

load_dotenv()
# same gitignored folder as src/tutor/llm.py; keys hash the full request, so no collisions
# shortcut: path copied from src/tutor/llm.py, import it once `tutor` is installed as a package
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "llm_cache"
_clients = {}

# Create each provider's connection the 1st time, then reuse it (keys from GROQ_API_KEY / OPENAI_API_KEY in .env)
def _get_client(provider):
    if provider not in _clients:
        _clients[provider] = (openai.OpenAI if provider == "openai" else groq.Groq)(max_retries=5)
    return _clients[provider]

# turns the whole request into a short, unique fingerprint that serves as the file name for its saved answer, so identical requests always find the same file.
def _cache_key(request):
    text = json.dumps(request, sort_keys=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Calls the provider; retries Groq's 400 tool_use_failed (OpenAI has no such error, its 400s raise)
def _call_with_retry(provider, request, attempts=3):
    for attempt in range(attempts):
        try:
            return _get_client(provider).chat.completions.create(**request)
        except groq.BadRequestError as e:
            if "tool_use_failed" not in str(e) or attempt == attempts - 1:
                raise

# 1. Build request: model, messages, temp, len limit is compulsory. Tool added only if needed
# 2. Check cache, if question same as one in cached, will return the answer w/o calling Groq
# 3. Call Groq and dump the results in hashmap
# 4. Write result to cache file
def chat(messages, model,
         tools=None, tool_choice="auto",
         temperature=0, max_tokens=1024):
    """Send one chat request, or read it from the cache.

    Returns {"message": dict, "usage": {"in": int, "out": int}, "cached": bool}.
    """

    request = {"model": model, "messages" : messages,
               "temperature": temperature, "max_tokens": max_tokens}

    if tools:
        request["tools"] = tools
        request["tool_choice"] = tool_choice

    # shortcut: only gpt- IDs route to OpenAI, add o-series prefixes if the team ever uses them
    provider = "openai" if model.startswith("gpt-") else "groq"
    if model.startswith("gpt-5"):
        del request["temperature"]
        request["max_completion_tokens"] = request.pop("max_tokens")

    path = CACHE_DIR / f"{_cache_key(request)}.json"

    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        result["cached"] = True
        return result

    response = _call_with_retry(provider, request)
    result = {
        "message": response.choices[0].message.model_dump(exclude_none=True),
        "usage": {
            "in": response.usage.prompt_tokens,
            "out": response.usage.completion_tokens,
        }
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["cached"] = False
    return result