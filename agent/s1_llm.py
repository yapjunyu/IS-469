"""Step 9: the only file that talks to Groq.

- Cache every successful response to disk, keyed by a hash of
  (model, messages, tools, tool_choice, temperature, max_tokens).
- On 429: the Groq SDK's own retries (max_retries) honour `retry-after`.
- On 400 `tool_use_failed`: retry; never append the bad call to history.
- Return plain dicts (not SDK objects) so cache and trace can JSON them.
"""
import hashlib
import json
from pathlib import Path

import groq
from dotenv import load_dotenv

load_dotenv()
# same gitignored folder as src/tutor/llm.py; keys hash the full request, so no collisions
# shortcut: path copied from src/tutor/llm.py, import it once `tutor` is installed as a package
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "llm_cache"
_client = None

# Create Groq connection 1st time then reuses it
def _get_client():
    global _client
    if not _client:
        _client = groq.Groq(max_retries = 5)
    return _client

# turns the whole request into a short, unique fingerprint that serves as the file name for its saved answer, so identical requests always find the same file.
def _cache_key(request):
    text = json.dumps(request, sort_keys=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Calls Groq & handle error types that doesn't retry
def _call_with_retry(request, attempts=3):
    for attempt in range(attempts):
        try:
            return _get_client().chat.completions.create(**request)
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

    path = CACHE_DIR / f"{_cache_key(request)}.json"

    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        result["cached"] = True
        return result

    response = _call_with_retry(request)
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