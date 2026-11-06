"""Step 13: search-again (corrective retrieval).

Called from inside search_notes, not from the loop.
After each search, ask the LLM: are these passages enough to answer?
If not, rephrase and search again, up to max_searches. Log each attempt to the trace.
With the switch off, return the first search result (fixed-pipeline baseline).
"""


def search_with_retries(query, search_fn, max_searches=3):
    """Return the final list of passages ({id, file, page, text})."""
    raise NotImplementedError
