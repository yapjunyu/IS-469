"""Step 13: search-again (corrective retrieval).

Called from inside search_notes, not from the loop, and only with the switch on
(off, search_notes uses the first result: the fixed-pipeline baseline).
After each search, one LLM call (purpose "search_again") says whether the passages are enough to
answer the query. If not, it gives a rephrased query and the search repeats, up to max_searches.
Each check writes its own llm row through the ask function search_notes passes in;
the search function counts each search on the turn.
"""
from agent.prompts import SEARCH_AGAIN_PROMPT

# shortcut: hardcoded, move to config.yaml once the team agrees on it
MAX_SEARCHES = 3


def search_with_retries(ctx, query, search, ask, max_searches=MAX_SEARCHES):
    """search(query) -> passages ({id, file, page, text}); ask is s2_tools.ask, passed in because s2_tools
    imports this module. Return the last search's passages."""
    passages = search(query)
    for _ in range(max_searches - 1):
        verdict = ask(ctx, "search_again", SEARCH_AGAIN_PROMPT, passages, f"Search query: {query}", "{")
        # an unreadable check keeps what we have: a bad reply must not cost extra searches
        if not isinstance(verdict, dict) or verdict.get("enough") is not False:
            break
        rephrased = verdict.get("query")
        if not (isinstance(rephrased, str) and rephrased.strip()):
            break
        query = rephrased
        passages = search(query)
    return passages
