"""Step 14: JSONL trace, one line per tool call.

Fields: ts, student, turn, step, tool, inputs, result_summary,
duration_ms, tokens {in, out}, ok, error.
"""


def log_call(path, **fields):
    """Append one JSON line to the trace file."""
    raise NotImplementedError
