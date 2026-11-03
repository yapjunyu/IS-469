"""Step 14: JSONL trace, one line per tool call.

Fields: ts, student, turn, step, tool, inputs, result_summary,
duration_ms, tokens {in, out}, ok, error.
"""

import json
import time

from pathlib import Path

def log_call(path, **fields):
    """Append one JSON line to the trace file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('a') as f:
        f.write(json.dumps({
            'ts': time.time(),
            **fields
        }, default=str) + "\n")
