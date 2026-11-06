"""Ask the tutor a question. The same as running src/tutor/pipeline.py.

    .venv/bin/python eval/ask.py "What is cloud computing, and which key problems does it solve?"
    .venv/bin/python eval/ask.py          # keeps asking until you press Enter on an empty line
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tutor.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
