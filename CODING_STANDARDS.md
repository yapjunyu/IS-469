# Coding standards

Rules a reviewer applies to every diff. Each is a judgement call. Mechanical rules live in tests instead (`agent/tests/test_seams.py`, `agent/tests/conftest.py`), so they fail on their own.

## Known limits

Mark every deliberate shortcut with a one-line comment: `# shortcut: <the limit>, <when to upgrade>`. A hard-coded setting gets one until the team's config file exists.

## LLM calls

`agent/s1_llm.py` is the only module that talks to a provider. Every other LLM call goes through `s4_trace.traced_chat(ctx, purpose, ...)`, which writes the call's `llm` trace row and adds its tokens to the turn. A new LLM caller (quiz, grading, judge, search-again) picks a short `purpose` name and passes the turn's `ctx`.

## Trace format

The `agent/s4_trace.py` module docstring is the single source of truth for trace rows. A diff that changes a row also updates that docstring and the step 14 example in `docs/build_and_test_steps.md`.

## Turn state

Per-turn state lives on `TurnContext` in `agent/s2_tools.py`: who and where (student, turn, step), running totals (tokens, searches, retrieved passage ids) and the switches. Anything that must count toward the turn is added there, so work done inside tools reaches the `final` row.

## Tests

- Agent tests are unit tests: the network, real LLMs and real retrieval stay out.
- Drive a whole turn through `run_turn` with a scripted fake of `s1_llm.chat` (see the helpers in `agent/tests/test_loop.py`). Assert on what `run_turn` returns, the trace rows and the learner model.
- A turn test switches on only the checks it tests, through the shared `turn` helper in `agent/tests/test_loop.py`. A new switch that adds LLM calls goes in its `EXTRA_CALLS_OFF`.
- Assert on external behaviour only: private helpers and prompt wording change freely.
- A tool's own return value (and the prompt it sends its LLM) may be checked by calling it through `run_tool` on a `TurnContext`; `run_turn` is for loop behaviour and trace rows.
- Turn state that no output shows yet (such as the retrieved passage ids) is checked by running one tool through `run_tool` on a `TurnContext`, as `agent/tests/test_memory.py` does.
- When a ticket asks for a rule in a prompt, assert on a stable token the rule names (a marker such as `<document>`), so the sentence can be reworded freely.
- Take expected values from a worked example or the spec (`0.65  # 0.5 + 0.3 * (1 - 0.5)`), so a test can disagree with the code.
- A new path that writes files gets redirected to the temp folder in the shared fixture in `agent/tests/conftest.py`.
