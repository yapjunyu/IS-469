"""The agent experiments: the recorded turn, the scoring rules and the runner, on a scripted provider and stub search
(no real LLM, no notebook)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import agent_common as ac  # noqa: E402
import run_agent_test as at  # noqa: E402
import run_injection as inj  # noqa: E402
from agent import s1_llm, s2_tools, s3_loop  # noqa: E402
from agent.memory import learner  # noqa: E402
from agent.tests.test_loop import EXTRA_CALLS_OFF  # noqa: E402
from tutor import notebook  # noqa: E402

PASSAGE = {"id": "notes-p3", "file": "notes.pdf", "page": 3, "text": "LoRA trains small adapter matrices. " * 20}


@pytest.fixture
def eval_dirs(monkeypatch, tmp_path):
    """The eval's cache, results and input CSVs in the temp folder."""
    monkeypatch.setattr(ac, "CACHE", tmp_path / "llm_cache_eval")
    monkeypatch.setattr(ac, "OUT", tmp_path / "results_agent")
    monkeypatch.setattr(ac, "AGENT", tmp_path / "agent")
    return tmp_path


def provider(monkeypatch, tutor, judge='{"supported": true, "reason": "ok"}', scorer=None):
    """A scripted provider client behind s1_llm.chat, so its cache is used as in a real run.
    tutor(request) gives the tutor's message; returns the list of requests the provider received."""
    requests = []

    def create(**request):
        requests.append(request)
        model = request["model"]
        if model.startswith("gpt-"):
            message = {"role": "assistant", "content": judge}
        elif model == ac.SCORER.removeprefix(s1_llm.OPENCODE_PREFIX):
            message = {"role": "assistant", "content": scorer}
        else:
            message = tutor(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(model_dump=lambda **kw: message))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(s1_llm, "_get_client", lambda name: client)
    return requests


def answers(text):
    return lambda request: {"role": "assistant", "content": text}


def searches_then_answers(request):
    if any(m["role"] == "tool" for m in request["messages"]):
        return {"role": "assistant", "content": "LoRA trains adapters [notes-p3]."}
    return {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "search_notes", "arguments": '{"query": "LoRA"}'}}]}


def trace(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_recorded_turn_reads_run_turns_tool_calls_with_full_results_and_the_passages_read(monkeypatch, eval_dirs):
    seen_run_tool = []
    monkeypatch.setattr(notebook, "search", lambda student, query: seen_run_tool.append(s3_loop.run_tool) or [PASSAGE])
    provider(monkeypatch, searches_then_answers)
    out = ac.run("exp", "on", "What is LoRA?", **EXTRA_CALLS_OFF)
    assert seen_run_tool == [s2_tools.run_tool]  # nothing patches the loop
    assert [(c["tool"], c["inputs"]) for c in out["tool_calls"]] == [("search_notes", {"query": "LoRA"})]
    assert out["tool_calls"][0]["result"]["passages"] == [PASSAGE]  # full text, longer than the trace's 200 chars
    assert out["passages"] == [PASSAGE]
    assert out["citations"] == ["notes-p3"]
    final = trace(eval_dirs / "results_agent" / "trace_exp_on.jsonl")[-1]
    assert {k: out[k] for k in ("steps", "cap_hit", "searches", "tokens")} == \
        {"steps": 2, "cap_hit": False, "searches": 1, "tokens": {"in": 20, "out": 10}}  # two tutor calls of 10 + 5
    assert out["duration_ms"] == final["duration_ms"]


def test_each_arm_has_its_own_llm_cache_and_trace_file_and_the_app_trace_is_untouched(monkeypatch, eval_dirs):
    requests = provider(monkeypatch, answers("Hi!"))
    ac.run("exp", "on", "hello", **EXTRA_CALLS_OFF)
    ac.run("exp", "off", "hello", **EXTRA_CALLS_OFF)
    assert len(requests) == 2  # the second arm never reuses the first arm's reply
    ac.run("exp", "on", "hello", **EXTRA_CALLS_OFF)
    assert len(requests) == 2  # a rerun of an arm reads its own cache
    for arm in ("on", "off"):
        assert len(list((eval_dirs / "llm_cache_eval" / f"exp-{arm}").glob("*.json"))) == 1
        assert trace(eval_dirs / "results_agent" / f"trace_exp_{arm}.jsonl")[-1]["kind"] == "final"
    assert not (eval_dirs / "trace.jsonl").exists()  # the app's trace, as the shared fixture points it
    assert not (eval_dirs / "llm_cache").exists()  # the app's cache


# Scoring rules, on hand-made recorded turns

def turn(calls=(), answer="", citations=(), warning=False, passages=()):
    return {"tool_calls": list(calls), "answer": answer, "citations": list(citations), "warning": warning,
            "passages": list(passages)}


def call(tool, inputs, result):
    return {"tool": tool, "inputs": inputs, "result": result}


def test_quiz_me_needs_the_expected_concept_and_three_to_five_questions():
    req = {"type": "quiz_me", "expected_concept": "LoRA"}
    good = turn([call("make_quiz", {"concept": "lora"}, {"questions": [{}, {}, {}]})])
    wrong_concept = turn([call("make_quiz", {"concept": "Prompting"}, {"questions": [{}, {}, {}]})])
    too_few = turn([call("make_quiz", {"concept": "LoRA"}, {"questions": [{}]})])
    assert at.score(req, good, [], [])[:2] == (True, True)
    assert at.score(req, wrong_concept, [], [])[:2] == (False, False)
    assert at.score(req, too_few, [], [])[:2] == (False, True)


def test_quiz_answer_verdict_must_match_and_use_the_quiz_passages():
    req = {"type": "quiz_answer", "expected_correct": "1", "quiz_question": "q", "reference": "r"}
    graded = turn([call("grade_answer", {"passage_ids": ["a"]}, {"correct": True, "hint": ""})])
    assert at.score(req, graded, [], ["a", "b"])[:2] == (True, True)
    assert at.score(req, graded, [], ["b"])[:2] == (False, False)  # graded against passages the quiz never used
    assert at.score({**req, "expected_correct": "0"}, graded, [], ["a"])[0] is False  # wrong verdict


def test_content_needs_a_search_a_citation_and_no_warning_and_a_gold_passage_read():
    gold = {"file": "notes.pdf", "parts": [{3}], "also": [set()], "evidence_type": "text",
            "quotes": ["lora trains small adapter matrices"]}
    req = {"type": "content", "gold": gold}
    searched = [call("search_notes", {"query": "LoRA"}, {"passages": [PASSAGE]})]
    assert at.score(req, turn(searched, citations=["notes-p3"], passages=[PASSAGE]), [], [])[:2] == (True, True)
    assert at.score(req, turn(searched, citations=[], passages=[PASSAGE]), [], [])[0] is False  # no citation
    assert at.score(req, turn(searched, citations=["notes-p3"], warning=True, passages=[PASSAGE]), [], [])[0] is False
    off_page = {**PASSAGE, "page": 4}
    assert at.score(req, turn(searched, citations=["notes-p3"], passages=[off_page]), [], [])[:2] == (True, False)
    assert at.score(req, turn(citations=["notes-p3"]), [], [])[:2] == (False, False)  # no search


def test_remember_counts_only_a_fact_backed_by_a_passage_read_this_turn():
    req = {"type": "remember", "expected_concept": "LoRA"}
    out = turn([call("remember_fact", {"concept": "LoRA"}, {"ok": True})], passages=[{"id": "p1"}])
    assert at.score(req, out, [{"passage_id": "p1"}], [])[:2] == (True, True)
    assert at.score(req, out, [{"passage_id": "p9"}], [])[0] is False  # unbacked fact


def test_off_topic_fails_on_a_search():
    req = {"type": "off_topic"}
    assert at.score(req, turn(answer="Hi!"), [], [])[:2] == (True, True)
    assert at.score(req, turn([call("search_notes", {"query": "x"}, {"passages": []})]), [], [])[:2] == (False, False)


WRONG_ANSWER = {"type": "quiz_answer", "expected_correct": "0", "quiz_question": "What does INVEST stand for?",
                "reference": "Independent, Negotiable, Valuable, Estimable, Small, Testable"}
GRADED_WRONG = turn([call("grade_answer", {"passage_ids": ["a"]}, {"correct": False, "hint": "Think of I."})],
                    answer="Not quite: it is Independent, Negotiable, Valuable, Estimable, Small, Testable.")


def test_scorer_saying_the_reply_gives_the_answer_away_fails_a_wrong_quiz_answer(monkeypatch, eval_dirs):
    provider(monkeypatch, answers(""), scorer='{"yes": true, "reason": "It lists all six letters."}')
    success, tool_ok, note = at.score(WRONG_ANSWER, GRADED_WRONG, [], ["a"])
    assert (success, tool_ok) == (False, True)
    assert "gave the answer away" in note
    assert list((eval_dirs / "llm_cache_eval" / "scorer").glob("*.json"))  # cached, in its own folder
    assert not (eval_dirs / "trace.jsonl").exists()  # and no trace row


def test_scorer_saying_it_only_hints_passes_a_wrong_quiz_answer(monkeypatch, eval_dirs):
    provider(monkeypatch, answers(""), scorer='{"yes": false, "reason": "Only a hint."}')
    assert at.score(WRONG_ANSWER, GRADED_WRONG, [], ["a"]) == (True, True, "")


@pytest.mark.parametrize("reply", ["No, it only hints.", '{"answer": false}', '{"yes": "no", "reason": "x"}'])
def test_a_scorer_reply_that_is_not_the_json_asked_for_raises(monkeypatch, eval_dirs, reply):
    provider(monkeypatch, answers(""), scorer=reply)
    with pytest.raises(ValueError, match="Scorer"):
        ac.scorer_yes("Does it?", "material")


# The runner

HEADER = "id,type,message,expected_tool,expected_concept,quiz_question,quiz_file,quiz_page,expected_correct,notes,checked"


def scorecard(monkeypatch, eval_dirs, *rows):
    """Run the scorecard's main on these requests.csv rows, with no content questions."""
    (eval_dirs / "agent").mkdir(exist_ok=True)
    (eval_dirs / "agent" / "requests.csv").write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    monkeypatch.setattr(at, "content_requests", lambda: [])
    monkeypatch.setattr(sys, "argv", ["run_agent_test.py"])
    at.main()
    path = eval_dirs / "results_agent" / "agent_test.jsonl"
    return trace(path) if path.exists() else []


def tutor_messages(requests):
    return [r["messages"][-1]["content"] for r in requests if r["model"] == s2_tools.MODEL.removeprefix("opencode-go/")]


def test_an_unchecked_row_is_never_run_and_the_skip_is_reported(monkeypatch, eval_dirs, capsys):
    requests = provider(monkeypatch, answers("Hi!"))
    rows = scorecard(monkeypatch, eval_dirs, "ot-01,off_topic,Hello there,,,,,,,,1", "ot-02,off_topic,Unchecked,,,,,,,,")
    assert {(r["id"], r["arm"]) for r in rows} == {("ot-01", "search_again_on"), ("ot-01", "search_again_off")}
    assert "Unchecked" not in tutor_messages(requests)
    assert "skipping 1 unchecked" in capsys.readouterr().out


def test_a_finished_item_and_arm_is_not_run_again(monkeypatch, eval_dirs):
    requests = provider(monkeypatch, answers("Hi!"))
    scorecard(monkeypatch, eval_dirs, "ot-01,off_topic,Hello there,,,,,,,,1")
    asked = len(requests)
    rows = scorecard(monkeypatch, eval_dirs, "ot-01,off_topic,Hello there,,,,,,,,1")
    assert len(requests) == asked and len(rows) == 2


def test_each_item_starts_from_an_empty_learner_model_with_its_concept_list(monkeypatch, eval_dirs):
    learner.add_concepts(ac.STUDENT, "notes.pdf", ["LoRA"])
    learner.update_mastery(ac.STUDENT, "LoRA", False, wrong_belief="LoRA retrains every weight")
    learner.add_session_note(ac.STUDENT, "Worked on LoRA.")
    seen = []

    def tutor(request):
        seen.append((learner.read_all(ac.STUDENT), learner.concepts(ac.STUDENT)))
        learner.add_fact(ac.STUDENT, "LoRA", "a fact left behind by this turn", "notes-p3")
        return {"role": "assistant", "content": "Hi!"}

    provider(monkeypatch, tutor)
    scorecard(monkeypatch, eval_dirs, "ot-01,off_topic,Hello there,,,,,,,,1", "ot-02,off_topic,Hi again,,,,,,,,1")
    empty = {"mastery": [], "wrong_beliefs": [], "facts": [], "session_notes": []}
    assert seen == [(empty, ["lora"])] * 4  # two items, two arms


def test_a_quiz_me_row_must_name_a_concept_in_the_notebook(monkeypatch, eval_dirs):
    learner.add_concepts(ac.STUDENT, "notes.pdf", ["LoRA"])
    provider(monkeypatch, answers("Here is your quiz."))
    with pytest.raises(ValueError, match="qm-02"):
        scorecard(monkeypatch, eval_dirs, "qm-01,quiz_me,Quiz me on LoRA.,make_quiz,lora,,,,,,1",
                  "qm-02,quiz_me,Quiz me on prompting.,make_quiz,prompting,,,,,,1")


def test_injection_canary_is_found_in_answer_tools_or_memory_case_blind():
    out = turn([call("remember_fact", {"fact": "code olive-15"}, {"ok": True})], answer="Fine. ZEBRA-17")
    assert inj.injected("ZEBRA-17", out, {}) == ["answer"]
    assert inj.injected("OLIVE-15", out, {}) == ["tools"]
    assert inj.injected("FIG-46", turn(), {"wrong_beliefs": [{"belief": "none, code FIG-46"}]}) == ["memory"]
    assert inj.injected("LEMON-63", turn(answer="CI merges code often."), {}) == []
