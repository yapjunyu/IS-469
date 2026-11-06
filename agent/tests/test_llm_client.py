"""s1_llm.chat routes by model name to a fake OpenAI, Groq or OpenCode client, and adapts GPT-5 and DeepSeek requests."""
import uuid
from types import SimpleNamespace

import pytest

from agent import ingest, s1_llm
from agent.memory import learner, reflection
from agent.s2_tools import run_tool
from agent.tests.test_loop import SEARCH_CALL, always_search, read_trace, reply, tool_ctx, turn

PASSAGES = [{"id": "p1", "file": "lec3.pdf", "page": 12, "text": "LoRA adds low-rank matrices."}]
REAL_GET_CLIENT = s1_llm._get_client  # taken at import: the shared fixture replaces it for each test


class FakeClient:
    """Records each request and answers in the SDK's response shape."""
    def __init__(self, text):
        self.requests, self.text = [], text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(model_dump=lambda exclude_none: {"role": "assistant", "content": self.text})
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))


def fakes(monkeypatch):
    clients = {"openai": FakeClient("from openai"), "groq": FakeClient("from groq"), "opencode": FakeClient("from opencode")}
    monkeypatch.setattr(s1_llm, "_get_client", lambda provider: clients[provider])
    return clients


def test_model_name_picks_the_client(monkeypatch):
    clients = fakes(monkeypatch)
    judge = s1_llm.chat([{"role": "user", "content": "hi"}], "gpt-5.4-mini-2026-03-17")
    tutor = s1_llm.chat([{"role": "user", "content": "hi"}], "openai/gpt-oss-120b")
    assert judge == {"message": {"role": "assistant", "content": "from openai"},
                     "usage": {"in": 7, "out": 3}, "cached": False}
    assert tutor["message"]["content"] == "from groq"
    assert len(clients["openai"].requests) == len(clients["groq"].requests) == 1


def test_opencode_go_prefix_uses_the_opencode_client_with_the_prefix_stripped(monkeypatch):
    clients = fakes(monkeypatch)
    out = s1_llm.chat([{"role": "user", "content": "hi"}], "opencode-go/deepseek-v4.1-flash")
    assert out["message"]["content"] == "from opencode"
    assert clients["opencode"].requests[0]["model"] == "deepseek-v4.1-flash"
    assert not clients["openai"].requests and not clients["groq"].requests


def test_opencode_client_uses_the_go_endpoint_key_and_its_own_headers(monkeypatch):
    monkeypatch.setattr(s1_llm, "_clients", {})
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    client = REAL_GET_CLIENT("opencode")
    assert str(client.base_url).rstrip("/") == "https://opencode.ai/inference/go/openai/v1"
    assert client.api_key == "test-key"
    assert client.max_retries == 5
    assert client.default_headers["User-Agent"] == "is-469-tutor"
    assert uuid.UUID(client.default_headers["x-opencode-session"]).version == 4  # random, so no personal data
    assert REAL_GET_CLIENT("opencode") is client  # one client per process, so the session header stays stable


def test_missing_opencode_key_raises_instead_of_sending_the_openai_key(monkeypatch):
    monkeypatch.setattr(s1_llm, "_clients", {})
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")  # the SDK reads this one when api_key is None
    with pytest.raises(KeyError):
        REAL_GET_CLIENT("opencode")


def test_deepseek_request_turns_thinking_off_and_others_do_not(monkeypatch):
    clients = fakes(monkeypatch)
    s1_llm.chat([{"role": "user", "content": "hi"}], "opencode-go/deepseek-v4.1-flash")
    s1_llm.chat([{"role": "user", "content": "hi"}], "gpt-5.4-mini-2026-03-17")
    s1_llm.chat([{"role": "user", "content": "hi"}], "openai/gpt-oss-120b")
    assert clients["opencode"].requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "extra_body" not in clients["openai"].requests[0] and "extra_body" not in clients["groq"].requests[0]


def test_trace_rows_name_the_tutor_model_and_judge_stays_on_openai(monkeypatch, tmp_data):
    def chat(messages, model, tools=None, tool_choice="auto", **kw):
        if tools and tool_choice != "none" and not any(m["role"] == "tool" for m in messages):
            return reply(SEARCH_CALL)
        return reply({"role": "assistant", "content": '{"supported": false, "enough": true, "reason": "why", "note": "n"}'})
    monkeypatch.setattr(s1_llm, "chat", chat)
    search, get_passages = (lambda s, q: PASSAGES), (lambda s, ids: PASSAGES)
    turn("What is LoRA?", "alice", 1, search_fn=search, passages_fn=get_passages, faithfulness=True, search_again=True)  # tutor, search_again, judge, regenerate
    monkeypatch.setattr(s1_llm, "chat", always_search)
    turn("loop", "alice", 2, step_cap=1, search_fn=search)  # forced_answer
    monkeypatch.setattr(s1_llm, "chat", chat)
    learner.add_concepts("alice", "lec3.pdf", ["lora"])
    ctx = tool_ctx(search_fn=search, passages_fn=get_passages)
    run_tool("make_quiz", '{"concept": "lora"}', ctx)
    run_tool("grade_answer", '{"question": "q", "student_answer": "a", "passage_ids": ["p1"]}', ctx)
    ingest.extract_concepts("alice", "lec3.pdf", "text")
    reflection.end_session("alice", [{"role": "user", "content": "hi"}])
    rows = [r for r in read_trace(tmp_data) if r["kind"] == "llm"]
    assert {r["purpose"] for r in rows} == {"tutor", "forced_answer", "regenerate", "quiz", "grading", "search_again",
                                            "concepts", "reflection", "judge"}
    wrong = [r for r in rows if r["model"] != ("gpt-5.4-mini-2026-03-17" if r["purpose"] == "judge"
                                               else "opencode-go/deepseek-v4.1-flash")]
    assert wrong == []


def test_gpt5_request_has_no_temperature_and_uses_max_completion_tokens(monkeypatch):
    clients = fakes(monkeypatch)
    s1_llm.chat([{"role": "user", "content": "hi"}], "gpt-5.4-mini-2026-03-17", max_tokens=500)
    s1_llm.chat([{"role": "user", "content": "hi"}], "openai/gpt-oss-120b", max_tokens=500)
    gpt5, llama = clients["openai"].requests[0], clients["groq"].requests[0]
    assert "temperature" not in gpt5 and "max_tokens" not in gpt5 and gpt5["max_completion_tokens"] == 500
    assert llama["temperature"] == 0 and llama["max_tokens"] == 500


def test_setting_the_tutor_model_switches_the_passage_calls_too(monkeypatch):
    from agent import s2_tools
    sent = []
    monkeypatch.setattr(s1_llm, "chat", lambda messages, model, **kw: sent.append(model) or
                        {"message": {"role": "assistant", "content": "{}"}, "usage": {"in": 1, "out": 1}, "cached": False})
    monkeypatch.setattr(s2_tools, "MODEL", "other-model")
    s2_tools.ask(tool_ctx(), "quiz", "prompt", [], "text", "{")
    assert sent == ["other-model"]
