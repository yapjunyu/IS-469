"""s1_llm.chat routes by model name to a fake OpenAI or Groq client, and adapts GPT-5 requests."""
from types import SimpleNamespace

from agent import s1_llm


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
    clients = {"openai": FakeClient("from openai"), "groq": FakeClient("from groq")}
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


def test_gpt5_request_has_no_temperature_and_uses_max_completion_tokens(monkeypatch):
    clients = fakes(monkeypatch)
    s1_llm.chat([{"role": "user", "content": "hi"}], "gpt-5.4-mini-2026-03-17", max_tokens=500)
    s1_llm.chat([{"role": "user", "content": "hi"}], "openai/gpt-oss-120b", max_tokens=500)
    gpt5, llama = clients["openai"].requests[0], clients["groq"].requests[0]
    assert "temperature" not in gpt5 and "max_tokens" not in gpt5 and gpt5["max_completion_tokens"] == 500
    assert llama["temperature"] == 0 and llama["max_tokens"] == 500


def test_setting_the_tutor_model_switches_the_passage_calls_too(monkeypatch):
    from agent import s2_tools
    from agent.tests.test_loop import tool_ctx
    sent = []
    monkeypatch.setattr(s1_llm, "chat", lambda messages, model, **kw: sent.append(model) or
                        {"message": {"role": "assistant", "content": "{}"}, "usage": {"in": 1, "out": 1}, "cached": False})
    monkeypatch.setattr(s2_tools, "MODEL", "other-model")
    s2_tools.ask(tool_ctx(), "quiz", "prompt", [], "text", "{")
    assert sent == ["other-model"]
