"""Unit tests for llm_router.py — config-driven Provider abstraction.

These tests use a temp YAML to avoid coupling to the live config, but
also assert one thing about the live config: deepseek must not appear
anywhere (per memory/feedback_no_deepseek.md).
"""
from __future__ import annotations
import os
from pathlib import Path
import tempfile

import pytest


@pytest.fixture
def temp_yaml_router(monkeypatch, tmp_path):
    """Point llm_router at a temp YAML and force a fresh load."""
    yaml_path = tmp_path / "llm_routes.yaml"
    yaml_path.write_text("""
providers:
  fake_oai:
    type: openai_compat
    base_url_env: FAKE_OAI_BASE
    api_key_env: FAKE_OAI_KEY
  fake_ollama:
    type: ollama
    base_url_env: FAKE_OLLAMA_BASE
    api_key_env: FAKE_OLLAMA_KEY
  fake_gemini:
    type: gemini_embed
    api_key_env: FAKE_GEMINI_KEY

routes:
  simple_chat:    [fake_oai:gpt-1]
  deep_reasoning: [fake_ollama:reasoner-x, fake_oai:gpt-1]

costs:
  "fake_oai:gpt-1": [1.5, 4.5]
  "fake_ollama:reasoner-x": [0.0, 0.0]

embeddings:
  default: fake_gemini:emb-1
  dim: 768
""")
    monkeypatch.setenv("FAKE_OAI_BASE", "http://example.com/v1")
    monkeypatch.setenv("FAKE_OAI_KEY", "k1")
    monkeypatch.setenv("FAKE_OLLAMA_BASE", "http://example.com/ollama")
    monkeypatch.setenv("FAKE_OLLAMA_KEY", "k2")
    monkeypatch.setenv("FAKE_GEMINI_KEY", "k3")

    from quant import llm_router as r
    monkeypatch.setattr(r, "CONFIG_PATH", yaml_path)
    r.reload_config()
    yield r
    r.reload_config()  # reset to live config


# ---- Live config invariants ----


def test_live_config_has_no_deepseek():
    """User reported deepseek hallucinations — must stay out of every chain.

    See memory/feedback_no_deepseek.md.
    """
    from quant import llm_router as r
    r.reload_config()
    routes = r.get_routes()
    flat = [m for chain in routes.values() for m in chain]
    deepseek = [m for m in flat if "deepseek" in m.lower()]
    assert deepseek == [], f"deepseek leaked back into routes: {deepseek}"
    costs = r.get_costs()
    deepseek_cost = [k for k in costs if "deepseek" in k.lower()]
    assert deepseek_cost == [], f"deepseek leaked back into costs: {deepseek_cost}"


def test_live_config_loads_expected_tasks():
    from quant import llm_router as r
    r.reload_config()
    routes = r.get_routes()
    for required in ("simple_chat", "reasoning", "deep_reasoning",
                      "long_context", "review", "vision"):
        assert required in routes, f"task {required} missing from live routes"


def test_live_config_every_route_entry_has_a_provider():
    """Routes can only reference providers that exist."""
    from quant import llm_router as r
    r.reload_config()
    providers = set(r._providers().keys())
    for task, chain in r.get_routes().items():
        for entry in chain:
            p, _ = entry.split(":", 1)
            assert p in providers, f"route {task} references unknown provider {p}"


# ---- YAML loader / Provider construction ----


def test_yaml_loader_builds_three_providers(temp_yaml_router):
    r = temp_yaml_router
    providers = r._providers()
    assert set(providers.keys()) == {"fake_oai", "fake_ollama", "fake_gemini"}
    assert isinstance(providers["fake_oai"], r.OpenAICompatProvider)
    assert isinstance(providers["fake_ollama"], r.OllamaProvider)
    assert isinstance(providers["fake_gemini"], r.GeminiEmbedProvider)


def test_yaml_loader_resolves_env_vars(temp_yaml_router):
    p = temp_yaml_router._providers()["fake_oai"]
    assert p.base_url == "http://example.com/v1"
    assert p.api_key == "k1"


def test_yaml_loader_unknown_provider_type_skipped(monkeypatch, tmp_path):
    yaml_path = tmp_path / "r.yaml"
    yaml_path.write_text("""
providers:
  good: {type: openai_compat, base_url_env: G_BASE, api_key_env: G_KEY}
  bad:  {type: nonexistent_kind, base_url_env: B_BASE, api_key_env: B_KEY}
routes:
  simple_chat: [good:m1]
costs: {}
embeddings: {default: good:e1, dim: 64}
""")
    monkeypatch.setenv("G_BASE", "http://x"); monkeypatch.setenv("G_KEY", "k")
    from quant import llm_router as r
    monkeypatch.setattr(r, "CONFIG_PATH", yaml_path)
    r.reload_config()
    try:
        providers = r._providers()
        assert "good" in providers
        assert "bad" not in providers
    finally:
        r.reload_config()


def test_chat_dispatches_to_first_provider_in_chain(temp_yaml_router, monkeypatch):
    """When chain is [fake_ollama:..., fake_oai:...], the ollama provider should be tried first."""
    r = temp_yaml_router
    calls = []

    def ollama_chat(self, model, messages, **kw):
        calls.append(("ollama", model))
        return {"text": "from ollama", "tokens_in": 5, "tokens_out": 10,
                "backend": f"fake_ollama:{model}"}

    def oai_chat(self, model, messages, **kw):
        calls.append(("oai", model))
        return {"text": "from oai", "tokens_in": 1, "tokens_out": 1,
                "backend": f"fake_oai:{model}"}

    monkeypatch.setattr(r.OllamaProvider, "chat", ollama_chat)
    monkeypatch.setattr(r.OpenAICompatProvider, "chat", oai_chat)

    out = r.chat("hi", task="deep_reasoning")
    assert out["text"] == "from ollama"
    assert calls == [("ollama", "reasoner-x")]


def test_chat_falls_back_when_first_provider_raises(temp_yaml_router, monkeypatch):
    r = temp_yaml_router

    def ollama_chat(self, model, messages, **kw):
        raise RuntimeError("ollama down")

    def oai_chat(self, model, messages, **kw):
        return {"text": "rescued", "tokens_in": 1, "tokens_out": 1,
                "backend": f"fake_oai:{model}"}

    monkeypatch.setattr(r.OllamaProvider, "chat", ollama_chat)
    monkeypatch.setattr(r.OpenAICompatProvider, "chat", oai_chat)

    out = r.chat("hi", task="deep_reasoning")
    assert out["text"] == "rescued"


def test_chat_unknown_task_falls_back_to_simple_chat(temp_yaml_router, monkeypatch):
    r = temp_yaml_router
    monkeypatch.setattr(r.OpenAICompatProvider, "chat",
                         lambda self, m, msgs, **kw: {"text": "ok", "tokens_in": 1,
                                                      "tokens_out": 1, "backend": f"fake_oai:{m}"})
    out = r.chat("hi", task="nonexistent_task")
    assert out["text"] == "ok"


def test_chat_thinking_field_salvaged_when_content_empty(temp_yaml_router, monkeypatch):
    """Some reasoning models return empty content but populated thinking field."""
    r = temp_yaml_router
    monkeypatch.setattr(r.OllamaProvider, "chat",
                         lambda self, m, msgs, **kw: {
                             "text": "", "thinking": "internal monologue here",
                             "tokens_in": 1, "tokens_out": 1,
                             "backend": f"fake_ollama:{m}"})
    out = r.chat("hi", task="deep_reasoning")
    assert out["text"] == "internal monologue here"


# ---- Cost estimation ----


def test_cost_estimate_uses_yaml_pricing(temp_yaml_router):
    r = temp_yaml_router
    cost = r._estimate_cost("fake_oai:gpt-1", tokens_in=1_000_000, tokens_out=1_000_000)
    assert cost == pytest.approx(1.5 + 4.5)


def test_cost_estimate_unknown_backend_returns_none(temp_yaml_router):
    assert temp_yaml_router._estimate_cost("nope:nope", 1000, 1000) is None


# ---- Embedding routing ----


def test_embedding_default_resolved_from_yaml(temp_yaml_router):
    p, m = temp_yaml_router.get_embedding_default()
    assert p == "fake_gemini"
    assert m == "emb-1"


def test_embed_uses_configured_provider(temp_yaml_router, monkeypatch):
    r = temp_yaml_router
    seen = []

    def fake_embed(self, model, text, **kw):
        seen.append((self.name, model, text))
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(r.GeminiEmbedProvider, "embed", fake_embed)
    out = r.embed("hello world")
    assert out == [[0.1, 0.2, 0.3]]
    assert seen == [("fake_gemini", "emb-1", "hello world")]


# ---- Provider class behaviour ----


def test_openai_compat_raises_when_no_api_key():
    from quant.llm_router import OpenAICompatProvider, ProviderError
    p = OpenAICompatProvider(name="x", base_url="http://x", api_key=None)
    with pytest.raises(ProviderError, match="api key"):
        p.chat("model", [{"role": "user", "content": "hi"}])


def test_ollama_raises_when_no_api_key():
    from quant.llm_router import OllamaProvider, ProviderError
    p = OllamaProvider(name="x", base_url="http://x", api_key=None)
    with pytest.raises(ProviderError, match="api key"):
        p.chat("model", [{"role": "user", "content": "hi"}])


def test_provider_default_does_not_implement_chat_or_embed():
    """Base Provider class should raise NotImplementedError, forcing subclasses."""
    from quant.llm_router import Provider, GeminiEmbedProvider, OpenAICompatProvider

    # Embed-only provider raises on chat
    p = GeminiEmbedProvider(name="x", api_key="k")
    with pytest.raises(NotImplementedError):
        p.chat("m", [])

    # Chat-only provider raises on embed
    p2 = OpenAICompatProvider(name="x", base_url="http://x", api_key="k")
    with pytest.raises(NotImplementedError):
        p2.embed("m", "text")
