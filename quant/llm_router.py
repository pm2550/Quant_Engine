"""统一 LLM 路由 — 配置驱动, 换模型/换厂商只改 config/llm_routes.yaml.

架构:
  Provider 抽象 (chat/embed)
    ├── OpenAICompatProvider — 适配所有 /v1/chat/completions (dashscope / OpenRouter / Together / vLLM ...)
    ├── OllamaProvider       — Ollama /api/chat 形状 (含 thinking 字段)
    └── GeminiEmbedProvider  — Gemini embedContent

加新厂商:
  - OpenAI 兼容: 加 YAML 一行就够 (例: openrouter: {type: openai_compat, base_url_env: ..., api_key_env: ...})
  - 新 API 形状: 写一个 Provider 子类, 在 _PROVIDER_TYPES 注册

每次调用都写一行到 SQLite `llm_audit` 表 (provider/任务/延迟/token/估算成本/错误).
跑 `python -c "from quant import db; print(db.llm_audit_summary(7))"` 看 7 天汇总.

⚠️ deepseek 系列已下架 — 见 memory/feedback_no_deepseek.md
"""
from __future__ import annotations
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
import yaml

log = logging.getLogger(__name__)

# Load secrets if not in env yet
_SECRETS = Path("/data2/quant/secrets/secrets.env")
if _SECRETS.exists():
    for line in _SECRETS.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "llm_routes.yaml"

# Default base URLs when env vars unset
_DEFAULT_BASE = {
    "DASHSCOPE_BASE": "http://localhost/dashscope/v1",
    "OLLAMA_CLOUD_BASE": "https://ollama.com",
}


# ---- Provider abstraction ----


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    """Abstract LLM provider. Subclasses implement chat() or embed() (or both)."""

    name: str  # set by subclass instance

    def chat(self, model: str, messages: list[dict], **kw) -> dict:
        raise NotImplementedError(f"{self.__class__.__name__} does not support chat")

    def embed(self, model: str, text: str, **kw) -> list[float]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support embed")


class OpenAICompatProvider(Provider):
    """OpenAI-compatible /v1/chat/completions. Works for dashscope / OpenRouter /
    Together / Anyscale / vLLM / DeepInfra / any OpenAI-shaped endpoint.
    """

    def __init__(self, name: str, base_url: str, api_key: str | None,
                  default_timeout: int = 120):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_timeout = default_timeout

    def chat(self, model: str, messages: list[dict], **kw) -> dict:
        if not self.api_key:
            raise ProviderError(f"{self.name}: api key not configured")
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 2048),
            "temperature": kw.get("temperature", 0.3),
        }
        if kw.get("response_format") == "json":
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                      "Content-Type": "application/json"},
            json=payload,
            timeout=kw.get("timeout", self.default_timeout),
        )
        r.raise_for_status()
        data = r.json()
        return {
            "text": (data["choices"][0]["message"].get("content") or ""),
            "tokens_in": data.get("usage", {}).get("prompt_tokens", 0),
            "tokens_out": data.get("usage", {}).get("completion_tokens", 0),
            "backend": f"{self.name}:{model}",
        }


class OllamaProvider(Provider):
    """Ollama Cloud / self-hosted Ollama. Uses /api/chat with options{} block,
    and surfaces the `thinking` field for reasoning models.
    """

    def __init__(self, name: str, base_url: str, api_key: str | None,
                  default_timeout: int = 240):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_timeout = default_timeout

    def chat(self, model: str, messages: list[dict], **kw) -> dict:
        if not self.api_key:
            raise ProviderError(f"{self.name}: api key not configured")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": kw.get("max_tokens", 2048),
                "temperature": kw.get("temperature", 0.3),
            },
        }
        # Ollama supports format="json" for strict JSON-object output —
        # equivalent to OpenAI's response_format={"type":"json_object"}.
        if kw.get("response_format") == "json":
            payload["format"] = "json"
        # Disable thinking mode for tasks that need deterministic structured
        # output (e.g. format/markdown rendering). Thinking-mode models
        # otherwise burn the entire num_predict budget on internal reasoning
        # and return empty content.
        if kw.get("disable_thinking"):
            payload["think"] = False
        r = requests.post(
            f"{self.base_url}/api/chat",
            headers={"Authorization": f"Bearer {self.api_key}",
                      "Content-Type": "application/json"},
            json=payload,
            timeout=kw.get("timeout", self.default_timeout),
        )
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        return {
            "text": msg.get("content", ""),
            "thinking": msg.get("thinking", ""),
            "tokens_in": data.get("prompt_eval_count", 0),
            "tokens_out": data.get("eval_count", 0),
            "backend": f"{self.name}:{model}",
        }


class GeminiEmbedProvider(Provider):
    """Gemini embedContent. Embeddings only — chat goes via other providers."""

    def __init__(self, name: str, api_key: str | None, default_timeout: int = 60):
        self.name = name
        self.api_key = api_key
        self.default_timeout = default_timeout

    def embed(self, model: str, text: str, **kw) -> list[float]:
        if not self.api_key:
            raise ProviderError(f"{self.name}: api key not configured")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
        r = requests.post(
            url,
            params={"key": self.api_key},
            json={"content": {"parts": [{"text": text[:2000]}]}},
            timeout=kw.get("timeout", self.default_timeout),
        )
        r.raise_for_status()
        return r.json().get("embedding", {}).get("values", [])


# Type registry — extension point for new API shapes
_PROVIDER_TYPES = {
    "openai_compat": OpenAICompatProvider,
    "ollama": OllamaProvider,
    "gemini_embed": GeminiEmbedProvider,
}


# ---- Config loading ----


_CONFIG: dict | None = None
_PROVIDERS: dict[str, Provider] | None = None


def _load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        if not CONFIG_PATH.exists():
            raise RuntimeError(f"llm_routes.yaml not found at {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f) or {}
    return _CONFIG


def _build_providers(cfg: dict) -> dict[str, Provider]:
    out: dict[str, Provider] = {}
    for name, spec in (cfg.get("providers") or {}).items():
        ptype = spec.get("type")
        cls = _PROVIDER_TYPES.get(ptype)
        if not cls:
            log.warning("unknown provider type %s for %s, skipping", ptype, name)
            continue
        # Resolve api_key: try primary env, then fallback env
        api_key = None
        for env_name in (spec.get("api_key_env"), spec.get("fallback_api_key_env")):
            if env_name:
                v = os.environ.get(env_name)
                if v:
                    api_key = v
                    break
        # Resolve base_url
        base_url = None
        if "base_url_env" in spec:
            env_name = spec["base_url_env"]
            base_url = os.environ.get(env_name) or _DEFAULT_BASE.get(env_name)
        elif "base_url" in spec:
            base_url = spec["base_url"]

        kwargs = {"name": name, "api_key": api_key,
                   "default_timeout": spec.get("default_timeout", 120)}
        if cls is GeminiEmbedProvider:
            out[name] = cls(**kwargs)  # no base_url
        else:
            out[name] = cls(base_url=base_url, **kwargs)
    return out


def _providers() -> dict[str, Provider]:
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = _build_providers(_load_config())
    return _PROVIDERS


def reload_config() -> None:
    """Force reload after editing llm_routes.yaml — useful for tests / hot reload."""
    global _CONFIG, _PROVIDERS
    _CONFIG = None
    _PROVIDERS = None


def get_routes() -> dict[str, list[str]]:
    return _load_config().get("routes") or {}


def get_costs() -> dict[str, tuple[float, float]]:
    raw = _load_config().get("costs") or {}
    return {k: tuple(v) for k, v in raw.items()}


def get_embedding_default() -> tuple[str, str]:
    """Returns (provider_name, model). Falls back to gemini:gemini-embedding-001."""
    emb = _load_config().get("embeddings") or {}
    full = emb.get("default", "gemini:gemini-embedding-001")
    p, m = full.split(":", 1)
    return p, m


EMBEDDING_DIM = 3072  # kept for back-compat; canonical value lives in YAML


# ---- Cost estimation ----


def _estimate_cost(backend: str, tokens_in: int, tokens_out: int) -> float | None:
    """Return USD estimate, or None if backend not in pricing table."""
    pricing = get_costs().get(backend)
    if pricing is None:
        return None
    in_rate, out_rate = pricing
    return (tokens_in / 1_000_000) * in_rate + (tokens_out / 1_000_000) * out_rate


# ---- Audit logging ----


def _detect_caller() -> str | None:
    """Best-effort: return 'module.function' of the nearest non-llm_router frame."""
    import inspect
    try:
        frame = inspect.currentframe()
        while frame:
            mod = frame.f_globals.get("__name__", "")
            if mod and not mod.startswith("quant.llm_router") and mod != __name__:
                return f"{mod}.{frame.f_code.co_name}"
            frame = frame.f_back
    except Exception:  # noqa: BLE001
        return None
    return None


def _log_audit(*, task: str | None, backend: str, success: bool,
                wall_time_s: float | None = None,
                tokens_in: int = 0, tokens_out: int = 0,
                prompt_chars: int = 0, response_chars: int = 0,
                error: str | None = None, caller: str | None = None) -> None:
    """Best-effort write to llm_audit table. Lazy-imports db to avoid circular issues."""
    try:
        from quant import db
        db.log_llm_call(
            task=task, backend=backend, success=success,
            wall_time_s=wall_time_s,
            tokens_in=tokens_in or None, tokens_out=tokens_out or None,
            cost_usd=_estimate_cost(backend, tokens_in, tokens_out),
            caller=caller or _detect_caller(),
            prompt_chars=prompt_chars or None,
            response_chars=response_chars or None,
            error=error,
        )
    except Exception:  # noqa: BLE001
        pass


# ---- Public chat / embed API ----


def chat(
    prompt: str | list[dict],
    *,
    task: str = "simple_chat",
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    timeout: int = 120,
    response_format: str | None = None,
    disable_thinking: bool = False,
) -> dict:
    """LLM dispatch via config-driven route chain. Returns dict with text/backend/tokens."""
    if isinstance(prompt, str):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
    else:
        messages = prompt

    routes = get_routes()
    chain = routes.get(task) or routes.get("simple_chat") or []
    if not chain:
        raise RuntimeError(f"no route configured for task={task} and no simple_chat fallback")

    prompt_chars = sum(len(m.get("content") or "") for m in messages)
    caller_name = _detect_caller()
    providers = _providers()
    last_err: Exception | None = None

    for entry in chain:
        try:
            provider_name, model = entry.split(":", 1)
        except ValueError:
            log.warning("malformed route entry %r, expected provider:model", entry)
            continue
        provider = providers.get(provider_name)
        if provider is None:
            log.warning("provider %s not configured, skipping", provider_name)
            continue

        kwargs = {"max_tokens": max_tokens, "temperature": temperature, "timeout": timeout}
        if response_format == "json":
            kwargs["response_format"] = "json"
        if disable_thinking:
            kwargs["disable_thinking"] = True
        t0 = time.time()
        try:
            out = provider.chat(model, messages, **kwargs)
            elapsed = round(time.time() - t0, 2)
            out["wall_time_s"] = elapsed
            out["task"] = task
            text = out.get("text", "") or ""
            if not text.strip():
                # Some thinking-mode models emit content="" when budget exhausted by thinking.
                think = (out.get("thinking") or "").strip()
                if think:
                    log.info("backend %s returned empty content, salvaging thinking field", entry)
                    out["text"] = think
                    _log_audit(task=task, backend=entry, success=True,
                                wall_time_s=elapsed,
                                tokens_in=out.get("tokens_in", 0),
                                tokens_out=out.get("tokens_out", 0),
                                prompt_chars=prompt_chars,
                                response_chars=len(think),
                                caller=caller_name)
                    return out
                raise RuntimeError(f"empty content from {entry}")
            _log_audit(task=task, backend=entry, success=True,
                        wall_time_s=elapsed,
                        tokens_in=out.get("tokens_in", 0),
                        tokens_out=out.get("tokens_out", 0),
                        prompt_chars=prompt_chars,
                        response_chars=len(text),
                        caller=caller_name)
            return out
        except Exception as e:  # noqa: BLE001
            elapsed = round(time.time() - t0, 2)
            _log_audit(task=task, backend=entry, success=False,
                        wall_time_s=elapsed, prompt_chars=prompt_chars,
                        error=repr(e)[:500], caller=caller_name)
            last_err = e
            log.warning("backend %s failed (%s), trying next", entry, e)

    raise RuntimeError(f"all backends in chain {chain} failed: {last_err}")


def chat_json(prompt: str, *, task: str = "simple_chat", **kw) -> Any:
    """Convenience: chat with response_format=json, parse, return dict."""
    out = chat(prompt, task=task, response_format="json", **kw)
    text = out["text"].strip()
    # Strip code fences if model added them
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed: %s; raw=%s", e, text[:200])
        raise


def embed(texts: list[str] | str, *, model: str | None = None) -> list[list[float]]:
    """Get embeddings via the configured embedding provider. Returns list of vectors."""
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []

    provider_name, default_model = get_embedding_default()
    use_model = model or default_model
    provider = _providers().get(provider_name)
    if provider is None:
        raise RuntimeError(f"embedding provider {provider_name} not configured")

    backend = f"{provider_name}:{use_model}"
    caller_name = _detect_caller()
    out: list[list[float]] = []
    for text in texts:
        t0 = time.time()
        try:
            vec = provider.embed(use_model, text)
            out.append(vec)
            _log_audit(task="embed", backend=backend, success=True,
                        wall_time_s=round(time.time() - t0, 2),
                        prompt_chars=len(text), response_chars=len(vec),
                        caller=caller_name)
        except Exception as e:  # noqa: BLE001
            _log_audit(task="embed", backend=backend, success=False,
                        wall_time_s=round(time.time() - t0, 2),
                        prompt_chars=len(text),
                        error=repr(e)[:500], caller=caller_name)
            raise
        time.sleep(0.05)  # gentle rate limit (free tier)
    return out


# ---- Sentiment (FinBERT-zh) - lazy load ----
_FINBERT_MODEL = None
_FINBERT_TOKENIZER = None


def classify_sentiment(text: str) -> dict:
    """Return {label: 'pos|neg|neu', score: 0..1}. Lazy loads FinBERT."""
    global _FINBERT_MODEL, _FINBERT_TOKENIZER
    if _FINBERT_MODEL is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch  # noqa: F401
            _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained("bardsai/finance-sentiment-zh-base")
            _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained("bardsai/finance-sentiment-zh-base")
        except Exception as e:  # noqa: BLE001
            log.warning("FinBERT not available (%s), using LLM fallback", e)
            return _llm_sentiment_fallback(text)

    import torch
    inputs = _FINBERT_TOKENIZER(text[:512], return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = _FINBERT_MODEL(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    labels = ["negative", "neutral", "positive"]
    idx = max(range(3), key=lambda i: probs[i])
    return {"label": labels[idx], "score": probs[idx], "probs": dict(zip(labels, probs))}


def _llm_sentiment_fallback(text: str) -> dict:
    out = chat_json(
        f"判断这段财经文本的情绪. 仅返回 JSON: {{\"label\":\"positive\"|\"neutral\"|\"negative\",\"score\":0..1}}\n\n文本:\n{text[:1000]}",
        task="simple_chat",
        max_tokens=100,
    )
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--task", default="simple_chat")
    ap.add_argument("--system")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = chat(args.prompt, task=args.task, system=args.system)
    print(f"=== {out['backend']} {out['wall_time_s']}s ({out['tokens_in']}→{out['tokens_out']} tok) ===")
    print(out["text"])
    if out.get("thinking"):
        print("\n--- thinking ---")
        print(out["thinking"][:500])
