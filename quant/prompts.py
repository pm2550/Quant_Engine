"""Prompt registry — load Markdown prompts from /data2/quant/prompts/.

Why a registry instead of inline strings:
  - prompts are diffable (PR review surfaces wording changes),
  - versioned via frontmatter (audit trail when output quality shifts),
  - swappable for A/B testing without code changes.

Format: each .md file has a YAML frontmatter block, then the prompt body.
The body uses Python `str.format()`-style placeholders ({portfolio}, etc).

Usage:
    from quant.prompts import load
    body = load("newswatch_severity").format(portfolio=p)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int | str
    body: str
    meta: dict


def _parse(text: str, name: str) -> Prompt:
    """Split frontmatter (between --- markers) from body."""
    if text.startswith("---\n"):
        try:
            _, fm, body = text.split("---\n", 2)
        except ValueError:
            return Prompt(name=name, version="?", body=text, meta={})
        meta = yaml.safe_load(fm) or {}
        return Prompt(name=name, version=meta.get("version", "?"),
                       body=body.lstrip("\n"), meta=meta)
    return Prompt(name=name, version="?", body=text, meta={})


@lru_cache(maxsize=None)
def _load_cached(name: str) -> Prompt:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    return _parse(path.read_text(encoding="utf-8"), name=name)


def get(name: str) -> Prompt:
    """Return the full Prompt object (body + metadata)."""
    return _load_cached(name)


def load(name: str) -> str:
    """Convenience: return just the body for direct .format() use."""
    return _load_cached(name).body


def reload() -> None:
    """Drop the cache — useful for tests and hot-edits during development."""
    _load_cached.cache_clear()


def list_all() -> list[Prompt]:
    """Enumerate all prompts in the registry. Used by audit/CLI."""
    out = []
    if PROMPTS_DIR.exists():
        for p in sorted(PROMPTS_DIR.glob("*.md")):
            out.append(_load_cached(p.stem))
    return out


if __name__ == "__main__":
    for p in list_all():
        print(f"{p.name:30s} v{p.version}  ({len(p.body)} chars)  {p.meta.get('purpose','')}")
