"""Unit tests for quant.prompts — frontmatter parsing, registry loading, cache."""
from __future__ import annotations
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def temp_prompts_dir(monkeypatch, tmp_path):
    from quant import prompts as p
    monkeypatch.setattr(p, "PROMPTS_DIR", tmp_path)
    p.reload()
    yield tmp_path
    p.reload()


def test_load_returns_body_only(temp_prompts_dir):
    (temp_prompts_dir / "greet.md").write_text(textwrap.dedent("""\
        ---
        name: greet
        version: 1
        ---
        Hello, {name}.
        """))
    from quant.prompts import load
    body = load("greet")
    assert body.startswith("Hello, {name}")
    # frontmatter is NOT in body
    assert "version: 1" not in body


def test_get_returns_full_prompt_with_metadata(temp_prompts_dir):
    (temp_prompts_dir / "greet.md").write_text(textwrap.dedent("""\
        ---
        name: greet
        version: 7
        purpose: say hi
        ---
        Hello, {name}.
        """))
    from quant.prompts import get
    p = get("greet")
    assert p.name == "greet"
    assert p.version == 7
    assert p.meta["purpose"] == "say hi"
    assert "Hello, {name}" in p.body


def test_load_without_frontmatter_returns_full_text(temp_prompts_dir):
    (temp_prompts_dir / "raw.md").write_text("Just a plain prompt.")
    from quant.prompts import get
    p = get("raw")
    assert p.body == "Just a plain prompt."
    assert p.version == "?"
    assert p.meta == {}


def test_missing_prompt_raises(temp_prompts_dir):
    from quant.prompts import load
    with pytest.raises(FileNotFoundError):
        load("nonexistent")


def test_list_all_returns_alphabetical(temp_prompts_dir):
    (temp_prompts_dir / "b.md").write_text("B")
    (temp_prompts_dir / "a.md").write_text("A")
    from quant.prompts import list_all
    names = [p.name for p in list_all()]
    assert names == ["a", "b"]


def test_reload_clears_cache(temp_prompts_dir):
    path = temp_prompts_dir / "x.md"
    path.write_text("v1")
    from quant.prompts import load, reload
    assert load("x") == "v1"
    path.write_text("v2")
    # without reload, the cached "v1" is returned
    assert load("x") == "v1"
    reload()
    assert load("x") == "v2"


# ---- Live prompt registry ----


def test_live_registry_has_newswatch_prompts():
    from quant.prompts import get
    sev = get("newswatch_severity")
    impact = get("newswatch_impact")
    assert "severity" in sev.body
    assert "{portfolio}" in sev.body
    assert "{snapshots}" in impact.body
    assert "{similar_history}" in impact.body


def test_live_prompts_have_versions():
    """Frontmatter must declare a version — the audit trail depends on it."""
    from quant.prompts import list_all
    for p in list_all():
        assert p.version not in ("?", None), f"{p.name} missing version frontmatter"
