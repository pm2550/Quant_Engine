"""Send messages to Telegram via Bot API."""
from __future__ import annotations
import logging
import os

import requests

log = logging.getLogger(__name__)

# Bot token must come from env. Loaded from /data2/quant/secrets/secrets.env
# at systemd unit start (EnvironmentFile=...). DO NOT hardcode here.


def _token() -> str:
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not tok:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set. Add it to /data2/quant/secrets/secrets.env "
            "(loaded by EnvironmentFile= in quant-*.service units)."
        )
    return tok


TG_LIMIT = 4096  # Telegram sendMessage hard limit (chars). Over this -> HTTP 400.


def _chunk(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Split on line boundaries so each piece fits Telegram's 4096-char hard limit.
    Root cause of the daily-digest 400 'byte offset 6996': the report is ~7k chars."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single over-long line -> hard split
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts


def _post(url: str, chunk: str, chat_id: str, parse_mode: str) -> "requests.Response":
    payload = {"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200 and parse_mode:
        # Markdown entity parse failure -> resend this chunk as plain text
        log.warning("telegram %s -> %s, resending chunk without parse_mode", r.status_code, r.text[:200])
        payload.pop("parse_mode", None)
        r = requests.post(url, json=payload, timeout=30)
    return r


def send(text: str, *, chat_id: str, parse_mode: str = "Markdown") -> dict:
    """Send to Telegram, chunking anything above the 4096-char limit. A single bad
    chunk is logged but does not abort the rest; raises only if every chunk fails."""
    url = f"https://api.telegram.org/bot{_token()}/sendMessage"
    parts = _chunk(text)
    last = None
    sent = 0
    for i, part in enumerate(parts):
        try:
            r = _post(url, part, chat_id, parse_mode)
            if r.status_code == 200:
                sent += 1
                last = r.json()
            else:
                log.error("telegram chunk %d/%d failed: %s %s", i + 1, len(parts), r.status_code, r.text[:200])
        except Exception as e:  # noqa: BLE001
            log.error("telegram chunk %d/%d exception: %s", i + 1, len(parts), e)
    if sent == 0:
        raise RuntimeError(f"telegram send failed for all {len(parts)} chunk(s)")
    return last or {"ok": True, "chunks": len(parts), "sent": sent}
