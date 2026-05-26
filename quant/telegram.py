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


def send(text: str, *, chat_id: str, parse_mode: str = "Markdown") -> dict:
    url = f"https://api.telegram.org/bot{_token()}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        # Fall back to plain text if Markdown parsing fails
        log.warning("telegram %s -> %s, retrying without parse_mode", r.status_code, r.text[:200])
        payload.pop("parse_mode", None)
        r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()
