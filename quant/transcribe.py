"""音频转文字 + (可选) 智能分析 - 自动路由.

后端优先级 (auto):
  1. Gemini 2.5 Flash  — 默认首选 (转录+分析一次完成, 秒级)
  2. 阿里云 Paraformer  — 备选 (纯转录, 中文最强)
  3. 本地 faster-whisper-small — 兜底 (1x 实时, 无限免费)

用法 (CLI):
  python -m quant.transcribe URL_or_path
  python -m quant.transcribe URL --analyze "提取 CFO 关于 Q2 收入的关键观点"
  python -m quant.transcribe URL --prefer local  # 强制本地

用法 (模块):
  from quant.transcribe import transcribe
  r = transcribe("https://.../call.mp3", analyze="鹰派还是鸽派?")
  print(r["text"])      # 转录
  print(r["analysis"])  # 分析 (有的话)

环境变量:
  GEMINI_API_KEY  — Google Gemini API key (首选)
  ALIYUN_ASR_KEY  — 百炼普通 API-KEY (sk-XXX, 不是 sk-sp-XXX coding plan)

返回:
  {"text": "...", "analysis": "..." or None, "backend": "gemini"|"aliyun"|"whisper",
   "duration_s": float, "model": str, "wall_time_s": float}
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

log = logging.getLogger(__name__)

ALIYUN_KEY = os.environ.get("ALIYUN_ASR_KEY") or os.environ.get("DASHSCOPE_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")  # set via secrets/secrets.env, no hardcode fallback

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")

# Gemini inline audio limit: 20MB (we'll use Files API above this in future)
GEMINI_INLINE_MAX_BYTES = 20 * 1024 * 1024


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False


def _download(url: str, dest_dir: str) -> Path:
    """Download URL to a temp file in dest_dir, return path."""
    suffix = Path(urlparse(url).path).suffix or ".audio"
    dest = Path(dest_dir) / f"audio{suffix}"
    log.info("downloading %s → %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "claude-transcribe/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            f.write(chunk)
    return dest


# ---- Gemini 2.5 Flash (audio + analysis) ----
GEMINI_MODELS_CHAIN = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

_AUDIO_MIME = {
    ".mp3": "audio/mp3",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".aiff": "audio/aiff",
    ".webm": "audio/webm",
}


def _gemini_transcribe(
    audio_path_or_url: str,
    *,
    language: str = "auto",
    analyze: str | None = None,
) -> dict:
    """Use Gemini 2.5 Flash with inline audio for transcription + optional analysis."""
    import requests as _r
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    # Resolve to local file
    if _is_url(audio_path_or_url):
        with tempfile.TemporaryDirectory() as td:
            path = _download(audio_path_or_url, td)
            return _gemini_transcribe_local(str(path), language=language, analyze=analyze)
    return _gemini_transcribe_local(audio_path_or_url, language=language, analyze=analyze)


def _gemini_transcribe_local(
    path: str,
    *,
    language: str = "auto",
    analyze: str | None = None,
) -> dict:
    import requests as _r, base64
    size = os.path.getsize(path)
    if size > GEMINI_INLINE_MAX_BYTES:
        raise RuntimeError(
            f"audio file too large for Gemini inline ({size/1e6:.1f}MB > 20MB); "
            "fall back to whisper or implement Files API upload"
        )
    suffix = Path(path).suffix.lower()
    mime = _AUDIO_MIME.get(suffix, "audio/mp3")
    with open(path, "rb") as f:
        data_b64 = base64.b64encode(f.read()).decode()

    lang_hint = (
        "Transcribe this audio."
        if language == "en"
        else "转录这段中文音频, 保持原文标点."
        if language == "zh"
        else "Transcribe this audio in its original language."
    )
    if analyze:
        prompt = (
            f"{lang_hint}\n\n"
            f"Then under a heading '## 分析', do this analysis: {analyze}\n"
            "Return as Markdown: '## 转录' followed by the transcript, "
            "then '## 分析' followed by the analysis."
        )
    else:
        prompt = lang_hint

    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": data_b64}},
            ]
        }]
    }

    last_err = None
    for model in GEMINI_MODELS_CHAIN:
        for attempt in range(3):
            try:
                t0 = time.time()
                r = _r.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    params={"key": GEMINI_KEY},
                    json=body,
                    timeout=300,
                )
                r.raise_for_status()
                resp = r.json()
                elapsed = time.time() - t0
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
                # Parse if markdown headings present
                transcript, analysis = _split_transcript_analysis(text)
                tokens = resp.get("usageMetadata", {})
                # Audio tokens ~= seconds × 32 (Gemini tokenizes audio at ~32 tokens/sec)
                audio_tokens = sum(
                    p.get("tokenCount", 0)
                    for p in tokens.get("promptTokensDetails", [])
                    if p.get("modality") == "AUDIO"
                )
                duration = audio_tokens / 32.0 if audio_tokens else None
                return {
                    "text": transcript or text,
                    "analysis": analysis,
                    "duration_s": duration,
                    "model": model,
                    "backend": "gemini",
                    "wall_time_s": round(elapsed, 1),
                    "tokens": tokens,
                }
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("gemini %s attempt %d: %s", model, attempt + 1, e)
                time.sleep(min(2 ** attempt, 5))
    raise RuntimeError(f"All Gemini models failed: {last_err}")


def _split_transcript_analysis(text: str) -> tuple[str | None, str | None]:
    """Split markdown text with '## 转录' / '## 分析' headings."""
    import re
    if "## 转录" in text or "## 分析" in text:
        parts = re.split(r"## (?:转录|分析)\s*\n?", text)
        # parts[0] is leading (often empty), parts[1] = transcript, parts[2] = analysis (if present)
        non_empty = [p.strip() for p in parts if p.strip()]
        if len(non_empty) >= 2:
            return non_empty[0], non_empty[1]
        elif len(non_empty) == 1:
            return non_empty[0], None
    return None, None


# ---- Aliyun (Bailian) ASR ----
def _aliyun_transcribe(audio_url_or_path: str, *, language: str = "auto") -> dict:
    """Async Paraformer-v2 transcription. Returns {text, duration_s, model}."""
    import requests
    if not ALIYUN_KEY:
        raise RuntimeError("ALIYUN_ASR_KEY env var not set")
    if not _is_url(audio_url_or_path):
        # Aliyun ASR needs a public URL. We don't host audio. So fall through.
        raise RuntimeError("Aliyun ASR needs a URL, not a local file (not implemented for upload)")

    submit_url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
    headers = {
        "Authorization": f"Bearer {ALIYUN_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    body = {
        "model": "paraformer-v2",
        "input": {"file_urls": [audio_url_or_path]},
        "parameters": {"language_hints": [language] if language != "auto" else ["zh", "en"]},
    }
    r = requests.post(submit_url, json=body, headers=headers, timeout=30)
    r.raise_for_status()
    submit = r.json()
    task_id = submit["output"]["task_id"]
    log.info("Aliyun ASR task submitted: %s", task_id)

    # Poll until done
    poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    deadline = time.time() + 600  # 10 min cap
    while time.time() < deadline:
        time.sleep(3)
        r = requests.get(poll_url, headers={"Authorization": f"Bearer {ALIYUN_KEY}"}, timeout=20)
        r.raise_for_status()
        status = r.json()["output"]["task_status"]
        if status == "SUCCEEDED":
            results = r.json()["output"]["results"]
            transcripts = []
            duration_total = 0.0
            for item in results:
                tx_url = item.get("transcription_url")
                if not tx_url:
                    continue
                tr = requests.get(tx_url, timeout=30).json()
                for sentence in tr.get("transcripts", []):
                    transcripts.append(sentence.get("text", ""))
                duration_total += tr.get("properties", {}).get("audio_duration_in_seconds", 0)
            return {
                "text": "".join(transcripts).strip(),
                "duration_s": duration_total,
                "model": "paraformer-v2",
                "backend": "aliyun",
            }
        if status == "FAILED":
            raise RuntimeError(f"Aliyun ASR failed: {r.json()}")
    raise TimeoutError("Aliyun ASR polling timed out (10 min)")


# ---- Local faster-whisper ----
_WHISPER_MODEL_CACHE = None


def _local_whisper_transcribe(audio_path: str, *, language: str = "auto") -> dict:
    global _WHISPER_MODEL_CACHE
    from faster_whisper import WhisperModel
    if _WHISPER_MODEL_CACHE is None:
        log.info("loading whisper-%s (compute=%s) ...", WHISPER_MODEL, WHISPER_COMPUTE)
        _WHISPER_MODEL_CACHE = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE)

    lang_arg = None if language == "auto" else language
    log.info("transcribing %s ...", audio_path)
    t0 = time.time()
    segments, info = _WHISPER_MODEL_CACHE.transcribe(
        audio_path,
        language=lang_arg,
        beam_size=5,
        vad_filter=True,
    )
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)
    elapsed = time.time() - t0
    return {
        "text": "".join(text_parts).strip(),
        "duration_s": float(info.duration),
        "model": f"whisper-{WHISPER_MODEL}",
        "backend": "whisper",
        "language_detected": info.language,
        "wall_time_s": round(elapsed, 1),
    }


# ---- Public API ----
def transcribe(
    audio: str,
    *,
    language: str = "auto",
    prefer: str = "auto",   # auto | gemini | aliyun | local
    analyze: str | None = None,
) -> dict:
    """Auto-route: gemini > aliyun > local whisper.

    `audio` can be a local file path or a public URL.
    `analyze`: optional prompt to also extract insights (only Gemini supports this in
    one shot; others ignore it and just return text).
    """
    backends_to_try: list[str] = []
    if prefer == "auto":
        if GEMINI_KEY:
            backends_to_try.append("gemini")
        if ALIYUN_KEY and _is_url(audio):
            backends_to_try.append("aliyun")
        backends_to_try.append("local")
    elif prefer == "gemini":
        backends_to_try = ["gemini"]
    elif prefer == "cloud" or prefer == "aliyun":
        backends_to_try = ["aliyun"]
    elif prefer == "local":
        backends_to_try = ["local"]

    last_err: Exception | None = None
    for backend in backends_to_try:
        try:
            if backend == "gemini":
                return _gemini_transcribe(audio, language=language, analyze=analyze)
            if backend == "aliyun":
                if not _is_url(audio):
                    raise RuntimeError("aliyun needs a URL, not local path")
                return _aliyun_transcribe(audio, language=language)
            if backend == "local":
                if _is_url(audio):
                    with tempfile.TemporaryDirectory() as td:
                        local = _download(audio, td)
                        return _local_whisper_transcribe(str(local), language=language)
                return _local_whisper_transcribe(audio, language=language)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("backend=%s failed: %s, trying next", backend, e)
    raise RuntimeError(f"All transcribe backends failed; last error: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="local audio file path or http(s) URL")
    ap.add_argument("--lang", default="auto", help="zh / en / auto")
    ap.add_argument("--prefer", default="auto", choices=["auto", "cloud", "local"])
    ap.add_argument("--json", action="store_true", help="output JSON")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    out = transcribe(args.audio, language=args.lang, prefer=args.prefer)
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"=== {out['backend']} / {out['model']} / 音频 {out['duration_s']:.1f}s ===")
        if out.get("wall_time_s"):
            print(f"耗时 {out['wall_time_s']}s")
        print()
        print(out["text"])


if __name__ == "__main__":
    main()
